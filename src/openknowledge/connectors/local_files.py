"""Local folder connector.

Point it at a directory of company documents and OpenKnowledge answers from
them - no cloud tenant, no OAuth registration, no admin consent. It is also the
fixture the SharePoint and Google Drive connectors are written against.

Every file goes through the document parsers, so a PDF policy, a Word procedure
and an Excel limits table all arrive as structured blocks with headings, tables
and real locators rather than as a wall of text.

Files it cannot read are **named**, not silently skipped. A document that
contributes nothing without saying so is how a corpus develops a hole that
surfaces months later as a wrong answer, and the two most common causes - a
scanned PDF and a pre-2007 Office file - both have a thirty-second fix if
somebody is told.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

from ..access import effective_principals
from ..documents import SUPPORTED_SUFFIXES, declares_superseded, parse_file, skip_reason
from ..documents.blocks import ParsedDocument
from ..documents.cache import ParseCache, cache_key
from ..retrieval.base import Document

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class SkippedFile:
    """A file in the corpus folder that produced nothing, and why."""

    path: str
    reason: str


@dataclass
class LocalFilesConnector:
    """Reads documents from a directory tree."""

    name: str = "local-files"
    root: Path = field(default_factory=Path)
    suffixes: frozenset[str] = SUPPORTED_SUFFIXES
    allowed_principals: frozenset[str] = frozenset()
    pdf_backend: str = "auto"
    #: Files from the last fetch that could not be indexed. Read by the engine
    #: so `index` can report them rather than leaving them invisible.
    skipped: list[SkippedFile] = field(default_factory=list)

    def __init__(
        self,
        root: str | Path,
        *,
        suffixes: frozenset[str] = SUPPORTED_SUFFIXES,
        allowed_principals: frozenset[str] = frozenset(),
        pdf_backend: str = "auto",
        folder_rules: Callable[[], Mapping[str, frozenset[str]]] | None = None,
        parses: ParseCache | None = None,
    ) -> None:
        self.name = "local-files"
        # Resolved eagerly: a relative root cannot be turned into the file:// URI
        # the citation carries, and it would also drift if the process chdirs.
        self.root = Path(root).expanduser().resolve()
        self.suffixes = suffixes
        self.allowed_principals = allowed_principals
        self.pdf_backend = pdf_backend
        # A callable, not a snapshot: rules are admin decisions that change
        # at runtime, and every re-index must see the current ones.
        self.folder_rules = folder_rules
        # Parsing is by far the most expensive thing a scan does on the formats
        # a company actually has - 780ms for one small PDF against 6ms for the
        # same words in markdown, almost all of it a Java process starting up.
        # None means parse every time, which is what a test wants.
        self.parses = parses
        self.skipped = []

    def _parse(self, path: Path) -> ParsedDocument:
        """Parse a file, or hand back the parse of these exact bytes.

        The bytes are the key, not the timestamp: ``rsync -t``, ``git
        checkout`` and every restore-from-backup put old mtimes on new
        content, and a cache that believed them would serve the previous
        version of a policy for ever. Reading the file to hash it costs
        milliseconds against the hundreds a PDF parse costs.
        """
        if self.parses is None:
            return parse_file(path, title=None, pdf_backend=self.pdf_backend)
        try:
            data = path.read_bytes()
        except OSError:
            # Let the parser produce the warning; it already has the words for
            # a file it could not read, and duplicating them here would mean
            # two ways to say the same thing and one of them going stale.
            return parse_file(path, title=None, pdf_backend=self.pdf_backend)

        key = cache_key(data, parser=self._parser_name(path))
        self._seen_keys.add(key)
        cached = self.parses.get(key)
        if cached is not None:
            return cached
        parsed = parse_file(path, title=None, pdf_backend=self.pdf_backend)
        # A parse that failed is not worth remembering: the next scan should
        # try again, because the reason is usually outside the file - a
        # missing backend, a JVM that did not start, a full disk.
        if parsed.blocks:
            self.parses.put(key, parsed)
        return parsed

    def _parser_name(self, path: Path) -> str:
        """Which parser's output this is. Two PDF backends extract slightly
        different text, so a cache shared between them would hand one
        backend's words to a corpus fingerprinted under the other's."""
        suffix = path.suffix.lower()
        return f"pdf:{self.pdf_backend}" if suffix == ".pdf" else suffix

    def access_map(self) -> dict[str, frozenset[str]]:
        """Who may read each document, without reading any of them.

        The same walk ``fetch`` does and none of the parsing: an access rule
        is decided by where a file sits, so recomputing it needs the tree and
        not the contents. Kept beside ``fetch`` on purpose - the rule that
        decides an audience must be written once, or the two will drift and
        the one that drifts is a document served to the wrong people.
        """
        rules = dict(self.folder_rules()) if self.folder_rules is not None else {}
        mapping: dict[str, frozenset[str]] = {}
        if not self.root.is_dir():
            return mapping
        for path in sorted(self.root.rglob("*")):
            if not path.is_file() or path.name.startswith("~$"):
                continue
            if path.suffix.lower() not in self.suffixes:
                continue
            relative = path.relative_to(self.root)
            ruled = effective_principals(relative.parent.as_posix(), rules)
            mapping[document_id_for(relative)] = ruled or self.allowed_principals
        return mapping

    def fetch(self) -> list[Document]:
        self.skipped = []
        self._seen_keys: set[str] = set()
        if not self.root.is_dir():
            log.warning("document root %s does not exist; no documents loaded", self.root)
            return []

        rules = dict(self.folder_rules()) if self.folder_rules is not None else {}
        documents: list[Document] = []
        for path in sorted(self.root.rglob("*")):
            if not path.is_file() or path.name.startswith("~$"):
                continue  # ~$ files are Office lock files, not documents

            relative = path.relative_to(self.root).as_posix()
            if path.suffix.lower() not in self.suffixes:
                reason = skip_reason(path)
                if reason:
                    self.skipped.append(SkippedFile(relative, reason))
                continue

            parsed = self._parse(path)
            for warning in parsed.warnings:
                self.skipped.append(SkippedFile(relative, warning))
                log.warning("%s: %s", relative, warning)

            if parsed.is_empty:
                if not parsed.warnings:
                    self.skipped.append(SkippedFile(relative, "no readable text"))
                continue

            # The deepest folder rule wins; unruled folders fall back to the
            # connector-wide default (empty means readable by everyone).
            folder = path.relative_to(self.root).parent.as_posix()
            ruled = effective_principals(folder, rules)
            text = parsed.text
            documents.append(
                Document(
                    document_id=document_id_for(path.relative_to(self.root)),
                    title=_usable_title(parsed.title) or _fallback_title(path),
                    text=text,
                    url=path.as_uri(),
                    allowed_principals=ruled or self.allowed_principals,
                    blocks=parsed.blocks,
                    superseded=declares_superseded(text),
                )
            )
        if self.parses is not None:
            # Everything this whole scan saw. Without it the cache grows by a
            # row per edit for ever - every draft of a policy ever saved into
            # the folder, kept because it existed once.
            self.parses.keep_only(self._seen_keys)
        return documents


def document_id_for(rel: Path) -> str:
    """A stable, citable id: 'policies/expenses.pdf' -> 'policies-expenses'.

    Models have to reproduce this exactly in citations, so it stays short and
    free of characters that invite quoting mistakes.
    """
    stem = rel.with_suffix("").as_posix()
    return "".join(ch if ch.isalnum() or ch in "-_/" else "-" for ch in stem).replace("/", "-")


_EMAIL_ADDRESS = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")


def _usable_title(title: str | None) -> str | None:
    """A parsed heading that can actually serve as a title, or None.

    An email printed to PDF opens with its From line, and the parser dutifully
    promoted 'Federico Sciuca <federico@...>' to the document's title - which
    is what the corpus listing then called the document. Reported from the
    first real-tenant corpus, which was exactly such a mail. A heading that
    contains an email address names a correspondent, not a document; the
    filename is the better title.
    """
    if not title or _EMAIL_ADDRESS.search(title):
        return None
    return title


def _fallback_title(path: Path) -> str:
    """A readable title when the document did not declare one."""
    return path.stem.replace("-", " ").replace("_", " ").strip().title()
