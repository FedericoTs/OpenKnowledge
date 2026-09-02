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
from ..documents.opendataloader import BATCH_SIZE
from ..documents.pdf import batching_helps, parse_pdfs
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
        file_principals: Callable[[], Mapping[str, frozenset[str]]] | None = None,
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
        # Readers a source stamped on individual files - the SharePoint mirror
        # writes what Graph said about each item. Where present they are the
        # source's own decision and win over a folder rule; a file the source
        # did not stamp falls back to the rules like any other.
        self.file_principals = file_principals
        # Parsing is by far the most expensive thing a scan does on the formats
        # a company actually has - 780ms for one small PDF against 6ms for the
        # same words in markdown, almost all of it a Java process starting up.
        # None means parse every time, which is what a test wants.
        self.parses = parses
        self.skipped = []
        # Per-scan working state, reset by every `fetch`. Named here so the
        # connector is a complete object before one has run.
        self._seen_keys: set[str] = set()
        self._pdf_queue: list[Path] = []
        self._parsed_ahead: dict[Path, ParsedDocument] = {}

    def _parse(self, path: Path) -> ParsedDocument:
        """Parse a file, or hand back the parse of these exact bytes.

        The bytes are the key, not the timestamp: ``rsync -t``, ``git
        checkout`` and every restore-from-backup put old mtimes on new
        content, and a cache that believed them would serve the previous
        version of a policy for ever. Reading the file to hash it costs
        milliseconds against the hundreds a PDF parse costs.
        """
        if path.suffix.lower() == ".pdf" and path not in self._parsed_ahead:
            self._parse_ahead()
        ahead = self._parsed_ahead.pop(path, None)
        if ahead is not None:
            return ahead  # already hashed, cached and counted by `_parse_ahead`
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

    def _parse_ahead(self) -> None:
        """Parse the next group of PDFs, so the JVM starts once for all of them.

        The parse cache made a *re*-index nearly free; a first index still paid
        a whole Java process per PDF, which on a corpus of a thousand policies
        is most of nine minutes spent starting and stopping JVMs. The parser
        takes a batch, so this hands it one - and the results wait here until
        the walk reaches each file.

        Driven from the walk rather than done up front, and only one group
        ahead: a batch holds every document in it in memory, and reading a
        whole corpus of large PDFs before indexing any of them would trade a
        JVM problem for a memory one. The queue is in the walk's own order, so
        the group starting at its head is the group about to be needed.

        A group already in the cache costs no JVM at all: the hit is kept and
        handed to ``_parse`` rather than being looked up a second time.
        """
        group, self._pdf_queue = self._pdf_queue[:BATCH_SIZE], self._pdf_queue[BATCH_SIZE:]
        pending: list[tuple[Path, str, bytes]] = []
        for path in group:
            try:
                data = path.read_bytes()
            except OSError:
                continue  # `_parse` reads it again and reports the reason
            key = cache_key(data, parser=self._parser_name(path))
            self._seen_keys.add(key)
            cached = self.parses.get(key) if self.parses is not None else None
            if cached is not None:
                self._parsed_ahead[path] = cached
                continue
            pending.append((path, key, data))
        if not pending:
            return
        parsed = parse_pdfs([data for _, _, data in pending], backend=self.pdf_backend)
        if parsed is None:
            return  # this backend gains nothing from a batch; `_parse` handles it
        for (path, key, _), document in zip(pending, parsed, strict=True):
            self._parsed_ahead[path] = document
            if self.parses is not None and document.blocks:
                self.parses.put(key, document)

    def access_map(self) -> dict[str, frozenset[str]]:
        """Who may read each document, without reading any of them.

        The same walk ``fetch`` does and none of the parsing: an access rule
        is decided by where a file sits, so recomputing it needs the tree and
        not the contents. Kept beside ``fetch`` on purpose - the rule that
        decides an audience must be written once, or the two will drift and
        the one that drifts is a document served to the wrong people.
        """
        rules = dict(self.folder_rules()) if self.folder_rules is not None else {}
        stamped = self._stamped()
        mapping: dict[str, frozenset[str]] = {}
        if not self.root.is_dir():
            return mapping
        for path in sorted(self.root.rglob("*")):
            if not path.is_file() or path.name.startswith("~$"):
                continue
            if path.suffix.lower() not in self.suffixes:
                continue
            relative = path.relative_to(self.root)
            mapping[document_id_for(relative)] = self._readers(relative, rules, stamped)
        return mapping

    def _stamped(self) -> Mapping[str, frozenset[str]]:
        return self.file_principals() if self.file_principals is not None else {}

    def _readers(
        self,
        relative: Path,
        rules: Mapping[str, frozenset[str]],
        stamped: Mapping[str, frozenset[str]],
    ) -> frozenset[str]:
        """Who may read one file: its source's stamp, else the deepest folder rule."""
        own = stamped.get(relative.as_posix())
        if own:
            return own
        ruled = effective_principals(relative.parent.as_posix(), rules)
        return ruled or self.allowed_principals

    def fetch(self) -> list[Document]:
        self.skipped = []
        self._seen_keys = set()
        self._pdf_queue = []
        self._parsed_ahead = {}
        if not self.root.is_dir():
            log.warning("document root %s does not exist; no documents loaded", self.root)
            return []

        rules = dict(self.folder_rules()) if self.folder_rules is not None else {}
        stamped = self._stamped()
        documents: list[Document] = []
        paths = sorted(self.root.rglob("*"))
        # The PDFs this walk will reach, in the order it will reach them, so
        # `_parse` can pull a whole group forward and start one JVM for it.
        # Left empty for a backend that gains nothing from a group, or every
        # file would be read once to build a batch that is never parsed.
        self._pdf_queue = (
            [
                path
                for path in paths
                if path.suffix.lower() == ".pdf"
                and ".pdf" in self.suffixes
                and not path.name.startswith("~$")
                and path.is_file()
            ]
            if batching_helps(self.pdf_backend)
            else []
        )
        for path in paths:
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

            # A stamp from the file's source wins; else the deepest folder rule;
            # else the connector-wide default (empty means readable by everyone).
            text = parsed.text
            documents.append(
                Document(
                    document_id=document_id_for(path.relative_to(self.root)),
                    title=_usable_title(parsed.title) or _fallback_title(path),
                    text=text,
                    url=path.as_uri(),
                    allowed_principals=self._readers(path.relative_to(self.root), rules, stamped),
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
