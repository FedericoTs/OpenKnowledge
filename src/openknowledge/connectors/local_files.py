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
from dataclasses import dataclass, field
from pathlib import Path

from ..documents import SUPPORTED_SUFFIXES, parse_file, skip_reason
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
    #: Files from the last fetch that could not be indexed. Read by the engine
    #: so `index` can report them rather than leaving them invisible.
    skipped: list[SkippedFile] = field(default_factory=list)

    def __init__(
        self,
        root: str | Path,
        *,
        suffixes: frozenset[str] = SUPPORTED_SUFFIXES,
        allowed_principals: frozenset[str] = frozenset(),
    ) -> None:
        self.name = "local-files"
        # Resolved eagerly: a relative root cannot be turned into the file:// URI
        # the citation carries, and it would also drift if the process chdirs.
        self.root = Path(root).expanduser().resolve()
        self.suffixes = suffixes
        self.allowed_principals = allowed_principals
        self.skipped = []

    def fetch(self) -> list[Document]:
        self.skipped = []
        if not self.root.is_dir():
            log.warning("document root %s does not exist; no documents loaded", self.root)
            return []

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

            parsed = parse_file(path, title=None)
            for warning in parsed.warnings:
                self.skipped.append(SkippedFile(relative, warning))
                log.warning("%s: %s", relative, warning)

            if parsed.is_empty:
                if not parsed.warnings:
                    self.skipped.append(SkippedFile(relative, "no readable text"))
                continue

            documents.append(
                Document(
                    document_id=_document_id(path.relative_to(self.root)),
                    title=parsed.title or _fallback_title(path),
                    text=parsed.text,
                    url=path.as_uri(),
                    allowed_principals=self.allowed_principals,
                    blocks=parsed.blocks,
                )
            )
        return documents


def _document_id(rel: Path) -> str:
    """A stable, citable id: 'policies/expenses.pdf' -> 'policies-expenses'.

    Models have to reproduce this exactly in citations, so it stays short and
    free of characters that invite quoting mistakes.
    """
    stem = rel.with_suffix("").as_posix()
    return "".join(ch if ch.isalnum() or ch in "-_/" else "-" for ch in stem).replace("/", "-")


def _fallback_title(path: Path) -> str:
    """A readable title when the document did not declare one."""
    return path.stem.replace("-", " ").replace("_", " ").strip().title()
