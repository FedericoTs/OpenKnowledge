"""Local folder connector.

The one that works today, and the one to test with. Point it at a directory of
text or Markdown files and OpenKnowledge answers from them - no cloud tenant, no
OAuth app registration, no admin consent. It is also the fixture the SharePoint
and Google Drive connectors are written against.
"""

from __future__ import annotations

import logging
from pathlib import Path

from ..retrieval.base import Document

log = logging.getLogger(__name__)

TEXT_SUFFIXES = frozenset({".txt", ".md", ".markdown", ".rst"})


class LocalFilesConnector:
    """Reads documents from a directory tree."""

    name = "local-files"

    def __init__(
        self,
        root: str | Path,
        *,
        suffixes: frozenset[str] = TEXT_SUFFIXES,
        allowed_principals: frozenset[str] = frozenset(),
    ) -> None:
        # Resolved eagerly: a relative root cannot be turned into the file:// URI
        # the citation carries, and it would also drift if the process chdirs.
        self.root = Path(root).expanduser().resolve()
        self.suffixes = suffixes
        self.allowed_principals = allowed_principals

    def fetch(self) -> list[Document]:
        if not self.root.is_dir():
            log.warning("document root %s does not exist; no documents loaded", self.root)
            return []

        documents: list[Document] = []
        for path in sorted(self.root.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in self.suffixes:
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError) as exc:
                # One unreadable file must not take down the whole re-index.
                log.warning("skipping %s: %s", path, exc)
                continue
            if not text.strip():
                continue

            rel = path.relative_to(self.root)
            documents.append(
                Document(
                    document_id=_document_id(rel),
                    title=_title(text, path),
                    text=text,
                    url=path.as_uri(),
                    allowed_principals=self.allowed_principals,
                )
            )
        return documents


def _document_id(rel: Path) -> str:
    """A stable, citable id: 'policies/expenses.md' -> 'policies-expenses'.

    Models have to reproduce this exactly in citations, so it stays short and
    free of characters that invite quoting mistakes.
    """
    stem = rel.with_suffix("").as_posix()
    return "".join(ch if ch.isalnum() or ch in "-_/" else "-" for ch in stem).replace("/", "-")


def _title(text: str, path: Path) -> str:
    """First Markdown heading if there is one, else the filename."""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            return stripped.lstrip("#").strip() or path.stem
        if stripped:
            break
    return path.stem.replace("-", " ").replace("_", " ").title()
