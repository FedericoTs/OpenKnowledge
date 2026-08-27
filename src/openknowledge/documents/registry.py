"""Choosing a parser for a file.

Dispatch is by extension rather than by sniffing content: every Office format is
a zip and every zip looks alike, so content sniffing gets .docx and .xlsx
confused for no benefit. An unrecognised extension is skipped and *named* in the
report, because a document silently contributing nothing is how a corpus grows
holes that only surface as a wrong answer months later.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from .blocks import ParsedDocument
from .pdf import parse_pdf
from .slides import parse_pptx
from .spreadsheet import parse_xlsx
from .text import parse_text
from .word import parse_docx

#: Formats read as text, decoded before parsing.
TEXT_SUFFIXES: frozenset[str] = frozenset({".txt", ".md", ".markdown", ".rst"})

#: Formats read as bytes.
BINARY_PARSERS: dict[str, Callable[..., ParsedDocument]] = {
    ".pdf": parse_pdf,
    ".docx": parse_docx,
    ".xlsx": parse_xlsx,
    ".xlsm": parse_xlsx,
    ".pptx": parse_pptx,
}

SUPPORTED_SUFFIXES: frozenset[str] = TEXT_SUFFIXES | frozenset(BINARY_PARSERS)

#: Legacy Office formats. Named separately so the report can say *why* they were
#: skipped - "unsupported" is unhelpful when the fix is a thirty-second re-save.
LEGACY_SUFFIXES: dict[str, str] = {
    ".doc": ".docx",
    ".xls": ".xlsx",
    ".ppt": ".pptx",
}


def is_supported(path: str | Path) -> bool:
    return Path(path).suffix.lower() in SUPPORTED_SUFFIXES


def skip_reason(path: str | Path) -> str | None:
    """Why this file will be ignored, or None if it will be read."""
    suffix = Path(path).suffix.lower()
    if suffix in SUPPORTED_SUFFIXES:
        return None
    if suffix in LEGACY_SUFFIXES:
        return (
            f"{suffix} is the pre-2007 Office format and cannot be read; "
            f"re-save it as {LEGACY_SUFFIXES[suffix]}"
        )
    return f"no parser for {suffix or 'files without an extension'}"


def parse_file(
    path: str | Path, *, title: str | None = None, pdf_backend: str = "auto"
) -> ParsedDocument:
    """Parse a file from disk, choosing the parser by extension.

    Never raises: an unreadable file returns an empty document carrying a
    warning, so one bad file cannot take down a whole re-index.
    """
    p = Path(path)
    suffix = p.suffix.lower()

    if suffix in TEXT_SUFFIXES:
        try:
            return parse_text(p.read_text(encoding="utf-8"), title=title)
        except (OSError, UnicodeDecodeError) as exc:
            return ParsedDocument(warnings=(f"could not read {p.name}: {exc}",))

    parser = BINARY_PARSERS.get(suffix)
    if parser is None:
        return ParsedDocument(warnings=(skip_reason(p) or f"no parser for {suffix}",))

    try:
        data = p.read_bytes()
    except OSError as exc:
        return ParsedDocument(warnings=(f"could not read {p.name}: {exc}",))
    if suffix == ".pdf":
        return parse_pdf(data, title=title, backend=pdf_backend)
    return parser(data, title=title)


def parse_bytes(
    data: bytes, *, suffix: str, title: str | None = None, pdf_backend: str = "auto"
) -> ParsedDocument:
    """Parse in-memory content - for connectors that fetch rather than read."""
    normalised = suffix.lower() if suffix.startswith(".") else f".{suffix.lower()}"
    if normalised in TEXT_SUFFIXES:
        try:
            return parse_text(data.decode("utf-8"), title=title)
        except UnicodeDecodeError as exc:
            return ParsedDocument(warnings=(f"not valid UTF-8 text: {exc}",))

    if normalised == ".pdf":
        return parse_pdf(data, title=title, backend=pdf_backend)
    parser = BINARY_PARSERS.get(normalised)
    if parser is None:
        return ParsedDocument(warnings=(f"no parser for {normalised}",))
    return parser(data, title=title)
