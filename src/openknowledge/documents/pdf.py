"""PDF.

The format that matters most - internal policy lives in PDFs - and the one that
carries no structure at all. A PDF knows where glyphs sit on a page and nothing
about headings, sections, or which numbers belong together. Everything useful
has to be recovered from geometry.

Three recoveries, each earning its complexity:

**Headings, from type size.** The most common size on a page is the body; lines
set materially larger, and short, are headings. This is what gives every block a
heading path, which is the difference between indexing "above EUR 500" and
"Expenses Policy > Approval thresholds: above EUR 500".

**Tables, ruled or not.** Most policy PDFs rule their tables and pdfplumber
finds those reliably. Plenty do not, and a threshold table that dissolves into
loose numbers is exactly the input that makes the numeric claim extractor
produce nonsense. So there is a text-alignment fallback - guarded, because on
prose it invents structure that is not there.

**Reading order, from vertical position.** Tables and prose are found by
separate passes, so they arrive in two unrelated sequences. Sorting everything
by its position on the page restores document order, which is what lets a table
inherit the heading printed above it rather than whichever heading happened to
be parsed last.

A PDF with no text layer is a scan. There is deliberately no OCR: it would mean
a heavyweight dependency and, on a policy document, a plausible-looking figure
with a character error in it. Better to report the file as unreadable.

Uses pdfplumber (MIT) rather than PyMuPDF, which is faster and better but AGPL -
compatible with this project's own licence, and a trap for the commercial
licensing ADR 0002 exists to keep possible.
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass
from typing import Any

from .blocks import (
    Block,
    BlockKind,
    ParsedDocument,
    looks_like_header_row,
    normalise,
    table_row_text,
)

log = logging.getLogger(__name__)

#: Below this many characters, a page has no usable text layer. Generous: a
#: sparse divider page is not a scan, a whole document of them is.
MIN_CHARS_PER_PAGE = 40

#: Bound the worst case. A garbled or adversarial PDF can otherwise expand into
#: something that costs real money to embed and index.
MAX_CHARS = 2_000_000

#: How much larger than the body a line must be set to read as a heading.
_HEADING_SIZE_RATIO = 1.15

#: Headings are short. A long line set large is a pull quote or a cover blurb.
_MAX_HEADING_CHARS = 90

#: Fraction of a column's cells that must contain a digit for it to count as a
#: value column.
_NUMERIC_COLUMN_RATIO = 0.6


def parse_pdf(data: bytes, *, title: str | None = None) -> ParsedDocument:
    """Parse a PDF into per-page blocks, keeping tables and headings intact."""
    try:
        import pdfplumber
    except ImportError:  # pragma: no cover - dependency is declared
        return ParsedDocument(warnings=("pdfplumber is not installed",))

    blocks: list[Block] = []
    warnings: list[str] = []
    heading_path: list[str] = []
    total_chars = 0
    empty_pages = 0
    pages = 0

    try:
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            pages = len(pdf.pages)
            for number, page in enumerate(pdf.pages, start=1):
                if total_chars >= MAX_CHARS:
                    warnings.append(f"stopped at page {number}: {MAX_CHARS:,} character cap")
                    break
                page_blocks, chars = _parse_page(page, f"p. {number}", heading_path)
                if chars < MIN_CHARS_PER_PAGE:
                    empty_pages += 1
                blocks.extend(page_blocks)
                total_chars += chars
    except Exception as exc:  # noqa: BLE001 - any malformed PDF, never fatal
        log.warning("could not read PDF: %s", exc)
        return ParsedDocument(warnings=(f"could not read this PDF: {exc}",))

    if pages and empty_pages == pages:
        warnings.append(
            f"no text layer on any of {pages} pages - this looks like a scan. "
            "OpenKnowledge does not OCR, so nothing from this file is searchable; "
            "supply a text-based copy."
        )
    elif empty_pages:
        warnings.append(f"{empty_pages} of {pages} pages had no text layer")

    resolved = title
    if resolved is None:
        resolved = next((b.text for b in blocks if b.kind is BlockKind.HEADING), None)

    return ParsedDocument(
        blocks=tuple(blocks), title=resolved, pages=pages, warnings=tuple(warnings)
    )


# -- geometry ---------------------------------------------------------------


def _body_font_size(words: list[dict[str, Any]]) -> float:
    """The size most of the page's text is set in.

    Weighted by characters and taken as the mode, so a single large title cannot
    drag the baseline up the way a mean would.
    """
    counts: dict[float, int] = {}
    for word in words:
        size = round(float(word.get("size") or 0), 1)
        if size:
            counts[size] = counts.get(size, 0) + len(str(word.get("text", "")))
    return max(counts, key=lambda k: counts[k]) if counts else 0.0


def _lines_from_words(
    words: list[dict[str, Any]],
) -> list[tuple[float, str, float, bool]]:
    """Group words into visual lines: ``(top, text, size, bold)``."""
    rows: dict[int, list[dict[str, Any]]] = {}
    for word in words:
        # Round the baseline so words on one line group despite sub-pixel drift.
        rows.setdefault(int(round(float(word.get("top", 0)))), []).append(word)

    lines: list[tuple[float, str, float, bool]] = []
    for top in sorted(rows):
        row = sorted(rows[top], key=lambda w: float(w.get("x0", 0)))
        text = normalise(" ".join(str(w.get("text", "")) for w in row))
        if not text:
            continue
        sizes = [float(w.get("size") or 0) for w in row if w.get("size")]
        fonts = " ".join(str(w.get("fontname", "")) for w in row).lower()
        lines.append((float(top), text, max(sizes) if sizes else 0.0, "bold" in fonts))
    return lines


def _is_heading(text: str, size: float, bold: bool, body_size: float) -> bool:
    if len(text) > _MAX_HEADING_CHARS or text.endswith((".", ";", ",")):
        return False
    if body_size and size >= body_size * _HEADING_SIZE_RATIO:
        return True
    # Bold and short at body size is the other common heading style.
    return bool(bold and len(text) <= 60 and body_size and size >= body_size)


# -- tables -----------------------------------------------------------------


def _numeric_columns(rows: list[list[str]]) -> set[int]:
    """Column indices whose cells mostly contain digits."""
    width = max((len(r) for r in rows), default=0)
    numeric: set[int] = set()
    for column in range(width):
        cells = [r[column] for r in rows if column < len(r) and r[column].strip()]
        if not cells:
            continue
        with_digits = sum(1 for c in cells if any(ch.isdigit() for ch in c))
        if with_digits / len(cells) >= _NUMERIC_COLUMN_RATIO:
            numeric.add(column)
    return numeric


def _clean_rows(raw: list[list[Any]], *, strict: bool) -> list[list[str]]:
    """Normalise a detected table, dropping rows that are really prose.

    ``strict`` is for the text-alignment pass, which will happily carve a
    paragraph into a grid. The guard is that a genuine policy table has at least
    one consistently numeric column - a threshold, a duration, an amount - and
    any row whose value cell holds no digit is a sentence that wandered in.
    """
    rows = [[normalise("" if c is None else str(c)) for c in row] for row in raw or []]
    rows = [r for r in rows if any(cell for cell in r)]
    if len(rows) < 2 or max((len(r) for r in rows), default=0) < 2:
        return []

    cells = [c for row in rows for c in row if c]
    if not cells or sum(1 for c in cells if len(c) <= 40) / len(cells) < 0.8:
        return []

    if not strict:
        return rows

    numeric = _numeric_columns([r for r in rows if not looks_like_header_row(r)])
    if not numeric:
        return []  # no value column: this is not a table, it is a paragraph
    return [
        row
        for row in rows
        if looks_like_header_row(row)
        or all(
            any(ch.isdigit() for ch in row[c]) for c in numeric if c < len(row) and row[c].strip()
        )
    ]


@dataclass(frozen=True, slots=True)
class _FoundTable:
    top: float
    bottom: float
    rows: list[list[str]]
    #: Whether ruling lines defined it. Only then is the bounding box worth
    #: trusting: the text-alignment strategy returns a box spanning the whole
    #: text area, so using it to suppress prose deletes the prose.
    ruled: bool

    @property
    def row_signatures(self) -> set[str]:
        """Each row as the prose pass will have read it - cells run together.

        A table row extracted as ``["Junior", "EUR 200", "5 days"]`` appears in
        the page's running text as ``Junior EUR 200 5 days``, so this is what
        lets an unruled table's lines be suppressed without a bounding box.
        """
        return {normalise(" ".join(cell for cell in row if cell.strip())) for row in self.rows}


def _find_tables(page: Any) -> list[_FoundTable]:
    """Tables on a page, ruled lines first and text alignment as a fallback."""
    ruled: list[_FoundTable] = []
    try:
        for table in page.find_tables():
            rows = _clean_rows(table.extract(), strict=False)
            if rows:
                ruled.append(_FoundTable(float(table.bbox[1]), float(table.bbox[3]), rows, True))
    except Exception:  # noqa: BLE001 - detection is best effort
        ruled = []
    if ruled:
        return ruled

    try:
        settings = {"vertical_strategy": "text", "horizontal_strategy": "text"}
        loose: list[_FoundTable] = []
        for table in page.find_tables(settings):
            rows = _clean_rows(table.extract(), strict=True)
            if rows:
                loose.append(_FoundTable(float(table.bbox[1]), float(table.bbox[3]), rows, False))
        return loose
    except Exception:  # noqa: BLE001
        return []


def _parse_page(page: Any, locator: str, heading_path: list[str]) -> tuple[list[Block], int]:
    """Blocks for one page, in reading order, with headings recovered."""
    tables = _find_tables(page)
    # Suppressing a table's own text from the prose pass, two ways. A ruled
    # table has a real bounding box, so geometry is exact. An unruled one does
    # not, so its rows are matched as text instead - using its box there would
    # delete the surrounding paragraphs along with the table.
    spans = [(t.top, t.bottom) for t in tables if t.ruled]
    signatures = {sig for t in tables if not t.ruled for sig in t.row_signatures}

    try:
        words = page.extract_words(extra_attrs=["size", "fontname"])
    except Exception:  # noqa: BLE001
        words = []

    body_size = _body_font_size(words)

    # Merge prose lines and tables into one sequence ordered by position on the
    # page, so a table inherits the heading printed above it rather than
    # whichever heading was parsed most recently.
    items: list[tuple[float, str, Any]] = [(t.top, "table", t.rows) for t in tables]
    for top, text, size, bold in _lines_from_words(words):
        if any(low - 2 <= top <= high + 2 for low, high in spans) or text in signatures:
            continue  # already emitted as part of a table
        kind = "heading" if _is_heading(text, size, bold, body_size) else "line"
        items.append((top, kind, (text, size)))
    items.sort(key=lambda item: item[0])

    blocks: list[Block] = []
    paragraph: list[str] = []
    chars = 0

    def flush() -> None:
        nonlocal chars
        if not paragraph:
            return
        text = " ".join(paragraph).strip()
        paragraph.clear()
        if not text:
            return
        blocks.append(
            Block(
                kind=BlockKind.PARAGRAPH,
                text=text,
                heading_path=tuple(heading_path),
                locator=locator,
            )
        )
        chars += len(text)

    for _, kind, payload in items:
        if kind == "heading":
            text, size = payload
            flush()
            # Larger type means a shallower heading, which nests a two-level
            # document correctly without real outline data.
            level = 1 if body_size and size >= body_size * 1.4 else 2
            del heading_path[level - 1 :]
            heading_path.append(text)
            blocks.append(
                Block(
                    kind=BlockKind.HEADING,
                    text=text,
                    heading_path=tuple(heading_path[:-1]),
                    locator=locator,
                    level=level,
                )
            )
            chars += len(text)
        elif kind == "line":
            paragraph.append(payload[0])
        else:
            flush()
            headers: list[str] = []
            for row in payload:
                if not headers and looks_like_header_row(row):
                    headers = row
                    continue
                text = table_row_text(headers, row)
                if not text:
                    continue
                blocks.append(
                    Block(
                        kind=BlockKind.TABLE_ROW,
                        text=text,
                        heading_path=tuple(heading_path),
                        locator=locator,
                    )
                )
                chars += len(text)

    flush()
    return blocks, chars
