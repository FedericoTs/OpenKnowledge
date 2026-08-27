"""Plain text and Markdown.

Markdown is the easy case and the reference one: its headings are explicit, so
this is where the heading-path model is at its most reliable. Every other parser
is trying to recover the same structure from a format that half-buried it.
"""

from __future__ import annotations

import re

from .blocks import Block, BlockKind, ParsedDocument, normalise

_ATX_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
_SETEXT_UNDERLINE = re.compile(r"^(=+|-{2,})\s*$")
_LIST_ITEM = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+(.*)$")
_TABLE_ROW = re.compile(r"^\s*\|(.+)\|\s*$")
_TABLE_DIVIDER = re.compile(r"^\s*\|?[\s:|-]+\|[\s:|-]*$")


def _split_row(line: str) -> list[str]:
    inner = _TABLE_ROW.match(line)
    raw = inner.group(1) if inner else line.strip().strip("|")
    return [cell.strip() for cell in raw.split("|")]


def parse_text(content: str, *, title: str | None = None) -> ParsedDocument:
    """Parse Markdown or plain text into blocks."""
    lines = normalise(content).split("\n")
    blocks: list[Block] = []
    heading_path: list[str] = []
    paragraph: list[str] = []
    table_headers: list[str] = []
    doc_title = title

    def flush_paragraph() -> None:
        if paragraph:
            blocks.append(
                Block(
                    kind=BlockKind.PARAGRAPH,
                    text=" ".join(paragraph).strip(),
                    heading_path=tuple(heading_path),
                )
            )
            paragraph.clear()

    def push_heading(text: str, level: int) -> None:
        nonlocal doc_title
        flush_paragraph()
        table_headers.clear()
        del heading_path[level - 1 :]
        heading_path.append(text)
        if doc_title is None and level == 1:
            doc_title = text
        blocks.append(
            Block(
                kind=BlockKind.HEADING,
                text=text,
                heading_path=tuple(heading_path[:-1]),
                level=level,
            )
        )

    for index, line in enumerate(lines):
        stripped = line.strip()

        if not stripped:
            flush_paragraph()
            table_headers.clear()
            continue

        atx = _ATX_HEADING.match(stripped)
        if atx:
            push_heading(atx.group(2).strip().rstrip("#").strip(), len(atx.group(1)))
            continue

        # Setext headings: the underline belongs to the line before it, which we
        # have already consumed into `paragraph`.
        if _SETEXT_UNDERLINE.match(stripped) and paragraph:
            level = 1 if stripped.startswith("=") else 2
            push_heading(paragraph.pop().strip(), level)
            continue

        if _TABLE_DIVIDER.match(stripped) and table_headers:
            continue  # the |---|---| separator carries no content

        if _TABLE_ROW.match(stripped):
            flush_paragraph()
            cells = _split_row(stripped)
            following = lines[index + 1].strip() if index + 1 < len(lines) else ""
            if not table_headers and _TABLE_DIVIDER.match(following):
                table_headers = cells
                continue
            from .blocks import table_row_text

            text = table_row_text(table_headers, cells)
            if text:
                blocks.append(
                    Block(
                        kind=BlockKind.TABLE_ROW,
                        text=text,
                        heading_path=tuple(heading_path),
                    )
                )
            continue

        item = _LIST_ITEM.match(line)
        if item:
            flush_paragraph()
            blocks.append(
                Block(
                    kind=BlockKind.LIST_ITEM,
                    text=item.group(1).strip(),
                    heading_path=tuple(heading_path),
                )
            )
            continue

        paragraph.append(stripped)

    flush_paragraph()
    return ParsedDocument(blocks=tuple(blocks), title=doc_title)
