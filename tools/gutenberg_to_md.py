#!/usr/bin/env python3
"""Turn a Project Gutenberg plain-text book into Markdown the parser can see.

The corpus in evals/golden-scope needs documents with real structure - acts,
chapters, a cast list - so that "what are the chapters?" has an answer that
lives in the document rather than in an exam somebody here wrote. Gutenberg
texts carry that structure as bare capitalised lines, which the Markdown parser
reads as paragraphs. This gives them heading markers and nothing else.

What it changes: CRLF to LF; the Gutenberg licence header and footer removed
(they are not the book); ``CHAPTER I.`` plus the title on the next line joined
into one ``## CHAPTER I. Title`` heading; ``FIRST ACT`` and the like made ``##``
headings; the two front-matter headings Wilde used made headings too.

The printed cast list under ``THE PERSONS IN THE PLAY`` - one name per line in
the original typography - becomes a Markdown list, one item per name, because a
list set as lines is a list, and the parser can only see one written as one.

What it never changes: a word of the text. Run it twice and the output is
byte-identical, which is what lets a test pin the committed copies.

    uv run python tools/gutenberg_to_md.py in.txt out.md
"""

from __future__ import annotations

import pathlib
import re
import sys

_START = re.compile(r"^\*\*\* START OF THE PROJECT GUTENBERG EBOOK .* \*\*\*$")
_END = re.compile(r"^\*\*\* END OF THE PROJECT GUTENBERG EBOOK .* \*\*\*$")
_CHAPTER = re.compile(r"^CHAPTER [IVXL]+\.$")
_ACT = re.compile(r"^(FIRST|SECOND|THIRD|FOURTH|FIFTH) ACT$")
_FRONT = frozenset({"THE PERSONS IN THE PLAY", "THE SCENES OF THE PLAY"})


def convert(text: str) -> str:
    lines = text.replace("\r\n", "\n").split("\n")

    # Keep only the book: everything between the START and END markers.
    start = next((i for i, line in enumerate(lines) if _START.match(line)), -1) + 1
    end = next((i for i, line in enumerate(lines) if _END.match(line)), len(lines))
    body = lines[start:end]

    out: list[str] = []
    i = 0
    while i < len(body):
        line = body[i].rstrip()
        if _CHAPTER.match(line):
            # The title is the next non-blank line.
            j = i + 1
            while j < len(body) and not body[j].strip():
                j += 1
            title = body[j].strip() if j < len(body) else ""
            out.append(f"## {line} {title}".rstrip())
            i = j + 1
            continue
        if line == "THE PERSONS IN THE PLAY":
            out.append(f"## {line}")
            out.append("")
            j = i + 1
            while j < len(body) and not body[j].strip():
                j += 1
            # The names run to the next blank line.
            while j < len(body) and body[j].strip():
                out.append(f"- {body[j].strip()}")
                j += 1
            i = j
            continue
        if _ACT.match(line) or line in _FRONT:
            out.append(f"## {line}")
            i += 1
            continue
        out.append(line)
        i += 1

    # Collapse the runs of blank lines the removals leave behind.
    text_out = "\n".join(out)
    text_out = re.sub(r"\n{3,}", "\n\n", text_out).strip() + "\n"
    return text_out


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(__doc__, file=sys.stderr)
        return 2
    src, dst = pathlib.Path(argv[1]), pathlib.Path(argv[2])
    dst.write_text(convert(src.read_text(encoding="utf-8")), encoding="utf-8")
    print(f"{dst}: {len(dst.read_text(encoding='utf-8').split())} words")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
