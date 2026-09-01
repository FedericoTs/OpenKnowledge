"""PDF via OpenDataLoader — the preferred backend when it is available.

OpenDataLoader PDF (Apache 2.0) is a Java parser that recovers a document's real
structure rather than inferring it from geometry. Where the pdfplumber backend
guesses headings from type size and reassembles tables from ruling lines, this
one reports them:

* ``heading`` elements carry an explicit **level**, so the heading trail is read
  rather than reconstructed.
* ``table`` elements carry rows, cells, and row and column spans, so a labelled
  row is a lookup instead of an alignment heuristic.
* Every element carries its **page number**, so citations point at a real page.
* With ``use_struct_tree`` it reads PDF/UA tags, which is true structure rather
  than any inference at all - and a good share of enterprise compliance
  documents are tagged.

It is also **deterministic**, which matters more here than it would elsewhere:
the same PDF must produce the same corpus version, or every cached answer is
invalidated on a re-index that changed nothing.

The cost is a JVM. That is why it stays optional and why the pure-Python
pdfplumber backend remains: a JVM is unremarkable inside a container and
impossible in a serverless function, so the choice is made at runtime by whether
Java is actually there.

The JSON output is used rather than the Markdown, because the Markdown discards
page numbers and heading levels - the two things worth having it for.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import subprocess
import tempfile
from collections.abc import Sequence
from pathlib import Path
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

#: Titles PDF writers leave in the metadata when nobody set one. Treated as
#: absent, so the document falls back to its first heading instead of being
#: indexed and cited as "(anonymous)".
_PLACEHOLDER_TITLES = frozenset(
    {"(anonymous)", "anonymous", "untitled", "unknown", "document", "-", "microsoft word"}
)

#: What a bound on one parse would be: 500 pages at the documented 60+
#: pages/sec is about 8 seconds, leaving room for JVM start-up and an unusually
#: heavy file.
#:
#: **Not enforced.** The wrapper package offers no timeout and runs the jar with
#: a blocking ``subprocess.run``, so nothing here can interrupt a parse that
#: never returns; the ``TimeoutExpired`` branch below is kept because that is
#: cheap and a later wrapper may grow one. Batching makes a hang cost a whole
#: group rather than one file. Recorded as a gap in docs/ROADMAP.md rather than
#: left as a number that reads like a protection nobody has.
TIMEOUT_SECONDS = 120

#: Same cap as the pdfplumber backend: bound a garbled or adversarial file.
MAX_CHARS = 2_000_000

#: How many PDFs go to one JVM.
#:
#: Measured on this box, 128 four-page PDFs, per document:
#:
#:     one JVM per file    656 ms
#:     batches of 32        71 ms
#:     batches of 64        49 ms
#:     batches of 128       40 ms
#:
#: Almost the whole win is in not paying JVM start-up per document (~640 ms of
#: that 656); past a few dozen the curve is nearly flat, and every document
#: added to a batch is one more file staged on disk and one more held by the
#: parser at once. 64 takes ~13x of a possible ~16x and keeps both bounded,
#: which is the trade a company server with a thousand policy PDFs wants.
BATCH_SIZE = 64


def is_available() -> bool:
    """Whether this backend can run: the wrapper installed and a JVM on PATH."""
    if shutil.which("java") is None:
        return False
    try:
        import opendataloader_pdf  # noqa: F401
    except ImportError:
        return False
    return True


def unavailable_reason() -> str | None:
    """Why this backend cannot run, for an operator to act on."""
    try:
        import opendataloader_pdf  # noqa: F401
    except ImportError:
        return "the opendataloader-pdf package is not installed"
    if shutil.which("java") is None:
        return "no Java runtime on PATH (OpenDataLoader is a Java parser)"
    return None


def _convert(sources: list[Path], output: Path, *, use_struct_tree: bool) -> Exception | None:
    """Run the parser over these files, handing back the failure rather than raising.

    The only place that decides how this parser is invoked. One document and
    sixty-four go through here identically, so the batch cannot come to read a
    document differently from the single-file path - which would be invisible
    until a corpus grew past the batching threshold and every cached answer
    was invalidated by a re-index that changed nothing.

    Not raising is the point of the return type: the CLI exits non-zero on a
    file it cannot read *having already written every good document*, so the
    caller reads the output directory either way.
    """
    import opendataloader_pdf

    try:
        opendataloader_pdf.convert(
            input_path=[str(source) for source in sources],
            output_dir=str(output),
            format="json",
            quiet=True,
            use_struct_tree=use_struct_tree,
        )
    except Exception as exc:  # noqa: BLE001 - a subprocess failure is not fatal
        return exc
    return None


def _failure_sentence(failure: Exception) -> str:
    """What the parser said, rather than the command line that said it.

    The wrapper raises ``CalledProcessError``, whose ``str`` is the whole java
    invocation - a temp path, a jar path and an exit code, none of which an
    operator can act on. The CLI's own stdout says the useful thing:

        Error: '000001.pdf' is not a valid PDF file (corrupted or truncated
        content).

    The quoted name is the one this parse staged the file under, never the
    operator's filename, so it is replaced rather than shown; the caller
    prefixes the real path when it records the file as skipped.
    """
    output = getattr(failure, "output", None)
    if isinstance(output, bytes):
        output = output.decode("utf-8", errors="replace")
    lines = [line.strip() for line in str(output or "").splitlines() if line.strip()]
    if not lines:
        return str(failure)
    sentence = lines[0].removeprefix("Error:").strip()
    return re.sub(r"^'[^']*'", "this file", sentence, count=1)


def parse_pdf_opendataloader(
    data: bytes, *, title: str | None = None, use_struct_tree: bool = False
) -> ParsedDocument:
    """Parse a PDF through OpenDataLoader. Never raises."""
    with tempfile.TemporaryDirectory(prefix="ok-odl-") as workspace:
        root = Path(workspace)
        source = root / "input.pdf"
        output = root / "out"
        source.write_bytes(data)

        failure = _convert([source], output, use_struct_tree=use_struct_tree)
        if isinstance(failure, subprocess.TimeoutExpired):
            return ParsedDocument(warnings=("OpenDataLoader timed out on this PDF",))
        if failure is not None:
            log.warning("OpenDataLoader failed: %s", failure)
            return ParsedDocument(warnings=(f"OpenDataLoader: {_failure_sentence(failure)}",))

        produced = sorted(output.rglob("*.json"))
        if not produced:
            return ParsedDocument(warnings=("OpenDataLoader produced no output",))

        try:
            document = json.loads(produced[0].read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return ParsedDocument(warnings=(f"OpenDataLoader output was unreadable: {exc}",))

    return _to_blocks(document, title=title)


def parse_pdfs_opendataloader(
    blobs: Sequence[bytes], *, use_struct_tree: bool = False
) -> list[ParsedDocument]:
    """Parse many PDFs in as few JVM starts as possible. Never raises.

    One parse per input, in the order given. This exists because the JVM, not
    the parsing, is what a first index spends its time on: ~640 ms of the
    ~656 ms a one-page PDF cost was the process starting up.

    **Staged under generated names, never their own.** The CLI names each
    output after its input's basename, so batching ``HR/policy.pdf`` and
    ``Finance/policy.pdf`` by their real paths writes one ``policy.json`` and
    silently loses a document - or worse, answers questions about one policy
    out of the other one's text. Verified, not assumed: two different PDFs
    with the same basename produce exactly one output file.

    A file the parser rejects does not take the batch with it. The CLI exits
    non-zero but still writes every good document first, so anything missing
    is re-parsed on its own, where it produces the same warning it would have
    produced had it never been batched.
    """
    results: list[ParsedDocument | None] = [None] * len(blobs)
    for start in range(0, len(blobs), BATCH_SIZE):
        chunk = blobs[start : start + BATCH_SIZE]
        with tempfile.TemporaryDirectory(prefix="ok-odl-batch-") as workspace:
            root = Path(workspace)
            staged = root / "in"
            output = root / "out"
            staged.mkdir()
            names = []
            for offset, data in enumerate(chunk):
                source = staged / f"{offset:06d}.pdf"
                source.write_bytes(data)
                names.append(source)
            failure = _convert(names, output, use_struct_tree=use_struct_tree)
            if failure is not None:
                log.debug("OpenDataLoader batch reported a failure: %s", failure)
            for offset in range(len(chunk)):
                produced = output / f"{offset:06d}.json"
                try:
                    document = json.loads(produced.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue  # parsed alone below, where it gets a real warning
                results[start + offset] = _to_blocks(document, title=None)

    return [
        parsed
        if parsed is not None
        else parse_pdf_opendataloader(data, use_struct_tree=use_struct_tree)
        for parsed, data in zip(results, blobs, strict=True)
    ]


# -- walking the document tree ---------------------------------------------


def _content(node: Any) -> str:
    """All text under a node, in order.

    Table cells hold their text in nested paragraphs rather than inline, so this
    has to recurse rather than read one field.
    """
    if isinstance(node, dict):
        parts = [str(node["content"])] if node.get("content") else []
        for key in ("kids", "rows", "cells"):
            for child in node.get(key) or []:
                text = _content(child)
                if text:
                    parts.append(text)
        return normalise(" ".join(parts))
    if isinstance(node, list):
        return normalise(" ".join(filter(None, (_content(item) for item in node))))
    return ""


def _locator(node: dict[str, Any]) -> str | None:
    page = node.get("page number")
    return f"p. {page}" if page else None


def _table_blocks(
    node: dict[str, Any], heading_path: tuple[str, ...], locator: str | None
) -> list[Block]:
    """Labelled rows from a table node.

    Cells arrive with explicit row and column numbers, so the header row is a
    lookup rather than the alignment guess the geometric backend has to make.
    """
    blocks: list[Block] = []
    headers: list[str] = []

    for row in node.get("rows") or []:
        cells = [_content(cell) for cell in row.get("cells") or []]
        if not any(cells):
            continue
        if not headers and looks_like_header_row(cells):
            headers = cells
            continue
        text = table_row_text(headers, cells)
        if text:
            blocks.append(
                Block(
                    kind=BlockKind.TABLE_ROW,
                    text=text,
                    heading_path=heading_path,
                    locator=_locator(row) or locator,
                )
            )
    return blocks


def _to_blocks(document: dict[str, Any], *, title: str | None) -> ParsedDocument:
    blocks: list[Block] = []
    heading_path: list[str] = []
    warnings: list[str] = []
    total = 0

    def visit(node: Any) -> None:
        nonlocal total
        if isinstance(node, list):
            for item in node:
                visit(item)
            return
        if not isinstance(node, dict):
            return
        if total >= MAX_CHARS:
            return

        kind = str(node.get("type") or "")
        locator = _locator(node)

        if kind == "heading":
            text = _content(node)
            if text:
                # The level is reported, not inferred from type size.
                level = int(node.get("heading level") or node.get("level") or 1)
                level = max(1, min(level, 6))
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
                total += len(text)
            return

        if kind == "table":
            for block in _table_blocks(node, tuple(heading_path), locator):
                blocks.append(block)
                total += len(block.text)
            return

        if kind in ("paragraph", "list item", "caption", "text"):
            text = _content(node)
            if text:
                blocks.append(
                    Block(
                        kind=BlockKind.LIST_ITEM
                        if kind == "list item"
                        else BlockKind.CAPTION
                        if kind == "caption"
                        else BlockKind.PARAGRAPH,
                        text=text,
                        heading_path=tuple(heading_path),
                        locator=locator,
                    )
                )
                total += len(text)
            return

        for key in ("kids", "rows", "cells"):
            visit(node.get(key))

    visit(document.get("kids"))

    if total >= MAX_CHARS:
        warnings.append(f"stopped at the {MAX_CHARS:,} character cap")

    pages = int(document.get("number of pages") or 0)
    if not blocks and pages:
        warnings.append(
            f"no text recovered from {pages} pages - this looks like a scan. "
            "OpenKnowledge does not OCR, so nothing from this file is searchable; "
            "supply a text-based copy."
        )

    resolved = title or _clean_title(document.get("title"))
    if not resolved:
        resolved = next((b.text for b in blocks if b.kind is BlockKind.HEADING), None)

    return ParsedDocument(
        blocks=tuple(blocks), title=resolved, pages=pages, warnings=tuple(warnings)
    )


def _clean_title(raw: Any) -> str | None:
    """A document's declared title, or None when it is a placeholder."""
    text = normalise(str(raw or ""))
    if not text:
        return None
    lowered = text.casefold()
    if lowered in _PLACEHOLDER_TITLES or lowered.startswith("microsoft word -"):
        return None
    # A filename is not a title, but it is what many writers put there.
    if lowered.endswith((".pdf", ".doc", ".docx")):
        return None
    return text
