# ADR 0007 — Parse documents into structured blocks, and chunk on that structure

**Status:** accepted · **Date:** 2026-08-27

## Context

OpenKnowledge could read `.txt` and `.md`. Every real corpus is PDF, Word and
Excel, so the tool could be evaluated on sample files and on nothing a company
actually has — which made every number in the project theoretical for the people
who would adopt it.

Two things had to be decided together. Getting text out of a PDF is the obvious
half. The half that matters for accuracy is what happens to it next: chunking by
word count cuts rules apart from their conditions, and the grounding gate cannot
detect that, because the answer it checks *is* faithful to the chunk it was
given. A dropped condition is invisible at every layer downstream.

## Decision

**Parse into blocks, not into text.** Every format reduces to an ordered list of
blocks — heading, paragraph, list item, table row — each carrying its heading
trail and a real locator (`p. 7`, `Limits!A7`, `slide 4`).

**Carry table headers onto every row.** `Grade: Senior | Limit: EUR 1,000`
rather than `Senior 1,000`, so a figure has a subject the claim extractors can
see.

**Chunk on structure.** A heading starts a chunk, atomic blocks are never split,
every chunk carries its heading trail. Word windows remain the fallback for
documents with no recovered structure.

**Report what could not be read.** Unsupported formats, scans and corrupt files
are named with a remedy, never silently skipped.

**Permissive dependencies only.** pdfplumber, python-docx, openpyxl, python-pptx.

## Consequences

**Good.** The tool can be pointed at a real corpus, which turns the project's
assumed numbers into measurable ones. Citations gain checkable locators — a
citation an employee cannot open is decoration. Table thresholds become claims
with subjects attached, so contradiction detection works on rows as it already
did on prose, and the numeric grounding check can tell EUR 1,000-for-Senior from
EUR 1,000-for-Junior. Heading trails improve lexical retrieval for free, because
section names are usually the words people search with.

**Bad.** Four new dependencies in the core install, and a container that now
needs `libgomp1`. PDF heading detection is inferential and produces one flat
section on a document that styles headings at body size. The unruled-table guard
requires a numeric column, so a purely textual table reads as prose. Spreadsheet
formulas are read as last-saved values, which can be stale. Parsing is
meaningfully slower than reading text, and it happens on every full re-index.

**Load-bearing.** Table rows must keep their headers. Dropping that to "simplify"
the renderer would silently degrade the numeric claim extractor, contradiction
detection and the grounding gate at once, and none of them would report an
error — they would just quietly get worse.

## Alternatives considered

- **PyMuPDF for PDF.** Faster and better. Rejected: AGPL, which is compatible
  with this project's licence but forecloses the commercial licensing that
  ADR 0002 exists to preserve. A dependency that quietly kills the business
  model is an expensive convenience.
- **OCR for scanned PDFs.** Would raise coverage. Rejected: a heavyweight
  dependency, and on a policy document it produces plausible figures with
  character errors — cited confidently. A project whose claim is that figures
  are checked should not introduce unchecked ones.
- **Reuse the parsing stack from dora-comply** (`pdfjs-dist`, `mammoth`,
  `exceljs`). Rejected on runtime: it is TypeScript, and adding Node to a Python
  container to run it is a bad trade for a self-hosted tool. Its *design*
  transferred instead — the per-page extraction for true page locators, the
  minimum-characters test for detecting scans, the character cap for garbled
  files, best-effort parsing that never throws, and the insight behind its
  DOCX-via-HTML approach, which exists because most extractors lose tables.
- **Keep word-window chunking.** Simpler. Rejected: it is the layer where a
  dropped condition becomes undetectable, so it is the wrong place to economise.
