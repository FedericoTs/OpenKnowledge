# Reading your documents

Until now OpenKnowledge could read `.txt` and `.md`. Real internal policy is a
PDF, a Word procedure, and an Excel table of limits — so the tool could be
evaluated on sample files and not on anything a company actually has.

| Format | | Locator in citations |
|---|---|---|
| `.pdf` | headings recovered from type size, ruled and unruled tables | `p. 7` |
| `.docx` | headings, lists, tables, in document order | heading trail |
| `.xlsx` / `.xlsm` | one labelled block per row | `Limits!A7` |
| `.pptx` | per slide, including speaker notes | `slide 4` |
| `.md` / `.txt` / `.rst` | headings, lists, pipe tables | heading trail |

Anything else is **named in the report**, not silently skipped:

```
$ openknowledge index
indexed 3 documents -> 5 chunks

2 file(s) contributed nothing:
  logo.png: no parser for .png
  old-handbook.doc: .doc is the pre-2007 Office format and cannot be read; re-save it as .docx
```

A document that contributes nothing without saying so is how a corpus develops a
hole that surfaces months later as a wrong answer. Both of the common causes — a
scanned PDF, a pre-2007 Office file — have a thirty-second fix if somebody is
told about them.

## Why tables get so much attention

Internal policy keeps its thresholds in tables, and the grounding gate is built
on figures being right. A table flattened into prose gives you this:

```
Grade Limit Notice Junior EUR 200 5 days Senior EUR 1,000 2 days
```

Six numbers with nothing attaching them to anything. The numeric claim extractor
reads them as unrelated figures, so it cannot tell you that EUR 1,000 belongs to
Senior — which means it cannot catch a model that says Junior. Every row is
therefore carried with its header:

```
Grade: Senior | Limit: EUR 1,000 | Notice: 2 days
```

Now the figure has a subject, the claim extractor sees `EUR 1,000` in a context
containing `senior`, and contradiction detection across documents works on rows
the way it already worked on prose.

## Structure is what makes chunking safe

Parsing produces blocks — heading, paragraph, list item, table row — each with
the trail of headings above it. Chunking then follows the document's own shape
rather than counting words:

- **A heading starts a chunk.** Content under different headings is about
  different things, and merging them produces a chunk whose heading contradicts
  half its body.
- **Atomic blocks are never split.** A table row cut in half is a number with no
  label.
- **Every chunk carries its heading trail**, so a passage retrieved alone still
  says what it is about.

That last point is worth being precise about, because it is an accuracy
property rather than a tidiness one. The grounding gate checks an answer against
the chunk it was given. If a window boundary landed mid-rule and dropped the
condition, the gate cannot tell — the answer *is* faithful to the chunk. Getting
the chunk right is the only place that failure can be prevented.

A document with no recovered structure still falls back to overlapping word
windows, so nothing regresses.

## PDFs, and what they cost

A PDF knows where glyphs sit on a page and nothing about headings or sections.
Three things are recovered from geometry:

**Headings, from type size.** The most common size on a page is the body; short
lines set materially larger are headings. This is what gives PDF blocks a
heading path at all.

**Tables, ruled or not.** Most policy PDFs rule their tables and those are found
reliably. Plenty do not, so there is a text-alignment fallback — guarded,
because on prose that strategy invents structure. The guard is that a genuine
policy table has at least one consistently numeric column; a row whose value
cell holds no digit is a sentence that wandered in. Without it, "Any single
expense above EUR 500 requires prior written approval" gets carved into a table
row reading `0 require | s prior written | approval.`

**Reading order, from vertical position.** Tables and prose are found by
separate passes and arrive in unrelated sequences. Sorting by position on the
page is what lets a table inherit the heading printed above it.

### No OCR, on purpose

A PDF with no text layer is reported as a scan and indexed as nothing. OCR would
mean a heavyweight dependency and, on a policy document, a plausible-looking
figure with a character error in it — `EUR 500` read as `EUR 600`, cited
confidently. Given that this project's whole claim is that figures are checked,
silently introducing unchecked ones is the wrong trade. Supply a text-based
copy.

## Dependency choices

All parsers are permissive — MIT or BSD-3 — and that is deliberate rather than
incidental.

**PyMuPDF is not used**, despite being faster and better at PDF than pdfplumber.
It is AGPL. That is compatible with OpenKnowledge's own licence, so it would
work today; it would also remove the option of selling a commercial licence,
which is the whole reason [ADR 0002](adr/0002-license-agpl-cla.md) put a CLA in
place. A dependency that quietly forecloses the business model is an expensive
convenience.

| | licence | |
|---|---|---|
| pdfplumber | MIT | PDF text, tables, glyph geometry |
| python-docx | MIT | Word |
| openpyxl | MIT | Excel |
| python-pptx | MIT | PowerPoint |

## Known limits

- **No OCR**, as above.
- **Headings in PDFs are inferred.** A document that sets its headings at body
  size with no bold gets one flat section. The blocks are still correct; they
  just carry a thinner heading trail.
- **Unruled table detection is conservative.** It requires a numeric column, so
  a purely textual table — a RACI matrix, say — is read as prose.
- **Spreadsheet formulas are read as their last computed value**, which comes
  from whatever the file last saved. An employee asking about the travel cap
  wants `EUR 500`, not `=B2*1.1`, but a workbook saved without recalculating
  will carry a stale number.
- **Password-protected files** are reported as unreadable.
- **Sheets are capped at 2,000 rows.** Someone's 200k-row export is data, not
  documentation, and indexing it would swamp the corpus.
