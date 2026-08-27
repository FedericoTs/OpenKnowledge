# ADR 0008 — Two PDF backends, preferring OpenDataLoader where a JVM exists

**Status:** accepted · **Date:** 2026-08-27 · **Amends:** [ADR 0007](0007-document-parsing.md)

## Context

ADR 0007 chose pdfplumber and recovered structure from geometry: headings from
type size, tables from ruling lines, reading order from vertical position. That
works, and it is roughly three hundred lines of heuristics standing in for
information the format sometimes already carries.

OpenDataLoader PDF (Apache 2.0) reports that information instead. It was
evaluated on a sibling project and rejected there for one reason: it is a Java
parser, and that project deploys to Vercel, where there is no JVM. OpenKnowledge
ships as a container, so the constraint that disqualified it does not apply.

Measured against the same document:

| | pdfplumber | OpenDataLoader |
|---|---|---|
| Heading levels | inferred from type size | **stated explicitly** |
| Tables | reconstructed from lines or alignment | **native cells with row and column spans** |
| Page numbers | tracked per page | per element |
| PDF/UA tagged documents | ignored | **read as true structure** |
| Borderless table (this fixture) | missed | missed |
| Runtime | pure Python | JVM + 23 MB JAR |

Neither handles the borderless case, so that is not the argument. The argument
is that heading levels and table cells are read rather than guessed, and that
a tagged PDF - which a good share of enterprise compliance documents are - stops
being inference altogether.

### Measured afterwards, on fifteen real contracts

The table above came from a synthetic fixture, which turned out to flatter
pdfplumber. Re-run against 15 real contracts and DPAs (204 pages):

| | OpenDataLoader | pdfplumber |
|---|---:|---:|
| Table rows | 813 | 798 |
| Documents with tables the other missed | 4 | 2 |
| Headings, uniformly-styled contract | 10 | 1 |
| Headings, bold-heavy DPA | 23 | 59 (over-detects) |

Two things this changed. First, the recommendation hardened: pdfplumber's
heading detection is wrong in both directions, so the backends are not peers and
the docs now say so. Second, it exposed a defect in ADR 0007's own PDF code -
the text-alignment table fallback fabricated 2,983 table rows across the corpus,
which is recorded below.

## Decision

**Keep both, choose at runtime.** `pdf_backend` is `auto` (default),
`opendataloader`, or `pdfplumber`.

`auto` uses OpenDataLoader when the wrapper is installed *and* `java` is on
PATH, and falls back to pdfplumber otherwise - including when OpenDataLoader
returns nothing, which costs one cheap second pass.

OpenDataLoader is an **extra**, not a core dependency; the container installs it
along with a headless JRE, and a bare `pip install` gets the pure-Python path.

The **JSON** output is consumed rather than the Markdown, because the Markdown
discards page numbers and heading levels - the two things worth having it for.

## Consequences

**Good.** Heading levels and table structure come from the document instead of
from heuristics, which is most of what ADR 0007's PDF code was doing by hand.
Tagged PDFs are read properly. The parser is deterministic by design, which
matters here more than usual: a corpus that fingerprints differently on each run
would invalidate every cached answer on a re-index that changed nothing. The
tool still installs and works with no JVM, so nothing regresses for a pip user.

**Bad.** The container carries a JRE (~180 MB) and a 23 MB JAR, and gains a
second runtime that can fail. Each PDF costs a subprocess and a temp directory
rather than an in-process call. Two backends is two code paths to keep working,
and the fixtures now have to exercise both.

**The sharp edge.** The backends extract slightly different text, so the *same*
corpus fingerprints differently under each. Answers regenerate rather than going
stale - the corpus version is part of the cache key, which is exactly what that
key is for - but moving a deployment between a machine with Java and one without
invalidates every cached answer, and the first day back is expensive. Deployments
that care should pin `OK_PDF_BACKEND` rather than leaving it on `auto`.

## What the measurement caught

ADR 0007 shipped a text-alignment fallback for borderless tables, guarded by a
numeric-column check. On real documents that guard was worthless: the fallback
found no genuine borderless table and turned prose into 2,983 fabricated rows -
151 of them from one three-page SLA - with words split mid-token and emitted as
labelled claims.

That is worse than the gap it was meant to close. A missed table costs recall; a
fabricated labelled claim corrupts the numeric claim extractor, contradiction
detection and the grounding gate simultaneously, because each one presents as a
fact with a subject attached. The fallback is removed, borderless tables are a
documented gap in both backends, and a regression test now generates a
prose-heavy PDF and asserts no table rows come back.

The general lesson is worth recording: the synthetic fixture said the fallback
was reasonable, and fifteen real documents said it was a liability. Parser
heuristics cannot be validated on documents you generated yourself.

## Alternatives considered

- **Replace pdfplumber entirely.** Simpler: one path, no divergence. Rejected
  because it makes a JVM mandatory, and a self-hosted tool that cannot be `pip
  install`ed and run loses the cheapest way for someone to try it.
- **Parse OpenDataLoader's Markdown.** Much less code. Rejected: it drops page
  numbers and heading levels, so citations would lose the locator that makes
  them checkable.
- **Reuse the sibling project's TypeScript integration.** Rejected on runtime -
  it shells out to the same JAR from Node, and adding Node to a Python container
  buys nothing. The Python wrapper ships the same CLI.
- **Keep geometry-only** (ADR 0007 as written). Still the fallback, and still
  usable. Rejected as the *preference* because inferring what a document states
  outright is work done twice, badly - and on real documents, done wrong.
- **`table_method=cluster`, `use_struct_tree`, `reading_order`.** All tested for
  the borderless case; none found it. `--hybrid` would, but it requires a running
  Docling or Hancom server, which breaks the promise that nothing leaves the
  machine. Not worth it for a gap both backends share.
