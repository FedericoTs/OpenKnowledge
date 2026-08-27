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
  correct. Rejected as the *preference* because inferring what a document states
  outright is work done twice, badly.
