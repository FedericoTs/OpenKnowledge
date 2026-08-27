# Evaluation sets

Two sets, deliberately in separate directories because they have different
schemas and different runners. Mixing them in one folder means a loader either
picks up a file it cannot parse, or silently skips it - and a silently skipped
eval is one that stops testing without telling you.

| | | |
|---|---|---|
| `golden/` | question → expected answer and citations | `openknowledge eval` |
| `conflicts/` | document pairs → contradiction or not | `openknowledge eval-conflicts` |

`golden/` needs a model configured to mean anything. `conflicts/` does not - it
is deterministic and runs in CI.

Add files freely to either directory; both runners load every `*.yaml` they
find.

## `corpus/` and `golden-aveline/`

A synthetic company's policy set, and a golden set written against it, so the
whole pipeline can be run end to end against a live model without needing
anybody's real documents. Built around traps: conditions that must survive,
negations that must not invert, two documents that disagree, a superseded copy,
facts that only exist in a spreadsheet or a PDF, and eight questions the corpus
does not answer.

Because the documents and the questions were written together it measures
whether the *pipeline* works, not how accurate the product is on real documents.
Do not quote its numbers as the latter. See [corpus/README.md](corpus/README.md)
and [docs/TEST-RUN.md](../docs/TEST-RUN.md).
