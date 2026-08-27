# Test corpora

## `aveline/` — a synthetic company, built to break things

Eleven documents for a fictional European company, in Markdown, Word, Excel and
PDF. It exists so the whole pipeline can be run end to end against a live model
without needing anybody's real policies, and it is written around traps rather
than around volume.

| trap | where |
|---|---|
| A rule that is wrong without its condition | 20 weeks parental leave *after 12 months*; 12 weeks below |
| Two figures for one thing in one sentence | meals EUR 45 domestic / EUR 65 international |
| A prohibition with a real exception | alcohol not reimbursable *except* pre-approved client entertainment |
| Two current documents that disagree | expenses EUR 500 vs travel guidelines EUR 1,000 |
| A superseded copy still in the folder | `archive/expenses-policy-2023.md`, every figure different |
| Two deadlines that look like a contradiction and are not | 4h security incident vs 72h data breach |
| The same number for two unrelated things | EUR 500 expense threshold vs EUR 500 equipment allowance |
| Facts only in a table, a spreadsheet, or a PDF | retention periods, payment terms, recovery objectives |
| A near-miss between two true things | security training in 30 days generally, 2 weeks for new joiners |

The free passes should find **two contradictions and one duplicated pair** and
nothing else:

```bash
uv run openknowledge audit evals/corpus/aveline
```

If it flags the 4-hour and 72-hour deadlines, or links the two EUR 500 figures,
precision has regressed.

## What this is not

**It is not evidence about accuracy on real documents.** The corpus and the
golden set in `../golden-aveline/` were written together, which is exactly the
circularity that makes a benchmark flattering. What it can honestly show is that
the pipeline behaves: retrieval finds the evidence, conditions survive, negations
do not invert, contradictions refuse, and unanswerable questions get refused.

For anything stronger, point it at a folder of real policies. Everything in
[TEST-RUN.md](../../docs/TEST-RUN.md) works identically.

## Licence

The Aveline documents are original, written for this repository, and carry the
project's licence. Every organisation, person, figure and address in them is
invented.
