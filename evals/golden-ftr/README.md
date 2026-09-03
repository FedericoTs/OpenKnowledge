# An exam neither of us wrote

The other golden sets in this repository share a weakness the README admits:
the corpus and the questions were written together, so they prove the pipeline
behaves rather than that it is accurate on documents it has never seen. This
set exists to close that gap, and the first thing it did was produce a much
worse number.

## Provenance

**The corpus** — `documents/`, 20 files, 159 kB, 205 sections — is the US
Federal Travel Regulation: Title 41, Subtitle F, chapters 300 and 301
(glossary, and temporary-duty travel allowances). Fetched from the eCFR API on
2026-09-03:

```sh
curl -sL --compressed \
  "https://www.ecfr.gov/api/versioner/v1/full/2026-09-01/title-41.xml?subtitle=F&chapter=301"
```

A work of the US federal government, so it is public domain and committed here
in full. That matters: anybody can rerun this exact evaluation, which is not
true of an eval measured against documents that cannot be redistributed.

**The questions** are GSA's own per-diem FAQ, verbatim, from
<https://www.gsa.gov/travel/plan-a-trip/per-diem-rates/faqs>. GSA writes both
the regulation and the FAQ, so the questions *and* the reference answers come
from the body that made the rules — not from whoever is being graded.

## How a case was classified

Not by judgement. For each FAQ entry, search the corpus for the fact GSA's own
answer turns on:

* **present** → answerable, and `must_say` carries the regulation's figure;
* **absent** — GSA answered from its website, its rate-setting process, or law
  outside these chapters → refusal, and the only correct behaviour is to say
  the documents do not cover it.

Twelve of the nineteen are refusals. That is not a thumb on the scale; it is
what a real FAQ looks like beside a real regulation, and it is the half that
matters most, because a system that answers those is a system that invents.

## What it found in the exam itself

Three of the first run's failures were defects in *this file*, not in the
product, and all three were the same mistake: a `must_not_say` substring that
occurs inside correct answers as well as wrong ones.

| case | what I got wrong |
|---|---|
| `gsa-17-receipts` | forbade "receipts are not required", which a correct answer used for a true, scoped caveat from the regulation |
| `gsa-15-first-and-last-day` | forbade "100 percent", which the regulation itself uses for *full* days of travel |
| `gsa-16-mix-and-match` | classed as answerable, but GSA answers it from absence and the phrase is nowhere in the corpus — refusing is right |

Each correction is in `ftr.yaml` with the reasoning in the case's own `notes`,
and the number before and after is in the measured record. Writing a fair exam
turned out to be harder than answering it.

## Reproducing

```sh
export OK_DOCUMENTS_DIR=$PWD/evals/golden-ftr/documents
export OK_LOCAL_ENABLED=true OK_EMBEDDING_ENABLED=false OK_ESCALATION_ENABLED=false
export OK_LOCAL_BASE_URL=http://127.0.0.1:8082/v1
uv run openknowledge eval --path evals/golden-ftr/ftr.yaml --verbose
```

Run it both ways. That last sentence used to read "BM25 is the weaker of the
two configurations, so this is a floor" — and the hybrid run disproved it:
with embeddings on, which is the **default**, the same set scored 57% instead
of 71% and produced the one false answer in the record. It lost the incidental
-expenses definition and the no-hotel-at-per-diem case, and answered a
question about whether hotels may refuse the rate, which these chapters do not
address at all.

So neither number is a floor, and the default is currently the worse one. To
reproduce the default, serve `nomic-embed-text-v1.5` and set
`OK_EMBEDDING_ENABLED=true` with `OK_EMBEDDING_BASE_URL` pointing at it. Probe
the endpoint before trusting a run to it: "embeddings enabled" and "embeddings
working" are different claims, and a missing embedding model degrades to BM25
silently by design.
