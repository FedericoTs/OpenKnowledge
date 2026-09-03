# An exam nobody here wrote

The other golden sets in this repository have a problem their own README admits:
the corpus and the questions were written together, by the same person, in the
same afternoon. That proves the pipeline behaves. It does not prove the system
is accurate on documents it has never seen, and it is the first thing a buyer
will ask about.

This set fixes the provenance of both halves.

## The corpus

`documents/` is the **US Federal Travel Regulation** — Title 41, Subtitle F,
chapters 300 and 301 — fetched from the eCFR API on 2026-09-03:

```sh
curl -sL --compressed \
  "https://www.ecfr.gov/api/versioner/v1/full/2026-09-01/title-41.xml?subtitle=F&chapter=301"
```

Twenty documents, 159 kB, 205 sections: one Markdown file per part of the
regulation, section headings preserved. A work of the US government, so it is
public domain and committed here in full — this run is reproducible by anyone,
which is the point.

It is also the right *shape*. Thresholds with conditions attached ("more than
12 but less than 24 hours"), rules that point at other rules, near-miss
material a search can trip over (twenty-one parts covering allowances,
transportation, lodging and claims), and definitions that live in a glossary
three chapters away from the rule that uses them.

## The questions

GSA's own [per-diem FAQ](https://www.gsa.gov/travel/plan-a-trip/per-diem-rates/faqs),
verbatim. GSA writes the regulation *and* the FAQ, so both the questions and
the reference answers come from the body that made the rules — not from
whoever is being graded.

## How answerable and refusal were decided

Not by taste. For each FAQ entry, search the corpus for the fact GSA's own
answer turns on:

- **present** → answerable, and `must_say` carries the regulation's figure;
- **absent** — because GSA answered from its website, its rate-setting
  process, or law outside these chapters → refusal, and the only correct
  behaviour is to say the documents do not cover it.

Eleven of nineteen came out refusals. That is not a thumb on the scale; it is
what a real FAQ looks like beside a real regulation. It is also the half that
matters most, because a system that answers those is a system that invents.

## What the run does and does not establish

Retrieval ran **BM25 only**, with the embedding model switched off — the
weaker of the two shipped configurations. The pre-flight
(`openknowledge eval --dry-run`) confirmed every answerable case's evidence
reaches the context that way, so a failure in the live run is the model's and
not the corpus's. Hybrid retrieval can only do better at finding the passage.

Escalation was off and no API key was present, so **every answer came from the
free local tier** — a 4-bit Qwen3-4B on four CPU cores. That is the tier the
`$0.00000` claim is about.

It is one corpus in one domain, graded against one organisation's FAQ. It says
nothing about how the system reads a scanned PDF, a spreadsheet, or a corpus
that contradicts itself. Those need their own sets, and saying so is cheaper
than being caught.

## One thing found on the way

GSA's answer to "are lodging taxes included in the CONUS per diem rate" cites
FTR **301-11.27**. In the current eCFR text that rule is at **301-11.16**. The
fact is unchanged; the pointer has drifted — which is exactly the kind of rot
this product exists to notice, found by accident while building its exam.

## Rerunning it

```sh
OK_DOCUMENTS_DIR=evals/golden-ftr/documents \
OK_EMBEDDING_ENABLED=false \
  uv run openknowledge eval --path evals/golden-ftr/ftr.yaml --verbose
```
