# Cost model

Every number here comes from [`tools/cost_model.py`](../tools/cost_model.py), which prices
scenarios using the same `openknowledge.costs` module and `pricing.yaml` rates the running
system uses. Run it instead of trusting the tables:

```bash
uv run python tools/cost_model.py            # default: 2,000 questions/day
uv run python tools/cost_model.py 10000      # your volume
```

Rates carry a `verified` date. Slots we have not verified ship **without numbers**, and
`cost_usd()` raises rather than reporting $0 for a call that really cost money.

**Token counts are measured, not assumed.** They come from
[`evals/measured/real-contracts.json`](../evals/measured/README.md), produced by
`tools/measure_prompts.py` against 15 real third-party vendor contracts, SLAs and DPAs —
around 100 pages. It parses the corpus, retrieves for real questions, assembles the exact
prompt the running system would send, and counts it. Point it at your own folder:

```bash
uv run python tools/measure_prompts.py --corpus ./policies --questions questions.txt
```

Only the answer length is assumed, and it is held constant on every row so it cancels out of
every comparison. Every run prints which numbers came from measurement and which did not.

## Where $0.10 per question comes from

| | tokens | rate | cost |
|---|---:|---:|---:|
| Retrieved context + system prompt | 15,000 in | $5.00 / M | $0.075 |
| Answer | 1,000 out | $25.00 / M | $0.025 |
| | | **total** | **$0.100** |

Nothing exotic: a frontier model, a fat context, no caching, on every call. The interesting
part is how little of that is buying anything.

That table was originally reverse-engineered from a reported figure. It has since been
**confirmed against real documents**: a build that retrieves generously — top 40 chunks,
keep everything that scored — over 100 pages of real contracts measures **13,097 input
tokens and $0.09048 per question**. The diagnosis was right, and it is no longer a guess.

| retrieval discipline | input tokens | $/question | $/year |
|---|---:|---:|---:|
| Whole corpus in context | 126,234 | $0.65617 | $328,085 |
| Top 40 chunks | 13,097 | $0.09048 | $45,242 |
| Top 20 chunks | 6,679 | $0.05840 | $29,198 |
| Top 10 chunks | 3,575 | $0.04288 | $21,438 |
| **Top 6 chunks — the default here** | **2,313** | **$0.03657** | **$18,282** |

Same corpus, same prompt, same model, same assumed answer length. One variable: how many
chunks were sent.

## The levers, one at a time

At 2,000 questions/day over 250 working days:

| | per question | per year | what changed |
|---|---:|---:|---|
| Generous retrieval | $0.09048 | $45,242 | top 40 chunks over a real corpus |
| \+ prompt caching | $0.09048 | $45,242 | **inert: 476-token prompt, 512-token floor** |
| \+ tighter retrieval | $0.03657 | $18,282 | 6 chunks, not everything that scored |
| \+ smaller model | $0.01463 | $7,313 | grounded extraction is not a reasoning task |
| \+ pins and cache | $0.00804 | $4,022 | 45% of questions never reach a model |

**11× cheaper, no local model, no new hardware.**

Three things are worth noticing, and the first only became visible on measurement.

**Prompt caching contributes nothing.** Not "a little" — nothing. The cacheable part is the
static system prompt, which measures **476 tokens**, and Anthropic declines to cache a prefix
under **512**. The `cache_control` marker in the provider is placed correctly and returns
zero, silently, forever. This is the failure mode caching has: everything keeps working and
the bill is just higher. Padding the prompt to clear the floor would earn about $0.002 a
question and is the wrong instinct; the real caching opportunity is the *documents*, which
is per-document caching for hot corpora, and is on the roadmap rather than in the numbers.

**Retrieval discipline and model choice are now the same size: 2.5× each.** Sending 13,097
tokens when 2,313 would do is not a modelling decision, it is a bug that bills you — and
this is the main reason to invest in reranking, because fewer, better chunks are cheaper
*and* more accurate. But it no longer dominates, which leads to the third point.

**Once retrieval is tight, the answer is the expensive part.** At six chunks on a frontier
model the context costs $0.0116 and the answer costs $0.0250 — output is 68% of the bill.
No amount of input-side cleverness touches that. The only two levers that reach it are a
smaller model and not calling one at all, which is the entire argument for the cascade.

### Why not cache the retrieved context too?

Tempting, and wrong for the default case. The retrieved chunks differ per question, so
marking them for caching pays the ~1.25× write premium on bytes nobody reads back — a pure
surcharge that looks like an optimisation in the code and shows up as a larger bill. This is
also why the Anthropic provider does not use top-level automatic caching: it places its
breakpoint at the end of the prompt, which here is the unique question.

Per-document caching for a small set of hot documents is a real optimisation and is on the
roadmap. It is not the default because it only pays when the same documents are retrieved
repeatedly within the cache TTL.

## The local tier, costed honestly

A self-hosted model has no per-token invoice. It has a GPU, and that cost is *fixed*: it is
incurred whether or not anyone asks a question. Divided across question volume:

| questions/day | hardware/question | cascade total | vs. API-only ($0.00804) |
|---:|---:|---:|:--|
| 250 | $0.03840 | $0.04206 | **5.2× more expensive** |
| 1,000 | $0.00960 | $0.01326 | **1.6× more expensive** |
| 2,000 | $0.00480 | $0.00846 | **1.1× more expensive** |
| 5,000 | $0.00192 | $0.00558 | 1.4× cheaper |
| 10,000 | $0.00096 | $0.00462 | 1.7× cheaper |
| 25,000 | $0.00038 | $0.00404 | 2.0× cheaper |

*(a $1.20/hour GPU running 8h/day; cascade = 45% free, 45% local, 10% escalated to frontier)*

**Break-even is around 2,200 questions/day.**

This is the number most self-hosting pitches leave out, so to be direct about it: **below
roughly 2,200 questions/day, running your own model costs more per question than the API
tier it replaces.** It can still be the right call — if documents must not leave your
network, the hardware is buying privacy and the price is reasonable. But it is a privacy
purchase, not a saving, and OpenKnowledge would rather say that than sell a number that
doesn't survive arithmetic.

Above that volume the fixed cost spreads thin and keeps improving, which is why the cascade
is built to support both shapes rather than assuming one.

### Moving your own break-even

- **Cheaper hardware.** The break-even scales linearly with the hourly rate. On-prem
  amortised capex is often well under a cloud GPU rate.
- **Don't run it 24/7.** Internal question traffic is bursty and office-hours shaped. An
  instance that sleeps outside working hours cuts the fixed cost directly.
- **Raise the free share.** Every point of pin or cache hit rate lowers both paths, and pins
  cost nothing but an admin's attention. Worth much less than it sounds once the cheap tier
  is an open-weight model — see [COST-STRATEGIES.md](COST-STRATEGIES.md), where raising the
  free share 45% → 85% saves $63/year and cutting escalation 10% → 5% saves $906.

## The cheap tier changes which lever matters

Everything above prices the paid tier as a frontier or mid-tier model. An open-weight model
on a serverless provider costs **$0.000316** for the same measured prompt — 116× less than
Opus 5 — through the same OpenAI-compatible adapter, with no new code.

At that price the arithmetic inverts: the free share stops being the dominant term and
**escalation becomes the entire remaining bill**. The full menu of strategies, what each is
worth, and the third-party accuracy numbers that decide whether a cheap tier is safe, are in
[COST-STRATEGIES.md](COST-STRATEGIES.md).

## Measure, don't estimate

The ledger records every answered question, **including the free ones** — a blended cost per
question is meaningless if cache hits are missing from the denominator.

```bash
openknowledge costs
```

```
1,284 questions · $2.9140 total
blended cost: $0.00227 per question

tier          questions   share       spend
pinned              412     32%$     0.0000
exact               338     26%$     0.0000
local               410     32%$     0.0000
frontier            124     10%$     2.9140

68% of questions were answered without calling a model.
```

That last line is the one to watch. It is the number that drives everything else, and it is
the one you can improve directly — with `openknowledge top` to find what to pin, and better
retrieval to raise the share the local tier can carry.
