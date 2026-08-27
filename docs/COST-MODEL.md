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

## Where $0.10 per question comes from

| | tokens | rate | cost |
|---|---:|---:|---:|
| Retrieved context + system prompt | 15,000 in | $5.00 / M | $0.075 |
| Answer | 1,000 out | $25.00 / M | $0.025 |
| | | **total** | **$0.100** |

Nothing exotic: a frontier model, a fat context, no caching, on every call. The interesting
part is how little of that is buying anything.

## The levers, one at a time

At 2,000 questions/day over 250 working days:

| | per question | per year | what changed |
|---|---:|---:|---|
| Naive RAG | $0.10000 | $50,000 | — |
| \+ prompt caching | $0.09100 | $45,500 | fixed prompt read at ~0.1× |
| \+ tighter retrieval | $0.02350 | $11,750 | 6 chunks, not everything that scored |
| \+ smaller model | $0.00940 | $4,700 | grounded extraction is not a reasoning task |
| \+ pins and cache | $0.00517 | $2,585 | 45% of questions never reach a model |

**19× cheaper, no local model, no new hardware.**

Two things are worth noticing about the order.

**Caching is the smallest win here, not the biggest.** It saves 9%, because in this workload
the cacheable part (the system prompt) is small and the expensive part (retrieved context)
changes every call. Caching is close to free, so take it — but a project that stops there has
left almost everything on the table.

**Retrieval discipline is the biggest single lever: 4× on its own.** Sending 15,000 tokens
when 2,500 would do is not a modelling decision, it is a bug that bills you. This is the main
reason to invest in reranking: fewer, better chunks are cheaper *and* more accurate.

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

| questions/day | hardware/question | cascade total | vs. API-only ($0.00517) |
|---:|---:|---:|:--|
| 250 | $0.03840 | $0.04075 | **7.9× more expensive** |
| 1,000 | $0.00960 | $0.01195 | **2.3× more expensive** |
| 2,000 | $0.00480 | $0.00715 | **1.4× more expensive** |
| 5,000 | $0.00192 | $0.00427 | 1.2× cheaper |
| 10,000 | $0.00096 | $0.00331 | 1.6× cheaper |
| 25,000 | $0.00038 | $0.00273 | 1.9× cheaper |

*(a $1.20/hour GPU running 8h/day; cascade = 45% free, 45% local, 10% escalated to frontier)*

**Break-even is around 3,400 questions/day.**

This is the number most self-hosting pitches leave out, so to be direct about it: **below
roughly 3,400 questions/day, running your own model costs more per question than the API
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
  cost nothing but an admin's attention.

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
