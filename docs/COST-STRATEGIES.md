# The space of cost strategies

Every option that makes a grounded answer cheaper, what it costs, what it risks,
and whether it survives the constraint that matters: **it must not make answers
worse.**

Numbers come from `tools/cost_model.py` and `tools/measure_prompts.py`, against
the measured prompt (2,313 input tokens over a real corpus, 1,000 output tokens
assumed and held constant). Run them rather than trusting this page.

---

## The finding that reorders everything

Once the cheap tier is an open-weight model, the free share stops mattering and
**escalation is the entire remaining bill.**

| escalation rate | per question | per year |
|---:|---:|---:|
| 20% | $0.00742 | $3,712 |
| 10% | $0.00380 | $1,899 |
| 5% | $0.00199 | $993 |
| 2% | $0.00090 | $449 |
| 0% | $0.00017 | $87 |

*(45% free, `gpt-oss-20b` cheap tier, escalating to Opus 5, 2,000 questions/day)*

Raising the free share from 45% to **85%** saves **$63/year**. Cutting escalation
from 10% to **5%** saves **$906/year** — fourteen times more. This directly
contradicts what this project's own README said until now ("the free share is the
single number the whole cost model turns on"), which was true when the paid tier
was a frontier model and is false now.

**Why this is good news for accuracy.** Escalation fires when the cheap tier's
answer *fails the grounding gate*. So every lever that lowers the escalation rate
— better retrieval, reranking, contextual chunks, a gentler escalation ladder —
lowers it by making the cheap answer **actually correct more often**. Cost and
accuracy point the same way. That is rare, and it is where the work should go.

The free share still matters, just not for money: a pinned answer is
**deterministic and instant**, which is why it exists.

---

## 1. The cheap tier: what answers the question

Measured on the same prompt:

| model | per call | vs frontier | notes |
|---|---:|---|---|
| Opus 5 | $0.036565 | 1× | |
| Sonnet 5 | $0.014626 | 2× | |
| Haiku 4.5 | $0.007313 | 5× | |
| gpt-oss-120b | $0.000947 | 39× | Together |
| Qwen3.5 9B | $0.000643 | 57× | Together |
| DeepSeek V4 Flash | $0.000604 | 61× | Together |
| Llama 3 8B Lite | $0.000464 | 79× | Together |
| **gpt-oss-20b** | **$0.000316** | **116×** | Together |
| Self-hosted | $0.000000 | free/token | hardware is separate |

Rates verified against [together.ai/pricing](https://www.together.ai/pricing) on
2026-08-27 and recorded in `pricing.yaml` with that date.

**No new code is needed.** Together, Groq, DeepInfra, Novita and Fireworks all
expose `/v1/chat/completions`, so the existing OpenAI-compatible provider reaches
them. Point `local_base_url` at the vendor and set `local_model`.

**The accuracy question, answered with third-party numbers.** On Vectara's
hallucination leaderboard, which measures exactly this task — summarise a
document without departing from it:

| model | hallucination rate | factual consistency | answer rate |
|---|---:|---:|---:|
| Qwen3-8b | 4.8% | 95.2% | 99.9% |
| Mistral-Small-2501 | 5.1% | 94.9% | 97.9% |
| Qwen3-4b | 5.7% | 94.3% | 99.9% |
| Gemma-3-4b-it | 6.4% | 93.6% | **67.3%** |
| Gemini-2.5-flash-lite | 3.3% | 96.7% | 99.5% |
| GPT-5.4-nano | 3.1% | 96.9% | 100.0% |
| Finix-S1-32b | 1.8% | 98.2% | 99.5% |

A good small open model is **1.7 points** behind a small frontier model on
grounded faithfulness, and that is *before* the grounding gate, which catches
invented citations and unsupported numbers for free, and before escalation
catches the rest.

Note Gemma-3-4b's **67.3% answer rate** — it declines a third of the time.
Choosing a cheap tier on hallucination rate alone would pick the model that
refuses most. Both numbers have to be read together, which is why the evaluation
harness scores accuracy and abstention separately.

**The trap this exposed in our own code.** An open-weight endpoint reaches the
*local tier* through the *local adapter* but **bills per token**. `_price()` used
to map every local-tier call to the $0 entry, so the recommended configuration
would have silently reported $0 for calls that cost money. Fixed: the cascade now
prices by whether there is an invoice behind the endpoint, derived from the base
URL and overridable. See `tests/test_cheap_tier.py`.

---

## 2. Escalation: the thing that now dominates

| configuration | per question | per year |
|---|---:|---:|
| 45% free, mid-tier for everything else | $0.00804 | $4,022 |
| 45% free + Haiku, 10% → Opus | $0.00695 | $3,474 |
| 45% free + open-weight, 10% → Opus | $0.00380 | $1,899 |
| 45% free + open-weight, 10% → Sonnet | $0.00160 | $802 |
| 45% free + open-weight, 10% → gpt-oss-120b | $0.00024 | $118 |

**A ladder, not a cliff.** Today the cascade escalates local → frontier. Adding a
middle rung (20b → 120b → frontier) means most gate failures are caught by a
model that costs $0.0009 rather than $0.037. The 120b tier is *also* an accuracy
gain over the 20b, so the rung is not a compromise.

**What lowers the escalation rate** — every one of these raises accuracy too:

| lever | effect | cost | status |
|---|---|---|---|
| Reranking | 67% fewer retrieval failures combined with contextual retrieval | free on CPU; `bge-reranker-v2-m3` is Apache-2.0 | not built |
| Contextual retrieval | 49% fewer retrieval failures | one-off at ingest, on the cheap tier | not built |
| Hybrid BM25 + dense | the two catch different misses | free, local embeddings | not built |
| Better cheap-tier prompting | fewer gate failures | free | partly |

Anthropic's [contextual retrieval](https://www.anthropic.com/engineering/contextual-retrieval)
measurements: contextual embeddings cut retrieval failures 35%, adding contextual
BM25 gets to 49%, adding reranking gets to 67%. Our chunker already carries a
heading trail on every chunk, which is most of the contextual-embeddings idea
built for a different reason.

---

## 3. Retrieval: cheaper *and* more accurate

Retrieval discipline was measured at 2.5× on its own (13,097 → 2,313 tokens).
Going further needs better ranking, not a smaller `k`.

**Embeddings are effectively free.** `nomic-embed-text` runs at ~580 chunks/sec
on a laptop CPU in ~0.3 GB; `bge-m3` (568M, Apache-2.0) does dense, sparse and
ColBERT retrieval in one model across 100+ languages; `Qwen3-Embedding-0.6B`
scores 70.7 MTEB in ~1.5 GB. Our whole 476-chunk test corpus embeds in about a
second. Self-hosting embeddings only loses to an API above roughly 10–15M
embeddings/month, which no internal corpus reaches.

**Reranking costs milliseconds.** MiniLM-class cross-encoders do 100 documents in
50–80 ms; `bge-reranker-v2-m3` in 80–200 ms. Typical lift is 5–15 NDCG@10 points,
20+ on lexically hard sets. Six chunks is a tiny rerank job.

**No vector database is needed.** At corpus scale (thousands of chunks, not
millions) a local index in SQLite is enough, and it keeps the "one file, no
service" deployment that makes this runnable at home.

---

## 4. Caching, and why ours does nothing

**Prompt caching is inert here** — measured, not assumed. The cacheable part is
the static system prompt at **476 tokens**, under Anthropic's **512-token** floor.
The `cache_control` marker is placed correctly and returns nothing. The real
opportunity is caching *documents* for a hot corpus, which is a different design.

**Semantic caching is the honest version of raising the free share.** Production
hit rates are 20–45%, not the 95% the marketing suggests. Threshold behaviour is
well characterised: ~0.92–0.93 gives 20–30% hit rate at acceptable precision;
0.97 gives ~5%; 0.88 gives real correctness risk — *"how do I reset my password"*
matching *"how do I change my email"*.

The literature's answer is **asynchronous** verification: serve the cached answer,
verify in the background, demote it if wrong. **That is wrong for this project.**
A wrong policy answer served now and corrected later has already been acted on.

We can do better, and cheaply, because we already have a **free synchronous
verifier**: retrieve for the *new* question, then run the existing grounding gate
on the *cached* answer against those fresh chunks. If the cached answer is still
supported by what the new question retrieves, serve it; otherwise treat it as a
miss. No model call, deterministic, and it catches the password/email case
directly. The paraphrase-consistency cases in the golden set are already the
labelled pairs needed to calibrate the threshold.

**Batch API** gives 50% off for anything not interactive — drafting the FAQ at
ingest, re-verification after a document changes. Free saving, no accuracy cost,
already priced in `pricing.yaml` via `batch_multiplier`.

---

## 5. Self-hosting: when it actually wins

With a cheap tier at $0.000316/question, **a GPU no longer pays for itself on
cost.** A $1.20/hour GPU at 8h/day is $2,400/year; the same volume on
`gpt-oss-20b` is $158/year at 100% paid share. Self-hosting is now a **privacy**
purchase, full stop — which is a perfectly good reason, and the project should say
so rather than dress it as a saving.

**But CPU-only self-hosting is real, and nearly free.** A 4B model on a modern CPU
generates 5–10 tokens/sec; a 30B MoE with ~3B active does 12–15 tok/s on 32 GB.
With short answers, a €7/month VPS or an idle office box handles a few hundred
questions a day — which covers a great many companies, at essentially zero
marginal cost and with nothing leaving the network. That is the "at home"
configuration, and it is the one worth proving.

The honest limit: at 2,000 questions/day with 45% going local, CPU inference does
not keep up. Above a few hundred a day, either buy the GPU (for privacy) or use
the open-weight API (for cost).

---

## 6. SharePoint and Microsoft 365: the constraint, costed

**Microsoft charges nothing for any of this.**

| | cost | notes |
|---|---|---|
| Microsoft Graph API | **free** | needs an Entra app registration and the M365 licences you already have |
| SharePoint enumeration + download | **free** | via Graph `/drives/{id}/root/delta` |
| Teams bot, standard channel | **free**, unlimited | premium channels are $0.50/1k messages; Teams is not one |
| Azure Bot Service | **free** tier | the bot itself is hosted by you |

**Delta queries are the important part.** `delta` returns only what changed since
a token, so a nightly or 15-minute sync reads changes rather than the corpus. That
matches this project's design exactly: re-index on change, draft once, never
re-read a document that did not move. Delta is not throttle-exempt — back off on
429, and pair it with change notifications rather than tight polling.

**`Sites.Selected` is the permission to ask for**: access to named sites only,
rather than every site in the tenant. Least privilege is also what makes the ACL
mapping tractable — and item permissions, group expansion and inheritance are the
real work in the connector, not the file download.

**One thing to plan around:** the Bot Framework SDK is archived, replaced by the
**Microsoft 365 Agents SDK**. A Teams channel should be written against the
current SDK, not the tutorials.

---

## Built since this was written

**The escalation ladder.** `OK_LADDER` puts rungs wherever an operator wants:

```
self-hosted → gpt-oss-20b → gpt-oss-120b → claude-opus-5
```

Every rung answers from the same passages under the same system prompt and is
graded by the same grounding gate — the invariant that makes climbing safe. A
rung's answer either passes the gate or nobody sees it, so adding a cheap rung
can lower the bill but cannot lower the standard. End to end on a scripted run:

| what happened | answered by | cost |
|---|---|---:|
| cheap rung grounds it | gpt-oss-20b | $0.00019 |
| cheap fails, middle catches it | gpt-oss-120b | $0.00076 |
| both fail, frontier catches it | claude-opus-5 | $0.02076 |

The middle rung is the point: a grounding failure that used to cost $0.021 now
costs $0.0008, and it is an accuracy gain over the rung below it rather than a
compromise.

**The budget governor.** `OK_BUDGET_DAILY_USD` turns a declared cap into a
ceiling on what one question may cost, recomputed from the ledger every time:

```
ceiling = budget remaining today ÷ questions still expected today
```

Spend ahead of pace and the ceiling drops, so expensive rungs stop being tried.
Spend behind it and it rises again. Nothing is scheduled and the arithmetic is
one division an operator can check. Three properties it deliberately has:

- **The first rung is never withheld.** A budget limits escalation, not service.
  A deployment that stops answering because it is 3% over pace has turned a cost
  control into an outage.
- **Refusal, never a guess.** When the ceiling blocks the rungs that could have
  grounded an answer, the question is refused *and says which model it could not
  afford*. Serving the cheap rung's rejected attempt is the one thing this
  project will not do.
- **Budget refusals are not cached**, so the question is retried freshly once the
  ceiling recovers. Answers that were *served* are unaffected — they are cached
  under a key that does not include spend.

**Reranking.** Free, deterministic, no model, on by default. It fixes three BM25
failures that are all recall failures — and a recall failure becomes a gate
failure, which escalates:

| | distinct docs in top 6 | slots taken by the dominant doc | near-duplicate pairs |
|---|---:|---:|---:|
| BM25 top-6 | 3.83 | 2.75 | 0.08 |
| \+ structural rerank | **4.42** | **1.92** | **0.00** |

*(15 real contracts, 476 chunks, 12 questions —
`tools/measure_retrieval.py`, recorded in `evals/measured/`)*

It uses the heading trails ADR 0007 already extracts, caps how many slots one
document may take, and drops windows that restate a neighbour. It is **not** a
cross-encoder and claims none of a cross-encoder's gains; the `Reranker`
protocol is what one would plug into. And these are **coverage** numbers, not
accuracy: they show the reranker does what it claims, not that answers improved.
That still needs a labelled set and a live model.

## What is next, in order

1. **A live run.** Every one of these levers is now measurable and none of the
   answer-side ones have been measured, because no model has been called. This
   is the only thing standing between "designed correctly" and "shown to work".
2. **Open-weight cheap tier as the documented default** — 116× cheaper than
   frontier, no new code, 1.7 points behind on faithfulness before the gate.
   Needs the live run to confirm the gate pass rate holds.
3. **Synchronously verified semantic cache** — reuse the grounding gate rather
   than the literature's async design.
4. **Hybrid retrieval + contextual chunks** — 49% fewer retrieval failures.
5. **A cross-encoder reranker** behind the existing protocol, for deployments
   willing to carry the dependency.
6. **SharePoint connector on Graph delta + `Sites.Selected`** — free, and the
   thing that makes any of it reachable.

Batch pricing for ingest work is already available and should just be used.

## Sources

- [Together AI pricing](https://www.together.ai/pricing) — open-weight rates, verified 2026-08-27
- [Vectara hallucination leaderboard](https://github.com/vectara/hallucination-leaderboard) — faithfulness by model
- [Anthropic, Contextual Retrieval](https://www.anthropic.com/engineering/contextual-retrieval) — 49% / 67% failure reduction
- [Microsoft Graph delta query](https://learn.microsoft.com/en-us/graph/delta-query-overview) and [throttling limits](https://learn.microsoft.com/en-us/graph/throttling-limits)
- [Closing the Calibration Gap in Semantic Caching](https://arxiv.org/abs/2606.19719)
- [Asynchronous Verified Semantic Caching for Tiered LLM Architectures](https://arxiv.org/pdf/2602.13165)
- [Best embedding models for RAG, 2026](https://www.premai.io/blog/best-embedding-models-for-rag-2026-ranked-by-mteb-score-cost-and-self-hosting/) — local embedding throughput
- [Reranker benchmarks](https://futureagi.com/blog/best-rerankers-for-rag-2026/) — CPU latency and NDCG lift
