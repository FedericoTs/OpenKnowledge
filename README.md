# OpenKnowledge

**A cheap, private, deterministic answer engine for your company's own documents.**

Self-hosted. Connects to SharePoint and Google Drive. Answers in Teams, Slack, or a web
widget. Runs on your own hardware, your own API keys, or both.

> **Status: pre-alpha (v0.1.0).** The cost architecture, the determinism layer, the
> grounding gate, the provider abstraction, and the evaluation harness are implemented
> and tested. Connectors beyond a local folder, the Teams channel, and the admin UI are
> scaffolded interfaces, not working integrations. See [ROADMAP](docs/ROADMAP.md).

---

## The problem

A mid-size company builds an internal document chatbot. It works. Then the bill arrives:
**around $0.10 per question.**

That is not a mystery, it is arithmetic. The usual build stuffs every retrieved document
chunk into a frontier model on every single call, with no caching:

| | tokens | rate | cost |
|---|---:|---:|---:|
| Retrieved context + system prompt | 15,000 in | $5.00 / M | $0.075 |
| Answer | 1,000 out | $25.00 / M | $0.025 |
| | | **total** | **$0.100** |

At 2,000 questions a day that is **$50,000 a year** to look things up in documents the
company already owns. And it still gets the answer subtly wrong sometimes, and gives two
different answers to the same question asked on Tuesday and Thursday.

The expensive part is not the intelligence. It's that nearly every one of those calls is
re-deriving an answer somebody already got last week.

## The idea

Most internal questions are not novel. In any company, a few hundred questions — expenses,
leave, onboarding, VPN, procurement thresholds — cover the large majority of traffic. Those
deserve to be **free and identical every time**. The genuinely hard, novel question deserves
a frontier model. The mistake is paying frontier prices for both.

So OpenKnowledge resolves each question at the cheapest tier that can answer it correctly,
and only escalates when it must:

| Tier | What it is | Marginal cost |
|---|---|---|
| **L0 · Pinned** | An admin wrote the canonical answer. Exact, by construction. | $0 |
| **L1 · Exact cache** | This question was answered before, under this same corpus. | $0 |
| **L2 · Semantic cache** | A near-identical question was, and the citations still check out. | $0 |
| **L3 · Local model** | Self-hosted model over retrieved context. No per-token invoice. | $0 marginal |
| **L4 · Frontier API** | The local answer failed the grounding check, or the question is hard. | full price |

Every tier answers from the same retrieved passages and passes the same grounding check, so
escalating changes the price, not the rules.

### What that's actually worth

Run `python tools/cost_model.py` and it prices each lever separately, from the same rate
table the running system uses. At 2,000 questions/day:

| | per question | per year |
|---|---:|---:|
| Naive RAG (today) | $0.10000 | $50,000 |
| \+ prompt caching | $0.09100 | $45,500 |
| \+ tighter retrieval (6 chunks, not everything) | $0.02350 | $11,750 |
| \+ mid-tier model instead of frontier | $0.00940 | $4,700 |
| \+ pins and cache (45% never reach a model) | **$0.00517** | **$2,585** |

**That is 19× cheaper with no local model and no new hardware** — just caching, retrieval
discipline, right-sized models, and not re-answering the same question twice.

### Where self-hosting actually helps — and where it doesn't

A local model has no per-token bill, but it does have a GPU behind it, and that cost is
*fixed*: it lands on every question whether or not anyone asks one. Carry it honestly and
the picture is less flattering than the usual pitch:

| questions/day | hardware/question | cascade total | vs. API-only |
|---:|---:|---:|:--|
| 1,000 | $0.00960 | $0.01195 | **more expensive** |
| 2,000 | $0.00480 | $0.00715 | **more expensive** |
| 5,000 | $0.00192 | $0.00427 | cheaper |
| 25,000 | $0.00038 | $0.00273 | cheaper |

*(a $1.20/hour GPU running 8h/day)*

**Break-even is around 3,400 questions/day.** Below that, running your own model costs more
per question than the API tier it replaces. It is still the right choice when documents must
not leave your network — but that is a **privacy** decision, and this project would rather
say so than sell a saving that isn't there. Above that volume, the fixed cost spreads thin
and the local tier wins by a widening margin.

Your own hit rates and hardware will differ, which is why `openknowledge costs` reports the
blended figure from the ledger rather than from this table.

### Is it actually right?

Cheapness is worthless if the answers are wrong, so correctness is measured rather than
asserted. `openknowledge eval` runs a golden set and reports accuracy **and** cost together —
either number alone is trivially gamed.

The metric that governs everything else is **false answers**: questions the corpus does not
cover that got an answer anyway. It is scored separately, and any increase fails the run
regardless of everything else. A bot that answers 95% of questions correctly and confidently
invents the other 5% is unusable, because nobody can tell which kind they are reading.

See [EVALUATION.md](docs/EVALUATION.md).

## Determinism

"The same question gets the same answer" is a hard requirement for policy and procedure
Q&A, and language models do not provide it. Temperature 0 helps and is not enough.

OpenKnowledge gets it from the cache key instead. Every answer is keyed on the question
plus **everything that produced it**:

```
sha256( canonical_question ‖ corpus_version ‖ prompt_version ‖ policy_version ‖ route_id )
```

Two consequences worth stating plainly:

- **Same inputs, same answer — byte for byte.** Not "usually". The second asker gets the
  first asker's answer.
- **Changed inputs, no stale answer.** Update the expenses policy in SharePoint and
  `corpus_version` changes, so every answer derived from the old text becomes unreachable
  in the same instant. A cache that can serve last year's rules is worse than no cache.

Question canonicalisation is deliberately conservative — it folds casing, whitespace, smart
quotes, and greetings, and it will never touch a word that could change meaning. `"which
expenses are not reimbursable"` must never collapse into `"which expenses are
reimbursable"`. That is why there is no stopword list; see
[`canonical.py`](src/openknowledge/canonical.py).

## Privacy

The default configuration sends **nothing** to anyone. Documents, index, cache, and logs
stay on the machine you run it on. Reaching a frontier model is opt-in, per-deployment, and
visible in the ledger. There is no OpenKnowledge-operated service to phone home to, and no
telemetry.

## Quick start

```bash
git clone https://github.com/FedericoTs/OpenKnowledge
cd OpenKnowledge
cp .env.example .env      # optional: add API keys for the escalation tier
docker compose up
```

Then open <http://localhost:8080> for the chat widget.

Pin the questions people actually ask — those become free and identical forever:

```bash
openknowledge top                     # what is asked most
openknowledge pin "How much parental leave do I get?" "20 weeks after 12 months." \
  --cite parental-leave --alias "what is the parental leave entitlement"
openknowledge costs                   # what it has cost so far
```

To develop against it directly:

```bash
uv venv && uv pip install -e ".[dev]"
uv run pytest
uv run openknowledge eval          # accuracy and cost, together
uv run uvicorn openknowledge.api.app:app --reload
```

## What's here now

```
src/openknowledge/
├── canonical.py     Query normalisation (conservative by design)
├── costs.py         Token accounting; prices from pricing.yaml
├── pricing.yaml     Rates, each with the date it was verified
├── types.py         Answer, Citation, Tier
├── cache/           Cache keys, pins, answer store, cost ledger
├── providers/       Anthropic (with prompt caching), OpenAI-compatible (incl. local)
├── cascade/         The router: try cheap, verify, escalate
├── retrieval/       Hybrid retrieval + the grounding gate
├── evaluation/      Golden set, scoring, baseline comparison
├── connectors/      SharePoint / Google Drive  (interfaces only)
├── channels/        Web / Teams / Slack        (web only)
└── api/             FastAPI app + admin endpoints
```

## Documentation

| | |
|---|---|
| [ARCHITECTURE.md](docs/ARCHITECTURE.md) | How the cascade fits together |
| [COST-MODEL.md](docs/COST-MODEL.md) | The arithmetic, and how to reproduce it |
| [DETERMINISM.md](docs/DETERMINISM.md) | What we guarantee, and what we don't |
| [EVALUATION.md](docs/EVALUATION.md) | How correctness is measured, and what fails a run |
| [ROADMAP.md](docs/ROADMAP.md) | What's built, what's next |
| [adr/](docs/adr/) | Why the load-bearing decisions went the way they did |

## Licence

[AGPL-3.0-only](LICENSE). Free to run, modify, and self-host, including commercially.
The AGPL's network clause means a hosted derivative must publish its source. Contributors
sign a [CLA](CLA.md) so the project can also offer commercial licences to organisations that
need different terms. See [ADR 0002](docs/adr/0002-license-agpl-cla.md).
