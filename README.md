# OpenKnowledge

[![Latest release](https://img.shields.io/github/v/release/FedericoTs/OpenKnowledge?label=release&color=2f6b4f)](https://github.com/FedericoTs/OpenKnowledge/releases/latest)
[![CI](https://github.com/FedericoTs/OpenKnowledge/actions/workflows/ci.yml/badge.svg)](https://github.com/FedericoTs/OpenKnowledge/actions/workflows/ci.yml)
[![Licence: AGPL-3.0](https://img.shields.io/badge/licence-AGPL--3.0-blue)](LICENSE)

**A cheap, private, deterministic answer engine for your company's own documents.**

It answers from your files, shows the passage it answered from, refuses when the
documents do not say, and gives the same answer to the same question every time.
Self-hosted, on your own hardware or your own API keys. Runs as a one-click Windows
app or as one server the whole company reaches from a browser.

![A question about parental leave, answered from the policy with the passage shown, by a local model for $0.00, with the share of the wording found in its sources](https://raw.githubusercontent.com/FedericoTs/OpenKnowledge/HEAD/docs/images/chat.png)

**Measured, not asserted.** Every number here comes from a script in this repository
(`evals/measured/`), run against a 4-bit Qwen3-4B on four CPU cores:

| | |
|---|---:|
| accuracy on the golden set | **100%** (17 answerable cases) |
| false answers on the must-refuse set | **0** (9 cases) |
| same answer, asked twice | **100%** |
| cost per question on the local tier | **$0.00000** |

The corpus and the questions were written together, so this proves the pipeline behaves,
not that it is accurate on your documents; that run is one command away, below. What a
real deployment costs, and why the usual build costs ten cents a question, is worked
through under *The problem*.

## Try it in sixty seconds

**Windows.** Download the [latest installer](https://github.com/FedericoTs/OpenKnowledge/releases/latest),
run it, and drop documents into the folder it opens. It is not yet code-signed, so
SmartScreen will ask first; every release's notes say exactly how the installer was
verified before it was published.

**Anywhere with Python.** Find where your documents disagree, with no model, no key, and
nothing uploaded anywhere:

```bash
uvx --from git+https://github.com/FedericoTs/OpenKnowledge openknowledge audit ~/policies
```

Then a model on your machine and the chat: see [Quick start](#quick-start).

![The management page: whether each model endpoint answers, what the install has cost, what people ask most](https://raw.githubusercontent.com/FedericoTs/OpenKnowledge/HEAD/docs/images/manage.png)

**Status.** Pre-1.0 and shipping weekly; the badge above is the current release. Built,
tested and measured: document parsing (PDF, Word, Excel, PowerPoint, Markdown), the cost
cascade, the determinism layer, the grounding gate, hybrid retrieval, drafting and review,
contradiction detection with a notion of scope, supersession, streaming chat, browser
uploads, per-folder access rules with Entra ID sign-in, a management page (costs, most
asked, gaps, wrong answers, pins, health, configuration, backup, admin log) and a Windows
installer that upgrades itself in CI on every release. Not yet: the paid tiers measured
(needs an API key), connectors beyond a folder (SharePoint, Google Drive, Teams), and a
signed installer. See [ROADMAP](docs/ROADMAP.md).

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
| **L2 · Drafted** | Answered when the document was uploaded, gate-checked, marked unreviewed. | $0 |
| **L3 · Local model** | Self-hosted model over retrieved context. No per-token invoice. | $0 marginal |
| **L4 · Frontier API** | The local answer failed the grounding check, or the question is hard. | full price |

Every tier answers from the same retrieved passages and passes the same grounding check, so
escalating changes the price, not the rules.

### What that's actually worth

These are **measured**, not modelled. `tools/measure_prompts.py` parses a real corpus —
15 third-party vendor contracts, SLAs and DPAs, around 100 pages — retrieves for real
questions, assembles the exact prompt that would be sent, and counts it. Only the answer
length is assumed, and it is held constant on every row so it cancels out.

| retrieval discipline | input tokens | $/question | $/year |
|---|---:|---:|---:|
| Whole corpus in context | 126,234 | $0.65617 | $328,085 |
| Top 40 chunks — *"keep everything that scored"* | 13,097 | $0.09048 | $45,242 |
| Top 20 chunks | 6,679 | $0.05840 | $29,198 |
| Top 10 chunks | 3,575 | $0.04288 | $21,438 |
| **Top 6 chunks — OpenKnowledge default** | **2,313** | **$0.03657** | **$18,282** |

The second row is the diagnosis, confirmed independently: a build that retrieves generously
over real documents lands at **$0.090 per question**. That is where the $0.10 comes from.

Then the remaining levers, at the measured prompt size:

| | per question | per year |
|---|---:|---:|
| Generous retrieval | $0.09048 | $45,242 |
| \+ prompt caching — *inert, see below* | $0.09048 | $45,242 |
| \+ tighter retrieval (6 chunks, not everything) | $0.03657 | $18,282 |
| \+ mid-tier model instead of frontier | $0.01463 | $7,313 |
| \+ pins and cache (45% never reach a model) | **$0.00804** | **$4,022** |

**11× cheaper with no local model and no new hardware** — retrieval discipline, right-sized
models, and not re-answering the same question twice.

That figure used to read 19×, from assumed token counts. Measuring found three errors that
had been compounding: the model assumed a 2,000-token cacheable system prompt (the real one
is **476 tokens, under Anthropic's 512-token floor, so it caches nothing at all**), assumed
a 4,500-token prompt at six chunks (real: 2,313), and quietly shortened the answer from
1,000 tokens to 400 in the same row as the retrieval change — crediting retrieval with an
output reduction it does not cause. 11× is lower and true. `tools/cost_model.py` now reads
the measurement file and prints which numbers are measured and which are assumed.

The other thing measurement showed: once retrieval is tight, **the answer is the expensive
part.** At 6 chunks on a frontier model, context costs $0.0116 and the answer costs $0.025.
Caching the input cannot fix that, and that is exactly why the architecture is a cascade —
the levers that work are a smaller model and not calling one at all.

### And a smaller model is very small indeed

The same measured prompt, priced across the tier the cascade can actually use:

| model | per call | vs frontier |
|---|---:|---|
| Opus 5 | $0.036565 | 1× |
| Sonnet 5 | $0.014626 | 2× |
| Haiku 4.5 | $0.007313 | 5× |
| **gpt-oss-20b** (Together) | **$0.000316** | **116×** |
| Self-hosted | $0 / token | hardware is separate |

Open-weight models on serverless providers speak `/v1/chat/completions`, so they reach
OpenKnowledge through the same adapter as a self-hosted model — configuration, not code. On
Vectara's hallucination leaderboard a good small open model (Qwen3-8b, **4.8%**) sits **1.7
points** behind a small frontier model (**3.1%**) on exactly this task, before the grounding
gate and before escalation.

At that price the whole argument inverts. Raising the free share from 45% to 85% saves
**$63/year**; cutting the escalation rate from 10% to 5% saves **$906**. Escalation fires
when the cheap tier fails the grounding gate — so better retrieval, reranking and a gentler
escalation ladder make answers **more accurate and cheaper at the same time**.

The whole menu, costed and sourced: **[COST-STRATEGIES.md](docs/COST-STRATEGIES.md)**.

### So the cascade is a ladder, and it has a budget

`OK_LADDER` puts as many rungs as you like between the cheap tier and the frontier:

```
self-hosted → gpt-oss-20b → gpt-oss-120b → claude-opus-5
```

Every rung answers from the same passages under the same system prompt and is graded by the
same grounding gate. That invariant is why climbing is safe rather than a quality gamble: a
rung's answer either passes the gate or nobody sees it, so a cheaper rung can lower the bill
but cannot lower the standard.

| what happened | answered by | cost |
|---|---|---:|
| cheap rung grounds it | gpt-oss-20b | $0.00019 |
| cheap fails, middle catches it | gpt-oss-120b | $0.00076 |
| both fail, frontier catches it | claude-opus-5 | $0.02076 |

`OK_BUDGET_DAILY_USD` then turns a declared cap into a ceiling on what one question may
cost — *budget remaining ÷ questions still expected today*, recomputed from the ledger on
every question. Spend ahead of pace and the expensive rungs stop being tried; spend behind it
and they come back. It limits **escalation, never service**: the cheapest rung is always
tried, and a question the ceiling blocks is refused *naming the model it could not afford*
rather than answered from an attempt the gate rejected.

```
claude-opus-5 not tried: forecast $0.04052 exceeds the $0.00020 budget ceiling
the rungs that might have grounded this were withheld by the budget ceiling; it
recovers as spending falls back on pace, and this refusal is not cached
```

### Retrieval that stops wasting slots

BM25 scores every chunk on its own, so its top 6 can be six views of one paragraph. That is a
recall failure, and a recall failure becomes a gate failure, which escalates — the expensive
thing. Reranking is on by default, free, deterministic and model-less:

| | distinct docs in top 6 | slots taken by dominant doc | near-duplicate pairs |
|---|---:|---:|---:|
| BM25 top-6 | 3.83 | 2.75 | 0.08 |
| \+ structural rerank | **4.42** | **1.92** | **0.00** |

*(15 real contracts, 476 chunks — `tools/measure_retrieval.py`)*

It caps how many slots one document may take, drops windows that restate a neighbour, and
lets a matching heading trail count for more than an incidental mention — using the structure
[ADR 0007](docs/adr/0007-document-parsing.md) already extracts. It is **not** a cross-encoder
and claims none of a cross-encoder's gains. And these are *coverage* numbers: they show the
reranker does what it claims, not that answers improved. That needs a live model.

### Where self-hosting actually helps — and where it doesn't

A local model has no per-token bill, but it does have a GPU behind it, and that cost is
*fixed*: it lands on every question whether or not anyone asks one. Carry it honestly and
the picture is less flattering than the usual pitch:

| questions/day | hardware/question | cascade total | vs. API-only ($0.00804) |
|---:|---:|---:|:--|
| 1,000 | $0.00960 | $0.01326 | **more expensive** |
| 2,000 | $0.00480 | $0.00846 | **more expensive** |
| 5,000 | $0.00192 | $0.00558 | cheaper |
| 25,000 | $0.00038 | $0.00404 | cheaper |

*(a $1.20/hour GPU running 8h/day)*

**Break-even is around 2,200 questions/day.** Below that, running your own model costs more
per question than the API tier it replaces. It is still the right choice when documents must
not leave your network — but that is a **privacy** decision, and this project would rather
say so than sell a saving that isn't there. Above that volume, the fixed cost spreads thin
and the local tier wins by a widening margin.

Your own hit rates and hardware will differ, which is why `openknowledge costs` reports the
blended figure from the ledger rather than from this table.

### It reads the documents you actually have

PDF, Word, Excel, PowerPoint and Markdown, parsed into structure rather than
flattened into text — headings, lists, and tables, each block carrying its heading trail
and a citable locator (`p. 7`, `Limits!A7`, `slide 4`).

Tables get particular care because that is where policy keeps its thresholds. Flattened,
a limits table reads as `Grade Limit Notice Junior EUR 200 5 days` — six numbers attached
to nothing, which the claim extractor cannot check. Every row is carried with its header
instead:

```
Grade: Senior | Limit: EUR 1,000 | Notice: 2 days
```

Chunking then follows the document's own shape: a heading starts a chunk, a table row is
never split, every chunk keeps its heading trail. This is an accuracy property, not a
tidy one — the grounding gate checks an answer against the chunk it was given, so a
window boundary that dropped a condition is invisible to every check downstream.

Files that cannot be read are named with a remedy rather than silently skipped:

```
2 file(s) contributed nothing:
  logo.png: no parser for .png
  old-handbook.doc: .doc is the pre-2007 Office format; re-save it as .docx
```

PDFs get two parsers, and they are not peers. Where a JVM is available — as in the
container — it uses [OpenDataLoader](https://github.com/opendataloader-project)
(Apache 2.0), which *reports* heading levels and table cells instead of inferring them,
and reads PDF/UA tags as true structure. Measured on 15 real contracts and DPAs it found
the section structure of a uniformly-styled agreement where the pure-Python path found
one heading in the whole document. That path exists so a bare `pip install` works, not
because it is equivalent.

See [DOCUMENTS.md](docs/DOCUMENTS.md).

### Try it before you configure anything

The cheapest useful thing here needs no API key, no model, no GPU and no database:

```bash
openknowledge audit ./policies
```

It reads the folder, extracts every figure and every stated rule, and tells you where your
own documents disagree with each other — quoting both sentences, so the finding is checkable
without opening either file. Nothing is written, nothing leaves the machine, and it exits
non-zero on findings, so it also works as a CI step on the repository where your policies
live.

```
OpenKnowledge audit - /srv/policies
23 document(s), 991 claim(s) checked, 0 model calls, $0.00

2 contradiction(s), in 1 document pair(s):

  1. expenses-policy vs travel-guidelines   (figure, 71% context match)
     [expenses-policy] says EUR 500
       "Travel above EUR 500 requires prior approval from a line manager."
     [travel-guidelines] says EUR 1,000
       "Travel above EUR 1,000 requires prior approval from a line manager."

1 pair(s) look like duplicated documents:

     register-2024 and register-2025 look like two versions of the same document:
     98 of the 154 figures they share disagree. Retire one rather than reconciling
     them line by line.
```

### Maintenance is a one-off at upload

When a document arrives or changes, OpenKnowledge drafts the FAQ from it, discards anything
that fails the grounding gate, and puts the rest to you ranked by what approving each one
saves. Drafting 500 documents costs about **$6.06, once** — after which those questions are
free for as long as the document stands.

It also notices when documents contradict each other, and three of the four passes are free:

| Pass | Catches | Cost |
|---|---|---|
| Numeric claims | A moved figure — threshold, deadline, allowance | free |
| Deontic claims | A changed permission — *eligible* → *excluded*, *required* → *optional* | free |
| FAQ cross-check | A **new** document disagreeing with an existing answer | free |
| Re-verification | The residue, on documents that changed | ~$0.075/upload |

The free passes run on every re-index. The paid one costs $0.075 instead of the $5.00 that
comparing an upload against a 500-document corpus would, because it re-asks only the approved
answers that *cite* the changed document — and it names the claim that moved rather than
pointing at two files.

A contested question is refused rather than guessed:

```
Your documents disagree on this, so I won't guess:
  - [expenses-policy] says EUR 500, [expenses-policy-2026] says EUR 1,000
Please ask your administrator which one currently applies.
```

See [KNOWLEDGE.md](docs/KNOWLEDGE.md).

### It has now been run against a real model

Everything above was measured with scripted providers until Qwen3-4B (Q4_K_M) was pointed at
the test corpus on four CPU cores — 10 documents across Markdown, Word, Excel and PDF:

| | |
|---|---:|
| accuracy | **100.0%** (17 answerable cases) |
| false answers | **0** (9 must-refuse cases) |
| determinism | **100.0%** |
| paraphrase consistency | **100.0%** |
| cost per question | **$0.00000** |

Tiers: 17 answered locally, 8 refused, 1 refused as contested — the live disagreement between
two documents, declined rather than guessed.

**What that is and is not.** The corpus and the questions were written together, which is the
circularity that makes a benchmark flattering, so this is evidence that the *pipeline*
behaves — retrieval finds the evidence, conditions survive, negations do not invert,
contradictions refuse, unanswerable questions get declined — not that the product is accurate
on your documents. It is also a 4-bit 4B model, so read it as a floor rather than a target.

The golden set is checked to be capable of failing: every answerable case is fed its own
plausible wrong answer, correctly cited, and must reject it. A test asserts this, because a
set that passes everything might mean the model is right or might mean the set is blind.

**Five runs before this one found four real bugs**, none of which 450+ unit tests had caught:
a contested-claim gate that a wordier question could slip past, a contested refusal scored as
a fabrication, a golden set requiring two spellings of one number at once, and figures
matching inside larger figures. That is the argument for a live run, made concretely.

See [TEST-RUN.md](docs/TEST-RUN.md) to reproduce it, including the appendix for running the
local tier through llama.cpp with no Ollama.

### Is it actually right?

Cheapness is worthless if the answers are wrong, so correctness is measured rather than
asserted. `openknowledge eval` runs a golden set and reports accuracy **and** cost together —
either number alone is trivially gamed.

The metric that governs everything else is **false answers**: questions the corpus does not
cover that got an answer anyway. It is scored separately, and any increase fails the run
regardless of everything else. A bot that answers 95% of questions correctly and confidently
invents the other 5% is unusable, because nobody can tell which kind they are reading.

Contradiction detection carries its own numbers, because it fails in two directions:

```
openknowledge eval-conflicts
  precision    100.0%  (of what it flagged, how much was real)
  recall       100.0%  (of what was real, how much it flagged)
```

Recall protects the employee who would otherwise be told the superseded policy. Precision
protects the feature — an admin who sees three bogus flags stops reading the fourth, and a
detector at 100% recall and 40% precision is switched off within a week. This one needs no
model, so unlike the golden set it runs as a real evaluation in CI.

**And then it was run on 100 pages of real contracts, where it produced 320 findings and
nothing useful.** A curated set of short policy pairs cannot see what a real folder does to
a detector: dense boilerplate reads as shared subject matter, and two copies of one document
read as ninety-eight separate contradictions. Weighting shared words by how rare they are,
and grouping findings by document pair, took that to six findings and six correctly named
duplicate pairs — on the same corpus, with the labelled set still at 100/100. The numbers,
what did not work, and what is still broken are in
[KNOWLEDGE.md](docs/KNOWLEDGE.md#what-a-labelled-set-could-not-tell-us).

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
- **Changed inputs, no stale answer.** Update the expenses policy in the document
  folder and `corpus_version` changes, so every answer derived from the old text becomes
  unreachable in the same instant. A cache that can serve last year's rules is worse than no cache.

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
curl -fsSL https://raw.githubusercontent.com/FedericoTs/OpenKnowledge/HEAD/install.sh | sh
```

Clones into `~/Documents/Projects/OpenKnowledge`, builds a virtualenv inside that folder,
and proves the install by running the audit against this repository's own test corpus.
No `sudo`, nothing installed system-wide, and deleting the folder uninstalls it. Prefer to
read it first? `git clone`, then `./install.sh` — same result.

Then, in order:

```bash
openknowledge audit ~/policies        # free: where your documents disagree, no model
openknowledge model use qwen3:8b      # a model on your machine, so answers cost nothing
openknowledge serve                   # chat widget on http://localhost:8080
```

The first of those needs no model, no key and no GPU, and writes nothing — you can point
it at a folder you would never upload anywhere, because nothing is uploaded.

Long documents want a bigger context window than a model's default. Ollama's
OpenAI-compatible endpoint has no field for that, so `--context` builds a copy of the
model carrying it, and records the window so a prompt that would not fit is refused
rather than silently truncated:

```bash
openknowledge model use qwen3:30b --context 131072
```

Pin the questions people actually ask — those become free and identical forever:

```bash
openknowledge learn                   # draft answers from your documents
openknowledge review                  # approve them, most valuable first
openknowledge conflicts               # documents that disagree with each other
openknowledge costs                   # what it has cost so far
```

Or run the whole thing in a container, with Java present for the better PDF backend:

```bash
cp .env.example .env      # optional: add API keys for the escalation tier
docker compose up
```

Step by step, including model sizes and what to do when something fails:
**[LOCAL-SETUP.md](docs/LOCAL-SETUP.md)**.

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
├── retrieval/       BM25 + structural reranking + the grounding gate
├── documents/       PDF, Word, Excel, PowerPoint, Markdown -> structured blocks
├── knowledge/       Draft at ingest, review queue, conflict detection
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
| [DOCUMENTS.md](docs/DOCUMENTS.md) | What formats are read, and why tables get special care |
| [LOCAL-SETUP.md](docs/LOCAL-SETUP.md) | Install, pick a model, size its context window, run it |
| [TEST-RUN.md](docs/TEST-RUN.md) | Step by step to your first real run against a live model |
| [web/site/](web/site/) | The public page, and how to serve it with or without a backend |
| [COST-STRATEGIES.md](docs/COST-STRATEGIES.md) | Every cost lever, costed, and whether it hurts accuracy |
| [KNOWLEDGE.md](docs/KNOWLEDGE.md) | Drafting at upload, review, and contradictions |
| [EVALUATION.md](docs/EVALUATION.md) | How correctness is measured, and what fails a run |
| [ROADMAP.md](docs/ROADMAP.md) | What's built, what's next |
| [RELEASING.md](docs/RELEASING.md) | How a version ships: the installer, the GitHub release, PyPI |
| [adr/](docs/adr/) | Why the load-bearing decisions went the way they did |

## Licence

[AGPL-3.0-only](LICENSE). Free to run, modify, and self-host, including commercially.
The AGPL's network clause means a hosted derivative must publish its source. Contributors
sign a [CLA](CLA.md) so the project can also offer commercial licences to organisations that
need different terms. See [ADR 0002](docs/adr/0002-license-agpl-cla.md).
