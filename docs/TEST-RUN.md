# The first real run

Everything in this repository is measured except the one thing that matters
most: **no model has ever been called.** Retrieval, chunking, contradiction
detection, cost arithmetic and the ladder are all measured. Draft yield, gate
pass rate, the escalation rate the whole cost model now turns on, and whether a
cheap model can actually ground an answer are all *assumed*.

This closes that. It takes about twenty minutes and costs a few cents.

You will end with three configurations measured side by side — self-hosted,
open-weight API, and frontier — against a corpus built to break them.

---

## 0. What you need

| | needed for | cost |
|---|---|---|
| A Together account | the open-weight rungs | a few cents for the whole run |
| An Anthropic account | the frontier rung | ~$0.50 for the whole run |
| Ollama | the self-hosted rung | free, needs ~5 GB disk |

You can do this with any one of them. Doing all three is what produces the
comparison, and it is the comparison that answers the question.

---

## 1. Install

```bash
git clone https://github.com/FedericoTs/OpenKnowledge
cd OpenKnowledge
uv venv && uv pip install -e ".[dev,anthropic]"
uv run pytest -q          # 416 passing before you change anything
```

If `uv` is not installed: `curl -LsSf https://astral.sh/uv/install.sh | sh`.

---

## 2. Point it at the test corpus

A corpus ships with this repository for exactly this purpose:
`evals/corpus/aveline` — eleven documents for a fictional European company, in
Markdown, Word, Excel and PDF. It is **synthetic and deliberately adversarial**:

| trap | where |
|---|---|
| A rule that is wrong without its condition | 20 weeks parental leave *after 12 months*; 12 weeks below that |
| Two figures for one thing in one sentence | meals EUR 45 domestic / EUR 65 international |
| A prohibition with a real exception | alcohol not reimbursable *except* pre-approved client entertainment |
| Two current documents that disagree | expenses says EUR 500, travel guidelines says EUR 1,000 |
| A superseded copy still in the folder | `archive/expenses-policy-2023.md`, every figure different |
| Two deadlines that look like a contradiction and are not | 4 hours security incident vs 72 hours data breach |
| The same number for two unrelated things | EUR 500 expense threshold vs EUR 500 equipment allowance |
| Facts only in a table, a spreadsheet, or a PDF | retention periods, payment terms, recovery objectives |
| Questions the corpus does not answer | sick leave, bonuses, pensions, dress code |

Because it is synthetic and the questions were written alongside it, **this
measures whether the pipeline works, not how accurate the product is on real
documents.** Do not quote its numbers as if they were the latter. When you have
a folder of real policies, point at that instead — everything below is identical.

```bash
export OK_DOCUMENTS_DIR=evals/corpus/aveline
```

### Check the free passes first — they cost nothing

```bash
uv run openknowledge audit evals/corpus/aveline
```

You should see **two contradictions** and **one duplicated pair**, and nothing
else. If it reports the 4-hour and 72-hour deadlines as a contradiction, or
links the two EUR 500 figures, something regressed.

```bash
uv run openknowledge index
uv run openknowledge eval --path evals/golden-aveline --dry-run
```

The dry run is the important one. It checks, **with no model and no cost**, that
every answerable case has its evidence in the retrieved context. Roughly half the
failures a fresh golden set produces are the set's own — a typo in `must_cite`, a
question worded unlike the corpus, a fact in a document retrieval never ranks —
and all of them read as model failures in the report. Get a `PASSED` here before
spending anything.

---

## 3. Rung one: a model on your own machine

Free, private, nothing leaves the box. Slower per answer, which is the trade.

```bash
# macOS / Linux
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen3:8b            # ~5 GB
ollama serve                    # leave running; it listens on 11434
```

Check it:

```bash
curl -s http://localhost:11434/v1/models | head -c 200
```

Then in `.env`:

```bash
OK_LOCAL_ENABLED=true
OK_LOCAL_MODEL=qwen3:8b
OK_LOCAL_BASE_URL=http://localhost:11434/v1
```

**Why `qwen3:8b`.** On Vectara's hallucination leaderboard Qwen3-8b sits at 4.8%
against 3.1% for a small frontier model on grounded summarisation — 1.7 points,
before the grounding gate and before escalation. A 4B fits in less RAM and costs
about a point more. Avoid picking a model on hallucination rate alone: one 4B
model scores well only because it declines to answer a third of the time.

OpenKnowledge prices this rung at **$0 per token**, because it derives from the
base URL that there is no invoice behind it. That is measured, not assumed — see
`tests/test_cheap_tier.py`.

Smoke test one question before going further:

```bash
uv run openknowledge ask "How much parental leave do I get?"
```

---

## 4. Rung two and three: open-weight models on Together

This is the configuration `COST-STRATEGIES.md` recommends: **$0.000316 per
measured question**, 116× cheaper than the frontier tier, through the *same
adapter* as Ollama. No new code, only configuration.

1. Sign up at <https://api.together.xyz>, create an API key.
2. Add a few dollars of credit. The whole run below costs cents.
3. In `.env`:

```bash
# Cheap rung: replaces Ollama, or use Ollama and put this on the ladder instead
OK_LOCAL_ENABLED=true
OK_LOCAL_MODEL=openai/gpt-oss-20b
OK_LOCAL_BASE_URL=https://api.together.xyz/v1
OK_LOCAL_API_KEY=<your together key>

# Middle rung: catches what the cheap one cannot ground, at $0.0009 not $0.037
OK_LADDER=["openai/gpt-oss-120b@https://api.together.xyz/v1"]
OK_LADDER_API_KEY=<your together key>
```

Because this endpoint **bills per token**, the cascade prices it at its real
rate rather than at zero. Rates for `gpt-oss-20b` and `gpt-oss-120b` are in
`pricing.yaml`, verified against together.ai/pricing on 2026-08-27. If Together
changes them, update that one file and every number in the project follows.

> If you point `local_base_url` at a vendor and see `cost not counted: no
> verified price for ... in pricing.yaml`, that is the system refusing to guess.
> Add the model to `pricing.yaml` with a `verified` date.

---

## 5. Rung four: the frontier

```bash
OK_ESCALATION_ENABLED=true
OK_ESCALATION_PROVIDER=anthropic
OK_ESCALATION_MODEL=claude-opus-5
OK_ESCALATION_EFFORT=low
OK_ANTHROPIC_API_KEY=<your anthropic key>
```

`low` effort is deliberate: this is grounded extraction from supplied passages,
not a reasoning task, and high effort buys nothing here while costing output
tokens.

Escalation is **off by default** in this project and stays off until you set that
flag. Nothing leaves the machine until an operator opts in.

Your ladder is now:

```
gpt-oss-20b → gpt-oss-120b → claude-opus-5
```

Confirm it before running — the engine logs the ladder at startup:

```bash
uv run openknowledge index 2>&1 | grep -i ladder
```

---

## 6. Put a budget on it

Before running anything against a paid key, cap it:

```bash
OK_BUDGET_DAILY_USD=2.00
OK_BUDGET_EXPECTED_QUESTIONS_PER_DAY=200
```

That is a ceiling of $0.01 per question. Spend ahead of pace and the expensive
rungs stop being tried; spend behind it and they come back. It limits
**escalation, never service** — the cheapest rung is always tried, and a question
the ceiling blocks is refused *naming the model it could not afford* rather than
answered from an attempt the gate rejected.

For the run below, $2.00 is far more headroom than you need. Set it anyway. It is
the difference between a mistake costing two dollars and a mistake costing two
hundred.

---

## 7. Run it

```bash
uv run openknowledge eval --path evals/golden-aveline --save-baseline evals/baseline-aveline.json
```

This asks all 26 cases, asks each answerable one twice to check determinism, and
asks each paraphrase. Read the report for four things, in this order:

**False answers.** Questions the corpus does not cover that got an answer anyway.
This is the metric that governs everything else, and any number above zero is the
result — a bot that invents 5% of its answers is unusable, because nobody can
tell which 5%.

**The contested case.** `contested-travel-threshold` must **refuse**, naming both
EUR 500 and EUR 1,000. Picking either one — even the more recent — is the failure
this project exists to prevent.

**Tier distribution.** How many questions each rung answered. This is the
escalation rate, measured for the first time. Everything in `COST-STRATEGIES.md`
turns on it.

**Cost per question.** From the ledger, including the free answers. A blended
cost that omits cache hits is meaningless.

```bash
uv run openknowledge costs          # what it actually cost, by tier and by model
uv run openknowledge conflicts      # the disagreements it found
```

---

## 8. Compare the three configurations

Run the same set three times, changing only which rung answers:

```bash
# self-hosted only
OK_ESCALATION_ENABLED=false OK_LADDER='[]' \
  uv run openknowledge eval --path evals/golden-aveline --save-baseline evals/run-local.json

# open-weight only
OK_ESCALATION_ENABLED=false \
  uv run openknowledge eval --path evals/golden-aveline --save-baseline evals/run-openweight.json

# the full ladder
uv run openknowledge eval --path evals/golden-aveline --save-baseline evals/run-ladder.json
```

Then compare any two:

```bash
uv run openknowledge eval --path evals/golden-aveline --baseline evals/run-local.json
```

The question this answers is the one the whole project rests on: **how much
accuracy does the cheap tier actually cost, and does the ladder buy it back?**

---

## 9. What to do with the numbers

Assumptions that become measurements:

| assumed today | measured by this run |
|---|---|
| 45% of questions answered without a model | the free share in the tier distribution |
| ~10% escalation rate | how often the cheap rung failed the gate |
| A small model can ground these answers | accuracy on the cheap-tier-only run |
| Drafting yields useful FAQ entries | `openknowledge learn` then `review` |
| 1,000 output tokens per answer | real output tokens in `costs --json` |

That last one matters more than it looks. The cost model holds answer length
constant at 1,000 tokens on every row because nothing in the architecture
shortens an answer — but nobody has checked what a grounded, cited answer from
six chunks actually costs. If it is 300 tokens rather than 1,000, every figure in
`COST-MODEL.md` moves, and it moves in our favour.

Send the numbers back and the docs get updated with measurements instead of
assumptions. That is the only remaining gap between "designed carefully" and
"shown to work".

---

## Troubleshooting

**Everything refuses.** No model is reachable. `openknowledge ask "test"` and read
the notes on the answer — they name the reason. The usual causes are Ollama not
running, `OK_ESCALATION_ENABLED` left false, or a key not picked up because `.env`
was not loaded.

**`cost not counted: no verified price`.** A model that is not in `pricing.yaml`.
Deliberate: reporting $0 for a call that cost money corrupts the ledger. Add the
model with a `verified` date.

**Answers are right but cite nothing.** The grounding gate rejects uncited
answers, so these show as refusals. Small models sometimes drop citation
formatting; check the rung's raw output, and if it is a formatting problem rather
than a grounding one, that is a prompt fix, not a model change.

**The run is slow on Ollama.** Expect 5–10 tokens/second on CPU for a 4–8B model,
so a 400-token answer takes about a minute. Twenty-six cases with determinism and
paraphrase checks is roughly 60 calls. Start with
`--only answerable --no-determinism` to halve it.

**Together returns 401.** The key goes in `OK_LOCAL_API_KEY` for the cheap rung
and `OK_LADDER_API_KEY` for the ladder rungs. They are separate on purpose — the
cheap rung is often a different provider from the ladder.
