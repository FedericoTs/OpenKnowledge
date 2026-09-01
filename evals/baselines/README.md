# Baselines

What a run of each golden set measured, so the next run can be compared
against it rather than eyeballed:

```sh
uv run openknowledge eval --path evals/golden-aveline \
  --baseline evals/baselines/aveline-qwen3-4b-instruct.json
```

That fails on a regression — accuracy down, a false answer appearing,
determinism down, cost up beyond a margin — and says which. It is the gate a
retrieval, prompt, chunking or grounding change has to clear, and it exists so
"nothing got worse" is a check rather than an opinion.

## What produced these

| | |
|---|---|
| model | Qwen3-4B-Instruct-2507-Q4_K_M, the chat model the desktop app ships |
| embeddings | nomic-embed-text-v1.5-Q4_K_M, likewise |
| runtime | llama.cpp via an OpenAI-compatible server, 8192-token window |
| profile | `self-hosted` from `../profiles.yaml` — local tier only, no ladder, no escalation |
| corpora | `golden` against `./documents`; `golden-aveline` against `../corpus/aveline` |
| hardware | 4 CPU cores, no GPU |

Both sets: **100% accuracy, no false answers, 100% determinism and paraphrase
consistency, $0.00**.

## What a baseline is not

**Not a claim about the product's ceiling.** This is a 4B model, where
`profiles.yaml` defaults to `qwen3:8b`. A number here measures this model on
this corpus; it says nothing about the paid rungs, and the escalation rate the
cost model turns on is still unmeasured — that needs an API key and is the
project's top open item.

**Not portable.** Sampling and quantisation differ between runtimes, so a
baseline is only meaningful against the same model and settings. Regenerate
rather than compare across a change of either, and name the file after what
produced it.

**Not a substitute for the pre-flight.** `--dry-run` checks every answerable
case has its evidence in the retrieved context, free and without a model. Run
it first: it is seconds, and it separates "the model got it wrong" from "the
corpus was never pointed at the right folder". Skipping it once here turned a
misconfigured `OK_DOCUMENTS_DIR` into forty minutes and eighteen failures that
meant nothing.
