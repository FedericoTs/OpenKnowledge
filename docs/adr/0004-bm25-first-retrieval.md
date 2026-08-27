# ADR 0004 — Lexical retrieval first, hybrid later

**Status:** accepted · **Date:** 2026-08-27

## Context

Retrieval quality drives both correctness and cost: it decides whether the answer is in the
context at all, and it is the single largest cost lever (sending 2,500 tokens instead of
15,000 is a 4× saving — see [COST-MODEL.md](../COST-MODEL.md)). The reflex choice is a vector
database and an embedding model.

## Decision

Ship BM25, in-process and dependency-free, as the v0 retriever, behind a `Retriever` protocol.
Add local dense retrieval and cross-encoder reranking as a fused hybrid next.

## Consequences

**Good.** No model download and no vector database on first run — OpenKnowledge answers
questions immediately after `docker compose up`, which matters enormously for a tool people
are evaluating. Lexical search is genuinely strong for this workload: people ask using the
exact nouns their company uses ("the T&E policy", "form RA-14", "GlobalProtect"), and BM25
matches those precisely where an embedding blurs them toward neighbours. It is also fully
inspectable, so a bad answer can be traced to a specific scoring decision. Deterministic
tie-breaking means identical questions retrieve identical context, without which the cache
would be unsound.

**Bad.** Pure lexical search misses paraphrases — "time off for a new baby" will not match a
document that only says "parental leave". That is a real quality gap today, and it is felt
most by exactly the casual phrasings a chatbot invites. Full re-indexing on every sync is
O(corpus). The in-memory index bounds corpus size to what fits in RAM.

**Mitigation.** The gap is bounded by the grounding gate: a missed document produces "I
don't know", not a wrong answer. That is the right failure to have while hybrid retrieval is
being built.

## Alternatives considered

- **Embeddings only.** Handles paraphrase well, handles exact identifiers badly, and adds a
  model download to first run. Worse than BM25 alone for this workload, not better.
- **Hosted vector database.** Rejected on the privacy requirement, and it would be an odd
  look for a project about not paying per query.
