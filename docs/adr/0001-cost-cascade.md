# ADR 0001 — Resolve questions through a cost cascade, gated by grounding

**Status:** accepted · **Date:** 2026-08-27

## Context

A frontier-model RAG chatbot costs roughly $0.10 per question. Most of that spend is not
buying correctness — it is re-deriving answers that already exist, and paying a frontier rate
to read a paragraph aloud. In any company a few hundred questions cover most traffic.

The obvious fix — "use a cheaper model" — is the reason most teams don't do it. A small model
will write a fluent, confident, invented answer about your expenses policy, and the saving is
worthless if that ships.

## Decision

Resolve each question at the cheapest tier that can answer it, and make escalation a
**consequence of verification failure** rather than a guess about difficulty:

```
pinned → exact cache → semantic cache → local model → frontier API → refuse
```

Every tier answers from the same retrieved evidence under the same system prompt, and every
tier's answer is graded by the same grounding gate. A cheap answer that cannot be verified
never reaches the user; it triggers the next tier.

## Consequences

**Good.** The cost lever is the *share of traffic that never reaches a paid model*, which is
measurable and improvable, unlike "pick a better model". Escalation is a quality decision the
system makes from evidence, not a heuristic. Adding a tier does not change the safety story,
because the gate is shared.

**Bad.** Escalated questions pay twice — once for the failed cheap attempt (near-zero if the
cheap tier is local) and once for the frontier call. Latency on escalated questions is the
sum of both. The gate has false positives: a well-grounded paraphrase can be rejected and
escalated unnecessarily, which costs money rather than correctness.

**Load-bearing.** The gate is what makes the cascade a cost optimisation rather than a
quality regression. Weakening it to "improve the local hit rate" reverses the entire
trade-off, and should be treated as a change to this decision.

## Alternatives considered

- **One mid-tier model for everything.** Simpler, and about 10× cheaper than naive. Rejected
  as the *only* strategy because it still pays per question forever and gives up determinism
  — but note that most of the win is available this way, which the cost model states plainly.
- **Classify difficulty upfront, route accordingly.** Needs a classifier that is right about
  a question before seeing the documents, and a classifier error is silent. Verifying an
  answer is easier than predicting difficulty.
- **Cache only, no local tier.** Nearly as cheap below ~3,400 questions/day (see
  [COST-MODEL.md](../COST-MODEL.md)), but sends every cache miss to a third party — which the
  privacy requirement rules out as a default.
