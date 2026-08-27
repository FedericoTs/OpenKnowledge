# ADR 0006 — Detect prose contradictions deontically, and cross-check at the FAQ level

**Status:** accepted · **Date:** 2026-08-27

## Context

Numeric conflict detection (ADR 0005) catches a moved figure. It is blind to the class that
does the next most damage:

> "Contractors are eligible for parental leave." / "Contractors are excluded from parental leave."

Nothing numeric changed, and an employee acting on the wrong one has a real problem.

Re-verification covers some of this, but has a structural blind spot: it re-asks the approved
answers that *cite* a changed document. The most common way a contradiction enters a corpus
is somebody uploading a **new** document — which no existing answer cites, because it did not
exist. Nothing is re-asked, and the disagreement sits there.

Closing either gap with a model means comparing documents pairwise: O(N²), and a call per
pair.

## Decision

Two additions, both free.

**Extract rules deontically.** Policy prose says things *must*, *may*, or *must not* happen —
a small closed vocabulary. A rule extracts like a figure does: a marker, a force, and a
context window. A contradiction is "the same subject under a different force".

Two gates protect precision:
- **Predicate families.** Claims are only compared when they are about the same kind of rule.
  Reimbursement cannot contradict VPN access, whatever words they share.
- **Hard versus soft pairs.** FORBIDDEN against another force is a logical contradiction, so
  recognisably the same subject suffices. MANDATORY against PERMITTED is not — "must submit
  within 60 days" and "may submit online" are both true — so it requires near-identical
  context.

**Cross-check stored answers against new documents.** BM25 already knows for free which
questions a new document ranks for; a stored answer is short text with extractable claims. So
the comparison is claim-versus-claim, with no model anywhere in the path.

Detection quality is measured by `openknowledge eval-conflicts` against a labelled set that
is more than half near-misses, reporting precision and recall separately.

## Consequences

**Good.** The two most damaging contradiction classes are now caught for nothing, on every
re-index, deterministically — which means it also runs as a real evaluation in CI rather than
a smoke test. The FAQ cross-check closes the new-document gap that re-verification cannot
reach, at zero marginal cost, and reports findings in the form a human can act on ("your
approved answer says not allowed, this document says allowed"). Thresholds are tunable and
the effect on both metrics is measurable rather than guessed.

**Bad.** Deontic detection is English-only and pattern-based; a corpus written in another
language, or one that states rules without modal verbs, gets nothing from it. It cannot see
contradictions requiring world knowledge, or ones with no deontic marker ("the policy was
withdrawn in March"). Precision depends on a hand-tuned threshold pair validated against a
21-case set — small enough that a real corpus will find failure modes it does not contain.
The cross-check compares against retrieved passages rather than whole documents, so a
contradiction stated outside the retrieved window is missed.

**Load-bearing.** The labelled set must keep its majority of clean cases. Measuring only
recall produces a detector that flags everything, which is worse than one that flags nothing —
it gets switched off, and takes its real findings with it.

## Alternatives considered

- **Ask a model whether two documents contradict.** Higher recall, catches what patterns
  cannot. Rejected as the default on cost and shape: O(N²) pairwise, and it makes keeping the
  corpus current expensive, which is backwards. It remains the right tool for the residue,
  which is what re-verification is.
- **Flag any differing force on overlapping words.** Simpler, and what the first
  implementation did. Rejected on measurement: it flags "must submit within 60 days" against
  "may submit online", and a detector with that failure mode gets turned off.
- **Sentence embeddings with a contradiction classifier (NLI).** Better recall on paraphrase
  than patterns. Rejected for now: it adds a model download to first run, and NLI models are
  trained on general text rather than policy prose. Worth revisiting once hybrid retrieval
  brings a local embedding model in anyway.
