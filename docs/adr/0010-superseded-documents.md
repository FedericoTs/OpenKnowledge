# ADR 0010 — A document that declares itself superseded stops competing

**Status:** accepted · **Date:** 2026-08-28 · **Amends:** [ADR 0009](0009-salience-and-variants.md)

## Context

Real corpora keep their old policies. The archived copy is not a mistake —
audit obligations require it — so "delete the old file" is not advice a
knowledge tool gets to give. The Aveline corpus models this: an
`archive/expenses-policy-2023.md` whose header reads

> **Status:** SUPERSEDED by Expenses Policy v4.1 (March 2026). Retained for
> audit only.

ADR 0009 stopped this pair from opening two dozen blocking conflicts: the
variant grouping collapses it to one non-blocking `versions` conflict, owed to
an admin in /manage. That fixed refusals. It exposed the next failure:

**with nothing blocking, retrieval served both copies, and the model was left
to adjudicate.** Measured on the aveline golden set, the local model:

- answered "what is the CFO approval limit?" **leading with the archived
  EUR 15,000**, then disclosed the current EUR 25,000 as a discrepancy;
- answered the meal-allowance question correctly (EUR 45) but disclosed the
  archived EUR 35 — which the golden case forbids, because an employee
  reading "the sources disagree, 35 vs 45" about a settled question has been
  failed just more politely.

Both answers passed the grounding gate: every figure genuinely appears in the
cited text. Grounding checks faithfulness to retrieved context; it cannot
know one source had resigned.

## Decision

Three parts, all deterministic and free:

**Detection at parse time.** A document whose *head* (first 800 characters)
declares supersession — "SUPERSEDED by …", "Status: superseded", "retained
for audit only", "no longer in force", "this policy has been replaced" — is
stamped `Document.superseded`. Only self-declaration counts: no guessing
from dates, folder names, or version numbers. "Supersedes:" — the current
copy naming what it replaced — deliberately does not match, and neither does
a body mention of some other superseded document.

**Demotion at search time.** Chunks inherit the flag. When a query's ranking
contains any current chunk, superseded chunks are **dropped** from the
result, not downranked — a downranked stale figure still lands in the
model's context, and the model is then asked to settle a versioning question
retrieval already knows the answer to. When *only* superseded documents
match, they are served: a corpus whose sole document on a topic was retired
without replacement should answer from it, cited as what it is, rather than
refuse blind. In hybrid mode the demotion runs on the fused ranking, because
the dense half scores every chunk and would otherwise smuggle the archive
back in.

**Ingest stops paying for and alarming about them.** Drafting FAQ entries
from a superseded document spends money writing answers the current copy
contradicts, so it is skipped and said so in the ingest report. The
crosscheck that flags "a new document contradicts an approved answer" skips
them too: an archive copy arriving in the corpus disagreeing with current
answers is expected, not news.

What does **not** change: the document stays indexed, listed, viewable, and
part of conflict detection — the `versions` conflict still lands in /manage,
where a human decides which copy stands. Demotion decides retrieval, not
governance.

The cache key gains a retrieval-policy revision (`rp1`), because answers
produced while the archive still competed must not be served under the new
policy.

## Consequences

- The two archived-figure failures on the aveline golden set are fixed at
  the root: the model never sees the stale figure, so no prompt rule about
  version arbitration has to hold under pressure.
- A question explicitly about the old policy ("what was the 2023 meal
  rate?") now usually retrieves only the current copy and refuses honestly,
  because the current copy shares the topic vocabulary. This is the accepted
  cost: the product answers what the policy *is*; history lives in /manage
  and the document viewer.
- Detection is a vocabulary heuristic with the usual obligation: every
  phrase in the pattern earns its place from a document seen in the field,
  and a corpus that marks retirement some other way ("VERALTET", a date-only
  banner) is not covered until someone shows us one.
