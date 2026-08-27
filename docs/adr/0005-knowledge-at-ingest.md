# ADR 0005 — Draft answers at ingest; detect contradictions through answers, not documents

**Status:** accepted · **Date:** 2026-08-27

## Context

Two problems with one root. First, cost: every question pays at query time, forever, and the
bill grows with adoption — while the documents behind those answers change perhaps monthly.
Second, maintenance: keeping an FAQ current is a standing chore that nobody owns, and a
document that quietly contradicts an older one produces confidently wrong answers with a
citation attached.

The obvious fix for the second problem — have a model compare each uploaded document against
every other one — is O(N²). At 500 documents that is about $5.00 per upload, a price that
discourages keeping the corpus current, which is precisely backwards.

## Decision

**Move answer generation to ingest time, and anchor contradiction detection to answers rather
than to documents.**

- When a document is added or changed, draft FAQ entries from it. Charged once per document,
  not once per question.
- Every draft passes the same grounding gate a live answer does. Failures are discarded, not
  queued.
- A gate-passed draft **serves immediately**, marked unreviewed, at cache-tier trust. Human
  approval promotes it to a pin.
- Detect numeric conflicts between documents with no model at all — free, deterministic, every
  re-index.
- For prose changes, re-ask only the approved answers that **cite** the changed document.
- Refuse questions bearing on an unresolved conflict, naming both figures and both documents.

## Consequences

**Good.** The cost profile inverts: a recurring variable cost becomes a one-off fixed one.
Drafting 500 documents costs ~$6.06 against a payback of about a day, and the larger win is
cold start — a fresh deployment begins near its steady-state free share instead of climbing
there over months of expensive traffic. Contradiction detection costs ~$0.075 per upload
instead of ~$5.00, and produces a better artefact: "EUR 500 → EUR 1,000" rather than "these
two documents differ somewhere". Serving unreviewed drafts is not a new risk, because they
cleared the same checks as any cached answer — the reframing from "unvalidated content" to
"precomputed cache entry" is what makes that defensible.

**Bad.** Drafting spends tokens on questions nobody may ever ask; the per-run cap and the
demand-ranked queue bound that but do not eliminate it. Numeric conflict detection cannot see
prose contradictions ("contractors are eligible" vs "contractors are excluded") — those need
the re-verification path, which only covers questions someone already approved. Refusing on
conflict trades availability for correctness, and a false positive means a question that
could have been answered is not. The relevance matcher folds word prefixes, which is the one
place in the project that normalises beyond what query canonicalisation allows.

**Load-bearing.** Approval must remain the only path into the pin tier. If drafts were written
straight to pins, machine output would sit in the tier that bypasses verification, and the
safety model would invert.

## Alternatives considered

- **Generate everything, ask a human to validate the list.** The original shape of the idea.
  Rejected: nobody reviews three thousand items carefully, so it becomes rubber-stamping, and
  rubber-stamped output in the highest-trust tier is worse than no feature. The ranked queue
  plus gate-checked drafts gets the same benefit without depending on review actually
  happening.
- **Queue drafts without serving them.** Maximally conservative, and the posture
  `OK_SERVE_DRAFTS=false` still offers. Rejected as the default: the cost benefit would wait
  on a review backlog that may never clear, making the queue the bottleneck the feature was
  meant to remove.
- **Document-pair contradiction detection with a model.** Rejected on cost and on output
  quality — see above.
- **Answer contested questions from the newest document, with a warning.** Rejected:
  "most recently uploaded" is not "currently in force", so re-uploading an old policy would
  silently flip the answer.
