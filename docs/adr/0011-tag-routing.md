# ADR 0011 — Tags at upload, and the radius they are allowed to shrink

**Status:** accepted · **Date:** 2026-08-28

## Context

Upload is the moment the system gives a document undivided attention, and
everything expensive in this design is pushed there: parsing, chunking,
conflict detection, FAQ drafting. Retrieval, by contrast, treats every
question as global - scored against every chunk of every document. At eleven
documents that costs nothing. At a thousand, it costs precision before it
costs time: chunks from unrelated documents crowd the top k, the grounding
gate then fails answers that would have been clean over the right context,
and every needless gate failure is a needless escalation - which is where a
tuned deployment's remaining money goes.

The operator also has no way to see what a document will be *found by*. The
listing shows filenames and sizes; whether "expenses" questions will land on
the right file is a matter of faith until someone asks one.

## Decision

**Every indexed document gets a derived tag set, and a question that names
its documents decisively is searched only against those documents.**

Derivation is free, deterministic, and runs at index time - no model call,
because indexing must never spend money by surprise. Four sources, in the
order a human would trust them: the document id (which encodes the
operator's own folder taxonomy - "hr-expenses-policy" says hr, expenses,
policy), the title, the headings, and the eight body words most distinctive
against the rest of the corpus by tf-idf. Tags are stored as readable words
and shown in the document listing; they are folded (plural stripping plus
prefix truncation, the same `fold_word` conflict relevance uses) only for
matching, so "travelling" finds the document tagged "travel".

Routing is deliberately cowardly, because the catastrophic failure is a
question routed *away* from the document that held its answer:

* a document counts as matched only on **two** folded tag hits - one shared
  word ("policy") is coincidence;
* the restriction applies only when the matched set is a small share of the
  corpus (a third, with a floor of two so a document and its archived twin
  still count as decisive - superseded demotion settles that pair);
* no match, or any ambiguity, means **no restriction** - retrieval behaves
  exactly as it did before tags existed.

In hybrid mode the same route filters both halves; the dense half scores
every chunk by cosine and would otherwise smuggle excluded documents back
into the fused ranking. Access control is unchanged and applied inside the
route; tag routing can only ever narrow, never widen, what an asker may see.
`OK_TAG_ROUTING=false` restores pre-tag retrieval exactly, and the flag is
part of the cache key.

## Consequences

- Both golden sets hold their bars with routing on - accuracy 1.0, zero
  false answers, determinism 1.0 - and the eval preflight now runs in the
  unit suite, pinning that no golden case's evidence is ever stranded
  outside a route.
- The listing answers "what will this be found by?" per file, which turns a
  retrieval mystery into something an operator can read.
- The honest claim is precision at scale, not speed: search time at the
  current corpus size is microseconds either way, and answer latency is
  dominated by generation. The speed that matters arrives indirectly -
  fewer off-topic chunks means fewer grounding failures means fewer
  escalations to slower, costlier tiers.
- Thresholds (two hits, a third of the corpus, floor of two) are choices,
  measured only against the shipped corpora. Changing them means bumping
  the retrieval-policy marker in the cache key, and any tuning claim needs
  a corpus where routing actually fires differently.
