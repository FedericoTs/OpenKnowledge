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
its documents decisively is guaranteed to find them among its candidates -
their best chunks rescued from below the cut, displacing the weakest
strangers. Nothing is filtered and nothing is reordered.**

Derivation is free, deterministic, and runs at index time - no model call,
because indexing must never spend money by surprise. Four sources, in the
order a human would trust them: the document id (which encodes the
operator's own folder taxonomy - "hr-expenses-policy" says hr, expenses,
policy), the title, the headings, and the eight body words most distinctive
against the rest of the corpus by tf-idf. Tags are stored as readable words
and shown in the document listing; they are folded (plural stripping plus
prefix truncation, the same `fold_word` conflict relevance uses) only for
matching, so "travelling" finds the document tagged "travel".

A route is a *guarantee of candidacy* - the third shape this took, each
stronger one rejected by a golden set within a run. The filter starved the
model: routed to a document that chunks to a single window, it saw a
one-chunk context and refused a question it answers happily with a fuller
one, while a paraphrase answered - exactly the inconsistency the paraphrase
check exists to catch. Routed-first ordering starved it differently: the
context filled with same-topic tables and the aveline set came back at
0.88, one question refused as scope-ambiguous, another answering with a
neighbouring row's figures. The mixed, score-earned context was doing
quiet work no design intuition predicted.

So rank is earned by score exactly as without tags, and the route's one
power is rescue: a named document with no chunk above the cut gets its
best-ranked chunk into the candidates. Measured on the shipped corpora,
every question retrieves an identical context with routing on and off,
because the named documents already rank - the rescue exists for the
thousand-document corpus where they will not. Demotion runs first, so a
superseded twin cannot be rescued back in.

The route itself is deliberately cowardly, because the catastrophic failure
is a question routed *away* from the document that held its answer:

* a document counts as matched only on **two** folded tag hits - one shared
  word ("policy") is coincidence;
* a route only forms when the matched set is a small share of the corpus
  (a third, with a floor of two so a document and its archived twin still
  count as decisive - superseded demotion settles that pair);
* no match, or any ambiguity, means **no route** - retrieval behaves
  exactly as it did before tags existed.

In hybrid mode the guarantee is applied to the fused ranking, after both
halves vote. Access control is unchanged and applied before the route, so
a rescue can only surface what the asker may already see.
`OK_TAG_ROUTING=false` restores pre-tag retrieval exactly, and the flag is
part of the cache key.

## Consequences

- Both golden sets hold their bars with routing on - accuracy 1.0, zero
  false answers, determinism 1.0 - and the eval preflight for both shipped
  corpus-and-set pairings runs in the unit suite. The two rejected designs
  are pinned as unit tests: a route never thins or reorders the context,
  and a buried named document is rescued into the cut.
- The listing answers "what will this be found by?" per file, which turns a
  retrieval mystery into something an operator can read.
- The honest claim is recall at scale, not speed: search time at the
  current corpus size is microseconds either way, and answer latency is
  dominated by generation. What the rescue buys is the named document
  reliably reaching the candidates on corpora large enough to bury it -
  fewer wrong-context answers, fewer grounding failures, fewer
  escalations to slower, costlier tiers.
- Thresholds (two hits, a third of the corpus, floor of two) are choices,
  measured only against the shipped corpora. Changing them means bumping
  the retrieval-policy marker in the cache key, and any tuning claim needs
  a corpus where routing actually fires differently.
