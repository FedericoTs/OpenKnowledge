# ADR 0009 — Weight shared words by rarity, and group findings by document pair

**Status:** accepted · **Date:** 2026-08-27 · **Amends:** [ADR 0006](0006-deontic-contradictions.md)

## Context

ADR 0006 decided contradictions between documents by asking whether two claims
are about the same subject, and answered that by counting the content words their
contexts share. That is measured by `eval-conflicts` against 21 labelled cases,
where it scores 100% precision and 100% recall.

The detector had never been run on a real corpus. Pointed at 15 real vendor
contracts and DPAs — around 100 pages, from different counterparties, including
two parallel copies of seven documents — it emitted **320 findings, none of them
useful.**

Reading them showed two failures, neither of which is a badly chosen threshold.

**Boilerplate reads as shared subject.** Contracts overlap enormously on *party*,
*agreement*, *notice*, *written*, *service*, *shall*. Counting shared words treats
those as evidence of a shared subject, which is how

> "Buyer and Licensor may be referred to collectively as the Parties"

scored **75%** against

> "Both parties shall be referred to as the Parties"

and was reported as a contradiction between two unrelated contracts.

**Duplicates read as contradictions.** The corpus contains two versions of seven
documents. One pair alone produced 98 findings. Each finding is individually
well-formed — two documents, two different figures, the same subject — and the
list is useless, because the reader's problem is not 98 disagreements, it is that
they have two copies of one register.

Raising the threshold fixes neither. The first failure scores *high* on overlap
precisely because boilerplate is shared. The second is not a per-claim property
at all: every one of the 98 findings looks exactly like a genuine contradiction
in isolation.

## Decision

**Weight each shared word by how rare it is across the corpus's own claims.**
Inverse document frequency over claim contexts, normalised so the mean weight is
1.0, applied to the overlap *score*. Boilerplate contributes almost nothing;
topic words contribute several words' worth. Normalisation is what keeps the
existing thresholds meaningful — they still read as proportions of shared
context, the words are just no longer all worth the same.

The *count* of shared words stays unweighted. Whether there is anything to
compare is a different question from what the comparison is worth, and weighting
it twice broke a small corpus where one genuinely shared subject word is all
there is to go on.

**Group findings by document pair, and treat a pair that shares a document's
worth of subject matter as duplication.** Reported once, as one sentence, rather
than enumerated. The threshold is on *how much the two documents both assert*,
not on what fraction disagrees:

| | duplicate pairs | distinct pairs |
|---|---|---|
| figures compared | 46 – 189 | ≤ 9 |
| disagreement ratio | 0.27 – 0.64 | 0.22 – 0.67 |

The ratio was tried first and does not separate them — the ranges run straight
through each other. It is reported, because it is worth seeing, and is not used
to classify. A knob that does not discriminate is worse than no knob.

## Consequences

On the same 15 contracts:

| | findings listed | duplicate pairs named |
|---|---:|---:|
| Before | 320 | — |
| Salience weighting | 287 | — |
| \+ pair grouping | **6** | **6** |

The labelled set stays at 100% precision and 100% recall throughout, which is the
point: the correction is invisible on the corpus shape the set describes and
decisive on the one it does not.

**What this does not claim.** Precision on that corpus is still not a number
worth quoting. The six remaining findings are all false — different SLA
components compared against each other — and the corpus contains no true
contradiction to find, being fifteen unrelated agreements. What the change bought
is an output short enough for a human to read, which is the precondition for
measuring anything. The six named duplicate pairs are true findings and are the
corpus's real problem.

**The gap this exposes and does not close:** the detector has no notion of
**scope**. Fifteen contracts with fifteen counterparties have no business
agreeing with each other. A folder of one company's own policies is the case this
works on. Per-vendor, per-country and per-client corpora need a scope signal that
does not exist yet, and that is now the top correctness item on the roadmap.

**A small corpus is deliberately left unweighted.** Below four claim contexts
there are no frequency statistics worth having, so weights stay uniform. The
correction arrives with the evidence that justifies it.

## Alternatives considered

**Raise `conflict_min_overlap`.** Does not address either failure, and costs
recall on the labelled set. The boilerplate pairs score high.

**A stopword list for legal boilerplate.** A hand-maintained list that is wrong
for the next corpus. Frequency over the corpus's own claims is the same idea
without the maintenance or the guessing.

**Near-duplicate document detection by content hash or shingling.** Would catch
the copies, and only the copies. The pair-shape test also catches two documents
that are not textually similar but cover the same ground at the same level of
detail — last year's register regenerated from a different source, for instance —
and needs no second index.

**Ask a model.** Would likely resolve all six remaining false positives. It also
costs money per pair, makes the result non-deterministic, and removes the
property that makes `audit` worth shipping: that it runs on a folder with no key,
no network and no trust decision.
