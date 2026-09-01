# Knowledge lifecycle

Maintaining an internal answer bot is usually a standing chore: someone notices a wrong
answer, hunts down which document changed, and edits an FAQ nobody else knew existed. The
goal here is to move that work to a single moment — when a document is uploaded or changed —
and to make everything after that automatic.

```
document uploaded or changed
        |
        +-- numeric conflicts detected              free, no model
        +-- drafts built on the old text retired    free, no model
        |
        +-- FAQ answers drafted from the document   one model call per document
        |     |
        |     +-- fails the grounding gate -> discarded, never queued
        |     +-- passes -> servable immediately, marked "not yet reviewed"
        |
        +-- approved answers citing it re-asked     one call per affected answer
              |
              +-- figures unchanged -> silent
              +-- a figure moved   -> raised for review, naming the change
```

Nothing in the free column ever calls a model, so `openknowledge index` is safe to run as
often as you like. `openknowledge learn` is the step that spends, and it reports what it
spent.

## Why this is where the money is

The cascade makes each question cheaper. Drafting at upload makes some questions cost nothing
at all, for a price paid once. Run `python tools/cost_model.py` for the current numbers:

| | |
|---|---:|
| Draft an FAQ across 500 documents | **$6.06, once** |
| Raising the free share from 45% → 75% | saves ~$1,410/year |
| **Payback** | **~1 day** |

The larger benefit is not on that table. A fresh deployment has an empty cache and climbs to
its free share over months of real traffic — the first month is the most expensive month it
will ever have. Drafting at upload means day one starts near the top of the range instead of
at the bottom.

## Drafts are precomputed cache entries, not pins

The obvious way to build this is to generate a few thousand Q&A pairs and ask an admin to
"validate" them. That fails on contact with a human: nobody reviews three thousand items
carefully, and rubber-stamped machine output sitting in the highest-trust tier is worse than
no feature at all — pins bypass verification precisely because a person wrote them.

So drafts are held to the same standard as any other machine answer:

- **Every draft passes the grounding gate before it is stored.** It must cite documents that
  were actually retrieved, use only figures present in them, and stay close to the source
  wording. Anything that fails is discarded, not queued — the reviewer never sees it.
- **A gate-passed draft serves immediately, marked as unreviewed.** This is the key
  reframing: it is exactly as trustworthy as a cached answer, because it went through exactly
  the same checks. It just went through them earlier. Serving it is not a new risk; refusing
  to serve it would only mean paying to re-derive it.
- **Approval promotes it to a pin.** That is the moment a machine draft becomes a human
  decision, and it is the only path into the top tier.

Set `OK_SERVE_DRAFTS=false` for a compliance posture where nothing machine-written may reach
an employee before sign-off. The cost benefit then waits on review, which is the honest
trade.

## The review queue is ranked, and short

A queue ordered by nothing in particular gets abandoned. `openknowledge review` ranks drafts
by what approving each one is actually worth — how often that question is asked, from the
ledger, times what one un-cached answer costs. Reviewing the top fifty is a morning's work
and captures most of the value; the tail can wait indefinitely without the system degrading,
because unreviewed drafts are still gate-checked.

```
openknowledge review
openknowledge approve <id> --reviewer hr@example.com
openknowledge reject  <id> --note "answers a question nobody asks"
```

A rejected draft is never proposed again. A queue that re-offers what you already declined
stops being read.

## Contradictions

Two passes, because they catch different things at very different prices.

### Three passes, ordered by price

| Pass | Catches | Cost |
|---|---|---|
| Numeric claims | A moved figure — threshold, deadline, allowance | free |
| Deontic claims | A changed permission — eligible → excluded, required → optional | free |
| FAQ cross-check | A **new** document disagreeing with an existing answer | free |
| Re-verification | Anything the above miss, on documents that changed | ~$0.075/upload |

The first three cost nothing and run on every `index`. Only the last one spends,
and it only runs on `learn`.

### Numeric conflicts — free, every re-index

Internal policy is mostly numbers, and when two documents disagree it is almost always about
a threshold, a deadline, or an allowance. Catching that class needs no model: a figure has a
value, a unit, and the words around it, and two claims conflict when the first two differ
while the third matches.

```
$ openknowledge conflicts
2 unresolved disagreements

  expenses-policy-2026:EUR 1,000|expenses-policy:EUR 500
    [expenses-policy] EUR 500
      Any single expense above EUR 500 requires prior written approval.
    [expenses-policy-2026] EUR 1,000
      Any single expense above EUR 1,000 requires prior written approval.
```

This runs on the whole corpus on every `index`, costs nothing, and is deterministic.

### Prose conflicts — free, and the same shape

Numeric detection is blind to the class that does the next most damage:

    "Contractors are eligible for parental leave."
    "Contractors are excluded from parental leave."

Nothing numeric changed. An employee acting on the wrong one has a real problem.

Catching it needs no model either, because **policy prose is deontic**: almost
every internal rule says something *must*, *may*, or *must not* happen. That is a
small closed vocabulary, so a rule extracts the same way a figure does — a
marker, a force, and the words around it — and a contradiction becomes "the same
subject under a different force".

Two gates keep the precision up, and the second is the interesting one:

**Predicate families.** A rule about reimbursement cannot contradict a rule about
VPN access, however many words they share. Claims are only compared when they are
about the same *kind* of rule.

**Hard versus soft pairs.** FORBIDDEN against either other force is a logical
contradiction — you cannot be both allowed and not allowed to do something —
so recognisably the same subject is enough. MANDATORY against PERMITTED is not:

    "Employees must submit expense claims within 60 days."
    "Employees may submit expense claims online through the portal."

Both true at once. One is a deadline, the other a channel. That pair only
contradicts when it describes the identical action, so it needs near-identical
context before anyone is interrupted. Without this rule, that example scores as a
contradiction on word overlap alone.

### Cross-checking answers against new documents — free

Re-verification has a structural blind spot: it re-asks the approved answers that
*cite* a changed document. It cannot see the most common way a contradiction
arrives — somebody uploads a **new** document. No answer cites it, because it did
not exist, so nothing is re-asked.

Closing that at the document level means comparing the new file against every
other file. Closing it at the FAQ level is free:

1. BM25 already knows, at no cost, which questions a new document has an opinion
   about — the ones it ranks highly for.
2. A stored answer is short text with extractable claims.
3. So it is claim-versus-claim: regex and set arithmetic.

```
'Can I expense alcohol for client entertainment?':
  your approved answer says not allowed,
  but [expenses-2026] says allowed
```

No model call anywhere in that path.

### Prose changes — anchored to answers, not documents

The obvious implementation compares each uploaded document against every other one. At 500
documents that is **$5.00 per upload** and O(N²) — a price that quietly discourages keeping
the corpus current, which is the opposite of the goal.

The cheap implementation notices that approved answers already carry citations. When a
document changes, only the answers that *cite it* can be affected, so those are the only ones
worth re-asking — typically a handful. That is **$0.075 per upload, 66× cheaper**, and it
produces a better artefact: instead of "these documents differ somewhere", the reviewer gets

```
"What is the approval threshold?": EUR 500 -> EUR 1,000
```

Comparison is figure-first. A reworded answer with identical numbers is not worth
interrupting anyone about; a changed threshold always is.

## What happens to a contested question

It is refused, naming both figures and both documents:

```
Your documents disagree on this, so I won't guess:
  - [expenses-policy] says EUR 500, [expenses-policy-2026] says EUR 1,000
Please ask your administrator which one currently applies.
```

A contested claim is exactly where the bot would otherwise be confidently wrong, which is the
one failure this project exists to prevent. Seeing both values is also often enough for the
reader to know which applies to them, and it creates useful pressure to resolve it.

Questions the documents still agree on are unaffected — relevance is judged against the
contested claim's own context, not against document identity. Blocking every question that
touches a document with one bad figure would make the feature intolerable, and an admin would
switch it off.

### The stale-pin trap

A pin written *before* a disagreement appeared was written by someone who had not seen the
new document. Serving it is the worst version of this failure: the answer looks
human-authored and authoritative, and it is out of date.

So pins are compared against the conflict's detection time. A pin that predates an unresolved
conflict is withheld and the question is reported as contested. A pin written *after* the
conflict was detected is a decision about it, and wins — pinning with the disagreement
visible is what resolving it looks like.

```
openknowledge resolve "<key>" --keep expenses-policy-2026 --reviewer finance
```

Resolving records the decision. It does not edit your documents — remove or correct the
superseded text so retrieval stops seeing it.

## Settings

| | default | |
|---|---|---|
| `OK_DRAFT_ON_INGEST` | `true` | Draft answers when documents are added or change |
| `OK_SERVE_DRAFTS` | `true` | Serve gate-passed drafts, marked as unreviewed |
| `OK_BLOCK_ON_CONFLICT` | `true` | Refuse contested questions instead of guessing |
| `OK_REVERIFY_ON_CHANGE` | `true` | Re-ask approved answers whose documents changed |
| `OK_MAX_DOCUMENTS_PER_INGEST` | `200` | Cap per run, so a first import cannot spend without warning |
| `OK_CONFLICT_MIN_OVERLAP` | `0.34` | Numeric threshold: raise to flag fewer |
| `OK_DEONTIC_STRICTNESS` | `1.0` | Prose thresholds: above 1.0 flags fewer |

Drafting prefers the local model when one is configured. It reads every changed document in
full, so it is the most token-hungry thing the system does — running it on a model with no
per-token invoice is the difference between a one-off cost and a genuinely free one.


## Measuring detection

Detection is an accuracy component, so it carries its own numbers — and the two
directions fail differently.

**Recall protects the employee.** A missed contradiction means somebody is told
the superseded policy, confidently, with a citation attached.

**Precision protects the feature.** A false flag blocks a question that could have
been answered, and an admin who sees three bogus flags stops reading the fourth —
which costs every real flag after that. A detector at 100% recall and 40%
precision gets switched off within a week, and then its recall is zero.

```bash
openknowledge eval-conflicts
```

```
21 cases  (9 real contradictions, 12 that must stay quiet)

Detection
  precision    100.0%  (of what it flagged, how much was real)
  recall       100.0%  (of what was real, how much it flagged)
  F1           100.0%
```

The set in `evals/conflicts/` is deliberately more than half **near-misses** —
restatements, carve-outs, deadline-versus-channel pairs, the same subject under
unrelated kinds of rule. A labelled set of only real contradictions measures
recall and cannot see a false positive, so the loader refuses to run without
clean cases.

This needs no model, so it runs in CI on every change. `--strictness` lets you
see the trade directly: at 1.8 the shipped set holds 100% precision and drops to
56% recall.

### What a labelled set could not tell us

The 21 cases are curated pairs of short policy documents. Run the same detector
over **15 real vendor contracts** — around 100 pages, boilerplate-heavy, from
different counterparties, including two parallel copies of seven documents — and
it emitted **320 findings, none of them useful.** A set that measures 100/100 was
not wrong; it was measuring a corpus shape that real folders do not have.

Two separate failures, and neither is a threshold that needed tuning.

**Boilerplate read as shared subject.** Contracts overlap enormously on
*party*, *agreement*, *notice*, *written*, *service*. Counting shared words
treats those as evidence that two sentences are about the same thing, which is
how *"Buyer and Licensor may be referred to collectively as the Parties"* scored
75% against *"Both parties shall be referred to as the Parties"*. The fix is to
weight each shared word by how rare it is across the corpus's own claims
(`salience.py`) — ordinary inverse document frequency, normalised so the mean
weight is one word, so the thresholds keep their meaning. The count of shared
words stays unweighted: whether there is anything to compare is a different
question from what the comparison is worth.

**Duplicates read as contradictions.** Two copies of one register disagreeing on
ninety-eight figures is not ninety-eight contradictions. No per-claim threshold
can see this, because each individual finding is identical in both cases — what
differs is the shape of the pair. The separator is **how much subject matter the
two documents share**: on that corpus, duplicate pairs compared 46 to 189 figures
each and every genuinely distinct pair compared nine or fewer. The disagreement
*ratio* was tried first and fails — duplicates ran 0.27–0.64, distinct pairs
0.22–0.67, straight through each other — so it is reported and not used to
classify. See `variants.py`.

Together, on the same corpus:

| | findings listed | duplicate pairs named |
|---|---:|---:|
| Before | 320 | — |
| Salience weighting | 287 | — |
| \+ pair grouping | **6** | **6** |

The six named duplicate pairs are the corpus's real problem, and they are the
finding a human can act on. The six remaining listed findings are all still false
positives — different SLA components compared against each other — so precision
on that corpus is not yet a number worth quoting. What changed is that the output
is now short enough for somebody to look at, which is the precondition for
improving it.

### What it still does not catch

Prose contradictions with no deontic marker — *"the policy was withdrawn in
March"* against a document that still states the policy — and contradictions
that need world knowledge to see. Those fall to re-verification, which covers
questions somebody already approved, and ultimately to the golden set. Adding
cases to `evals/conflicts/` is the cheapest way to find out what else is missing.

**A per-entity corpus is now handled**, and it was the largest of these. See
`scope.py`: an agreement's parties are read from the positions that define
them — *between X and Y*, `X ("the Supplier")` — in the document's opening,
and the party common to the most documents is dropped, because your own
company signs all of them. Two documents are compared unless both name
parties and share none. On a corpus of the shape that broke this, findings
went from 534 across 136 pairs to 3 across 3, none of them cross-vendor,
with a same-vendor contradiction and a policy contradiction both still
caught. Four cases went into `evals/conflicts/` for it, and against the
previous detector the set now falls to 84.6% precision — it is no longer
blind to this class.

What decided the design is the control that could have gone wrong: two
versions of one expenses policy naming **different** booking agents must
still be compared. A mention is not a party, so a policy corpus gets no
scope at all. A suppressed contradiction means somebody is told the wrong
policy with a citation attached, which is worse than the noise this removes.

The original note, kept because the reasoning still holds: fifteen contracts
with fifteen different counterparties have no business agreeing with each
other, and nothing in the
detector knows that. A folder of one company's own policies is the case this
works on; a folder of per-vendor or per-country documents needs a notion of scope
that does not exist yet.
