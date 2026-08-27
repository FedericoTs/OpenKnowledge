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
| `OK_CONFLICT_MIN_OVERLAP` | `0.34` | Raise to flag fewer, lower to flag more |

Drafting prefers the local model when one is configured. It reads every changed document in
full, so it is the most token-hungry thing the system does — running it on a model with no
per-token invoice is the difference between a one-off cost and a genuinely free one.
