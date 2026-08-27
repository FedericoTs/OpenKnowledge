# Determinism

## What is promised

**Under an unchanged corpus, prompt, policy, and model configuration, the same question
returns the same answer — byte for byte.** Not "usually", not "semantically similar". The
second person to ask gets exactly what the first person got.

## What is not promised

**That the answer is the same after anything changes.** It is not, deliberately. Edit a
document, edit the system prompt, change a retrieval setting, or swap a model, and the
answer is regenerated. An answer engine that kept returning last quarter's policy because it
was cached would be worse than one with no cache at all.

**That two different phrasings always agree.** Canonicalisation folds casing, whitespace,
smart quotes, and leading greetings. It does not paraphrase. "How much parental leave do I
get?" and "What is the parental leave entitlement?" are different cache entries today, and
both are answered from the same sources under the same rules. Closing that gap is the
semantic cache tier (L2), which needs a similarity threshold *and* a citation check — a
hash cannot safely decide that two sentences mean the same thing.

**That the underlying model is deterministic.** It isn't. Temperature 0 is necessary and
insufficient: batching and floating-point non-determinism mean even greedy decoding can vary
between runs. This is precisely why determinism here comes from the cache rather than from
the sampler.

## How it works

```
sha256( canonical_question ‖ corpus_version ‖ prompt_version ‖ policy_version ‖ route_id )
```

| Component | Changes when |
|---|---|
| `canonical_question` | The question does, in a way that could change its meaning |
| `corpus_version` | Any indexed document is added, edited, or removed |
| `prompt_version` | The system prompt, or the admin's appended instructions, change |
| `policy_version` | Retrieval depth, the support threshold, or the citation rule change |
| `route_id` | The configured local or escalation model changes |

Fields are joined with `\x1f`, a byte that cannot occur in any of them, so `("ab","c")` and
`("a","bc")` cannot collide into one key.

The practical consequence: **"clear the cache" is almost never the right operation.** Bump
the relevant version and invalidation is precise and reversible.

## Why canonicalisation stops where it does

This is the part most likely to be "improved" into a correctness bug, so the reasoning is
worth stating plainly.

Aggressive normalisation raises the cache hit rate. Stemming, stopword removal, and
lemmatisation all make more questions collapse onto one entry, and the cost graph looks
better immediately. They are also catastrophic here, because in policy and procedure
questions the small words are the content:

| These must never collapse | |
|---|---|
| "which expenses are reimbursable" | "which expenses are **not** reimbursable" |
| "approval needed **before** travel" | "approval needed **after** travel" |
| "who **may** approve this" | "who **must** approve this" |
| "leave **with** pay" | "leave **without** pay" |

Drop "not" as a stopword and half the company is told the opposite of the policy, with a
citation attached, in a tone of complete confidence. A cache miss costs a fraction of a cent.
This costs a compliance incident.

So `canonical.py` only removes what provably cannot carry meaning: unicode presentation
differences, casing, whitespace, terminal punctuation, and a closed list of leading
pleasantries. There is a `NEVER_STRIP` set in that module and a test file full of pairs that
must stay distinct — both exist so this stays a deliberate decision rather than something a
later optimisation quietly reverses.

## Pinned answers: determinism by construction

The cache makes an answer *reproducible*. A pin makes it *correct*, because a person wrote it.

For the few hundred questions that make up most internal traffic, that is the right tool:
HR writes the canonical answer to "how much parental leave do I get", it is returned exactly,
every time, for free, with no model involved. `openknowledge top` ranks questions by
frequency so an admin can see which ones are worth the ten minutes.

```bash
openknowledge pin "How much parental leave do I get?" \
  "20 weeks fully paid after 12 months of continuous service; 6 weeks at statutory rate below that." \
  --cite parental-leave \
  --alias "what is the parental leave entitlement" \
  --alias "how much time off do I get for a new baby" \
  --author hr@example.com
```

Two flags worth using every time:

`--cite` attaches the source documents, so a pinned answer carries the same provenance a
model-generated one does. A pin without sources asks the reader to take it on trust, which is
the one thing this project is trying not to do. Citing a document that is not in the corpus
produces a visible marker rather than a silent omission — the grounding rules that apply to
models should apply to people too.

`--alias` catches the phrasings canonicalisation deliberately will not collapse. Until the
semantic cache lands, this is how "what is the parental leave entitlement" reaches the same
answer as "how much parental leave do I get". `openknowledge eval` will tell you when you have
missed one: paraphrase consistency drops.

Pins are access-checked like any other answer.

## Verifying it yourself

```bash
openknowledge ask "How much parental leave do I get?"   # local model, cached
openknowledge ask "how much PARENTAL LEAVE do i get??"  # exact cache hit, $0
openknowledge costs                                     # tier breakdown

# Now edit a document and watch the answer regenerate:
echo "Updated: 24 weeks." >> documents/parental-leave.md
openknowledge index
openknowledge ask "How much parental leave do I get?"   # fresh answer, new corpus version
```
