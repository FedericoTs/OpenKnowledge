# Architecture

## The shape of the problem

An internal-documents chatbot has three jobs, and they pull against each other:

1. **Be correct.** Wrong policy answers cause real damage and the user has no way to tell.
2. **Be cheap.** Frontier-model calls on every question do not survive contact with finance.
3. **Be private.** The documents are the company's, and often nobody wants them leaving.

The usual build optimises (1), pays for it in (2), and gives up on (3). OpenKnowledge's
position is that most of the cost is not buying correctness at all — it is re-deriving
answers that already exist, and paying frontier prices to read a paragraph aloud.

## The cascade

One question enters. It leaves at the first tier that can answer it *and prove it*.

```
                    question
                        |
              canonicalise (fold casing, whitespace, greetings;
                        |    never touch a word that carries meaning)
                        v
      do the documents disagree here? - yes -> refuse, naming both sides
                        | no                  (unless a pin postdates the conflict)
                        v
      L0  pinned answer?  ------------ yes -> return, $0
                        | no                  (access-checked)
                        v
      L1  exact cache hit? ----------- yes -> return, $0
                        | no                  (access-checked)
                        v
      L2  drafted at upload?  -------- yes -> return, $0
                        | no                  (gate-checked, marked unreviewed)
                        v
              retrieve (BM25 today; hybrid + rerank next)
                        |
                   no hits -> refuse
                        v
      L3  local model  --> grounding gate --> pass -> cache, return, $0 marginal
                        |                     fail
                        v
      L4  frontier API --> grounding gate --> pass -> cache, return, $
                        |                     fail
                        v
                     refuse ("I don't know") - never cached
```

Two properties make this safe rather than a gamble:

- **Every tier answers from the same retrieved evidence, under the same system prompt.**
  Escalation changes the price, not the rules.
- **Every tier's answer is graded by the same grounding gate.** A cheap model that would
  have invented a number never reaches the user; it just triggers the next tier.

So "use the cheap model first" is not a quality compromise. It is a bet that the cheap model
can often do the job, with a verifier that collects when the bet loses.

## The grounding gate

Implemented in [`retrieval/grounding.py`](../src/openknowledge/retrieval/grounding.py). An
answer must clear four checks before anyone sees it:

| Check | Catches |
|---|---|
| Cites at least one source | Confident free-form answers with no basis |
| Every cited id was actually retrieved | The classic RAG failure: a plausible invented filename |
| Every number appears in the cited text | The most damaging error in a policy bot — a wrong threshold or deadline |
| Content words overlap the sources enough | Fluent invention that stays on topic |

An explicit "I don't know" is reported separately (`abstained`). It is correct behaviour and
must not be cached as if it were an answer.

The number check is cheap and disproportionately valuable. Internal policy is mostly
numbers — *20 weeks*, *EUR 500*, *60 days* — and a model that gets the prose right and the
figure wrong produces something that reads perfectly and is actively harmful.

## The cache key

```
sha256( canonical_question ‖ corpus_version ‖ prompt_version ‖ policy_version ‖ route_id )
```

Every input that could change the answer is in the key, which gives two guarantees at once:
identical inputs return an identical answer, and changed inputs cannot return a stale one.
`corpus_version` is a hash of the whole indexed corpus, so editing one document in SharePoint
makes every answer derived from the old text unreachable — instantly, and without a cache
flush. See [DETERMINISM.md](DETERMINISM.md).

`route_id` fingerprints the *configured models* rather than naming one, because an answer
here may come from either tier. Swapping the local model invalidates; which tier happened to
answer does not.

## Access control

Enterprise search that ignores permissions is a leak generator. Two places enforce it:

- **At retrieval.** ACL filtering happens during scoring, not as a filter over the top-k —
  otherwise a restricted user silently gets fewer results instead of different ones.
- **At cache read.** The cache is shared across users and its key deliberately excludes
  identity (per-user caches would destroy the hit rate the cost model depends on). So a
  cached answer is served only if the asker could have retrieved each of its cited sources
  themselves. Otherwise it is treated as a miss and re-answered over what they *can* see.
  An unknown document id fails closed.

Connectors are responsible for populating `allowed_principals` from the source system's own
ACLs. That is the bulk of the work in the SharePoint and Drive connectors, and the part that
leaks documents if done casually.

## Module map

| Module | Responsibility |
|---|---|
| `canonical.py` | Question → cache-key form. Conservative on purpose. |
| `costs.py` / `pricing.yaml` | Token accounting. Rates carry a verification date. |
| `cache/keys.py` | Key derivation, corpus fingerprinting |
| `cache/store.py` | Pins, cached answers, cost ledger (SQLite) |
| `retrieval/bm25.py` | Lexical retrieval, chunking, ACL-aware scoring |
| `retrieval/grounding.py` | The gate |
| `providers/` | `ChatProvider` protocol; Anthropic (with caching), OpenAI-compatible |
| `cascade/router.py` | Tier ordering, escalation, pricing of each answer |
| `knowledge/` | Ingest-time drafting, review queue, conflict detection, re-verification |
| `evaluation/` | Golden set, scoring, baseline comparison |
| `connectors/` | Document sources |
| `api/` | FastAPI app, engine assembly, admin |

## Deliberate omissions

Things a v0.1 could have had and does not, with reasons:

- **No vector database.** BM25 answers policy questions well and adds no dependency. Hybrid
  retrieval is on the roadmap; a Pinecone bill would have been funny in this project.
- **No conversation memory.** Follow-up questions ("what about contractors?") need it, and
  it interacts badly with a cache keyed on a single question. Designing that properly is
  worth its own decision, not a default.
- **No agent loop.** Retrieve-then-answer covers the traffic. Multi-step retrieval is a
  large cost multiplier and should be earned by evidence.
- **No fine-tuning.** It gives up determinism and pins you to a model version, in exchange
  for something retrieval already does.
