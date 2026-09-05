# Questions whose answer is a whole document

Every other set here asks whether the system finds the right fact. This one
asks whether it can find *all* of something.

"What are the priorities?", "who are the main characters?", "summarise this
document" - the answer is not a sentence sitting in one passage. It is the
whole document, and retrieval hands the model `retrieval_k` chunks of it. The
shipped default is 6 chunks of ~350 words: about 2,100 words, four pages.

## Where this came from

The `/manage` gap report on a real install, sixteen questions. Eleven were this
shape:

| shape | n |
|---|---|
| enumerate a list | 5 |
| characters in a work | 3 |
| summarise a document | 3 |

They route correctly - `evals/capability/questions.py` already asserts six of
them must reach retrieval, and they do. They fail after it.

## Two measurements

The first is modelless and already existed: `tools/measure_scope.py`, the
ceiling. The second is `scope.yaml`, a model-in-the-loop set run like every
other golden set, which records what the whole pipeline then does with that
ceiling.

### The set: `scope.yaml`

Ten cases over `./documents`, none of which was written here:

| document | source | words | why |
|---|---|---:|---|
| `alice.md` | Project Gutenberg #11, Carroll | 26,537 | twelve chapters, one title per ~2,400 words |
| `earnest.md` | Project Gutenberg #844, Wilde | 20,719 | a printed cast list of nine, three acts |
| `ftr-300-1.md` | copied from `../golden-ftr` | 6,314 | the 82-term glossary, 22 chunks |
| `ftr-301-10.md` | copied from `../golden-ftr` | 4,706 | a four-item list in one chunk: the control |

The two books were converted by `tools/gutenberg_to_md.py`, which turns
`CHAPTER I.` and `FIRST ACT` and `THE PERSONS IN THE PLAY` into headings and
changes nothing else. The originals are in `./sources`; the converter is
idempotent and a test proves the committed copy is its output word for word.

Every reference list is read out of the documents by the script that wrote the
YAML, and `tests/test_golden_scope.py` regenerates each one. Wilde's cast list,
Carroll's chapter titles, the regulation's terms, GSA's section headings -
nothing in the exam is a list somebody here made up. The one judgement is
`scope-07`, which names five people any summary of Act I mentions.

Shapes: an enumeration inside the budget (the control, must stay complete);
three far outside it; an ordinal ("the second chapter"); a summary; and three
refusals in the same vocabulary - "the fourth act" of a three-act play,
"chapter thirteen" of twelve - so the ordinal machinery is measured for
fabrication as well as recall.

`must_list` with `min_share` is new in the scorer for this set. `must_say`
demands every fact, which is right for "how much" and would make an 82-term
list 82 impossible requirements. The scorer now reports *listed 26 of 82
(32%), needed 90%* - the number every other set was unable to state.

```sh
export OK_DOCUMENTS_DIR=$PWD/evals/golden-scope/documents
export OK_LOCAL_ENABLED=true OK_EMBEDDING_ENABLED=false OK_ESCALATION_ENABLED=false
export OK_LOCAL_BASE_URL=http://127.0.0.1:8081/v1
uv run openknowledge eval --path evals/golden-scope/scope.yaml --verbose
```

### Results

Seven of the ten cases are tagged `structure`: the document's own headings,
list or glossary answer them, and they run through the real cascade with no
model configured - in CI, on every push.

```
uv run openknowledge eval --path evals/golden-scope/scope.yaml --tag structure

  7 cases  (5 answerable, 2 must-refuse)
  accuracy 100.0%   false answers 0   determinism 100.0%   paraphrase 100.0%
  tiers  outline=5  refused=2          cost $0
```

The two refusals are "the fourth act" of a three-act play and "chapter
thirteen" of twelve, each refused with the count.

With a real model over all ten cases: **42.9% to 100% accuracy, one false
answer to none**, determinism 90% to 100%, paraphrase consistency 75% to 100%,
and the run itself from 58 minutes to 14 because half the set stopped calling a
model.

The false answer is worth naming. Asked what happens in the fourth act of a
three-act play, the system invented one - "the characters confront the truth
about Jack's identity and the deception surrounding his supposed brother,
Ernest" - and the grounding gate passed it at 96% support, higher than the mean
on answers that passed. `cascade/scope.py` cites that risk as the reason an
ordinal past the end refuses with the count; it was written as a hypothetical
and turned out to be a transcript.

### The ceiling: `tools/measure_scope.py`

`tools/measure_scope.py`. No model is called, deliberately. This measures the
ceiling: of the passages carrying the answer, how many does `search()` return?
A model cannot list what it was never shown, so this is the best any prompt or
any model could do. That separation matters - a 32% ceiling is a retrieval
problem, and no amount of prompt work will move it.

The corpus is `../golden-ftr/documents` - the US Federal Travel Regulation,
public domain, already committed, written by nobody here. That last part is the
point: an enumeration corpus I wrote myself would be one where I had
unconsciously kept the answer inside the budget.

`§ 300-1.1` is a glossary: 82 defined terms over 22 chunks. The reference
answer is extracted from the regulation rather than typed out, because a
hand-copied list drifts from the corpus and drifts in the direction that
flatters the score.

```
                                        k=6    k=12   k=25   k=50   all 92
  what terms does the glossary define    32%    56%    77%    89%    100%
  list all the terms defined ...         32%    49%    74%    90%    100%
  what are all the definitions in ...    26%    40%    59%    84%    100%
```

**At the shipped default the model is shown 1 of the 22 chunks, and 26 of the
82 terms.** It then answers with what it saw and says the rest is not in the
sources - which the ledger records as `partial`, and the gap report presents as
"the document exists and is missing a section". The section is not missing. It
was in chunk seven.

## Raising k does not fix this

k=50 is eight times the shipped budget and more than half the entire corpus,
and it still reaches only 89%. BM25 ranks by term overlap, and a glossary
chunk defining "Spouse" does not match "what terms does the glossary define"
any better than an unrelated chunk does. The query has no lexical purchase on
the content it is asking to enumerate.

That is the useful part of this result. The obvious fix - retrieve more - is
measured here and does not work. A whole-document question needs the document
assembled, not ranked.

## The control

`§ 301-10.2` authorises four transportation methods, in one chunk. It scores
4/4 at every k including the shipped 6, which is what says the harness measures
the system rather than itself.

Its floor is chance, not zero: unrelated queries still score 1-2 of 4, because
"common carrier" and "privately owned" are ordinary phrasing in a travel
regulation. The control's signal is 4/4 against that 1-2, not against 0.

## Sabotages

- Shown all 92 chunks, the glossary case reaches exactly 82/82 and 22/22. If it
  did not, the extractor or the coverage check would be wrong rather than the
  system - this is the one that makes the 32% believable.
- Unrelated queries drop the control from 4/4 to 1-2/4, so it is not passing on
  ambient vocabulary.

## What is not measured here

Whether the model, shown 26 of 82 terms, lists 26 or invents 82. That needs a
model and belongs in a generation set. This file establishes only the ceiling.
