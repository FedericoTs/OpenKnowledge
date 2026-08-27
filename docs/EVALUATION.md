# Evaluation

Without this, "the cheap tier is good enough" is an opinion — and the entire cost argument
rests on it. The harness exists to turn that into a number that a change can break.

Two sets, two runners:

| | | |
|---|---|---|
| `evals/golden/` | question → expected answer | `openknowledge eval` — needs a model |
| `evals/conflicts/` | document pair → contradiction or not | `openknowledge eval-conflicts` — no model |

```bash
openknowledge eval                      # the whole golden set
openknowledge eval --only refusal       # just the safety set
openknowledge eval --tag finance        # one area
openknowledge eval --json               # machine-readable
openknowledge eval-conflicts            # contradiction precision and recall
openknowledge audit ./policies          # the same detectors on a real folder, free
```

## What it measures

Accuracy and cost **together**, because either alone is trivially gamed: accuracy is
maximised by sending everything to a frontier model, cost is minimised by answering
everything from a stale cache. The pair is the real objective.

| Metric | Question it answers |
|---|---|
| **accuracy** | Of the questions the corpus covers, how many were answered correctly? |
| **false answers** | How many questions it *cannot* answer did it answer anyway? |
| **determinism** | Asked twice, did it give the same answer byte for byte? |
| **paraphrase consistency** | Asked in other words, did it state the same facts? |
| **cost per question** | What did this run actually cost, blended across all tiers? |
| **free share** | What fraction never reached a model? |

**False answers is the metric to watch.** It is reported separately from accuracy, and any
increase fails the run regardless of everything else. A bot that answers 95% of questions
correctly and confidently invents the other 5% is unusable in a company, because nobody can
tell which kind they are reading. Missing an answer is an inconvenience; fabricating one is
a liability.

## Writing cases

`evals/golden.yaml` is a starting set for the sample documents. Replace it with your own.

```yaml
- id: parental-leave-duration
  question: How many weeks of parental leave do I get?
  must_cite: [parental-leave]
  must_say: ["20 weeks"]
  must_not_say: ["26 weeks", "16 weeks"]     # the plausible wrong answers
  paraphrases:
    - "how much parental leave am I entitled to"
  tags: [hr]
```

Three habits that make a golden set worth having:

**Guard every number.** `must_say: ["20 weeks"]` is half a test. A model that says "20 weeks"
is only right if it is not also willing to say "26 weeks" on a different run — so put the
plausible wrong answer in `must_not_say`. This is where invented figures get caught, and
invented figures are the most damaging error this system can make.

**Assert the conditions, not just the headline.** Most internal rules are conditional. "20
weeks" without "12 months of continuous service" is wrong for anyone who has not been there
a year, so both belong in `must_say`.

**Write the safety set first, and keep adding to it.** These are questions employees would
plausibly ask that your documents do not answer:

```yaml
- id: refuse-mileage-rate
  question: What is the mileage reimbursement rate for using my own car?
  kind: refusal
  notes: >
    The expenses policy covers travel, so retrieval surfaces it confidently.
    It says nothing about mileage. This is exactly the shape of question that
    produces invented numbers.
```

The most dangerous questions are the ones *adjacent* to something you do document — the
retriever returns a confident, topically-relevant document that happens not to contain the
answer. Those cases are cheap to write and they are the ones that catch real regressions.

## Measuring contradiction detection

`openknowledge eval-conflicts` scores the free contradiction passes on a labelled
set of document pairs. It reports **precision and recall separately**, because
they protect different things: recall protects the employee who would otherwise
be told the superseded policy, precision protects the feature from being switched
off after three bogus flags.

The shipped set is more than half near-misses — restatements, carve-outs, the
same subject under unrelated kinds of rule. A set of only real contradictions
measures recall and cannot see a false positive, so the loader refuses to run
without clean cases.

It needs no model, so unlike the golden set it is a real evaluation in CI rather
than a smoke test. See [KNOWLEDGE.md](KNOWLEDGE.md) for what the detectors do.

### And its limits, which are severe

This set has scored 100% precision and 100% recall throughout the project's life,
including at the point where the same detector produced **320 findings and no
useful ones** on 15 real vendor contracts. That is not a contradiction: the set
measures what it contains, which is curated pairs of short single-authority
policy documents, and real folders are not that shape.

A green `eval-conflicts` therefore means "no regression against the cases we
wrote down". It does not mean the detector works on your documents. The way to
find out is `openknowledge audit ./your-folder`, which is free, and the way to
turn the answer into a number is to label what it finds and add the cases here.
See [KNOWLEDGE.md](KNOWLEDGE.md#what-a-labelled-set-could-not-tell-us) for the
full account of what that run exposed and what it cost to fix.

## Catching regressions in CI

```bash
openknowledge eval --save-baseline evals/baseline.json    # once, on a good run
openknowledge eval --baseline evals/baseline.json         # thereafter
```

Exits non-zero on any drop in accuracy or determinism, any new false answer, or a cost jump
beyond 25%. Cost gets a tolerance because it moves for legitimate reasons — a cold cache on
a fresh run — while correctness gets none.

### An honest caveat about CI

The harness needs a model to mean anything. In a CI job with no model configured, every
question is refused, which makes the safety set pass trivially and accuracy zero. That is a
smoke test, not an evaluation.

So this repository splits it:

- **Every CI run** exercises the *scorer* through `tests/test_evaluation.py`, which scripts
  the model's answers and asserts the harness catches a wrong number, a fabricated answer, a
  non-deterministic pair, and paraphrase drift. That is fast, free, and deterministic.
- **A real evaluation** runs against a configured model — nightly, or before a release — and
  is what the baseline file is for.

A harness that cannot detect a fabricated answer is worse than no harness, because it
certifies whatever it is pointed at. The scorer tests exist so that failure mode is itself
tested.

## Reading a run

```
13 cases  (8 answerable, 5 must-refuse)

Correctness
  accuracy                   100.0%  (8 cases)
  false answers                   0  (0.0% of must-refuse cases)
  determinism                100.0%  (same question twice)
  paraphrase consistency     100.0%  (same facts, other words)

Cost
  per question             $0.00000
  total for this run       $0.00000
  answered without a model   100.0%
  tiers                    pinned=13
```

`answered without a model` is the number that drives cost. Raising it — by pinning the
questions `openknowledge top` says are asked most, and by improving retrieval so the local
tier can carry more — is the whole optimisation, and this is where you watch it move.
