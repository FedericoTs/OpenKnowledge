# Does it apply the rule, or only recite it?

Asked "what is the procurement threshold?", the system answers. Asked "we want
to buy a EUR 40,000 platform - do we need quotes?", it refuses, though the rule
is in the document and retrieval finds it.

Two evaluations found this independently and neither was built to measure it:
`golden-injection`'s `inj-07`, where it failed in all three arms of the control
and so is not injection's doing, and `golden-ftr`, whose notes record the same
shape. This set exists so a change can be judged.

## What it is

Sixteen cases over `../corpus/aveline`, already committed and used by
`golden-aveline`. Nothing here is new writing: every case's `notes` quotes the
sentence it turns on, so the exam can be checked against the documents rather
than trusted.

| group | n | what it asks |
|---|---:|---|
| interior | 7 | a figure clearly inside one band |
| boundary | 7 | a figure *on* a threshold |
| refusal | 2 | a figure with no rule that reaches it |

The boundaries are the point. "Up to EUR 5,000", "above EUR 25,000", "at or
below EUR 500", "more than 20 working days", "at least 14 characters" - a
system that matches a number to a nearby band gets the interior right and the
edges wrong, and the edges are where somebody is told they may commit money
they may not. Exactly EUR 25,000 is the sharpest: it is *inside* the
Head-of-Department approval band and *outside* the three-quotes rule, so one
figure has opposite answers in two rules on the same page.

The two refusals carry a figure and a plausible-sounding subject with no rule
behind them - payment in cryptocurrency, a relocation allowance. A system newly
taught to compare numbers must not start inventing bands to compare them
against.

## Scoring an inference set with substring matching

This is harder than it looks and the first draft got it wrong twice.

`must_not_say: [no]` fails a **correct** answer, because "not" contains "no" -
the mistake `golden-ftr`'s notes already record, made again here. Every
forbidden phrase is now checked against a written-out correct answer before it
is trusted, and `rule-10` ended up with none at all: a correct answer quotes
"will not be approved" while explaining that 20 days does not reach it.

The refusal is the failure being measured, and the scorer already fails an
answerable case that was refused. So `must_say` carries the document's own
words - the band, the figure, the named approver - and a wrong band fails on
that alone. That is why most cases need no `must_not_say`.

## Running it

```sh
export OK_DOCUMENTS_DIR=$PWD/evals/corpus/aveline
uv run openknowledge eval --path evals/golden-rules/rules.yaml --dry-run   # free
uv run openknowledge eval --path evals/golden-rules/rules.yaml --verbose
```

The pre-flight passes all fourteen answerable cases: retrieval reaches the
right passage for every one, so a failure in a live run is the model's and not
the corpus's. It passes the derived verdicts too, and not because they are
present - for `rule-07` none of "not required", "are not needed", "do not need"
or "no quotes" appear in what retrieval returns. The pre-flight fails a case
only when *every* one of its fact groups is missing, which it documents as
deliberate; the extractive group confirms retrieval, the derived group is
expected to be absent.

## Status

Written and pre-flighted. The model-in-the-loop baseline has not been run yet -
the machine here answers one question every two to three minutes on CPU, and
the scope set is ahead of it in the queue. No number is claimed until it has.
