"""Why this file no longer scores confidence.

A confidence score was built here from signals the cascade already computed -
how much of the answer's wording appeared in its sources, whether the sources
offered a competing figure of the same unit, whether an open disagreement came
close to blocking the question, whether one document was cited out of several
that matched. It was free, deterministic, and wrong.

Measured against a live model on 17 cases, then re-measured with retrieval
deliberately starved to k=2 with reranking off:

    mean confidence with more evidence (k=6, reranked)   0.905
    mean confidence with less evidence (k=2, no rerank)  0.939
    13 of 17 cases got MORE confident on LESS evidence

At k=2 **not one penalty fired**. Every signal turned out to be a property of
how much was retrieved rather than of how reliable the answer was: fewer chunks
means fewer competing figures, fewer documents to have cited only one of, fewer
conflicts in range. So the score rose as the system was degraded, and would have
told an operator that narrowing retrieval had improved their answers.

On the one case that did fail, confidence was 0.941 against a passing mean of
0.939 - higher on the wrong answer.

The lesson is not that the thresholds needed tuning. It is that "how much
competing material was in the context" is a fact about a retrieval setting, and
dressing it as a judgement about an answer put a number in front of readers that
moved the wrong way.

What replaced it is the measurement without the verdict: `Answer.support` is the
grounding gate's own `support_ratio`, the share of the answer's content words
that appear in the text it cited. That is a fact rather than a prediction, it is
already computed, and it claims nothing about whether the answer is right.

If you want a real confidence score, `openknowledge eval` reports mean support on
answers that passed against answers that failed, and says plainly when it does
not separate. Build a candidate, run it on a labelled corpus, and keep it only if
it separates - and check it against degraded retrieval before believing it, which
is the test this one failed.
"""

from __future__ import annotations
