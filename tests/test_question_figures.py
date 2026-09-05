"""The asker's own figure is not an invention, but parroting it is.

The grounding gate rejects figures that appear in no source, which is what stops
a model making numbers up. Measured on evals/golden-rules, that rule was also
the whole of the apply-the-rule gap: "does a EUR 40,000 contract need quotes?"
cannot be answered without saying 40,000, 40,000 appears in no policy, and so a
correct answer was thrown away and the user saw "that isn't covered by the
documents I have". Six of six correct answers were rejected when they repeated
the question's figure; six of six passed when they avoided it.

The model was shown the question, exactly as it was shown the passage headers
whose numbers already count. But admitting the asker's figures outright opens a
hole - a leading question could put a figure into policy just by containing it -
so they are admitted only in an answer that also states a figure from the
sources. That is the difference between comparing and parroting.
"""

from __future__ import annotations

import pytest

from openknowledge.retrieval.base import Chunk
from openknowledge.retrieval.grounding import check_grounding

POLICY = (
    "Three competitive quotes are required for any contract with an annual value above "
    "EUR 25,000. Contracts up to EUR 5,000 are approved by a line manager; EUR 5,001 to "
    "EUR 25,000 by the Head of Department; EUR 25,001 to EUR 50,000 by the Chief "
    "Financial Officer."
)


def chunks() -> list[Chunk]:
    return [
        Chunk(
            chunk_id="procurement#0",
            document_id="procurement",
            document_title="Procurement Policy",
            text=POLICY,
            locator="chunk 1",
        )
    ]


def verdict(answer: str, question: str = "") -> bool:
    return check_grounding(answer, chunks(), question=question).passed


ASKED = "Do we need competitive quotes for a contract worth EUR 40,000 a year?"


def test_the_askers_figure_alone_is_still_an_invention() -> None:
    """Without the question, 40,000 is a figure from nowhere - the behaviour
    that made this gate worth having in the first place."""
    assert not verdict("Yes, EUR 40,000 is above EUR 25,000 [procurement].")


def test_comparing_the_askers_figure_to_a_threshold_is_allowed() -> None:
    """The answer carries the threshold it compared against, so the reader can
    check the comparison. This is the case the whole set exists for."""
    assert verdict("Yes, EUR 40,000 is above EUR 25,000 [procurement].", question=ASKED)


def test_parroting_the_askers_figure_as_policy_is_not() -> None:
    """A leading question must not be able to put a number into the policy.

    Written in the policy's own words on purpose. The obvious version of this
    test - "the approval limit is EUR 87,500" - scores 0.20 support and is
    rejected for that alone, so it passes even when the figure rule is removed
    and proves nothing. This one scores 0.82, comfortably over the floor, and
    the only thing wrong with it is the number.
    """
    answer = (
        "Yes, three competitive quotes are required for any contract with an annual "
        "value above EUR 87,500 [procurement]."
    )
    assert check_grounding(answer, chunks()).support_ratio > 0.45, "must fail on the figure alone"
    assert not verdict(
        answer,
        question="Is the threshold for three competitive quotes an annual value above EUR 87,500?",
    )


def test_an_invented_figure_beside_a_real_comparison_is_still_caught() -> None:
    """Comparing correctly does not buy a licence for a second, invented number."""
    assert not verdict(
        "Yes, EUR 40,000 is above EUR 25,000, and the filing fee is EUR 93,400 [procurement].",
        question=ASKED,
    )


def test_a_figure_from_neither_source_nor_question_is_caught() -> None:
    assert not verdict(
        "The Chief Financial Officer approves it, and the cap is EUR 87,500 [procurement].",
        question="Who approves a EUR 40,000 contract?",
    )


def test_an_answer_quoting_only_the_sources_is_unaffected() -> None:
    """The common case, and it must not depend on the question at all."""
    answer = "Contracts up to EUR 5,000 are approved by a line manager [procurement]."
    assert verdict(answer)
    assert verdict(answer, question="Who approves a small contract?")


@pytest.mark.parametrize("spelling", ["EUR 40,000", "EUR 40000", "40,000", "40000"])
def test_the_figure_is_matched_however_either_side_spells_it(spelling: str) -> None:
    """The asker types "40000" and the answer writes "EUR 40,000", or the
    reverse. Normalisation already handles this for sources; it has to hold for
    the question too, or the fix works only when both happen to agree."""
    assert verdict(
        f"Yes, {spelling} is above EUR 25,000 [procurement].",
        question="Do we need quotes for a contract worth 40000 a year?",
    )


async def test_the_router_hands_the_question_to_the_gate(store, settings) -> None:
    """Everything above calls `check_grounding` directly, so none of it notices
    if the router stops forwarding the question - which a sabotage confirmed.

    This asks the cascade a question whose answer must restate the asker's
    figure, and fails if that comes back as a refusal. Self-contained: the
    shared corpus has no procurement policy, and a citation resolving to
    nothing would be refused for a reason unrelated to the one under test.
    """
    from fakes import FakeProvider
    from openknowledge.cascade import Cascade
    from openknowledge.retrieval.base import Document
    from openknowledge.retrieval.bm25 import BM25Retriever
    from openknowledge.types import Tier

    own = BM25Retriever()
    own.index([Document(document_id="procurement", title="Procurement Policy", text=POLICY)])
    reply = (
        "Yes. EUR 40,000 is above the EUR 25,000 threshold, so three competitive "
        "quotes are required [procurement]."
    )
    provider = FakeProvider(replies=[reply] * 4)
    cascade = Cascade(store=store, retriever=own, settings=settings, local=provider)

    answer = await cascade.answer(
        "Do we need competitive quotes for a contract worth EUR 40,000 a year?"
    )
    assert answer.tier is not Tier.REFUSED, answer.notes
    assert "40,000" in answer.text
