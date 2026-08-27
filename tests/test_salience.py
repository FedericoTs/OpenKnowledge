"""How much evidence a shared word is.

The point of weighting is that a folder of contracts shares an enormous
boilerplate vocabulary, and two sentences overlapping on nothing but that are
not about the same subject. These tests pin the two properties that has to
have: boilerplate must be worth less than a topic word, and a corpus too small
to have statistics must not be weighted on noise.
"""

from __future__ import annotations

from openknowledge.knowledge.salience import UNIFORM, Salience, salience_from


def contexts(*groups: str) -> list[frozenset[str]]:
    return [frozenset(g.split()) for g in groups]


def test_a_word_in_every_claim_is_worth_less_than_a_word_in_one() -> None:
    weights = salience_from(
        contexts(
            "agreement party notice expenses",
            "agreement party notice travel",
            "agreement party notice mileage",
            "agreement party notice alcohol",
            "agreement party notice subsistence",
        )
    )
    assert weights.weight("agreement") < weights.weight("expenses")
    assert weights.weight("party") < 1.0 < weights.weight("alcohol")


def test_an_unseen_word_is_neutral() -> None:
    """A word with no corpus statistics is worth exactly one word, not zero."""
    weights = salience_from(contexts("a b", "a c", "a d", "a e"))
    assert weights.weight("never-seen") == 1.0


def test_a_corpus_too_small_to_have_statistics_stays_uniform() -> None:
    assert salience_from(contexts("a b", "c d")) is UNIFORM
    assert salience_from([]) is UNIFORM


def test_overlapping_on_boilerplate_scores_lower_than_on_topic_words() -> None:
    """The property the whole module exists for."""
    corpus = contexts(*[f"agreement party notice topic{n}" for n in range(20)])
    weights = salience_from(corpus)

    boilerplate = weights.jaccard(
        frozenset({"agreement", "party", "notice", "topic1"}),
        frozenset({"agreement", "party", "notice", "topic2"}),
    )
    topical = weights.jaccard(
        frozenset({"agreement", "topic1", "topic2"}),
        frozenset({"party", "topic1", "topic2"}),
    )
    assert topical > boilerplate


def test_uniform_reproduces_plain_jaccard() -> None:
    a, b = frozenset({"x", "y", "z"}), frozenset({"y", "z", "w"})
    assert UNIFORM.jaccard(a, b) == 2 / 4
    assert UNIFORM.coefficient(a, b) == 2 / 3
    assert UNIFORM.shared_mass(a, b) == 2


def test_empty_contexts_score_zero_rather_than_dividing_by_zero() -> None:
    weights = Salience({"a": 2.0})
    assert weights.jaccard(frozenset(), frozenset()) == 0.0
    assert weights.coefficient(frozenset({"a"}), frozenset()) == 0.0
