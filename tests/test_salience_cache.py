"""Remembering a claim's own weight, without remembering every comparison.

`evals/measured/fortyfirst-what-happens-at-a-thousand.json` profiled a
300-document index: `deontic.conflicts_between` was 98% of it, and inside that
`Salience.mass` was called 8.9 million times over a few thousand distinct claim
contexts. Contradiction detection compares every pair of claims, and every
comparison re-added the weights of both sides from scratch.

A claim's own total does not depend on what it is being compared against, so it
is worth remembering. An intersection or a union belongs to a *pair*, of which
there are millions, so it is not - and the first version of this cached inside
`mass`, which would have remembered every intersection and grown a dictionary
the size of the comparison itself. That is why `total` and `mass` are separate
methods rather than one method with a flag.

The numbers are identical before and after: the same conflicts, byte for byte,
across every corpus committed here, and the labelled set still scores 100%
precision and 100% recall.
"""

from __future__ import annotations

from openknowledge.knowledge.salience import UNIFORM, Salience, salience_from


def test_a_remembered_total_is_the_total() -> None:
    weights = Salience({"leave": 0.25, "annual": 0.5, "days": 2.0})
    context = frozenset({"leave", "annual", "days", "unseen"})
    # 0.25 + 0.5 + 2.0 + 1.0 for the word with no statistics.
    assert weights.total(context) == 3.75
    assert weights.total(context) == weights.mass(context), "asked twice, same answer"


def test_only_claim_contexts_are_remembered() -> None:
    """The memory bug this design exists to avoid.

    `mass` is what the pair-shaped arguments go through - an intersection, a
    union - and it must not accumulate them. At 300 documents that would be
    2.9 million entries.
    """
    weights = Salience({"a": 0.5, "b": 1.5})
    left, right = frozenset({"a", "b", "c"}), frozenset({"b", "c", "d"})
    weights.coefficient(left, right)
    remembered = set(weights._totals)
    assert remembered == {left, right}, remembered
    assert (left & right) not in remembered, "an intersection belongs to a pair, not a claim"

    for _ in range(50):
        weights.mass(frozenset({"x", "y"}) | left)
    assert set(weights._totals) == {left, right}, "mass must never add to the cache"


def test_uniform_weights_need_no_arithmetic_at_all() -> None:
    """Every word worth one word, so a total is a count - and the cache stays
    empty, which matters because UNIFORM is a module-level singleton shared by
    every caller that supplies no corpus statistics."""
    context = frozenset({"one", "two", "three"})
    assert UNIFORM.total(context) == 3.0
    assert UNIFORM.mass(context) == 3.0
    assert UNIFORM.mass(w for w in ("one", "two")) == 2.0
    assert UNIFORM._totals == {}, "a shared singleton must not accumulate anything"


def test_the_cache_cannot_change_what_a_salience_is() -> None:
    """Two objects with the same weights compare equal whatever either has
    computed, so nothing downstream can be made to depend on cache state."""
    a, b = Salience({"x": 2.0}), Salience({"x": 2.0})
    a.total(frozenset({"x", "y"}))
    assert a == b
    assert a._totals and not b._totals


def test_weights_still_come_from_the_corpus() -> None:
    """The uniform fast path must trigger on 'no weights', not swallow a real
    corpus. Four claims is the floor `salience_from` sets for a frequency to
    mean anything; below it there is nothing to weight and UNIFORM is returned
    on purpose."""
    weights = salience_from(
        [
            frozenset({"party", "leave", "annual"}),
            frozenset({"party", "notice", "written"}),
            frozenset({"party", "notice", "term"}),
            frozenset({"party", "notice", "renewal"}),
        ]
    )
    assert weights.weights, "four claims is enough for real weights"
    assert weights.weight("party") < weights.weight("leave"), "boilerplate is worth less"
    # And the remembered total agrees with the arithmetic it replaces.
    context = frozenset({"party", "leave"})
    assert weights.total(context) == weights.weight("party") + weights.weight("leave")


def test_too_small_a_corpus_stays_uniform_and_still_totals_correctly() -> None:
    weights = salience_from([frozenset({"party", "leave"}), frozenset({"party", "notice"})])
    assert weights is UNIFORM
    assert weights.total(frozenset({"party", "leave"})) == 2.0


# -- the comparison itself, remembered once --------------------------------------


#: A pair that really does disagree. The first version of this fixture did not
#: - "must submit within 30 days" against "may submit within 60 days" produces
#: no conflict at all - which made the mutation test below assert that an empty
#: list was still empty. Sabotaging the copy on the way out did not fail it, and
#: that is how the fixture was found.
FIRST = "Annual leave entitlement for full-time employees is 25 days per calendar year."
SECOND = "Annual leave entitlement for full-time employees is 30 days per calendar year."


def _corpus(second: str = SECOND) -> list:
    from openknowledge.retrieval.base import Document

    return [Document("a", "A", FIRST), Document("b", "B", second)]


def test_the_same_corpus_is_not_compared_twice() -> None:
    """An access-rule change rebuilds the index without touching a document.

    At 500 documents that was 32 seconds of comparing 125,000 pairs to reach an
    answer that could not have moved.
    """
    from openknowledge.knowledge import claims as claims_module
    from openknowledge.knowledge.claims import ClaimCache, compare_documents

    docs, cache = _corpus(), ClaimCache()
    calls = 0
    real = claims_module._compare_documents

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return real(*args, **kwargs)

    claims_module._compare_documents = counted
    try:
        first, _ = compare_documents(docs, cache=cache)
        second, _ = compare_documents(docs, cache=cache)
    finally:
        claims_module._compare_documents = real

    assert calls == 1, "the second rebuild compared the corpus again"
    assert [c.key for c in first] == [c.key for c in second]


def test_a_changed_document_is_compared_again() -> None:
    from openknowledge.knowledge import claims as claims_module
    from openknowledge.knowledge.claims import ClaimCache, compare_documents

    cache = ClaimCache()
    calls = 0
    real = claims_module._compare_documents

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return real(*args, **kwargs)

    claims_module._compare_documents = counted
    try:
        compare_documents(_corpus(), cache=cache)
        compare_documents(_corpus(SECOND.replace("30 days", "90 days")), cache=cache)
    finally:
        claims_module._compare_documents = real
    assert calls == 2, "edited text must not be answered from the last comparison"


def test_different_thresholds_are_a_different_question() -> None:
    from openknowledge.knowledge import claims as claims_module
    from openknowledge.knowledge.claims import ClaimCache, compare_documents

    docs, cache = _corpus(), ClaimCache()
    calls = 0
    real = claims_module._compare_documents

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return real(*args, **kwargs)

    claims_module._compare_documents = counted
    try:
        compare_documents(docs, cache=cache)
        compare_documents(docs, cache=cache, deontic_strictness=2.0)
    finally:
        claims_module._compare_documents = real
    assert calls == 2, "strictness changes what counts as a conflict"


def test_what_comes_back_can_be_edited_without_poisoning_the_next_rebuild() -> None:
    from openknowledge.knowledge.claims import ClaimCache, compare_documents

    docs, cache = _corpus(), ClaimCache()
    first, agreements = compare_documents(docs, cache=cache)
    assert first, "the fixture has to disagree, or this test proves nothing"
    before, agreed = len(first), len(agreements)
    first.clear()
    agreements.clear()
    second, still = compare_documents(docs, cache=cache)
    assert len(second) == before, "a caller emptied the list the cache was holding"
    assert len(still) == agreed


def test_the_comparison_survives_the_eviction_that_runs_on_every_rebuild() -> None:
    """The bug that made the cache do nothing at all.

    `Engine.reindex` calls `keep_only` on every scan, and the first version of
    this cleared the remembered comparison there - so the entry was thrown away
    at exactly the moment it was supposed to be used, and the warm rebuild cost
    the same as the cold one. Every unit test still passed. What noticed was the
    measurement: `tools/measure_scale.py` kept printing the same number in both
    columns.

    Nothing is needed here beyond leaving it alone. The entry is keyed by every
    document's id and content hash, so a corpus that lost a document misses on
    its own, and one entry is not the leak `keep_only` guards against.
    """
    from openknowledge.knowledge import claims as claims_module
    from openknowledge.knowledge.claims import ClaimCache, compare_documents

    docs, cache = _corpus(), ClaimCache()
    compare_documents(docs, cache=cache)
    cache.keep_only(docs)  # what a rebuild does: the same corpus, re-swept

    calls = 0
    real = claims_module._compare_documents

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return real(*args, **kwargs)

    claims_module._compare_documents = counted
    try:
        compare_documents(docs, cache=cache)
    finally:
        claims_module._compare_documents = real
    assert calls == 0, "the rebuild compared a corpus that had not changed"
