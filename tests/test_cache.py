"""Cache keys, pins, and the ledger."""

from __future__ import annotations

from openknowledge.cache import AnswerStore, KeyContext, answer_key
from openknowledge.cache.keys import corpus_fingerprint
from openknowledge.costs import Usage
from openknowledge.types import Answer, Citation, Tier

CTX = KeyContext(corpus_version="corpus-1", model_id="route-a")


def test_equivalent_questions_share_a_key() -> None:
    assert answer_key("What is the leave policy?", CTX) == answer_key(
        "  the LEAVE policy  ", CTX
    ) or answer_key("What is the leave policy?", CTX) == answer_key("what is the leave policy", CTX)


def test_updating_a_document_makes_old_answers_unreachable() -> None:
    """The property that stops the cache serving last year's rules."""
    before = answer_key("what is the leave policy", CTX)
    after = answer_key(
        "what is the leave policy", KeyContext(corpus_version="corpus-2", model_id="route-a")
    )
    assert before != after


def test_prompt_and_policy_changes_invalidate() -> None:
    base = answer_key("q", CTX)
    from dataclasses import replace

    assert answer_key("q", replace(CTX, prompt_version="v2")) != base
    assert answer_key("q", replace(CTX, policy_version="v2")) != base
    assert answer_key("q", replace(CTX, model_id="route-b")) != base


def test_field_boundaries_cannot_be_confused() -> None:
    """('ab','c') and ('a','bc') must not hash to the same key."""
    a = answer_key("q", KeyContext(corpus_version="ab", prompt_version="c"))
    b = answer_key("q", KeyContext(corpus_version="a", prompt_version="bc"))
    assert a != b


def test_corpus_fingerprint_ignores_enumeration_order() -> None:
    assert corpus_fingerprint({"a": "1", "b": "2"}) == corpus_fingerprint({"b": "2", "a": "1"})


def test_corpus_fingerprint_tracks_content() -> None:
    assert corpus_fingerprint({"a": "1"}) != corpus_fingerprint({"a": "2"})


def _answer(text: str = "20 weeks.", tier: Tier = Tier.FRONTIER, cost: float = 0.02) -> Answer:
    return Answer(
        text=text,
        tier=tier,
        model_id="claude-opus-5",
        cache_key="k",
        citations=(Citation("hr-handbook", "HR Handbook", "20 weeks", "p.14"),),
        usage=Usage(input_tokens=3000, output_tokens=200),
        cost_usd=cost,
    )


def test_round_trip_preserves_citations(store: AnswerStore) -> None:
    store.put("k1", "q", _answer(), "corpus-1")
    entry = store.get("k1")
    assert entry is not None
    assert entry.citations[0].document_title == "HR Handbook"
    assert entry.usage.input_tokens == 3000
    assert entry.hits == 1


def test_hits_are_counted(store: AnswerStore) -> None:
    store.put("k1", "q", _answer(), "corpus-1")
    for expected in (1, 2, 3):
        entry = store.get("k1")
        assert entry is not None and entry.hits == expected


def test_refusals_are_never_cached(store: AnswerStore) -> None:
    """A refusal must get a fresh attempt once the corpus improves."""
    store.put("k1", "q", _answer(tier=Tier.REFUSED), "corpus-1")
    assert store.get("k1") is None


def test_pins_round_trip_and_can_be_removed(store: AnswerStore) -> None:
    store.pin("what is the leave policy", "20 weeks.", author="hr@example.com")
    pin = store.get_pin("what is the leave policy")
    assert pin is not None and pin.author == "hr@example.com"
    assert store.unpin("what is the leave policy") is True
    assert store.get_pin("what is the leave policy") is None
    assert store.unpin("what is the leave policy") is False


def test_pinning_twice_overwrites(store: AnswerStore) -> None:
    store.pin("q", "old")
    store.pin("q", "new")
    pin = store.get_pin("q")
    assert pin is not None and pin.answer == "new"
    assert len(store.list_pins()) == 1


def test_evicting_superseded_corpus_versions(store: AnswerStore) -> None:
    store.put("old", "q", _answer(), "corpus-1")
    store.put("new", "q", _answer(), "corpus-2")
    assert store.evict_other_corpus_versions("corpus-2") == 1
    assert store.get("old") is None
    assert store.get("new") is not None


def test_blended_cost_counts_the_free_answers(store: AnswerStore) -> None:
    """The headline number is only honest if cache hits are in the denominator."""
    store.record("q", _answer(cost=0.02))
    for _ in range(9):
        store.record("q", _answer(tier=Tier.EXACT_CACHE, cost=0.0))

    report = store.cost_report()
    assert report["questions"] == 10
    assert report["spend_usd"] == 0.02
    assert report["cost_per_question_usd"] == 0.002
    assert report["by_tier"]["exact"]["questions"] == 9


def test_empty_ledger_reports_zero_not_a_crash(store: AnswerStore) -> None:
    assert store.cost_report()["cost_per_question_usd"] == 0.0


def test_top_questions_surfaces_pin_candidates(store: AnswerStore) -> None:
    for _ in range(5):
        store.record("how do i book leave", _answer())
    store.record("what is the vpn host", _answer())
    assert store.top_questions(limit=1) == [("how do i book leave", 5)]


# -- pin provenance --------------------------------------------------------


def test_citations_are_built_from_the_indexed_corpus(retriever) -> None:
    from openknowledge.cache import citations_for

    (citation,) = citations_for(retriever, ("hr-handbook",))
    assert citation.document_title == "HR Handbook"
    assert "parental leave" in citation.snippet.lower()


def test_citing_a_document_that_is_not_indexed_is_visible_not_silent(retriever) -> None:
    """An admin can pin a wrong source; the reader should be able to tell."""
    from openknowledge.cache import citations_for

    (citation,) = citations_for(retriever, ("does-not-exist",))
    assert citation.document_id == "does-not-exist"
    assert "not currently in the indexed corpus" in citation.snippet


def test_citations_for_tolerates_a_retriever_without_the_method() -> None:
    from openknowledge.cache import citations_for

    (citation,) = citations_for(object(), ("some-doc",))
    assert citation.document_id == "some-doc"
