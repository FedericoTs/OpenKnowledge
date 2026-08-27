"""Canonicalisation must raise the cache hit rate without ever changing meaning."""

from __future__ import annotations

import pytest

from openknowledge.canonical import canonicalize_query as c


@pytest.mark.parametrize(
    ("a", "b"),
    [
        ("What is the parental leave policy?", "what is the parental leave policy"),
        ("Hi, what is the parental leave policy?", "what is the parental leave policy"),
        ("Hello please can you tell me how do I book leave?", "how do i book leave"),
        ("What   is  the\tpolicy", "what is the policy"),
        ("What is the “remote work” policy?", 'what is the "remote work" policy'),
        ("Café allowance?", "café allowance"),
    ],
)
def test_equivalent_phrasings_collapse(a: str, b: str) -> None:
    assert c(a) == c(b)


@pytest.mark.parametrize(
    ("a", "b"),
    [
        # These are the pairs a stopword list would destroy. If any of these ever
        # collapse, the bot starts telling people the opposite of the policy.
        ("which expenses are reimbursable", "which expenses are not reimbursable"),
        ("can I expense alcohol", "can I never expense alcohol"),
        ("approval needed before travel", "approval needed after travel"),
        ("who may approve this", "who must approve this"),
        ("leave with pay", "leave without pay"),
    ],
)
def test_meaning_changing_words_are_never_stripped(a: str, b: str) -> None:
    assert c(a) != c(b)


def test_idempotent() -> None:
    for q in ["Hi! What's the VPN setup?", "  spaced   out  ", "PLAIN"]:
        assert c(c(q)) == c(q)


def test_invisible_characters_do_not_split_the_cache() -> None:
    # A question pasted out of Word carries zero-width joiners and a BOM.
    assert c("what is​ the policy﻿") == c("what is the policy")


def test_empty_and_punctuation_only() -> None:
    assert c("") == ""
    assert c("???") == ""
