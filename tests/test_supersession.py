"""A document retired by the one that replaced it.

`declares_superseded` reads a document's own head and trusts it, which needs
somebody to have gone back and edited the old file. In practice nobody does:
the statement that gets written is in the *new* document's header, and this
project's own sample corpus is an example - `hr/expenses-policy.md` says
"**Supersedes:** Expenses Policy v3.0 (January 2023)" and that line was
ignored until now.

The risk these hold down is specific. Retrieval does not downrank a
superseded document, it **excludes** it whenever any current document matches
(`demote_superseded`), so resolving a name to the wrong document takes a live
policy out of almost every answer. Every test here is about not doing that.
"""

from __future__ import annotations

from openknowledge.knowledge.supersession import HEAD_CHARS, announced_by, apply
from openknowledge.retrieval.base import Document

HEADER = "**Owner:** Finance · **Version:** {version} **Supersedes:** {named}\n\n{body}"


def doc(doc_id: str, title: str, text: str, *, superseded: bool = False) -> Document:
    return Document(doc_id, title, text, superseded=superseded)


def test_the_newer_document_retires_the_one_it_names() -> None:
    """The case the sample corpus already contains and nothing read."""
    corpus = [
        doc(
            "hr-expenses",
            "Expenses Policy",
            HEADER.format(
                version="4.1",
                named="Expenses Policy v3.0 (January 2023)",
                body="Approval above EUR 1,000.",
            ),
        ),
        doc("archive-expenses-2023", "Expenses Policy (2023)", "Approval above EUR 500."),
    ]
    assert announced_by(corpus) == {"archive-expenses-2023": "hr-expenses"}

    marked, retired = apply(corpus)
    assert retired == {"archive-expenses-2023": "hr-expenses"}
    assert [d.superseded for d in marked] == [False, True]


def test_the_announcer_never_retires_itself() -> None:
    """Its own title fits inside the phrase it wrote - "Expenses Policy" is a
    subset of "Expenses Policy v3.0". Without excluding it, the current policy
    would declare itself gone and vanish from every answer."""
    corpus = [
        doc(
            "hr-expenses",
            "Expenses Policy",
            HEADER.format(
                version="4.1",
                named="Expenses Policy v3.0 (January 2023)",
                body="Approval above EUR 1,000.",
            ),
        ),
    ]
    assert announced_by(corpus) == {}


def test_a_name_that_fits_two_documents_retires_neither() -> None:
    """A tie means nobody can tell which was meant, and the safe answer to
    "which of these did they retire?" is to retire neither."""
    corpus = [
        doc(
            "new",
            "Travel Standard",
            HEADER.format(version="2", named="Travel Policy", body="Economy only."),
        ),
        doc("one", "Travel Policy", "Economy."),
        doc("two", "Travel Policy", "Economy, elsewhere."),
    ]
    assert announced_by(corpus) == {}


def test_a_shared_word_is_not_a_match() -> None:
    """Almost every document in a policy folder has "Policy" in its title.
    Matching on that alone would retire whatever sorted first."""
    corpus = [
        doc(
            "new",
            "Expenses Standard",
            HEADER.format(version="2", named="the old Policy", body="EUR 1,000."),
        ),
        doc("leave", "Parental Leave Policy", "20 weeks."),
        doc("travel", "Travel Policy", "Economy."),
    ]
    assert announced_by(corpus) == {}


def test_prose_is_not_a_declaration() -> None:
    """The colon is what makes it a field rather than a sentence.

    Inferring supersession from prose would be a guess, and a wrong guess
    hides a live policy from almost every question. Deliberately out of scope.
    """
    corpus = [
        doc(
            "new",
            "Travel Standard",
            "This standard supersedes the Travel Guidelines that came before it.",
        ),
        doc("old", "Travel Guidelines", "Book economy."),
    ]
    assert announced_by(corpus) == {}


def test_a_notice_buried_in_the_body_is_a_mention() -> None:
    """Headers are at the top. A line about some other policy on page nine is
    the document discussing the world, not declaring a replacement."""
    corpus = [
        doc(
            "new",
            "Travel Standard",
            ("Booking rules. " * 200) + "\n\nSupersedes: Travel Guidelines 2023",
        ),
        doc("old", "Travel Guidelines 2023", "Book economy."),
    ]
    assert len(corpus[0].text) > HEAD_CHARS
    assert announced_by(corpus) == {}


def test_a_document_that_already_says_so_itself_is_not_reported_twice() -> None:
    """It was not this that retired it. Reporting it would tell an operator
    something happened when nothing did."""
    corpus = [
        doc(
            "hr-expenses",
            "Expenses Policy",
            HEADER.format(
                version="4.1", named="Expenses Policy v3.0 (January 2023)", body="EUR 1,000."
            ),
        ),
        doc("archive", "Expenses Policy (2023)", "EUR 500.", superseded=True),
    ]
    marked, retired = apply(corpus)
    assert retired == {}
    assert marked is corpus, "an unchanged corpus should not be rebuilt"


def test_a_corpus_nobody_wrote_a_notice_in_is_untouched() -> None:
    corpus = [doc("a", "Travel Policy", "Economy."), doc("b", "Leave Policy", "20 weeks.")]
    marked, retired = apply(corpus)
    assert retired == {} and marked is corpus


def test_naming_a_document_that_is_not_here_does_nothing() -> None:
    corpus = [
        doc(
            "new",
            "Expenses Policy",
            HEADER.format(
                version="4.1", named="Expenses Policy v3.0 (January 2023)", body="EUR 1,000."
            ),
        ),
        doc("other", "Travel Guidelines", "Economy."),
    ]
    assert announced_by(corpus) == {}


def test_a_one_word_title_is_not_enough_to_retire_a_document() -> None:
    """A title that reduces to a single word could sit inside almost any
    notice. Two words is the floor, and it is what stops a document called
    "Expenses" being retired by a sentence that merely says "expenses"."""
    corpus = [
        doc(
            "new",
            "Corporate Expenses Standard",
            HEADER.format(version="2", named="Expenses Policy 2023 edition", body="EUR 1,000."),
        ),
        doc("bare", "Expenses", "EUR 500."),
    ]
    from openknowledge.knowledge.supersession import _significant

    # It would match on the subset rule alone - the floor is the only guard.
    assert _significant("Expenses") <= _significant("Expenses Policy 2023 edition")
    assert announced_by(corpus) == {}


def test_the_more_specific_title_wins_not_the_looser_one() -> None:
    """The dangerous case, and the reason the best match is the *largest*.

    A notice naming "Expenses Policy v3.0 (January 2023)" fits both the
    archived copy and the current policy. Taking the loosest match would
    retire the current one - and retrieval excludes a superseded document
    whenever anything current matches, so the live policy would vanish from
    almost every answer while the archived figure answered in its place.
    """
    corpus = [
        doc(
            "new",
            "Corporate Expenses Standard",
            HEADER.format(
                version="5", named="Expenses Policy v3.0 (January 2023)", body="EUR 2,000."
            ),
        ),
        doc("archive", "Expenses Policy (2023)", "EUR 500."),
        doc("current", "Expenses Policy", "EUR 1,000."),
    ]
    from openknowledge.knowledge.supersession import _significant

    named = _significant("Expenses Policy v3.0 (January 2023)")
    assert _significant("Expenses Policy") <= named, "the loose one really does fit"
    assert _significant("Expenses Policy (2023)") <= named

    assert announced_by(corpus) == {"archive": "new"}
