"""Which documents have any business agreeing with each other.

The detector's largest error, and the roadmap said so before this existed:
it assumed every document in the folder speaks for the same authority about
the same world. On fifteen real vendor contracts that produced findings that
were all false - one supplier's uptime commitment compared against another's
- and no weighting of shared words can fix it, because the words really are
shared. Contracts come from the same boilerplate.

What these hold is the direction of the risk. A false positive is noise; a
**suppressed real contradiction** means somebody is told the wrong policy
with a citation attached. So scope may only ever come from a party-defining
position, never from a mention, and a document that names no parties is
compared with everything exactly as before.
"""

from __future__ import annotations

from openknowledge.knowledge.claims import compare_documents
from openknowledge.knowledge.scope import (
    HEAD_CHARS,
    comparable,
    counterparties,
    named_parties,
)
from openknowledge.retrieval.base import Document

MSA = """MASTER SERVICES AGREEMENT

This Agreement is made and entered into by and between {vendor} ("Supplier")
and Aveline Holdings Limited ("Customer").

The Supplier shall maintain availability of {uptime}% measured monthly, and
either party may terminate on {notice} days prior written notice. The
Supplier's aggregate liability shall not exceed EUR {cap} in any period.
"""

POLICY = """# Travel and Expenses Policy

Travel expenses require prior written approval for any amount above EUR {cap}.
Bookings are made through {agent}, our appointed booking agent, and receipts
must be submitted within 30 days of the date incurred.
"""


def msa(doc_id: str, vendor: str, *, uptime: float, notice: int, cap: int) -> Document:
    return Document(
        doc_id, doc_id, MSA.format(vendor=vendor, uptime=uptime, notice=notice, cap=cap)
    )


def policy(doc_id: str, *, cap: int, agent: str) -> Document:
    return Document(doc_id, doc_id, POLICY.format(cap=cap, agent=agent))


def pairs_of(documents: list[Document]) -> set[tuple[str, str]]:
    conflicts, _ = compare_documents(documents)
    return {tuple(sorted((c.left.document_id, c.right.document_id))) for c in conflicts}


def test_a_mention_is_not_a_party() -> None:
    """The control that decides whether this is safe to ship at all.

    Two versions of one expenses policy that name *different* booking agents
    must still be compared - the moved figure between them is exactly the kind
    of contradiction this product exists to catch. Scope comes only from a
    party-defining position, so a company named in passing scopes nothing.
    """
    old = policy("expenses-2025", cap=500, agent="Corporate Travel Ltd")
    new = policy("expenses-2026", cap=1000, agent="Skyline Travel Ltd")

    assert named_parties(old) == frozenset()
    assert named_parties(new) == frozenset()
    assert ("expenses-2025", "expenses-2026") in pairs_of([old, new])


def test_two_agreements_with_different_companies_are_not_disagreeing() -> None:
    """The failure this was built for: 534 findings across 136 pairs on a
    corpus of the shape that broke it, and every cross-vendor pair false."""
    contracts = [
        msa("northwind", "Northwind Systems Ltd", uptime=99.5, notice=30, cap=500),
        msa("kestrel", "Kestrel Analytics GmbH", uptime=99.9, notice=60, cap=1000),
        msa("beacon", "Beacon Cloud Services Inc", uptime=99.95, notice=90, cap=2500),
    ]
    assert pairs_of(contracts) == set()


def test_two_versions_of_one_vendors_agreement_still_are() -> None:
    """Same counterparty, a figure moved. Scope must not touch this."""
    contracts = [
        msa("northwind-2025", "Northwind Systems Ltd", uptime=99.5, notice=30, cap=500),
        msa("northwind-2026", "Northwind Systems Ltd", uptime=99.5, notice=30, cap=2000),
        msa("kestrel", "Kestrel Analytics GmbH", uptime=99.9, notice=60, cap=1000),
    ]
    assert pairs_of(contracts) == {("northwind-2025", "northwind-2026")}


def test_your_own_company_is_not_what_tells_two_agreements_apart() -> None:
    """It is a party to everything you signed, so it is found and dropped.

    Without this every pair would share your own name, nothing would ever be
    scoped apart, and the whole thing would be inert.
    """
    contracts = [
        msa("a", "Northwind Systems Ltd", uptime=99.5, notice=30, cap=500),
        msa("b", "Kestrel Analytics GmbH", uptime=99.9, notice=60, cap=1000),
        msa("c", "Beacon Cloud Services Inc", uptime=99.95, notice=90, cap=2500),
    ]
    scoped = counterparties(contracts)
    assert scoped["a"] == frozenset({"northwind systems ltd"})
    assert scoped["b"] == frozenset({"kestrel analytics gmbh"})
    assert not any("aveline" in name for names in scoped.values() for name in names)


def test_one_relationship_stays_one_relationship() -> None:
    """Three agreements with the same vendor are about one world.

    That vendor is then as common as your own company, both are dropped, and
    the three compare - which is right. The rule that removes your own name
    has to land this way round or it would wall off the documents that most
    need comparing.
    """
    contracts = [
        msa("y1", "Northwind Systems Ltd", uptime=99.5, notice=30, cap=500),
        msa("y2", "Northwind Systems Ltd", uptime=99.5, notice=30, cap=900),
        msa("y3", "Northwind Systems Ltd", uptime=99.5, notice=30, cap=1500),
    ]
    assert all(scope == frozenset() for scope in counterparties(contracts).values())
    assert len(pairs_of(contracts)) == 3


def test_silence_is_not_a_scope() -> None:
    """One side naming nobody means no scope was established, and an
    unestablished scope must never be why a contradiction goes unreported."""
    assert comparable(frozenset(), frozenset({"acme ltd"}))
    assert comparable(frozenset({"acme ltd"}), frozenset())
    assert comparable(frozenset(), frozenset())
    assert comparable(frozenset({"acme ltd"}), frozenset({"acme ltd", "globex inc"}))
    assert not comparable(frozenset({"acme ltd"}), frozenset({"globex inc"}))


def test_parties_are_read_from_the_opening_not_from_a_schedule() -> None:
    """An agreement names its parties up front. A company named on page forty
    is a mention, and reading it as a party would scope documents apart on the
    strength of an appendix."""
    buried = Document(
        "long",
        "Long",
        "# Schedule\n\n"
        + ("filler text about the service. " * 200)
        + '\n\nThis Agreement is between Deepwater Systems Ltd ("Supplier") and others.',
    )
    assert len(buried.text) > HEAD_CHARS
    assert named_parties(buried) == frozenset()


def test_a_capitalised_phrase_is_not_a_company() -> None:
    """The patterns are case-sensitive on purpose.

    With ``re.IGNORECASE`` the ``[A-Z]`` that starts a name matches anything,
    and the role pattern reads backwards across the whole preamble - "March
    2025 by and between Northwind Systems Ltd" becomes the company's name.
    Two documents would then never agree on how that company is spelled, and
    the corpus-wide rule that drops your own name would stop working.
    """
    document = Document(
        "d",
        "D",
        "This Agreement is made as of 1 March 2025 by and between "
        'Northwind Systems Ltd, a company registered in England ("Supplier"), '
        'and Aveline Holdings Limited ("Customer").',
    )
    assert named_parties(document) == frozenset(
        {"northwind systems ltd", "aveline holdings limited"}
    )


def test_an_empty_corpus_is_not_a_special_case() -> None:
    assert counterparties([]) == {}
    assert pairs_of([]) == set()


def test_documents_that_name_only_their_own_counterparty_are_still_scoped() -> None:
    """No name repeats anywhere, so there is no common party to drop.

    An SLA schedule often names the provider and nobody else - your own
    company is on the covering agreement, not on every annex. Dropping the
    most-common name when nothing is shared would erase every scope there was
    and quietly put this back to comparing all of them.
    """
    schedules = [
        Document(
            "sla-beacon",
            "SLA",
            'SERVICE LEVEL SCHEDULE\n\nBeacon Cloud Services Inc. ("Provider") shall '
            "maintain availability of 99.9% measured monthly and shall retain "
            "personal data for no more than 30 days after termination.",
        ),
        Document(
            "sla-thornfield",
            "SLA",
            'SERVICE LEVEL SCHEDULE\n\nThornfield Security Ltd ("Provider") shall '
            "maintain availability of 99.5% measured monthly and shall retain "
            "personal data for no more than 90 days after termination.",
        ),
    ]
    scoped = counterparties(schedules)
    assert scoped["sla-beacon"] == frozenset({"beacon cloud services inc"})
    assert scoped["sla-thornfield"] == frozenset({"thornfield security ltd"})
    assert pairs_of(schedules) == set()
