"""A quotation is a claim that these exact words are in the document.

v0.12.10 shipped a correct answer whose supporting citation was invented.
Asked who approves a contract of exactly EUR 25,000, the model answered "Head
of Department" - right - and quoted, in the policy table's own column format,
a row that does not exist:

    "Contract value (annual): Above EUR 25,000 | Approver: Chief Financial
     Officer | Additional requirement: Three competitive quotes"

"Above EUR 25,000" is in the document, as prose about competitive quotes. The
Chief Financial Officer is in the document, in a different row. The support
ratio is word and number overlap, so a sentence welded together out of the
corpus's own vocabulary is perfectly grounded by construction - which is also
how an invented fourth act of a three-act play passed at 96% support.

The floor of eight words and the shape of the normalisation are not judgement
calls; `tools/measure_quotations.py` measured them over 99 quoted spans from
this repository's own evaluation runs. Every case below is drawn from that
sample, including the three honest spans a naive check would have rejected.
"""

from __future__ import annotations

from openknowledge.retrieval import check_grounding
from openknowledge.retrieval.base import Chunk

#: The real table, verbatim, as evals/corpus/aveline states it.
_POLICY = Chunk(
    chunk_id="finance-procurement-policy#1",
    document_id="finance-procurement-policy",
    document_title="Procurement Policy",
    locator="chunk 2",
    text=(
        "## 1. Authority to commit\n\n"
        "| Contract value (annual) | Approver | Additional requirement |\n"
        "|---|---|---|\n"
        "| Up to EUR 5,000 | Line manager | None |\n"
        "| EUR 5,001 to EUR 25,000 | Head of Department | Written business case |\n"
        "| EUR 25,001 to EUR 50,000 | Chief Financial Officer | Three competitive quotes |\n"
        "| Above EUR 50,000 | Board | Three quotes and legal review |\n\n"
        "## 2. Competitive quotes\n\n"
        "Three competitive quotes are required for any contract with an annual value "
        "above **EUR 25,000**."
    ),
)


def _report(answer: str, question: str = ""):
    return check_grounding(answer, [_POLICY], question=question)


def test_the_fabricated_row_v0_12_10_shipped_is_caught() -> None:
    """The transcript, not a hypothetical. The answer is correct and every word
    of its 'quotation' occurs somewhere in the document."""
    report = _report(
        "A contract of exactly EUR 25,000 is approved by the Head of Department "
        "[finance-procurement-policy].\n"
        '"Contract value (annual): Above EUR 25,000 | Approver: Chief Financial Officer '
        '| Additional requirement: Three competitive quotes"'
    )
    assert not report.passed
    assert report.unquotable
    assert "Chief Financial Officer" in report.unquotable[0]
    assert any("not in the cited sources" in r for r in report.reasons)


def test_a_real_quotation_of_the_same_table_passes() -> None:
    report = _report(
        "The approver is the Head of Department [finance-procurement-policy]. The table "
        'reads "EUR 5,001 to EUR 25,000 | Head of Department | Written business case".'
    )
    assert not report.unquotable, report.unquotable


def test_the_normalisation_the_measurement_earned() -> None:
    """Exact matching finds 70.7% of real quotations; Markdown emphasis adds
    21.2 points and case-folding 3.0. Each rung is one of these."""
    for label, quoted in (
        ("emphasis the model added", "**EUR 5,001 to EUR 25,000** | Head of Department"),
        ("re-wrapped across lines", "EUR 5,001 to EUR 25,000 |\n        Head of Department"),
        ("case changed", "eur 5,001 to eur 25,000 | head of department"),
    ):
        report = _report(
            f'The approver is the Head of Department [finance-procurement-policy]: "{quoted}".'
        )
        assert not report.unquotable, f"{label}: {report.unquotable}"


def test_a_paraphrase_in_quotation_marks_is_not_a_quotation() -> None:
    """From the sample, on a different corpus: the source says the allowance is
    a daily payment instead of reimbursement, and the answer quoted a sentence
    of its own invention that says something else."""
    report = _report(
        "Quotes are required above EUR 25,000 [finance-procurement-policy]. The policy "
        'states "every contract of any size must be supported by three written quotes '
        'from unrelated suppliers".'
    )
    assert report.unquotable


# --- the three honest spans a naive check would have rejected ---------------


def test_a_schematic_placeholder_is_not_a_quotation() -> None:
    """From the sample: "both documents use "from X to Y" bands". The answer is
    not claiming the document contains the letters X and Y."""
    report = _report(
        "The approver is the Head of Department [finance-procurement-policy], because "
        'both documents use "from X to Y" bands that start above a threshold.'
    )
    assert not report.unquotable, report.unquotable


def test_quoting_a_phrase_in_order_to_deny_it_is_not_a_quotation() -> None:
    """Also from the sample, and the subtler of the two: the answer quotes
    wording precisely to say the document does NOT use it."""
    report = _report(
        "Quotes are not required at exactly EUR 25,000 [finance-procurement-policy]. "
        'The policy specifies "above EUR 25,000" and not "equal to or above".'
    )
    assert not report.unquotable, report.unquotable


def test_a_long_denial_is_exempt_too_though_the_sample_had_none() -> None:
    """The length floor already excludes both denials the sample contained, at
    four words each. This is the case the sample did not have: a denial long
    enough to clear the floor, which the floor alone would reject."""
    report = _report(
        "Quotes are not required at exactly EUR 25,000 [finance-procurement-policy]. "
        'The policy says "above EUR 25,000", not "any contract at a value equal to or '
        'greater than twenty-five thousand euro whatsoever".'
    )
    assert not report.unquotable, report.unquotable


def test_a_leading_question_cannot_legitimise_its_own_quotation() -> None:
    """The figure check admits the asker's numbers; this check does not admit
    the asker's prose, and the asymmetry is deliberate.

    A number has to be repeatable or "is EUR 40,000 above the limit?" cannot be
    answered at all - and even there it is admitted only beside a figure from
    the sources, because attacking the first version of that fix showed a
    leading question could otherwise put a number into policy. Quoted prose has
    no such need and the identical hole, so the asker is not a source.
    """
    question = (
        'Does the policy say "all contracts of any value require three competitive '
        'quotes from unrelated suppliers"?'
    )
    report = _report(
        "Yes [finance-procurement-policy]. The policy says "
        '"all contracts of any value require three competitive quotes from unrelated '
        'suppliers".',
        question=question,
    )
    assert report.unquotable, "the question must not vouch for the answer's quotation"


def test_a_short_quoted_phrase_is_a_mention_not_a_claim() -> None:
    """Under the eight-word floor nothing is checked, because every honest span
    the measurement would have wrongly rejected was four words long."""
    report = _report('The rule turns on "above", not on the band [finance-procurement-policy].')
    assert not report.unquotable
