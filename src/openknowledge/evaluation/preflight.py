"""Is this golden set answerable at all, before anybody spends money on it?

A live run costs tokens and takes minutes, and roughly half the failures a new
golden set produces have nothing to do with the model. The fact was in a document
the retriever never surfaced; the case cited a document id that does not exist;
the expected phrase was written differently from the corpus. All three look
exactly like a model failure in the report, and all three are free to detect.

So this runs the retrieval half only - no model, no cost, no network - and asks
one question per case: **if the model answered perfectly from what it was given,
could it pass?** A case that fails here is a bug in the case or in retrieval, and
fixing it before the run is the difference between a report that says something
and a report that says "seventeen failures, cause unknown".

The commonest of the three, and the least obvious in a report, is vocabulary.
Retrieval here does not stem, deliberately - "which expenses are not reimbursable"
must never collapse into "which expenses are reimbursable" - and the price is that
a question asking about *subsistence* against a corpus that says *meals* retrieves
nothing. In a live run that reads as the model refusing a question it was never
shown the answer to.

It is not a quality measurement. Passing means the evidence was present, not that
an answer will be right.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..retrieval.base import Chunk
from ..retrieval.bm25 import BM25Retriever
from ..retrieval.rerank import Reranker
from .dataset import Case


@dataclass(frozen=True, slots=True)
class CaseCheck:
    """What retrieval could and could not supply for one case."""

    case_id: str
    question: str
    #: Documents the case requires a citation from that were not retrieved.
    missing_citations: tuple[str, ...] = ()
    #: Document ids named by the case that are not in the corpus at all. A
    #: separate failure from "not retrieved": one is a ranking problem, the
    #: other is a typo, and telling an operator to tune retrieval for a typo
    #: wastes an afternoon.
    unknown_documents: tuple[str, ...] = ()
    #: Facts none of whose accepted spellings appear in the retrieved text,
    #: named by their first form.
    missing_phrases: tuple[str, ...] = ()
    retrieved: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return not (self.missing_citations or self.unknown_documents or self.unsupported)

    @property
    def unsupported(self) -> bool:
        """True when none of the case's facts were retrieved at all.

        A case listing several facts may legitimately have one of them stated
        somewhere the top-k did not reach; a case with none of them present was
        not retrievable and will fail for reasons no model can fix.
        """
        return bool(self.missing_phrases) and len(self.missing_phrases) == self._expected

    _expected: int = 0


@dataclass
class PreflightReport:
    checks: list[CaseCheck] = field(default_factory=list)
    skipped_refusals: int = 0

    @property
    def failures(self) -> list[CaseCheck]:
        return [c for c in self.checks if not c.ok]

    @property
    def passed(self) -> bool:
        return not self.failures


def preflight(
    cases: list[Case],
    *,
    retriever: BM25Retriever,
    k: int = 6,
    candidates: int = 30,
    reranker: Reranker | None = None,
) -> PreflightReport:
    """Check every answerable case against retrieval alone."""
    known = {doc_id for doc_id in retriever.document_ids()}
    report = PreflightReport()

    for case in cases:
        if case.kind != "answerable":
            # A refusal case is checked by the run itself: there is nothing
            # retrieval can promise about a question the corpus should not
            # answer, and demanding that nothing be retrieved would be wrong -
            # refusals usually retrieve plenty and ground none of it.
            report.skipped_refusals += 1
            continue

        hits = retriever.search(case.question, k=max(k, candidates) if reranker else k)
        if reranker is not None:
            hits = reranker.rerank(case.question, hits, k=k)
        chunks: list[Chunk] = [h.chunk for h in hits]
        retrieved = tuple(dict.fromkeys(c.document_id for c in chunks))
        haystack = " ".join(c.text.lower() for c in chunks)

        report.checks.append(
            CaseCheck(
                case_id=case.id,
                question=case.question,
                unknown_documents=tuple(d for d in case.must_cite if d not in known),
                missing_citations=tuple(
                    d for d in case.must_cite if d in known and d not in retrieved
                ),
                missing_phrases=tuple(
                    alternatives[0]
                    for alternatives in case.must_say
                    if not any(form.lower() in haystack for form in alternatives)
                ),
                retrieved=retrieved,
                _expected=len(case.must_say),
            )
        )
    return report


def format_preflight(report: PreflightReport) -> str:
    lines = [
        f"Retrieval pre-flight: {len(report.checks)} answerable case(s), "
        f"{report.skipped_refusals} refusal case(s) not checked here.",
        "",
    ]
    for check in report.checks:
        mark = "ok  " if check.ok else "FAIL"
        lines.append(f"  [{mark}] {check.case_id}")
        for doc in check.unknown_documents:
            lines.append(f"      must_cite names {doc!r}, which is not in the corpus")
        for doc in check.missing_citations:
            lines.append(f"      {doc} exists but was not retrieved for this question")
        if check.unsupported:
            lines.append(f"      none of {list(check.missing_phrases)} appear in the context")

    lines.append("")
    if report.passed:
        lines.append(
            "PASSED - every answerable case has its evidence in the context. A failure "
            "in the live run will be the model's, not the corpus's."
        )
    else:
        lines.append(
            f"FAILED - {len(report.failures)} case(s) cannot be answered from what "
            "retrieval supplies. Fix these before paying for a run: a model cannot "
            "cite what it was never shown."
        )
    return "\n".join(lines)
