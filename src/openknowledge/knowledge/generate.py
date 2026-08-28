"""Drafting FAQ answers from documents at ingest time.

This is where the project's economics invert. Answering at query time costs
money on every question, forever, and grows with adoption. Drafting at ingest
time costs money once per document, and documents change rarely. Moving work
across that line converts a recurring variable cost into a one-off fixed one.

The output is not trusted because a model produced it. Every draft goes through
exactly the same grounding gate a live answer does - cites real retrieved
sources, uses only numbers present in them, stays close to the source wording -
and anything that fails is discarded rather than queued. A draft that survives
is a *precomputed cache entry*: no more trustworthy than a live answer, and no
less. Human approval is what promotes it to a pin.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass

from ..canonical import canonicalize_query
from ..prompts import format_context
from ..providers.base import ChatProvider, ProviderError
from ..retrieval.base import Chunk, Document, chunk_document
from ..retrieval.grounding import check_grounding
from ..types import Citation

log = logging.getLogger(__name__)

DRAFT_PROMPT_VERSION = "v1"

#: Deliberately narrow. We are not asking for a summary or for anything
#: creative - we want the questions an employee would actually type, answered
#: with the document's own words and figures.
DRAFT_SYSTEM_PROMPT = """\
You are preparing a frequently-asked-questions list from one internal company \
document, so that employees can get answers without anyone re-reading the \
document each time.

You will be given the document as SOURCES, each part introduced by a line of the \
form:

    [document-id] Document Title (location)

Produce the questions employees would genuinely type, and answer each one from \
the document alone.

Return ONLY a JSON array. No preamble, no explanation, no markdown fence. Each \
element must be an object with exactly two string fields:

    [{"question": "...", "answer": "..."}]

Rules:

1. Every answer must come from the SOURCES and nothing else. You have no other \
knowledge of this organisation. If part of a question cannot be answered from \
the document, do not ask that question.

2. Copy figures exactly. Amounts, durations, deadlines, thresholds, percentages \
and dates must appear in the answer exactly as the document writes them. Never \
round, convert, or infer a figure.

3. Keep the conditions. Most internal rules depend on tenure, department, \
amount, or prior approval. An answer that states the headline figure and drops \
the condition is wrong. If a rule has exceptions, state them.

4. Cite the source in every answer using its document id in square brackets, \
like [expenses-policy], exactly as it appears in the SOURCES block.

5. Ask the question the way an employee would - "How much can I claim for \
meals?", not "What is the per-diem subsistence allowance threshold?". Ask one \
thing at a time.

6. Prefer questions about rules people must follow: limits, deadlines, who \
approves what, what is and is not allowed. Skip questions about the document's \
own structure or revision history.

7. Produce at most 8 questions. Fewer is fine. Do not pad.
"""

_JSON_ARRAY = re.compile(r"\[.*\]", re.DOTALL)


@dataclass(frozen=True, slots=True)
class DraftedAnswer:
    """One question and answer drafted from a document, already gate-checked."""

    question: str
    canonical_query: str
    answer: str
    citations: tuple[Citation, ...]
    origin_documents: tuple[str, ...]
    support_ratio: float


@dataclass(frozen=True, slots=True)
class DraftResult:
    drafted: tuple[DraftedAnswer, ...]
    #: Pairs the model produced that failed the gate, with the reason. Kept so
    #: an operator can see whether a model is worth its tokens rather than only
    #: seeing what survived.
    rejected: tuple[tuple[str, str], ...]
    cost_usd: float
    model_id: str


def _parse_pairs(text: str) -> list[tuple[str, str]]:
    """Pull question/answer pairs out of a model response.

    Tolerant of a stray preamble or a markdown fence, because rejecting a whole
    document's worth of drafting over a backtick would be a poor trade. Anything
    that is not a well-formed pair is dropped silently - the gate is the real
    filter, this is just parsing.
    """
    match = _JSON_ARRAY.search(text)
    if match is None:
        return []
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []

    pairs: list[tuple[str, str]] = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        question = item.get("question")
        answer = item.get("answer")
        if isinstance(question, str) and isinstance(answer, str):
            question, answer = question.strip(), answer.strip()
            if question and answer:
                pairs.append((question, answer))
    return pairs


def _citations_from(chunks: list[Chunk], cited_ids: tuple[str, ...]) -> tuple[Citation, ...]:
    wanted = set(cited_ids)
    seen: dict[str, Citation] = {}
    for chunk in chunks:
        if chunk.document_id not in wanted or chunk.document_id in seen:
            continue
        snippet = chunk.text[:280] + ("..." if len(chunk.text) > 280 else "")
        seen[chunk.document_id] = Citation(
            document_id=chunk.document_id,
            document_title=chunk.document_title,
            snippet=snippet,
            locator=chunk.locator,
            url=chunk.url,
        )
    return tuple(seen.values())


async def draft_from_document(
    provider: ChatProvider,
    document: Document,
    *,
    min_support_ratio: float = 0.45,
    min_support_ratio_cited: float = 0.30,
    max_tokens: int = 2000,
    target_words: int = 350,
    overlap_words: int = 60,
) -> DraftResult:
    """Draft FAQ entries from one document and keep only what passes the gate."""
    chunks = chunk_document(document, target_words=target_words, overlap_words=overlap_words)
    if not chunks:
        return DraftResult((), (), 0.0, getattr(provider, "model_id", "none"))

    context = format_context(chunks)
    try:
        completion = await provider.complete(
            system=DRAFT_SYSTEM_PROMPT,
            context=context,
            question=("Write the FAQ for this document now. Return only the JSON array."),
            max_tokens=max_tokens,
        )
    except ProviderError as exc:
        log.warning("drafting failed for %s: %s", document.document_id, exc)
        return DraftResult((), (), 0.0, getattr(provider, "model_id", "none"))

    from ..costs import PricingError, cost_usd, get_price

    model_id = "local" if getattr(provider, "tier", "") == "local" else completion.model_id
    try:
        cost = cost_usd(completion.usage, get_price(model_id))
    except PricingError:
        cost = 0.0

    drafted: list[DraftedAnswer] = []
    rejected: list[tuple[str, str]] = []
    seen: set[str] = set()

    for question, answer in _parse_pairs(completion.text):
        canonical = canonicalize_query(question)
        if not canonical:
            rejected.append((question, "question is empty after normalisation"))
            continue
        if canonical in seen:
            rejected.append((question, "duplicate of an earlier question in this document"))
            continue

        report = check_grounding(
            answer,
            chunks,
            min_support_ratio=min_support_ratio,
            min_support_ratio_cited=min_support_ratio_cited,
        )
        if not report.passed:
            rejected.append((question, "; ".join(report.reasons)))
            continue

        seen.add(canonical)
        drafted.append(
            DraftedAnswer(
                question=question,
                canonical_query=canonical,
                answer=answer,
                citations=_citations_from(chunks, report.cited_ids),
                origin_documents=(document.document_id,),
                support_ratio=report.support_ratio,
            )
        )

    return DraftResult(tuple(drafted), tuple(rejected), cost, completion.model_id)
