"""The grounding gate.

This is what stands between "cheap" and "cheap and wrong". A small local model
is perfectly capable of writing a fluent, confident, invented answer about your
expenses policy, and the cost saving is worthless if that ships. So no answer
from any tier is returned until it survives these checks:

1. **It cites something.** An uncited answer to a policy question is a guess.
2. **It cites something real.** Every referenced document id must be in the set
   actually retrieved for this question - a model naming a plausible-sounding
   document it never saw is the classic RAG failure.
3. **Its numbers appear in the sources.** Policy answers are made of numbers -
   "20 weeks", "30 days", "EUR 50" - and a wrong number is the most damaging and
   most confident-looking error. Any figure in the answer must occur in the text
   it cites.
4. **It stays close to the source wording.** Enough of the answer's content words
   must appear in the cited passages. This is a blunt instrument, tuned to catch
   free-form invention rather than to judge style.

A failure is not an error - it is the escalation signal. The local tier failing
this gate is exactly the case the frontier tier exists to serve, which is what
makes the cost saving safe rather than a gamble.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .base import Chunk, tokenize

#: Citation markers the answer template asks models to emit, e.g. ``[hr-handbook]``.
_CITATION_RE = re.compile(r"\[([A-Za-z0-9][A-Za-z0-9._#/-]{0,127})\]")

#: Numbers that carry meaning. Bare list indices and years are noisy, but a
#: figure attached to a policy is exactly what we want to verify.
_NUMBER_RE = re.compile(r"\d[\d,.]*")

#: Function words carry no evidence, so they should not inflate the overlap
#: score. This is a scoring aid only - unlike query canonicalisation, dropping
#: these cannot change an answer, only how strictly we grade it.
_FUNCTION_WORDS = frozenset(
    (  # noqa: SIM905 - a 60-item list literal here is far less readable
        "a an the and or but if then than that this these those of to in on at by for "
        "with from as is are was were be been being do does did have has had will would "
        "can could should may might must it its you your we our they their he she them "
        "there here what which who whom when where how why not no yes all any some each "
        "per about into over under between within"
    ).split()
)

_PHRASES_MEANING_NO_ANSWER = (
    "i don't know",
    "i do not know",
    "not in the provided",
    "no information",
    "cannot find",
    "could not find",
    "unable to answer",
    "not covered by",
    "insufficient information",
)


@dataclass(frozen=True, slots=True)
class GroundingReport:
    passed: bool
    cited_ids: tuple[str, ...] = ()
    unknown_ids: tuple[str, ...] = ()
    unsupported_numbers: tuple[str, ...] = ()
    support_ratio: float = 0.0
    abstained: bool = False
    reasons: tuple[str, ...] = ()
    #: Share of the answer's substantive claims that carry a resolving
    #: citation. 1.0 is the citation discipline that earns the lower floor.
    cited_coverage: float = 0.0


#: How answers refer to passages by the context's own labels. The SOURCES
#: block introduces every passage as ``[doc-id] Title (chunk 4)``, and models
#: - frontier models especially - faithfully echo that vocabulary: "(chunk 2)",
#: "reinforced in chunk 4", "chunks 4, 5 and 6". Measured in the field: three
#: Azure answers rejected in a row for "inventing" the numbers 17, 4, 5, 6 -
#: every one a chunk label we ourselves put in front of the model.
_CHUNK_REF_RE = re.compile(r"\bchunks?\s+(\d{1,3}(?:\s*(?:,|and|&)\s*\d{1,3})*)", re.IGNORECASE)


def _resolve_chunk_references(
    answer_text: str, retrieved: list[Chunk]
) -> tuple[tuple[Chunk, ...], str]:
    """Chunks the answer names by label, and the text with those names removed.

    A reference resolves only when every number in it names a retrieved
    chunk's locator - "chunk 4" with chunk 4 in context is the model citing
    what it saw; "chunk 99" resolves to nothing and stays in the text, where
    the figure check will flag 99 exactly as before. Removal matters as much
    as resolution: the number regex reads "chunks 4,5,6" as the single
    figure 4,5,6 and no source will ever contain it.
    """
    by_locator: dict[str, Chunk] = {}
    for chunk in retrieved:
        if chunk.locator:
            by_locator.setdefault(chunk.locator, chunk)

    resolved: dict[str, Chunk] = {}
    kept: list[str] = []
    last = 0
    for match in _CHUNK_REF_RE.finditer(answer_text):
        named = [by_locator.get(f"chunk {n}") for n in re.findall(r"\d+", match.group(1))]
        hits = [c for c in named if c is not None]
        if hits and len(hits) == len(named):
            for hit in hits:
                resolved[hit.chunk_id] = hit
            kept.append(answer_text[last : match.start()])
            last = match.end()
    kept.append(answer_text[last:])
    return tuple(resolved.values()), "".join(kept)


#: A claim needs at least this many words to be worth a citation. Below it,
#: a line is connective tissue ("Key changes include:") that asserts nothing.
_CLAIM_MIN_WORDS = 8

_SENTENCES = re.compile(r"(?<=[.!?])\s+")
_BULLET = re.compile(r"^\s*(?:[-*•·]|\d{1,3}[.)])\s+")


def _claim_coverage(answer_text: str, resolving_ids: frozenset[str]) -> tuple[int, int]:
    """(claims, cited claims): how disciplined the answer's citing is.

    A bullet is one claim however many sentences it holds - its trailing
    citation covers the bullet. Prose splits into sentences. Lines too short
    to assert anything are not claims and need no citation.
    """
    claims = 0
    cited = 0
    for raw_line in answer_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        is_bullet = bool(_BULLET.match(line))
        pieces = [line] if is_bullet else _SENTENCES.split(line)
        # A citation after the sentence's final period splits into its own
        # fragment; it belongs to the sentence it follows.
        merged: list[str] = []
        for piece in pieces:
            if merged and not tokenize(_CITATION_RE.sub("", piece)):
                merged[-1] += " " + piece
            else:
                merged.append(piece)
        for piece in merged:
            words = tokenize(_CITATION_RE.sub("", piece))
            if len(words) < _CLAIM_MIN_WORDS:
                continue
            claims += 1
            if any(cid in resolving_ids for cid in _CITATION_RE.findall(piece)):
                cited += 1
    return claims, cited


def _normalise_number(raw: str) -> str:
    """Compare numbers by value, not spelling: '1,200' and '1200' are the same,
    and a trailing sentence period is not a decimal point."""
    cleaned = raw.rstrip(".").replace(",", "")
    try:
        value = float(cleaned)
    except ValueError:
        return cleaned
    return str(int(value)) if value.is_integer() else str(value)


def check_grounding(
    answer_text: str,
    retrieved: list[Chunk],
    *,
    min_support_ratio: float = 0.45,
    require_citations: bool = True,
    min_support_ratio_cited: float = 0.30,
) -> GroundingReport:
    """Grade ``answer_text`` against the chunks it was allowed to see.

    Two support floors, because summaries and extractions fail differently.
    A faithful summary compresses and rephrases, which is exactly what a
    word-overlap ratio penalises - measured in the field, a correct
    six-bullet summary with every bullet cited scored 42% against the 45%
    floor and was withdrawn. So an answer that shows full citation
    discipline - every substantive claim cites a retrieved source, no
    unknown ids, no unverified figures - is graded against
    ``min_support_ratio_cited`` instead. Everything else keeps the
    original floor: relaxation is earned per answer, never granted to the
    model in general.
    """
    lowered = answer_text.lower().strip()

    if not lowered:
        return GroundingReport(passed=False, reasons=("empty answer",))

    # An explicit "I don't know" is correct behaviour, not a grounding failure -
    # but it is not an answer either, so it must not be cached as one.
    if any(phrase in lowered for phrase in _PHRASES_MEANING_NO_ANSWER):
        return GroundingReport(
            passed=False, abstained=True, reasons=("model declined to answer from the sources",)
        )

    reasons: list[str] = []
    known_ids = {c.document_id for c in retrieved} | {c.chunk_id for c in retrieved}
    cited = tuple(dict.fromkeys(_CITATION_RE.findall(answer_text)))

    # References in the context's own vocabulary - "(chunk 2)", "chunks 4, 5
    # and 6" - are citations too: they name passages the model was shown, in
    # the labels we introduced them with. They count toward *having* cited,
    # and the named chunks join the evidence; the earned lower support floor
    # stays reserved for the [id] discipline the prompt actually asks for.
    referenced, scrubbed = _resolve_chunk_references(answer_text, retrieved)
    referenced_chunk_ids = {c.chunk_id for c in referenced}

    if require_citations and not cited and not referenced:
        reasons.append("answer cites no sources")

    unknown = tuple(cid for cid in cited if cid not in known_ids)
    if unknown:
        reasons.append(f"cites sources not retrieved for this question: {', '.join(unknown)}")

    # Grade against the chunks the answer names - by id or by label -
    # otherwise against everything retrieved, so an uncited answer is still
    # measured.
    cited_chunks = [
        c
        for c in retrieved
        if c.document_id in cited or c.chunk_id in cited or c.chunk_id in referenced_chunk_ids
    ]
    evidence = cited_chunks or retrieved
    evidence_text = " ".join(c.text for c in evidence)
    evidence_tokens = set(tokenize(evidence_text))
    evidence_numbers = {_normalise_number(n) for n in _NUMBER_RE.findall(evidence_text)}
    # The header line above every passage - [doc-id] Title (chunk 4) - is
    # part of what the model saw, so its numbers are evidence: an answer
    # saying "the 2023 policy" with 2023 only in the title, or naming a cell
    # like A2, is reading the context, not inventing figures. Headers of all
    # retrieved chunks count, because the model saw all of them.
    evidence_numbers.update(
        _normalise_number(n)
        for c in retrieved
        for n in _NUMBER_RE.findall(f"{c.document_id} {c.document_title} {c.locator or ''}")
    )

    answer_numbers = tuple(
        dict.fromkeys(
            n
            for n in _NUMBER_RE.findall(_CITATION_RE.sub("", scrubbed))
            if _normalise_number(n) not in evidence_numbers
        )
    )
    if answer_numbers:
        reasons.append(f"figures not found in the cited text: {', '.join(answer_numbers)}")

    content_words = [
        w for w in tokenize(_CITATION_RE.sub("", answer_text)) if w not in _FUNCTION_WORDS
    ]
    if content_words:
        supported = sum(1 for w in content_words if w in evidence_tokens)
        support_ratio = supported / len(content_words)
    else:
        support_ratio = 0.0

    resolving = frozenset(cid for cid in cited if cid in known_ids)
    claims, cited_claims = _claim_coverage(answer_text, resolving)
    coverage = cited_claims / claims if claims else 0.0

    # The lower floor is earned by this answer's own discipline: every
    # substantive claim cited, every citation real, every figure verified.
    # An answer with no claims long enough to need citations earns nothing -
    # short answers pass or fail exactly as before.
    fully_cited = (
        claims > 0 and cited_claims == claims and bool(cited) and not unknown and not answer_numbers
    )
    floor = min(min_support_ratio, min_support_ratio_cited) if fully_cited else min_support_ratio

    if support_ratio < floor:
        cited_note = " even for a fully cited answer" if fully_cited else ""
        reasons.append(
            f"only {support_ratio:.0%} of the answer's content words appear in the sources "
            f"(need {floor:.0%}{cited_note})"
        )

    return GroundingReport(
        passed=not reasons,
        cited_ids=tuple(dict.fromkeys((*cited, *(c.document_id for c in referenced)))),
        unknown_ids=unknown,
        unsupported_numbers=answer_numbers,
        support_ratio=round(support_ratio, 4),
        reasons=tuple(reasons),
        cited_coverage=round(coverage, 4),
    )
