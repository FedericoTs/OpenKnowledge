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
) -> GroundingReport:
    """Grade ``answer_text`` against the chunks it was allowed to see."""
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

    if require_citations and not cited:
        reasons.append("answer cites no sources")

    unknown = tuple(cid for cid in cited if cid not in known_ids)
    if unknown:
        reasons.append(f"cites sources not retrieved for this question: {', '.join(unknown)}")

    # Grade against the cited chunks where the answer names them; otherwise
    # against everything retrieved, so an uncited answer is still measured.
    cited_chunks = [c for c in retrieved if c.document_id in cited or c.chunk_id in cited]
    evidence = cited_chunks or retrieved
    evidence_text = " ".join(c.text for c in evidence)
    evidence_tokens = set(tokenize(evidence_text))
    evidence_numbers = {_normalise_number(n) for n in _NUMBER_RE.findall(evidence_text)}

    answer_numbers = tuple(
        dict.fromkeys(
            n
            for n in _NUMBER_RE.findall(_CITATION_RE.sub("", answer_text))
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

    if support_ratio < min_support_ratio:
        reasons.append(
            f"only {support_ratio:.0%} of the answer's content words appear in the sources "
            f"(need {min_support_ratio:.0%})"
        )

    return GroundingReport(
        passed=not reasons,
        cited_ids=cited,
        unknown_ids=unknown,
        unsupported_numbers=answer_numbers,
        support_ratio=round(support_ratio, 4),
        reasons=tuple(reasons),
    )
