"""How much to trust an answer that already passed the grounding gate.

The gate is binary: grounded enough to serve, or not. That is the right shape
for a safety decision and the wrong shape for a reader, because "passed" covers
both an answer that restates one unambiguous sentence and one where the model
had to choose between two figures in the same paragraph and picked one.

Confidence is derived entirely from signals the cascade already computes, so it
costs nothing and adds no model call. Asking the model how sure it is would cost
output tokens to obtain a number that language models are famously bad at, and
would not be reproducible; every input here is deterministic, so the same answer
over the same corpus always carries the same confidence.

The signals, and why each one:

**Support** - what share of the answer's content words appear in the cited text.
Already computed by the gate as `support_ratio`. An answer that closely tracks
its sources is safer than one that paraphrases freely.

**Competing figures** - the retrieved passages contain another number of the same
unit that the answer did not mention. This is the one that matters: a real run
found two cases where a single sentence gave a general rule and then an
exception - "three days per week... two days for client-facing roles" - and the
whole risk sat in choosing between them. When the answer states one of several
candidate figures and ignores the rest, the reader should know a choice was made.

**Contested near-misses** - an open disagreement that was relevant to the
question but scored just under the threshold that would have refused it. That is
precisely the band where the refusal decision was closest, and saying so costs
nothing.

**Single-source answers over a broad match** - the answer cites one document when
several were retrieved and scored comparably. Not damning; worth a small
deduction.

What this is not: a probability. It is an ordering, and the only claim made for
it is that lower scores deserve a second look. Whether it actually separates
right answers from wrong ones on a given corpus is an empirical question, and
`openknowledge eval` reports it per case so it can be checked rather than
believed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .base import Chunk
from .grounding import GroundingReport

#: Numbers with an optional unit, matched in the answer and in the sources.
_FIGURE = re.compile(
    r"(\d[\d.,]*)\s*(%|percent|hours?|days?|weeks?|months?|years?)?",
    re.IGNORECASE,
)
_CITATION = re.compile(r"\[([a-z0-9][a-z0-9_\-/]*)\]", re.IGNORECASE)

#: Confidence at or below this reads as "check this one".
LOW = 0.55
#: Above this the answer tracked its sources closely and had nothing to choose.
HIGH = 0.8


@dataclass(frozen=True, slots=True)
class Confidence:
    """A score in [0, 1], and the reasons it is not 1."""

    score: float
    reasons: tuple[str, ...] = field(default_factory=tuple)

    @property
    def label(self) -> str:
        if self.score >= HIGH:
            return "high"
        return "low" if self.score <= LOW else "medium"

    def describe(self) -> str:
        if not self.reasons:
            return f"{self.label} ({self.score:.0%})"
        return f"{self.label} ({self.score:.0%}) - {'; '.join(self.reasons)}"


def _figures(text: str) -> set[tuple[str, str]]:
    """Numbers with their unit, normalised for comparison."""
    found: set[tuple[str, str]] = set()
    for raw, unit in _FIGURE.findall(text):
        value = raw.rstrip(".,").replace(",", "")
        if not value:
            continue
        unit = (unit or "").lower().rstrip("s")
        found.add((value, unit))
    return found


def assess(
    answer_text: str,
    *,
    grounding: GroundingReport,
    retrieved: list[Chunk],
    near_miss_conflicts: int = 0,
) -> Confidence:
    """Score an answer that has already passed the gate."""
    score = 1.0
    reasons: list[str] = []

    # Support is the base. The gate's own floor is 0.45, so an answer scraping
    # past it starts well below one that restates its sources closely.
    support = max(0.0, min(1.0, grounding.support_ratio))
    if support < 0.75:
        penalty = (0.75 - support) * 0.8
        score -= penalty
        reasons.append(f"{support:.0%} of the answer's wording appears in the sources")

    # The lever this exists for: the sources offered more than one figure of the
    # same unit and the answer used one of them.
    answered = _figures(_CITATION.sub("", answer_text))
    if answered:
        available = _figures(" ".join(c.text for c in retrieved))
        units_used = {unit for _, unit in answered if unit}
        competing = {
            (value, unit)
            for value, unit in available
            if unit in units_used and (value, unit) not in answered
        }
        if competing:
            score -= min(0.25, 0.08 * len(competing))
            shown = ", ".join(f"{v} {u}" for v, u in sorted(competing)[:3])
            reasons.append(f"the sources also give {shown} for the same kind of quantity")

    if near_miss_conflicts:
        score -= min(0.2, 0.1 * near_miss_conflicts)
        reasons.append(
            f"{near_miss_conflicts} open disagreement(s) were close to blocking this question"
        )

    documents = {c.document_id for c in retrieved}
    if len(grounding.cited_ids) == 1 and len(documents) >= 4:
        score -= 0.05
        reasons.append(f"answered from one of {len(documents)} documents that matched")

    return Confidence(score=max(0.0, round(score, 3)), reasons=tuple(reasons))
