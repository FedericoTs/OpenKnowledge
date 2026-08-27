"""Detecting contradictions in policy prose, without a model.

Numeric conflicts catch a moved threshold. They are blind to the other class
that does real damage:

    "Contractors are eligible for parental leave."
    "Contractors are excluded from parental leave."

Nothing numeric changed, and an employee acting on the wrong one has a problem.

The trick is that policy prose is *deontic*: almost every internal rule says
that something must happen, may happen, or must not happen. That is a small,
closed vocabulary - must, shall, required, may, permitted, eligible, prohibited,
excluded - and it can be extracted with the same shape of machinery as a number:
a marker, a polarity, and the words around it. A contradiction is then "the same
subject under a different force", which is a set comparison rather than an
inference.

This will not catch every prose disagreement. It is not meant to - it is the
free pass that runs on every re-index and catches the expensive cases. What it
misses falls to the FAQ cross-check, which is also free, and then to
re-verification, which is not.

Precision matters more than recall here. A missed contradiction is caught later
by the other passes; a false one blocks an answerable question, and an admin who
sees three bogus flags stops reading the fourth. So the thresholds are stricter
than the numeric ones - a number anchors a claim to a specific value, while
prose has nothing to anchor on but its own words.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum

from ..retrieval.base import Document, tokenize
from .claims import _SENTENCE_SPLIT, _STOPWORDS

#: Content words either side of a marker that form its context. Wider than the
#: numeric window: a number is its own anchor, a modal verb is not.
_CONTEXT_WINDOW = 10


class Force(StrEnum):
    """What a rule does to the behaviour it describes."""

    MANDATORY = "mandatory"  # must, shall, is required
    PERMITTED = "permitted"  # may, can, is eligible, is reimbursable
    FORBIDDEN = "forbidden"  # must not, is prohibited, is excluded

    @property
    def label(self) -> str:
        return {
            Force.MANDATORY: "required",
            Force.PERMITTED: "allowed",
            Force.FORBIDDEN: "not allowed",
        }[self]


#: Ordered: negated forms must be tested before the plain ones, or "must not"
#: reads as "must". Each pattern maps a phrase onto the force it asserts.
#:
#: "not required" is PERMITTED rather than FORBIDDEN on purpose - saying a step
#: is not required makes it optional, not prohibited. Getting that backwards
#: would invent disagreements between documents that agree.
_PATTERNS: tuple[tuple[re.Pattern[str], Force], ...] = tuple(
    (re.compile(pattern, re.IGNORECASE), force)
    for pattern, force in [
        # --- explicitly forbidden ---
        (r"\bmust\s+not\b", Force.FORBIDDEN),
        (r"\bshall\s+not\b", Force.FORBIDDEN),
        (r"\bmay\s+not\b", Force.FORBIDDEN),
        (r"\bcan\s?not\b|\bcan't\b", Force.FORBIDDEN),
        (r"\bis\s+not\s+(?:permitted|allowed|eligible|reimbursable|available)\b", Force.FORBIDDEN),
        (r"\bare\s+not\s+(?:permitted|allowed|eligible|reimbursable|available)\b", Force.FORBIDDEN),
        (r"\bnot\s+(?:reimbursable|eligible|permitted|allowed|entitled)\b", Force.FORBIDDEN),
        (r"\b(?:is|are)\s+(?:prohibited|forbidden|excluded|ineligible|barred)\b", Force.FORBIDDEN),
        (r"\b(?:prohibited|forbidden|ineligible)\b", Force.FORBIDDEN),
        (r"\bexcluded\s+from\b", Force.FORBIDDEN),
        (r"\bunder\s+no\s+circumstances\b", Force.FORBIDDEN),
        (r"\bnever\b", Force.FORBIDDEN),
        # --- explicitly optional (a negated obligation) ---
        (r"\b(?:is|are)\s+not\s+(?:required|mandatory|obliged)\b", Force.PERMITTED),
        (
            r"\bno\s+(?:approval|authorisation|authorization|sign-?off)\s+is\s+required\b",
            Force.PERMITTED,
        ),
        (r"\bwithout\s+(?:prior\s+)?(?:approval|authorisation|authorization)\b", Force.PERMITTED),
        (r"\bdo(?:es)?\s+not\s+(?:require|need)\b", Force.PERMITTED),
        # --- mandatory ---
        (r"\bmust\b", Force.MANDATORY),
        (r"\bshall\b", Force.MANDATORY),
        (r"\b(?:is|are)\s+required\b", Force.MANDATORY),
        (r"\brequires?\b", Force.MANDATORY),
        (r"\b(?:is|are)\s+mandatory\b", Force.MANDATORY),
        (r"\bhas\s+to\b|\bhave\s+to\b", Force.MANDATORY),
        (r"\bonly\s+.{0,30}?\s+may\b", Force.MANDATORY),
        # --- permitted ---
        (r"\bmay\b", Force.PERMITTED),
        (r"\b(?:is|are)\s+(?:eligible|entitled|permitted|allowed|reimbursable)\b", Force.PERMITTED),
        (r"\b(?:eligible|entitled|reimbursable)\b", Force.PERMITTED),
        (r"\bcan\b", Force.PERMITTED),
        (r"\boptional\b", Force.PERMITTED),
    ]
)


#: Predicate families. Two rules can only contradict each other if they are
#: about the same *kind* of thing - a rule about reimbursement cannot contradict
#: a rule about VPN access, however many words they happen to share.
#:
#: Matched by prefix against every token in the sentence, so "reimbursable",
#: "reimbursed" and "reimbursement" all land together. Membership is a set: a
#: sentence about expense claims is about both submission and reimbursement, and
#: it should be comparable with either.
_FAMILIES: dict[str, tuple[str, ...]] = {
    "reimbursement": ("reimburs", "expens", "refund", "allowance", "subsisten", "receipt"),
    "eligibility": ("eligib", "entitl", "qualif", "exclud", "ineligib", "barred", "applies"),
    "approval": ("approv", "authoris", "authoriz", "signoff", "permission", "consent"),
    "access": ("access", "permit", "allow", "grant", "credential", "login"),
    "submission": ("submit", "submiss", "lodge", "notif", "claim", "request"),
    "payment": ("pay", "paid", "salary", "wage", "bonus", "compensat"),
    "leave": ("leave", "holiday", "vacation", "absence", "sabbatical"),
    "confidentiality": ("confidential", "disclos", "share", "secret", "nda"),
}


def predicate_families(sentence: str) -> frozenset[str]:
    """Which kinds of rule this sentence is about."""
    tokens = tokenize(sentence)
    return frozenset(
        family
        for family, prefixes in _FAMILIES.items()
        if any(token.startswith(prefix) for token in tokens for prefix in prefixes)
    )


@dataclass(frozen=True, slots=True)
class DeonticClaim:
    """A rule a document states about what is required, allowed, or forbidden."""

    document_id: str
    document_title: str
    force: Force
    marker: str
    context: frozenset[str]
    sentence: str
    #: The kinds of rule this sentence is about; empty when none matched.
    families: frozenset[str] = frozenset()

    @property
    def raw(self) -> str:
        """How the claim reads in a conflict report."""
        return self.force.label

    @property
    def unit(self) -> str:
        """Kept structurally parallel to a numeric claim's unit.

        A fixed value rather than the force, so that two rules are compared on
        their subject and their forces are compared separately - and so that a
        deontic claim can never be paired against a numeric one.
        """
        return "rule"

    @property
    def value(self) -> str:
        return self.force.value

    @property
    def display(self) -> str:
        return f"{self.force.label} ({self.marker})"


def _context_words(sentence: str, start: int, end: int) -> frozenset[str]:
    before = [w for w in tokenize(sentence[:start]) if w not in _STOPWORDS]
    after = [w for w in tokenize(sentence[end:]) if w not in _STOPWORDS]
    return frozenset(before[-_CONTEXT_WINDOW:] + after[:_CONTEXT_WINDOW])


def extract_deontic_claims(doc: Document) -> list[DeonticClaim]:
    """Pull the rules a document states, one per deontic marker."""
    claims: list[DeonticClaim] = []
    for sentence in _SENTENCE_SPLIT.split(doc.text):
        if not sentence.strip():
            continue

        taken: list[tuple[int, int]] = []
        for pattern, force in _PATTERNS:
            for match in pattern.finditer(sentence):
                # Patterns are ordered strongest-first, so an earlier match at
                # this position already decided the force ("must not" wins over
                # the "must" inside it).
                if any(match.start() < end and match.end() > start for start, end in taken):
                    continue
                taken.append((match.start(), match.end()))
                claims.append(
                    DeonticClaim(
                        document_id=doc.document_id,
                        document_title=doc.title,
                        force=force,
                        marker=match.group(0).strip().lower(),
                        context=_context_words(sentence, match.start(), match.end()),
                        sentence=sentence.strip(),
                        families=predicate_families(sentence),
                    )
                )
    return claims


def _overlap_coefficient(a: frozenset[str], b: frozenset[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


#: Not every pair of differing forces is a contradiction, and the difference is
#: deontic rather than statistical.
#:
#: FORBIDDEN against either other force is a *hard* contradiction: you cannot be
#: both allowed and not allowed to do something. Recognisably the same subject is
#: enough to flag it.
#:
#: MANDATORY against PERMITTED is *soft*. "You must submit within 60 days" and
#: "you may submit online" are both true at once - one is a deadline, the other a
#: channel. Those only contradict when they describe the identical action, so
#: they need near-identical context before we interrupt anyone.
_HARD_OVERLAP = 0.45
_SOFT_OVERLAP = 0.85
_SAME_FAMILY_SHARED = 1


def _is_hard(a: Force, b: Force) -> bool:
    return Force.FORBIDDEN in (a, b)


#: With no family to anchor on, the words have to carry the whole argument, so
#: the bar is much higher.
_NO_FAMILY_OVERLAP = 0.6
_NO_FAMILY_SHARED = 3


def conflicts_between(
    left: list[DeonticClaim],
    right: list[DeonticClaim],
    *,
    strictness: float = 1.0,
) -> list[tuple[DeonticClaim, DeonticClaim, float]]:
    """Pairs of rules about the same subject that assert different forces.

    Two gates, in order. First the claims must be about the same *kind* of rule -
    reimbursement, eligibility, approval - because that is what separates "the
    same thing, contradicted" from "two different rules that share a noun".
    Then they must be recognisably about the same subject.

    Without the family gate, "employees must submit within 60 days" and
    "employees may submit expense claims online" score as a contradiction on
    word overlap alone. They are not one: same subject, different aspect - and
    the soft-pair threshold is what finally rejects them.

    ``strictness`` scales both thresholds for tuning; above 1.0 flags less.
    """
    found: list[tuple[DeonticClaim, DeonticClaim, float]] = []
    for a in left:
        for b in right:
            if a.force is b.force:
                continue

            shares_family = bool(a.families & b.families)
            if a.families and b.families and not shares_family:
                continue  # different kinds of rule cannot contradict each other

            if shares_family:
                base = _HARD_OVERLAP if _is_hard(a.force, b.force) else _SOFT_OVERLAP
                min_overlap = base * strictness
                min_shared = _SAME_FAMILY_SHARED
            else:
                min_overlap = _NO_FAMILY_OVERLAP * strictness
                min_shared = _NO_FAMILY_SHARED

            if len(a.context & b.context) < min_shared:
                continue
            score = _overlap_coefficient(a.context, b.context)
            if score < min_overlap:
                continue
            found.append((a, b, round(score, 4)))

    found.sort(key=lambda triple: -triple[2])
    return found
