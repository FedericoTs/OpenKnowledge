"""Numeric claim extraction and conflict detection - no model required.

Internal policy is mostly numbers. Thresholds, deadlines, allowances, notice
periods. When two documents disagree, it is almost always about one of those,
and that is also the disagreement that does real damage: an employee told the
approval limit is EUR 1,000 when the current policy says EUR 500 will submit an
expense they are not entitled to, with a citation attached.

Catching that class does not need a language model. A number carries a value, a
unit, and the words around it, and two claims conflict when the first two differ
while the third matches. That is a regex and a set intersection, which means it
runs on every upload for free, deterministically, and can be unit-tested - none
of which is true of asking a model "do these documents contradict each other".

What this deliberately does not attempt: prose contradictions ("contractors are
eligible" vs "contractors are excluded"). Those need the FAQ re-verification
path in ``reverify.py``, which is targeted and therefore still cheap. This
module is the free first pass, not the whole answer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..retrieval.base import Document, tokenize

#: Words that carry no topical signal, so they should not make two unrelated
#: claims look similar. Scoring aid only - dropping them cannot change an
#: answer, only whether we raise a flag for a human to look at.
_STOPWORDS = frozenset(
    [
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "if",
        "then",
        "than",
        "that",
        "this",
        "these",
        "those",
        "of",
        "to",
        "in",
        "on",
        "at",
        "by",
        "for",
        "with",
        "from",
        "as",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "do",
        "does",
        "did",
        "have",
        "has",
        "had",
        "will",
        "would",
        "can",
        "could",
        "should",
        "may",
        "might",
        "must",
        "it",
        "its",
        "you",
        "your",
        "we",
        "our",
        "they",
        "their",
        "any",
        "per",
        "up",
        "no",
        "not",
        "more",
        "less",
        "over",
        "under",
        "above",
        "below",
        "within",
        "each",
    ]  # noqa: SIM905 - a 60-item list literal here is far less readable
)

#: Currency codes and symbols, normalised to an ISO-ish lowercase code.
_CURRENCIES = {
    "eur": "eur",
    "€": "eur",
    "euro": "eur",
    "euros": "eur",
    "usd": "usd",
    "$": "usd",
    "dollar": "usd",
    "dollars": "usd",
    "gbp": "gbp",
    "£": "gbp",
    "pound": "gbp",
    "pounds": "gbp",
    "chf": "chf",
}

#: Time and count units, normalised to a plural form so "1 day" and "30 days"
#: compare as the same unit.
_UNITS = {
    "day": "days",
    "days": "days",
    "week": "weeks",
    "weeks": "weeks",
    "month": "months",
    "months": "months",
    "year": "years",
    "years": "years",
    "hour": "hours",
    "hours": "hours",
    "percent": "percent",
    "%": "percent",
}

_NUMBER = r"\d[\d.,]*"
_CURRENCY_BEFORE = re.compile(
    rf"(?P<cur>EUR|USD|GBP|CHF|€|\$|£)\s?(?P<num>{_NUMBER})", re.IGNORECASE
)
_CURRENCY_AFTER = re.compile(
    rf"(?P<num>{_NUMBER})\s?(?P<cur>EUR|USD|GBP|CHF|€|\$|£|euros?|dollars?|pounds?)",
    re.IGNORECASE,
)
_UNIT_AFTER = re.compile(
    rf"(?P<num>{_NUMBER})\s*(?P<unit>%|percent|business\s+days?|days?|weeks?|months?|"
    r"years?|hours?)",
    re.IGNORECASE,
)
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")

#: How many content words either side of a number form its context.
_CONTEXT_WINDOW = 8


@dataclass(frozen=True, slots=True)
class Claim:
    """A number asserted by a document, with enough context to compare it."""

    document_id: str
    document_title: str
    raw: str
    value: float
    unit: str
    context: frozenset[str]
    sentence: str

    @property
    def display(self) -> str:
        return self.raw


@dataclass(frozen=True, slots=True)
class Conflict:
    """Two documents asserting different values for what looks like one thing."""

    left: Claim
    right: Claim
    overlap: float

    @property
    def key(self) -> str:
        """Stable identifier, independent of which claim was found first."""
        ends = sorted(
            [
                f"{self.left.document_id}:{self.left.raw}",
                f"{self.right.document_id}:{self.right.raw}",
            ]
        )
        return f"{ends[0]}|{ends[1]}"

    def describe(self) -> str:
        return (
            f"[{self.left.document_id}] says {self.left.raw} "
            f"but [{self.right.document_id}] says {self.right.raw} "
            f"- {self.left.sentence.strip()[:120]!r} vs {self.right.sentence.strip()[:120]!r}"
        )


def _parse_number(raw: str) -> float | None:
    """Parse a figure as written in a document.

    Handles thousands separators in both conventions, so "1,200" and "1.200"
    both read as 1200 while "1.5" stays 1.5. Getting this wrong would invent
    conflicts between a document and itself.
    """
    text = raw.strip().rstrip(".,")
    if not text:
        return None
    if "," in text and "." in text:
        # Whichever separator comes last is the decimal point.
        text = (
            text.replace(".", "").replace(",", ".")
            if text.rindex(",") > text.rindex(".")
            else text.replace(",", "")
        )
    elif "," in text:
        left, _, right = text.rpartition(",")
        text = f"{left.replace(',', '')}.{right}" if len(right) in (1, 2) else text.replace(",", "")
    elif text.count(".") == 1:
        left, _, right = text.partition(".")
        if len(right) == 3 and len(left) <= 3:
            text = left + right  # "1.200" is twelve hundred, not 1.2
    try:
        return float(text)
    except ValueError:
        return None


def _context_words(sentence: str, start: int, end: int) -> frozenset[str]:
    before = [w for w in tokenize(sentence[:start]) if w not in _STOPWORDS]
    after = [w for w in tokenize(sentence[end:]) if w not in _STOPWORDS]
    return frozenset(before[-_CONTEXT_WINDOW:] + after[:_CONTEXT_WINDOW])


def extract_claims(doc: Document) -> list[Claim]:
    """Pull every numeric claim out of a document."""
    claims: list[Claim] = []
    for sentence in _SENTENCE_SPLIT.split(doc.text):
        if not sentence.strip():
            continue
        spans: list[tuple[int, int, str, float, str]] = []

        for pattern, unit_group in (
            (_CURRENCY_BEFORE, "cur"),
            (_CURRENCY_AFTER, "cur"),
            (_UNIT_AFTER, "unit"),
        ):
            for match in pattern.finditer(sentence):
                value = _parse_number(match.group("num"))
                if value is None:
                    continue
                token = match.group(unit_group).lower().strip()
                token = re.sub(r"\s+", " ", token)
                if unit_group == "cur":
                    unit = _CURRENCIES.get(token, token)
                else:
                    unit = _UNITS.get(token.removeprefix("business ").strip(), token)
                spans.append((match.start(), match.end(), match.group(0), value, unit))

        # A currency match and a unit match can overlap ("EUR 45 per day");
        # keep the longest span at each position so one figure is one claim.
        spans.sort(key=lambda s: (s[0], -(s[1] - s[0])))
        taken: list[tuple[int, int]] = []
        for start, end, raw, value, unit in spans:
            if any(start < t_end and end > t_start for t_start, t_end in taken):
                continue
            taken.append((start, end))
            claims.append(
                Claim(
                    document_id=doc.document_id,
                    document_title=doc.title,
                    # Trailing sentence punctuation gets caught by the number
                    # pattern; it is not part of the figure.
                    raw=raw.strip().rstrip(".,"),
                    value=value,
                    unit=unit,
                    context=_context_words(sentence, start, end),
                    sentence=sentence.strip(),
                )
            )
    return claims


def _overlap(a: frozenset[str], b: frozenset[str]) -> float:
    """Jaccard similarity of two context windows."""
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def find_conflicts(
    documents: list[Document],
    *,
    min_overlap: float = 0.34,
    min_shared_words: int = 3,
) -> list[Conflict]:
    """Find pairs of claims that look like the same fact with different numbers.

    Only compares claims from *different* documents. A single document giving
    two figures is usually a rule with a condition ("EUR 500 for travel, EUR 45
    for meals"), not a contradiction, and flagging those would train the admin
    to dismiss the whole feature.

    ``min_shared_words`` guards against short contexts scoring a high Jaccard on
    one incidental word - a real match should share several topic words.
    """
    by_doc: dict[str, list[Claim]] = {}
    for doc in documents:
        by_doc.setdefault(doc.document_id, []).extend(extract_claims(doc))

    conflicts: list[Conflict] = []
    seen: set[str] = set()
    doc_ids = sorted(by_doc)

    for i, left_id in enumerate(doc_ids):
        for right_id in doc_ids[i + 1 :]:
            for left in by_doc[left_id]:
                for right in by_doc[right_id]:
                    if left.unit != right.unit or left.value == right.value:
                        continue
                    shared = left.context & right.context
                    if len(shared) < min_shared_words:
                        continue
                    score = _overlap(left.context, right.context)
                    if score < min_overlap:
                        continue
                    conflict = Conflict(left=left, right=right, overlap=round(score, 4))
                    if conflict.key in seen:
                        continue
                    seen.add(conflict.key)
                    conflicts.append(conflict)

    conflicts.sort(key=lambda c: (-c.overlap, c.key))
    return conflicts
