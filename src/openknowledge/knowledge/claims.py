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
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ..retrieval.base import Document, tokenize
from .salience import salience_from
from .scope import comparable, counterparties

#: An unscoped document compares with everything, which is what an absent
#: entry means: no scope was established, and an unestablished scope must
#: never be the reason a contradiction goes unreported.
EVERYTHING: frozenset[str] = frozenset()

if TYPE_CHECKING:  # pragma: no cover - import cycle at runtime
    from .deontic import DeonticClaim

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
    """Two documents asserting different things about what looks like one subject.

    Holds either kind of claim. A numeric claim and a deontic one are never
    paired - they carry different units, and a figure cannot contradict a
    permission.
    """

    left: Claim | DeonticClaim
    right: Claim | DeonticClaim
    overlap: float
    #: "numeric" for a moved figure, "deontic" for a changed permission.
    kind: str = "numeric"

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


def compare_numeric_claims(
    documents: list[Document],
    *,
    min_overlap: float = 0.34,
    min_shared_words: int = 3,
    cache: ClaimCache | None = None,
    scopes: Mapping[str, frozenset[str]] | None = None,
) -> tuple[list[Conflict], dict[tuple[str, str], int]]:
    """Compare every pair of documents on the figures they both assert.

    Returns the disagreements and, per document pair, how many figures they
    agreed on. The agreements are not noise to be discarded: they are what
    distinguishes two documents that contradict each other from two documents
    that are variants of one another, which is a distinction no per-claim
    threshold can make. See `variants.py`.

    Only compares claims from *different* documents. A single document giving
    two figures is usually a rule with a condition ("EUR 500 for travel, EUR 45
    for meals"), not a contradiction, and flagging those would train the admin
    to dismiss the whole feature.

    ``min_shared_words`` guards against short contexts scoring a high Jaccard on
    one incidental word - a real match should share several topic words. It stays
    a plain count: how *much* each shared word is worth is what the score
    measures, and applying the weighting twice punished a small corpus where a
    single genuinely shared subject word is all there is to go on.
    """
    scopes = counterparties(documents) if scopes is None else scopes
    pull = cache.numeric if cache is not None else extract_claims
    by_doc: dict[str, list[Claim]] = {}
    for doc in documents:
        by_doc.setdefault(doc.document_id, []).extend(pull(doc))

    weights = salience_from(claim.context for claims in by_doc.values() for claim in claims)

    conflicts: list[Conflict] = []
    seen: set[str] = set()
    doc_ids = sorted(by_doc)

    agreements: dict[tuple[str, str], int] = {}

    for i, left_id in enumerate(doc_ids):
        for right_id in doc_ids[i + 1 :]:
            # Two agreements with different companies are not disagreeing;
            # they are about different worlds. See `scope.py`.
            if not comparable(scopes.get(left_id, EVERYTHING), scopes.get(right_id, EVERYTHING)):
                continue
            for left in by_doc[left_id]:
                for right in by_doc[right_id]:
                    if left.unit != right.unit:
                        continue
                    if len(left.context & right.context) < min_shared_words:
                        continue
                    score = weights.jaccard(left.context, right.context)
                    if score < min_overlap:
                        continue
                    if left.value == right.value:
                        # Same subject, same figure. Worth counting: a pair of
                        # documents that agrees forty times and differs twice is
                        # telling a very different story from one that differs
                        # forty times, and only the second reading is available
                        # if agreements are discarded here. See `variants.py`.
                        agreements[left_id, right_id] = agreements.get((left_id, right_id), 0) + 1
                        continue
                    conflict = Conflict(
                        left=left, right=right, overlap=round(score, 4), kind="numeric"
                    )
                    if conflict.key in seen:
                        continue
                    seen.add(conflict.key)
                    conflicts.append(conflict)

    conflicts.sort(key=lambda c: (-c.overlap, c.key))
    return conflicts, agreements


def find_numeric_conflicts(
    documents: list[Document],
    *,
    min_overlap: float = 0.34,
    min_shared_words: int = 3,
) -> list[Conflict]:
    """Just the disagreements, for callers that do not need the agreement counts."""
    conflicts, _ = compare_numeric_claims(
        documents, min_overlap=min_overlap, min_shared_words=min_shared_words
    )
    return conflicts


class ClaimCache:
    """Claims already pulled out of a document, keyed by that document's text.

    Extraction is a pure function of ``doc.text``: the same words yield the
    same claims, every time. Comparison is not - it is a property of the whole
    corpus, and a disagreement between two untouched documents is still a
    disagreement - so this caches only the extracting, and every pair is still
    compared on every scan. What is found does not change; what is re-derived
    does.

    It earns its place on the clock rather than the invoice. Conflict detection
    costs no money, which is why it re-ran over everything, but at 400 markdown
    documents it was 57% of a rebuild and claim extraction alone was 48% - and
    a rebuild happens on every upload, every delete and every access rule an
    admin changes, in the request that is waiting for it.

    Keyed by ``(document_id, content_hash)`` rather than by the hash alone: two
    documents can carry identical text, and their claims cite where they came
    from.
    """

    __slots__ = ("_comparison", "_deontic", "_numeric")

    def __init__(self) -> None:
        self._numeric: dict[tuple[str, str], tuple[Claim, ...]] = {}
        self._deontic: dict[tuple[str, str], tuple[object, ...]] = {}
        #: The last comparison, and the corpus it was made over. One entry, not
        #: a dictionary: a rebuild either faces the corpus the last one faced or
        #: it does not, and keeping every corpus ever compared would be the leak
        #: `keep_only` exists to prevent.
        self._comparison: tuple[object, object] | None = None

    @staticmethod
    def _key(doc: Document) -> tuple[str, str]:
        return (doc.document_id, doc.content_hash)

    def numeric(self, doc: Document) -> list[Claim]:
        key = self._key(doc)
        cached = self._numeric.get(key)
        if cached is None:
            cached = tuple(extract_claims(doc))
            self._numeric[key] = cached
        return list(cached)

    def deontic(self, doc: Document, extract: Callable[[Document], list[Any]]) -> list[Any]:
        key = self._key(doc)
        cached = self._deontic.get(key)
        if cached is None:
            cached = tuple(extract(doc))
            self._deontic[key] = cached
        return list(cached)

    def comparison(self, fingerprint: object, compute: Callable[[], Any]) -> Any:
        """The comparison over this exact corpus, made once.

        Extraction is a pure function of one document; comparison is a pure
        function of all of them, which is why it was re-derived on every scan
        and why that was 98% of the clock. But "all of them" is a thing that can
        be recognised: the same documents with the same text, compared with the
        same thresholds, disagree in exactly the same places.

        This does not make a rebuild after an upload cheaper - the corpus really
        did change, and the salience weights every score depends on are computed
        over the whole of it, so nothing from before still applies. What it
        makes free is a rebuild where no document changed at all, which is what
        every access-rule change is: an admin waited 32 seconds at 500 documents
        for an answer that could not have moved.
        """
        if self._comparison is not None and self._comparison[0] == fingerprint:
            return self._comparison[1]
        result = compute()
        self._comparison = (fingerprint, result)
        return result

    def keep_only(self, documents: Iterable[Document]) -> None:
        """Forget documents that are no longer in the corpus.

        Without this the cache is a slow leak keyed by everything that was ever
        indexed - and an install that re-uploads a document under a new name
        every week would keep both forever.
        """
        live = {self._key(doc) for doc in documents}
        for store in (self._numeric, self._deontic):
            for key in [k for k in store if k not in live]:
                del store[key]
        # The remembered comparison is deliberately left alone. It is a single
        # entry, so it is not the leak this method exists to stop, and it is
        # keyed by every document's id and content hash - a corpus that lost a
        # document has a different key and misses on its own. Clearing it here
        # was the first version, and it silently disabled the whole cache: the
        # engine calls this on every rebuild, which is precisely the moment the
        # comparison is meant to survive. The tests passed; the measurement is
        # what noticed, because the warm column never moved.


def compare_documents(
    documents: list[Document],
    *,
    min_overlap: float = 0.34,
    min_shared_words: int = 3,
    deontic_strictness: float = 1.0,
    cache: ClaimCache | None = None,
) -> tuple[list[Conflict], dict[tuple[str, str], int]]:
    """Every disagreement between documents that we can find without a model.

    Two detectors with different failure modes. The numeric one catches a moved
    figure - a threshold, a deadline, an allowance - and is nearly free of false
    positives because a number anchors the claim. The deontic one catches a
    changed permission - eligible becoming excluded, required becoming optional -
    and needs more care, because prose has nothing to anchor on but its own
    words.

    Between them they cover the two classes that cause real harm. What they miss
    goes to the FAQ cross-check, and then to re-verification.
    """
    from .deontic import conflicts_between, extract_deontic_claims

    cache = cache if cache is not None else ClaimCache()
    # The same documents with the same text, compared with the same thresholds,
    # disagree in exactly the same places. An upload changes the corpus and gets
    # no benefit from this; an access-rule change does not, and stops paying for
    # a comparison whose answer cannot have moved.
    fingerprint = (
        frozenset((doc.document_id, doc.content_hash) for doc in documents),
        min_overlap,
        min_shared_words,
        deontic_strictness,
    )

    def compare() -> tuple[list[Conflict], dict[tuple[str, str], int]]:
        return _compare_documents(
            documents,
            min_overlap=min_overlap,
            min_shared_words=min_shared_words,
            deontic_strictness=deontic_strictness,
            cache=cache,
            conflicts_between=conflicts_between,
            extract_deontic_claims=extract_deontic_claims,
        )

    conflicts, agreements = cache.comparison(fingerprint, compare)
    # Copies, because a caller that sorts or appends must not edit what the next
    # rebuild will be handed.
    return list(conflicts), dict(agreements)


def _compare_documents(
    documents: list[Document],
    *,
    min_overlap: float,
    min_shared_words: int,
    deontic_strictness: float,
    cache: ClaimCache,
    conflicts_between: Any,
    extract_deontic_claims: Any,
) -> tuple[list[Conflict], dict[tuple[str, str], int]]:
    """The comparison itself, with nothing remembered."""
    # Worked out once for the corpus and shared by both detectors: which party
    # is *yours* is a property of the folder, not of a pair.
    scopes = counterparties(documents)
    conflicts, agreements = compare_numeric_claims(
        documents,
        min_overlap=min_overlap,
        min_shared_words=min_shared_words,
        cache=cache,
        scopes=scopes,
    )

    by_doc = {doc.document_id: cache.deontic(doc, extract_deontic_claims) for doc in documents}
    doc_ids = sorted(by_doc)
    seen = {c.key for c in conflicts}

    # Built once over the whole corpus, not per pair: how ordinary a word is is a
    # property of the folder, and computing it from two documents at a time would
    # make a claim's weight depend on which pair it happened to be scored in.
    weights = salience_from(claim.context for claims in by_doc.values() for claim in claims)

    for i, left_id in enumerate(doc_ids):
        for right_id in doc_ids[i + 1 :]:
            if not comparable(scopes.get(left_id, EVERYTHING), scopes.get(right_id, EVERYTHING)):
                continue
            for left, right, score in conflicts_between(
                by_doc[left_id],
                by_doc[right_id],
                strictness=deontic_strictness,
                weights=weights,
            ):
                conflict = Conflict(left=left, right=right, overlap=score, kind="deontic")
                if conflict.key in seen:
                    continue
                seen.add(conflict.key)
                conflicts.append(conflict)

    conflicts.sort(key=lambda c: (-c.overlap, c.key))
    return conflicts, agreements


def find_conflicts(
    documents: list[Document],
    *,
    min_overlap: float = 0.34,
    min_shared_words: int = 3,
    deontic_strictness: float = 1.0,
) -> list[Conflict]:
    """Just the disagreements, for callers that do not group them by document pair."""
    conflicts, _ = compare_documents(
        documents,
        min_overlap=min_overlap,
        min_shared_words=min_shared_words,
        deontic_strictness=deontic_strictness,
    )
    return conflicts
