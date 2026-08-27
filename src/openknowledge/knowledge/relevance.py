"""Deciding whether an open conflict applies to the question being asked.

A conflict is about one specific claim, not about a whole document. Blocking
every question that touches the expenses policy because two of its figures
disagree would make the feature intolerable, and an admin would switch it off.
So relevance is judged against the *contested claim's* context words rather than
against document identity.

The measure is an overlap coefficient rather than Jaccard: questions are short
and claim contexts are not, so Jaccard would score almost every genuine match as
a miss purely because of the length difference.

Words are compared after light suffix stripping, so "submit an expense claim"
matches a claim written as "expense claims must be submitted". This is the one
place in the project where that is safe: query canonicalisation refuses to stem
because it decides which cached answer you get, whereas here the only outcome is
whether a human is asked to look at something. Getting it wrong in the generous
direction costs a needless review; getting it wrong in the strict direction
means somebody is told EUR 500 while the current policy says EUR 1,000.
"""

from __future__ import annotations

from ..retrieval.base import tokenize
from .claims import _STOPWORDS
from .store import StoredConflict

#: Fraction of the smaller word set that must be shared.
DEFAULT_MIN_OVERLAP = 0.5

#: Below this, a single coincidental word could trigger a refusal.
DEFAULT_MIN_SHARED = 2


#: Longest prefix compared when matching words. Folding to a prefix beats
#: suffix stripping here: a stripper turns "expenses" into "expens" while
#: leaving "expense" alone, so the two stop matching - which is worse than not
#: normalising at all.
_PREFIX = 5

#: Never fold these. Negations and modals decide what a rule means, and a
#: normaliser that mangles them could match a claim to its own opposite.
_NEVER_FOLD = frozenset(
    {"is", "was", "has", "does", "goes", "less", "unless", "yes", "this", "its", "as"}
)


def _fold(word: str) -> str:
    """Reduce a word to a comparable form.

    Strips a plural ending, then truncates, so "expense"/"expenses",
    "submit"/"submitted" and "day"/"days" all land on the same token. Short
    words are left alone - truncating them would collide unrelated terms.
    """
    if word in _NEVER_FOLD or len(word) <= 3:
        return word
    if word.endswith("es") and len(word) > 5:
        word = word[:-2]
    elif word.endswith("s") and not word.endswith("ss"):
        word = word[:-1]
    return word[:_PREFIX]


def _content_words(text: str) -> frozenset[str]:
    return frozenset(_fold(w) for w in tokenize(text) if w not in _STOPWORDS)


def overlap_coefficient(a: frozenset[str], b: frozenset[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


def relevant_conflicts(
    question: str,
    conflicts: list[StoredConflict],
    *,
    min_overlap: float = DEFAULT_MIN_OVERLAP,
    min_shared: int = DEFAULT_MIN_SHARED,
) -> list[StoredConflict]:
    """Open conflicts that bear on this question, most relevant first."""
    words = _content_words(question)
    if not words:
        return []

    scored: list[tuple[float, StoredConflict]] = []
    for conflict in conflicts:
        context = frozenset(_fold(w) for w in conflict.context)
        shared = words & context
        if len(shared) < min_shared:
            continue
        score = overlap_coefficient(words, context)
        if score < min_overlap:
            continue
        scored.append((score, conflict))

    scored.sort(key=lambda pair: (-pair[0], pair[1].key))
    return [conflict for _, conflict in scored]


def describe_for_user(conflicts: list[StoredConflict]) -> str:
    """The message an employee sees when their question is contested.

    It names both figures and both documents, because "ask your admin" without
    saying what is in dispute wastes everyone's time - and because seeing the
    two values is often enough for the reader to know which one applies to them.
    """
    lines = [
        "Your documents disagree on this, so I won't guess:",
    ]
    for conflict in conflicts[:3]:
        lines.append(
            f"  - [{conflict.left_document}] says {conflict.left_raw}, "
            f"[{conflict.right_document}] says {conflict.right_raw}"
        )
    lines.append("Please ask your administrator which one currently applies.")
    return "\n".join(lines)
