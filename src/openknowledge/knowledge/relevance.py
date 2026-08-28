"""Deciding whether an open conflict applies to the question being asked.

A conflict is about one specific claim, not about a whole document. Blocking
every question that touches the expenses policy because two of its figures
disagree would make the feature intolerable, and an admin would switch it off.
So relevance is judged against the *contested claim's* context words rather than
against document identity.

The measure is **how much of the contested claim the question covers** - shared
words over the size of the claim's context. Jaccard is wrong here because
questions are short and claim contexts are not, so a genuine match would score as
a miss purely on the length difference.

An overlap coefficient - dividing by the smaller of the two - was the first
attempt and is worse than it looks. It divides by the question's length whenever
the question is the shorter side, so **adding words to a question lowers its
score**, and a contested claim can be answered simply by asking about it at
greater length. A live run found exactly that: "what is the approval threshold
for travel expenses" was refused as contested while "above what amount do I need
approval before booking travel" - the same question - was answered with one of
the two disputed figures. Dividing by the claim's context instead makes the score
independent of how the question was phrased around the subject, which is the same
property canonicalisation gives the cache key.

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

#: Fraction of the contested claim's context the question must cover.
#:
#: Lower than it looks, because the denominator changed: it is now the claim's
#: full context rather than whichever side was shorter, so the same question
#: scores lower than it used to and the threshold moves with it. Measured on the
#: Aveline corpus, 0.3 catches every phrasing of a genuinely contested question
#: that shares vocabulary with the claim, and raises no false alarm across eight
#: unrelated questions over the same conflicts.
DEFAULT_MIN_OVERLAP = 0.3

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
    """Shared words over the smaller side.

    Kept because it is the right measure where both sides are the same kind of
    thing. It is **not** used to decide whether a question is contested - see
    :func:`claim_coverage` and the module docstring for why.
    """
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


def claim_coverage(question: frozenset[str], context: frozenset[str]) -> float:
    """How much of a contested claim's subject this question covers.

    Deliberately independent of the question's length: extra words the claim does
    not mention neither raise nor lower the score. A safety gate that a wordier
    question slips past is not a gate.
    """
    if not question or not context:
        return 0.0
    return len(question & context) / len(context)


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
        # A "versions" conflict is a duplicated document pair - the decision
        # it needs is which copy stands, owed once in /manage, not a
        # per-question contradiction. Gating answers on it turned one stale
        # archive copy into a refusal of the whole expenses domain.
        if conflict.kind == "versions":
            continue
        context = frozenset(_fold(w) for w in conflict.context)
        shared = words & context
        if len(shared) < min_shared:
            continue
        score = claim_coverage(words, context)
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
