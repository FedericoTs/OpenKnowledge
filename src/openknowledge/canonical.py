"""Query canonicalisation.

Two people asking the same thing should hit the same cache entry. "What's the
parental leave policy?" and "what is the parental leave policy" must normalise
to one string, or the cache never fills up and the whole cost argument
collapses.

The hard part is knowing when to stop. Aggressive normalisation - stemming,
stopword removal - raises the hit rate and destroys correctness, because in
policy and procedure questions the throwaway words carry the meaning:

    "which expenses are reimbursable"
    "which expenses are not reimbursable"

Drop "not" as a stopword and those two collapse into one cache entry, and the
bot confidently gives half the company the opposite of the right answer. That is
a worse failure than an expensive one.

So the rules here are deliberately conservative. We only remove things that
cannot change what was asked: unicode presentation differences, casing,
whitespace, terminal punctuation, and a closed list of leading pleasantries.
Anything that could be load-bearing stays. Near-miss phrasings are the semantic
cache's job (``cascade`` tier L1), where a similarity threshold and a citation
check can catch a bad match - a hash cannot.
"""

from __future__ import annotations

import re
import unicodedata

#: Leading pleasantries that carry no question content. Matched only at the
#: start of the query, only as whole words, and only before the real question.
#: Keep this list closed and boring - every addition is a chance to eat meaning.
_LEADING_FILLERS: tuple[str, ...] = (
    "hi",
    "hello",
    "hey",
    "good morning",
    "good afternoon",
    "please",
    "quick question",
    "just wondering",
    "i was wondering",
    "i'd like to know",
    "i would like to know",
    "can you tell me",
    "could you tell me",
    "can you please tell me",
    "do you know",
)

#: Words we must never strip. Documented so nobody "optimises" the hit rate by
#: adding a stopword list later without reading why that is a correctness bug.
NEVER_STRIP: frozenset[str] = frozenset(
    {
        "no",
        "not",
        "never",
        "cannot",
        "can't",
        "don't",
        "doesn't",
        "isn't",
        "aren't",
        "won't",
        "shouldn't",
        "must",
        "may",
        "except",
        "unless",
        "before",
        "after",
        "without",
    }
)

_FILLER_RE = re.compile(
    rf"^(?:(?:{'|'.join(re.escape(f) for f in _LEADING_FILLERS)})\b[\s,]*)+",
    re.IGNORECASE,
)
_WHITESPACE_RE = re.compile(r"\s+")
_TRAILING_PUNCT_RE = re.compile(r"[?!.,;:\s]+$")

#: C0/C1 controls (minus the whitespace ones we fold separately), zero-width
#: marks, bidi overrides, and the BOM. These are invisible in a chat client but
#: change the bytes, so a paste from Word would otherwise miss the cache.
_CONTROL_RE = re.compile(
    "[\\u0000-\\u0008\\u000b\\u000c\\u000e-\\u001f\\u007f-\\u009f"
    "\\u200b-\\u200f\\u202a-\\u202e\\u2060\\ufeff]"
)

_PUNCT_FOLD = str.maketrans(
    {
        "‘": "'",  # left single quote
        "’": "'",  # right single quote / apostrophe
        "‚": "'",
        "“": '"',  # left double quote
        "”": '"',  # right double quote
        "–": "-",  # en dash
        "—": "-",  # em dash
        "−": "-",  # minus sign
        " ": " ",  # non-breaking space
        "…": "...",  # ellipsis
    }
)


def canonicalize_query(query: str) -> str:
    """Reduce ``query`` to its cache-key form.

    Idempotent: ``canonicalize_query(canonicalize_query(q)) == canonicalize_query(q)``.

    >>> canonicalize_query("Hi, what's the PARENTAL LEAVE policy?")
    "what's the parental leave policy"
    >>> canonicalize_query("Which expenses are not reimbursable?")
    'which expenses are not reimbursable'
    """
    text = unicodedata.normalize("NFKC", query)
    text = text.translate(_PUNCT_FOLD)
    text = _CONTROL_RE.sub("", text)
    text = text.casefold()
    text = _WHITESPACE_RE.sub(" ", text).strip()

    # Strip pleasantries, then re-strip punctuation they may have left behind
    # ("hi, what is x" -> ", what is x" -> "what is x").
    text = _FILLER_RE.sub("", text).lstrip(" ,:-")
    text = _TRAILING_PUNCT_RE.sub("", text)
    return _WHITESPACE_RE.sub(" ", text).strip()
