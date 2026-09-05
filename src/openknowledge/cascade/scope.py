"""Questions whose answer is a whole document, not a passage of it.

"What are the chapters?", "who are the characters?", "what are all the
priorities?", "summarise the first act." The answer is not a sentence that
retrieval can rank; it is the document, or one section of it, read end to end.
Retrieval hands the model `retrieval_k` passages ranked by word overlap, and a
question like "what terms does the glossary define?" shares no words with the
terms it is asking for. Measured on a 22-chunk glossary: one chunk shown, 26 of
82 terms visible, and raising `k` eight-fold reaches 89%, not 100%
(evals/golden-scope/README.md). The obvious fix is measured and does not work.

Two things here, and a rule about when either may run.

**The rule: two votes, never one.** A question is treated as whole-document
only when it has the *shape* - it asks to list, name all, summarise, or say
what something covers - *and* it has a *target*: retrieval concentrates on one
document, or the question names one. Either alone changes nothing. "How much
is the meal allowance?" has neither and is never touched; "list the vendors"
over a corpus where the hits spread across five documents is left to ordinary
retrieval too. Vocabulary lists have been the recurring hole in the free tiers,
and the second vote is the mitigation: the shape words can be as generous as
they like, because they cannot act alone.

**Structure answers enumerations, free.** A document's parsed blocks already
know its headings, its list items, its glossary's term-definition paragraphs.
"What are the chapters?" is the headings. "What are the priorities?" is the list
under the heading that says priorities. "Who are the persons in the play?" is
the list under the heading that says persons. These are read from the blocks
with no model, for $0, deterministically, and they are complete by construction
in exactly the sense the corpus tier's answers are. An ordinal - "the second
chapter", "priority 2" - indexes the same list, and one past the end is a
refusal that says how many there are, because the alternative is a model
inventing a thirteenth chapter. The ordinal carries its noun: "the fourth act"
indexes the *acts*, not every heading in the play, which is the difference
between refusing and answering "SECOND ACT".

**Assembly serves summaries, within the window.** When the shape is summary and
no structure answers it, the six ranked passages are replaced by the target
document's passages in order - or one section's, when the question names it -
up to what the model's context window can take. The local model runs 8,192
tokens; the provider refuses anything past 90% of that; the system prompt is
~1,070 tokens; about ten passages fit. A document longer than that is read from
the start and the answer's notes say so - an honest partial rather than a
silent one. Map-reduce over windows is the next step and is not built here.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from ..documents.blocks import Block, BlockKind

if TYPE_CHECKING:
    from ..retrieval.base import Chunk, ScoredChunk

_WORDS = re.compile(r"[a-z0-9']+")

#: Verbs and quantifiers that ask for the whole of something.
_ENUMERATE = frozenset(
    [
        "list",
        "listing",
        "all",
        "every",
        "each",
        "name",
        "outline",
        "contents",
        "cover",
        "covers",
        "covering",
        "contain",
        "contains",
        "define",
        "defines",
        "definitions",
        "defined",
        "enumerate",
    ]
)
#: Openers that ask for a set rather than a fact.
_SET_OPENERS = ("what are the", "who are the", "which are the", "what are all", "who are all")
#: What kind of structure answers a noun.
_HEADING_NOUNS = (
    "chapter",
    "chapters",
    "section",
    "sections",
    "act",
    "acts",
    "part",
    "parts",
    "heading",
    "headings",
    "topic",
    "topics",
)
_TERM_NOUNS = ("term", "terms", "definition", "definitions", "glossary")
_PERSON_NOUNS = ("character", "characters", "person", "persons", "people", "cast")
_ITEM_NOUNS = (
    "priority",
    "priorities",
    "step",
    "steps",
    "activity",
    "activities",
    "item",
    "items",
    "point",
    "points",
    "rule",
    "rules",
    "requirement",
    "requirements",
    "action",
    "actions",
    "task",
    "tasks",
)
_NOUNS: Mapping[str, str] = {
    **dict.fromkeys(_HEADING_NOUNS, "headings"),
    **dict.fromkeys(_TERM_NOUNS, "terms"),
    **dict.fromkeys(_PERSON_NOUNS, "persons"),
    **dict.fromkeys(_ITEM_NOUNS, "items"),
}
#: Heading kinds specific enough to filter on. "Section" and "part" are not:
#: a regulation's sections are headed with a § and say neither word.
_SPECIFIC_HEADINGS = {"chapter": "chapter", "chapters": "chapter", "act": "act", "acts": "act"}
#: "What does this document cover?" names no noun; the headings are the answer.
_COVERAGE = frozenset(["cover", "covers", "covering", "contain", "contains", "outline", "contents"])
_SUMMARISE = frozenset(
    [
        "summarise",
        "summarize",
        "summarising",
        "summarizing",
        "summary",
        "overview",
        "recap",
        "gist",
        "synopsis",
    ]
)

_ORDINAL_WORDS = (
    "first",
    "second",
    "third",
    "fourth",
    "fifth",
    "sixth",
    "seventh",
    "eighth",
    "ninth",
    "tenth",
    "eleventh",
    "twelfth",
    "thirteenth",
    "fourteenth",
    "fifteenth",
)
_CARDINAL_WORDS = (
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
)
_ORDINALS = {w: i for i, w in enumerate(_ORDINAL_WORDS, 1)}
_CARDINALS = {w: i for i, w in enumerate(_CARDINAL_WORDS, 1)}
#: The things an ordinal can count. Singular; the pattern allows a plural s.
_COUNTABLE = (
    "chapter",
    "act",
    "section",
    "part",
    "priority",
    "step",
    "item",
    "point",
    "rule",
    "activity",
    "task",
)
#: "the second chapter", "chapter two", "priority 2", "the last act". Two
#: digits at most, with a suffix when they lead: "part 301-10" is a document's
#: name, not its three-hundred-and-first part, and the lookahead refuses a
#: number that continues with a dash or a point.
_ORDINAL = re.compile(
    rf"\b(?:(\d{{1,2}})(?:st|nd|rd|th)|({'|'.join(_ORDINAL_WORDS)}|last))\s+"
    rf"({'|'.join(_COUNTABLE)})s?\b"
    rf"|\b({'|'.join(_COUNTABLE)})\s+(\d{{1,2}}|{'|'.join(_CARDINAL_WORDS)})\b(?![-.\d])"
)

#: A glossary entry opens its paragraph: "Accompanied baggage. Government
#: property ...". Six words or fewer, or it is a sentence.
TERM_PATTERN = re.compile(r"^([A-Z][A-Za-z0-9\-/'’()., ]{1,60}?)\.\s")


@dataclass(frozen=True, slots=True)
class Scope:
    """A whole-document question, resolved to what it wants and from where."""

    kind: Literal["enumerate", "summarise"]
    document_id: str
    #: Which structure answers it: headings, items, persons, terms - or "any"
    #: when the question says "cover" and names nothing.
    wants: str = "any"
    #: "the second chapter" -> 2; "the last act" -> -1. None for the whole list.
    ordinal: int | None = None
    #: What the ordinal counts - "chapter", "act" - so it indexes that list.
    ordinal_noun: str | None = None
    #: A specific kind of heading the question names without an ordinal -
    #: "what are the chapters?" - so a document with no chapters is not
    #: answered with whatever headings it does have. Measured: over a corpus
    #: holding a novel and a play, "what are the chapters?" concentrated on
    #: the play and would have listed its acts.
    heading_noun: str | None = None

    @property
    def noun(self) -> str | None:
        """What the headings must say, if anything."""
        return self.ordinal_noun or self.heading_noun


def recognise_scope(
    question: str,
    hits: Iterable[ScoredChunk],
    *,
    titles: Mapping[str, str] | None = None,
) -> Scope | None:
    """Whether ``question`` asks for a whole document, and which one.

    Both votes are required. The shape vote reads the question; the target vote
    reads where retrieval landed, or which title the question names. ``titles``
    may widen the title search beyond the ranked hits.
    """
    lowered = question.lower()
    words = _WORDS.findall(lowered)
    if not words:
        return None
    vocabulary = set(words)

    ordinal = _ordinal_in(lowered)
    wants = next((_NOUNS[w] for w in words if w in _NOUNS), None)
    if ordinal is not None and wants is None:
        wants = _NOUNS.get(ordinal[1], "items")
    summarise = bool(vocabulary & _SUMMARISE)
    enumerate_ = (
        ordinal is not None
        or (wants is not None and (vocabulary & _ENUMERATE or lowered.startswith(_SET_OPENERS)))
        or (wants is None and bool(vocabulary & _COVERAGE))
    )
    if not summarise and not enumerate_:
        return None

    target = _target(lowered, list(hits), titles or {})
    if target is None:
        return None

    # A summary cue wins: "summarise the first act" wants the act summarised,
    # and the ordinal scopes which section is read rather than what is listed.
    kind: Literal["enumerate", "summarise"] = "summarise" if summarise else "enumerate"
    return Scope(
        kind=kind,
        document_id=target,
        wants=wants or "any",
        ordinal=ordinal[0] if ordinal else None,
        ordinal_noun=ordinal[1] if ordinal else None,
        heading_noun=next((_SPECIFIC_HEADINGS[w] for w in words if w in _SPECIFIC_HEADINGS), None),
    )


def _ordinal_in(lowered: str) -> tuple[int, str] | None:
    found = _ORDINAL.search(lowered)
    if not found:
        return None
    digits, word, noun, noun_after, trailing = found.groups()
    if digits:
        return int(digits), noun
    if word:
        return (-1 if word == "last" else _ORDINALS[word]), noun
    value = int(trailing) if trailing.isdigit() else _CARDINALS[trailing]
    return value, noun_after


def _target(lowered: str, hits: list[ScoredChunk], titles: Mapping[str, str]) -> str | None:
    """One document, or nothing. Never a guess between two."""
    candidates = dict(titles)
    for hit in hits:
        candidates.setdefault(hit.chunk.document_id, hit.chunk.document_title)
    # A title the question names wins outright: "the chapters of Alice in
    # Wonderland" is about that book whatever retrieval ranked first.
    named = [doc for doc, title in candidates.items() if _names(lowered, title)]
    if len(named) == 1:
        return named[0]
    if len(named) > 1:
        return None
    # Otherwise the hits must agree: at least half of them from one document,
    # and never fewer than two.
    if not hits:
        return None
    counts = Counter(h.chunk.document_id for h in hits)
    doc, n = counts.most_common(1)[0]
    if n >= max(2, math.ceil(len(hits) / 2)):
        return doc
    return None


def _names(lowered: str, title: str) -> bool:
    """Whether the question mentions this title.

    Two distinctive title words, or the whole title when it is one word. A
    single word out of a longer title is not enough: a question about a person
    called Alice must not hijack a book whose title merely contains the name.
    """
    # Apostrophes split: "Alice's Adventures" is named by "alice", not "alice's".
    # A number of three digits or more is distinctive too: "part 301-10" names
    # the part whose title carries 301, and nothing else does.
    significant = [
        w
        for w in re.findall(r"[a-z0-9]+", title.lower())
        if len(w) > 3 or (w.isdigit() and len(w) >= 3)
    ]
    if not significant:
        return False
    present = [w for w in significant if re.search(rf"\b{re.escape(w)}\b", lowered)]
    if len(significant) == 1:
        return len(present) == 1
    return len(present) >= 2


# --- structure -------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Outline:
    """What a document's structure says in answer to an enumeration."""

    #: The list, in document order.
    items: tuple[str, ...]
    #: What was listed, plural: "chapters", "sections", "items", "persons", "terms".
    noun: str
    #: The heading the list sits under, when it does.
    under: str | None = None


def outline(
    blocks: Iterable[Block], wants: str, question: str, *, noun: str | None = None
) -> Outline | None:
    """Read the enumeration ``question`` asks for out of ``blocks``, or None.

    None means "the structure does not answer this", and the caller falls back
    to ordinary retrieval. It is returned freely: a wrong list is a confident
    wrong answer, and a fallback is only a slower right one. ``noun`` is what
    an ordinal counts, and narrows the headings to the ones that say it.
    """
    blocks = tuple(blocks)
    if not blocks:
        return None
    if wants in ("headings", "any"):
        found = _headings(blocks, noun)
        if found is not None:
            return found
        if wants == "headings":
            return None
    if wants in ("items", "persons", "any"):
        found = _list_under(blocks, question, wants)
        if found is not None:
            return found
        if wants != "any":
            return None
    if wants in ("terms", "any"):
        return _terms(blocks)
    return None


def _headings(blocks: tuple[Block, ...], noun: str | None) -> Outline | None:
    """The document's sections, minus a lone title above them.

    With a noun, only the headings that say it: asked about acts, a play's
    "THE PERSONS IN THE PLAY" is not one. Nothing matching means the structure
    does not answer this, not that it answers with everything.
    """
    headings = [b for b in blocks if b.kind is BlockKind.HEADING]
    if noun is not None:
        said = [b for b in headings if re.search(rf"\b{noun}s?\b", b.text, re.IGNORECASE)]
        if not said:
            return None
        return Outline(items=tuple(b.text.strip() for b in said), noun=f"{noun}s")
    if not headings:
        return None
    top_level = min(b.level for b in headings)
    at_top = [b for b in headings if b.level == top_level]
    sections = headings[1:] if len(at_top) == 1 and len(headings) > 1 else headings
    if len(sections) < 2:
        return None
    return Outline(items=tuple(b.text.strip() for b in sections), noun="sections")


def _list_under(blocks: tuple[Block, ...], question: str, wants: str) -> Outline | None:
    """List items under the heading the question is about, else the document's one list."""
    q_words = {w for w in _WORDS.findall(question.lower()) if len(w) > 3}
    if wants == "persons":
        q_words |= {"persons", "characters", "cast", "dramatis", "people"}
    runs: list[tuple[str | None, list[Block]]] = []
    current_heading: str | None = None
    for block in blocks:
        if block.kind is BlockKind.HEADING:
            current_heading = block.text.strip()
            continue
        if block.kind is BlockKind.LIST_ITEM:
            if runs and runs[-1][0] == current_heading:
                runs[-1][1].append(block)
            else:
                runs.append((current_heading, [block]))
    if not runs:
        return None
    noun = "persons" if wants == "persons" else "items"
    # A run whose heading shares a word with the question is the one asked for.
    for heading, items in runs:
        heading_words = set(_WORDS.findall((heading or "").lower()))
        if heading_words & q_words:
            return Outline(items=tuple(b.text.strip() for b in items), noun=noun, under=heading)
    # Otherwise only an unambiguous document - one list - may answer, and a
    # question about people is never answered from a list that does not say so.
    if len(runs) == 1 and wants != "persons":
        heading, items = runs[0]
        return Outline(items=tuple(b.text.strip() for b in items), noun="items", under=heading)
    return None


def _terms(blocks: tuple[Block, ...]) -> Outline | None:
    """A glossary: paragraphs that open with the term they define."""
    terms = []
    for block in blocks:
        if block.kind is not BlockKind.PARAGRAPH:
            continue
        found = TERM_PATTERN.match(block.text.strip())
        if found and len(found.group(1).split()) <= 6:
            terms.append(found.group(1).strip())
    if len(terms) < 3:
        return None
    return Outline(items=tuple(terms), noun="terms")


def section_chunks(chunks: Iterable[Chunk], heading: str) -> list[Chunk]:
    """The passages that sit under ``heading``, by the trail each carries."""
    wanted = heading.strip().lower()
    return [c for c in chunks if c.section and c.section.strip().lower().endswith(wanted)]


# --- rendering -----------------------------------------------------------------------


def render(
    found: Outline, *, title: str, document_id: str, ordinal: int | None
) -> tuple[str, bool]:
    """The answer text, and whether it is a refusal.

    An ordinal past the end refuses, and says how long the list is: "there is
    no fourth act" is only convincing next to "the play has three".
    """
    n = len(found.items)
    where = f" under “{found.under}”" if found.under else ""
    if ordinal is not None:
        index = n if ordinal == -1 else ordinal
        if index < 1 or index > n:
            return (
                f"{title} has {n} {found.noun}{where}; there is no "
                f"{_ordinal_word(ordinal)} one. [{document_id}]",
                True,
            )
        return (
            f"The {_ordinal_word(index)} of the {n} {found.noun} in {title}{where} is "
            f"“{found.items[index - 1]}”. [{document_id}]",
            False,
        )
    lines = [f"{title} has {n} {found.noun}{where}:"]
    lines.extend(f"  {i}. {item}" for i, item in enumerate(found.items, 1))
    lines.append(f"[{document_id}]")
    return "\n".join(lines), False


def _ordinal_word(n: int) -> str:
    if n == -1:
        return "last"
    for word, value in _ORDINALS.items():
        if value == n:
            return word
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


# --- assembly --------------------------------------------------------------------------


def assemble(
    chunks: Iterable[Chunk],
    *,
    context_tokens: int,
    reserved_tokens: int,
    fallback_max: int,
) -> tuple[list[Chunk], int]:
    """A document's passages in order, as many as the window will take.

    Returns the passages and how many were left out. ``reserved_tokens`` is
    what the prompt already spends - the system prompt, the question, and the
    answer's own budget. When the window is unknown (``context_tokens`` is 0)
    nothing can be fitted against, and a runtime that trims silently would
    drop the front of the prompt where the grounding rules live; so the cap is
    ``fallback_max`` passages rather than the whole document.
    """
    ordered = list(chunks)
    if context_tokens <= 0:
        capped = ordered[:fallback_max]
        return capped, len(ordered) - len(capped)
    budget_chars = max(0, int(context_tokens * 0.9) - reserved_tokens) * 4
    kept: list[Chunk] = []
    spent = 0
    for chunk in ordered:
        cost = len(chunk.text) + 80  # the SOURCES header line for this passage
        if kept and spent + cost > budget_chars:
            break
        kept.append(chunk)
        spent += cost
    return kept, len(ordered) - len(kept)
