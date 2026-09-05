"""Questions about the corpus, answered from the corpus's own index.

"What documents do you have?" is the first thing a person types into a document
assistant, and it is the one question no retriever can answer: it asks *about*
the collection, not *from* it. BM25 finds nothing to match on, the model is
handed either nothing or a random passage, and the honest reply - "that isn't
covered by the documents I have" - reads as a system that does not know what it
is holding. It knows exactly what it is holding.

So this answers those from the index: free, instant, no model call, and correct
by construction rather than by a model reading a passage and getting it right.

The whole risk here is a false positive. "Which documents mention parental
leave?" is a *document* question wearing similar words, and hijacking it would
be far worse than missing a meta-question. The test is therefore not "does this
look like a meta-question" but "is there anything else in it": a question
qualifies only when removing the meta-vocabulary leaves no content behind.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..retrieval.base import Chunk

_WORDS = re.compile(r"[a-z0-9']+")

#: Phrases that make a question about the collection rather than its contents.
_META = frozenset(
    [
        "document",
        "documents",
        "doc",
        "docs",
        "file",
        "files",
        "format",
        "formats",
        "policy",
        "policies",
        "paper",
        "papers",
        "index",
        "corpus",
        "knowledge",
        # The collection's subject matter, asked for in the abstract: "what
        # topics do you cover?" is a question about the collection exactly as
        # "what documents do you have?" is. Measured in the field: "what are
        # the macro-categories of info they covers?" cost a frontier call and
        # ended refused, when the index knew its own tags all along.
        "topic",
        "topics",
        "category",
        "categories",
        "tag",
        "tags",
        "subject",
        "subjects",
        "area",
        "areas",
        "content",
        "contents",
        "info",
        "information",
    ]
)

#: Words that make a question about the assistant itself. "What can you help
#: me with?" is the first thing many people type, mentions no document, and
#: deserves the same free index answer - measured in the field, it cost a
#: frontier call and came back "that isn't covered by the documents I have."
_CAPABILITY = frozenset(
    ["help", "helping", "helps", "assist", "assists", "assistance", "able", "capabilities"]
)

#: Things people ask whether the assistant does: "do you summarize
#: documents?" Only spent in the you-frame below, because the same verbs open
#: real work requests - "summarize the documents" is an instruction, not a
#: question about abilities, and hijacking it would be worse than any miss.
_ASSISTANT_VERBS = frozenset(
    (  # noqa: SIM905 - a list literal this long is far less readable
        "summarize summarise summarizing summarising translate translating compare "
        "comparing analyze analyse analyzing analysing explain explaining cite "
        "citing quote quoting extract extracting provide provided providing upload "
        "uploaded uploading remember work works "
        # Measured in the field: "what files can you manage?" - a question
        # every new user asks - fell out of this frame on the verb alone,
        # escalated to a paid model, and was refused. The nouns were already
        # here; the verbs people pair them with were not.
        "manage manages managing handle handles handling support supports "
        "supported read reads reading accept accepts accepting ingest ingests "
        "ingesting process processes take takes open opens"
    ).split()
)

#: The openings of a question *about* the assistant rather than an instruction
#: to it. An imperative starts with its verb; these start with an auxiliary or
#: an interrogative, and the "you" requirement seals the frame.
_ASKING_ABOUT_YOU = frozenset(
    ["do", "does", "can", "could", "will", "would", "are", "is", "what", "how", "who", "which"]
)

#: Words that carry no subject: interrogatives, auxiliaries, pronouns, and the
#: verbs people use to ask what exists. Anything left after these and the meta
#: vocabulary is a real subject, and the question is not about the corpus.
_EMPTY = frozenset(
    [
        "a",
        "about",
        "all",
        "an",
        "and",
        "any",
        "are",
        "available",
        "as",
        "at",
        "av",
        "be",
        "been",
        "by",
        "can",
        "could",
        "currently",
        "do",
        "does",
        "doing",
        "done",
        "exist",
        "exists",
        "get",
        "got",
        "give",
        "got",
        "has",
        "have",
        "having",
        "here",
        "hold",
        "holding",
        "how",
        "i",
        "in",
        "indexed",
        "is",
        "it",
        "its",
        "know",
        "list",
        "listed",
        "loaded",
        "many",
        "me",
        "much",
        "of",
        "on",
        "or",
        "our",
        "ours",
        "read",
        "search",
        "see",
        "show",
        "stored",
        "tell",
        "that",
        "the",
        "their",
        "there",
        "these",
        "they",
        "this",
        "those",
        "to",
        "up",
        "us",
        "use",
        "used",
        "we",
        "what",
        "whats",
        "which",
        "who",
        "whose",
        "with",
        "you",
        "your",
        "yours",
        "access",
        "answer",
        "answers",
        "contain",
        "contains",
        "covered",
        "from",
        "kind",
        "kinds",
        "loaded",
        "name",
        "names",
        "sort",
        "sorts",
        "thing",
        "things",
        "total",
        "type",
        "types",
        "cover",
        "covers",
        "covering",
        "macro",
        "main",
        "please",
        "for",
        "my",
        "questions",
    ]
)


#: One named document, not the collection. "What does *this* document cover?"
#: is asked with a file open or just uploaded, and answering it with an
#: inventory of everything on the server is not a near miss - it is a confident
#: answer to a question nobody asked, and it is delivered free, ungated, from
#: the first branch of the cascade.
#:
#: Grammatical number carries the whole distinction and this used to discard
#: it: "cover" and "covers" joined the no-subject vocabulary in 17818b9 so that
#: "what are the macro-categories of info they covers?" would be recognised,
#: and the side effect was that "what does the document cover?" stopped having
#: a subject too. A singular collection noun under a definite or demonstrative
#: determiner is that subject, and it belongs to retrieval.
_ONE_DOCUMENT = re.compile(
    r"\b(?:the|this|that)\s+"
    r"(?:document|file|doc|paper|report|attachment|policy|handbook)\b"
)


def _names_one_document(question: str) -> bool:
    """Whether the question is about a single document rather than the shelf.

    Plurals are deliberately not matched: "what do the documents cover?" and
    "what documents do you have?" are questions about the collection and must
    keep their free answer.
    """
    return _ONE_DOCUMENT.search(question.lower()) is not None


@dataclass(frozen=True, slots=True)
class CorpusQuestion:
    """A recognised question about the collection, and which kind it is."""

    #: "count" wants a number, "list" wants the titles. Both get both, because
    #: the difference is not worth a wrong guess - but the phrasing follows.
    wants: str


def recognise(question: str) -> CorpusQuestion | None:
    """Whether ``question`` asks about the corpus rather than from it.

    Conservative by design: a question qualifies only when it mentions the
    collection (or the assistant's abilities) *and* has no other subject in
    it. "How many documents do you have" qualifies; "how many days of
    parental leave" does not, and neither does "which documents mention
    contractors" or "how can you help me claim expenses".
    """
    words = _WORDS.findall(question.lower())
    if not words:
        return None
    # "What does this document cover?" names a subject; it is just not a
    # subject made of content words, so the residue test below cannot see it.
    if _names_one_document(question):
        return None
    vocabulary = set(words)
    about_collection = bool(_META & vocabulary)
    about_abilities = bool(_CAPABILITY & vocabulary)

    # The you-frame: a question that opens interrogatively and addresses
    # "you" is asking about the assistant, so verbs like "summarize" stop
    # being subjects - "do you summarize documents?" is a capability
    # question. An imperative ("summarize the expenses policy") never enters
    # this frame, and a named subject still leaves residue either way.
    asking_about_you = "you" in vocabulary and words[0] in _ASKING_ABOUT_YOU
    allowed_verbs = _ASSISTANT_VERBS if asking_about_you else frozenset()

    residue = [
        word
        for word in words
        if word not in _META
        and word not in _CAPABILITY
        and word not in _EMPTY
        and word not in allowed_verbs
    ]
    if residue:
        return None
    # "What can you do for me?" carries no trigger word at all - every word
    # is function vocabulary. A question with no subject cannot be answered
    # from documents by construction; addressed to "you", it is about the
    # assistant. Measured in the field: it reached the frontier model, which
    # answered as if it were the organisation in the documents.
    if not about_collection and not about_abilities and not asking_about_you:
        return None
    used_verbs = bool(vocabulary & allowed_verbs)
    if about_abilities or used_verbs or (asking_about_you and not about_collection):
        # The help answer includes the listing, so it is never less than the
        # list - and "do you summarize documents?" deserves the sentence
        # about summarising, not just an inventory.
        return CorpusQuestion(wants="help")
    return CorpusQuestion(wants="count" if "many" in words or "much" in words else "list")


#: What people call the formats, rather than what the filesystem calls them.
#: Unmapped suffixes fall through to the bare extension, so a parser added
#: without a name here is still named to the person - understated, never
#: omitted.
_FORMAT_NAMES = {
    ".pdf": "PDF",
    ".docx": "Word",
    ".pptx": "PowerPoint",
    ".xlsx": "Excel",
    ".xlsm": "Excel",
    ".md": "Markdown",
    ".markdown": "Markdown",
    ".rst": "Markdown",
    ".txt": "plain text",
}


#: Ways to address the assistant. Unlike verbs and nouns, this is a genuinely
#: closed class - English has no long tail of words for "you" - which is what
#: makes the check below immune to the vocabulary gap that sent "what files
#: can you manage?" to a paid refusal.
_SECOND_PERSON = frozenset(["you", "your", "yours", "yourself", "u"])

#: The openings of a question rather than an instruction. Also closed-class.
#: A wider set than ``_ASKING_ABOUT_YOU`` because this check does not have to
#: carry the whole decision on its own - the corpus gets the second vote.
_QUESTION_OPENERS = frozenset(
    (  # noqa: SIM905
        "do does did can could will would shall should are is was were am have "
        "has what how who which why when where whats"
    ).split()
)


def asks_about_the_assistant(question: str) -> bool:
    """Whether the question addresses the assistant itself.

    Grammar, not vocabulary. Every word list in this module is open-class -
    there is always another verb someone will use - and each gap costs a
    paid call and a refusal before anyone notices. Measured against a set
    written to defeat those lists, the lists recognised 25% of real
    self-referential questions.

    This asks two closed-class questions instead: does the sentence address
    "you", and does it open like a question rather than an instruction. That
    is deliberately loose - "can you tell me what the handbook says?" passes
    it - because on its own it decides nothing. The caller pairs it with
    what the documents actually contain, and only a question the corpus has
    no answer for is treated as being about the assistant.
    """
    words = re.findall(r"[a-z']+", question.lower())
    if not words:
        return False
    if not _SECOND_PERSON & set(words):
        return False
    # An imperative addressed at the assistant ("summarise the handbook for
    # yourself") opens with its verb, not with an interrogative. Looking at
    # the first two words rather than the first keeps that exclusion - an
    # imperative has no interrogative in either slot - while surviving the
    # typo a real user made twice: "whaat can you do for me?".
    return any(w in _QUESTION_OPENERS for w in words[:2])


def evidence_text(chunks: Sequence[Chunk]) -> str:
    """What the retrieved passages offer, labels included.

    The title carries the subject as often as the body does - a document
    called "Expenses Policy" may never repeat the word "expenses" in its
    text - and an early version of the check below missed exactly that,
    treating a question the corpus could answer as one about the assistant.
    The grounding gate learned the same lesson as policy g2.
    """
    return " ".join(
        f"{c.document_id} {c.document_title} {c.locator or ''} {c.text}" for c in chunks
    )


def corpus_has_nothing_to_say(question: str, evidence: str, *, floor: float = 0.34) -> bool:
    """Whether the best passages retrieved say anything about this question.

    The second vote, and the one that keeps the first from doing harm. A
    real document question - "can you tell me what the handbook says about
    parental leave?" - lands its content words in the passages it retrieves.
    A question about the assistant does not, because no corpus describes the
    software reading it.

    The floor is deliberately low: firing this when the documents *do* have
    an answer would replace a real answer with a description of the product,
    which is far worse than the refusal it exists to prevent.
    """
    # Function words carry no subject, so counting them dilutes the ratio and
    # makes the net fire on questions the corpus can answer: "what does the
    # policy say you should do about expenses?" landed its only real word and
    # still scored 1/3, because "say" and "should" were counted as subject.
    # The grounding gate already maintains this list for the same reason.
    from ..retrieval.grounding import _FUNCTION_WORDS

    ignored = _SECOND_PERSON | _EMPTY | _META | _CAPABILITY | _FUNCTION_WORDS
    subject = [w for w in re.findall(r"[a-z']+", question.lower()) if w not in ignored]
    if not subject:
        return True
    seen = set(re.findall(r"[a-z']+", evidence.lower()))
    landed = sum(1 for w in subject if w in seen)
    return landed / len(subject) < floor


def _readable_formats() -> str:
    """The file types this build can actually parse, in prose.

    Read from the registry that does the parsing: a list typed out here by
    hand would drift the moment a parser is added or dropped, and this
    sentence is a promise the ingest path has to keep.
    """
    from ..documents import SUPPORTED_SUFFIXES

    # Named in the order above - the order a person would say them in -
    # then anything the registry has that this map does not.
    seen: list[str] = []
    for suffix, name in _FORMAT_NAMES.items():
        if suffix in SUPPORTED_SUFFIXES and name not in seen:
            seen.append(name)
    for suffix in sorted(SUPPORTED_SUFFIXES):
        if suffix not in _FORMAT_NAMES:
            seen.append(suffix.lstrip(".").upper())
    if len(seen) == 1:
        return seen[0]
    return ", ".join(seen[:-1]) + f" and {seen[-1]}"


def describe(
    titles: list[str],
    *,
    chunks: int,
    hidden: int = 0,
    limit: int = 40,
    tags: dict[str, tuple[str, ...]] | None = None,
    wants: str = "list",
) -> str:
    """State what is indexed, in the words a person asked for it.

    Titles and their derived tags, never contents: this path exists because it
    can answer without reading anything, and the moment it summarises what is
    *in* a document it is making a claim no gate has checked. The tags are the
    honest middle ground - derived at index time from each document's own
    name, headings and vocabulary, they answer "what topics do you cover?"
    without asserting a single sentence of content.
    """
    if not titles:
        return (
            "I have no documents indexed yet. Put files in the documents folder "
            "and run `openknowledge index`, and I will be able to answer from them."
        )

    count = len(titles)
    noun = "document" if count == 1 else "documents"
    lines = []
    if wants == "help":
        lines.append(
            "I answer questions from the documents indexed here, citing the "
            "source for every claim. I can summarise any of them - ask "
            '"summarise <document name>" - and when the documents do not '
            "cover something I say so rather than guessing."
        )
        lines.append("")
        # "What files can you manage?" deserves the real list, and it is
        # read from the registry that actually parses them so the answer
        # cannot drift from the code that has to honour it.
        lines.append(f"You can add {_readable_formats()} files by dragging them in.")
        lines.append("")
    lines.append(f"I have {count} {noun} indexed, in {chunks} passage{'' if chunks == 1 else 's'}:")
    for title in titles[:limit]:
        doc_tags = (tags or {}).get(title, ())
        if doc_tags:
            shown = ", ".join(doc_tags[:8])
            lines.append(f"  - {title} - covering: {shown}")
        else:
            lines.append(f"  - {title}")
    if count > limit:
        lines.append(f"  ... and {count - limit} more")
    if hidden:
        lines.append(
            f"\n({hidden} further document{'' if hidden == 1 else 's'} you do not have access to.)"
        )
    lines.append(
        "\nAsk me about what is in any of them and I will answer from the text, with the source."
    )
    return "\n".join(lines)
