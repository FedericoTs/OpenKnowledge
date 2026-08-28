"""Document tags, derived at index time, and the routing they allow.

Uploading a document is the one moment the system gives it undivided
attention, so that is when each document gets a small set of tags: the words
that say what it is about. Nothing here calls a model - tags come from the
document's own name, title, headings, and its statistically distinctive
vocabulary - so indexing stays free and byte-identical across runs, which the
answer cache depends on.

The point of the tags is to shrink the search radius. On a corpus of a
thousand documents, a question about the expenses policy is scored against
every chunk of every engineering runbook, and the noise has two costs: chunks
from unrelated documents crowd the top k, and the grounding gate then fails
answers that would have been clean over the right context. When the
question's own words match a *few* documents' tags decisively, retrieval is
restricted to those documents.

The failure mode to fear is the opposite one: a question routed away from
the document that held its answer. So routing is deliberately cowardly -
**any doubt means no restriction**:

* a document only counts as matched on two folded tag hits, because one
  shared word ("policy") is coincidence;
* the restriction only applies when the matched set is a small share of the
  corpus - if a third of the documents match, the question is broad and
  narrowing it would only add risk;
* no match at all means no restriction, not an empty result.

Tags are stored as readable words - they are shown in the document listing -
and folded only for matching, by the same ``fold_word`` conflict relevance
uses, so "travelling" finds the document tagged "travel".
"""

from __future__ import annotations

from collections import Counter
from math import log

from .base import Document, fold_word, tokenize

#: Generic document-furniture words that would tag every file in the corpus.
#: Small on purpose: real topic words ("expenses", "policy", "security") stay,
#: because they are exactly what questions share with the right documents.
_NOISE = frozenset(
    (  # noqa: SIM905 - a 50-item list literal here is far less readable
        "the and for with from into over under about this that these those are was "
        "were has have had can may must not all any each per within a an of to in on "
        "at by or if it its you your we our they their is be been do does did will "
        "would should"
    ).split()
)

#: How many statistically distinctive body words join the tag set. Enough to
#: catch the topics the title does not name; few enough that the tags stay a
#: description rather than an index.
_DISTINCTIVE_TERMS = 8

#: A tag set is capped so the listing stays readable and a pathological
#: document cannot tag itself into every question's route.
_MAX_TAGS = 16

#: A document is matched only when the question shares at least this many
#: folded tags with it. One shared word is coincidence.
_MIN_HITS = 2

#: Restrict only when the matched documents are at most this share of the
#: corpus. A question matching more than a third of the documents is a broad
#: question, and narrowing it would only add risk for no precision.
_MAX_MATCHED_SHARE = 1 / 3


def derive_tags(
    document: Document, document_frequency: Counter[str], corpus_size: int
) -> tuple[str, ...]:
    """One document's tags: readable words, deduplicated by folded form.

    Four sources, in the order a human would trust them - earlier sources
    keep their spot when the cap bites:

    * the document id, which encodes the operator's own folder taxonomy
      ("hr-expenses-policy" says hr, expenses, policy);
    * the title;
    * the headings, which name the topics the title does not;
    * the ``_DISTINCTIVE_TERMS`` body words with the highest tf-idf against
      the rest of the corpus, so a document about GDPR gets tagged "gdpr"
      even when no heading says it.

    ``document_frequency`` counts, per *folded* word, how many documents
    contain it; computed once per index build and shared.
    """
    chosen: dict[str, str] = {}  # folded form -> display word, insertion-ordered

    def consider(word: str, minimum: int) -> None:
        lower = word.lower()
        if len(lower) < minimum or lower in _NOISE or lower.isdigit():
            return
        chosen.setdefault(fold_word(lower), lower)

    # Two letters is enough for a *named* source: "hr" is a real taxonomy.
    for part in document.document_id.replace("/", "-").split("-"):
        consider(part, 2)
    for word in tokenize(document.title):
        consider(word, 2)
    for block in document.blocks:
        if block.kind.name == "HEADING":
            for word in tokenize(block.text):
                consider(word, 2)

    named = len(chosen)
    body = Counter(
        w for w in tokenize(document.text) if len(w) > 3 and w not in _NOISE and not w.isdigit()
    )
    scored = sorted(
        (
            (tf * log(1 + corpus_size / (1 + document_frequency[fold_word(word)])), word)
            for word, tf in body.items()
        ),
        key=lambda pair: (-pair[0], pair[1]),
    )
    for _, word in scored:
        if len(chosen) >= named + _DISTINCTIVE_TERMS:
            break
        consider(word, 4)

    return tuple(list(chosen.values())[:_MAX_TAGS])


def corpus_document_frequency(documents: list[Document]) -> Counter[str]:
    """Per folded word, how many documents contain it."""
    frequency: Counter[str] = Counter()
    for document in documents:
        frequency.update({fold_word(w) for w in tokenize(document.text)})
    return frequency


def fold_tags(tags: tuple[str, ...]) -> frozenset[str]:
    """The matching form of a tag set. Folded once at index time, because
    folding every document's tags on every question is avoidable work."""
    return frozenset(fold_word(t) for t in tags)


def route_by_tags(question: str, folded_tags: dict[str, frozenset[str]]) -> frozenset[str] | None:
    """The documents this question should be answered from, or None for all.

    None is the common and safe answer: it means "search everything", which
    is exactly what retrieval did before tags existed. A frozenset is only
    returned when a few documents match the question decisively - and the
    caller treats it as a restriction, never as a ranking.
    """
    if not folded_tags:
        return None
    words = frozenset(
        fold_word(w) for w in tokenize(question) if w not in _NOISE and not w.isdigit()
    )
    if not words:
        return None

    matched = frozenset(
        doc_id for doc_id, tags in folded_tags.items() if len(words & tags) >= _MIN_HITS
    )
    if not matched:
        return None
    # Floor of two: a document and its archived twin matching together is the
    # decisive case, not a broad one - superseded demotion settles the pair.
    if len(matched) > max(2, int(len(folded_tags) * _MAX_MATCHED_SHARE)):
        return None
    return matched
