"""Document tags, derived at index time, and the routing they allow.

Uploading a document is the one moment the system gives it undivided
attention, so that is when each document gets a small set of tags: the words
that say what it is about. Nothing here calls a model - tags come from the
document's own name, title, headings, and its statistically distinctive
vocabulary - so indexing stays free and byte-identical across runs, which the
answer cache depends on.

The point of the tags is to keep the right documents inside the search
radius. On a corpus of a thousand documents, the chunk that answers an
expenses question can be buried below a thousand near-matches and never
reach the candidate list at all. When the question's own words match a
*few* documents' tags decisively, those documents are *guaranteed*
representation among the candidates - their best chunks rescued from below
the cut, displacing the weakest strangers. Nothing is reordered: rank is
earned by score exactly as without tags, because two stronger designs -
filtering to the routed documents, and putting them first - both measurably
degraded the local model's answers (see :func:`guarantee_routed`).

The failure mode to fear is a question routed away from the document that
held its answer. So the route itself is deliberately cowardly - **any doubt
means no route**:

* a document only counts as matched on two folded tag hits, because one
  shared word ("policy") is coincidence;
* a route only forms when the matched set is a small share of the corpus -
  if a third of the documents match, the question is broad and narrowing it
  would only add risk;
* no match at all means no route, not an empty result.

Tags are stored as readable words - they are shown in the document listing -
and folded only for matching, by the same ``fold_word`` conflict relevance
uses, so "travelling" finds the document tagged "travel".
"""

from __future__ import annotations

from collections import Counter
from math import log

from .base import Document, ScoredChunk, fold_word, tokenize

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

#: And never more than this many documents, whatever the share works out to.
#:
#: The share rule alone was written against a corpus of three, where the
#: floor of two did all the work. On a thousand documents a third is three
#: hundred and thirty, so a question sharing two common words with a hundred
#: boilerplate documents counted as naming them decisively. It then spent
#: every rescue slot on them: measured at a thousand documents, a passage
#: ranked FIRST for its question was evicted from a k=6 search and the
#: question was refused.
#:
#: A route is a document or two that a question names. Past a handful it is
#: not a route, it is a topic - and topics are what the score is for.
_MAX_MATCHED_DOCS = 5


def tag_sources(document: Document) -> tuple[tuple[str, str], ...]:
    """The tags a document names itself, needing no corpus to know.

    Three sources, in the order a human would trust them - the document id,
    which encodes the operator's own folder taxonomy ("hr-expenses-policy"
    says hr, expenses, policy); the title; and the headings, which name the
    topics the title does not. Folded form to display form, insertion-ordered,
    because the order is what survives when the cap bites.

    Split out from ``derive_tags`` so an index rebuild can keep it: this is
    the half that depends on one document, and the tf-idf half that needs the
    whole corpus is recomputed every time. ``derive_tags`` still composes the
    two, and a test asserts the composition equals what it always produced.
    """
    chosen: dict[str, str] = {}
    for part in document.document_id.replace("/", "-").split("-"):
        _consider(chosen, part, 2)
    for word in tokenize(document.title):
        _consider(chosen, word, 2)
    for block in document.blocks:
        if block.kind.name == "HEADING":
            for word in tokenize(block.text):
                _consider(chosen, word, 2)
    return tuple(chosen.items())


def tag_body(document: Document) -> Counter[str]:
    """The body words a tf-idf ranking chooses from, counted."""
    return Counter(
        w for w in tokenize(document.text) if len(w) > 3 and w not in _NOISE and not w.isdigit()
    )


def folded_vocabulary(document: Document) -> frozenset[str]:
    """Every folded word this document contains - one document's share of the
    corpus document frequency. Every token, not only the taggable ones."""
    return frozenset(fold_word(w) for w in tokenize(document.text))


def _consider(chosen: dict[str, str], word: str, minimum: int) -> None:
    lower = word.lower()
    if len(lower) < minimum or lower in _NOISE or lower.isdigit():
        return
    chosen.setdefault(fold_word(lower), lower)


def rank_tags(
    sources: tuple[tuple[str, str], ...],
    body: Counter[str],
    document_frequency: Counter[str],
    corpus_size: int,
) -> tuple[str, ...]:
    """The named sources, then the body words most distinctive against the corpus.

    The corpus-dependent half, kept separate because it must run on every
    index build: adding one document changes what every other document's
    words are distinctive *against*, so a cached tag set would slowly stop
    being true. It is also the cheap half - the tokenising is what costs.
    """
    chosen = dict(sources)
    named = len(chosen)
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
        _consider(chosen, word, 4)
    return tuple(list(chosen.values())[:_MAX_TAGS])


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
    return rank_tags(tag_sources(document), tag_body(document), document_frequency, corpus_size)


def corpus_document_frequency(documents: list[Document]) -> Counter[str]:
    """Per folded word, how many documents contain it."""
    frequency: Counter[str] = Counter()
    for document in documents:
        frequency.update(folded_vocabulary(document))
    return frequency


def fold_tags(tags: tuple[str, ...]) -> frozenset[str]:
    """The matching form of a tag set. Folded once at index time, because
    folding every document's tags on every question is avoidable work."""
    return frozenset(fold_word(t) for t in tags)


def route_by_tags(question: str, folded_tags: dict[str, frozenset[str]]) -> frozenset[str] | None:
    """The documents this question names decisively, or None for all.

    None is the common and safe answer: it means "search everything", which
    is exactly what retrieval did before tags existed. A frozenset is a
    *guarantee of candidacy*, applied by :func:`guarantee_routed` - never a
    filter and never a reordering; both were measured and both made the
    local model worse.
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
    if len(matched) > _MAX_MATCHED_DOCS:
        return None
    return matched


def guarantee_routed(
    ranked: list[ScoredChunk], within: frozenset[str] | None, k: int
) -> list[ScoredChunk]:
    """Cut to k, first making sure every named document is represented.

    Two stronger designs were measured and rejected on the golden sets. A
    hard filter starved the model: routed to a document that chunks to one
    window, it saw a one-chunk context and refused a question it answers
    with a fuller one. Putting routed chunks first starved it differently:
    the context filled with same-topic tables, and on the aveline set the
    local model refused one scope-ambiguous question and recited a
    neighbouring row's figures on another (accuracy 0.88 against a passing
    1.0). The mixed, score-earned context was doing quiet work.

    So order is earned by score, exactly as without tags. The route's one
    power is rescue: a named document with no chunk above the cut gets its
    best-ranked chunk in, displacing the weakest strangers. On a small
    corpus the named documents are already in the head and routing changes
    nothing - measured: identical contexts with routing on and off. On the
    thousand-document corpus this exists for, it is the difference between
    the right document being in the candidates and being buried below them.
    """
    head = ranked[:k]
    if within is None or k <= 0:
        return head
    present = {s.chunk.document_id for s in head}
    missing = set(within) - present
    if not missing:
        return head

    # Rescue takes a minority of k, never the whole of it. The head is
    # earned by score, and the arithmetic that replaced it - keep
    # ``k - len(rescued)``, floored at zero - quietly meant "keep none"
    # as soon as the route named k documents or more. Measured on a
    # thousand-document corpus: the passage ranked FIRST for its question
    # was evicted from a k=6 search, and the question came back refused
    # with the answer sitting in the index. Half of k, at least one.
    budget = max(1, k // 2)
    rescued: list[ScoredChunk] = []
    for scored in ranked[k:]:
        if len(rescued) >= budget:
            break
        doc_id = scored.chunk.document_id
        if doc_id in missing:
            # Best-ranked first, because the tail is in score order: when
            # the budget cannot cover every named document, it should
            # spend on the strongest rather than the alphabetically first.
            rescued.append(scored)
            missing.discard(doc_id)
    if not rescued:
        return head
    return head[: k - len(rescued)] + rescued
