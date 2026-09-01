"""Per-document index work, reused across rebuilds.

Every upload and every delete re-indexes the whole corpus, inside the request
the operator is waiting on. Measured once parses were cached and PDFs batched
and this was the cost left: 1.04s at 400 documents, 3.42s at 1,200, 7.23s at
2,400 - linear, about 3ms a document, so five thousand policies is fifteen
seconds after dragging in one file.

Chunking a document, tokenising its passages and counting its words depend on
that document alone. Only the tf-idf behind its tags needs the corpus, because
a word is distinctive against the documents that never use it. So the first
half is remembered and the second is recomputed every time.

What these hold: the index must be **identical** to the one a full rebuild
produced, and nothing a chunk carries may ever come back stale - least of all
who is allowed to read it.
"""

from __future__ import annotations

from collections import Counter

from openknowledge.documents.blocks import Block, BlockKind
from openknowledge.retrieval.base import Document, fold_word, tokenize
from openknowledge.retrieval.bm25 import BM25Retriever
from openknowledge.retrieval.derived import DerivedCache, fingerprint
from openknowledge.retrieval.tags import (
    corpus_document_frequency,
    derive_tags,
    rank_tags,
    tag_body,
    tag_sources,
)

_WORDS = ["expense", "travel", "approval", "manager", "receipt", "threshold", "quarter"]


def corpus(n: int, *, marker: str = "", principals: frozenset[str] = frozenset()) -> list[Document]:
    documents = []
    for i in range(n):
        body = " ".join(_WORDS[(i + j) % len(_WORDS)] for j in range(120))
        documents.append(
            Document(
                document_id=f"policy-{i:03d}",
                title=f"Policy {i}",
                text=f"{marker} {body} document number {i} " * 3,
                allowed_principals=principals,
            )
        )
    return documents


def snapshot(retriever: BM25Retriever) -> tuple[object, ...]:
    """Every field a search reads, so 'identical' means identical."""
    index = retriever._index
    return (
        index.chunks,
        index.term_freqs,
        index.lengths,
        index.doc_freq,
        index.avg_length,
        index.corpus_version,
        index.doc_principals,
        index.doc_tags,
        index.doc_tags_folded,
    )


def test_a_warm_cache_builds_the_same_index_as_a_cold_one() -> None:
    """The property everything else rests on.

    Every answer this product gives is scored against these numbers, and a
    rebuild that reused work slightly wrongly would change retrieval without
    changing anything anybody could see.
    """
    documents = corpus(30)
    warm = BM25Retriever()
    warm.index(documents)  # cold: every document derived
    warm.index(documents)  # warm: every document a hit
    cold = BM25Retriever()
    cold.index(documents)

    assert snapshot(warm) == snapshot(cold)
    assert warm._derived.hits == 30
    assert warm._derived.misses == 30


def test_the_tag_split_reproduces_what_derive_tags_always_produced() -> None:
    """The refactor that made this possible must not have moved a single tag.

    ``derive_tags`` was one function; it is now the composition of a
    per-document half that gets cached and a corpus-wide half that does not.
    Composition is the only thing that makes the cache safe, so it is asserted
    rather than assumed.
    """
    documents = corpus(25)
    frequency = corpus_document_frequency(documents)
    for document in documents:
        composed = rank_tags(tag_sources(document), tag_body(document), frequency, len(documents))
        assert composed == derive_tags(document, frequency, len(documents))


def test_tags_are_recomputed_because_they_depend_on_the_whole_corpus() -> None:
    """Adding a document changes what every other document is distinctive against.

    This is why tags are the half that is *not* cached. Shown with a word that
    earns its place and then loses it: "northstar" appears in one document out
    of twelve, so it is the most distinctive thing that document says; once
    twenty more documents use it, it is just a word. A cached tag set would be
    right on the day it was computed and quietly drift from then on.
    """
    # More filler words than the eight distinctive terms a document is allowed,
    # so the ranking is what decides rather than the cap.
    filler = (
        "expense travel approval manager receipt threshold quarter contractor "
        "allowance invoice deadline submission entitlement probation overtime"
    )
    body = " ".join([filler] * 20)
    rare = Document("policy-000", "Policy 0", "northstar " * 12 + body)
    others = [Document(f"policy-{i:03d}", f"Policy {i}", body) for i in range(1, 12)]

    retriever = BM25Retriever()
    retriever.index([rare, *others])
    assert "northstar" in retriever._index.doc_tags["policy-000"]

    # Twenty documents that all say it: it stops being distinctive of anything.
    common = [Document(f"extra-{i:03d}", f"Extra {i}", "northstar " * 12 + body) for i in range(20)]
    everything = [rare, *others, *common]
    retriever.index(everything)

    assert "northstar" not in retriever._index.doc_tags["policy-000"]
    frequency = corpus_document_frequency(everything)
    for document in everything:
        assert retriever._index.doc_tags[document.document_id] == derive_tags(
            document, frequency, len(everything)
        )


def test_an_access_change_never_serves_a_chunk_with_the_old_audience() -> None:
    """A chunk carries ``allowed_principals``, so the key must too.

    ``content_hash`` is the obvious key and would have been wrong here: the
    text of a document does not change when its folder's rule does, and a
    cache keyed on text alone would hand back passages stamped with the
    audience that folder used to have. That is the worst mistake this product
    can make, so it gets its own test rather than a comment.
    """
    retriever = BM25Retriever()
    retriever.index(corpus(6, principals=frozenset({"group:hr"})))
    assert {c.allowed_principals for c in retriever._index.chunks} == {frozenset({"group:hr"})}

    retriever.index(corpus(6, principals=frozenset({"group:finance"})))
    assert {c.allowed_principals for c in retriever._index.chunks} == {frozenset({"group:finance"})}
    assert retriever._index.doc_principals["policy-000"] == frozenset({"group:finance"})


def test_the_same_words_in_a_different_shape_are_different_chunks() -> None:
    """Chunking reads blocks, not text, so the key cannot stop at the text.

    A heading rewritten as a paragraph leaves ``document.text`` byte-identical
    and the passages different - a heading starts a chunk and a paragraph does
    not, so the same words come back as one passage or as two. Rare, and
    exactly the kind of thing a content hash would wave through.
    """
    said = ("Travel rules apply", "Expenses", "Receipts are required")
    flat = tuple(Block(kind=BlockKind.PARAGRAPH, text=t) for t in said)
    with_heading = (flat[0], Block(kind=BlockKind.HEADING, text=said[1], level=2), flat[2])

    text = "\n\n".join(said)
    one = Document("d", "D", text, blocks=flat)
    other = Document("d", "D", text, blocks=with_heading)
    assert one.content_hash == other.content_hash, "a content hash cannot tell these apart"

    cache = DerivedCache()
    first = cache.get(one, target_words=350, overlap_words=60)
    second = cache.get(other, target_words=350, overlap_words=60)
    assert cache.misses == 2 and cache.hits == 0
    assert len(first.chunks) == 1 and len(second.chunks) == 2


def test_a_different_chunk_window_is_a_different_derivation() -> None:
    """The same document cut at a different width is a different set of passages."""
    documents = corpus(4)
    cache = DerivedCache()
    for document in documents:
        cache.get(document, target_words=350, overlap_words=60)
        cache.get(document, target_words=80, overlap_words=10)
    assert cache.misses == 8 and cache.hits == 0


def test_only_the_document_that_changed_is_derived_again() -> None:
    """The whole point: an upload pays for one document, not for the corpus."""
    documents = corpus(20)
    retriever = BM25Retriever()
    retriever.index(documents)
    assert (retriever._derived.hits, retriever._derived.misses) == (0, 20)

    edited = [
        *documents[:-1],
        Document("policy-019", "Policy 19", "an entirely new expenses threshold of EUR 2000"),
    ]
    retriever.index(edited)
    assert retriever._derived.hits == 19
    assert retriever._derived.misses == 21  # the twenty, plus the one that changed


def test_the_cache_does_not_grow_by_an_entry_per_edit() -> None:
    """Each entry holds a copy of a document's passages, which is the one thing
    here big enough to matter. A version nobody has any more is not kept."""
    retriever = BM25Retriever()
    documents = corpus(10)
    for revision in range(5):
        edited = [
            *documents[:-1],
            Document("policy-009", "Policy 9", f"revision {revision} of the expenses policy"),
        ]
        retriever.index(edited)
    assert len(retriever._derived) == 10


def test_a_deleted_document_leaves_no_trace() -> None:
    """The rebuild is still a rebuild: what is reused is the work, not the result."""
    documents = corpus(8)
    retriever = BM25Retriever()
    retriever.index(documents)
    version = retriever.corpus_version

    retriever.index(documents[:-1])
    assert retriever.corpus_version != version
    assert "policy-007" not in {c.document_id for c in retriever._index.chunks}
    assert "policy-007" not in retriever._index.doc_tags
    assert len(retriever._derived) == 7


def test_indexing_twice_does_not_double_the_corpus_statistics() -> None:
    """The cached counters are handed out by reference.

    Accumulating into one of them instead of into a fresh Counter would turn
    one document's statistics into the whole corpus's, and every BM25 score
    with them - silently, and only after the second index.
    """
    documents = corpus(6)
    retriever = BM25Retriever()
    retriever.index(documents)
    first = Counter(retriever._index.doc_freq)
    retriever.index(documents)
    assert retriever._index.doc_freq == first


def test_the_fingerprint_covers_what_a_chunk_is_made_of() -> None:
    """Named field by field, because each one is a way to serve a stale chunk."""
    base = Document("d", "Title", "the expenses policy text", url="file:///a")
    key = fingerprint(base, target_words=350, overlap_words=60)
    variants = [
        Document("other", base.title, base.text, url=base.url),
        Document(base.document_id, "Other", base.text, url=base.url),
        Document(base.document_id, base.title, "other text", url=base.url),
        Document(base.document_id, base.title, base.text, url="file:///b"),
        Document(base.document_id, base.title, base.text, url=base.url, superseded=True),
        Document(
            base.document_id,
            base.title,
            base.text,
            url=base.url,
            allowed_principals=frozenset({"group:hr"}),
        ),
    ]
    for variant in variants:
        assert fingerprint(variant, target_words=350, overlap_words=60) != key


def test_the_index_statistics_are_what_they_claim_to_be() -> None:
    """Recomputed by hand from the chunks, not compared against another build.

    Warm agreeing with cold proves only that they agree: a derivation that
    counted the wrong thing would produce the same wrong index both ways and
    every test above would pass. These are the numbers every BM25 score is
    computed from, so they are checked against what they are supposed to mean.
    """
    documents = corpus(15)
    retriever = BM25Retriever()
    retriever.index(documents)
    index = retriever._index

    # A chunk's length is how many tokens it has, not how many distinct ones.
    assert list(index.lengths) == [len(tokenize(chunk.text)) for chunk in index.chunks]

    # Document frequency counts, per term, how many *chunks* contain it - once
    # each, however often the chunk repeats the word.
    expected: Counter[str] = Counter()
    for chunk in index.chunks:
        expected.update(set(tokenize(chunk.text)))
    assert index.doc_freq == expected

    assert index.avg_length == sum(index.lengths) / len(index.lengths)


def test_the_corpus_vocabulary_is_every_word_not_only_the_taggable_ones() -> None:
    """What a document frequency counts is every word in it.

    The tag ranking asks this how common a word is; narrowing it to the words
    that could *become* tags would change every tf-idf score in the corpus,
    and would do it invisibly - the tags would still look like tags.
    """
    documents = [
        Document("a", "A", "the cap is 12 eur per quarter"),
        Document("b", "B", "eur only, and no cap at all"),
    ]
    frequency = corpus_document_frequency(documents)

    assert frequency[fold_word("the")] == 1
    assert frequency[fold_word("is")] == 1
    assert frequency[fold_word("12")] == 1
    assert frequency[fold_word("eur")] == 2
    assert frequency[fold_word("cap")] == 2
    assert frequency[fold_word("quarter")] == 1
