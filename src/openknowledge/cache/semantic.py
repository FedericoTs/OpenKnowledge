"""The semantic cache: a cached answer, re-judged for a new phrasing.

"How much parental leave do I get?" and "How many weeks of parental leave do
employees get?" are the same question, and the exact cache cannot see it -
its key is a hash. Embeddings can. But a measurement stopped the obvious
design cold: on the real embedding model, genuine paraphrases scored
0.727-0.849 cosine while "parental leave weeks" vs "annual leave days" - two
questions with different correct answers - scored 0.810, inside the
paraphrase band. There is no threshold that catches the paraphrases without
sometimes serving the wrong cached answer, and a cache that is occasionally
confidently wrong is worse than no cache.

So similarity only *nominates*. The nominated answer is then judged by the
grounding gate against the NEW question's own retrieval - the same
check_grounding, the same support threshold, the same standard every live
answer meets. The paraphrase retrieves the same passages, so the cached
answer grounds and is served; the annual-leave question retrieves
annual-leave passages, the parental answer's figures find no support there,
and the gate refuses the shortcut. In cascade terms the semantic cache is a
zero-cost rung: a candidate completion that happens to come from disk,
under the gate like everything else.

Vectors live beside the answers they describe and are evicted with them:
a vector for an answer that no longer exists nominates nothing.
"""

from __future__ import annotations

import array
import logging
from dataclasses import dataclass

from ..retrieval.embed import Embedder, EmbeddingError, normalise

log = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS question_vectors (
    cache_key      TEXT PRIMARY KEY,
    corpus_version TEXT NOT NULL,
    value          BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_qvec_corpus ON question_vectors(corpus_version);
"""


#: Words that shape a question without naming its subject. The shared
#: stopword list was built for policy prose, where "much" and "get" can
#: matter; in a question they are scaffolding, and requiring the cached
#: answer to contain them would dismiss "how much parental leave do I get"
#: over the word "much". Folded the same way _content_words folds.
_QUESTION_NOISE = frozenset(
    [
        "i",
        "you",
        "we",
        "me",
        "my",
        "our",
        "your",
        "how",
        "what",
        "when",
        "where",
        "who",
        "which",
        "why",
        "whose",
        "much",
        "many",
        "more",
        "less",
        "get",
        "got",
        "give",
        "need",
        "want",
        "know",
        "tell",
        "show",
        "do",
        "does",
        "did",
        "can",
        "could",
        "should",
        "would",
        "will",
        "may",
        "might",
        "must",
        "please",
        "anyone",
        "someone",
    ]
)


def covers(new_question: str, cached_question: str) -> bool:
    """Whether the cached QUESTION asks about everything the new one asks about.

    The third arbiter, and the one whose scope the golden set corrected twice.

    It exists because a live run served the parental-leave answer to "how many
    days of annual leave do employees get?" at similarity 0.81 - on that
    corpus there is no annual-leave document, so retrieval's top-ranked
    document for the trap WAS the cited one and the grounding gate passed at
    full support. Both judge "closest thing we have"; neither asks whether the
    closest thing answers the question. So: every content word of the new
    question must appear in the cached question, folded the way the
    contested-claims gate folds words. "annual" and "days" die here.

    The first version also searched the cached *answer* for those words, and
    the golden set caught what that allows: the entitlement answer happened to
    volunteer a sentence about contractors, so "do contractors get parental
    leave?" matched it and was served "20 weeks" - a forbidden answer for that
    question. An answer's ramblings must not widen what the entry claims to
    answer; a cache entry can vouch for its question and nothing else.

    The cost of this strictness is honest and bounded: a paraphrase using
    genuinely different vocabulary is dismissed and pays the model call it
    would have paid anyway. This cache accelerates re-phrasings; refusing the
    ones it cannot vouch for is what keeps it a cache and not a guesser.
    """
    from ..knowledge.relevance import _content_words

    asked = _content_words(new_question) - _QUESTION_NOISE
    if not asked:
        return False
    return asked <= _content_words(cached_question)


@dataclass(frozen=True)
class Nominee:
    """A cached answer worth showing to the gate, and why."""

    cache_key: str
    similarity: float


class SemanticIndex:
    """Question vectors over the answer cache, brute-force and deterministic.

    Brute force is right here by a wide margin: one vector per *cached
    answer*, so even a busy deployment holds a few thousand rows - a single
    numpy matrix-vector product. An approximate index would add a dependency
    and a source of nondeterminism to search a list that fits in L2 cache.
    """

    def __init__(self, store: object, embedder: Embedder) -> None:
        # Duck-typed store: anything with the AnswerStore's connection/lock.
        self._store = store
        self._embedder = embedder
        conn = getattr(store, "_conn")  # noqa: B009 - one component, split for clarity
        with getattr(store, "_lock"):  # noqa: B009
            conn.executescript(_SCHEMA)
            conn.commit()

    def embed(self, question: str) -> list[float] | None:
        """The question's vector, or None when the endpoint cannot answer."""
        try:
            return normalise(self._embedder.embed_query(question))
        except (EmbeddingError, IndexError) as exc:
            log.debug("semantic cache: could not embed the question: %s", exc)
            return None

    def nominate(
        self, vector: list[float], corpus_version: str, *, threshold: float
    ) -> Nominee | None:
        """The nearest cached question under this corpus, if near enough.

        The threshold is a candidate-finder, not a safety judgement - the gate
        is the safety judgement. Ties break on cache_key so the same question
        always nominates the same answer.
        """
        conn = getattr(self._store, "_conn")  # noqa: B009
        with getattr(self._store, "_lock"):  # noqa: B009
            rows = conn.execute(
                "SELECT cache_key, value FROM question_vectors WHERE corpus_version = ?",
                (corpus_version,),
            ).fetchall()
        if not rows:
            return None

        import numpy as np

        keys = [row[0] for row in rows]
        matrix = np.frombuffer(b"".join(row[1] for row in rows), dtype=np.float32).reshape(
            len(rows), -1
        )
        scores = matrix @ np.asarray(vector, dtype=np.float32)
        order = np.lexsort((np.asarray(keys), -scores))
        best = int(order[0])
        if float(scores[best]) < threshold:
            return None
        return Nominee(cache_key=keys[best], similarity=float(scores[best]))

    def remember(self, question: str, cache_key: str, corpus_version: str) -> None:
        """Store this question's vector beside the answer it produced.

        Failing to embed is a lost future shortcut, never a lost answer, so it
        is logged and swallowed.
        """
        vector = self.embed(question)
        if vector is None:
            return
        blob = array.array("f", vector).tobytes()
        conn = getattr(self._store, "_conn")  # noqa: B009
        with getattr(self._store, "_lock"):  # noqa: B009
            conn.execute(
                "INSERT OR REPLACE INTO question_vectors (cache_key, corpus_version, value)"
                " VALUES (?, ?, ?)",
                (cache_key, corpus_version, blob),
            )
            conn.commit()

    def evict_other_corpus_versions(self, current: str) -> int:
        conn = getattr(self._store, "_conn")  # noqa: B009
        with getattr(self._store, "_lock"):  # noqa: B009
            cur = conn.execute("DELETE FROM question_vectors WHERE corpus_version != ?", (current,))
            conn.commit()
        return cur.rowcount
