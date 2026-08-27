"""Dense retrieval: the half of hybrid search that understands paraphrase.

BM25 matches words. Someone who asks "how much can I spend on dinner?" against
a document that says "meals are reimbursed up to EUR 45 per day" shares almost
no vocabulary with it, and gets nothing - which reads as the system not knowing
something it plainly does. That gap is felt most by exactly the casual phrasings
a chat box invites, and it is the reason a keyword index alone is not a chatbot.

What this is not: a replacement for BM25. Lexical search is the stronger half
for the things this workload is full of - EUR 500, form RA-14, GlobalProtect -
where an embedding blurs a specific identifier toward its neighbours. The two
fail in different directions, which is the whole argument for running both.

Three properties this has to keep, because the rest of the system rests on them:

* **Deterministic.** A fixed model over fixed text returns fixed vectors, and
  the fusion below is rank-based arithmetic. Identical questions retrieve
  identical context, without which the answer cache is unsound.
* **Free.** The embedding model runs on the same local endpoint as the chat
  model, so this adds no per-token invoice.
* **Optional.** No embedding endpoint, no embedding model, a runtime that is
  down - all fall back to BM25 alone and say so. Retrieval degrades; it never
  fails.
"""

from __future__ import annotations

import hashlib
import logging
import math
from dataclasses import dataclass

import httpx

log = logging.getLogger(__name__)

#: Small, fast, and made for retrieval rather than chat. 274 MB.
DEFAULT_MODEL = "nomic-embed-text"

#: Models trained with task prefixes, and what they expect. Getting this wrong
#: is silent and expensive: measured on ten real policy documents, nomic scored
#: 7/8 on paraphrased questions without its prefixes and 8/8 with them. Nothing
#: errors, the vectors just come out of a slightly different space than the one
#: the model was trained to put them in.
PREFIXES: dict[str, tuple[str, str]] = {
    # model-name substring -> (document prefix, query prefix)
    "nomic-embed": ("search_document: ", "search_query: "),
    "e5-": ("passage: ", "query: "),
    "multilingual-e5": ("passage: ", "query: "),
    "bge-": ("", "Represent this sentence for searching relevant passages: "),
}


def prefixes_for(model: str) -> tuple[str, str]:
    """The (document, query) prefixes ``model`` was trained with, if any.

    Matched on the name because that is what an operator sets and what the
    runtime reports. An unknown model gets no prefixes, which is right far more
    often than guessing at one.
    """
    name = model.lower()
    for marker, pair in PREFIXES.items():
        if marker in name:
            return pair
    return ("", "")


#: Ollama takes a list, so chunks go up in batches rather than one call each.
_BATCH = 64


class EmbeddingError(RuntimeError):
    """The endpoint could not embed. Callers fall back rather than fail."""


@dataclass
class Embedder:
    """Vectors from an OpenAI-compatible or Ollama endpoint.

    Ollama's own ``/api/embed`` is preferred where it exists because it batches;
    the OpenAI-compatible ``/v1/embeddings`` is the fallback, which vLLM,
    LM Studio and llama.cpp all serve.
    """

    model: str = DEFAULT_MODEL
    base_url: str = "http://localhost:11434/v1"
    api_key: str | None = None
    timeout: float = 120.0
    #: Set from the model name unless given. Asymmetric on purpose: these models
    #: are trained to put a question and the passage answering it in different
    #: places, and swapping the two is worse than using neither.
    document_prefix: str | None = None
    query_prefix: str | None = None

    def __post_init__(self) -> None:
        document, query = prefixes_for(self.model)
        if self.document_prefix is None:
            self.document_prefix = document
        if self.query_prefix is None:
            self.query_prefix = query

    @property
    def _root(self) -> str:
        trimmed = self.base_url.rstrip("/")
        return trimmed[: -len("/v1")] if trimmed.endswith("/v1") else trimmed

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Vectors for passages being indexed."""
        return self.embed([f"{self.document_prefix}{text}" for text in texts])

    def embed_query(self, text: str) -> list[float]:
        """A vector for a question. Not the same prefix as a passage."""
        return self.embed([f"{self.query_prefix}{text}"])[0]

    @property
    def fingerprint(self) -> str:
        """Identifies the vector space, so a model change invalidates the cache.

        Vectors from two models are not comparable, and a corpus half-embedded
        by each would rank by nothing at all. This goes into `corpus_version`.
        """
        return f"{self.model}@{self._root}|{self.document_prefix}"

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Vectors for ``texts``, in order. Raises rather than returning junk."""
        if not texts:
            return []
        out: list[list[float]] = []
        for start in range(0, len(texts), _BATCH):
            out.extend(self._batch(texts[start : start + _BATCH]))
        return out

    def _batch(self, texts: list[str]) -> list[list[float]]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(
                    f"{self._root}/api/embed",
                    json={"model": self.model, "input": texts},
                    headers=headers,
                )
                if response.status_code == 404:
                    return self._openai_batch(client, texts, headers)
                response.raise_for_status()
                vectors = response.json().get("embeddings")
        except (httpx.HTTPError, ValueError) as exc:
            raise EmbeddingError(f"{self.model}: {exc or type(exc).__name__}") from exc

        if not isinstance(vectors, list) or len(vectors) != len(texts):
            raise EmbeddingError(
                f"{self.model}: expected {len(texts)} vectors, got {str(vectors)[:80]!r}"
            )
        return [[float(x) for x in vector] for vector in vectors]

    def _openai_batch(
        self, client: httpx.Client, texts: list[str], headers: dict[str, str]
    ) -> list[list[float]]:
        """For runtimes that serve /v1/embeddings and not Ollama's endpoint."""
        response = client.post(
            f"{self.base_url.rstrip('/')}/embeddings",
            json={"model": self.model, "input": texts},
            headers=headers,
        )
        response.raise_for_status()
        rows = sorted(response.json()["data"], key=lambda row: row.get("index", 0))
        if len(rows) != len(texts):
            raise EmbeddingError(f"{self.model}: expected {len(texts)} vectors, got {len(rows)}")
        return [[float(x) for x in row["embedding"]] for row in rows]


def normalise(vector: list[float]) -> list[float]:
    """Unit length, so cosine similarity is a dot product."""
    length = math.sqrt(sum(x * x for x in vector))
    return [x / length for x in vector] if length else vector


def text_key(text: str, fingerprint: str) -> str:
    """Cache key for one chunk's vector under one model.

    Keyed on the text rather than the chunk id: re-indexing renumbers chunks but
    leaves most of their text identical, and re-embedding an unchanged paragraph
    is the slowest pointless thing this system could do.
    """
    digest = hashlib.sha256()
    digest.update(fingerprint.encode("utf-8"))
    digest.update(b"\0")
    digest.update(text.encode("utf-8"))
    return digest.hexdigest()
