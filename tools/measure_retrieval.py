#!/usr/bin/env python3
"""What reranking recovers, measured on a real corpus.

    uv run python tools/measure_retrieval.py --corpus ./policies --questions qs.txt

BM25 scores every chunk on its own, so its top *k* can be six views of one
paragraph and still look excellent. This measures three things that go wrong as
a result, all of them recall failures that end in an escalation:

* **distinct documents** in the context - a question whose answer spans two
  policies needs both of them to survive the cut;
* **slots taken by the single most dominant document** - how badly one verbose
  file crowds out the rest;
* **near-duplicate pairs** - the chunker overlaps windows by design, so adjacent
  chunks share text and two slots can carry one fact.

What this does *not* measure is whether answers get better. That needs a
labelled set and a live model. These are coverage numbers: they show the
reranker does what it claims, not that the claim was worth making.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

from openknowledge.connectors import LocalFilesConnector
from openknowledge.retrieval import BM25Retriever
from openknowledge.retrieval.base import ScoredChunk, tokenize
from openknowledge.retrieval.rerank import StructuralReranker


@dataclass(frozen=True)
class Coverage:
    distinct_documents: float
    dominant_document_slots: float
    near_duplicate_pairs: float


def _jaccard(a: str, b: str) -> float:
    left, right = frozenset(tokenize(a)), frozenset(tokenize(b))
    return len(left & right) / len(left | right) if left and right else 0.0


def coverage(batches: list[list[ScoredChunk]], *, threshold: float) -> Coverage:
    distinct = dominant = duplicates = 0
    for hits in batches:
        chunks = [h.chunk for h in hits]
        per_document: dict[str, int] = {}
        for chunk in chunks:
            per_document[chunk.document_id] = per_document.get(chunk.document_id, 0) + 1
        distinct += len(per_document)
        dominant += max(per_document.values(), default=0)
        duplicates += sum(
            1
            for i in range(len(chunks))
            for j in range(i + 1, len(chunks))
            if _jaccard(chunks[i].text, chunks[j].text) >= threshold
        )
    n = max(len(batches), 1)
    return Coverage(distinct / n, dominant / n, duplicates / n)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--questions", required=True)
    parser.add_argument("-k", type=int, default=6, help="chunks sent to the model")
    parser.add_argument("--candidates", type=int, default=30, help="chunks the reranker picks from")
    parser.add_argument("--max-per-document", type=int, default=2)
    parser.add_argument("--pdf-backend", default="auto")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    questions = [
        line.strip()
        for line in Path(args.questions).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    documents = LocalFilesConnector(args.corpus, pdf_backend=args.pdf_backend).fetch()
    if not documents or not questions:
        sys.exit("need both readable documents and questions")

    retriever = BM25Retriever()
    retriever.index(documents)
    reranker = StructuralReranker(max_per_document=args.max_per_document)

    raw = [retriever.search(q, k=args.candidates) for q in questions]
    plain = [hits[: args.k] for hits in raw]
    reranked = [reranker.rerank(q, hits, k=args.k) for q, hits in zip(questions, raw, strict=True)]

    before = coverage(plain, threshold=reranker.redundancy_threshold)
    after = coverage(reranked, threshold=reranker.redundancy_threshold)

    payload = {
        "documents": len(documents),
        "chunks": len(retriever),
        "questions": len(questions),
        "k": args.k,
        "candidates": args.candidates,
        "max_per_document": args.max_per_document,
        "bm25": vars(before),
        "reranked": vars(after),
    }
    if args.json:
        print(json.dumps(payload, indent=2))
        return 0

    print(f"{len(documents)} documents, {len(retriever)} chunks, {len(questions)} questions")
    print(
        f"top {args.k} of {args.candidates} candidates, at most {args.max_per_document}/document\n"
    )
    print(f"{'':<22}{'distinct docs':>15}{'top doc slots':>15}{'near-dupe pairs':>17}")
    for label, c in (("BM25 top-k", before), ("+ structural rerank", after)):
        print(
            f"{label:<22}{c.distinct_documents:>15.2f}"
            f"{c.dominant_document_slots:>15.2f}{c.near_duplicate_pairs:>17.2f}"
        )
    print("\nCoverage, not accuracy: this shows the reranker does what it claims,")
    print("not that answers improved. That needs a labelled set and a live model.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
