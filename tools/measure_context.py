#!/usr/bin/env python3
"""What the heading trail inside an embedded passage is worth.

    uv run python tools/measure_context.py --corpus evals/corpus/aveline \
        --cases evals/golden-aveline/aveline.yaml \
        --embedding-url http://127.0.0.1:8082/v1

The chunker states each passage's heading trail once, at the top of the
passage (see `_passage` in retrieval/base.py), so the trail is already part of
what gets embedded - this is the "contextual chunk embedding" idea, and it is
built. What was never established is whether it earns the space it takes.

This measures that directly: the rank of each labelled case's required
citation, with the trail in the embedded text and with it removed, by cosine
alone. Dense-only on purpose - BM25 does not care where in the passage a word
sits, and fusing the two hides the effect being measured behind a lexical
match that would have found the document anyway.

What it cannot tell you is whether answers improve. That needs the golden set
and a live model. This is a retrieval number: it shows whether the trail moves
the right passage up the list.

**A null result here is about the corpus as much as the trail.** On a corpus
small enough that every required document already ranks first, nothing can
move, and the honest report is that this corpus cannot measure it.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

import yaml

from openknowledge.connectors import LocalFilesConnector
from openknowledge.retrieval.base import Chunk, chunk_document
from openknowledge.retrieval.embed import Embedder


def without_trail(chunk: Chunk) -> str:
    """The passage as it would read if the trail had never been prepended.

    Exactly the line the chunker wrote, never "whatever the first line is": a
    passage may legitimately open with a line ending in a colon, and stripping
    that would remove content and measure something else.
    """
    prefix = f"{chunk.section}:" if chunk.section else ""
    if prefix and chunk.text.startswith(prefix):
        return chunk.text[len(prefix) :].lstrip("\n")
    return chunk.text


def _unit(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vector)) or 1.0
    return [x / norm for x in vector]


def measure(corpus: Path, cases_path: Path, embedder: Embedder) -> dict[str, Any]:
    documents = LocalFilesConnector(corpus).fetch()
    chunks = [c for d in documents for c in chunk_document(d)]
    cases = [c for c in yaml.safe_load(cases_path.read_text()) if c.get("must_cite")]
    if not chunks or not cases:
        raise SystemExit("nothing to measure: no chunks, or no case carries must_cite")

    variants = {
        "with_trail": [c.text for c in chunks],
        "trail_stripped": [without_trail(c) for c in chunks],
    }
    differing = sum(
        1 for a, b in zip(variants["with_trail"], variants["trail_stripped"], strict=True) if a != b
    )
    if not differing:
        raise SystemExit(
            "the two variants are identical, so this would measure nothing - "
            "no passage in this corpus carries a heading trail"
        )

    vectors = {n: [_unit(v) for v in embedder.embed_documents(t)] for n, t in variants.items()}
    queries = {c["id"]: _unit(embedder.embed_query(c["question"])) for c in cases}

    out: dict[str, Any] = {
        "chunks": len(chunks),
        "cases": len(cases),
        "chunks_carrying_a_trail": differing,
        "share_of_embedded_words_that_are_the_trail": round(
            1
            - sum(len(t.split()) for t in variants["trail_stripped"])
            / sum(len(t.split()) for t in variants["with_trail"]),
            4,
        ),
        "ranks": {},
    }
    for name, vecs in vectors.items():
        ranks = {}
        for case in cases:
            query = queries[case["id"]]
            order = sorted(
                (
                    (sum(a * b for a, b in zip(query, vecs[i], strict=True)), i)
                    for i in range(len(chunks))
                ),
                reverse=True,
            )
            wanted = set(case["must_cite"])
            ranks[case["id"]] = next(
                (p for p, (_s, i) in enumerate(order, 1) if chunks[i].document_id in wanted),
                len(chunks),
            )
        out["ranks"][name] = ranks
        values = list(ranks.values())
        out[name] = {
            "median_rank": statistics.median(values),
            "mean_rank": round(statistics.mean(values), 3),
            "top_1": sum(1 for v in values if v == 1),
            "top_3": sum(1 for v in values if v <= 3),
        }

    before, after = out["ranks"]["trail_stripped"], out["ranks"]["with_trail"]
    moved = {k: [before[k], after[k]] for k in after if before[k] != after[k]}
    out["moved"] = moved
    out["better"] = sum(1 for w, n in moved.values() if n < w)
    out["worse"] = sum(1 for w, n in moved.values() if n > w)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--embedding-url", default="http://localhost:11434/v1")
    parser.add_argument("--embedding-model", default="nomic-embed-text")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    result = measure(
        args.corpus,
        args.cases,
        Embedder(model=args.embedding_model, base_url=args.embedding_url, api_key=None),
    )
    if args.json:
        print(json.dumps(result, indent=2))
        return 0

    share = result["share_of_embedded_words_that_are_the_trail"]
    print(
        f"{result['chunks']} chunks, {result['cases']} cases with a required citation; "
        f"{share:.1%} of the embedded words are the trail\n"
    )
    for name in ("with_trail", "trail_stripped"):
        s = result[name]
        print(
            f"  {name.replace('_', ' '):16} median {s['median_rank']:4.1f}  "
            f"mean {s['mean_rank']:5.2f}  top-1 {s['top_1']:2d}  top-3 {s['top_3']:2d}"
        )
    print()
    for case_id, (was, now) in sorted(result["moved"].items(), key=lambda kv: kv[1][1] - kv[1][0]):
        print(f"  {case_id:34} {was:3d} -> {now:3d}  {'better' if now < was else 'WORSE'}")
    if not result["moved"]:
        print("  no case changed rank - on this corpus the trail is worth nothing measurable,")
        print("  which is a fact about the corpus as much as about the trail.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
