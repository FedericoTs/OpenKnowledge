"""How long the map takes at a size a real company has.

The layout is an all-pairs force simulation, so its cost grows with the
square of the document count; a number measured at four documents says
nothing about a thousand. This builds a synthetic corpus of the requested
size with the relations the map draws - contradictions, citations, gaps -
and times each step. The picture itself is not judged here; only whether
/manage would wait on it.

    uv run python tools/measure_graph.py --documents 1000 --json
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass

from openknowledge.graph import from_engine, layout, render_svg
from openknowledge.retrieval.base import Document

FOLDERS = ("hr", "finance", "security", "legal", "it", "ops", "sales", "archive")


@dataclass(frozen=True)
class _Conflict:
    left_document: str
    right_document: str


def synthetic(documents: int, seed: int = 7) -> dict[str, object]:
    rng = random.Random(seed)
    ids = [f"d{i}" for i in range(documents)]
    docs = [
        Document(
            document_id=ident,
            title=f"Document {i}",
            text="x",
            url=f"file:///corpus/{FOLDERS[i % len(FOLDERS)]}/{ident}.md",
            superseded=(i % 37 == 0),
        )
        for i, ident in enumerate(ids)
    ]
    conflicts = [_Conflict(*rng.sample(ids, 2)) for _ in range(max(documents * 3 // 10, 1))]
    citations = [tuple(rng.sample(ids, rng.randint(1, 3))) for _ in range(documents * 2)]
    gaps = [{"question": f"question {i}", "asked": rng.randint(1, 20)} for i in range(30)]
    return {"documents": docs, "conflicts": conflicts, "citations": citations, "gaps": gaps}


def measure(documents: int) -> dict[str, object]:
    data = synthetic(documents)
    started = time.perf_counter()
    graph = from_engine(
        data["documents"],  # type: ignore[arg-type]
        root="/corpus",
        conflicts=data["conflicts"],  # type: ignore[arg-type]
        citations=data["citations"],  # type: ignore[arg-type]
        gaps=data["gaps"],  # type: ignore[arg-type]
        viewer=None,
    )
    built = time.perf_counter()
    positions = layout(graph)
    laid_out = time.perf_counter()
    svg = render_svg(graph, positions)
    rendered = time.perf_counter()
    return {
        "documents": documents,
        "nodes": len(graph.nodes),
        "edges": len(graph.edges),
        "build_seconds": round(built - started, 3),
        "layout_seconds": round(laid_out - built, 3),
        "render_seconds": round(rendered - laid_out, 3),
        "total_seconds": round(rendered - started, 3),
        "svg_kilobytes": round(len(svg.encode()) / 1000, 1),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--documents", type=int, nargs="+", default=[100, 1000])
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    results = [measure(n) for n in args.documents]
    if args.json:
        print(json.dumps(results, indent=2))
    else:
        for r in results:
            print(
                f"{r['documents']:>5} documents, {r['edges']:>5} edges: "
                f"build {r['build_seconds']}s, layout {r['layout_seconds']}s, "
                f"render {r['render_seconds']}s, svg {r['svg_kilobytes']} kB"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
