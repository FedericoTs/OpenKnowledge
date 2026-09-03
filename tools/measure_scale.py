"""How long indexing and retrieval take as a corpus grows.

Every accuracy number in this repository was measured on a corpus of twenty
documents. A company has thousands, and nothing here has ever said what happens
then - not "it is probably fine", nothing at all. This measures it.

The corpus is grown from the Federal Travel Regulation set rather than from
lorem ipsum: real sentence lengths, real heading trails, real near-miss
vocabulary between neighbouring sections, which is what a retriever actually
has to separate. Each copy is given a distinct organisation name and section
numbering so that the index is not N identical documents - BM25 on duplicates
is a much easier problem than BM25 on a real corpus, and would flatter the
result.

No model is called. This measures the free half of the pipeline: reading the
corpus, chunking it, building the index, and answering a retrieval query. That
is deliberate - it is the half whose cost grows with the corpus, and it can be
measured exactly, on this machine, without a GPU or a network.

    uv run python tools/measure_scale.py --sizes 20 200 1000 --queries 25
"""

from __future__ import annotations

import argparse
import gc
import json
import resource
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "evals/golden-ftr/documents"

#: Real questions, not keyword soup: these are the phrasings the FTR eval uses,
#: so the retrieval work is representative of a question somebody would ask.
QUERIES = [
    "What is the M&IE reimbursement rate during the first and last travel day?",
    "Do I need to provide receipts?",
    "Are lodging taxes included in the per diem rate?",
    "What is the difference between per diem and actual expenses?",
    "When may I use actual expense reimbursement?",
    "How is mileage reimbursed for a privately owned vehicle?",
    "What happens if my travel is interrupted by illness?",
    "Who authorises premium class travel?",
    "May I keep frequent flyer miles earned on official travel?",
    "What are the rules for a rental car on official travel?",
]


def _grow(target: int, into: Path) -> int:
    """Write ``target`` documents into ``into``, derived from the FTR corpus."""
    into.mkdir(parents=True, exist_ok=True)
    for stale in into.glob("*.md"):
        stale.unlink()
    originals = sorted(SOURCE.glob("*.md"))
    if not originals:  # pragma: no cover - the corpus is committed
        raise SystemExit(f"no documents in {SOURCE}")
    written = 0
    generation = 0
    while written < target:
        generation += 1
        for original in originals:
            if written >= target:
                break
            text = original.read_text(encoding="utf-8")
            if generation > 1:
                # A distinct organisation and a distinct section prefix, so the
                # copies compete on vocabulary the way real neighbours do
                # rather than being detected as duplicates.
                org = f"Region {generation}"
                text = text.replace("Federal", f"{org} Federal")
                text = text.replace("§ 3", f"§ {generation}3")
                text = f"# {org} — {original.stem}\n\n{text}"
            name = original.stem if generation == 1 else f"{original.stem}-r{generation}"
            (into / f"{name}.md").write_text(text, encoding="utf-8")
            written += 1
    return written


def _peak_mb() -> float:
    # ru_maxrss is kilobytes on Linux.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def measure(size: int, corpus: Path, state: Path, queries: int) -> dict[str, object]:
    from openknowledge.api.engine import build_engine
    from openknowledge.config import Settings

    _grow(size, corpus)
    for stale in sorted(state.rglob("*"), reverse=True):
        stale.unlink() if stale.is_file() else stale.rmdir()
    state.mkdir(parents=True, exist_ok=True)

    settings = Settings(
        documents_dir=str(corpus),
        data_dir=str(state),
        embedding_enabled=False,
        local_enabled=False,
        escalation_enabled=False,
    )
    engine = build_engine(settings)

    gc.collect()
    started = time.perf_counter()
    documents, chunks, _version, _evicted = engine.reindex()
    index_seconds = time.perf_counter() - started

    latencies: list[float] = []
    for i in range(queries):
        query = QUERIES[i % len(QUERIES)]
        began = time.perf_counter()
        engine.retriever.search(query, k=settings.retrieval_k)
        latencies.append((time.perf_counter() - began) * 1000)

    latencies.sort()
    return {
        "documents": documents,
        "chunks": chunks,
        "index_seconds": round(index_seconds, 2),
        "documents_per_second": round(documents / index_seconds, 1) if index_seconds else None,
        "query_ms_median": round(statistics.median(latencies), 1),
        "query_ms_p95": round(latencies[min(len(latencies) - 1, int(len(latencies) * 0.95))], 1),
        "query_ms_max": round(latencies[-1], 1),
        "peak_rss_mb": round(_peak_mb(), 1),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=[20, 100, 500, 1000])
    parser.add_argument("--queries", type=int, default=25)
    parser.add_argument("--out", type=Path, help="write the rows as JSON here")
    parser.add_argument(
        "--workdir", type=Path, required=True, help="scratch directory for corpora and state"
    )
    args = parser.parse_args(argv)

    rows = []
    print(
        f"{'docs':>7} {'chunks':>8} {'index s':>9} {'docs/s':>8} "
        f"{'q ms p50':>9} {'q ms p95':>9} {'peak MB':>8}"
    )
    for size in args.sizes:
        row = measure(
            size, args.workdir / f"corpus-{size}", args.workdir / f"state-{size}", args.queries
        )
        rows.append(row)
        print(
            f"{row['documents']:>7} {row['chunks']:>8} {row['index_seconds']:>9} "
            f"{row['documents_per_second']:>8} {row['query_ms_median']:>9} "
            f"{row['query_ms_p95']:>9} {row['peak_rss_mb']:>8}"
        )
    if args.out:
        args.out.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
        print(f"\nwritten to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
