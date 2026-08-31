#!/usr/bin/env python3
"""Measure what a question actually costs to ask, on a real corpus.

`tools/cost_model.py` prices the architecture from assumed token counts. This
prices it from *measured* ones: it parses a real folder, retrieves for real
questions, assembles the exact prompt that would be sent, and counts it.

    uv run python tools/measure_prompts.py --corpus ./policies --questions qs.txt

Why this exists. The claim this project rests on is that $0.10 per question is
not a law of nature, it is what you pay for sending 15,000 tokens to a frontier
model on every call. That is an arithmetic claim until somebody checks what a
real corpus actually produces, and the retrieved-context term is around three
quarters of the bill.

What is measured and what is not:

* **Input tokens are measured.** The prompt is assembled by the same code the
  running system uses - same system prompt, same SOURCES formatting, same
  retrieval, same chunks. Nothing here is an estimate of the input side.
* **Output tokens are assumed** (`--output-tokens`, default 1,000). No model is
  called, so nothing here can know how long an answer would be. It is the same
  assumption for every row, so it cancels out of every comparison, and it is
  named in the output rather than buried.
* **Token counts use `cl100k_base`**, an OpenAI BPE, as a vendor-neutral
  counter. Anthropic's tokenizer differs on English prose by a few percent.
  That matters for an invoice and not for a comparison between rows measured
  the same way.

The rows are retrieval disciplines, not vendors. Everything else is held
constant, so the difference between the top and the bottom of the table is the
thing this project is actually about.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

from openknowledge.connectors import LocalFilesConnector
from openknowledge.costs import Usage, cost_usd, get_price
from openknowledge.prompts import SYSTEM_PROMPT, format_context
from openknowledge.providers.anthropic_provider import CACHE_MIN_TOKENS
from openknowledge.retrieval import BM25Retriever
from openknowledge.retrieval.base import Chunk

WORKING_DAYS = 250


@dataclass(frozen=True)
class Discipline:
    """One way of deciding how much context to send."""

    name: str
    #: Chunks retrieved per question; None means the entire corpus.
    k: int | None
    note: str


DISCIPLINES = [
    Discipline("Whole corpus in context", None, "no retrieval at all"),
    Discipline("Top 40 chunks", 40, "retrieval, but keep everything that scored"),
    Discipline("Top 20 chunks", 20, "the usual 'be generous' default"),
    Discipline("Top 10 chunks", 10, "tightening"),
    Discipline("Top 6 chunks", 6, "OpenKnowledge default"),
]


def count_tokens(text: str) -> int:
    try:
        import tiktoken
    except ImportError:  # pragma: no cover - dev extra
        sys.exit(
            "tiktoken is needed to count tokens: uv pip install 'openknowledge[measure]'\n"
            "It is a dev extra on purpose - the running system takes token counts from "
            "the provider's own usage response, never from an estimate."
        )
    return len(tiktoken.get_encoding("cl100k_base").encode(text))


def prompt_for(question: str, chunks: list[Chunk]) -> str:
    """The exact text that would be sent, system prompt included."""
    return f"{SYSTEM_PROMPT}\n\n{format_context(chunks)}\n\nQUESTION: {question}"


def _levers(
    *,
    measured_input: int,
    system_tokens: int,
    output_tokens: int,
    mid_tier: str,
    frontier: str,
    per_day: int,
    free_shares: tuple[float, float],
) -> list[dict[str, object]]:
    """Price the remaining levers on top of the measured prompt.

    Each row adds one thing to the row above it, so the reader can see which
    lever did the work rather than being handed a single ratio.
    """
    # The API silently declines to cache a prefix below its floor, so a system
    # prompt under it earns nothing however the cache_control marker is placed.
    # Measured rather than assumed, because this is exactly the kind of lever a
    # cost model claims and a deployment never receives.
    caches = system_tokens >= CACHE_MIN_TOKENS
    cached = system_tokens if caches else 0
    variable = max(measured_input - cached, 0)
    caching_label = (
        "+ prompt caching (static system prompt)"
        if caches
        else f"+ prompt caching (inert: {system_tokens} < {CACHE_MIN_TOKENS} floor)"
    )
    rows: list[tuple[str, Usage, str]] = [
        (
            "measured prompt, frontier model",
            Usage(input_tokens=measured_input, output_tokens=output_tokens),
            frontier,
        ),
        (
            caching_label,
            Usage(
                input_tokens=variable,
                cache_read_tokens=cached,
                output_tokens=output_tokens,
            ),
            frontier,
        ),
        (
            "+ mid-tier model",
            Usage(
                input_tokens=variable,
                cache_read_tokens=cached,
                output_tokens=output_tokens,
            ),
            mid_tier,
        ),
    ]

    out: list[dict[str, object]] = []
    paid = 1.0
    for name, usage, model in rows:
        per_call = cost_usd(usage, get_price(model))
        out.append(
            {
                "name": name,
                "usd_per_question": round(per_call * paid, 5),
                "usd_per_year": round(per_call * paid * per_day * WORKING_DAYS, 2),
            }
        )

    _, last_usage, last_model = rows[-1]
    per_call = cost_usd(last_usage, get_price(last_model))
    for share in free_shares:
        out.append(
            {
                "name": f"+ {share:.0%} of questions never reach a model",
                "usd_per_question": round(per_call * (1 - share), 5),
                "usd_per_year": round(per_call * (1 - share) * per_day * WORKING_DAYS, 2),
            }
        )
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True, help="folder of real documents")
    parser.add_argument("--questions", required=True, help="one question per line")
    parser.add_argument("--model", default="claude-opus-5", help="rate to price against")
    parser.add_argument("--output-tokens", type=int, default=1_000)
    parser.add_argument("--per-day", type=int, default=2_000)
    parser.add_argument("--pdf-backend", default="auto")
    parser.add_argument("--mid-tier-model", default="claude-sonnet-5")
    parser.add_argument(
        "--free-share",
        type=float,
        default=0.45,
        help="share of questions a pin, cache hit or draft answers without a model",
    )
    parser.add_argument("--free-share-grown", type=float, default=0.75)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    questions = [
        line.strip()
        for line in Path(args.questions).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    if not questions:
        sys.exit(f"no questions in {args.questions}")

    connector = LocalFilesConnector(args.corpus, pdf_backend=args.pdf_backend)
    documents = connector.fetch()
    if not documents:
        sys.exit(f"no readable documents in {args.corpus}")

    retriever = BM25Retriever()
    retriever.index(documents)
    every_chunk = list(retriever.chunks)
    price = get_price(args.model)

    rows = []
    for discipline in DISCIPLINES:
        totals = []
        for question in questions:
            chunks = (
                every_chunk
                if discipline.k is None
                else [hit.chunk for hit in retriever.search(question, k=discipline.k)]
            )
            totals.append(count_tokens(prompt_for(question, chunks)))

        mean_in = sum(totals) // len(totals)
        usage = Usage(input_tokens=mean_in, output_tokens=args.output_tokens)
        per_call = cost_usd(usage, price)
        rows.append(
            {
                "discipline": discipline.name,
                "note": discipline.note,
                "chunks": discipline.k or len(every_chunk),
                "input_tokens": mean_in,
                "min_input_tokens": min(totals),
                "max_input_tokens": max(totals),
                "usd_per_question": round(per_call, 5),
                "usd_per_year": round(per_call * args.per_day * WORKING_DAYS, 2),
            }
        )

    tight = next(r for r in rows if r["chunks"] == DISCIPLINES[-1].k)
    payload = {
        "levers": _levers(
            measured_input=int(tight["input_tokens"]),
            system_tokens=count_tokens(SYSTEM_PROMPT),
            output_tokens=args.output_tokens,
            mid_tier=args.mid_tier_model,
            frontier=args.model,
            per_day=args.per_day,
            free_shares=(args.free_share, args.free_share_grown),
        ),
        "system_prompt_tokens": count_tokens(SYSTEM_PROMPT),
        "corpus": str(connector.root),
        "documents": len(documents),
        "chunks": len(every_chunk),
        "questions": len(questions),
        "model": args.model,
        "output_tokens_assumed": args.output_tokens,
        "questions_per_day": args.per_day,
        "rows": rows,
    }

    if args.json:
        print(json.dumps(payload, indent=2))
        return 0

    print(f"Corpus     {connector.root}")
    print(f"           {len(documents)} documents, {len(every_chunk)} chunks")
    print(f"Questions  {len(questions)}")
    print(f"Priced at  {args.model} (${price.input_per_mtok}/${price.output_per_mtok} per MTok)")
    print(f"Assumed    {args.output_tokens:,} output tokens per answer - the only estimate here")
    print()
    header = f"{'retrieval discipline':<28}{'in tokens':>11}{'range':>16}"
    print(f"{header}{'$/question':>12}{'$/year':>12}")
    for row in rows:
        span = f"{row['min_input_tokens']:,}-{row['max_input_tokens']:,}"
        print(
            f"{row['discipline']:<28}{row['input_tokens']:>11,}{span:>16}"
            f"{row['usd_per_question']:>12.5f}{row['usd_per_year']:>12,.0f}"
        )
    print()
    print(f"at {args.per_day:,} questions/day x {WORKING_DAYS} working days, paid tier only.")
    print("Rows differ in one thing: how many chunks were sent. Same corpus, same")
    print("prompt, same model, same assumed answer length.")

    print()
    print("-- and then, at the measured prompt size " + "-" * 36)
    print()
    width = max(len(str(lever["name"])) for lever in payload["levers"]) + 2
    print(f"{'lever':<{width}}{'$/question':>12}{'$/year':>12}")
    for lever in payload["levers"]:
        print(
            f"{lever['name']!s:<{width}}{lever['usd_per_question']:>12.5f}"
            f"{lever['usd_per_year']:>12,.0f}"
        )
    print()
    system_tokens = int(payload["system_prompt_tokens"])
    if system_tokens < CACHE_MIN_TOKENS:
        print(
            f"NOTE: the system prompt measures {system_tokens:,} tokens, under the "
            f"{CACHE_MIN_TOKENS}-token"
        )
        print("minimum cacheable prefix. The cache_control marker on it is inert, so")
        print("prompt caching is not a lever at this prompt size and the row above is")
        print("flat on purpose. Retrieval discipline and the free tier are the levers.")
    else:
        print(
            f"The system prompt is {system_tokens:,} tokens and never changes, so it caches; "
            "the SOURCES block does not."
        )
    print()
    print("Free share is the only assumed figure in this second table - it is what pins,")
    print("cache hits and drafted answers cover, and `openknowledge costs` reports yours")
    print("from the ledger rather than from here.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
