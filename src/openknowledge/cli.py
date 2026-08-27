"""Command line interface.

``openknowledge costs`` is the one that matters most: it reports what the bot
actually cost from the ledger, so the claim on the front page is something an
operator can check rather than take on faith.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from typing import Any

from .canonical import canonicalize_query
from .config import load_settings
from .costs import load_price_table
from .types import Tier


def _engine() -> Any:
    from .api.engine import build_engine

    return build_engine(load_settings())


def _cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    uvicorn.run("openknowledge.api.app:app", host=args.host, port=args.port, reload=args.reload)
    return 0


def _cmd_index(_: argparse.Namespace) -> int:
    engine = _engine()
    documents, chunks, version, evicted = engine.reindex()
    print(f"indexed {documents} documents -> {chunks} chunks")
    print(f"corpus version: {version}")
    if evicted:
        print(f"dropped {evicted} answers derived from a previous corpus")
    if documents == 0:
        print(
            f"\nNo documents found in {engine.settings.documents_dir!r}. "
            "Put .md or .txt files there, or set OK_DOCUMENTS_DIR.",
            file=sys.stderr,
        )
    return 0


def _cmd_ask(args: argparse.Namespace) -> int:
    engine = _engine()
    answer = asyncio.run(engine.cascade.answer(args.question, channel="cli"))

    print(answer.text)
    if answer.citations:
        print("\nSources:")
        for c in answer.citations:
            where = f" ({c.locator})" if c.locator else ""
            print(f"  [{c.document_id}] {c.document_title}{where}")

    price = "free" if answer.tier.is_cache_hit else f"${answer.cost_usd:.5f}"
    print(f"\n{answer.tier.value} · {answer.model_id} · {price}")
    for note in answer.notes:
        print(f"  note: {note}")
    return 0 if answer.tier is not Tier.REFUSED else 1


def _cmd_costs(args: argparse.Namespace) -> int:
    engine = _engine()
    report = engine.store.cost_report()

    if args.json:
        print(json.dumps(report, indent=2))
        return 0

    total = report["questions"]
    if not total:
        print("No questions answered yet - nothing to report.")
        return 0

    print(f"{total} questions · ${report['spend_usd']:.4f} total")
    print(f"blended cost: ${report['cost_per_question_usd']:.5f} per question\n")
    print(f"{'tier':<12}{'questions':>10}{'share':>8}{'spend':>12}")
    for tier, row in sorted(report["by_tier"].items(), key=lambda kv: -kv[1]["questions"]):
        share = row["questions"] / total
        print(f"{tier:<12}{row['questions']:>10}{share:>7.0%}${row['spend_usd']:>11.4f}")

    free = sum(r["questions"] for t, r in report["by_tier"].items() if Tier(t).is_cache_hit)
    print(f"\n{free / total:.0%} of questions were answered without calling a model.")
    return 0


def _cmd_top(args: argparse.Namespace) -> int:
    engine = _engine()
    rows = engine.store.top_questions(args.limit)
    if not rows:
        print("No questions answered yet.")
        return 0
    print("Most-asked questions - pinning these makes them free and identical every time:\n")
    for question, count in rows:
        print(f"{count:>5}x  {question}")
    return 0


def _cmd_pin(args: argparse.Namespace) -> int:
    engine = _engine()
    canonical = canonicalize_query(args.question)
    if not canonical:
        print("Question is empty after normalisation.", file=sys.stderr)
        return 2
    engine.store.pin(canonical, args.answer, author=args.author)
    print(f"pinned: {canonical}")
    return 0


def _cmd_unpin(args: argparse.Namespace) -> int:
    engine = _engine()
    canonical = canonicalize_query(args.question)
    if engine.store.unpin(canonical):
        print(f"removed pin: {canonical}")
        return 0
    print(f"no pin found for: {canonical}", file=sys.stderr)
    return 1


def _cmd_pricing(_: argparse.Namespace) -> int:
    print(f"{'model':<22}{'tier':<10}{'in $/M':>9}{'out $/M':>10}  verified")
    for price in load_price_table().values():
        if price.is_priced:
            print(
                f"{price.model_id:<22}{price.tier:<10}{price.input_per_mtok:>9.2f}"
                f"{price.output_per_mtok:>10.2f}  {price.verified}"
            )
        else:
            print(
                f"{price.model_id:<22}{price.tier:<10}{'-':>9}{'-':>10}  not set (see pricing.yaml)"
            )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="openknowledge", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("serve", help="run the HTTP server and chat widget")
    p.add_argument("--host", default="0.0.0.0")  # noqa: S104 - container default
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--reload", action="store_true")
    p.set_defaults(func=_cmd_serve)

    p = sub.add_parser("index", help="re-read the document folder")
    p.set_defaults(func=_cmd_index)

    p = sub.add_parser("ask", help="ask one question from the terminal")
    p.add_argument("question")
    p.set_defaults(func=_cmd_ask)

    p = sub.add_parser("costs", help="what the bot has actually cost")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=_cmd_costs)

    p = sub.add_parser("top", help="most-asked questions, i.e. what to pin")
    p.add_argument("--limit", type=int, default=20)
    p.set_defaults(func=_cmd_top)

    p = sub.add_parser("pin", help="set the canonical answer to a question")
    p.add_argument("question")
    p.add_argument("answer")
    p.add_argument("--author")
    p.set_defaults(func=_cmd_pin)

    p = sub.add_parser("unpin", help="remove a pinned answer")
    p.add_argument("question")
    p.set_defaults(func=_cmd_unpin)

    p = sub.add_parser("pricing", help="show the model price table")
    p.set_defaults(func=_cmd_pricing)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
