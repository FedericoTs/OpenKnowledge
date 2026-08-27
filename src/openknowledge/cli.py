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
from pathlib import Path
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
    from .cache import citations_for

    engine = _engine()
    phrasings = [args.question, *(args.alias or [])]
    canonicals = [c for c in (canonicalize_query(p) for p in phrasings) if c]
    if not canonicals:
        print("Question is empty after normalisation.", file=sys.stderr)
        return 2

    citations = citations_for(engine.retriever, tuple(args.cite or ()))
    unknown = [c.document_id for c in citations if "not currently in the indexed" in c.snippet]
    if unknown:
        print(
            f"warning: cited document(s) not in the corpus: {', '.join(unknown)}",
            file=sys.stderr,
        )

    for canonical in canonicals:
        engine.store.pin(canonical, args.answer, citations=citations, author=args.author)
        print(f"pinned: {canonical}")
    if citations:
        print(f"  sources: {', '.join(c.document_id for c in citations)}")
    return 0


def _cmd_unpin(args: argparse.Namespace) -> int:
    engine = _engine()
    canonical = canonicalize_query(args.question)
    if engine.store.unpin(canonical):
        print(f"removed pin: {canonical}")
        return 0
    print(f"no pin found for: {canonical}", file=sys.stderr)
    return 1


def _cmd_learn(args: argparse.Namespace) -> int:
    """Draft answers for changed documents and re-check approved ones."""
    engine = _engine()
    report = asyncio.run(engine.learn(max_documents=args.max_documents))

    print(report.summary())
    for note in report.notes:
        print(f"  {note}")

    if report.drafts_rejected:
        print(
            f"\n{report.drafts_rejected} drafted answers were discarded by the grounding "
            "gate before reaching the queue."
        )
    if report.needs_review:
        print(f"\n{report.needs_review} items need review:")
        if report.drafts_created:
            print(f"  openknowledge review      # {report.drafts_created} drafted answers")
        if report.conflicts_open:
            print(f"  openknowledge conflicts   # {report.conflicts_open} disagreements")
    return 0


def _cmd_review(args: argparse.Namespace) -> int:
    """Show drafted answers awaiting review, most valuable first."""
    from .knowledge import rank_by_demand

    engine = _engine()
    pending = engine.knowledge.pending(limit=args.limit)
    if not pending:
        print("Nothing waiting for review.")
        return 0

    demand = dict(engine.store.top_questions(limit=500))
    ranked = rank_by_demand(pending, demand=demand, cost_per_answer_usd=args.cost_per_answer)

    print(f"{len(ranked)} drafted answers awaiting review\n")
    for proposal, value in ranked:
        asked = demand.get(proposal.canonical_query, 0)
        worth = f"saves ~${value:.4f}/period" if value else "not asked yet"
        print(f"  {proposal.id}  [{worth}, asked {asked}x, support {proposal.support_ratio:.0%}]")
        print(f"    Q: {proposal.question}")
        print(f"    A: {proposal.answer[:200]}{'...' if len(proposal.answer) > 200 else ''}")
        print(f"    sources: {', '.join(c.document_id for c in proposal.citations) or '(none)'}")
        if proposal.supersedes:
            print("    ! replaces an answer you previously approved")
        print()
    print("openknowledge approve <id>   /   openknowledge reject <id>")
    return 0


def _cmd_approve(args: argparse.Namespace) -> int:
    engine = _engine()
    if engine.approve(args.proposal_id, reviewer=args.reviewer):
        print(f"approved {args.proposal_id} and pinned it")
        return 0
    print(f"no draft awaiting review with id {args.proposal_id}", file=sys.stderr)
    return 1


def _cmd_reject(args: argparse.Namespace) -> int:
    engine = _engine()
    if engine.knowledge.reject(args.proposal_id, reviewer=args.reviewer, note=args.note):
        print(f"rejected {args.proposal_id}; it will not be proposed again")
        return 0
    print(f"no draft awaiting review with id {args.proposal_id}", file=sys.stderr)
    return 1


def _cmd_conflicts(args: argparse.Namespace) -> int:
    """Show documents that disagree with each other."""
    engine = _engine()
    conflicts = engine.knowledge.open_conflicts()
    if not conflicts:
        print("No unresolved disagreements between your documents.")
        return 0

    print(f"{len(conflicts)} unresolved disagreements\n")
    for conflict in conflicts[: args.limit]:
        print(f"  {conflict.key}")
        print(f"    [{conflict.left_document}] {conflict.left_raw}")
        print(f"      {conflict.left_sentence[:150]}")
        print(f"    [{conflict.right_document}] {conflict.right_raw}")
        print(f"      {conflict.right_sentence[:150]}")
        print()
    print("Questions touching these are refused until resolved:")
    print("  openknowledge resolve <key> --keep <document-id>")
    return 0


def _cmd_resolve(args: argparse.Namespace) -> int:
    engine = _engine()
    resolution = args.note or (f"authoritative: {args.keep}" if args.keep else "resolved")
    if engine.knowledge.resolve_conflict(args.key, resolution=resolution, resolver=args.reviewer):
        print(f"resolved {args.key}: {resolution}")
        if args.keep:
            print(
                "\nThis records the decision but does not edit your documents. "
                "Remove or correct the superseded text so retrieval stops seeing it."
            )
        return 0
    print(f"no open conflict with key {args.key}", file=sys.stderr)
    return 1


def _cmd_eval(args: argparse.Namespace) -> int:
    """Run the golden set and report accuracy and cost together."""
    import json as _json

    from .evaluation import (
        DatasetError,
        compare,
        filter_cases,
        format_report,
        load_cases,
        run_eval,
    )

    try:
        cases = load_cases(args.path)
    except DatasetError as exc:
        print(f"golden set: {exc}", file=sys.stderr)
        return 2

    cases = filter_cases(
        cases,
        kind=None if args.only == "all" else args.only,
        tags=tuple(args.tag or ()),
    )
    if not cases:
        print("no cases matched the given filters", file=sys.stderr)
        return 2

    engine = _engine()
    report = asyncio.run(run_eval(engine.cascade, cases, check_determinism=not args.no_determinism))

    if args.json:
        print(_json.dumps(report.to_dict(), indent=2))
    else:
        print(format_report(report, verbose=args.verbose))

    if args.save_baseline:
        Path(args.save_baseline).write_text(_json.dumps(report.to_dict(), indent=2) + "\n")
        print(f"\nbaseline written to {args.save_baseline}", file=sys.stderr)

    if args.baseline:
        baseline = _json.loads(Path(args.baseline).read_text())
        result = compare(report, baseline)
        print()
        for line in result.improvements:
            print(f"  improved:   {line}")
        for line in result.regressions:
            print(f"  REGRESSED:  {line}")
        if not result.ok:
            return 1

    return 0 if report.passed else 1


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
    p.add_argument(
        "--cite",
        action="append",
        metavar="DOC_ID",
        help="document this answer comes from; repeat for several",
    )
    p.add_argument(
        "--alias",
        action="append",
        metavar="PHRASING",
        help="another way people ask this; repeat for several",
    )
    p.set_defaults(func=_cmd_pin)

    p = sub.add_parser("unpin", help="remove a pinned answer")
    p.add_argument("question")
    p.set_defaults(func=_cmd_unpin)

    p = sub.add_parser("learn", help="draft answers for changed documents (costs tokens)")
    p.add_argument("--max-documents", type=int, help="cap this run")
    p.set_defaults(func=_cmd_learn)

    p = sub.add_parser("review", help="drafted answers awaiting review")
    p.add_argument("--limit", type=int, default=20)
    p.add_argument(
        "--cost-per-answer",
        type=float,
        default=0.0094,
        help="what one un-cached answer costs, for ranking by value",
    )
    p.set_defaults(func=_cmd_review)

    p = sub.add_parser("approve", help="approve a drafted answer and pin it")
    p.add_argument("proposal_id")
    p.add_argument("--reviewer")
    p.set_defaults(func=_cmd_approve)

    p = sub.add_parser("reject", help="reject a drafted answer for good")
    p.add_argument("proposal_id")
    p.add_argument("--reviewer")
    p.add_argument("--note")
    p.set_defaults(func=_cmd_reject)

    p = sub.add_parser("conflicts", help="documents that disagree with each other")
    p.add_argument("--limit", type=int, default=20)
    p.set_defaults(func=_cmd_conflicts)

    p = sub.add_parser("resolve", help="record which document wins a disagreement")
    p.add_argument("key")
    p.add_argument("--keep", metavar="DOC_ID", help="the document that is authoritative")
    p.add_argument("--note")
    p.add_argument("--reviewer")
    p.set_defaults(func=_cmd_resolve)

    p = sub.add_parser("eval", help="run the golden set: accuracy and cost together")
    p.add_argument("--path", default="evals", help="golden set file or directory")
    p.add_argument("--json", action="store_true")
    p.add_argument("--verbose", action="store_true", help="show failing answers in full")
    p.add_argument(
        "--only",
        choices=("all", "answerable", "refusal"),
        default="all",
        help="run only one kind of case; 'refusal' is the safety set",
    )
    p.add_argument("--tag", action="append", help="run only cases carrying this tag")
    p.add_argument(
        "--no-determinism",
        action="store_true",
        help="skip the ask-twice check (halves the run cost)",
    )
    p.add_argument("--baseline", help="compare against a saved baseline and fail on regressions")
    p.add_argument("--save-baseline", help="write this run's metrics to a file")
    p.set_defaults(func=_cmd_eval)

    p = sub.add_parser("pricing", help="show the model price table")
    p.set_defaults(func=_cmd_pricing)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
