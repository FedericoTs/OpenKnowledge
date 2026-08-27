"""Command line interface.

``openknowledge costs`` is the one that matters most: it reports what the bot
actually cost from the ledger, so the claim on the front page is something an
operator can check rather than take on faith.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import sys
import time
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

    skipped = engine.connector.skipped
    if skipped:
        print(f"\n{len(skipped)} file(s) contributed nothing:")
        for item in skipped[:20]:
            print(f"  {item.path}: {item.reason}")
        if len(skipped) > 20:
            print(f"  ... and {len(skipped) - 20} more")
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
    if answer.support is not None:
        print(f"  support: {answer.support:.0%} of this answer's wording appears in its sources")
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
    """Show documents that disagree with each other, grouped by pair.

    Grouped for the same reason `audit` groups: two documents disagreeing on
    twenty figures is one stale copy, and listing it twenty times buries the
    pair that disagrees on one - which is the one somebody has to decide.
    """
    from .knowledge.variants import group_stored

    engine = _engine()
    conflicts = engine.knowledge.open_conflicts()
    if not conflicts:
        print("No unresolved disagreements between your documents.")
        return 0

    pairs = group_stored(list(conflicts))
    real = [p for p in pairs if not p.is_variant]
    duplicated = [p for p in pairs if p.is_variant]

    shown = 0
    if real:
        print(f"{sum(len(p.conflicts) for p in real)} disagreement(s) to decide\n")
    for pair in real:
        for conflict in pair.conflicts:
            if shown >= args.limit:
                break
            shown += 1
            print(f"  {conflict.key}")
            print(f"    [{conflict.left_document}] {conflict.left_raw}")
            print(f"      {conflict.left_sentence[:150]}")
            print(f"    [{conflict.right_document}] {conflict.right_raw}")
            print(f"      {conflict.right_sentence[:150]}")
            print()

    if duplicated:
        print(f"{len(duplicated)} pair(s) look like duplicated documents:\n")
        for pair in duplicated:
            print(f"  {pair.describe()}")
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

    if args.dry_run:
        # Free, offline, and the first thing to run against a new golden set:
        # about half of a new set's failures are the set's own, not the model's.
        from .evaluation import format_preflight, preflight

        checked = preflight(
            cases,
            retriever=engine.retriever,
            k=engine.settings.retrieval_k,
            candidates=engine.settings.rerank_candidates,
            reranker=engine.cascade.reranker,
        )
        print(format_preflight(checked))
        return 0 if checked.passed else 1

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


def _cmd_eval_conflicts(args: argparse.Namespace) -> int:
    """Measure contradiction detection: precision and recall, no model needed."""
    import json as _json

    from .evaluation import (
        ConflictSetError,
        format_conflict_report,
        load_conflict_cases,
        run_conflict_eval,
    )

    try:
        cases = load_conflict_cases(args.path)
    except ConflictSetError as exc:
        print(f"conflict set: {exc}", file=sys.stderr)
        return 2

    report = run_conflict_eval(cases, deontic_strictness=args.strictness)
    if args.json:
        print(_json.dumps(report.to_dict(), indent=2))
    else:
        print(format_conflict_report(report))
    return 0 if report.passed else 1


def _cmd_audit(args: argparse.Namespace) -> int:
    """Read a folder and report where its documents disagree. No key, no model.

    Deliberately the one command that needs nothing configured: it builds no
    store, touches no data directory, and calls no provider, so it can be run
    against a folder of real policies by somebody who has not decided whether
    they trust this project yet.
    """
    from .audit import audit_folder, render

    settings = load_settings()
    report = audit_folder(
        args.path or settings.documents_dir,
        min_overlap=(
            settings.conflict_min_overlap if args.min_overlap is None else args.min_overlap
        ),
        deontic_strictness=args.strictness,
        pdf_backend=args.pdf_backend or settings.pdf_backend,
    )

    if args.json:
        print(json.dumps(report.as_dict(), indent=2))
    else:
        print(render(report))

    if report.documents == 0:
        print(f"\nNo readable documents in {report.root}.", file=sys.stderr)
        return 2
    # Non-zero on findings so this can be a CI step: a pull request that makes
    # two policies disagree fails before anybody is answered from either.
    return 0 if args.exit_zero or report.clean else 1


def _cmd_contacts(args: argparse.Namespace) -> int:
    """People who filled in the form on the website.

    Read from the deployment's own file. There is nowhere else they could be:
    the form posts to the same container that served the page.
    """
    from pathlib import Path as _Path

    from .contacts import ContactStore

    settings = load_settings()
    path = _Path(settings.data_dir) / settings.contacts_db
    if not path.exists():
        print(f"No contacts yet ({path} does not exist).")
        print("The form is served only when OK_WEBSITE_ENABLED=true.")
        return 0

    store = ContactStore(path)
    people = store.recent(args.limit)
    if args.json:
        print(json.dumps([dataclasses.asdict(c) for c in people], indent=2))
        return 0

    if not people:
        print("No contacts yet.")
        return 0

    print(f"{len(people)} of {store.count()} contact(s), newest first\n")
    for c in people:
        when = time.strftime("%Y-%m-%d %H:%M", time.localtime(c.ts))
        org = f" · {c.organisation}" if c.organisation else ""
        print(f"  {when}  {c.name} <{c.email}>{org}")
        if c.interest:
            print(f"      interested in: {c.interest}")
        if c.message:
            print(f"      {c.message[:160]}")
        print()
    return 0


def _cmd_pricing(_: argparse.Namespace) -> int:
    prices = list(load_price_table().values())
    width = max((len(p.model_id) for p in prices), default=20) + 2
    print(f"{'model':<{width}}{'tier':<10}{'in $/M':>9}{'out $/M':>10}  verified")
    for price in prices:
        if price.is_priced:
            print(
                f"{price.model_id:<{width}}{price.tier:<10}{price.input_per_mtok:>9.2f}"
                f"{price.output_per_mtok:>10.2f}  {price.verified}"
            )
        else:
            print(
                f"{price.model_id:<{width}}{price.tier:<10}{'-':>9}{'-':>10}  "
                "not set (see pricing.yaml)"
            )
    return 0


def _cmd_model(args: argparse.Namespace) -> int:
    """Show, or change, which model answers on the cheap rung.

    Everything printed here was asked of the runtime rather than looked up in a
    table: which models exist, and what window each declares. A table would be
    wrong within a release, and a wrong context window is the one setting that
    fails quietly.
    """
    from . import models as local_models

    settings = load_settings()
    runtime = local_models.probe(settings.local_base_url)

    if args.action == "list":
        print(f"runtime: {runtime.kind} at {settings.local_base_url}", end="")
        print(f" (v{runtime.version})" if runtime.version else "")
        if not runtime.reachable:
            print("\n  Nothing is answering there. Start it, or point OK_LOCAL_BASE_URL")
            print("  at the machine that runs it.")
        else:
            have = local_models.installed(runtime)
            if have:
                print(f"\n{len(have)} model(s) installed:")
                width = max(len(m.name) for m in have) + 2
                for model in sorted(have, key=lambda m: m.name):
                    window = local_models.context_window(runtime, model.name)
                    shown = f"{window:,} tokens" if window else "window not reported"
                    mark = " <- in use" if model.name == settings.local_model else ""
                    print(f"  {model.name:<{width}}{model.size:>9}  {shown}{mark}")
            else:
                print("\n  No models installed yet.")
        print("\nWorth trying:")
        width = max(len(name) for name, _ in local_models.SUGGESTED) + 2
        for name, note in local_models.SUGGESTED:
            print(f"  {name:<{width}}{note}")
        print("\n  openknowledge model use <name> [--context N]")
        return 0

    if args.action == "status":
        window = settings.local_context_tokens
        print(f"model:    {settings.local_model}")
        print(f"runtime:  {runtime.kind} at {settings.local_base_url}")
        print(f"window:   {window:,} tokens" if window else "window:   not recorded")
        if not settings.local_enabled:
            print("\n  The local tier is off (OK_LOCAL_ENABLED=false).")
        if not runtime.reachable:
            print("\n  Not reachable. `openknowledge ask` will refuse rather than answer.")
            return 1
        live = local_models.context_window(runtime, settings.local_model)
        if live and window and live != window:
            print(
                f"\n  The runtime reports {live:,} tokens, not the {window:,} recorded here."
                "\n  Re-run `openknowledge model use` to bring them back into line."
            )
            return 1
        names = {m.name for m in local_models.installed(runtime)}
        if names and settings.local_model not in names:
            print(f"\n  {settings.local_model!r} is not installed on that runtime.")
            return 1
        return 0

    # --- use ---------------------------------------------------------------
    # Rewriting one line in place needs a terminal; *saying something* does not.
    # Gating the whole report on isatty made `model use` completely silent under
    # Git Bash, where MinTTY is a named pipe and Python reports isatty() False -
    # so a five-gigabyte download looked like a hung command on the one platform
    # least able to tell. Progress is now unconditional; only the redraw is not.
    redraw = sys.stderr.isatty()
    last_step = -1

    def report(status: str, done: int, total: int) -> None:
        nonlocal last_step
        if total:
            gb = f"{done / 1_000_000_000:.1f} of {total / 1_000_000_000:.1f} GB"
            percent = done * 100 // total
            line = f"  {status}: {gb} ({percent}%)"
        else:
            percent = -1
            line = f"  {status}"

        if redraw:
            # Padded, to clear whatever the previous, longer line left behind.
            print(f"\r{line:<64}", end="", file=sys.stderr, flush=True)
            return

        # No redraw available: one line every ten percent, so it is neither
        # silent nor thousands of lines of scroll.
        step = percent // 10
        if step == last_step:
            return
        last_step = step
        print(line, file=sys.stderr, flush=True)

    try:
        result = local_models.switch(
            runtime,
            args.model,
            context=args.context,
            allow_download=not args.no_download,
            on_progress=report,
        )
    except local_models.ModelError as exc:
        print(f"\nerror: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        # Interrupting a download is a normal thing to do, not a crash. Say what
        # survives it, because "did I just lose four gigabytes" is the question.
        print(
            f"\nstopped. {args.model} was not switched to; anything already "
            f"downloaded is kept, so `ollama pull {args.model}` resumes rather "
            "than starting over.",
            file=sys.stderr,
        )
        return 130
    finally:
        if redraw:
            print("\r" + " " * 64 + "\r", end="", file=sys.stderr, flush=True)

    values = {"OK_LOCAL_MODEL": result.model, "OK_LOCAL_ENABLED": "true"}
    if result.context:
        values["OK_LOCAL_CONTEXT_TOKENS"] = str(result.context)
    env = Path(args.env_file)
    written = local_models.write_env(env, values)

    if result.pulled:
        print(f"downloaded {args.model}")
    if result.derived_from:
        print(
            f"built {result.model} from {result.derived_from} "
            f"with a {result.context:,}-token window"
        )
    print(f"model:  {result.model}")
    print(f"window: {result.context:,} tokens" if result.context else "window: not reported")
    print(f"wrote:  {', '.join(written)} -> {env}")
    for note in result.notes:
        print(f"\n  note: {note}")
    print("\n  Restart `openknowledge serve` for this to take effect.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """The whole command surface, as an object rather than a side effect.

    Separate from :func:`main` so that the documentation tests can walk it -
    every command and flag a guide tells a reader to run is checked against this
    tree, which is cheaper and more exact than running each one.
    """
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
    p.add_argument("--path", default="evals/golden", help="golden set file or directory")
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
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="check the set is answerable from retrieval alone - free, no model, no network",
    )
    p.add_argument("--baseline", help="compare against a saved baseline and fail on regressions")
    p.add_argument("--save-baseline", help="write this run's metrics to a file")
    p.set_defaults(func=_cmd_eval)

    p = sub.add_parser(
        "eval-conflicts", help="measure contradiction detection (precision and recall)"
    )
    p.add_argument("--path", default="evals/conflicts", help="labelled set file or directory")
    p.add_argument("--json", action="store_true")
    p.add_argument(
        "--strictness",
        type=float,
        default=1.0,
        help="scale the prose thresholds; above 1.0 flags less",
    )
    p.set_defaults(func=_cmd_eval_conflicts)

    p = sub.add_parser(
        "audit", help="where a folder of documents disagrees with itself (free, no model)"
    )
    p.add_argument("path", nargs="?", help="folder to read; defaults to OK_DOCUMENTS_DIR")
    p.add_argument("--json", action="store_true")
    p.add_argument("--pdf-backend", choices=("auto", "opendataloader", "pdfplumber"))
    p.add_argument(
        "--strictness",
        type=float,
        default=1.0,
        help="scale the prose thresholds; above 1.0 flags less",
    )
    p.add_argument(
        "--min-overlap",
        type=float,
        default=None,
        metavar="R",
        help=(
            "how much context two figures must share before disagreeing about "
            "them counts (default 0.34); raise it to flag less"
        ),
    )
    p.add_argument(
        "--exit-zero",
        action="store_true",
        help="always exit 0; without it, findings exit 1 so this can gate CI",
    )
    p.set_defaults(func=_cmd_audit)

    p = sub.add_parser("contacts", help="people who filled in the website contact form")
    p.add_argument("--limit", type=int, default=50)
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=_cmd_contacts)

    p = sub.add_parser("model", help="which local model answers, and how big its window is")
    model_sub = p.add_subparsers(dest="action", required=True)

    q = model_sub.add_parser("list", help="models the runtime has, and their windows")
    q.set_defaults(func=_cmd_model)

    q = model_sub.add_parser("status", help="check the configured model against the runtime")
    q.set_defaults(func=_cmd_model)

    q = model_sub.add_parser("use", help="switch to a model, downloading it if needed")
    q.add_argument("model", help="model name, e.g. qwen3:8b")
    q.add_argument(
        "--context",
        type=int,
        default=None,
        metavar="N",
        help=(
            "run it with an N-token window. Ollama takes no per-call context "
            "setting, so this builds a copy of the model carrying it."
        ),
    )
    q.add_argument(
        "--no-download",
        action="store_true",
        help="fail rather than fetching a model that is not installed",
    )
    q.add_argument("--env-file", default=".env", help="where to record it (default: .env)")
    q.set_defaults(func=_cmd_model)

    p = sub.add_parser("pricing", help="show the model price table")
    p.set_defaults(func=_cmd_pricing)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
