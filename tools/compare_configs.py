#!/usr/bin/env python3
"""Run one golden set against several configurations, and diff them.

    uv run python tools/compare_configs.py

The question this exists to answer is not "how good is it" but **"how much does
the cheap tier cost me in accuracy, and does the ladder buy it back"** - which is
a difference between two runs, not a number from one. A single configuration
measures a setup; two measure a decision.

Profiles live in `evals/profiles.yaml` and name environment variables rather than
holding values, so nothing secret is committed. A profile whose keys are absent
is **skipped with a reason**, never failed: the same command has to work for
somebody with no keys, somebody with one, and somebody with both, or it will only
ever be run by the person who wrote it.

Each profile runs in its own subprocess with its own data directory, because a
cache warmed by one configuration would make the next one look free.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Profile:
    name: str
    description: str = ""
    requires: tuple[str, ...] = ()
    env: dict[str, str] = field(default_factory=dict)

    def missing_keys(self) -> tuple[str, ...]:
        return tuple(name for name in self.requires if not os.environ.get(name))

    def resolved_env(self) -> dict[str, str]:
        """Expand `$VAR` and `$VAR:-fallback` against the real environment.

        The fallback form is what keeps the local profile usable without editing
        this file: Ollama listens on 11434, LM Studio on 1234, llama.cpp's server
        on whatever you told it, and none of those is more correct than the
        others.
        """
        return {key: _expand(value) for key, value in self.env.items()}


def _expand(value: str) -> str:
    if not value.startswith("$"):
        return value
    name, sep, fallback = value[1:].partition(":-")
    resolved = os.environ.get(name, "")
    return resolved or (fallback if sep else "")


def load_profiles(path: Path) -> list[Profile]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        sys.exit(f"{path}: expected a list of profiles")
    return [
        Profile(
            name=str(entry["name"]),
            description=str(entry.get("description", "")).strip(),
            requires=tuple(entry.get("requires") or ()),
            env={str(k): str(v) for k, v in (entry.get("env") or {}).items()},
        )
        for entry in raw
    ]


def run_profile(
    profile: Profile,
    *,
    golden: str,
    documents: str,
    out_dir: Path,
    extra: list[str],
) -> dict[str, object] | None:
    """Run the golden set under one profile. Returns its metrics, or None."""
    data_dir = out_dir / f"data-{profile.name}"
    # A fresh store per profile: a cache warmed by the previous configuration
    # would report the next one as almost entirely free, which is true and
    # meaningless.
    shutil.rmtree(data_dir, ignore_errors=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    baseline = out_dir / f"{profile.name}.json"

    env = {
        **os.environ,
        **profile.resolved_env(),
        "OK_DOCUMENTS_DIR": documents,
        "OK_DATA_DIR": str(data_dir),
    }
    command = [
        sys.executable,
        "-m",
        "openknowledge.cli",
        "eval",
        "--path",
        golden,
        "--save-baseline",
        str(baseline),
        *extra,
    ]

    print(f"\n{'=' * 72}\n{profile.name}  -  {profile.description}\n{'=' * 72}", flush=True)
    started = time.monotonic()
    result = subprocess.run(command, env=env, cwd=REPO, check=False)
    elapsed = time.monotonic() - started

    if not baseline.exists():
        print(f"  {profile.name}: no metrics written (exit {result.returncode})")
        return None

    metrics = json.loads(baseline.read_text())
    metrics["_profile"] = profile.name
    metrics["_seconds"] = round(elapsed, 1)
    # A non-zero exit means cases failed, which is a result, not an error. Only
    # a missing baseline means the run itself did not happen.
    metrics["_passed"] = result.returncode == 0
    return metrics


def render(rows: list[dict[str, object]], skipped: list[tuple[str, tuple[str, ...]]]) -> str:
    if not rows:
        return (
            "No profile could run.\n"
            "Set at least one of the keys named in evals/profiles.yaml, or start a "
            "local model, and try again."
        )

    def col(row: dict[str, object], key: str, fmt: str) -> str:
        value = row.get(key)
        return format(value, fmt) if isinstance(value, (int, float)) else "-"

    width = max(len(str(r["_profile"])) for r in rows) + 2
    lines = [
        "",
        "=" * 72,
        "Same golden set, same corpus, same prompt. Only the models differ.",
        "=" * 72,
        "",
        f"{'configuration':<{width}}{'accuracy':>10}{'false':>7}{'determ':>9}"
        f"{'$/question':>12}{'free':>7}{'minutes':>9}",
    ]
    for row in rows:
        lines.append(
            f"{str(row['_profile']):<{width}}"
            f"{col(row, 'accuracy', '>10.1%')}"
            f"{col(row, 'false_answers', '>7.0f')}"
            f"{col(row, 'determinism', '>9.1%')}"
            f"{col(row, 'cost_per_question_usd', '>12.5f')}"
            f"{col(row, 'free_share', '>7.0%')}"
            f"{float(row.get('_seconds', 0) or 0) / 60:>9.1f}"
        )

    lines += ["", "Which tier answered:", ""]
    for row in rows:
        tiers = row.get("tiers")
        spread = (
            ", ".join(f"{k} {v}" for k, v in sorted(tiers.items()))
            if isinstance(tiers, dict)
            else "-"
        )
        lines.append(f"  {row['_profile']:<{width}}{spread}")

    lines += ["", "How to read this:", ""]
    lines += [
        "  false      Questions the corpus does not answer that got an answer anyway.",
        "             Any number above zero outranks every other column here: a bot",
        "             that invents some of its answers is unusable, because nobody",
        "             can tell which ones.",
        "  accuracy   Of the answerable cases, how many were right AND cited.",
        "  determ     Same question twice, same answer.",
        "  free       Answered with no model at all - pinned, cached or drafted.",
        "",
        "  The difference between two rows is the finding. A cheap tier that matches",
        "  the frontier on false answers and trails on accuracy is a different",
        "  decision from one that trails on both.",
    ]

    if skipped:
        lines += ["", "Not run:", ""]
        for name, missing in skipped:
            lines.append(f"  {name:<{width}}needs {', '.join(missing)}")
        lines.append("")
        lines.append("  Set those and re-run; the profiles that already ran are cached in")
        lines.append("  the output directory and can be compared against directly.")

    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles", default="evals/profiles.yaml")
    parser.add_argument("--golden", default="evals/golden-aveline")
    parser.add_argument("--documents", default="evals/corpus/aveline")
    parser.add_argument("--out", default="evals/runs", help="where baselines are written")
    parser.add_argument("--only", action="append", help="run just this profile; repeatable")
    parser.add_argument(
        "--no-determinism",
        action="store_true",
        help="skip the ask-twice check; roughly halves the run time",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    profiles = load_profiles(REPO / args.profiles)
    if args.only:
        wanted = set(args.only)
        profiles = [p for p in profiles if p.name in wanted]
        if not profiles:
            sys.exit(f"no profile matched {sorted(wanted)}")

    out_dir = REPO / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    extra = ["--no-determinism"] if args.no_determinism else []

    rows: list[dict[str, object]] = []
    skipped: list[tuple[str, tuple[str, ...]]] = []
    for profile in profiles:
        missing = profile.missing_keys()
        if missing:
            skipped.append((profile.name, missing))
            print(f"skipping {profile.name}: needs {', '.join(missing)}", flush=True)
            continue
        metrics = run_profile(
            profile,
            golden=args.golden,
            documents=args.documents,
            out_dir=out_dir,
            extra=extra,
        )
        if metrics is not None:
            rows.append(metrics)

    if args.json:
        print(json.dumps({"runs": rows, "skipped": [list(s) for s in skipped]}, indent=2))
    else:
        print(render(rows, skipped))

    # Non-zero only when nothing ran at all. Failing cases are the point of the
    # exercise, not a reason to exit unsuccessfully.
    return 0 if rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
