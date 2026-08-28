"""Ask quantprobe how to place the model before spawning llama-server.

The field failure this answers: a laptop's integrated GPU refused the KV
allocation, and the launcher's remedy was a coin flip - all layers on the
GPU, else all on the CPU. quantprobe (the author's own probe-then-quantize
tool, MIT) answers the real question in about a second, offline, from the
GGUF already on disk: it detects the machine (VRAM, RAM and their measured
bandwidths - on Windows via the driver registry, dodging the 4 GB
AdapterRAM trap), prices every placement by its tiered decode law, and
emits the llama.cpp command for the winner.

Integration is deliberately narrow and fail-open:

- quantprobe is called **in-process** through its own CLI entry with argv
  patched and stdout captured - the frozen app has no python to subprocess,
  and the CLI path is the one their audit hardened ("the command a user
  SEES must be the command run EXECUTES").
- Only the ``run it: llama-server ...`` line is read, and only flags on an
  allowlist survive: placement (``-ngl``, ``-ot``), threads and batch
  sizing, mmap. Ports, hosts, context and speculation flags never pass -
  this app owns its ports and context, and determinism is not negotiable.
- Anything unexpected - quantprobe missing, a version drift, a crash, a
  timeout - returns ``None`` and the launcher behaves exactly as before.
  The plan is an upgrade, never a dependency.
"""

from __future__ import annotations

import contextlib
import io
import re
import shlex
import sys
import threading
from dataclasses import dataclass
from pathlib import Path

#: Flags that take a value and are safe to forward to our llama-server.
_VALUED = {
    "-ngl",
    "--n-gpu-layers",
    "-ot",
    "--override-tensor",
    "--threads",
    "-t",
    "-b",
    "--batch-size",
    "-ub",
    "--ubatch-size",
}
#: Flags that stand alone and are safe to forward.
_BOOLEAN = {"--no-mmap", "--mlock"}

_RUN_LINE = re.compile(r"run it:\s*llama-server\s+(.+)$", re.MULTILINE)
_PICK_LINE = re.compile(r"^\s*\*\s+([\d.]+)\s+tok/s\s+(.+?)(?:\s{3,}\[|$)", re.MULTILINE)


@dataclass(frozen=True)
class PlannedLaunch:
    """What quantprobe recommends, reduced to what we are willing to use."""

    extra_args: tuple[str, ...]
    summary: str


def extract_planned_flags(text: str) -> tuple[str, ...]:
    """The allowlisted argv from quantprobe's ``run it:`` line.

    Unknown flags are dropped together with what looks like their value -
    forwarding a flag this code has never heard of into a server that owns
    loopback ports is how "planned" becomes "surprising".
    """
    match = _RUN_LINE.search(text)
    if not match:
        return ()
    tokens = shlex.split(match.group(1))
    kept: list[str] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token == "-m":
            index += 2  # the model path is ours to supply
            continue
        if token in _VALUED and index + 1 < len(tokens):
            kept.extend((token, tokens[index + 1]))
            index += 2
            continue
        if token in _BOOLEAN:
            kept.append(token)
            index += 1
            continue
        if (
            token.startswith("-")
            and index + 1 < len(tokens)
            and not tokens[index + 1].startswith("-")
        ):
            index += 2  # unknown valued flag: skip it and its value
            continue
        index += 1
    return tuple(kept)


def extract_summary(text: str) -> str:
    """The winning placement, as one line a setup page can show."""
    match = _PICK_LINE.search(text)
    if not match:
        return ""
    tok_s, name = match.groups()
    return f"{name.strip()} — about {tok_s} tok/s expected"


def plan_chat_flags(
    gguf_path: Path, ctx_tokens: int, timeout_seconds: float = 45.0
) -> PlannedLaunch | None:
    """Run quantprobe's planner for the model at ``gguf_path``. None on any doubt."""
    result: dict[str, str] = {}

    def _run() -> None:
        buffer = io.StringIO()
        argv = [
            "quantprobe",
            "plan",
            "--gguf",
            str(gguf_path),
            "--ctx",
            str(ctx_tokens),
        ]
        try:
            from quantprobe import cli  # noqa: PLC0415 - optional, resolved at call time

            with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(io.StringIO()):
                old_argv = sys.argv
                sys.argv = argv
                try:
                    cli.main()
                finally:
                    sys.argv = old_argv
        except SystemExit:
            pass  # argparse exits are part of a CLI's normal life
        except Exception:
            return  # missing, drifted, or broken: the floor behaviour takes over
        result["text"] = buffer.getvalue()

    worker = threading.Thread(target=_run, daemon=True)
    worker.start()
    worker.join(timeout=timeout_seconds)
    text = result.get("text", "")
    if not text:
        return None
    flags = extract_planned_flags(text)
    if not flags:
        return None
    return PlannedLaunch(extra_args=flags, summary=extract_summary(text))
