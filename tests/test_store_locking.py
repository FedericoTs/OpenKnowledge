"""Every SQLite connection this server shares between threads stays serialised.

Measured, not assumed. ``tools/measure_locking.py`` runs the four-thread load
below against a store whose readers do not take the lock; on this machine 82%
of every attempt failed - 4,556 "database is locked", 2,322 cursors reporting
"another row available", and 41 errors whose message was scrambled to "not an
error" by a second thread clearing the connection's error state mid-report.

Two details are why the bug survived so long, and why the first attempt to
measure it produced nonsense:

*Sharing a connection is safe on its own.* SQLite here is built
``THREADSAFE=1``, so it holds its own mutex per connection and the same load
with nothing else touching the file raises nothing at all. The failures need
a second connection on the same file - which is exactly what a CLI ``costs``,
a backup, or a second worker creates.

*The busy timeout does not save you.* Python sets ``busy_timeout=5000`` by
default, but SQLite skips the busy handler entirely when waiting could
deadlock, and one connection holding a read open in one thread while another
thread writes is that case. Measured, the first "database is locked" came
back after 0.5 ms, not 5 s.

So the fix is not WAL - measured, WAL left 6,125 errors in place - and it is
not a longer timeout. It is that every use of a shared connection happens
under the store's own lock.

Two tests hold that, and the sabotage run says why both are needed. Taking
the lock back off ``recent_questions`` fails both. Taking it off
``cost_report`` or ``spend_since`` fails only the first: a single aggregate
that executes and fetches in one call is far less exposed than a row scan, so
the reproduction misses it while the reading of the source does not. The
stress test proves the failure is real; the static one is what actually keeps
it from coming back.
"""

from __future__ import annotations

import ast
import contextlib
import threading
import time
from pathlib import Path

import pytest

from openknowledge.cache.store import AnswerStore
from openknowledge.types import Answer, Tier

SOURCE = Path(__file__).resolve().parent.parent / "src" / "openknowledge"


def _shared_connection_modules() -> list[Path]:
    """Every module that hands one connection to more than one thread."""
    return sorted(p for p in SOURCE.rglob("*.py") if "check_same_thread=False" in p.read_text())


def _holds_the_lock(item: ast.withitem) -> bool:
    return isinstance(item.context_expr, ast.Attribute) and item.context_expr.attr == "_lock"


class _Unguarded(ast.NodeVisitor):
    """Lines that reach ``self._conn`` from outside ``with self._lock``."""

    def __init__(self) -> None:
        self.depth = 0
        self.lines: list[int] = []

    def visit_With(self, node: ast.With) -> None:  # noqa: N802 - ast API
        inside = any(_holds_the_lock(item) for item in node.items)
        self.depth += inside
        self.generic_visit(node)
        self.depth -= inside

    def visit_Attribute(self, node: ast.Attribute) -> None:  # noqa: N802 - ast API
        if node.attr == "_conn" and self.depth == 0:
            self.lines.append(node.lineno)
        self.generic_visit(node)


def test_shared_connections_are_only_used_under_the_lock() -> None:
    modules = _shared_connection_modules()
    assert modules, "no module opens a connection with check_same_thread=False any more"
    offenders = []
    for path in modules:
        tree = ast.parse(path.read_text())
        for cls in (n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)):
            for fn in (n for n in cls.body if isinstance(n, ast.FunctionDef)):
                # __init__ builds the lock, so it cannot yet take it, and
                # nothing else can reach the connection before it returns.
                if fn.name == "__init__":
                    continue
                probe = _Unguarded()
                for statement in fn.body:
                    probe.visit(statement)
                if probe.lines:
                    where = ", ".join(f"{path.name}:{line}" for line in probe.lines)
                    offenders.append(f"{cls.name}.{fn.name} ({where})")
    assert not offenders, (
        "these touch a shared SQLite connection outside the store's lock, which "
        "fails once a second process opens the same file: " + "; ".join(offenders)
    )


def test_a_second_connection_on_the_file_does_not_break_the_shared_one(tmp_path: Path) -> None:
    """The reproduction, shrunk to a second: reads, writes, and an outsider."""
    path = tmp_path / "answers.db"
    served = AnswerStore(path)
    outsider = AnswerStore(path)
    answer = Answer(text="An answer.", tier=Tier.LOCAL, model_id="test", cache_key="k")
    failures: list[str] = []
    lock = threading.Lock()
    until = time.monotonic() + 1.0

    def note(exc: BaseException) -> None:
        with lock:
            failures.append(f"{type(exc).__name__}: {exc}")

    def read() -> None:
        highest = 0
        while time.monotonic() < until:
            try:
                _, count = served.spend_since(0.0)
                served.cost_report()
                served.recent_questions(20)
            except BaseException as exc:  # noqa: BLE001 - the test is what it raises
                note(exc)
                continue
            # Only inserts happen here, so a shrinking count would be a query
            # that answered wrongly without raising - the worse failure.
            if count < highest:
                note(AssertionError(f"ledger count fell from {highest} to {count}"))
            highest = max(highest, count)

    def write(store: AnswerStore) -> None:
        n = 0
        while time.monotonic() < until:
            n += 1
            try:
                store.record(f"question {n}", answer, channel="test")
            except BaseException as exc:  # noqa: BLE001 - the test is what it raises
                note(exc)

    threads = [threading.Thread(target=read) for _ in range(2)]
    threads.append(threading.Thread(target=write, args=(served,)))
    threads.append(threading.Thread(target=write, args=(outsider,)))
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    for store in (served, outsider):
        with contextlib.suppress(Exception):
            store.close()

    if failures:
        pytest.fail(f"{len(failures)} concurrent operations failed, e.g. {failures[:5]}")
