"""Does concurrent use of a store actually break, and how?

Written after a reading of the code suggested it must, and a first harness
produced a result so odd - more errors *with* the fix than without - that the
harness was clearly the thing under test. Both corrections are recorded here
so the next person does not repeat them:

*Python already sets a busy timeout.* ``sqlite3.connect`` defaults to
``timeout=5.0``, which is ``PRAGMA busy_timeout=5000``. The stores never set
one because they never had to. The original suspicion - "no busy timeout
anywhere" - was simply wrong, and checking took one line.

*Journal mode is a property of the file, not the connection.* Setting WAL on
one open connection while another holds the same file in rollback mode
produces exactly the storm of lock errors the first run showed. It has to be
set before any other connection opens, and its result row has to be read.

With those out of the way the remaining question splits in two, and the two
must not be measured together - the first harness mixed them and the numbers
said nothing.

**One connection, several threads.** What the server does: FastAPI serves on
a thread pool and every store hands the same ``Connection`` to all of them.
Python permits that and requires the caller to serialise access. Writes always
did; reads did not until this was measured. The cases marked *(before)* issue
the same three ledger queries straight at the connection, which is what the
reads used to do - kept here on purpose, so the fix stays measurable after
the code it fixed is gone.

**One file, several connections.** What a second process creates - a CLI
``costs`` while the server runs, a backup, ``--workers 2``. The in-process
lock coordinates nothing across connections, so this is SQLite's own
contention, measured with and without WAL.

Two things are counted, and the second matters more than the first. An
*error* is a raised exception. A *wrong answer* is a read that returned a
smaller ledger count than an earlier read on the same thread - impossible
when the only writes are inserts, so every one is a query that silently
returned something untrue.

    uv run python tools/measure_locking.py --seconds 5
    uv run python tools/measure_locking.py --seconds 5 --json
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sqlite3
import threading
import time
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory

from openknowledge.cache.store import AnswerStore
from openknowledge.types import Answer, Tier


def _answer(n: int) -> Answer:
    return Answer(text=f"An answer {n}.", tier=Tier.LOCAL, model_id="measure", cache_key=f"k{n}")


def _set_wal(path: Path) -> None:
    """Put the file in WAL mode before anything else opens it."""
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=WAL").fetchone()
    conn.close()


class _Tally:
    """What happened, by kind - not just how much."""

    def __init__(self) -> None:
        self.ok: Counter[str] = Counter()
        self.errors: Counter[str] = Counter()
        self.wrong = 0
        self._lock = threading.Lock()

    def did(self, what: str) -> None:
        with self._lock:
            self.ok[what] += 1

    def read_went_backwards(self) -> None:
        with self._lock:
            self.wrong += 1

    def failed(self, exc: BaseException) -> None:
        text = str(exc)
        if "locked" in text or "busy" in text:
            kind = "database is locked"
        elif "another row available" in text or "no more rows" in text:
            kind = "cursor interference"
        else:
            kind = f"{type(exc).__name__}: {text[:60]}"
        with self._lock:
            self.errors[kind] += 1

    def as_dict(self, seconds: float) -> dict[str, object]:
        done = self.ok["write"] + self.ok["read"]
        errors = sum(self.errors.values())
        return {
            "writes": self.ok["write"],
            "reads": self.ok["read"],
            "ops_per_second": round(done / seconds, 1),
            "errors": dict(self.errors),
            "error_total": errors,
            # Per attempt, so a case that blocked its way to few operations
            # is comparable with one that ran fast.
            "error_rate": round(errors / (done + errors), 4) if done + errors else 0.0,
            "wrong_answers": self.wrong,
        }


#: The three queries a /manage page load makes of the ledger. Kept here in
#: full so the "before" cases can issue them straight at the connection - the
#: way the store did before its readers took the lock - long after the store
#: itself stopped doing that. Without this the fix would be unmeasurable a
#: week after it shipped.
_READS = (
    (
        "SELECT COALESCE(SUM(cost_usd), 0.0) AS spend, COUNT(*) AS n FROM ledger WHERE ts >= ?",
        (0.0,),
    ),
    ("SELECT tier, COUNT(*) AS n, SUM(cost_usd) AS spend FROM ledger GROUP BY tier", ()),
    (
        "SELECT ts, canonical_query, tier, model_id, cost_usd, channel"
        " FROM ledger ORDER BY ts DESC, id DESC LIMIT ?",
        (20,),
    ),
)


def _read_through_the_store(store: AnswerStore) -> int:
    """What a page load does now: public methods, each taking the lock."""
    _, count = store.spend_since(0.0)
    store.cost_report()
    store.recent_questions(20)
    return count


def _read_past_the_lock(store: AnswerStore) -> int:
    """What a page load did before: the same queries, straight at the connection."""
    count = 0
    for sql, params in _READS:
        rows = store._conn.execute(sql, params).fetchall()  # noqa: SLF001
        if "COUNT(*) AS n" in sql and "GROUP BY" not in sql:
            count = int(rows[0]["n"])
    return count


def _write(store: AnswerStore, tally: _Tally, until: float) -> None:
    n = 0
    while time.monotonic() < until:
        n += 1
        try:
            store.record(f"question {n}", _answer(n), channel="measure")
            tally.did("write")
        except BaseException as exc:  # noqa: BLE001 - a measurement reports, never raises
            tally.failed(exc)


def _read(store: AnswerStore, tally: _Tally, until: float, *, locked: bool) -> None:
    highest = 0
    query = _read_through_the_store if locked else _read_past_the_lock
    while time.monotonic() < until:
        try:
            count = query(store)
            # Only inserts happen in this harness, so the ledger count can
            # never shrink. A read that says it did is a wrong answer that
            # raised nothing - the failure mode worth finding.
            if count < highest:
                tally.read_went_backwards()
            highest = max(highest, count)
            tally.did("read")
        except BaseException as exc:  # noqa: BLE001 - a measurement reports, never raises
            tally.failed(exc)


def _case(
    label: str,
    *,
    shared_connection: bool,
    readers: int,
    writers: int,
    seconds: float,
    wal: bool = False,
    locked_reads: bool = True,
    contenders: int = 0,
) -> dict[str, object]:
    """One configuration, run flat out for ``seconds``.

    ``shared_connection`` picks the question: one store for every thread (the
    server today), or one store per thread on the same file.

    ``contenders`` adds that many further connections, one writer thread
    each, standing in for the other processes that open the same file - a
    CLI ``costs`` while the server runs, a backup, a second worker. They are
    what makes the server's own connection meet SQLITE_BUSY at all.
    """
    with TemporaryDirectory() as tmp:
        path = Path(tmp) / "answers.db"
        if wal:
            _set_wal(path)
        count = 1 if shared_connection else readers + writers
        stores = [AnswerStore(path) for _ in range(count)]
        outside = [AnswerStore(path) for _ in range(contenders)]

        def store_for(index: int) -> AnswerStore:
            return stores[0] if shared_connection else stores[index]

        tally = _Tally()
        until = time.monotonic() + seconds
        threads = [
            threading.Thread(target=_write, args=(store_for(i), tally, until))
            for i in range(writers)
        ]
        threads += [
            threading.Thread(
                target=_read,
                args=(store_for(writers + i), tally, until),
                kwargs={"locked": locked_reads},
            )
            for i in range(readers)
        ]
        threads += [threading.Thread(target=_write, args=(s, tally, until)) for s in outside]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        for store in [*stores, *outside]:
            with contextlib.suppress(Exception):
                store.close()
    return {
        "case": label,
        "connections": count + contenders,
        "wal": wal,
        "locked_reads": locked_reads,
        "other_processes": contenders,
        **tally.as_dict(seconds),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument("--readers", type=int, default=4)
    parser.add_argument("--writers", type=int, default=2)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    load = {"readers": args.readers, "writers": args.writers, "seconds": args.seconds}
    alone = {"shared_connection": True, **load}
    crowded = {**alone, "contenders": 1}
    results = [
        _case("one connection, reads only", **{**alone, "writers": 0}),
        _case("one connection, reads and writes", **alone),
        _case("...with reads going past the lock (before)", locked_reads=False, **alone),
        _case("one connection, and a second process writing", **crowded),
        _case("...with reads going past the lock (before)", locked_reads=False, **crowded),
        _case("...that, in WAL mode", locked_reads=False, wal=True, **crowded),
        _case("one connection per thread, one file", shared_connection=False, **load),
        _case(
            "one connection per thread, one file, WAL", shared_connection=False, wal=True, **load
        ),
    ]
    if args.json:
        print(json.dumps(results, indent=2))
        return 0
    head = f"{'case':<48}{'ops/s':>9}{'errors':>9}{'rate':>8}{'wrong':>8}"
    print(head)
    print("-" * len(head))
    for row in results:
        print(
            f"{row['case']:<48}{row['ops_per_second']:>9}{row['error_total']:>9}"
            f"{row['error_rate']:>8}{row['wrong_answers']:>8}"
        )
        errors: dict[str, int] = row["errors"]  # type: ignore[assignment]
        for kind, n in sorted(errors.items(), key=lambda kv: -kv[1]):
            print(f"      {n:>6}  {kind}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
