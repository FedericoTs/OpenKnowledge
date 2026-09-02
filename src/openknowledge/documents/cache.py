"""Parsed documents, kept so the same bytes are never read twice.

Measured on this project's own parsers, per document, for a corpus of 120
files of roughly a page each:

    markdown    5.9 ms
    docx       56.7 ms
    PDF       780.3 ms

That last one is not a slow parser. Profiling a PDF rebuild put **99.2%** of
it in ``opendataloader``, which spawns a Java process once per file: almost
all of the time is JVM startup and pipe traffic rather than reading the
document. A hundred and twenty small PDFs rebuilt in 93.6 seconds, and a
thousand policy PDFs - an entirely ordinary corpus for the company this is
built for - is about thirteen minutes, paid again on every upload and every
delete.

None of that work is new each time. A file whose bytes have not changed
parses to exactly what it parsed to before, so this remembers the result.

**Keyed on the content, not on the clock.** ``mtime`` and size are the
obvious key and the wrong one: ``rsync -t``, ``git checkout`` and every
restore-from-backup put old timestamps on new bytes, and a cache that
believed them would serve the previous version of a policy forever. Reading
the file to hash it costs a few milliseconds against the hundreds this
saves, and it cannot be fooled.

The key also carries the parser's identity. Two PDF backends extract
slightly different text - the parser says so itself - so a cache shared
between them would hand one backend's output to a corpus fingerprinted
under the other's.

Stored in SQLite beside the rest of the state, because the first build is
where the whole thirteen minutes lands and a restart should not pay it
twice.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import threading
from pathlib import Path

from .blocks import Block, BlockKind, ParsedDocument

log = logging.getLogger(__name__)

#: Bumped when the stored shape changes, so an older row is a miss rather
#: than something to be interpreted. Cheaper than a migration for a cache:
#: everything in here can be recomputed from the file it came from.
FORMAT = 2

_SCHEMA = """
CREATE TABLE IF NOT EXISTS parses (
    key        TEXT PRIMARY KEY,
    parsed     TEXT NOT NULL,
    stored_at  REAL NOT NULL
);
"""


def cache_key(data: bytes, *, parser: str) -> str:
    """What this parse is of, and what produced it."""
    digest = hashlib.sha256(data).hexdigest()
    return f"{FORMAT}:{parser}:{digest}"


def _as_json(parsed: ParsedDocument) -> str:
    return json.dumps(
        {
            "title": parsed.title,
            "pages": parsed.pages,
            # Computed once here rather than on every scan that reads this row.
            # It is derived from the blocks below, so it cannot disagree with
            # them; FORMAT covers the shape changing.
            "text": parsed.text,
            "warnings": list(parsed.warnings),
            "blocks": [
                {
                    "kind": block.kind.value,
                    "text": block.text,
                    "heading_path": list(block.heading_path),
                    "locator": block.locator,
                    "level": block.level,
                }
                for block in parsed.blocks
            ],
        },
        separators=(",", ":"),
    )


def _from_json(raw: str) -> ParsedDocument | None:
    """A stored parse, or None if this row cannot be trusted.

    Anything unreadable is a miss rather than an error: the file it came from
    is still on disk, so the worst case is paying for the parse again, and
    that is much better than a crash or - far worse - half a document.
    """
    try:
        payload = json.loads(raw)
        blocks = tuple(
            Block(
                kind=BlockKind(block["kind"]),
                text=block["text"],
                heading_path=tuple(block["heading_path"]),
                locator=block["locator"],
                level=block["level"],
            )
            for block in payload["blocks"]
        )
        return ParsedDocument(
            blocks=blocks,
            title=payload["title"],
            pages=int(payload["pages"]),
            warnings=tuple(payload["warnings"]),
            flattened=payload["text"],
        )
    except (TypeError, ValueError, KeyError):
        return None


class ParseCache:
    """Parsed documents by content hash. Never raises into a parse."""

    def __init__(self, path: str | Path = ":memory:") -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        self.hits = 0
        self.misses = 0
        with self._lock:
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __enter__(self) -> ParseCache:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def get(self, key: str) -> ParsedDocument | None:
        with self._lock:
            try:
                row = self._conn.execute(
                    "SELECT parsed FROM parses WHERE key = ?", (key,)
                ).fetchone()
            except sqlite3.Error:
                return None
            if row is None:
                self.misses += 1
                return None
            parsed = _from_json(row["parsed"])
            if parsed is None:
                self.misses += 1
                return None
            self.hits += 1
            return parsed

    def put(self, key: str, parsed: ParsedDocument) -> None:
        """Remember a parse. A failure here costs speed, never correctness."""
        import time

        try:
            with self._lock:
                self._conn.execute(
                    "INSERT INTO parses (key, parsed, stored_at) VALUES (?, ?, ?)"
                    " ON CONFLICT(key) DO UPDATE SET"
                    "   parsed=excluded.parsed, stored_at=excluded.stored_at",
                    (key, _as_json(parsed), time.time()),
                )
                self._conn.commit()
        except sqlite3.Error as exc:  # pragma: no cover - disk full, read-only db
            log.warning("could not cache a parse: %s", exc)

    def keep_only(self, keys: set[str]) -> int:
        """Forget parses of bytes the corpus no longer contains.

        Called with everything one whole scan saw. Without it this grows by a
        row per edit for ever - every draft of a policy that was ever saved
        into the folder, kept because it once existed.
        """
        try:
            with self._lock:
                rows = self._conn.execute("SELECT key FROM parses").fetchall()
                stale = [row["key"] for row in rows if row["key"] not in keys]
                for key in stale:
                    self._conn.execute("DELETE FROM parses WHERE key = ?", (key,))
                self._conn.commit()
        except sqlite3.Error:
            return 0
        return len(stale)

    def __len__(self) -> int:
        with self._lock:
            try:
                row = self._conn.execute("SELECT COUNT(*) AS n FROM parses").fetchone()
            except sqlite3.Error:
                return 0
            return int(row["n"])
