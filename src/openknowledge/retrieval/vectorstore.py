"""Chunk vectors on disk, so a restart is not a re-embedding.

Embedding a corpus is the one genuinely slow part of indexing - minutes for
ten thousand chunks on a laptop - and almost all of it is repeated work: an
edit to one document leaves every other paragraph byte-identical. Keyed on the
text and the model, an unchanged paragraph is embedded exactly once, ever.

Its own file, not the answer store. These are derived, disposable and much the
largest thing here: deleting vectors.db costs one re-embed and nothing else,
which is not true of answers people have approved.
"""

from __future__ import annotations

import array
import sqlite3
import threading
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS vectors (
    key   TEXT PRIMARY KEY,
    dims  INTEGER NOT NULL,
    value BLOB NOT NULL
);
"""


class VectorCache:
    """A dict-shaped SQLite table. Reads once at open, writes as it goes."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._db = sqlite3.connect(self._path, check_same_thread=False)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.executescript(_SCHEMA)
        self._db.commit()

    def load(self) -> dict[str, list[float]]:
        """Every cached vector. Called once per index build, not per query."""
        with self._lock:
            rows = self._db.execute("SELECT key, value FROM vectors").fetchall()
        out: dict[str, list[float]] = {}
        for key, blob in rows:
            values = array.array("f")
            values.frombytes(blob)
            out[key] = list(values)
        return out

    def put_many(self, vectors: dict[str, list[float]]) -> None:
        if not vectors:
            return
        rows = [
            (key, len(value), array.array("f", value).tobytes()) for key, value in vectors.items()
        ]
        with self._lock:
            self._db.executemany(
                "INSERT OR REPLACE INTO vectors (key, dims, value) VALUES (?, ?, ?)", rows
            )
            self._db.commit()

    def evict_all(self) -> int:
        """Drop everything - what a model change means. Returns how many went."""
        with self._lock:
            count = self._db.execute("SELECT count(*) FROM vectors").fetchone()[0]
            self._db.execute("DELETE FROM vectors")
            self._db.commit()
        return int(count)

    def __len__(self) -> int:
        with self._lock:
            return int(self._db.execute("SELECT count(*) FROM vectors").fetchone()[0])

    def close(self) -> None:
        with self._lock:
            self._db.close()
