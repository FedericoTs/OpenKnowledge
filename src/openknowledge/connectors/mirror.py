"""What a mirrored source remembers between syncs, and how it says what it did.

SharePoint and Google Drive are the same shape of problem: read a remote
library through a changes feed, keep the files in the documents folder, and
stamp each one with the readers the source says it has. What differs is the
API. What does not is the bookkeeping - which sync token each source is up
to, which remote item became which local file, what its readers were and
when they were last asked - so that lives here, once.

``WITHHELD`` is the load-bearing constant. A file whose readers a connector
cannot express is stamped with a principal nobody holds: indexed, listed,
and shown to no one, rather than falling into the local convention that an
empty principal set means public. A permissions connector must never widen a
grant it did not understand.
"""

from __future__ import annotations

import json
import re
import sqlite3
import threading
from dataclasses import dataclass, field
from pathlib import Path

#: Stamped on a file whose readers could not be mapped: nobody holds it, so
#: the file is indexed and shown to no one rather than to everyone.
WITHHELD = "mirror:unmapped"
#: What v0.10.0's SharePoint sync stamped before this bookkeeping was shared.
#: Equally held by nobody, so such a file was never visible; counted here too
#: so the number an admin reads is right before the next permissions refresh
#: rewrites those rows.
LEGACY_WITHHELD = "sharepoint:unmapped"
_BAD_SEGMENT = re.compile(r'[<>:"|?*\x00-\x1f]')


_SCHEMA = """
CREATE TABLE IF NOT EXISTS drives (
    drive_id   TEXT PRIMARY KEY,
    name       TEXT NOT NULL,
    folder     TEXT NOT NULL,
    delta_link TEXT,
    synced_at  REAL
);
CREATE TABLE IF NOT EXISTS items (
    item_id        TEXT PRIMARY KEY,
    drive_id       TEXT NOT NULL,
    relative_path  TEXT NOT NULL,
    etag           TEXT NOT NULL,
    principals     TEXT NOT NULL,
    unmapped       INTEGER NOT NULL DEFAULT 0,
    permissions_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS status (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


@dataclass(frozen=True, slots=True)
class ItemRow:
    item_id: str
    drive_id: str
    relative_path: str
    etag: str
    principals: frozenset[str]
    unmapped: int
    permissions_at: float


class SyncStore:
    """Delta links, mirrored items and their principals, in one SQLite file."""

    def __init__(self, path: str | Path = ":memory:") -> None:
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._lock = threading.RLock()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def drive(self, drive_id: str) -> sqlite3.Row | None:
        with self._lock:
            return self._conn.execute(
                "SELECT * FROM drives WHERE drive_id = ?", (drive_id,)
            ).fetchone()

    def set_drive(
        self, drive_id: str, name: str, folder: str, delta_link: str | None, now: float
    ) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO drives (drive_id, name, folder, delta_link, synced_at)"
                " VALUES (?, ?, ?, ?, ?)"
                " ON CONFLICT(drive_id) DO UPDATE SET name = excluded.name,"
                " folder = excluded.folder, delta_link = excluded.delta_link,"
                " synced_at = excluded.synced_at",
                (drive_id, name, folder, delta_link, now),
            )
            self._conn.commit()

    def items_for(self, drive_id: str) -> dict[str, ItemRow]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM items WHERE drive_id = ?", (drive_id,)
            ).fetchall()
            return {r["item_id"]: self._row(r) for r in rows}

    def upsert(self, row: ItemRow) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO items (item_id, drive_id, relative_path, etag, principals, unmapped,"
                " permissions_at) VALUES (?, ?, ?, ?, ?, ?, ?)"
                " ON CONFLICT(item_id) DO UPDATE SET drive_id = excluded.drive_id,"
                " relative_path = excluded.relative_path, etag = excluded.etag,"
                " principals = excluded.principals, unmapped = excluded.unmapped,"
                " permissions_at = excluded.permissions_at",
                (
                    row.item_id,
                    row.drive_id,
                    row.relative_path,
                    row.etag,
                    json.dumps(sorted(row.principals)),
                    row.unmapped,
                    row.permissions_at,
                ),
            )
            self._conn.commit()

    def remove(self, item_id: str) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM items WHERE item_id = ?", (item_id,))
            self._conn.commit()

    def forget_drive(self, drive_id: str) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM items WHERE drive_id = ?", (drive_id,))
            self._conn.execute("DELETE FROM drives WHERE drive_id = ?", (drive_id,))
            self._conn.commit()

    def principals_map(self) -> dict[str, frozenset[str]]:
        with self._lock:
            rows = self._conn.execute("SELECT relative_path, principals FROM items").fetchall()
            return {r["relative_path"]: frozenset(json.loads(r["principals"])) for r in rows}

    def counts(self) -> tuple[int, int, int]:
        """Documents mirrored, withheld, and grants left unmapped."""
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*) AS n, SUM(principals LIKE ? OR principals LIKE ?) AS withheld,"
                " SUM(unmapped) AS unmapped FROM items",
                (f'%"{WITHHELD}"%', f'%"{LEGACY_WITHHELD}"%'),
            ).fetchone()
            return int(row["n"]), int(row["withheld"] or 0), int(row["unmapped"] or 0)

    def set_status(self, key: str, value: object) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO status (key, value) VALUES (?, ?)"
                " ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, json.dumps(value)),
            )
            self._conn.commit()

    def get_status(self, key: str) -> object:
        with self._lock:
            row = self._conn.execute("SELECT value FROM status WHERE key = ?", (key,)).fetchone()
            return json.loads(row["value"]) if row else None

    @staticmethod
    def _row(r: sqlite3.Row) -> ItemRow:
        return ItemRow(
            item_id=r["item_id"],
            drive_id=r["drive_id"],
            relative_path=r["relative_path"],
            etag=r["etag"],
            principals=frozenset(json.loads(r["principals"])),
            unmapped=int(r["unmapped"]),
            permissions_at=float(r["permissions_at"]),
        )


@dataclass
class SyncSummary:
    """What one run did, so it can be printed rather than trusted."""

    drives: int = 0
    added: int = 0
    updated: int = 0
    removed: int = 0
    unchanged: int = 0
    skipped: int = 0
    permissions_read: int = 0
    documents: int = 0
    withheld: int = 0
    unmapped_grants: int = 0
    errors: list[str] = field(default_factory=list)
    took_seconds: float = 0.0

    @property
    def changed(self) -> bool:
        return bool(self.added or self.updated or self.removed)

    def as_dict(self) -> dict[str, object]:
        return {
            "drives": self.drives,
            "added": self.added,
            "updated": self.updated,
            "removed": self.removed,
            "unchanged": self.unchanged,
            "skipped": self.skipped,
            "permissions_read": self.permissions_read,
            "documents": self.documents,
            "withheld": self.withheld,
            "unmapped_grants": self.unmapped_grants,
            "errors": list(self.errors),
            "took_seconds": round(self.took_seconds, 3),
        }


def safe_segment(name: str) -> str:
    cleaned = _BAD_SEGMENT.sub("_", name).strip().rstrip(".")
    return cleaned or "_"
