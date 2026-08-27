"""People who asked to be contacted, stored where the deployment can see them.

OpenKnowledge collects nothing. That is the promise the whole project is built
on, and a marketing site with a contact form is the obvious place to quietly
break it - a third-party form service, an analytics tag, an email hosted
somewhere the operator never chose.

So the form on the website posts here, to the same container that serves the
page, into a SQLite file next to everything else. Nobody's details reach a
service the operator did not set up. If the page is hosted statically with no
endpoint behind it, the form says so and points at the issue tracker rather than
failing silently or pretending to succeed.

Off unless an operator turns it on. A running answer engine has no business
accepting public writes by default, and most deployments serve the widget
internally and never need this at all.
"""

from __future__ import annotations

import re
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path

#: Deliberately permissive. Rejecting valid-but-unusual addresses is a worse
#: failure than accepting a junk one, which a human deletes in a second.
_EMAIL = re.compile(r"^[^@\s]+@[^@\s.]+\.[^@\s]+$")

#: Long enough for anyone with something to say, short enough that the table
#: cannot be used as free storage.
_MAX = {"name": 200, "email": 320, "organisation": 200, "interest": 60, "message": 4000}

_SCHEMA = """
CREATE TABLE IF NOT EXISTS contacts (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    ts           REAL NOT NULL,
    name         TEXT NOT NULL,
    email        TEXT NOT NULL,
    organisation TEXT,
    interest     TEXT,
    message      TEXT,
    source       TEXT
);
CREATE INDEX IF NOT EXISTS idx_contacts_ts ON contacts(ts);
"""


class ContactError(ValueError):
    """The submission was not usable. The message is shown to the sender."""


@dataclass(frozen=True, slots=True)
class Contact:
    id: int
    ts: float
    name: str
    email: str
    organisation: str = ""
    interest: str = ""
    message: str = ""
    source: str = ""


def clean(payload: dict[str, object]) -> dict[str, str]:
    """Validate and trim a submission, or raise with a reason worth showing.

    The honeypot is checked by the caller, not here: a bot that fills a hidden
    field should be answered with a cheerful 200 and dropped, because telling it
    what failed is how it learns to pass.
    """
    fields = {key: str(payload.get(key) or "").strip() for key in _MAX}

    if not fields["name"]:
        raise ContactError("a name is required")
    if not _EMAIL.match(fields["email"]):
        raise ContactError("that does not look like an email address")

    for key, limit in _MAX.items():
        if len(fields[key]) > limit:
            raise ContactError(f"{key} is longer than {limit} characters")
    return fields


class ContactStore:
    """SQLite-backed, in its own file so it can be handled separately.

    Kept out of the answer store on purpose: one holds questions employees
    asked, the other holds people who want to be emailed, and they have
    different retention rules and different people who should be able to read
    them.
    """

    def __init__(self, path: str | Path = ":memory:") -> None:
        self._lock = threading.Lock()
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def add(self, fields: dict[str, str], *, source: str = "website") -> int:
        with self._lock:
            cursor = self._conn.execute(
                "INSERT INTO contacts (ts, name, email, organisation, interest, message, source)"
                " VALUES (?,?,?,?,?,?,?)",
                (
                    time.time(),
                    fields["name"],
                    fields["email"],
                    fields.get("organisation", ""),
                    fields.get("interest", ""),
                    fields.get("message", ""),
                    source,
                ),
            )
            self._conn.commit()
            return int(cursor.lastrowid or 0)

    def recent(self, limit: int = 50) -> list[Contact]:
        rows = self._conn.execute(
            "SELECT * FROM contacts ORDER BY ts DESC LIMIT ?", (limit,)
        ).fetchall()
        return [
            Contact(
                id=r["id"],
                ts=r["ts"],
                name=r["name"],
                email=r["email"],
                organisation=r["organisation"] or "",
                interest=r["interest"] or "",
                message=r["message"] or "",
                source=r["source"] or "",
            )
            for r in rows
        ]

    def count(self) -> int:
        return int(self._conn.execute("SELECT COUNT(*) FROM contacts").fetchone()[0])

    def submissions_since(self, since: float) -> int:
        """Used to rate-limit. A public write endpoint without one is an invitation."""
        row = self._conn.execute("SELECT COUNT(*) FROM contacts WHERE ts >= ?", (since,)).fetchone()
        return int(row[0])
