"""Server-side sessions for signed-in people.

Sessions live in SQLite rather than in the cookie because Entra group claims
are lists of GUIDs that do not fit in one - and because a session the server
holds is a session the server can end. The browser carries only an opaque
token; the database stores its SHA-256, so a copied database file yields no
usable cookies.

Two tables, both boring on purpose:

``sessions``
    Who signed in, their groups, and when it stops being true.

``pending_logins``
    The state/nonce/verifier triple minted at the redirect out, consumed
    exactly once by the callback. Single use is what makes a replayed
    callback fail instead of minting a second session.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import secrets
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from .oidc import Identity, PendingLogin

#: How long a login redirect may take before the callback refuses it.
#: Generous enough for a first-ever consent screen, short enough that a
#: forgotten tab's state is worthless by lunch.
PENDING_MAX_AGE_SECONDS = 600.0

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    token_hash  TEXT PRIMARY KEY,
    subject     TEXT NOT NULL,
    name        TEXT NOT NULL,
    groups_json TEXT NOT NULL DEFAULT '[]',
    email       TEXT NOT NULL DEFAULT '',
    created_at  REAL NOT NULL,
    expires_at  REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS pending_logins (
    state         TEXT PRIMARY KEY,
    nonce         TEXT NOT NULL,
    code_verifier TEXT NOT NULL,
    created_at    REAL NOT NULL,
    next_path     TEXT NOT NULL DEFAULT '/'
);
"""


def _now() -> float:
    """Wall time behind one seam, so tests age sessions without sleeping."""
    return time.time()


@dataclass(frozen=True, slots=True)
class Session:
    """A signed-in person, as the rest of the server sees them."""

    subject: str
    name: str
    groups: tuple[str, ...]
    expires_at: float
    email: str = ""

    @property
    def principals(self) -> frozenset[str]:
        """What this identity may read, in the ACL machinery's vocabulary.

        ``authenticated`` lets a document say "anyone signed in";
        ``user:{id}`` and ``group:{id}`` say someone in particular. Minted
        here and only here - the server never accepts these from the wire
        while sign-in is on.

        A verified email is minted as a second ``user:`` principal, because
        not every source names people by directory id: Google Drive grants
        read to an address. An address and a directory id can never collide,
        so both live in one namespace safely, and a person matches only
        their own verified address.
        """
        found = {"authenticated", f"user:{self.subject}", *(f"group:{g}" for g in self.groups)}
        if self.email:
            found.add(f"user:{self.email}")
        return frozenset(found)


class SessionStore:
    """Sessions and pending logins over one SQLite file."""

    def __init__(self, path: str | Path = ":memory:") -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False + an explicit lock: FastAPI serves requests
        # on a thread pool, and SQLite is happy to be shared as long as we
        # serialise writes ourselves.
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        with self._lock:
            self._conn.executescript(_SCHEMA)
            # A database written before sessions carried an email has the
            # table without the column. Adding it is idempotent and keeps a
            # running install's people signed in across the upgrade.
            with contextlib.suppress(sqlite3.OperationalError):
                self._conn.execute("ALTER TABLE sessions ADD COLUMN email TEXT NOT NULL DEFAULT ''")
            # A database created before next_path existed gains the column;
            # CREATE IF NOT EXISTS cannot add it to an existing table.
            columns = {
                row["name"] for row in self._conn.execute("PRAGMA table_info(pending_logins)")
            }
            if "next_path" not in columns:
                self._conn.execute(
                    "ALTER TABLE pending_logins ADD COLUMN next_path TEXT NOT NULL DEFAULT '/'"
                )
                self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # -- sessions ----------------------------------------------------------

    def create(self, identity: Identity, *, ttl_seconds: float) -> str:
        """Mint a session and return the opaque token the cookie will carry."""
        token = secrets.token_urlsafe(32)
        now = _now()
        with self._lock:
            self._conn.execute(
                "INSERT INTO sessions (token_hash, subject, name, groups_json, email,"
                " created_at, expires_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    _hashed(token),
                    identity.subject,
                    identity.name,
                    json.dumps(list(identity.groups)),
                    identity.email,
                    now,
                    now + ttl_seconds,
                ),
            )
            self._conn.commit()
        return token

    def get(self, token: str) -> Session | None:
        """The live session for this token, or None. Expired rows are removed."""
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM sessions WHERE token_hash = ?", (_hashed(token),)
            ).fetchone()
            if row is None:
                return None
            if row["expires_at"] <= _now():
                self._conn.execute("DELETE FROM sessions WHERE token_hash = ?", (_hashed(token),))
                self._conn.commit()
                return None
        return Session(
            subject=row["subject"],
            name=row["name"],
            groups=tuple(json.loads(row["groups_json"])),
            expires_at=row["expires_at"],
            email=row["email"],
        )

    def delete(self, token: str) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM sessions WHERE token_hash = ?", (_hashed(token),))
            self._conn.commit()

    # -- pending logins ----------------------------------------------------

    def save_pending(self, pending: PendingLogin) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO pending_logins (state, nonce, code_verifier,"
                " created_at, next_path) VALUES (?, ?, ?, ?, ?)",
                (
                    pending.state,
                    pending.nonce,
                    pending.code_verifier,
                    pending.created_at,
                    pending.next_path,
                ),
            )
            self._conn.commit()

    def take_pending(self, state: str) -> PendingLogin | None:
        """The pending login for this state - removed as it is read.

        Single use by construction: a second callback with the same state
        finds nothing, so a replayed or forged callback cannot mint a
        session. Stale entries are refused the same way.
        """
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM pending_logins WHERE state = ?", (state,)
            ).fetchone()
            if row is None:
                return None
            self._conn.execute("DELETE FROM pending_logins WHERE state = ?", (state,))
            self._conn.commit()
        if _now() - row["created_at"] > PENDING_MAX_AGE_SECONDS:
            return None
        return PendingLogin(
            state=row["state"],
            nonce=row["nonce"],
            code_verifier=row["code_verifier"],
            created_at=row["created_at"],
            next_path=row["next_path"],
        )


def _hashed(token: str) -> str:
    return hashlib.sha256(token.encode("ascii", errors="replace")).hexdigest()
