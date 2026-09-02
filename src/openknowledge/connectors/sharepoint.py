"""SharePoint document libraries, mirrored through Microsoft Graph.

The shape: a library is synced into ``<documents>/sharepoint/<library>/…`` as
ordinary files, and the local-files connector reads them like anything else -
same parsers, same parse cache, same rescan timer - with one difference: each
mirrored file carries the readers SharePoint says it has, mapped onto the
``user:<id>`` / ``group:<id>`` principals sign-in already puts in a session.
Nothing is re-invented on the answering side; the corpus simply has more
files in it, each stamped with who may read it.

Two decisions are load-bearing.

*Changes only.* Each library is read through Graph's ``delta`` feed: the first
run walks everything, every later run asks "what changed since the link you
gave me" and downloads only that. Permissions are asked per file when a file
is new or changed, and re-asked on a slow cadence for unchanged files, so a
revoked grant is honoured within that bound rather than never.

*Fail closed.* A grant this connector cannot map - a SharePoint site group, a
device, a shape Graph has not documented - is never widened into "everyone".
Its readers are simply not added; if that leaves a file with no mappable
reader at all, the file is stamped with a principal nobody holds and stays
indexed but invisible, and the sync says how many files that happened to.
The local convention that an empty principal set means public is exactly the
convention a permissions connector must never fall into by accident.

Honesty about what this is: built and tested against ``tests/fake_graph.py``,
whose payloads follow Microsoft's documented shapes for sites, drives, delta,
content and permissions. It has not yet been run against a real tenant; the
first such run is the measurement this module still owes (ROADMAP item 8).
"""

from __future__ import annotations

import json
import logging
import os
import re
import sqlite3
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import unquote

import httpx

from ..documents import is_supported

log = logging.getLogger(__name__)

GRAPH_URL = "https://graph.microsoft.com/v1.0"
LOGIN_URL = "https://login.microsoftonline.com"
#: The folder under the documents root that the mirror owns.
MIRROR_FOLDER = "sharepoint"
#: Stamped on a file whose readers could not be mapped: nobody holds it, so
#: the file is indexed and shown to no one rather than to everyone.
WITHHELD = "sharepoint:unmapped"
#: Retry-After when Graph throttles and does not say how long.
_DEFAULT_BACKOFF = 2.0
_MAX_BACKOFF = 60.0
_BAD_SEGMENT = re.compile(r'[<>:"|?*\x00-\x1f]')


class GraphError(RuntimeError):
    """Graph answered with something the sync cannot act on."""


class GraphGone(GraphError):
    """410: the delta link has expired and the library must be re-read."""


@dataclass(frozen=True, slots=True)
class GraphConfig:
    tenant_id: str
    client_id: str
    client_secret: str
    #: Path-addressed site, e.g. ``contoso.sharepoint.com:/sites/HR``.
    site: str
    #: Library display names to sync; empty means every library on the site.
    drives: tuple[str, ...] = ()
    graph_url: str = GRAPH_URL
    login_url: str = LOGIN_URL
    timeout: float = 30.0


class GraphClient:
    """The few Graph calls a mirror needs, with the retries Graph expects.

    App-only: a client-credentials token for ``https://graph.microsoft.com/.default``,
    cached until a minute before it expires. Throttling (429, 503) is retried
    after ``Retry-After``; an expired token is refreshed once; a 410 on a
    delta link is raised as :class:`GraphGone` so the sync re-reads the
    library instead of failing.
    """

    def __init__(
        self,
        config: GraphConfig,
        *,
        http: httpx.Client | None = None,
        clock: Callable[[], float] = time.time,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.config = config
        self._http = http or httpx.Client(timeout=config.timeout, follow_redirects=True)
        self._clock = clock
        self._sleep = sleep
        self._token: str | None = None
        self._expires_at = 0.0
        #: How many times a token was fetched - what a test reads to prove a
        #: 401 was answered with a refresh rather than a retry of the same token.
        self.token_fetches = 0

    # -- auth ------------------------------------------------------------------

    def token(self) -> str:
        if self._token is None or self._clock() >= self._expires_at:
            self._refresh_token()
        assert self._token is not None
        return self._token

    def _refresh_token(self) -> None:
        url = f"{self.config.login_url.rstrip('/')}/{self.config.tenant_id}/oauth2/v2.0/token"
        response = self._http.post(
            url,
            data={
                "client_id": self.config.client_id,
                "client_secret": self.config.client_secret,
                "scope": "https://graph.microsoft.com/.default",
                "grant_type": "client_credentials",
            },
        )
        self.token_fetches += 1
        if response.status_code != 200:
            raise GraphError(
                f"token request failed: HTTP {response.status_code} {response.text[:200]}"
            )
        body = response.json()
        self._token = str(body["access_token"])
        self._expires_at = self._clock() + float(body.get("expires_in", 3600)) - 60.0

    # -- requests ------------------------------------------------------------------

    def _request(self, method: str, url: str, **kwargs: object) -> httpx.Response:
        refreshed = False
        for attempt in range(6):
            headers = {"Authorization": f"Bearer {self.token()}"}
            response = self._http.request(method, url, headers=headers, **kwargs)  # type: ignore[arg-type]
            if response.status_code == 401 and not refreshed:
                refreshed = True
                self._token = None
                continue
            if response.status_code in (429, 503) and attempt < 5:
                wait = _DEFAULT_BACKOFF
                header = response.headers.get("Retry-After")
                if header and header.isdigit():
                    wait = min(float(header), _MAX_BACKOFF)
                self._sleep(wait)
                continue
            if response.status_code == 410:
                raise GraphGone(f"HTTP 410 for {url}: the delta link has expired")
            if response.status_code >= 400:
                raise GraphError(f"HTTP {response.status_code} for {url}: {response.text[:200]}")
            return response
        raise GraphError(f"gave up on {url} after repeated throttling")

    def get_json(self, url: str) -> dict:
        response = self._request("GET", url)
        try:
            body = response.json()
        except ValueError as exc:
            raise GraphError(f"{url} did not return JSON") from exc
        if not isinstance(body, dict):
            raise GraphError(f"{url} returned {type(body).__name__}, not an object")
        return body

    def _collect(self, url: str) -> list[dict]:
        """Every page of a ``value`` collection."""
        rows: list[dict] = []
        next_url: str | None = url
        while next_url:
            body = self.get_json(next_url)
            rows.extend(body.get("value") or [])
            next_url = body.get("@odata.nextLink")
        return rows

    # -- what the mirror asks --------------------------------------------------------

    def site_id(self) -> str:
        body = self.get_json(f"{self.config.graph_url}/sites/{self.config.site}")
        site = body.get("id")
        if not site:
            raise GraphError(f"site {self.config.site!r} has no id in Graph's answer")
        return str(site)

    def drives(self, site_id: str) -> list[dict]:
        """The document libraries on the site, filtered to the configured names."""
        found = self._collect(f"{self.config.graph_url}/sites/{site_id}/drives")
        if self.config.drives:
            wanted = {name.casefold() for name in self.config.drives}
            found = [d for d in found if str(d.get("name", "")).casefold() in wanted]
        return found

    def delta(self, drive_id: str, link: str | None) -> tuple[list[dict], str | None]:
        """Every changed item since ``link`` (or every item, without one), and the new link."""
        url = link or f"{self.config.graph_url}/drives/{drive_id}/root/delta"
        items: list[dict] = []
        delta_link: str | None = None
        next_url: str | None = url
        while next_url:
            body = self.get_json(next_url)
            items.extend(body.get("value") or [])
            delta_link = body.get("@odata.deltaLink") or delta_link
            next_url = body.get("@odata.nextLink")
        return items, delta_link

    def permissions(self, drive_id: str, item_id: str) -> list[dict]:
        url = f"{self.config.graph_url}/drives/{drive_id}/items/{item_id}/permissions"
        return self._collect(url)

    def content(self, drive_id: str, item_id: str) -> bytes:
        response = self._request(
            "GET", f"{self.config.graph_url}/drives/{drive_id}/items/{item_id}/content"
        )
        return response.content


# -- permissions to principals -----------------------------------------------------------


def principals_from(permissions: list[dict]) -> tuple[frozenset[str], int]:
    """Who may read an item, in the vocabulary sign-in uses, and how many grants
    could not be expressed in it.

    Entra users and groups map one to one. A sharing link scoped to the
    organisation (or, more permissively, to anyone) becomes ``authenticated``.
    A SharePoint site group, a device, an unknown shape - anything else - is
    counted and dropped: readers the mapping cannot name are not added, and a
    file left with no mappable reader is stamped :data:`WITHHELD`. Application
    grants (this connector's own, typically) are neither readers nor a
    reason to withhold.
    """
    mapped: set[str] = set()
    unmapped = 0
    grants = 0
    for permission in permissions:
        identities: list[dict] = []
        direct = permission.get("grantedToV2") or permission.get("grantedTo")
        if direct:
            identities.append(direct)
        identities.extend(
            permission.get("grantedToIdentitiesV2") or permission.get("grantedToIdentities") or []
        )
        link = permission.get("link")
        if link is not None:
            grants += 1
            scope = link.get("scope")
            if scope in ("organization", "anonymous"):
                mapped.add("authenticated")
            elif scope != "users":  # "users" links name their people below
                unmapped += 1
        for identity in identities:
            user = identity.get("user") or {}
            group = identity.get("group") or {}
            if user.get("id"):
                grants += 1
                mapped.add(f"user:{user['id']}")
            elif group.get("id"):
                grants += 1
                mapped.add(f"group:{group['id']}")
            elif identity.get("application"):
                continue
            else:
                grants += 1
                unmapped += 1
    if not mapped:
        return frozenset({WITHHELD}), unmapped
    return frozenset(mapped), unmapped


# -- what the sync remembers -----------------------------------------------------------

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
        self._lock = threading.Lock()

    def close(self) -> None:
        self._conn.close()

    def drive(self, drive_id: str) -> sqlite3.Row | None:
        return self._conn.execute("SELECT * FROM drives WHERE drive_id = ?", (drive_id,)).fetchone()

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
        rows = self._conn.execute("SELECT * FROM items WHERE drive_id = ?", (drive_id,)).fetchall()
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
        rows = self._conn.execute("SELECT relative_path, principals FROM items").fetchall()
        return {r["relative_path"]: frozenset(json.loads(r["principals"])) for r in rows}

    def counts(self) -> tuple[int, int, int]:
        """Documents mirrored, withheld, and grants left unmapped."""
        row = self._conn.execute(
            "SELECT COUNT(*) AS n, SUM(principals LIKE ?) AS withheld, SUM(unmapped) AS unmapped"
            " FROM items",
            (f'%"{WITHHELD}"%',),
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


# -- the sync --------------------------------------------------------------------------


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


def _safe_segment(name: str) -> str:
    cleaned = _BAD_SEGMENT.sub("_", name).strip().rstrip(".")
    return cleaned or "_"


class SharePointSync:
    """Mirror the configured libraries into the documents folder.

    ``refusal`` is a sentence the wiring sets when the sync must not run - the
    one case today being sign-in off, where no principal can be enforced and a
    mirrored library would be readable by whoever reaches the widget.
    """

    def __init__(
        self,
        graph: GraphClient,
        *,
        documents_dir: str | Path,
        store: SyncStore,
        permissions_refresh_seconds: float = 3600.0,
        clock: Callable[[], float] = time.time,
        refusal: str | None = None,
    ) -> None:
        self.graph = graph
        self.documents_dir = Path(documents_dir)
        self.store = store
        self.permissions_refresh_seconds = permissions_refresh_seconds
        self._clock = clock
        self.refusal = refusal
        self._running = threading.Lock()

    @property
    def mirror_root(self) -> Path:
        return self.documents_dir / MIRROR_FOLDER

    def principals_map(self) -> dict[str, frozenset[str]]:
        return self.store.principals_map()

    def owns(self, relative_path: str) -> bool:
        """Whether a documents-folder path is the mirror's to change."""
        return relative_path == MIRROR_FOLDER or relative_path.startswith(MIRROR_FOLDER + "/")

    def status(self) -> dict[str, object]:
        documents, withheld, unmapped = self.store.counts()
        return {
            "site": self.graph.config.site,
            "last_sync_at": self.store.get_status("last_sync_at"),
            "last_error": self.store.get_status("last_error"),
            "last_summary": self.store.get_status("last_summary"),
            "documents": documents,
            "withheld": withheld,
            "unmapped_grants": unmapped,
            "refusal": self.refusal,
        }

    def run(self) -> SyncSummary:
        summary = SyncSummary()
        started = self._clock()
        if self.refusal:
            summary.errors.append(self.refusal)
            self._record(summary, started)
            return summary
        if not self._running.acquire(blocking=False):
            summary.errors.append("a sync is already running")
            return summary
        try:
            try:
                site = self.graph.site_id()
                drives = self.graph.drives(site)
            except (GraphError, httpx.HTTPError) as exc:
                summary.errors.append(str(exc))
                self._record(summary, started)
                return summary
            summary.drives = len(drives)
            for drive in drives:
                try:
                    self._sync_drive(drive, summary)
                except (GraphError, httpx.HTTPError, OSError) as exc:
                    summary.errors.append(f"{drive.get('name', drive.get('id'))}: {exc}")
            self._record(summary, started)
            return summary
        finally:
            self._running.release()

    def _record(self, summary: SyncSummary, started: float) -> None:
        summary.took_seconds = self._clock() - started
        summary.documents, summary.withheld, summary.unmapped_grants = self.store.counts()
        self.store.set_status("last_sync_at", self._clock())
        self.store.set_status("last_error", summary.errors[0] if summary.errors else None)
        self.store.set_status("last_summary", summary.as_dict())
        if summary.errors:
            log.warning("sharepoint sync: %s", "; ".join(summary.errors))

    def _sync_drive(self, drive: dict, summary: SyncSummary) -> None:
        drive_id = str(drive["id"])
        name = str(drive.get("name") or drive_id)
        folder = f"{MIRROR_FOLDER}/{_safe_segment(name)}"
        known_drive = self.store.drive(drive_id)
        link = known_drive["delta_link"] if known_drive is not None else None
        try:
            items, new_link = self.graph.delta(drive_id, link)
        except GraphGone:
            log.info("sharepoint: delta link for %s expired; re-reading the library", name)
            self._forget(drive_id)
            items, new_link = self.graph.delta(drive_id, None)
        if known_drive is not None and known_drive["folder"] != folder:
            self._forget(drive_id)  # the library was renamed; its mirror folder moves with it

        known = self.store.items_for(drive_id)
        now = self._clock()
        touched: set[str] = set()
        for item in items:
            item_id = str(item.get("id", ""))
            if not item_id:
                continue
            touched.add(item_id)
            if "deleted" in item:
                self._remove(known.get(item_id), summary)
                continue
            if "file" not in item:
                continue
            relative = self._relative_path(folder, item)
            if relative is None:
                summary.skipped += 1
                continue
            row = known.get(item_id)
            etag = str(item.get("eTag") or item.get("cTag") or "")
            target = self.documents_dir / relative
            if row is not None and row.relative_path != relative:
                self._unlink(self.documents_dir / row.relative_path)
                if row.etag == etag and not target.exists():
                    # Moved or renamed, same bytes: no download needed.
                    pass
            fresh = row is None or row.etag != etag or not target.is_file()
            if fresh:
                data = self.graph.content(drive_id, item_id)
                self._write(target, data)
                if row is None:
                    summary.added += 1
                else:
                    summary.updated += 1
            else:
                summary.unchanged += 1

            stale = row is None or (now - row.permissions_at) >= self.permissions_refresh_seconds
            if row is not None and not fresh and not stale:
                principals, unmapped = row.principals, row.unmapped
                permissions_at = row.permissions_at
            else:
                principals, unmapped = principals_from(self.graph.permissions(drive_id, item_id))
                summary.permissions_read += 1
                permissions_at = now
            self.store.upsert(
                ItemRow(
                    item_id=item_id,
                    drive_id=drive_id,
                    relative_path=relative,
                    etag=etag,
                    principals=principals,
                    unmapped=unmapped,
                    permissions_at=permissions_at,
                )
            )
        # A revoked grant changes nothing the delta feed reports, so files the
        # feed did not mention have their readers re-read on a clock. That is
        # the staleness bound on a revocation: the refresh interval, not never.
        for item_id, row in known.items():
            if item_id in touched or (now - row.permissions_at) < self.permissions_refresh_seconds:
                continue
            principals, unmapped = principals_from(self.graph.permissions(drive_id, item_id))
            summary.permissions_read += 1
            self.store.upsert(
                ItemRow(
                    item_id=item_id,
                    drive_id=drive_id,
                    relative_path=row.relative_path,
                    etag=row.etag,
                    principals=principals,
                    unmapped=unmapped,
                    permissions_at=now,
                )
            )
        self.store.set_drive(drive_id, name, folder, new_link, now)

    def _relative_path(self, folder: str, item: dict) -> str | None:
        """Where the item lives under the documents folder, or None if it cannot be placed."""
        name = str(item.get("name") or "")
        parent = str((item.get("parentReference") or {}).get("path") or "")
        _, _, tail = parent.partition("root:")
        segments = [unquote(part) for part in tail.split("/") if part]
        segments.append(name)
        if not name or any(part in (".", "..") for part in segments):
            return None
        if not is_supported(name):
            return None
        return "/".join([folder, *(_safe_segment(part) for part in segments)])

    def _write(self, target: Path, data: bytes) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(target.name + ".part")
        tmp.write_bytes(data)
        os.replace(tmp, target)

    def _remove(self, row: ItemRow | None, summary: SyncSummary) -> None:
        if row is None:
            return
        self._unlink(self.documents_dir / row.relative_path)
        self.store.remove(row.item_id)
        summary.removed += 1

    def _unlink(self, path: Path) -> None:
        try:
            path.unlink()
        except FileNotFoundError:
            return
        # Empty folders left behind are noise in the listing; prune upward.
        parent = path.parent
        while parent != self.mirror_root and parent != self.documents_dir:
            try:
                parent.rmdir()
            except OSError:
                break
            parent = parent.parent

    def _forget(self, drive_id: str) -> None:
        for row in self.store.items_for(drive_id).values():
            self._unlink(self.documents_dir / row.relative_path)
        self.store.forget_drive(drive_id)
