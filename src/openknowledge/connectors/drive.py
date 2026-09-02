"""Google Drive shared drives, mirrored with the readers Drive gives them.

The same shape as the SharePoint mirror and the same discipline: read each
shared drive through a changes feed, keep the files in the documents folder
where the ordinary parsers handle them, and stamp every file with who may
read it. What differs is Google's API, and one thing that is not cosmetic.

**Drive names people by email; a directory names them by id.** SharePoint
hands back the same Entra object ids sign-in puts in a session, so the two
vocabularies already meet. Drive hands back ``alice@contoso.com`` and
``hr-team@contoso.com``. So this maps a user grant to ``user:<email>`` and a
group grant to ``group:<email>`` - and sign-in was taught to mint
``user:<verified email>`` alongside ``user:<subject>``, so a person matches
their own files without anything being guessed. An email principal and an
object-id principal can never collide, so both live in one namespace safely.

Group grants are the part a deployment must think about: a person matches
``group:hr-team@contoso.com`` only if their sign-in emits that group's email.
Where it does not, those files are simply not visible to them - the failure
this fails towards, not away from.

A grant this cannot express - a domain that is not the configured one, a
shape Google has not documented - is dropped, never widened. A file left with
no mappable reader is stamped :data:`~openknowledge.connectors.mirror.WITHHELD`
and shown to nobody.

Google-native documents have no bytes to download, so they are exported:
a Doc as .docx, a Sheet as .xlsx, a Slide deck as .pptx, which is what gives
the parsers real headings, tables and locators rather than a wall of text.

Built against ``tests/fake_drive.py``, whose payloads follow Google's
documented shapes. It has not yet been run against a real Workspace; see
docs/DRIVE.md.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import httpx

from ..documents import is_supported
from .mirror import WITHHELD, ItemRow, SyncStore, SyncSummary, safe_segment

log = logging.getLogger(__name__)

DRIVE_URL = "https://www.googleapis.com/drive/v3"
TOKEN_URL = "https://oauth2.googleapis.com/token"
SCOPE = "https://www.googleapis.com/auth/drive.readonly"
#: The folder under the documents root that this mirror owns.
MIRROR_FOLDER = "gdrive"
#: What a Google-native file is exported as, and the extension it lands under.
#: A form, a drawing or a shortcut has no document in it and is skipped.
EXPORTS = {
    "application/vnd.google-apps.document": (
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ".docx",
    ),
    "application/vnd.google-apps.spreadsheet": (
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        ".xlsx",
    ),
    "application/vnd.google-apps.presentation": (
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        ".pptx",
    ),
}
FOLDER_MIME = "application/vnd.google-apps.folder"
#: The fields worth asking for. Drive returns almost nothing without this.
FILE_FIELDS = "id,name,mimeType,parents,modifiedTime,md5Checksum,version,trashed,driveId"
_DEFAULT_BACKOFF = 2.0
_MAX_BACKOFF = 60.0


class DriveError(RuntimeError):
    """Drive answered with something the sync cannot act on."""


class DriveGone(DriveError):
    """The saved page token is too old: the drive must be read again."""


@dataclass(frozen=True, slots=True)
class DriveConfig:
    #: The service account's own address, from its JSON key file.
    client_email: str
    #: Its PEM private key, from the same file.
    private_key: str
    #: The person to impersonate, when domain-wide delegation is configured.
    #: Empty means act as the service account itself, which then sees only
    #: what has been shared with it.
    subject: str = ""
    #: The Workspace domain a ``domain`` grant must name to count as "anyone
    #: signed in". A grant naming any other domain is not this company's.
    domain: str = ""
    #: Shared drive ids to mirror; empty means every one the account can see.
    drive_ids: tuple[str, ...] = ()
    api_url: str = DRIVE_URL
    token_url: str = TOKEN_URL
    timeout: float = 30.0


class DriveClient:
    """The Drive calls a mirror needs, with the retries Google expects.

    Authenticates the way a service account does: a short-lived JWT signed
    with its own private key, exchanged for an access token. Nothing is
    stored but the token, and only until a minute before it expires.
    """

    def __init__(
        self,
        config: DriveConfig,
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
        self.token_fetches = 0

    def close(self) -> None:
        self._http.close()

    # -- auth ------------------------------------------------------------------

    def token(self) -> str:
        if self._token is None or self._clock() >= self._expires_at:
            self._refresh_token()
        assert self._token is not None
        return self._token

    def _refresh_token(self) -> None:
        # Imported here, not at the top: signing a service-account assertion
        # needs PyJWT, which is the `auth` extra. A base install has no PyJWT
        # and no Drive mirror, and importing it eagerly would stop the server
        # starting at all - see tests/test_base_install.py.
        import jwt  # noqa: PLC0415 - needs the auth extra

        now = int(self._clock())
        claims: dict[str, object] = {
            "iss": self.config.client_email,
            "scope": SCOPE,
            "aud": self.config.token_url,
            "iat": now,
            "exp": now + 3600,
        }
        if self.config.subject:
            # Domain-wide delegation: act as this person, so the mirror sees
            # what they see rather than only what was shared with the robot.
            claims["sub"] = self.config.subject
        try:
            assertion = jwt.encode(claims, self.config.private_key, algorithm="RS256")
        except (ValueError, TypeError, jwt.PyJWTError) as exc:  # noqa: B902
            raise DriveError(f"the service account key could not sign an assertion: {exc}") from exc
        response = self._http.post(
            self.config.token_url,
            data={
                "grant_type": "urn:ietf:params:oauth:grant-type:jwt-bearer",
                "assertion": assertion,
            },
        )
        self.token_fetches += 1
        if response.status_code != 200:
            raise DriveError(
                f"the token request failed: HTTP {response.status_code} {response.text[:200]}"
            )
        body = response.json()
        self._token = str(body["access_token"])
        self._expires_at = self._clock() + float(body.get("expires_in", 3600)) - 60.0

    # -- requests --------------------------------------------------------------

    def _request(self, url: str, *, params: dict | None = None) -> httpx.Response:
        refreshed = False
        for attempt in range(6):
            headers = {"Authorization": f"Bearer {self.token()}"}
            response = self._http.get(url, headers=headers, params=params)
            if response.status_code == 401 and not refreshed:
                refreshed = True
                self._token = None
                continue
            if response.status_code in (429, 500, 502, 503) and attempt < 5:
                wait = _DEFAULT_BACKOFF * (2**attempt)
                header = response.headers.get("Retry-After")
                if header and header.isdigit():
                    wait = float(header)
                self._sleep(min(wait, _MAX_BACKOFF))
                continue
            if response.status_code == 404 and "startPageToken" not in url:
                raise DriveError(f"HTTP 404 for {url}")
            if response.status_code >= 400:
                # A page token Drive no longer knows is the one error that
                # means "read it all again" rather than "give up".
                if response.status_code == 410 or "pageToken" in response.text[:400]:
                    raise DriveGone(f"HTTP {response.status_code} for {url}: the page token is old")
                raise DriveError(f"HTTP {response.status_code} for {url}: {response.text[:200]}")
            return response
        raise DriveError(f"gave up on {url} after repeated throttling")

    def get_json(self, url: str, params: dict | None = None) -> dict:
        response = self._request(url, params=params)
        try:
            body = response.json()
        except ValueError as exc:
            raise DriveError(f"{url} did not return JSON") from exc
        if not isinstance(body, dict):
            raise DriveError(f"{url} returned {type(body).__name__}, not an object")
        return body

    def _paged(self, url: str, params: dict, key: str) -> list[dict]:
        rows: list[dict] = []
        page: str | None = None
        while True:
            body = self.get_json(url, {**params, **({"pageToken": page} if page else {})})
            rows.extend(body.get(key) or [])
            page = body.get("nextPageToken")
            if not page:
                return rows

    # -- what the mirror asks ----------------------------------------------------

    def shared_drives(self) -> list[dict]:
        found = self._paged(
            f"{self.config.api_url}/drives", {"fields": "nextPageToken,drives(id,name)"}, "drives"
        )
        if self.config.drive_ids:
            wanted = set(self.config.drive_ids)
            found = [d for d in found if str(d.get("id")) in wanted]
        return found

    def start_page_token(self, drive_id: str) -> str:
        body = self.get_json(
            f"{self.config.api_url}/changes/startPageToken",
            {"driveId": drive_id, "supportsAllDrives": "true"},
        )
        token = body.get("startPageToken")
        if not token:
            raise DriveError(f"drive {drive_id} returned no startPageToken")
        return str(token)

    def files(self, drive_id: str) -> list[dict]:
        """Every file in the drive - the first walk, before there is a token."""
        return self._paged(
            f"{self.config.api_url}/files",
            {
                "corpora": "drive",
                "driveId": drive_id,
                "includeItemsFromAllDrives": "true",
                "supportsAllDrives": "true",
                "q": "trashed = false",
                "pageSize": "100",
                "fields": f"nextPageToken,files({FILE_FIELDS})",
            },
            "files",
        )

    def changes(self, drive_id: str, token: str) -> tuple[list[dict], str]:
        """What changed since ``token``, and the token to use next time."""
        url = f"{self.config.api_url}/changes"
        params = {
            "driveId": drive_id,
            "includeItemsFromAllDrives": "true",
            "supportsAllDrives": "true",
            "includeRemoved": "true",
            "pageSize": "100",
            "fields": (
                f"nextPageToken,newStartPageToken,changes(fileId,removed,file({FILE_FIELDS}))"
            ),
        }
        rows: list[dict] = []
        page = token
        while True:
            body = self.get_json(url, {**params, "pageToken": page})
            rows.extend(body.get("changes") or [])
            following = body.get("nextPageToken")
            if not following:
                return rows, str(body.get("newStartPageToken") or token)
            page = str(following)

    def file(self, file_id: str) -> dict:
        return self.get_json(
            f"{self.config.api_url}/files/{file_id}",
            {"supportsAllDrives": "true", "fields": FILE_FIELDS},
        )

    def permissions(self, file_id: str) -> list[dict]:
        return self._paged(
            f"{self.config.api_url}/files/{file_id}/permissions",
            {
                "supportsAllDrives": "true",
                "pageSize": "100",
                "fields": ("nextPageToken,permissions(id,type,role,emailAddress,domain,deleted)"),
            },
            "permissions",
        )

    def content(self, file_id: str, mime_type: str) -> bytes:
        """The file's bytes, exported first when Google holds it natively."""
        export = EXPORTS.get(mime_type)
        if export is not None:
            response = self._request(
                f"{self.config.api_url}/files/{file_id}/export",
                params={"mimeType": export[0]},
            )
        else:
            response = self._request(
                f"{self.config.api_url}/files/{file_id}",
                params={"alt": "media", "supportsAllDrives": "true"},
            )
        return response.content


# -- permissions to principals ----------------------------------------------------


def principals_from(permissions: Sequence[dict], *, domain: str) -> tuple[frozenset[str], int]:
    """Who may read a file, in the vocabulary sign-in mints, and what was dropped.

    Drive's roles all imply reading, so the role is not filtered on; what
    matters is who the grant names. A ``domain`` grant counts as "anyone
    signed in" only when it names the configured Workspace domain - another
    domain's people are not this company's. An ``anyone`` grant (a public
    link) is treated the same way: everyone who can reach this server is
    already inside the company, so it widens nothing. A deleted grant, or one
    naming a shape this does not know, is counted and dropped.
    """
    mapped: set[str] = set()
    unmapped = 0
    for permission in permissions:
        if permission.get("deleted"):
            continue
        kind = str(permission.get("type") or "")
        email = str(permission.get("emailAddress") or "").strip().lower()
        if kind == "user" and email:
            mapped.add(f"user:{email}")
        elif kind == "group" and email:
            mapped.add(f"group:{email}")
        elif kind == "domain":
            granted = str(permission.get("domain") or "").strip().lower()
            if domain and granted == domain.strip().lower():
                mapped.add("authenticated")
            else:
                unmapped += 1
        elif kind == "anyone":
            mapped.add("authenticated")
        else:
            unmapped += 1
    if not mapped:
        return frozenset({WITHHELD}), unmapped
    return frozenset(mapped), unmapped


# -- the sync ----------------------------------------------------------------------


class DriveSync:
    """Mirror the configured shared drives into the documents folder."""

    label = "Google Drive"

    def __init__(
        self,
        drive: DriveClient,
        *,
        documents_dir: str | Path,
        store: SyncStore,
        permissions_refresh_seconds: float = 3600.0,
        clock: Callable[[], float] = time.time,
        refusal: str | None = None,
    ) -> None:
        self.drive = drive
        self.documents_dir = Path(documents_dir)
        self.store = store
        self.permissions_refresh_seconds = permissions_refresh_seconds
        self._clock = clock
        self.refusal = refusal
        self._running = threading.Lock()
        #: folder id -> (name, parent id), so a path is built without asking twice.
        self._folders: dict[str, tuple[str, str]] = {}

    @property
    def mirror_root(self) -> Path:
        return self.documents_dir / MIRROR_FOLDER

    def principals_map(self) -> dict[str, frozenset[str]]:
        return self.store.principals_map()

    def owns(self, relative_path: str) -> bool:
        return relative_path == MIRROR_FOLDER or relative_path.startswith(MIRROR_FOLDER + "/")

    def status(self) -> dict[str, object]:
        documents, withheld, unmapped = self.store.counts()
        return {
            "subject": self.drive.config.subject or self.drive.config.client_email,
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
                drives = self.drive.shared_drives()
            except (DriveError, httpx.HTTPError) as exc:
                summary.errors.append(str(exc))
                self._record(summary, started)
                return summary
            summary.drives = len(drives)
            for drive in drives:
                try:
                    self._sync_drive(drive, summary)
                except (DriveError, httpx.HTTPError, OSError) as exc:
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
            log.warning("drive sync: %s", "; ".join(summary.errors))

    def _sync_drive(self, drive: dict, summary: SyncSummary) -> None:
        drive_id = str(drive["id"])
        name = str(drive.get("name") or drive_id)
        folder = f"{MIRROR_FOLDER}/{safe_segment(name)}"
        known_drive = self.store.drive(drive_id)
        token = known_drive["delta_link"] if known_drive is not None else None
        if known_drive is not None and known_drive["folder"] != folder:
            self._forget(drive_id)  # the drive was renamed; its mirror folder moves with it
            token = None

        self._folders.clear()
        if token:
            try:
                changes, next_token = self.drive.changes(drive_id, str(token))
                items = [c.get("file") or {"id": c.get("fileId"), "removed": True} for c in changes]
                for change, item in zip(changes, items, strict=True):
                    if change.get("removed") or (change.get("file") or {}).get("trashed"):
                        item["removed"] = True
            except DriveGone:
                log.info("drive: the page token for %s is too old; reading it again", name)
                self._forget(drive_id)
                token = None
        if not token:
            # Order matters: take the token first, so a file changed while
            # this walk runs is reported by the next sync rather than missed.
            next_token = self.drive.start_page_token(drive_id)
            items = list(self.drive.files(drive_id))

        known = self.store.items_for(drive_id)
        now = self._clock()
        touched: set[str] = set()
        for item in items:
            file_id = str(item.get("id") or "")
            if not file_id:
                continue
            touched.add(file_id)
            if item.get("removed") or item.get("trashed"):
                self._remove(known.get(file_id), summary)
                continue
            if str(item.get("mimeType") or "") == FOLDER_MIME:
                continue
            relative = self._relative_path(folder, drive_id, item)
            if relative is None:
                summary.skipped += 1
                continue
            row = known.get(file_id)
            etag = str(
                item.get("md5Checksum") or item.get("version") or item.get("modifiedTime") or ""
            )
            target = self.documents_dir / relative
            if row is not None and row.relative_path != relative:
                self._unlink(self.documents_dir / row.relative_path)
            fresh = row is None or row.etag != etag or not target.is_file()
            if fresh:
                data = self.drive.content(file_id, str(item.get("mimeType") or ""))
                self._write(target, data)
                if row is None:
                    summary.added += 1
                else:
                    summary.updated += 1
            else:
                summary.unchanged += 1

            stale = row is None or (now - row.permissions_at) >= self.permissions_refresh_seconds
            if row is None or fresh or stale:
                principals, unmapped = principals_from(
                    self.drive.permissions(file_id), domain=self.drive.config.domain
                )
                summary.permissions_read += 1
                permissions_at = now
            else:
                principals, unmapped = row.principals, row.unmapped
                permissions_at = row.permissions_at
            self.store.upsert(
                ItemRow(
                    item_id=file_id,
                    drive_id=drive_id,
                    relative_path=relative,
                    etag=etag,
                    principals=principals,
                    unmapped=unmapped,
                    permissions_at=permissions_at,
                )
            )

        # A revoked grant changes nothing the changes feed reports, so files
        # it did not mention have their readers re-read on a clock.
        for file_id, row in known.items():
            if file_id in touched or (now - row.permissions_at) < self.permissions_refresh_seconds:
                continue
            principals, unmapped = principals_from(
                self.drive.permissions(file_id), domain=self.drive.config.domain
            )
            summary.permissions_read += 1
            self.store.upsert(
                ItemRow(
                    item_id=file_id,
                    drive_id=drive_id,
                    relative_path=row.relative_path,
                    etag=row.etag,
                    principals=principals,
                    unmapped=unmapped,
                    permissions_at=now,
                )
            )
        self.store.set_drive(drive_id, name, folder, next_token, now)

    def _relative_path(self, folder: str, drive_id: str, item: dict) -> str | None:
        """Where the file lands, or None when it is not a document to read."""
        name = str(item.get("name") or "")
        if not name:
            return None
        mime = str(item.get("mimeType") or "")
        export = EXPORTS.get(mime)
        if export is not None:
            name = f"{name}{export[1]}" if not name.endswith(export[1]) else name
        elif mime.startswith("application/vnd.google-apps."):
            return None  # a form, a drawing, a shortcut: no document in it
        if not is_supported(name):
            return None
        parts = [*self._path_of(item, drive_id), name]
        if any(part in (".", "..") for part in parts):
            return None
        return "/".join([folder, *(safe_segment(part) for part in parts)])

    def _path_of(self, item: dict, drive_id: str) -> list[str]:
        """The folder names above this file, nearest last.

        Walked through a per-sync cache: a drive of a thousand files in fifty
        folders asks about fifty folders, not a thousand times.
        """
        parts: list[str] = []
        parents = item.get("parents") or []
        current = str(parents[0]) if parents else ""
        seen: set[str] = set()
        while current and current != drive_id and current not in seen:
            seen.add(current)
            known = self._folders.get(current)
            if known is None:
                try:
                    folder = self.drive.file(current)
                except DriveError:
                    break
                ancestors = folder.get("parents") or []
                known = (str(folder.get("name") or current), str(ancestors[0]) if ancestors else "")
                self._folders[current] = known
            parts.append(known[0])
            current = known[1]
        return list(reversed(parts))

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
