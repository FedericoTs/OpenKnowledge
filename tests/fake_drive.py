"""A loopback Google Drive, so the mirror can be tested with no Workspace.

Serves what the sync asks for: the service-account token exchange (whose JWT
assertion it actually verifies, so a wrong key is a wrong key), the shared
drive list, a start page token, the first full walk, the changes feed with
paging and removals, a file's parents for path building, its permissions,
and its bytes - downloaded for ordinary files and exported for the
Google-native ones. Tests mutate its state between syncs and read its request
log, so "changes only" is proved by the URLs asked rather than by the result
looking right.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlparse

import jwt
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

CLIENT_EMAIL = "openknowledge@project.iam.gserviceaccount.com"
SUBJECT = "librarian@contoso.com"
DOMAIN = "contoso.com"
PAGE_SIZE = 2
DOC_MIME = "application/vnd.google-apps.document"
FOLDER_MIME = "application/vnd.google-apps.folder"

_SHARED_KEY: rsa.RSAPrivateKey | None = None


def private_key_pem() -> str:
    global _SHARED_KEY  # noqa: PLW0603 - a cache, not state under test
    if _SHARED_KEY is None:
        _SHARED_KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    return _SHARED_KEY.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    ).decode()


def user_grant(email: str, role: str = "reader") -> dict[str, Any]:
    return {"id": f"p-{email}", "type": "user", "role": role, "emailAddress": email}


def group_grant(email: str) -> dict[str, Any]:
    return {"id": f"g-{email}", "type": "group", "role": "reader", "emailAddress": email}


def domain_grant(domain: str) -> dict[str, Any]:
    return {"id": f"d-{domain}", "type": "domain", "role": "reader", "domain": domain}


def anyone_grant() -> dict[str, Any]:
    return {"id": "anyone", "type": "anyone", "role": "reader"}


def deleted_group_grant(email: str) -> dict[str, Any]:
    return {
        "id": f"x-{email}",
        "type": "group",
        "role": "reader",
        "emailAddress": email,
        "deleted": True,
    }


@dataclass
class _File:
    file_id: str
    name: str
    parent: str
    mime_type: str
    content: bytes
    permissions: list[dict[str, Any]]
    version: int
    trashed: bool = False


@dataclass
class _Drive:
    drive_id: str
    name: str
    files: dict[str, _File] = field(default_factory=dict)
    changes: list[tuple[int, str]] = field(default_factory=list)


@dataclass
class FakeDrive:
    server: ThreadingHTTPServer = field(init=False)
    base: str = field(init=False)
    drives: dict[str, _Drive] = field(default_factory=dict)
    requests: list[str] = field(default_factory=list)
    version: int = 0
    token_calls: int = 0
    issued: list[str] = field(default_factory=list)
    #: Consumed once each: throttle a request, expire the page token.
    throttle_once: int | None = None
    expire_token_once: bool = False
    stale_page_token_once: bool = False

    def __post_init__(self) -> None:
        drive = self
        self._pem = private_key_pem()

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args: object) -> None:  # noqa: D102 - quiet
                pass

            def _json(self, payload: dict, status: int = 200, headers: dict | None = None) -> None:
                self._bytes(json.dumps(payload).encode(), status, "application/json", headers)

            def _bytes(
                self,
                body: bytes,
                status: int = 200,
                kind: str = "application/octet-stream",
                headers: dict | None = None,
            ) -> None:
                self.send_response(status)
                self.send_header("Content-Type", kind)
                self.send_header("Content-Length", str(len(body)))
                for k, v in (headers or {}).items():
                    self.send_header(k, v)
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self) -> None:  # noqa: N802
                drive.requests.append(f"POST {self.path}")
                length = int(self.headers.get("Content-Length", 0))
                form = {k: v[0] for k, v in parse_qs(self.rfile.read(length).decode()).items()}
                if urlparse(self.path).path != "/token":
                    self._json({"error": "not found"}, 404)
                    return
                if form.get("grant_type") != "urn:ietf:params:oauth:grant-type:jwt-bearer":
                    self._json({"error": "unsupported_grant_type"}, 400)
                    return
                try:
                    # Generous leeway on purpose: what this endpoint is for
                    # is proving the assertion was signed by the right key
                    # for the right audience. Tests move their own clock by
                    # hours to age caches, and Google's clock is not what
                    # those tests are about.
                    claims = jwt.decode(
                        form.get("assertion", ""),
                        key=drive.public_pem,
                        algorithms=["RS256"],
                        audience=f"{drive.base}/token",
                        leeway=86_400,
                    )
                except jwt.PyJWTError:
                    self._json({"error": "invalid_grant"}, 401)
                    return
                if claims.get("iss") != CLIENT_EMAIL:
                    self._json({"error": "invalid_client"}, 401)
                    return
                drive.token_calls += 1
                token = f"tok-{drive.token_calls}"
                drive.issued.append(token)
                self._json({"access_token": token, "token_type": "Bearer", "expires_in": 3600})

            def _authorised(self) -> bool:
                header = self.headers.get("Authorization", "")
                return header.startswith("Bearer ") and header[7:] in drive.issued

            def do_GET(self) -> None:  # noqa: N802
                drive.requests.append(f"GET {self.path}")
                parsed = urlparse(self.path)
                path, query = parsed.path, {k: v[0] for k, v in parse_qs(parsed.query).items()}
                if drive.expire_token_once:
                    drive.expire_token_once = False
                    drive.issued.clear()
                    self._json({"error": {"code": 401}}, 401)
                    return
                if not self._authorised():
                    self._json({"error": {"code": 401}}, 401)
                    return
                if drive.throttle_once is not None:
                    wait = drive.throttle_once
                    drive.throttle_once = None
                    self._json({"error": {"code": 429}}, 429, {"Retry-After": str(wait)})
                    return
                if path == "/v3/drives":
                    listed = [{"id": d.drive_id, "name": d.name} for d in drive.drives.values()]
                    self._json({"drives": listed})
                elif path == "/v3/changes/startPageToken":
                    self._json({"startPageToken": str(drive.version)})
                elif path == "/v3/changes":
                    self._changes(query)
                elif path == "/v3/files":
                    self._files(query)
                elif path.endswith("/permissions"):
                    found = drive.find(path.split("/v3/files/")[1].split("/")[0])
                    self._json({"permissions": found.permissions})
                elif path.endswith("/export"):
                    found = drive.find(path.split("/v3/files/")[1].split("/")[0])
                    self._bytes(found.content)
                elif path.startswith("/v3/files/"):
                    found = drive.find(path.split("/v3/files/")[1])
                    if query.get("alt") == "media":
                        self._bytes(found.content)
                    else:
                        self._json(drive.render(found))
                else:
                    self._json({"error": "not found"}, 404)

            def _files(self, query: dict[str, str]) -> None:
                d = drive.drives[query["driveId"]]
                live = [f for f in d.files.values() if not f.trashed]
                start = int(query.get("pageToken", "0"))
                page = live[start : start + PAGE_SIZE]
                body: dict[str, Any] = {"files": [drive.render(f) for f in page]}
                if start + PAGE_SIZE < len(live):
                    body["nextPageToken"] = str(start + PAGE_SIZE)
                self._json(body)

            def _changes(self, query: dict[str, str]) -> None:
                if drive.stale_page_token_once:
                    drive.stale_page_token_once = False
                    self._json({"error": {"code": 410, "message": "Invalid pageToken"}}, 410)
                    return
                d = drive.drives[query["driveId"]]
                since = int(query["pageToken"])
                seen: list[str] = []
                for version, file_id in d.changes:
                    if version > since and file_id not in seen:
                        seen.append(file_id)
                rows = []
                for file_id in seen:
                    found = d.files[file_id]
                    if found.trashed:
                        rows.append({"fileId": file_id, "removed": True})
                    else:
                        rows.append({"fileId": file_id, "file": drive.render(found)})
                self._json({"changes": rows, "newStartPageToken": str(drive.version)})

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.base = f"http://127.0.0.1:{self.server.server_port}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    @property
    def public_pem(self) -> str:
        assert _SHARED_KEY is not None
        return (
            _SHARED_KEY.public_key()
            .public_bytes(
                serialization.Encoding.PEM,
                serialization.PublicFormat.SubjectPublicKeyInfo,
            )
            .decode()
        )

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()

    # -- the state tests shape --------------------------------------------------

    def add_drive(self, drive_id: str, name: str) -> None:
        self.drives[drive_id] = _Drive(drive_id, name)

    def add_folder(self, drive_id: str, folder_id: str, name: str, parent: str) -> None:
        self.version += 1
        self.drives[drive_id].files[folder_id] = _File(
            folder_id, name, parent, FOLDER_MIME, b"", [], self.version
        )

    def add_file(
        self,
        drive_id: str,
        file_id: str,
        name: str,
        content: bytes,
        permissions: list[dict[str, Any]],
        *,
        parent: str | None = None,
        mime_type: str = "text/markdown",
    ) -> None:
        self.version += 1
        self.drives[drive_id].files[file_id] = _File(
            file_id, name, parent or drive_id, mime_type, content, permissions, self.version
        )
        self.drives[drive_id].changes.append((self.version, file_id))

    def change_content(self, drive_id: str, file_id: str, content: bytes) -> None:
        self.version += 1
        found = self.drives[drive_id].files[file_id]
        found.content = content
        found.version = self.version
        self.drives[drive_id].changes.append((self.version, file_id))

    def set_permissions(self, drive_id: str, file_id: str, permissions: list[dict]) -> None:
        # No version bump: a permission change is invisible to the changes
        # feed, which is why the sync re-reads readers on a clock.
        self.drives[drive_id].files[file_id].permissions = permissions

    def rename(self, drive_id: str, file_id: str, name: str, parent: str | None = None) -> None:
        self.version += 1
        found = self.drives[drive_id].files[file_id]
        found.name = name
        if parent:
            found.parent = parent
        found.version = self.version
        self.drives[drive_id].changes.append((self.version, file_id))

    def trash(self, drive_id: str, file_id: str) -> None:
        self.version += 1
        self.drives[drive_id].files[file_id].trashed = True
        self.drives[drive_id].changes.append((self.version, file_id))

    def find(self, file_id: str) -> _File:
        for d in self.drives.values():
            if file_id in d.files:
                return d.files[file_id]
        raise KeyError(file_id)

    def render(self, found: _File) -> dict[str, Any]:
        return {
            "id": found.file_id,
            "name": found.name,
            "mimeType": found.mime_type,
            "parents": [found.parent],
            "modifiedTime": "2026-09-02T10:00:00Z",
            "version": str(found.version),
            "trashed": found.trashed,
            "driveId": next(d.drive_id for d in self.drives.values() if found.file_id in d.files),
        }
