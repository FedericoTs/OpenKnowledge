"""A loopback Microsoft Graph, for a SharePoint sync with no tenant anywhere.

Serves the six calls the sync makes - the token endpoint, a path-addressed
site, its drives, a drive's delta feed with paging and tokens, an item's
content (through the redirect Graph really sends) and an item's permissions -
with payloads in the shapes Microsoft documents. Tests mutate its state
between syncs (add, change, rename, delete, re-permission) and read its
request log, so "changes only" is proved by the URLs asked rather than by
the result looking right. It can also misbehave on command: throttle once,
expire a token once, expire a delta link once.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlparse

TENANT = "tenant-1"
CLIENT_ID = "app-id"
CLIENT_SECRET = "s3cret"
SITE_ID = "contoso.sharepoint.com,1111,2222"
PAGE_SIZE = 2


def user_grant(object_id: str, roles: tuple[str, ...] = ("read",)) -> dict[str, Any]:
    return {
        "id": f"p-{object_id}",
        "roles": list(roles),
        "grantedToV2": {"user": {"id": object_id}},
    }


def group_grant(object_id: str, *, inherited_from: str | None = None) -> dict[str, Any]:
    grant: dict[str, Any] = {
        "id": f"p-{object_id}",
        "roles": ["read"],
        "grantedToV2": {"group": {"id": object_id, "displayName": "a group"}},
    }
    if inherited_from:
        grant["inheritedFrom"] = {"driveId": "drive-1", "id": inherited_from}
    return grant


def link_grant(scope: str, users: tuple[str, ...] = ()) -> dict[str, Any]:
    grant: dict[str, Any] = {
        "id": f"link-{scope}",
        "roles": ["read"],
        "link": {"scope": scope, "type": "view", "webUrl": "https://contoso.sharepoint.com/:x"},
    }
    if users:
        grant["grantedToIdentitiesV2"] = [{"user": {"id": u}} for u in users]
    return grant


def site_group_grant(name: str) -> dict[str, Any]:
    return {
        "id": f"sg-{name}",
        "roles": ["read"],
        "grantedToV2": {
            "siteGroup": {"id": "5", "displayName": name, "loginName": name},
        },
    }


def application_grant(app_id: str) -> dict[str, Any]:
    return {
        "id": f"app-{app_id}",
        "roles": ["read"],
        "grantedToV2": {"application": {"id": app_id, "displayName": "OpenKnowledge"}},
    }


@dataclass
class _Item:
    item_id: str
    parent: str  # folder path under the library root, "" for the root
    name: str
    content: bytes
    permissions: list[dict[str, Any]]
    etag: str
    deleted: bool = False


@dataclass
class _Drive:
    drive_id: str
    name: str
    items: dict[str, _Item] = field(default_factory=dict)
    #: (version, item_id): what changed when, for the delta feed.
    changes: list[tuple[int, str]] = field(default_factory=list)


@dataclass
class FakeGraph:
    server: ThreadingHTTPServer = field(init=False)
    base: str = field(init=False)
    drives: dict[str, _Drive] = field(default_factory=dict)
    requests: list[str] = field(default_factory=list)
    version: int = 0
    token_calls: int = 0
    #: Misbehaviour on command, each consumed once.
    throttle_once: int | None = None
    expire_token_once: bool = False
    expire_delta_once: bool = False
    issued_tokens: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        graph = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args: object) -> None:  # noqa: D102 - quiet
                pass

            def _json(self, payload: dict, status: int = 200, headers: dict | None = None) -> None:
                body = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                for k, v in (headers or {}).items():
                    self.send_header(k, v)
                self.end_headers()
                self.wfile.write(body)

            def _authorised(self) -> bool:
                header = self.headers.get("Authorization", "")
                return header.startswith("Bearer ") and header[7:] in graph.issued_tokens

            def do_POST(self) -> None:  # noqa: N802
                graph.requests.append(f"POST {self.path}")
                length = int(self.headers.get("Content-Length", 0))
                form = {k: v[0] for k, v in parse_qs(self.rfile.read(length).decode()).items()}
                if self.path != f"/{TENANT}/oauth2/v2.0/token":
                    self._json({"error": "not found"}, 404)
                    return
                if form.get("client_id") != CLIENT_ID or form.get("client_secret") != CLIENT_SECRET:
                    self._json({"error": "invalid_client"}, 401)
                    return
                if form.get("scope") != "https://graph.microsoft.com/.default":
                    self._json({"error": "invalid_scope"}, 400)
                    return
                graph.token_calls += 1
                token = f"tok-{graph.token_calls}"
                graph.issued_tokens.append(token)
                self._json({"access_token": token, "token_type": "Bearer", "expires_in": 3600})

            def do_GET(self) -> None:  # noqa: N802
                graph.requests.append(f"GET {self.path}")
                parsed = urlparse(self.path)
                path = parsed.path
                query = {k: v[0] for k, v in parse_qs(parsed.query).items()}
                if path.startswith("/download/"):
                    _, _, drive_id, item_id = path.split("/", 3)
                    item = graph.drives[drive_id].items[item_id]
                    self.send_response(200)
                    self.send_header("Content-Type", "application/octet-stream")
                    self.send_header("Content-Length", str(len(item.content)))
                    self.end_headers()
                    self.wfile.write(item.content)
                    return
                if not path.startswith("/v1.0/"):
                    self._json({"error": "not found"}, 404)
                    return
                if graph.expire_token_once:
                    graph.expire_token_once = False
                    graph.issued_tokens.clear()
                    self._json({"error": {"code": "InvalidAuthenticationToken"}}, 401)
                    return
                if not self._authorised():
                    self._json({"error": {"code": "InvalidAuthenticationToken"}}, 401)
                    return
                if graph.throttle_once is not None:
                    wait = graph.throttle_once
                    graph.throttle_once = None
                    self._json(
                        {"error": {"code": "TooManyRequests"}}, 429, {"Retry-After": str(wait)}
                    )
                    return
                rest = path[len("/v1.0/") :]
                if rest.startswith("sites/") and ":/" in rest:
                    self._json({"id": SITE_ID, "displayName": "HR", "webUrl": "https://x"})
                elif rest == f"sites/{SITE_ID}/drives":
                    self._json(
                        {
                            "value": [
                                {"id": d.drive_id, "name": d.name, "driveType": "documentLibrary"}
                                for d in graph.drives.values()
                            ]
                        }
                    )
                elif rest.startswith("drives/") and rest.endswith("/root/delta"):
                    drive_id = rest.split("/")[1]
                    self._delta(drive_id, query)
                elif rest.startswith("drives/") and rest.endswith("/content"):
                    _, drive_id, _, item_id, _ = rest.split("/")
                    location = f"{graph.base}/download/{drive_id}/{item_id}"
                    self.send_response(302)
                    self.send_header("Location", location)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                elif rest.startswith("drives/") and rest.endswith("/permissions"):
                    _, drive_id, _, item_id, _ = rest.split("/")
                    item = graph.drives[drive_id].items[item_id]
                    self._json({"value": item.permissions})
                else:
                    self._json({"error": {"code": "itemNotFound"}}, 404)

            def _delta(self, drive_id: str, query: dict[str, str]) -> None:
                drive = graph.drives[drive_id]
                url = f"{graph.base}/v1.0/drives/{drive_id}/root/delta"
                if "token" in query:
                    if graph.expire_delta_once:
                        graph.expire_delta_once = False
                        self._json({"error": {"code": "resyncRequired"}}, 410)
                        return
                    since = int(query["token"])
                    changed = [item_id for v, item_id in drive.changes if v > since]
                    seen: list[str] = []
                    for item_id in changed:
                        if item_id not in seen:
                            seen.append(item_id)
                    rows = [graph.render(drive, drive.items[i]) for i in seen]
                    self._json({"value": rows, "@odata.deltaLink": f"{url}?token={graph.version}"})
                    return
                live = [i for i in drive.items.values() if not i.deleted]
                start = int(query.get("skiptoken", "0"))
                page = live[start : start + PAGE_SIZE]
                body: dict[str, Any] = {"value": [graph.render(drive, i) for i in page]}
                if start + PAGE_SIZE < len(live):
                    body["@odata.nextLink"] = f"{url}?skiptoken={start + PAGE_SIZE}"
                else:
                    body["@odata.deltaLink"] = f"{url}?token={graph.version}"
                self._json(body)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.base = f"http://127.0.0.1:{self.server.server_port}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()

    # -- the state tests shape ------------------------------------------------------

    def add_drive(self, drive_id: str, name: str) -> None:
        self.drives[drive_id] = _Drive(drive_id, name)

    def add_file(
        self,
        drive_id: str,
        item_id: str,
        path: str,
        content: bytes,
        permissions: list[dict[str, Any]],
    ) -> None:
        parent, _, name = path.rpartition("/")
        self.version += 1
        self.drives[drive_id].items[item_id] = _Item(
            item_id, parent, name, content, permissions, etag=f"etag-{self.version}"
        )
        self.drives[drive_id].changes.append((self.version, item_id))

    def change_content(self, drive_id: str, item_id: str, content: bytes) -> None:
        self.version += 1
        item = self.drives[drive_id].items[item_id]
        item.content = content
        item.etag = f"etag-{self.version}"
        self.drives[drive_id].changes.append((self.version, item_id))

    def set_permissions(self, drive_id: str, item_id: str, permissions: list[dict]) -> None:
        # Permissions change without the item changing: no version bump, no
        # delta entry - which is exactly why the sync re-reads them on a clock.
        self.drives[drive_id].items[item_id].permissions = permissions

    def rename(self, drive_id: str, item_id: str, path: str) -> None:
        self.version += 1
        item = self.drives[drive_id].items[item_id]
        item.parent, _, item.name = path.rpartition("/")
        self.drives[drive_id].changes.append((self.version, item_id))

    def delete(self, drive_id: str, item_id: str) -> None:
        self.version += 1
        self.drives[drive_id].items[item_id].deleted = True
        self.drives[drive_id].changes.append((self.version, item_id))

    def render(self, drive: _Drive, item: _Item) -> dict[str, Any]:
        if item.deleted:
            return {"id": item.item_id, "deleted": {"state": "deleted"}}
        return {
            "id": item.item_id,
            "name": item.name,
            "eTag": item.etag,
            "cTag": f"c{item.etag}",
            "size": len(item.content),
            "lastModifiedDateTime": "2026-09-02T10:00:00Z",
            "file": {"mimeType": "application/octet-stream"},
            "parentReference": {
                "driveId": drive.drive_id,
                "id": "folder-x",
                "path": f"/drives/{drive.drive_id}/root:"
                + (f"/{item.parent}" if item.parent else ""),
            },
            "webUrl": f"https://contoso.sharepoint.com/sites/HR/{item.parent}/{item.name}",
        }
