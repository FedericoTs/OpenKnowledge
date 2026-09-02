"""A loopback Bot Service, Graph and connector, so Teams can be tested with no tenant.

Five endpoints: the OpenID metadata and JWKS the inbound tokens are validated
against, a token endpoint for the bot's own outward calls, the connector's
activities endpoint (which records what the bot said), and Graph's transitive
group membership (which decides what the asker may read). It signs inbound
tokens on demand and will sign bad ones - wrong audience, wrong issuer,
expired, a serviceUrl that is not the activity's - so every refusal can be
proved rather than assumed.
"""

from __future__ import annotations

import base64
import json
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlparse

import jwt
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

KID = "bot-key"
APP_ID = "11112222-3333-4444-5555-666677778888"
APP_PASSWORD = "bot-secret"
TENANT_ID = "aaaabbbb-cccc-dddd-eeee-ffff00001111"
OTHER_TENANT = "99998888-7777-6666-5555-444433332222"

_SHARED_KEY: rsa.RSAPrivateKey | None = None


def _test_key() -> rsa.RSAPrivateKey:
    global _SHARED_KEY  # noqa: PLW0603 - a cache, not state under test
    if _SHARED_KEY is None:
        _SHARED_KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    return _SHARED_KEY


def _b64url_uint(value: int) -> str:
    raw = value.to_bytes((value.bit_length() + 7) // 8, "big")
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


@dataclass
class FakeBotFramework:
    server: ThreadingHTTPServer = field(init=False)
    base: str = field(init=False)
    #: What the bot posted back, in order: (conversation id, activity).
    replies: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    #: object id -> group ids Graph will report for them.
    memberships: dict[str, list[str]] = field(default_factory=dict)
    #: When set, Graph answers the group lookup with this status instead.
    graph_status: int | None = None
    requests: list[str] = field(default_factory=list)
    token_calls: int = 0

    def __post_init__(self) -> None:
        self._key = _test_key()
        self._pem = self._key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
        bot = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args: object) -> None:  # noqa: D102 - quiet
                pass

            def _json(self, payload: dict, status: int = 200) -> None:
                body = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self) -> None:  # noqa: N802 - http.server API
                bot.requests.append(f"GET {self.path}")
                path = urlparse(self.path).path
                if path == "/v1/.well-known/openidconfiguration":
                    self._json({"issuer": bot.issuer, "jwks_uri": f"{bot.base}/v1/keys"})
                elif path == "/v1/keys":
                    self._json(bot.jwks())
                elif "/transitiveMemberOf/microsoft.graph.group" in path:
                    if bot.graph_status is not None:
                        self._json({"error": {"code": "Forbidden"}}, bot.graph_status)
                        return
                    who = path.split("/users/")[1].split("/")[0]
                    groups = bot.memberships.get(who)
                    if groups is None:
                        self._json({"error": {"code": "ResourceNotFound"}}, 404)
                        return
                    self._json({"value": [{"id": g} for g in groups]})
                else:
                    self._json({"error": "not found"}, 404)

            def do_POST(self) -> None:  # noqa: N802
                bot.requests.append(f"POST {self.path}")
                length = int(self.headers.get("Content-Length", 0))
                raw = self.rfile.read(length)
                path = urlparse(self.path).path
                if path.endswith("/oauth2/v2.0/token"):
                    form = {k: v[0] for k, v in parse_qs(raw.decode()).items()}
                    if form.get("client_id") != APP_ID or form.get("client_secret") != APP_PASSWORD:
                        self._json({"error": "invalid_client"}, 401)
                        return
                    bot.token_calls += 1
                    self._json(
                        {
                            "access_token": f"outbound-{bot.token_calls}",
                            "token_type": "Bearer",
                            "expires_in": 3600,
                        }
                    )
                    return
                if "/v3/conversations/" in path and path.endswith("/activities"):
                    if not self.headers.get("Authorization", "").startswith("Bearer outbound-"):
                        self._json({"error": "unauthorized"}, 401)
                        return
                    conversation = path.split("/v3/conversations/")[1].split("/")[0]
                    bot.replies.append((conversation, json.loads(raw or b"{}")))
                    self._json({"id": f"reply-{len(bot.replies)}"}, 201)
                    return
                self._json({"error": "not found"}, 404)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.base = f"http://127.0.0.1:{self.server.server_port}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()

    @property
    def issuer(self) -> str:
        return f"{self.base}/botframework"

    def jwks(self) -> dict[str, Any]:
        numbers = self._key.public_key().public_numbers()
        return {
            "keys": [
                {
                    "kty": "RSA",
                    "kid": KID,
                    "use": "sig",
                    "alg": "RS256",
                    "n": _b64url_uint(numbers.n),
                    "e": _b64url_uint(numbers.e),
                }
            ]
        }

    def mint(
        self,
        *,
        service_url: str | None = None,
        audience: str = APP_ID,
        issuer: str | None = None,
        expires_in: float = 300.0,
        kid: str = KID,
    ) -> str:
        """An inbound token, correct by default and wrong on request."""
        now = time.time()
        claims: dict[str, Any] = {
            "iss": issuer if issuer is not None else self.issuer,
            "aud": audience,
            "iat": now,
            "exp": now + expires_in,
        }
        if service_url is not None:
            claims["serviceurl"] = service_url
        return jwt.encode(claims, self._pem, algorithm="RS256", headers={"kid": kid})

    def activity(
        self,
        text: str,
        *,
        user_id: str,
        tenant_id: str = TENANT_ID,
        conversation_id: str = "a:conversation-1",
        activity_type: str = "message",
        entities: list[dict] | None = None,
    ) -> dict[str, Any]:
        """An activity shaped the way Teams sends one."""
        return {
            "type": activity_type,
            "id": "activity-1",
            "timestamp": "2026-09-02T10:00:00Z",
            "serviceUrl": self.base,
            "channelId": "msteams",
            "from": {"id": "29:1abc", "name": "A Colleague", "aadObjectId": user_id},
            "conversation": {"conversationType": "channel", "id": conversation_id},
            "recipient": {"id": f"28:{APP_ID}", "name": "OpenKnowledge"},
            "text": text,
            "textFormat": "plain",
            "entities": entities or [],
            "channelData": {"tenant": {"id": tenant_id}},
        }
