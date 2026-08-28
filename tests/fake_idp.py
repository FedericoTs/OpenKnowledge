"""A loopback OpenID Connect provider for testing the whole sign-in flow.

Three endpoints and a throwaway RSA key: discovery, JWKS, and a token
endpoint that signs whatever identity a test registered for a code. This is
what lets CI prove the complete flow - redirect, exchange, validation,
session - with no tenant anywhere, and it is deliberately obedient: tests
tell it to lie (wrong issuer, expired token, overage marker) to prove the
client refuses each lie.
"""

from __future__ import annotations

import base64
import json
import secrets
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs

import jwt
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

KID = "test-key"

#: One throwaway key for the whole test run: generating RSA keys is the
#: slowest thing these tests do, and nothing about them needs distinct keys.
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
class FakeIdp:
    """One provider instance per test that wants one."""

    server: ThreadingHTTPServer = field(init=False)
    issuer: str = field(init=False)
    #: code -> the claims that code's token will carry (already merged).
    codes: dict[str, dict[str, Any]] = field(default_factory=dict)
    #: The form the client posted to the token endpoint, for assertions.
    last_token_request: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._key = _test_key()
        self._pem = self._key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
        idp = self

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
                if self.path == "/.well-known/openid-configuration":
                    self._json(idp.discovery())
                elif self.path == "/jwks":
                    self._json(idp.jwks())
                else:
                    self._json({"error": "not found"}, status=404)

            def do_POST(self) -> None:  # noqa: N802 - http.server API
                length = int(self.headers.get("Content-Length", 0))
                form = {k: v[0] for k, v in parse_qs(self.rfile.read(length).decode()).items()}
                idp.last_token_request = form
                if self.path != "/token":
                    self._json({"error": "not found"}, status=404)
                    return
                claims = idp.codes.pop(form.get("code", ""), None)
                if claims is None:
                    self._json({"error": "invalid_grant"}, status=400)
                    return
                token = jwt.encode(claims, idp._pem, algorithm="RS256", headers={"kid": KID})
                self._json({"id_token": token, "token_type": "Bearer"})

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.issuer = f"http://127.0.0.1:{self.server.server_port}"
        thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        thread.start()

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()

    # -- what the endpoints serve -----------------------------------------

    def discovery(self) -> dict[str, Any]:
        return {
            "issuer": self.issuer,
            "authorization_endpoint": f"{self.issuer}/authorize",
            "token_endpoint": f"{self.issuer}/token",
            "jwks_uri": f"{self.issuer}/jwks",
        }

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

    # -- what tests register -----------------------------------------------

    def mint_code(
        self,
        *,
        audience: str,
        nonce: str,
        subject: str = "user-1",
        name: str = "Test Person",
        groups: tuple[str, ...] = (),
        claims: dict[str, Any] | None = None,
        drop: tuple[str, ...] = (),
        overage: bool = False,
    ) -> str:
        """Register a code whose token will describe this identity.

        ``claims`` overrides win over everything, ``drop`` removes claims
        entirely, and ``overage`` reproduces Entra's too-many-groups shape:
        no groups claim, just the pointer saying where they could be fetched.
        """
        now = int(time.time())
        payload: dict[str, Any] = {
            "iss": self.issuer,
            "aud": audience,
            "iat": now,
            "exp": now + 300,
            "oid": subject,
            "sub": f"pairwise-{subject}",
            "name": name,
            "nonce": nonce,
            "groups": list(groups),
        }
        if overage:
            del payload["groups"]
            payload["_claim_names"] = {"groups": "src1"}
            payload["_claim_sources"] = {"src1": {"endpoint": f"{self.issuer}/never-called"}}
        payload.update(claims or {})
        for claim in drop:
            payload.pop(claim, None)
        code = secrets.token_urlsafe(16)
        self.codes[code] = payload
        return code
