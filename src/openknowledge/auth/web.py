"""The sign-in gate in front of the app.

``install_auth`` wires three routes and one middleware onto the FastAPI app
when ``OK_AUTH_MODE=oidc``. With sign-in off none of this is imported, let
alone registered - the desktop app and personal servers keep exactly today's
behaviour, including the trusted-caller mode where a request may assert its
own principals.

Who gets through the gate:

- ``/auth/*``, ``/healthz`` and the favicon - the door, the monitoring
  probe, and the icon on the door.
- A browser with a live session cookie. The session rides
  ``request.state.session`` so endpoints can mint principals from it.
- A caller presenting the admin token. That token already grants every
  admin write, so it also stands in for a trusted backend (a bot relay
  asserting per-user principals is its legitimate job).

Everyone else: HTML requests are redirected to sign-in, API requests get 401.
"""

from __future__ import annotations

import secrets
from collections.abc import Awaitable, Callable

from fastapi import FastAPI, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

from ..config import Settings
from .oidc import OidcClient, OidcError
from .sessions import SessionStore

#: The cookie carrying the opaque session token.
COOKIE_NAME = "ok_session"

_OPEN_PATHS = frozenset({"/healthz", "/favicon.ico"})
_OPEN_PREFIX = "/auth/"


def install_auth(app: FastAPI, settings: Settings) -> None:
    """Register the sign-in routes and the gate. Fails loud on bad config."""
    missing = [
        name
        for name, value in (
            ("OK_OIDC_ISSUER", settings.oidc_issuer),
            ("OK_OIDC_CLIENT_ID", settings.oidc_client_id),
        )
        if not value
    ]
    if missing:
        raise RuntimeError(f"OK_AUTH_MODE=oidc needs {' and '.join(missing)} set")

    oidc = OidcClient(
        issuer=settings.oidc_issuer,
        client_id=settings.oidc_client_id,
        client_secret=settings.oidc_client_secret,
        groups_claim=settings.oidc_groups_claim,
    )
    store = SessionStore(settings.auth_db_path)
    app.state.oidc = oidc
    app.state.auth_sessions = store

    async def _close() -> None:
        await oidc.aclose()
        store.close()

    app.state.auth_close = _close

    def _base_url(request: Request) -> str:
        return (settings.public_url or str(request.base_url)).rstrip("/")

    def _redirect_uri(request: Request) -> str:
        return f"{_base_url(request)}/auth/callback"

    @app.get("/auth/login", include_in_schema=False)
    async def login(request: Request) -> Response:
        try:
            url, pending = await oidc.begin_login(_redirect_uri(request))
        except OidcError as exc:
            return _error_page(str(exc), status_code=502)
        store.save_pending(pending)
        return RedirectResponse(url, status_code=302)

    @app.get("/auth/callback", include_in_schema=False)
    async def callback(request: Request) -> Response:
        params = request.query_params
        if params.get("error"):
            reason = params.get("error_description") or params["error"]
            return _error_page(f"the identity provider said no: {reason}", status_code=400)
        code, state = params.get("code"), params.get("state")
        if not code or not state:
            return _error_page("the callback carried no code or state", status_code=400)
        pending = store.take_pending(state)
        if pending is None:
            return _error_page(
                "this sign-in attempt is unknown, already used, or took too long",
                status_code=400,
            )
        try:
            identity = await oidc.complete_login(code, pending, _redirect_uri(request))
        except OidcError as exc:
            return _error_page(str(exc), status_code=400)
        token = store.create(identity, ttl_seconds=settings.session_hours * 3600)
        response = RedirectResponse("/", status_code=302)
        response.set_cookie(
            COOKIE_NAME,
            token,
            max_age=int(settings.session_hours * 3600),
            httponly=True,
            samesite="lax",
            secure=_base_url(request).startswith("https"),
            path="/",
        )
        return response

    @app.post("/auth/logout", include_in_schema=False)
    async def logout(request: Request) -> Response:
        token = request.cookies.get(COOKIE_NAME)
        if token:
            store.delete(token)
        response = RedirectResponse("/", status_code=302)
        response.delete_cookie(COOKIE_NAME, path="/")
        return response

    @app.middleware("http")
    async def gate(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        path = request.url.path
        if path in _OPEN_PATHS or path.startswith(_OPEN_PREFIX):
            return await call_next(request)
        token = request.cookies.get(COOKIE_NAME)
        session = store.get(token) if token else None
        if session is not None:
            request.state.session = session
            return await call_next(request)
        if _admin_token_matches(settings, request.headers.get("authorization")):
            # A trusted caller, not a person: no session, and endpoints that
            # mint principals treat it as the pre-sign-in trusted mode.
            request.state.session = None
            return await call_next(request)
        if "text/html" in request.headers.get("accept", ""):
            return RedirectResponse("/auth/login", status_code=302)
        return JSONResponse({"detail": "sign in required"}, status_code=401)


def _admin_token_matches(settings: Settings, authorization: str | None) -> bool:
    expected = settings.admin_token
    if not expected or not authorization:
        return False
    supplied = authorization.removeprefix("Bearer ").strip()
    return bool(supplied) and secrets.compare_digest(supplied, expected)


def _error_page(reason: str, *, status_code: int) -> HTMLResponse:
    """A plain page that says what failed and where the door is."""
    return HTMLResponse(
        "<!doctype html><meta charset='utf-8'><title>Sign-in failed</title>"
        "<body style='font:16px/1.6 system-ui;max-width:36em;margin:15vh auto;padding:0 1em'>"
        f"<h1 style='font-size:1.2em'>Sign-in failed</h1><p>{_escaped(reason)}</p>"
        "<p><a href='/auth/login'>Try again</a></p>",
        status_code=status_code,
    )


def _escaped(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
