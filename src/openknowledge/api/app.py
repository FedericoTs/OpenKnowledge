"""FastAPI application.

Two surfaces: a chat endpoint anyone in the company can reach, and an admin
surface that is fail-closed - if no admin token is configured, the admin routes
refuse to serve rather than defaulting to open. An internal tool where anyone can
rewrite the canonical answer to "what is our refund policy" is a worse problem
than an inconvenient setup step.

Dependencies are declared at module level on purpose. This module uses
``from __future__ import annotations``, so FastAPI resolves every route
annotation as a string against module globals - an ``Annotated`` alias defined
inside ``create_app`` is invisible to it, and each dependency silently degrades
into a required query parameter.
"""

from __future__ import annotations

import contextlib
import json
import logging
import re
import secrets
import threading
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, Response, StreamingResponse
from starlette.middleware.trustedhost import TrustedHostMiddleware

from ..assets import find_asset
from ..cache import citations_for
from ..canonical import canonicalize_query
from ..config import Settings, load_settings
from ..contacts import ContactError, ContactStore, clean
from ..desktop import setup as first_run
from ..paths import state_paths
from ..providers.base import Message
from .engine import Engine, build_engine
from .runtime_settings import (
    EDITABLE,
    SettingsChangeError,
    needs_rebuild,
    to_env_value,
    validate_changes,
)
from .schemas import (
    ChatRequest,
    ChatResponse,
    ContactRequest,
    ContactResponse,
    LearnRequest,
    PinRequest,
    ReindexResponse,
    ResolveRequest,
    ReviewRequest,
)

log = logging.getLogger(__name__)


def _find_widget() -> Path | None:
    return find_asset("widget/index.html")


def get_engine(request: Request) -> Engine:
    engine: Engine | None = getattr(request.app.state, "engine", None)
    if engine is None:  # pragma: no cover - only if lifespan did not run
        raise HTTPException(503, "engine not ready")
    return engine


def require_admin(
    request: Request,
    authorization: Annotated[str | None, Header()] = None,
) -> None:
    """Fail closed: no configured token means the admin API is unavailable."""
    settings: Settings = request.app.state.settings
    expected = settings.admin_token
    if not expected:
        raise HTTPException(
            503,
            "Admin API is disabled because no admin token is set. Set OK_ADMIN_TOKEN to enable it.",
        )
    supplied = (authorization or "").removeprefix("Bearer ").strip()
    # Constant-time compare so the token cannot be recovered byte by byte.
    if not supplied or not secrets.compare_digest(supplied, expected):
        raise HTTPException(401, "invalid admin token")


EngineDep = Annotated[Engine, Depends(get_engine)]
AdminOnly = Depends(require_admin)


def _asker_principals(request: Request, supplied: list[str] | None) -> frozenset[str] | None:
    """Who is asking, in the vocabulary the ACL machinery enforces.

    With sign-in off (the default), the caller is trusted and may assert
    principals - the mode a bot backend relaying per-user identity needs.
    With sign-in on, a signed-in person's principals come from their session
    and nowhere else: a request that asserts its own is refused loudly,
    because an escalation attempt should fail, not be silently ignored. The
    admin-token caller keeps the trusted-caller mode - that token already
    grants every admin write, so it stands in for a trusted backend.
    """
    settings: Settings = request.app.state.settings
    if settings.auth_mode != "oidc":
        return frozenset(supplied) if supplied is not None else None
    session = getattr(request.state, "session", None)
    if session is not None:
        if supplied is not None:
            raise HTTPException(
                400, "principals are minted from your sign-in; a request cannot assert its own"
            )
        principals: frozenset[str] = session.principals
        return principals
    return frozenset(supplied) if supplied is not None else None


def _find_site() -> Path | None:
    return find_asset("site/index.html")


def _contact_store(app: FastAPI, settings: Settings) -> ContactStore:
    """One store per app, opened on first use rather than at import."""
    store = getattr(app.state, "contacts", None)
    if store is None:
        store = ContactStore(Path(settings.data_dir) / settings.contacts_db)
        app.state.contacts = store
    return store


#: The brand mark from the site, as a tab icon: a ticked box.
_FAVICON = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16">'
    '<rect x="1" y="1" width="14" height="14" rx="3" fill="none" '
    'stroke="currentColor" stroke-width="1.4"/>'
    '<path d="M4.6 8.2l2.2 2.2 4.6-4.8" fill="none" stroke="currentColor" '
    'stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/>'
    "</svg>"
)


def _safe_document_name(raw: str) -> str | None:
    """A filename that cannot leave the documents folder, or None.

    Browsers send bare names, but nothing forces a client to: "../../etc/x",
    an absolute path, or a Windows drive prefix are all legal multipart
    filenames. Everything is flattened to its final component and then
    whitelisted, so the only thing a hostile name can do is be refused.
    """
    final = raw.replace("\\", "/").rsplit("/", 1)[-1].strip()
    if not final or final.startswith(".") or ":" in final:
        return None
    if not re.fullmatch(r"[\w][\w \-.()+&,']{0,150}", final):
        return None
    return final


def _safe_document_path(raw: str) -> str | None:
    """A relative path inside the documents folder, or None.

    The listing names files by their folder path ('HR/handbook.pdf') because
    folders are the corpus's categories - so delete must address the same
    names, and an upload may choose one as its destination. Flattening here
    once made deleting 'HR/x.md' remove a root-level 'x.md' instead. Every
    segment is held to the same whitelist as an uploaded filename, so a
    hostile path cannot traverse; it can only be refused. Depth is capped
    because Windows still has a path-length ceiling to respect.
    """
    parts = [part for part in raw.replace("\\", "/").split("/") if part.strip()]
    if not parts or len(parts) > 8:
        return None
    safe_parts = []
    for part in parts:
        safe = _safe_document_name(part)
        if safe is None:
            return None
        safe_parts.append(safe)
    return "/".join(safe_parts)


def _upload_skip_reason(name: str) -> str | None:
    from ..documents import skip_reason

    return skip_reason(name)


def _provision_admin_token(settings: Settings) -> None:
    """In app mode - and only there - mint the admin token.

    On a server, admin-disabled-until-someone-sets-a-token is a security
    stance: a deliberate act enables the write surface. That stance holds for
    project mode AND for OK_STATE_DIR overrides - an operator pointing state
    at /srv/openknowledge is running a server, and minting there would
    silently enable every admin write behind their back. Only the per-user
    app directory means "a desktop install with nobody to set variables",
    which is the one place a dead admin API helps no one.

    The token file is owner-only before the secret touches it, and an
    existing token in the file wins over minting a new one, so two processes
    starting together converge instead of diverging.
    """
    from ..models import write_env
    from ..paths import state_paths

    state = state_paths()
    if settings.admin_token or state.mode != "app":
        return

    state.root.mkdir(parents=True, exist_ok=True)
    with contextlib.suppress(OSError):  # some filesystems refuse; not fatal
        state.root.chmod(0o700)

    # Another process may have minted between our settings load and now.
    if state.env_file.is_file():
        for line in state.env_file.read_text(encoding="utf-8").splitlines():
            if line.startswith("OK_ADMIN_TOKEN=") and line.split("=", 1)[1].strip():
                settings.admin_token = line.split("=", 1)[1].strip()
                return

    token = secrets.token_urlsafe(32)
    write_env(state.env_file, {"OK_ADMIN_TOKEN": token}, private=True)
    settings.admin_token = token
    log.info(
        "admin token generated and stored in %s (`openknowledge token` prints it)",
        state.env_file,
    )


def _provision_uploads(settings: Settings) -> None:
    """On a desktop install, drag-and-drop is how documents arrive.

    Same reasoning and same guard-rails as the token: only app mode, only when
    the operator has not spoken (an explicit OK_UPLOAD_ENABLED - true or false,
    environment or file - always wins), and the choice is recorded so it is a
    setting the person can see and flip, not behaviour that appears from
    nowhere.
    """
    from ..models import write_env
    from ..paths import state_paths

    state = state_paths()
    if state.mode != "app" or "upload_enabled" in settings.model_fields_set:
        return
    settings.upload_enabled = True
    state.root.mkdir(parents=True, exist_ok=True)
    write_env(state.env_file, {"OK_UPLOAD_ENABLED": "true"}, private=True)


def _warm_the_model_in_the_background(settings: Settings) -> None:
    """Start loading the model now; answer questions the moment it is done.

    A daemon thread rather than a task on the event loop: the load is one long
    blocking HTTP call, the loop is about to start serving, and if the process
    exits there is nothing worth waiting for.
    """
    if not (settings.local_enabled and settings.local_warmup):
        return
    from ..models import ModelError, probe, warm

    runtime = probe(settings.local_base_url, timeout=2.0)
    if not runtime.reachable:
        return  # the unreachable warning below already covers this

    def _run() -> None:
        try:
            took = warm(
                runtime,
                settings.local_model,
                keep_alive=settings.local_keep_alive,
                timeout=settings.local_timeout_seconds,
            )
            log.info("local model %s is warm (%.1fs)", settings.local_model, took)
        except ModelError as exc:
            log.warning("warmup: %s (the first question will pay the load instead)", exc)

    threading.Thread(target=_run, name="model-warmup", daemon=True).start()


def _warn_if_the_model_is_unreachable(settings: Settings) -> None:
    """Say at startup that the local endpoint is down, not one question later.

    The operator is looking at this terminal now. Finding out from a refused
    question in a chat widget - which is how this was reported - costs them a
    confusing round trip through their own documents first.

    One probe, at startup, non-fatal: a server that cannot reach its model still
    serves the audit tier, pinned answers and the cache, and refusing to boot
    over it would be worse than saying so.
    """
    if not settings.local_enabled:
        return
    from ..models import probe

    runtime = probe(settings.local_base_url, timeout=2.0)
    if runtime.reachable:
        return
    log.warning(
        "no model is answering at %s, so questions will be refused rather than answered "
        "(pins, the cache and `openknowledge audit` still work). Start it - `ollama serve` "
        "if you use Ollama - or check `openknowledge model status`.",
        settings.local_base_url,
    )


def create_app(settings: Settings | None = None) -> FastAPI:
    resolved = settings or load_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        _provision_admin_token(app.state.settings)
        _provision_uploads(app.state.settings)
        app.state.engine = build_engine(app.state.settings)
        _warn_if_the_model_is_unreachable(app.state.settings)
        _warm_the_model_in_the_background(app.state.settings)
        try:
            yield
        finally:
            app.state.engine.store.close()
            app.state.engine.knowledge.close()
            close_auth = getattr(app.state, "auth_close", None)
            if close_auth is not None:
                await close_auth()

    app = FastAPI(
        title="OpenKnowledge",
        version="0.1.0",
        summary="Cheap, private, deterministic answers from your own documents.",
        lifespan=lifespan,
    )
    app.state.settings = resolved

    # A browser will happily point an attacker's domain at 127.0.0.1 (DNS
    # rebinding) and then fetch this server from a webpage. The Host header
    # survives that trick, so when serving loopback - the personal-machine
    # case - only loopback names are accepted. A deployment that binds a
    # network interface set OK_BIND_HOST itself and keeps full control via
    # OK_TRUSTED_HOSTS.
    if resolved.bind_host in ("127.0.0.1", "localhost", "::1"):
        app.add_middleware(TrustedHostMiddleware, allowed_hosts=resolved.trusted_hosts)

    if resolved.auth_mode == "oidc":
        try:
            from ..auth.web import install_auth  # noqa: PLC0415 - needs the auth extra
        except ImportError as exc:
            raise RuntimeError(
                "OK_AUTH_MODE=oidc needs the auth extra: pip install 'openknowledge[auth]'"
            ) from exc
        install_auth(app, resolved)

    # -- public --------------------------------------------------------------
    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    async def widget() -> str:
        widget_path = _find_widget()
        if widget_path is None:  # pragma: no cover - packaging fallback
            return "<h1>OpenKnowledge</h1><p>Chat widget not found. POST to /chat.</p>"
        return widget_path.read_text(encoding="utf-8")

    @app.get("/favicon.ico", include_in_schema=False)
    async def favicon() -> Response:
        """The site's own mark, so the widget stops logging a 404 for it.

        SVG rather than an .ico: every browser that can run the widget renders
        it, and it means no binary asset in the repository. `currentColor` picks
        up the tab's own foreground, so it reads in either theme.
        """
        return Response(content=_FAVICON, media_type="image/svg+xml")

    @app.post("/chat/stream", include_in_schema=False)
    async def chat_stream(
        req: ChatRequest, engine: EngineDep, request: Request
    ) -> StreamingResponse:
        """The same resolution as /chat, narrated as server-sent events.

        Instant tiers arrive as a single `final`. A slow local answer arrives
        as `provisional` + `delta` events while it generates, then either the
        gated `final` or a `retract` - the reader watched ungated text appear,
        so the reader watches it withdrawn. The `final` payload is exactly what
        /chat would have returned, produced by the same code path, so nothing
        about caching or determinism depends on which endpoint was used.
        """
        principals = _asker_principals(request, req.principals)

        async def events() -> AsyncIterator[str]:
            history = tuple(Message(role=t.role, content=t.content) for t in req.history or ())
            stream = engine.cascade.answer_stream(
                req.question, principals=principals, channel=req.channel, history=history
            )
            async for event in stream:
                if event["type"] == "final":
                    payload: dict[str, Any] = {
                        "type": "final",
                        "response": ChatResponse.from_answer(event["answer"]).model_dump(),
                    }
                else:
                    payload = event
                yield f"data: {json.dumps(payload)}\n\n"

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            # Proxies love to buffer event streams back into one big response,
            # which would un-stream the stream.
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # ---- first run: the desktop app's setup, surfaced in the browser ------
    # On a plain server deployment nothing touches first_run.STATUS and
    # /setup/status reports "ready" forever; the widget then never redirects.

    @app.get("/setup", response_class=HTMLResponse, include_in_schema=False)
    async def setup_page() -> str:
        page = find_asset("setup/index.html")
        if page is None:  # pragma: no cover - packaging fallback
            return "<h1>OpenKnowledge</h1><p>Setup page not found.</p>"
        return page.read_text(encoding="utf-8")

    @app.get("/setup/status")
    async def setup_status() -> dict[str, Any]:
        return first_run.STATUS.snapshot()

    @app.post("/setup/download")
    async def setup_download() -> dict[str, Any]:
        """The Download or Resume button. Consent is the point: 2.6 GB is
        not a surprise to spring on someone's connection."""
        first_run.STATUS.request_proceed()
        return first_run.STATUS.snapshot()

    @app.get("/manage", response_class=HTMLResponse, include_in_schema=False)
    async def manage() -> str:
        """The management surface: documents, review, conflicts, settings.

        A static page like the widget; everything privileged it does goes
        through the admin API with the bearer token the person pastes once.
        Serving the page itself is harmless - it is the calls that are guarded.
        """
        page = find_asset("manage/index.html")
        if page is None:  # pragma: no cover - packaging fallback
            return "<h1>OpenKnowledge</h1><p>Manage page not found.</p>"
        return page.read_text(encoding="utf-8")

    @app.get("/healthz")
    async def healthz(engine: EngineDep) -> dict[str, Any]:
        return {
            "status": "ok",
            # These are not the same number and were reported as if they were:
            # `documents_indexed` carried the chunk count, so a corpus of four
            # files reported six documents.
            "documents_indexed": engine.retriever.document_count,
            "chunks_indexed": len(engine.retriever),
            "corpus_version": engine.retriever.corpus_version,
            "escalation_enabled": engine.settings.escalation_enabled,
            "upload_enabled": engine.settings.upload_enabled,
        }

    def require_uploads(engine: EngineDep) -> None:
        """Checked per request, not at route registration, so the settings
        surface can turn uploads on and off without a restart."""
        if not engine.settings.upload_enabled:
            raise HTTPException(status_code=404, detail="uploads are not enabled")

    @app.get("/documents", dependencies=[Depends(require_uploads)])
    async def list_documents(engine: EngineDep) -> dict[str, Any]:
        """What is in the documents folder, as the folder sees it.

        The corpus tier already tells any asker the indexed titles, so a
        filename listing raises nothing new - and it shows the files that
        did NOT index, with the reason, which is where "I uploaded it and
        it knows nothing" gets diagnosed.
        """
        from ..documents import skip_reason

        root = Path(engine.settings.documents_dir)
        rows = []
        folders = []
        if root.is_dir():
            for path in sorted(root.rglob("*")):
                if path.is_dir():
                    # Named even when empty: a folder an admin created is a
                    # category that exists, not an artifact of its contents.
                    folders.append(path.relative_to(root).as_posix())
                    continue
                if not path.is_file():
                    continue
                rows.append(
                    {
                        "name": path.relative_to(root).as_posix(),
                        "size": path.stat().st_size,
                        "skipped": skip_reason(path),
                    }
                )
        return {"documents_dir": str(root), "folders": folders, "files": rows}

    @app.post("/documents", status_code=201, dependencies=[Depends(require_uploads)])
    async def upload_documents(
        engine: EngineDep,
        files: Annotated[list[UploadFile], File()],
        folder: Annotated[str, Form()] = "",
    ) -> dict[str, Any]:
        """Accept documents and make them knowledge in the same breath.

        The whole batch is written first and the corpus re-indexed once -
        re-indexing is free by construction (no model call), so a drop of
        forty files costs one scan, not forty. The response says what a
        person needs to trust it: what was stored, what was refused and
        why, and what the corpus looks like now.

        ``folder`` files the whole batch under a category. It is a separate
        field rather than part of the filenames because multipart filenames
        stay hostile-by-default and flattened; the folder is an explicit
        request, validated segment by segment.
        """
        into = ""
        if folder.strip():
            safe_folder = _safe_document_path(folder)
            if safe_folder is None:
                raise HTTPException(status_code=400, detail="unusable folder name")
            into = safe_folder
        limit = engine.settings.upload_max_mb * 1_000_000
        root = Path(engine.settings.documents_dir)
        target_dir = root / into if into else root
        target_dir.mkdir(parents=True, exist_ok=True)

        stored: list[dict[str, Any]] = []
        skipped: list[dict[str, str]] = []
        for upload in files:
            name = _safe_document_name(upload.filename or "")
            if name is None:
                skipped.append(
                    {"name": upload.filename or "(unnamed)", "reason": "unusable file name"}
                )
                continue
            reason = _upload_skip_reason(name)
            if reason is not None:
                skipped.append({"name": name, "reason": reason})
                continue
            data = await upload.read(limit + 1)
            if len(data) > limit:
                skipped.append(
                    {
                        "name": name,
                        "reason": f"larger than the {engine.settings.upload_max_mb} MB "
                        "limit; a corpus document this size is usually a scan, which "
                        "cannot be read anyway",
                    }
                )
                continue
            if not data:
                skipped.append({"name": name, "reason": "the file is empty"})
                continue
            target = target_dir / name
            replaced = target.exists()
            target.write_bytes(data)
            stored.append(
                {
                    "name": f"{into}/{name}" if into else name,
                    "bytes": len(data),
                    "replaced": replaced,
                }
            )

        corpus: dict[str, Any] = {}
        if stored:
            documents, chunks, version, _ = engine.reindex()
            corpus = {
                "documents": documents,
                "chunks": chunks,
                "corpus_version": version,
                "conflicts_open": (engine.last_scan.conflicts_open if engine.last_scan else 0),
            }
        return {"stored": stored, "skipped": skipped, "corpus": corpus}

    @app.delete("/documents/{name:path}", dependencies=[Depends(require_uploads)])
    async def delete_document(name: str, engine: EngineDep) -> dict[str, Any]:
        safe = _safe_document_path(name)
        if safe is None:
            raise HTTPException(status_code=400, detail="unusable file name")
        target = Path(engine.settings.documents_dir) / safe
        if not target.is_file():
            raise HTTPException(status_code=404, detail=f"no document called {safe!r}")
        target.unlink()
        documents, chunks, version, evicted = engine.reindex()
        return {
            "deleted": safe,
            "corpus": {
                "documents": documents,
                "chunks": chunks,
                "corpus_version": version,
                "answers_evicted": evicted,
            },
        }

    if resolved.website_enabled:

        @app.get("/site", response_class=HTMLResponse, include_in_schema=False)
        async def site() -> str:
            page = _find_site()
            if page is None:  # pragma: no cover - packaging fallback
                return "<h1>OpenKnowledge</h1><p>Site page not found.</p>"
            return page.read_text(encoding="utf-8")

        # Both paths, because the page is served at /site with no trailing
        # slash: a browser resolves its relative `fonts/x.woff2` against the
        # parent and asks for /fonts/x.woff2. The /site/fonts/ form is what a
        # static host serving the same folder would use. Registering only one
        # of them is how these 404'd silently the first time - the page still
        # rendered, in fallback faces, and looked fine.
        @app.get("/fonts/{name}", include_in_schema=False)
        @app.get("/site/fonts/{name}", include_in_schema=False)
        async def site_font(name: str) -> FileResponse:
            """The page's three typefaces, served from here rather than a CDN.

            The page claims it makes no third-party requests, and that claim is
            only true if the fonts come from the same origin.

            Deliberately narrow: one extension, no separators, no dots, so the
            name cannot walk out of the folder whatever is asked for.
            """
            page = _find_site()
            stem, _, suffix = name.rpartition(".")
            if page is None or suffix != "woff2" or not re.fullmatch(r"[a-z0-9-]+", stem):
                raise HTTPException(status_code=404, detail="not found")
            font = page.parent / "fonts" / name
            if not font.is_file():
                raise HTTPException(status_code=404, detail="not found")
            return FileResponse(
                font,
                media_type="font/woff2",
                headers={"Cache-Control": "public, max-age=31536000, immutable"},
            )

        @app.post("/api/contact", response_model=ContactResponse, status_code=201)
        async def contact(req: ContactRequest, request: Request) -> ContactResponse:
            store = _contact_store(app, resolved)

            # A bot that filled the hidden field gets a cheerful 201 and is
            # dropped. Telling it what failed is how it learns to pass.
            if req.website:
                log.info("contact submission dropped: honeypot filled")
                return ContactResponse(received=True)

            if store.submissions_since(time.time() - 3600) >= resolved.contact_max_per_hour:
                raise HTTPException(
                    status_code=429,
                    detail="too many submissions in the last hour; please try later",
                )

            try:
                fields = clean(req.model_dump())
            except ContactError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc

            store.add(fields, source=request.headers.get("referer", "website")[:200])
            return ContactResponse(received=True)

    @app.post("/chat", response_model=ChatResponse)
    async def chat(req: ChatRequest, engine: EngineDep, request: Request) -> ChatResponse:
        history = tuple(Message(role=t.role, content=t.content) for t in req.history or ())
        answer = await engine.cascade.answer(
            req.question,
            principals=_asker_principals(request, req.principals),
            channel=req.channel,
            history=history,
        )
        return ChatResponse.from_answer(answer)

    # -- admin ---------------------------------------------------------------
    @app.get("/admin/costs", dependencies=[AdminOnly])
    async def costs(engine: EngineDep) -> dict[str, Any]:
        """What the bot actually costs, measured rather than estimated."""
        return engine.store.cost_report()

    @app.get("/admin/questions", dependencies=[AdminOnly])
    async def questions(engine: EngineDep, limit: int = 20) -> dict[str, Any]:
        """Most-asked questions: the shortlist worth pinning."""
        return {
            "top": [{"question": q, "count": n} for q, n in engine.store.top_questions(limit)],
            "recent": [
                {
                    "question": e.canonical_query,
                    "tier": e.tier.value,
                    "cost_usd": round(e.cost_usd, 6),
                    "channel": e.channel,
                }
                for e in engine.store.recent_questions(limit)
            ],
        }

    @app.get("/admin/pins", dependencies=[AdminOnly])
    async def list_pins(engine: EngineDep) -> list[dict[str, Any]]:
        return [
            {
                "question": p.canonical_query,
                "answer": p.answer,
                "author": p.author,
                "cited": [c.document_id for c in p.citations],
            }
            for p in engine.store.list_pins()
        ]

    @app.post("/admin/pins", dependencies=[AdminOnly], status_code=201)
    async def create_pin(req: PinRequest, engine: EngineDep) -> dict[str, Any]:
        phrasings = [req.question, *req.aliases]
        canonicals = [c for c in (canonicalize_query(p) for p in phrasings) if c]
        if not canonicals:
            raise HTTPException(422, "question is empty after normalisation")

        citations = citations_for(engine.retriever, tuple(req.cite))
        for canonical in canonicals:
            engine.store.pin(canonical, req.answer, citations=citations, author=req.author)
        return {
            "question": canonicals[0],
            "aliases": canonicals[1:],
            "cited": [c.document_id for c in citations],
            "pinned": True,
        }

    @app.delete("/admin/pins", dependencies=[AdminOnly])
    async def delete_pin(question: str, engine: EngineDep) -> dict[str, Any]:
        canonical = canonicalize_query(question)
        return {"question": canonical, "removed": engine.store.unpin(canonical)}

    @app.post("/admin/reindex", dependencies=[AdminOnly], response_model=ReindexResponse)
    async def reindex(engine: EngineDep) -> ReindexResponse:
        documents, chunks, version, evicted = engine.reindex()
        return ReindexResponse(
            documents=documents,
            chunks=chunks,
            corpus_version=version,
            evicted_cache_entries=evicted,
        )

    # -- knowledge lifecycle -------------------------------------------------
    @app.post("/admin/learn", dependencies=[AdminOnly])
    async def learn(req: LearnRequest, engine: EngineDep) -> dict[str, Any]:
        """Draft answers for changed documents. This one spends tokens."""
        report = await engine.learn(max_documents=req.max_documents)
        return {
            "summary": report.summary(),
            "added": list(report.added),
            "changed": list(report.changed),
            "removed": list(report.removed),
            "drafts_created": report.drafts_created,
            "drafts_rejected": report.drafts_rejected,
            "drafts_superseded": report.drafts_superseded,
            "revisions_raised": report.revisions_raised,
            "conflicts_open": report.conflicts_open,
            "cost_usd": round(report.cost_usd, 6),
            "needs_review": report.needs_review,
            "notes": report.notes,
        }

    @app.get("/admin/proposals", dependencies=[AdminOnly])
    async def proposals(engine: EngineDep, limit: int = 50) -> dict[str, Any]:
        """Drafted answers awaiting review, ranked by what approving them saves."""
        from ..knowledge import rank_by_demand

        pending = engine.knowledge.pending(limit=limit)
        demand = dict(engine.store.top_questions(limit=500))
        ranked = rank_by_demand(pending, demand=demand, cost_per_answer_usd=0.0094)
        return {
            "counts": engine.knowledge.counts(),
            "pending": [
                {
                    "id": p.id,
                    "question": p.question,
                    "answer": p.answer,
                    "cited": [c.document_id for c in p.citations],
                    "support_ratio": round(p.support_ratio, 4),
                    "source": p.source,
                    "times_asked": demand.get(p.canonical_query, 0),
                    "estimated_value_usd": round(value, 6),
                    "supersedes": p.supersedes,
                }
                for p, value in ranked
            ],
        }

    @app.post("/admin/proposals/{proposal_id}/approve", dependencies=[AdminOnly])
    async def approve(proposal_id: str, req: ReviewRequest, engine: EngineDep) -> dict[str, Any]:
        if not engine.approve(proposal_id, reviewer=req.reviewer):
            raise HTTPException(404, "no draft awaiting review with that id")
        return {"id": proposal_id, "approved": True, "pinned": True}

    @app.post("/admin/proposals/{proposal_id}/reject", dependencies=[AdminOnly])
    async def reject(proposal_id: str, req: ReviewRequest, engine: EngineDep) -> dict[str, Any]:
        rejected = engine.knowledge.reject(proposal_id, reviewer=req.reviewer, note=req.note)
        if rejected is None:
            raise HTTPException(404, "no draft awaiting review with that id")
        return {"id": proposal_id, "rejected": True}

    @app.get("/admin/conflicts", dependencies=[AdminOnly])
    async def conflicts(engine: EngineDep) -> list[dict[str, Any]]:
        """Documents that disagree. Questions touching these are refused."""
        return [
            {
                "key": c.key,
                "unit": c.unit,
                "overlap": round(c.overlap, 4),
                "left": {
                    "document": c.left_document,
                    "value": c.left_raw,
                    "sentence": c.left_sentence,
                },
                "right": {
                    "document": c.right_document,
                    "value": c.right_raw,
                    "sentence": c.right_sentence,
                },
            }
            for c in engine.knowledge.open_conflicts()
        ]

    @app.post("/admin/conflicts/{key}/resolve", dependencies=[AdminOnly])
    async def resolve(key: str, req: ResolveRequest, engine: EngineDep) -> dict[str, Any]:
        resolution = req.note or (f"authoritative: {req.keep}" if req.keep else "resolved")
        resolved = engine.knowledge.resolve_conflict(
            key, resolution=resolution, resolver=req.reviewer
        )
        if resolved is None:
            raise HTTPException(404, "no open conflict with that key")
        return {
            "key": key,
            "resolved": True,
            "resolution": resolution,
            "note": (
                "Recorded. This does not edit your documents - remove or correct the "
                "superseded text so retrieval stops seeing it."
            ),
        }

    @app.get("/admin/settings", dependencies=[AdminOnly])
    async def get_settings(engine: EngineDep) -> dict[str, Any]:
        """The editable settings, their current values, and how each applies."""
        s = engine.settings
        return {
            "settings": {
                key: {"value": getattr(s, key), "applies": how} for key, how in EDITABLE.items()
            },
            "persists_to": str(state_paths().env_file),
        }

    @app.put("/admin/settings", dependencies=[AdminOnly])
    async def put_settings(changes: dict[str, Any], request: Request) -> dict[str, Any]:
        """Apply and persist a set of changes, saying exactly what happened.

        Live fields take effect on the next request because the running
        objects read the settings instance rather than copies of it. Rebuild
        fields swap in a freshly built engine inside this request - the old
        one is closed only after the new one exists, so a failed rebuild
        leaves the server answering exactly as before. Everything applied is
        also written to the state dotenv, so a restart agrees with the page.
        """
        from ..models import write_env

        try:
            validated = validate_changes(changes)
        except SettingsChangeError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        settings: Settings = request.app.state.settings
        previous = {key: getattr(settings, key) for key in validated}
        for key, value in validated.items():
            setattr(settings, key, value)

        rebuilt = False
        if needs_rebuild(validated):
            try:
                fresh = build_engine(settings)
            except Exception as exc:
                # The new configuration could not even be built; put the old
                # values back and say so rather than serving a half-state.
                for key, value in previous.items():
                    setattr(settings, key, value)
                raise HTTPException(
                    status_code=422, detail=f"these settings do not build: {exc}"
                ) from exc
            old_engine: Engine = request.app.state.engine
            request.app.state.engine = fresh
            old_engine.store.close()
            old_engine.knowledge.close()
            _warm_the_model_in_the_background(settings)
            rebuilt = True

        state = state_paths()
        state.root.mkdir(parents=True, exist_ok=True)
        written = write_env(
            state.env_file,
            {f"OK_{key.upper()}": to_env_value(value) for key, value in validated.items()},
        )
        return {
            "applied": validated,
            "engine_rebuilt": rebuilt,
            "persisted": written,
            "persists_to": str(state.env_file),
        }

    @app.get("/admin/config", dependencies=[AdminOnly])
    async def config(engine: EngineDep) -> dict[str, Any]:
        """Effective settings, with secrets redacted."""
        s = engine.settings
        return {
            "documents_dir": s.documents_dir,
            "retrieval_k": s.retrieval_k,
            "min_support_ratio": s.min_support_ratio,
            "require_citations": s.require_citations,
            "local": {
                "enabled": s.local_enabled,
                "model": s.local_model,
                "base_url": s.local_base_url,
            },
            "escalation": {
                "enabled": s.escalation_enabled,
                "provider": s.escalation_provider,
                "model": s.escalation_model,
                "effort": s.escalation_effort,
                "api_key_configured": bool(s.anthropic_api_key or s.openai_api_key),
            },
            "system_prompt_suffix": s.system_prompt_suffix,
        }

    return app


app = create_app()
