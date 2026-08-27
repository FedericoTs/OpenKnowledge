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

import logging
import re
import secrets
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse

from ..cache import citations_for
from ..canonical import canonicalize_query
from ..config import Settings, load_settings
from ..contacts import ContactError, ContactStore, clean
from .engine import Engine, build_engine
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
    """Locate the widget in a source checkout or an installed layout."""
    here = Path(__file__).resolve()
    candidates = [
        here.parents[3] / "web" / "widget" / "index.html",  # source checkout
        here.parents[2] / "web" / "widget" / "index.html",
        Path("/app/web/widget/index.html"),  # container image
    ]
    return next((c for c in candidates if c.is_file()), None)


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


def _find_site() -> Path | None:
    for candidate in (
        Path(__file__).resolve().parents[3] / "web" / "site" / "index.html",
        Path("/app/web/site/index.html"),  # container image
    ):
        if candidate.is_file():
            return candidate
    return None


def _contact_store(app: FastAPI, settings: Settings) -> ContactStore:
    """One store per app, opened on first use rather than at import."""
    store = getattr(app.state, "contacts", None)
    if store is None:
        store = ContactStore(Path(settings.data_dir) / settings.contacts_db)
        app.state.contacts = store
    return store


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
        app.state.engine = build_engine(app.state.settings)
        _warn_if_the_model_is_unreachable(app.state.settings)
        try:
            yield
        finally:
            app.state.engine.store.close()
            app.state.engine.knowledge.close()

    app = FastAPI(
        title="OpenKnowledge",
        version="0.1.0",
        summary="Cheap, private, deterministic answers from your own documents.",
        lifespan=lifespan,
    )
    app.state.settings = resolved

    # -- public --------------------------------------------------------------
    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    async def widget() -> str:
        widget_path = _find_widget()
        if widget_path is None:  # pragma: no cover - packaging fallback
            return "<h1>OpenKnowledge</h1><p>Chat widget not found. POST to /chat.</p>"
        return widget_path.read_text(encoding="utf-8")

    @app.get("/healthz")
    async def healthz(engine: EngineDep) -> dict[str, Any]:
        return {
            "status": "ok",
            "documents_indexed": len(engine.retriever),
            "corpus_version": engine.retriever.corpus_version,
            "escalation_enabled": engine.settings.escalation_enabled,
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
    async def chat(req: ChatRequest, engine: EngineDep) -> ChatResponse:
        answer = await engine.cascade.answer(
            req.question,
            principals=frozenset(req.principals) if req.principals is not None else None,
            channel=req.channel,
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
