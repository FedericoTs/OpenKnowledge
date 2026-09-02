"""Whether the model endpoints answer, asked of them rather than assumed.

A question that arrives while the local model is down is refused with a
sentence about the model; the person who can do something about that reads
/manage, not the chat. This asks each configured endpoint the cheapest thing
it will answer - Ollama its version, an OpenAI-compatible server its model
list, a paid API its model list - times the answer, and reports what came
back: reachable or not, which runtime, whether the configured model is among
the ones served, whether the key was accepted.

Nothing here is on the path of a question. Readings are cached for a short
while and taken with a short timeout, so a dead endpoint costs the page a
few seconds once, not every reader every time. ``/healthz`` stays a liveness
check and carries none of this: a monitor that restarts the app because a
*dependency* is down turns an outage into a longer one.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from typing import Any
from urllib.parse import urlparse

import httpx

from . import models
from .config import Settings

#: How long a reading stands before the endpoint is asked again.
TTL_SECONDS = 30.0
#: How long one endpoint gets to answer the cheapest call it has.
TIMEOUT_SECONDS = 3.0
ANTHROPIC_API = "https://api.anthropic.com"


@dataclass(frozen=True)
class Target:
    """One endpoint to ask, and how to ask it.

    ``kind`` is the dialect: ``runtime`` for a self-hosted model server
    (probed the way ``openknowledge model`` does), ``openai`` / ``azure`` /
    ``anthropic`` for a paid API, ``off`` for a role that is not configured -
    which is a reading in itself, and ``detail`` says why.
    """

    role: str
    kind: str
    base_url: str = ""
    model: str = ""
    api_key: str | None = None
    api_version: str = ""
    detail: str = ""


@dataclass(frozen=True)
class Reading:
    """What one endpoint said when asked. Never carries the key it was asked with."""

    role: str
    configured: bool
    #: reachable | unreachable | key refused | error | off
    state: str
    #: Whether a question would get through: reachable, key accepted, and the
    #: configured model served (or not knowable). None when the role is off.
    ok: bool | None
    host: str = ""
    model: str = ""
    model_served: bool | None = None
    runtime: str = ""
    version: str = ""
    latency_ms: int | None = None
    checked_at: float = 0.0
    detail: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def targets(settings: Settings, frontier: object | None) -> list[Target]:
    """The endpoints this install depends on, from the settings in force.

    ``frontier`` is the built escalation provider, or None: settings can
    switch escalation on while the key it needs is missing, and the built
    provider is the truth about whether anything would actually be called.
    """
    found: list[Target] = []
    if settings.local_enabled:
        found.append(Target("chat", "runtime", settings.local_base_url, settings.local_model))
    else:
        found.append(Target("chat", "off", detail="the local model is turned off"))

    if settings.embedding_enabled:
        base = settings.embedding_base_url or settings.local_base_url
        found.append(Target("embedding", "runtime", base, settings.embedding_model))
    else:
        found.append(
            Target("embedding", "off", detail="embeddings are off; retrieval is BM25 only")
        )

    found.append(_escalation_target(settings, frontier))
    return found


def _escalation_target(settings: Settings, frontier: object | None) -> Target:
    from .providers.anthropic_provider import AnthropicProvider
    from .providers.azure_openai import AzureOpenAIProvider
    from .providers.openai_compat import OpenAICompatProvider

    if not settings.escalation_enabled:
        return Target("escalation", "off", detail="escalation is off; nothing leaves the machine")
    if frontier is None:
        return Target(
            "escalation",
            "off",
            detail=(
                "escalation is on but nothing is configured to call: a key or endpoint is missing"
            ),
        )
    if isinstance(frontier, AzureOpenAIProvider):
        return Target(
            "escalation",
            "azure",
            frontier.base_url,
            frontier.model_id,
            frontier._api_key,
            frontier.api_version,
        )
    if isinstance(frontier, OpenAICompatProvider):
        return Target(
            "escalation", "openai", frontier.base_url, frontier.model_id, frontier._api_key
        )
    if isinstance(frontier, AnthropicProvider):
        return Target(
            "escalation", "anthropic", ANTHROPIC_API, frontier.model_id, frontier._api_key
        )
    return Target(
        "escalation",
        "off",
        detail=f"escalation goes through {type(frontier).__name__}, which this cannot probe",
    )


def _host(base_url: str) -> str:
    return urlparse(base_url).netloc


def _serves(names: Sequence[str], model: str) -> bool | None:
    """Whether ``model`` is among ``names``; None when the list could not be read."""
    if not names:
        return None
    wanted = {model, models.tagged(model), model.removesuffix(":latest")}
    return any(name in wanted for name in names)


def probe(target: Target, *, timeout: float = TIMEOUT_SECONDS) -> Reading:
    """Ask one endpoint, synchronously. Runs in a worker thread under the monitor."""
    if target.kind == "off":
        return Reading(
            role=target.role,
            configured=False,
            state="off",
            ok=None,
            model=target.model,
            checked_at=time.time(),
            detail=target.detail,
        )
    if target.kind == "runtime":
        return _probe_runtime(target, timeout)
    return _probe_api(target, timeout)


def _probe_runtime(target: Target, timeout: float) -> Reading:
    started = time.perf_counter()
    runtime = models.probe(target.base_url, timeout=timeout)
    if not runtime.reachable:
        return Reading(
            role=target.role,
            configured=True,
            state="unreachable",
            ok=False,
            host=_host(target.base_url),
            model=target.model,
            latency_ms=_ms(started),
            checked_at=time.time(),
            detail=f"nothing answered at {_host(target.base_url)} within {timeout:.0f}s",
        )
    names = [m.name for m in models.installed(runtime, timeout=timeout)]
    served = _serves(names, target.model)
    detail = ""
    if served is False:
        detail = f"the server answers but does not serve {target.model}"
    elif served is None:
        detail = "the server answers; it did not list its models"
    return Reading(
        role=target.role,
        configured=True,
        state="reachable",
        ok=served is not False,
        host=_host(target.base_url),
        model=target.model,
        model_served=served,
        runtime=runtime.kind,
        version=runtime.version,
        latency_ms=_ms(started),
        checked_at=time.time(),
        detail=detail,
    )


def _models_request(target: Target) -> tuple[str, dict[str, str]]:
    """The cheapest authenticated call each paid dialect answers: its model list."""
    key = target.api_key or ""
    if target.kind == "anthropic":
        return (
            f"{target.base_url.rstrip('/')}/v1/models",
            {"x-api-key": key, "anthropic-version": "2023-06-01"},
        )
    if target.kind == "azure":
        if target.api_version == "v1":
            return f"{target.base_url}/models", {"api-key": key}
        # The legacy surface names the deployment in the URL; the model list
        # lives beside it, one level up, and takes the same dated version.
        resource = target.base_url.split("/openai/", 1)[0]
        return f"{resource}/openai/models?api-version={target.api_version}", {"api-key": key}
    return f"{target.base_url}/models", {"Authorization": f"Bearer {key}"}


def _probe_api(target: Target, timeout: float) -> Reading:
    url, headers = _models_request(target)
    started = time.perf_counter()
    common: dict[str, Any] = {
        "role": target.role,
        "configured": True,
        "host": _host(target.base_url),
        "model": target.model,
    }
    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.get(url, headers=headers)
    except httpx.HTTPError as exc:
        return Reading(
            **common,
            state="unreachable",
            ok=False,
            latency_ms=_ms(started),
            checked_at=time.time(),
            detail=f"{type(exc).__name__}: {exc}".rstrip(": "),
        )
    latency = _ms(started)
    if response.status_code in (401, 403):
        return Reading(
            **common,
            state="key refused",
            ok=False,
            latency_ms=latency,
            checked_at=time.time(),
            detail=f"the endpoint answered HTTP {response.status_code} to the configured key",
        )
    if response.status_code != 200:
        return Reading(
            **common,
            state="error",
            ok=False,
            latency_ms=latency,
            checked_at=time.time(),
            detail=f"HTTP {response.status_code} from the model list",
        )
    served = _serves(_listed_ids(response, target.kind), target.model)
    return Reading(
        **common,
        state="reachable",
        ok=served is not False,
        model_served=served,
        latency_ms=latency,
        checked_at=time.time(),
        detail=f"{target.model} is not in the model list" if served is False else "",
    )


def _listed_ids(response: httpx.Response, kind: str) -> list[str]:
    """Model ids from a list response; empty when the shape is not the one expected.

    Azure's list names models, not deployments, so a deployment can never be
    found in it - report nothing rather than a false "not served".
    """
    if kind == "azure":
        return []
    try:
        rows = response.json().get("data") or []
        return [str(row["id"]) for row in rows if isinstance(row, dict) and row.get("id")]
    except (ValueError, AttributeError, KeyError, TypeError):
        return []


def _ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)


class HealthMonitor:
    """Readings, cached per target for ``ttl`` seconds; fresh on request.

    Keyed by the target rather than the role, so a changed setting is asked
    about at once rather than after the old reading expires.
    """

    def __init__(
        self,
        *,
        ttl: float = TTL_SECONDS,
        timeout: float = TIMEOUT_SECONDS,
        clock: Callable[[], float] = time.monotonic,
        prober: Callable[..., Reading] = probe,
    ) -> None:
        self.ttl = ttl
        self._timeout = timeout
        self._clock = clock
        self._prober = prober
        self._cache: dict[Target, tuple[float, Reading]] = {}

    async def readings(self, wanted: Sequence[Target], *, fresh: bool = False) -> list[Reading]:
        now = self._clock()
        due = [
            t
            for t in wanted
            if fresh or t not in self._cache or now - self._cache[t][0] >= self.ttl
        ]
        if due:
            # Each probe blocks on its own socket; three of them in threads
            # finish in the time of the slowest, and the event loop keeps
            # answering questions meanwhile.
            results = await asyncio.gather(
                *(asyncio.to_thread(self._prober, t, timeout=self._timeout) for t in due)
            )
            taken = self._clock()
            for target, reading in zip(due, results, strict=True):
                self._cache[target] = (taken, reading)
        return [self._cache[t][1] for t in wanted]
