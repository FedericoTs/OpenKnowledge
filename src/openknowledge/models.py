"""Choosing, installing and resizing the local model.

The cheapest rung of the cascade is a model on a machine you own, and the one
thing that decides whether a given model can do the job is its context window.
Runtimes disagree about what an over-long prompt means, and the
OpenAI-compatible API does not report which kind you are talking to - so
switching models has to carry the window with it.

Two things follow from that, and they are the reason this module exists rather
than a line in the README:

* **Nothing here is claimed, it is asked.** There is no table of models with
  their sizes and windows baked in - those go stale, and a stale number here is
  worse than none. Every figure printed comes from the runtime that is actually
  running: `/api/tags` for what is installed, `/api/show` for the window the
  weights declare.
* **The window is recorded, not assumed.** ``model use`` writes
  ``OK_LOCAL_CONTEXT_TOKENS`` alongside the model name, and the provider checks
  the fit before sending - see
  :meth:`~openknowledge.providers.openai_compat.OpenAICompatProvider._check_fit`
  for what that is worth and what it is not.

Ollama's OpenAI-compatible endpoint takes no context-length parameter, so a
larger window cannot be requested per call. The documented route is a derived
model carrying ``num_ctx``, which is what ``--context`` builds.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

import httpx

#: Names worth trying first, with no numbers attached: the sizes and windows
#: are whatever the registry is serving today, and `model list` reads those
#: from the runtime instead of repeating a guess.
SUGGESTED = (
    ("qwen3:4b", "smallest that still grounds well; fine on a laptop CPU"),
    ("qwen3:8b", "the default - the balance point for a 16 GB machine"),
    ("qwen3:14b", "noticeably better on long policies, wants a GPU"),
    ("qwen3:30b", "mixture-of-experts: 30B stored, a fraction of it active"),
    ("gpt-oss:20b", "strong refusals, useful when abstaining matters most"),
)


class ModelError(RuntimeError):
    """Something the operator has to decide about, reported rather than guessed."""


def _root(base_url: str) -> str:
    """The runtime's own root, from the OpenAI-compatible URL pointing at it."""
    trimmed = base_url.rstrip("/")
    return trimmed[: -len("/v1")] if trimmed.endswith("/v1") else trimmed


@dataclass(frozen=True)
class Runtime:
    """What is answering at ``base_url``, established by asking it."""

    base_url: str
    kind: str  # "ollama" | "openai-compatible" | "unreachable"
    version: str = ""

    @property
    def root(self) -> str:
        return _root(self.base_url)

    @property
    def is_ollama(self) -> bool:
        return self.kind == "ollama"

    @property
    def reachable(self) -> bool:
        return self.kind != "unreachable"

    @property
    def hostname(self) -> str:
        return urlparse(self.base_url).hostname or "?"


@dataclass(frozen=True)
class InstalledModel:
    name: str
    size_bytes: int = 0
    context: int | None = None

    @property
    def size(self) -> str:
        if not self.size_bytes:
            return "-"
        gb = self.size_bytes / 1_000_000_000
        return f"{gb:.1f} GB" if gb >= 1 else f"{self.size_bytes / 1_000_000:.0f} MB"


@dataclass
class Switch:
    """What ``model use`` did, so the command can report it rather than assert it."""

    model: str
    context: int | None = None
    native_context: int | None = None
    pulled: bool = False
    derived_from: str | None = None
    notes: list[str] = field(default_factory=list)


def probe(base_url: str, *, timeout: float = 4.0) -> Runtime:
    """Identify the runtime at ``base_url`` without assuming which one it is."""
    root = _root(base_url)
    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.get(f"{root}/api/version")
            if response.status_code == 200:
                version = str(response.json().get("version", ""))
                if version:
                    return Runtime(base_url=base_url, kind="ollama", version=version)
            if client.get(f"{base_url.rstrip('/')}/models").status_code == 200:
                return Runtime(base_url=base_url, kind="openai-compatible")
    except (httpx.HTTPError, ValueError, json.JSONDecodeError):
        pass
    return Runtime(base_url=base_url, kind="unreachable")


def _context_from_info(info: dict[str, object]) -> int | None:
    """Pull the window out of ``model_info``, whatever the architecture is called.

    The key is namespaced by architecture - ``qwen3.context_length``,
    ``llama.context_length`` - so matching on the suffix survives a model family
    this code has never heard of.
    """
    for key, value in info.items():
        if key.endswith(".context_length") and isinstance(value, int):
            return value
    return None


def context_window(runtime: Runtime, model: str, *, timeout: float = 10.0) -> int | None:
    """The window ``model`` declares, or None where the runtime will not say."""
    if not runtime.is_ollama:
        return None
    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.post(f"{runtime.root}/api/show", json={"model": model})
            if response.status_code != 200:
                return None
            body = response.json()
    except (httpx.HTTPError, ValueError):
        return None

    # A derived model's own num_ctx wins: it is the window it will actually run
    # with, where model_info only reports what the weights were trained for.
    parameters = str(body.get("parameters") or "")
    override = re.search(r"^\s*num_ctx\s+(\d+)", parameters, re.M)
    if override:
        return int(override.group(1))
    info = body.get("model_info")
    return _context_from_info(info) if isinstance(info, dict) else None


def installed(runtime: Runtime, *, timeout: float = 10.0) -> tuple[InstalledModel, ...]:
    """What the runtime already has, cheapest possible call - no model is loaded."""
    if not runtime.reachable:
        return ()
    try:
        with httpx.Client(timeout=timeout) as client:
            if runtime.is_ollama:
                response = client.get(f"{runtime.root}/api/tags")
                response.raise_for_status()
                rows = response.json().get("models") or []
                return tuple(
                    InstalledModel(
                        name=str(row.get("model") or row.get("name") or ""),
                        size_bytes=int(row.get("size") or 0),
                    )
                    for row in rows
                    if row.get("model") or row.get("name")
                )
            response = client.get(f"{runtime.base_url.rstrip('/')}/models")
            response.raise_for_status()
            rows = response.json().get("data") or []
            return tuple(InstalledModel(name=str(row["id"])) for row in rows if row.get("id"))
    except (httpx.HTTPError, ValueError, KeyError):
        return ()


def pull(runtime: Runtime, model: str, *, timeout: float = 3600.0) -> None:
    """Download ``model`` into the runtime. Minutes, and gigabytes, so it is loud."""
    if not runtime.is_ollama:
        raise ModelError(
            f"{runtime.hostname} is not Ollama, so OpenKnowledge cannot download "
            f"{model!r} for you. Load it into that runtime yourself, then re-run this."
        )
    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.post(
                f"{runtime.root}/api/pull", json={"model": model, "stream": False}
            )
            response.raise_for_status()
            body = response.json()
    except (httpx.HTTPError, ValueError) as exc:
        raise ModelError(f"could not download {model!r}: {exc}") from exc
    if isinstance(body, dict) and body.get("error"):
        raise ModelError(f"could not download {model!r}: {body['error']}")


def derived_name(model: str, context: int) -> str:
    """A stable, legal tag for the resized copy.

    Deterministic on purpose: running the same `model use` twice rebuilds the
    same name rather than accumulating near-identical models on disk.
    """
    stem = re.sub(r"[^a-z0-9._-]+", "-", model.lower().replace(":", "-")).strip("-")
    return f"{stem}-ok{context}"


def create_resized(runtime: Runtime, *, base: str, context: int, timeout: float = 600.0) -> str:
    """Build a copy of ``base`` that runs with a ``context``-token window.

    Ollama's OpenAI-compatible endpoint has no field for this, so the window has
    to be baked into a model. Cheap: the weights are shared, only a manifest is
    written.
    """
    if not runtime.is_ollama:
        raise ModelError(f"{runtime.hostname} is not Ollama; cannot resize a model there")

    name = derived_name(base, context)
    payload = {
        "model": name,
        "from": base,
        "parameters": {"num_ctx": context},
        "stream": False,
    }
    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.post(f"{runtime.root}/api/create", json=payload)
            response.raise_for_status()
            body = response.json()
    except (httpx.HTTPError, ValueError) as exc:
        raise ModelError(
            f"could not build a {context:,}-token copy of {base!r}: {exc}. "
            f"Ollama 0.5.13 or newer is needed for this; check `ollama --version`."
        ) from exc
    if isinstance(body, dict) and body.get("error"):
        raise ModelError(f"could not build {name!r}: {body['error']}")
    return name


def switch(
    runtime: Runtime,
    model: str,
    *,
    context: int | None = None,
    allow_download: bool = True,
) -> Switch:
    """Work out what ``OK_LOCAL_MODEL`` should become, doing the work it implies.

    Returns what happened rather than printing it, so the CLI reports and the
    tests assert on the same object.
    """
    result = Switch(model=model)

    if not runtime.reachable:
        raise ModelError(
            f"nothing is answering at {runtime.base_url}. Start your model runtime "
            "first - `ollama serve` if you use Ollama - or point OK_LOCAL_BASE_URL "
            "at the machine that runs it."
        )

    if not runtime.is_ollama:
        # vLLM, llama.cpp, LM Studio: the window is fixed when the server is
        # launched, so the honest thing is to record it and say what to relaunch.
        have = {m.name for m in installed(runtime)}
        if have and model not in have:
            result.notes.append(
                f"{runtime.hostname} does not list {model!r} "
                f"(it offers: {', '.join(sorted(have)[:6])})"
            )
        result.context = context
        if context:
            # Named precisely, because the two llama.cpp servers spell it
            # differently and a flag that is nearly right fails at launch.
            result.notes.append(
                f"{runtime.hostname} is not Ollama, so the window could not be set from "
                f"here. Relaunch that server with the window it needs - llama-server: "
                f"`--ctx-size {context}`, llama-cpp-python: `--n_ctx {context}`, "
                f"vLLM: `--max-model-len {context}` - and this setting will match it."
            )
        return result

    have = {m.name for m in installed(runtime)}
    if model not in have:
        if not allow_download:
            raise ModelError(
                f"{model!r} is not installed. Run it without --no-download, or "
                f"`ollama pull {model}` yourself."
            )
        pull(runtime, model)
        result.pulled = True

    native = context_window(runtime, model)
    result.native_context = native

    if context is None:
        result.context = native
        return result

    if native is not None and context <= native:
        # Already fits: a derived copy would only pin a smaller window than the
        # weights allow, which is a downgrade dressed as a setting.
        result.context = context
        if context < native:
            result.notes.append(
                f"{model} already declares {native:,} tokens; recording {context:,} as the "
                "working limit without building a resized copy."
            )
        return result

    if native is not None and context > native:
        result.notes.append(
            f"{context:,} tokens is beyond the {native:,} this model declares. Ollama will "
            "extend it, but quality past a model's trained window is a thing to measure, "
            "not assume - run `openknowledge eval` before trusting it."
        )
    result.model = create_resized(runtime, base=model, context=context)
    result.derived_from = model
    result.context = context
    return result


# --- persistence -----------------------------------------------------------


def write_env(path: Path, values: dict[str, str]) -> list[str]:
    """Set ``values`` in a dotenv file, leaving every other line exactly as it was.

    An operator's .env has their comments and their ordering in it. Rewriting it
    from the settings object would silently drop both, so this edits in place and
    appends only what is genuinely new.
    """
    lines = path.read_text().splitlines() if path.exists() else []
    remaining = dict(values)
    changed: list[str] = []

    for index, line in enumerate(lines):
        key = line.split("=", 1)[0].strip().lstrip("#").strip()
        if key in remaining:
            new = f"{key}={remaining.pop(key)}"
            if new != line:
                lines[index] = new
                changed.append(key)

    for key, value in remaining.items():
        lines.append(f"{key}={value}")
        changed.append(key)

    path.write_text("\n".join(lines).rstrip("\n") + "\n")
    return changed
