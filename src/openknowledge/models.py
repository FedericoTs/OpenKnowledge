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
import os
import re
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

import httpx

#: What `model use` pins when no window is asked for. Twice Ollama's own
#: default, which this project's defaults - k=6 retrieval plus 1,500 answer
#: tokens, about 3,800 - very nearly fill. Comfortable on any machine that can
#: hold the model at all, and a number an operator can raise knowing what it is.
DEFAULT_CONTEXT = 8192

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


@dataclass(frozen=True)
class Window:
    """What a model *may* do, and what the runtime *will* do. Not the same number.

    ``declared`` is the context length in the weights - 40,960 for qwen3:8b.
    ``pinned`` is ``num_ctx`` on the model itself, which is the only thing that
    decides what Ollama actually allocates.

    Ollama runs a model at its own default (4,096 unless OLLAMA_CONTEXT_LENGTH
    says otherwise) regardless of what the weights allow, and nothing in its API
    reports that server-side setting. So an unpinned model's real window is not
    knowable from here - and recording the declared length as if it were would
    be ten times too large for qwen3:8b, waving through prompts the runtime then
    truncates. Which is the exact failure this whole feature exists to prevent.
    """

    declared: int | None = None
    pinned: int | None = None

    @property
    def effective(self) -> int | None:
        """What may be recorded and enforced. None means genuinely unknown."""
        return self.pinned


def context_window(runtime: Runtime, model: str, *, timeout: float = 10.0) -> Window:
    """Ask what ``model`` declares and what, if anything, is pinned on it."""
    if not runtime.is_ollama:
        return Window()
    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.post(f"{runtime.root}/api/show", json={"model": model})
            if response.status_code != 200:
                return Window()
            body = response.json()
    except (httpx.HTTPError, ValueError):
        return Window()

    parameters = str(body.get("parameters") or "")
    override = re.search(r"^\s*num_ctx\s+(\d+)", parameters, re.M)
    info = body.get("model_info")
    return Window(
        declared=_context_from_info(info) if isinstance(info, dict) else None,
        pinned=int(override.group(1)) if override else None,
    )


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


def pull(
    runtime: Runtime,
    model: str,
    *,
    on_progress: Callable[[str, int, int], None] | None = None,
) -> None:
    """Download ``model`` into the runtime, reporting progress as it arrives.

    Gigabytes and minutes. Asking for it without ``stream`` returns one response
    at the end, which means a terminal that sits blank for ten minutes and looks
    hung - so this streams, and hands each update to ``on_progress`` as
    ``(status, completed_bytes, total_bytes)``.

    The read timeout is per-chunk rather than for the whole download: a slow
    connection should not fail, but a stalled one should.
    """
    if not runtime.is_ollama:
        raise ModelError(
            f"{runtime.hostname} is not Ollama, so OpenKnowledge cannot download "
            f"{model!r} for you. Load it into that runtime yourself, then re-run this."
        )

    timeout = httpx.Timeout(connect=15.0, read=180.0, write=30.0, pool=15.0)
    try:
        with (
            httpx.Client(timeout=timeout) as client,
            client.stream(
                "POST", f"{runtime.root}/api/pull", json={"model": model, "stream": True}
            ) as response,
        ):
            response.raise_for_status()
            for line in response.iter_lines():
                if not line.strip():
                    continue
                event = json.loads(line)
                if event.get("error"):
                    raise ModelError(f"could not download {model!r}: {event['error']}")
                if on_progress is not None:
                    on_progress(
                        str(event.get("status", "")),
                        int(event.get("completed") or 0),
                        int(event.get("total") or 0),
                    )
    except (httpx.HTTPError, ValueError) as exc:
        raise ModelError(f"could not download {model!r}: {exc}") from exc


def tagged(name: str) -> str:
    """``qwen3-8b-ok8192`` and ``qwen3-8b-ok8192:latest`` are the same model.

    Ollama stores an untagged name with the implicit ``:latest`` and reports it
    that way from ``/api/tags``, while accepting either form everywhere else.
    Comparing the two strings exactly is how ``model use`` could build a model
    and ``model status`` then report it missing, in the same minute.

    The tag is whatever follows a colon in the last path segment, so a registry
    path like ``hf.co/user/repo`` is still untagged.
    """
    return name if ":" in name.rsplit("/", 1)[-1] else f"{name}:latest"


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
    pin: bool = True,
    allow_download: bool = True,
    on_progress: Callable[[str, int, int], None] | None = None,
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
        have = {tagged(m.name) for m in installed(runtime)}
        if have and tagged(model) not in have:
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

    have = {tagged(m.name) for m in installed(runtime)}
    if tagged(model) not in have:
        if not allow_download:
            raise ModelError(
                f"{model!r} is not installed. Run it without --no-download, or "
                f"`ollama pull {model}` yourself."
            )
        pull(runtime, model, on_progress=on_progress)
        result.pulled = True

    window = context_window(runtime, model)
    result.native_context = window.declared

    if context is None and not pin:
        # Explicitly asked to leave it alone. Record only what is knowable, so
        # the fit check stays off rather than checking against a guess.
        result.context = window.pinned
        if window.pinned is None:
            result.notes.append(
                "no window recorded, so no prompt will be checked for fit. Ollama will "
                "run this at its own default, which its API does not report."
            )
        return result

    if context is None:
        # Nothing asked for. Keep whatever is already pinned; otherwise pin the
        # default, because the alternative is a configuration that is quietly
        # marginal. Ollama runs an unpinned model at 4,096 (unless
        # OLLAMA_CONTEXT_LENGTH says otherwise, which its API does not report),
        # and this project's own defaults - k=6 retrieval, 1,500 answer tokens -
        # come to roughly 3,800. That fits, barely, until one longer document
        # does not, and then the prompt is truncated from the front where the
        # grounding rules are. Leaving that to chance is not a default.
        if window.pinned is not None:
            result.context = window.pinned
            return result
        context = DEFAULT_CONTEXT
        result.notes.append(
            f"pinned at {DEFAULT_CONTEXT:,} tokens. Ollama runs an unpinned model at "
            "4,096, which this project's own defaults very nearly fill, and its API does "
            "not report the server-side setting - so the window is pinned rather than "
            f"assumed. {model} declares "
            f"{f'{window.declared:,}' if window.declared else 'more'}: raise it with "
            "`--context N` if you have the memory for it, or keep the runtime's own "
            "default with `--no-pin`."
        )

    if window.pinned == context:
        # Already exactly this. Rebuilding would be a no-op with a download's
        # worth of ceremony.
        result.context = context
        return result

    if window.declared is not None and context > window.declared:
        result.notes.append(
            f"{context:,} tokens is beyond the {window.declared:,} this model declares. "
            "Ollama will extend it, but quality past a model's trained window is a thing "
            "to measure, not assume - run `openknowledge eval` before trusting it."
        )

    # Always build, even well under the declared length. Pinning num_ctx is the
    # only way to know what Ollama allocates; recording a number without it
    # would be a setting that describes nothing.
    result.model = create_resized(runtime, base=model, context=context)
    result.derived_from = model
    result.context = context
    return result


# --- persistence -----------------------------------------------------------


def write_env(path: Path, values: dict[str, str], *, private: bool = False) -> list[str]:
    """Set ``values`` in a dotenv file, leaving every other line exactly as it was.

    An operator's .env has their comments and their ordering in it. Rewriting it
    from the settings object would silently drop both, so this edits in place and
    appends only what is genuinely new.

    The write is atomic - a temp file beside the target, then rename - so two
    writers cannot leave a half-file or silently drop each other's keys.
    ``private`` restricts the file to its owner *before* any content lands in
    it: a token written under umask 022 is world-readable, and a bearer token
    any local account can read is not a secret.
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

    content = "\n".join(lines).rstrip("\n") + "\n"
    descriptor, temp_name = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.")
    try:
        if private:
            os.fchmod(descriptor, 0o600)
        elif path.exists():
            os.fchmod(descriptor, path.stat().st_mode & 0o777)
        with os.fdopen(descriptor, "w") as handle:
            handle.write(content)
        os.replace(temp_name, path)
    except BaseException:
        os.unlink(temp_name)
        raise
    if private:
        os.chmod(path, 0o600)
    return changed


# --- keeping the model warm -------------------------------------------------


def warm(
    runtime: Runtime,
    model: str,
    *,
    keep_alive: str = "",
    timeout: float = 600.0,
) -> float:
    """Load ``model`` into memory now, so the next question does not.

    The first call after idle silently absorbs a full model load - minutes on a
    laptop CPU - and it lands on whoever happens to ask the first question.
    Paying that cost at startup, in the background, moves it to the one moment
    nobody is waiting.

    On Ollama this uses the native generate endpoint with an empty prompt,
    which loads the model without producing a token, and ``keep_alive`` pins
    how long it stays resident (Ollama's own default is five minutes; the
    OpenAI-compatible endpoint has no field for this, which is why the native
    one is used). Other runtimes - llama.cpp, vLLM - keep their model loaded
    for the life of the process, so a one-token completion both warms them and
    proves the pipe.

    Returns how long the load took, so the caller can log something truthful.
    Raises ModelError when nothing could be warmed; callers treat that as a
    note, never an outage - the cascade already handles a cold model, just
    slowly.
    """
    import time as _time

    started = _time.monotonic()
    try:
        with httpx.Client(timeout=timeout) as client:
            if runtime.is_ollama:
                payload: dict[str, object] = {"model": model, "prompt": "", "stream": False}
                if keep_alive:
                    payload["keep_alive"] = keep_alive
                response = client.post(f"{runtime.root}/api/generate", json=payload)
                response.raise_for_status()
            else:
                response = client.post(
                    f"{runtime.base_url.rstrip('/')}/chat/completions",
                    json={
                        "model": model,
                        "messages": [{"role": "user", "content": "ok"}],
                        "max_tokens": 1,
                        "temperature": 0,
                    },
                )
                response.raise_for_status()
    except httpx.HTTPError as exc:
        raise ModelError(f"could not warm {model!r}: {exc or type(exc).__name__}") from exc
    return _time.monotonic() - started
