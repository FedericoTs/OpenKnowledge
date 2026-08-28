"""Model downloading that survives the realities of a 2.5 GB file.

A first run pulls about 2.6 GB over whatever network the machine has. The
rules here are the ones a person would want if the download died at 94%:

- **Resume, never restart.** Partial bytes land in ``<name>.part`` and a
  retry continues from where it stopped with an HTTP Range request.
- **Verify, never trust.** The file is finished only when its SHA-256
  matches the manifest. A mismatch deletes the bytes and says so - serving
  answers from a model other than the one the accuracy numbers were
  measured on is not a degraded mode, it is a different product.
- **Verify once.** A ``<name>.sha256-ok`` marker records a successful check
  so every later launch skips re-hashing 2.5 GB. The marker holds the hash
  it verified; a manifest change invalidates it.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from pathlib import Path

import httpx

from .manifest import ModelFile

# progress(model, bytes_done, bytes_total) - called about once per chunk.
Progress = Callable[[ModelFile, int, int], None]

_CHUNK = 1024 * 1024


class DownloadError(Exception):
    """A model could not be fetched or failed verification."""


def ensure_model(
    model: ModelFile,
    into: Path,
    progress: Progress | None = None,
    *,
    base_url: str | None = None,
) -> Path:
    """Return a verified local path for ``model``, downloading if needed.

    ``base_url`` rebases the manifest URL onto another host - it exists for
    tests, which should not need Hugging Face to prove resume logic works.
    """
    into.mkdir(parents=True, exist_ok=True)
    final = into / model.filename
    marker = into / (model.filename + ".sha256-ok")

    if final.is_file() and final.stat().st_size == model.size_bytes:
        if marker.is_file() and marker.read_text().strip() == model.sha256:
            return final
        if _sha256_of(final) == model.sha256:
            marker.write_text(model.sha256)
            return final
        # Right size, wrong bytes: worse than nothing. Start clean.
        final.unlink()
        marker.unlink(missing_ok=True)

    url = model.url
    if base_url is not None:
        url = base_url.rstrip("/") + "/" + model.filename

    part = into / (model.filename + ".part")
    digest = _download(model, url, part, progress)

    if digest != model.sha256:
        part.unlink(missing_ok=True)
        raise DownloadError(
            f"{model.filename}: downloaded bytes hash to {digest[:12]}…, the manifest "
            f"pins {model.sha256[:12]}…. The upstream file changed or the transfer was "
            "corrupted; the download was discarded. Nothing runs on unverified bytes."
        )
    part.replace(final)
    marker.write_text(model.sha256)
    return final


def _download(model: ModelFile, url: str, part: Path, progress: Progress | None) -> str:
    """Fetch ``url`` into ``part``, resuming what is already there.

    Returns the SHA-256 of the complete file. The hash is fed incrementally:
    existing partial bytes first, then each chunk as it lands, so verifying
    costs no second pass over the file.
    """
    hasher = hashlib.sha256()
    done = 0
    if part.is_file():
        size = part.stat().st_size
        if 0 < size <= model.size_bytes:
            with part.open("rb") as f:
                while chunk := f.read(_CHUNK):
                    hasher.update(chunk)
            done = size
        else:
            part.unlink()

    if done == model.size_bytes:
        return hasher.hexdigest()

    headers = {"Range": f"bytes={done}-"} if done else {}
    try:
        with (
            httpx.Client(follow_redirects=True, timeout=httpx.Timeout(30.0, read=120.0)) as client,
            client.stream("GET", url, headers=headers) as response,
        ):
            if done and response.status_code == 200:
                # The server ignored the Range request; the body is the whole
                # file, so the partial bytes and their hash start over.
                hasher = hashlib.sha256()
                done = 0
                part.unlink(missing_ok=True)
            elif response.status_code not in (200, 206):
                raise DownloadError(f"{model.filename}: HTTP {response.status_code} from {url}")
            mode = "ab" if done else "wb"
            with part.open(mode) as f:
                for chunk in response.iter_bytes(_CHUNK):
                    f.write(chunk)
                    hasher.update(chunk)
                    done += len(chunk)
                    if progress is not None:
                        progress(model, done, model.size_bytes)
    except httpx.HTTPError as exc:
        raise DownloadError(
            f"{model.filename}: download interrupted after {done:,} bytes ({exc}). "
            "Run again - it resumes from here."
        ) from exc

    if done != model.size_bytes:
        raise DownloadError(
            f"{model.filename}: server sent {done:,} bytes, the manifest pins "
            f"{model.size_bytes:,}. Partial data kept for resume."
        )
    return hasher.hexdigest()


def _sha256_of(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(_CHUNK):
            hasher.update(chunk)
    return hasher.hexdigest()
