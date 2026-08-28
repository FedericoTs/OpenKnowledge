"""Check the desktop model manifest against upstream, without downloading.

The manifest pins each model by URL, size and SHA-256. Hugging Face stores
these files in Git LFS, and the resolve endpoint reports the LFS object id -
which IS the file's SHA-256 - in the ``x-linked-etag`` header, alongside the
byte size in ``x-linked-size``. Two HEAD requests therefore prove, without
moving 2.6 GB, either that the URLs still serve exactly the pinned bytes or
that upstream changed and the manifest (and the accuracy claims that go with
it) needs re-measuring.

Run it whenever the manifest changes, and before cutting an installer:

    uv run python tools/verify_model_manifest.py

Exits non-zero on any mismatch, so it can gate a release job.
"""

from __future__ import annotations

import sys

import httpx

from openknowledge.desktop.manifest import MODELS


def main() -> int:
    failures = 0
    # No redirect-following: the x-linked-* headers ride on Hugging Face's
    # own 302 to the CDN; the CDN's final response does not repeat them.
    with httpx.Client(follow_redirects=False, timeout=30.0) as client:
        for model in MODELS:
            response = client.head(model.url)
            etag = response.headers.get("x-linked-etag", "").strip('"')
            size = response.headers.get("x-linked-size", "")
            problems = []
            if response.status_code not in (200, 302):
                problems.append(f"HTTP {response.status_code}")
            if etag != model.sha256:
                problems.append(f"upstream sha {etag or '(missing)'} != pinned {model.sha256}")
            if size != str(model.size_bytes):
                problems.append(f"upstream size {size or '(missing)'} != pinned {model.size_bytes}")
            if problems:
                failures += 1
                print(f"MISMATCH  {model.filename}")
                for problem in problems:
                    print(f"          {problem}")
            else:
                print(f"ok        {model.filename}  ({model.size_human}, sha verified upstream)")

    if failures:
        print(
            f"\n{failures} model(s) no longer match the manifest. Do not ship: the pinned "
            "bytes are what the published numbers were measured on. Re-pin and re-measure."
        )
        return 1
    print("\nEvery pinned model is still served byte-for-byte at its URL.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
