"""The models the desktop app runs on - pinned to the bytes we measured.

Every figure the project publishes for the self-hosted tier (100% accuracy,
$0 per question, the golden runs in ``evals/measured/``) was produced by two
specific GGUF files. The desktop app downloads exactly those bytes, verified
by SHA-256, or it does not run. "Latest" is not a version.

URLs, sizes and hashes verified against Hugging Face on 2026-08-28: the
``x-linked-etag`` of each resolve URL equals the SHA-256 recorded here, so a
mismatch at download time means the file changed upstream, not that this
table went stale. ``tools/verify_model_manifest.py`` re-checks without
downloading.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelFile:
    """One downloadable model artifact, pinned by content."""

    filename: str
    url: str
    sha256: str
    size_bytes: int
    purpose: str  # "chat" | "embedding"
    license: str
    context_tokens: int

    @property
    def size_human(self) -> str:
        mb = self.size_bytes / 1_000_000
        return f"{mb / 1000:.1f} GB" if mb >= 1000 else f"{mb:.0f} MB"


CHAT_MODEL = ModelFile(
    filename="Qwen3-4B-Instruct-2507-Q4_K_M.gguf",
    url=(
        "https://huggingface.co/unsloth/Qwen3-4B-Instruct-2507-GGUF"
        "/resolve/main/Qwen3-4B-Instruct-2507-Q4_K_M.gguf"
    ),
    sha256="3605803b982cb64aead44f6c1b2ae36e3acdb41d8e46c8a94c6533bc4c67e597",
    size_bytes=2_497_281_120,
    purpose="chat",
    license="Apache-2.0",
    context_tokens=8192,
)

EMBEDDING_MODEL = ModelFile(
    filename="nomic-embed-text-v1.5.Q4_K_M.gguf",
    url=(
        "https://huggingface.co/nomic-ai/nomic-embed-text-v1.5-GGUF"
        "/resolve/main/nomic-embed-text-v1.5.Q4_K_M.gguf"
    ),
    sha256="d4e388894e09cf3816e8b0896d81d265b55e7a9fff9ab03fe8bf4ef5e11295ac",
    size_bytes=84_106_624,
    purpose="embedding",
    license="Apache-2.0",
    context_tokens=2048,
)

MODELS = (CHAT_MODEL, EMBEDDING_MODEL)

TOTAL_DOWNLOAD_BYTES = sum(m.size_bytes for m in MODELS)
