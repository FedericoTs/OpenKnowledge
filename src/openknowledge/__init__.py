"""OpenKnowledge - cheap, private, deterministic enterprise document Q&A.

The design goal is stated as a cost target, not a feature list: answering an
internal-documents question should cost close to nothing per call, and the same
question asked twice should produce the same answer.

See ``docs/ARCHITECTURE.md`` for how the pieces fit together.
"""

# Resolved from the installed package so it cannot drift from pyproject -
# the literal it replaced sat at 0.1.0 while the product shipped 0.2.1.
try:
    from importlib.metadata import version as _version

    __version__ = _version("openknowledge")
except Exception:  # pragma: no cover - source tree without install metadata
    __version__ = "0.0.0"

__all__ = ["__version__"]
