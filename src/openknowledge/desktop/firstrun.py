"""First-run feedback: 2.6 GB must never download in silence.

The launcher is a windowed executable on Windows - stdout goes nowhere - so
a first run that quietly pulls models for ten minutes is indistinguishable
from a hang, and a person will kill it at 94%. A small tkinter window shows
which file, how far, and how big. Where tkinter cannot open (no display, a
stripped Python), progress falls back to the console, which is where a
person running from a terminal is looking anyway.

tkinter is pumped from the download's own progress callback rather than a
second thread: a progress bar needs no event loop of its own, and one
thread means no half of the state is ever stale.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Protocol

from .download import ensure_model
from .manifest import ModelFile


class Reporter(Protocol):
    def update(self, model: ModelFile, done: int, total: int, position: str) -> None: ...

    def close(self) -> None: ...


class ConsoleReporter:
    """Progress on stderr-adjacent stdout, one redrawn line per file."""

    def __init__(self) -> None:
        self._last_line = ""

    def update(self, model: ModelFile, done: int, total: int, position: str) -> None:
        line = (
            f"{position}  {model.filename}: {done / 1_000_000:,.0f} / {total / 1_000_000:,.0f} MB"
        )
        if line != self._last_line:
            print("\r" + line.ljust(len(self._last_line)), end="", flush=True)
            self._last_line = line

    def close(self) -> None:
        if self._last_line:
            print()


class TkReporter:
    """A small window with a name, a bar and a number. Nothing else."""

    def __init__(self) -> None:
        import tkinter
        from tkinter import ttk

        self.root = tkinter.Tk()
        self.root.title("OpenKnowledge — first run")
        self.root.resizable(False, False)
        frame = ttk.Frame(self.root, padding=16)
        frame.grid()
        ttk.Label(
            frame,
            text="Downloading the models this machine will answer from.\n"
            "This happens once; every answer after it is local and free.",
            justify="left",
        ).grid(sticky="w")
        self.label = ttk.Label(frame, text="")
        self.label.grid(sticky="w", pady=(12, 4))
        self.bar = ttk.Progressbar(frame, length=380, maximum=1000)
        self.bar.grid()
        self.root.update()

    def update(self, model: ModelFile, done: int, total: int, position: str) -> None:
        self.label.config(
            text=f"{position}  {model.filename}  "
            f"({done / 1_000_000:,.0f} / {total / 1_000_000:,.0f} MB)"
        )
        self.bar["value"] = 1000 * done / max(total, 1)
        self.root.update()

    def close(self) -> None:
        self.root.destroy()


def make_reporter() -> Reporter:
    """Console when a console exists; the Tk window only when nothing else does.

    The Tk window exists for the windowed executable, where ``sys.stdout``
    is ``None`` and printed progress would vanish. Anywhere a real stdout
    exists - the ``desktop`` CLI command, a terminal, CI - the console line
    is where the person is already looking, and unlike Tk it cannot fail:
    a broken Tcl installation can abort the whole process natively, below
    Python's ability to catch it, which a progress nicety must never risk.
    ``OK_HEADLESS`` forces console mode for automation either way.
    """
    if os.environ.get("OK_HEADLESS") or sys.stdout is not None:
        return ConsoleReporter()
    try:
        return TkReporter()
    except Exception:  # no display, no tkinter - degrade to silent no-op prints
        return ConsoleReporter()


def fetch_models(models: tuple[ModelFile, ...], into: Path) -> None:
    """Download whatever is missing, with a face on it. Raises DownloadError."""
    reporter = make_reporter()
    try:
        for index, model in enumerate(models, start=1):
            position = f"model {index} of {len(models)}"

            def report(m: ModelFile, done: int, total: int, _at: str = position) -> None:
                reporter.update(m, done, total, _at)

            reporter.update(model, 0, model.size_bytes, position)
            ensure_model(model, into, progress=report)
            reporter.update(model, model.size_bytes, model.size_bytes, position)
    finally:
        reporter.close()
