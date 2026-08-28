"""Where state lives, decided by how OpenKnowledge is being run.

Two audiences use this code and they keep state in opposite places. A server
operator works inside a directory that is the deployment - `.env`, `./data`,
`./documents`, all version-adjacent, all mounted into the container. A person
who double-clicked an installer has no working directory in any meaningful
sense: their process starts wherever the shortcut says, and CWD-relative state
would scatter databases across whatever folder was current at launch.

So there are exactly two modes, chosen by evidence rather than configuration:

* **project mode** - the working directory already carries this deployment's
  state (a `.env`, or a previously created `data/`). Everything stays
  CWD-relative, exactly as before this module existed. Every current install,
  the container, and the test suite all look like this.
* **app mode** - nothing in the working directory says "this is a deployment",
  so state goes where the platform keeps per-user application data:
  ``%LOCALAPPDATA%\\OpenKnowledge`` on Windows, ``~/Library/Application
  Support/OpenKnowledge`` on macOS, ``$XDG_DATA_HOME/openknowledge`` elsewhere.

``OK_STATE_DIR`` overrides both, which is also what tests use.

Nothing here creates directories. Deciding where state *would* live must stay
free of side effects, because commands that promise to write nothing - `audit`
above all - call this too.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

import platformdirs

_APP_NAME = "OpenKnowledge"


@dataclass(frozen=True)
class StatePaths:
    """The resolved locations, and which rule chose them."""

    mode: str  # "project" | "app" | "override"
    root: Path

    @property
    def env_file(self) -> Path:
        return self.root / ".env"

    @property
    def data_dir(self) -> Path:
        return self.root / "data"

    @property
    def documents_dir(self) -> Path:
        return self.root / "documents"


def _project_markers(cwd: Path) -> bool:
    """Whether ``cwd`` is already a deployment.

    A `.env` is the deliberate marker; an existing `data/` directory covers
    installs made before `.env` was written (install.sh creates both). The
    checkout itself carries `.env.example` and `pyproject.toml`, so a developer
    running from source without a `.env` yet still lands in project mode.
    """
    return any(
        (cwd / marker).exists() for marker in (".env", "data", ".env.example", "pyproject.toml")
    )


def state_paths(cwd: Path | None = None) -> StatePaths:
    override = os.environ.get("OK_STATE_DIR", "").strip()
    if override:
        return StatePaths(mode="override", root=Path(override).expanduser())

    here = (cwd or Path.cwd()).resolve()
    if _project_markers(here):
        return StatePaths(mode="project", root=here)

    return StatePaths(
        mode="app",
        root=Path(platformdirs.user_data_dir(_APP_NAME, appauthor=False)),
    )


def is_frozen() -> bool:
    """Whether this process is a PyInstaller-style bundle rather than a checkout."""
    return getattr(sys, "frozen", False) is True
