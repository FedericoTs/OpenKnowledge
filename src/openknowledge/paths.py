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


def _mentions_openknowledge(path: Path) -> bool:
    """Whether a config file is *ours*, not just present.

    Read defensively: an unreadable file is simply not evidence.
    """
    try:
        head = path.read_text(errors="replace")[:16384]
    except OSError:
        return False
    return "OK_" in head or "openknowledge" in head.lower()


def _project_markers(cwd: Path) -> bool:
    """Whether ``cwd`` is *an OpenKnowledge deployment*, not just any project.

    The first version accepted any `.env`, any `pyproject.toml`, or anything
    named `data` - which meant cd-ing into an unrelated Python or Node repo
    silently switched which corpus, token and config every command saw, and
    `openknowledge index` run there would scatter state into a stranger's
    project. Evidence now has to be specific to this tool:

    * a `.env` or `.env.example` that actually contains OK_ settings;
    * this tool's own databases under `data/`;
    * the paired `data/` + `documents/` directories an install creates
      before its `.env` exists;
    * the source checkout itself (`src/openknowledge`, or a `pyproject.toml`
      that names openknowledge).
    """
    for env_name in (".env", ".env.example"):
        env = cwd / env_name
        if env.is_file() and _mentions_openknowledge(env):
            return True
    data = cwd / "data"
    if any((data / db).is_file() for db in ("openknowledge.db", "knowledge.db", "vectors.db")):
        return True
    if data.is_dir() and (cwd / "documents").is_dir():
        return True
    if (cwd / "src" / "openknowledge").is_dir():
        return True
    pyproject = cwd / "pyproject.toml"
    return pyproject.is_file() and _mentions_openknowledge(pyproject)


def state_paths(cwd: Path | None = None) -> StatePaths:
    override = os.environ.get("OK_STATE_DIR", "").strip()
    if override:
        return StatePaths(mode="override", root=Path(override).expanduser())

    try:
        here = (cwd or Path.cwd()).resolve()
    except OSError:
        # The working directory no longer exists (a removed worktree, a cleaned
        # tmp dir). No directory means no deployment there, by definition -
        # not a traceback.
        return StatePaths(
            mode="app",
            root=Path(platformdirs.user_data_dir(_APP_NAME, appauthor=False)),
        )
    if _project_markers(here):
        return StatePaths(mode="project", root=here)

    return StatePaths(
        mode="app",
        root=Path(platformdirs.user_data_dir(_APP_NAME, appauthor=False)),
    )


def is_frozen() -> bool:
    """Whether this process is a PyInstaller-style bundle rather than a checkout."""
    return getattr(sys, "frozen", False) is True
