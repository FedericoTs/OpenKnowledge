"""Find the shipped web assets, wherever this build put them.

The widget and the site are files, and "next to the source tree" is only true
in a checkout. A wheel ships them inside the package (pyproject force-includes
``web/`` as ``openknowledge/web/``), a PyInstaller bundle unpacks them under
``sys._MEIPASS``, and the container bakes them at ``/app/web``. The old code
knew one of those layouts - ``Path(__file__).parents[3]`` - which is why a
wheel install served "Chat widget not found" while every test passed against
the checkout.

One resolver, all layouts, in the order a packaged build should win them.
"""

from __future__ import annotations

import sys
from pathlib import Path

_PACKAGE_DIR = Path(__file__).resolve().parent


def _roots() -> list[Path]:
    roots: list[Path] = []
    # A frozen bundle: PyInstaller unpacks data files under _MEIPASS, keeping
    # the destination paths given at build time ("web/..." and the package's
    # own "openknowledge/web/...").
    bundle = getattr(sys, "_MEIPASS", None)
    if bundle:
        roots += [Path(bundle) / "web", Path(bundle) / "openknowledge" / "web"]
    # A wheel install: web/ lives inside the package itself.
    roots.append(_PACKAGE_DIR / "web")
    # A source checkout: src/openknowledge/assets.py -> repo root / web.
    roots.append(_PACKAGE_DIR.parents[1] / "web")
    # The container image bakes the checkout at /app.
    roots.append(Path("/app/web"))
    return roots


def find_asset(relative: str) -> Path | None:
    """The first real file called ``relative`` under any known web root."""
    for root in _roots():
        candidate = root / relative
        if candidate.is_file():
            return candidate
    return None
