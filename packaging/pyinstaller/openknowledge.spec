# -*- mode: python ; coding: utf-8 -*-
"""One folder, two executables: the app people click, the CLI people script.

``OpenKnowledge`` (windowed) is the Start-menu entry - the desktop launcher.
``openknowledge`` (console) is the same CLI a pip install provides, for the
person who opens a terminal. Both share one _internal folder, so the bundle
costs one runtime, not two.

Data files are added explicitly rather than through collect_data_files():
the development install is editable, where package-data collection is the
kind of thing that works until the day it silently doesn't. assets.py
resolves ``web`` from the bundle root (sys._MEIPASS), pricing.yaml through
importlib.resources - both paths below match those lookups exactly.

Build (any OS; CI does Windows):

    uv pip install -e ".[desktop,packaging,anthropic]"
    uv run pyinstaller packaging/pyinstaller/openknowledge.spec

Output lands in dist/OpenKnowledge/.
"""

import sys
from pathlib import Path

ROOT = Path(SPECPATH).resolve().parents[1]  # noqa: F821 - SPECPATH is injected

datas = [
    (str(ROOT / "src" / "openknowledge" / "pricing.yaml"), "openknowledge"),
    (str(ROOT / "web"), "web"),
]

hiddenimports = []
if sys.platform == "win32":
    # pystray picks its backend with a runtime importlib call the static
    # analysis cannot follow.
    hiddenimports.append("pystray._win32")

common = dict(
    pathex=[str(ROOT / "src")],
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=["pytest", "playwright"],
    noarchive=False,
)

a_cli = Analysis([str(ROOT / "packaging" / "pyinstaller" / "entry_cli.py")], **common)
a_app = Analysis([str(ROOT / "packaging" / "pyinstaller" / "entry_app.py")], **common)

icon = str(ROOT / "packaging" / "windows" / "openknowledge.ico")

exe_cli = EXE(
    PYZ(a_cli.pure),
    a_cli.scripts,
    [],
    exclude_binaries=True,
    name="openknowledge",
    console=True,
    icon=icon if sys.platform == "win32" else None,
)

exe_app = EXE(
    PYZ(a_app.pure),
    a_app.scripts,
    [],
    exclude_binaries=True,
    name="OpenKnowledge",
    console=False,
    icon=icon if sys.platform == "win32" else None,
)

COLLECT(
    exe_cli,
    a_cli.binaries,
    a_cli.datas,
    exe_app,
    a_app.binaries,
    a_app.datas,
    name="OpenKnowledge",
)
