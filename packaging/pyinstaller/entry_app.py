"""PyInstaller entry for the windowed executable: the desktop launcher.

Separate from the CLI entry so the Start-menu shortcut opens no console
window; everything else - models, servers, the app - is the launcher's job.
"""

import multiprocessing
import sys

from openknowledge.desktop.launcher import main

if __name__ == "__main__":
    multiprocessing.freeze_support()
    sys.exit(main())
