"""Print the update handoff command, as JSON, for the packaging workflow.

The CI upgrade job runs the command the app itself builds rather than a
hand-written approximation of it, so that what is measured on Windows is the
same string `spawn_installer` would have run. Keeping the call here rather
than inline in the workflow avoids a PowerShell here-string, whose closing
delimiter has to sit at column 0 and cannot live inside YAML.

    python tools/print_handoff.py <installer> <relaunch> <pid>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from openknowledge.desktop.update import spawn_command


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(__doc__, file=sys.stderr)
        return 2
    installer, relaunch, pid = argv
    print(json.dumps(spawn_command(Path(installer), Path(relaunch), wait_for_pid=int(pid))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
