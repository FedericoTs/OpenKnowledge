"""Leave the machine somewhere to write.

A disk at zero does not fail politely. SQLite cannot commit, the index
cannot be written, the log the operator would read to find out why cannot
be appended to, and the first symptom is usually an answer engine that has
stopped answering for a reason nothing on the page explains. Every one of
those is downstream of somebody adding the last file that fitted.

So both of the ways this product writes a lot - a colleague dropping
documents in, and a first launch fetching 2.6 GB of model weights - ask
first whether there will be room left afterwards, and refuse in a sentence
that names the number rather than failing in the middle with an OSError.

The floor is free space *after* the write, not before it: "there is 40 MB
free and this file is 30 MB" is not a reason to proceed. What it protects
is the space everything else needs, which is why the default is not zero.
A deployment that genuinely wants to run its disk to the edge sets
``OK_DISK_FLOOR_MB=0`` and owns the consequence.

This is a floor under free space, not a ceiling on the corpus. Two
different questions: this one is "will the machine still work", and it
holds whatever else is filling the disk. "How much may this corpus grow
to" is a quota, and nothing here answers it.
"""

from __future__ import annotations

import shutil
from pathlib import Path


def free_bytes(path: str | Path) -> int:
    """Free space on the filesystem holding ``path``.

    Walks up to the first parent that exists, because the caller often means
    a directory it is about to create.
    """
    here = Path(path).absolute()
    while not here.exists():
        parent = here.parent
        if parent == here:  # the root of a filesystem that is not there
            return 0
        here = parent
    return shutil.disk_usage(here).free


def no_room_for(path: str | Path, wanted: int, floor_mb: int) -> str | None:
    """Why ``wanted`` bytes cannot be written at ``path``, or None.

    ``floor_mb`` of zero turns the check off entirely.
    """
    if floor_mb <= 0 or wanted <= 0:
        return None
    floor = floor_mb * 1_000_000
    free = free_bytes(path)
    if free - wanted >= floor:
        return None
    return (
        f"this needs {wanted / 1_000_000:.1f} MB and the disk has "
        f"{free / 1_000_000:.1f} MB free, which would leave less than the "
        f"{floor_mb} MB this server keeps spare so it can still write its "
        "index, its databases and its log. Free some space, or lower "
        "OK_DISK_FLOOR_MB if you mean to run this close to the edge."
    )
