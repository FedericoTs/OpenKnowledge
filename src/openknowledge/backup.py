"""Everything an install would hate to lose, in one file it can carry.

The documents are the easy half: they are already files, and whoever put them
there usually still has them. The half worth protecting is what people *did* -
the pinned answers somebody wrote by hand, the folder rules somebody decided,
the contradictions somebody adjudicated, and the ledger that knows what has
been asked and what it cost. None of that exists anywhere else, and until this
module there was no way to copy it off a laptop and onto a server.

Three decisions worth stating, because each could reasonably have gone the
other way:

* **Secrets are never written into a backup.** An archive is a file that gets
  emailed, dropped in shared storage, and occasionally committed. A backup that
  quietly carries an API key is a leak waiting for someone to be helpful with
  it. What the archive records instead is the *names* of the settings that were
  set, so a restore can say exactly what has to be typed in again rather than
  leaving somebody to discover it when answers stop.

* **The vector index is left out.** It is derived from the documents and is
  rebuilt by the first index, so carrying it would double the archive to save
  a few minutes of CPU. The manifest says so rather than leaving a reader to
  wonder what happened to it.

* **Databases are copied through SQLite's own backup API, not the filesystem.**
  A server does not stop being asked questions because somebody is taking a
  backup, and `cp` of a database mid-write produces a file that looks fine and
  is not. `Connection.backup()` is the supported way to snapshot a live
  database and it is barely more code.
"""

from __future__ import annotations

import json
import shutil
import sqlite3
import time
import zipfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from . import __version__
from .config import Settings

#: Bumped when the archive layout changes in a way an older reader would get
#: wrong. A reader refuses a format from the future rather than guessing.
FORMAT = 1

MANIFEST = "manifest.json"

#: The databases worth carrying, and why each one is here.
_DATABASES = {
    "openknowledge.db": "pinned answers, the answer cache and the ledger",
    "knowledge.db": "folder access rules, resolved contradictions, proposals, document versions",
    "auth.db": "sign-in sessions",
}

#: Settings whose values must never reach an archive. Matched by suffix so a
#: setting added later is covered by naming it the way the others are named.
_SECRET_SUFFIXES = ("_key", "_secret", "_token", "_password")


class BackupError(RuntimeError):
    """A backup or restore that could not be completed, said plainly."""


@dataclass(frozen=True, slots=True)
class BackupSummary:
    path: Path
    databases: tuple[str, ...]
    documents: int
    bytes: int
    secrets_omitted: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RestoreSummary:
    databases: tuple[str, ...]
    documents: int
    #: Settings that were set when the backup was taken and are not in it.
    #: Answers will be refused until these are set again, so they are the
    #: first thing the caller should say out loud.
    secrets_to_reenter: tuple[str, ...] = ()
    from_version: str = ""


def _is_secret(name: str) -> bool:
    return name.endswith(_SECRET_SUFFIXES)


def _settings_for_manifest(settings: Settings) -> tuple[dict[str, object], tuple[str, ...]]:
    """The configuration worth recording, and the names of what was withheld."""
    kept: dict[str, object] = {}
    withheld: list[str] = []
    for name, value in settings.model_dump().items():
        if _is_secret(name):
            if value:  # only the ones that were actually set need re-entering
                withheld.append(f"OK_{name.upper()}")
            continue
        if isinstance(value, (str, int, float, bool)) or value is None:
            kept[name] = value
    return kept, tuple(sorted(withheld))


def _snapshot(source: Path, into: Path) -> bool:
    """A consistent copy of a live SQLite database, or False if there is none."""
    if not source.is_file():
        return False
    src = sqlite3.connect(f"file:{source}?mode=ro", uri=True)
    try:
        dest = sqlite3.connect(into)
        try:
            src.backup(dest)
        finally:
            dest.close()
    finally:
        src.close()
    return True


def _document_files(documents_dir: Path) -> Iterator[Path]:
    if not documents_dir.is_dir():
        return
    for path in sorted(documents_dir.rglob("*")):
        if path.is_file():
            yield path


def write_backup(settings: Settings, out: Path, *, include_documents: bool = True) -> BackupSummary:
    """Write one archive holding this install's state."""
    out = Path(out)
    if out.is_dir():
        raise BackupError(f"{out} is a directory; give a file path to write")
    out.parent.mkdir(parents=True, exist_ok=True)

    data_dir = Path(settings.data_dir)
    documents_dir = Path(settings.documents_dir)
    kept, withheld = _settings_for_manifest(settings)

    staged = out.parent / f".{out.name}.staging"
    if staged.exists():
        shutil.rmtree(staged)
    staged.mkdir(parents=True)
    try:
        carried: list[str] = []
        for name in _DATABASES:
            if _snapshot(data_dir / name, staged / name):
                carried.append(name)

        documents: list[Path] = list(_document_files(documents_dir)) if include_documents else []
        manifest = {
            "format": FORMAT,
            "created_at": time.time(),
            "created_by_version": __version__,
            "databases": {name: _DATABASES[name] for name in carried},
            "documents": len(documents),
            "documents_included": include_documents,
            "settings": kept,
            "secrets_not_included": list(withheld),
            "omitted": {
                "vectors.db": (
                    "derived from the documents and rebuilt by the first index, "
                    "so it is not carried"
                )
            },
        }

        with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr(MANIFEST, json.dumps(manifest, indent=2, sort_keys=True))
            for name in carried:
                archive.write(staged / name, f"data/{name}")
            for path in documents:
                archive.write(path, f"documents/{path.relative_to(documents_dir).as_posix()}")
    finally:
        shutil.rmtree(staged, ignore_errors=True)

    return BackupSummary(
        path=out,
        databases=tuple(carried),
        documents=len(documents),
        bytes=out.stat().st_size,
        secrets_omitted=withheld,
    )


def _names(manifest: dict[str, object], key: str) -> tuple[str, ...]:
    """A list of names out of a manifest, whatever it actually contains.

    The manifest is a file somebody handed us, so its shape is a claim rather
    than a fact. Anything that is not a collection of names reads as no names,
    which surfaces as "says it carries X and does not" or as an empty list of
    secrets - both of which are true and neither of which is a traceback.
    """
    value = manifest.get(key)
    if isinstance(value, dict):
        return tuple(str(k) for k in value)
    if isinstance(value, (list, tuple)):
        return tuple(str(v) for v in value)
    return ()


def read_manifest(archive: Path) -> dict[str, object]:
    """The archive's own account of itself, refused if it is not one of ours."""
    try:
        with zipfile.ZipFile(archive) as zf:
            manifest = json.loads(zf.read(MANIFEST))
    except (OSError, KeyError, zipfile.BadZipFile, ValueError) as exc:
        raise BackupError(f"{archive} is not an OpenKnowledge backup ({exc})") from exc
    if not isinstance(manifest, dict):
        raise BackupError(f"{archive} carries an unreadable manifest")
    found = manifest.get("format")
    if not isinstance(found, int) or found > FORMAT:
        raise BackupError(
            f"{archive} was written in backup format {found}, and this build reads "
            f"up to {FORMAT}. Restore it with the version that wrote it, or newer."
        )
    return manifest


def restore_backup(archive: Path, settings: Settings, *, force: bool = False) -> RestoreSummary:
    """Put an archive's state back, refusing to overwrite by accident.

    Everything is extracted and checked before anything is moved into place, so
    a truncated or foreign archive is refused with the install untouched. Once
    the moving starts it is file by file: a machine that loses power halfway
    through leaves a half-restored directory, which is why this refuses to run
    over existing state unless somebody says to.
    """
    manifest = read_manifest(archive)
    data_dir = Path(settings.data_dir)
    documents_dir = Path(settings.documents_dir)

    existing = [name for name in _DATABASES if (data_dir / name).is_file()]
    if existing and not force:
        raise BackupError(
            f"{data_dir} already holds {', '.join(existing)}. Restoring would replace "
            "the pins, access rules and history in them. Move them aside, or pass "
            "--force if that is what you want."
        )

    staged = data_dir.parent / ".restore-staging"
    if staged.exists():
        shutil.rmtree(staged)
    try:
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(staged)

        promised = _names(manifest, "databases")
        carried = [n for n in promised if (staged / "data" / n).is_file()]
        missing = [n for n in promised if n not in carried]
        if missing:
            raise BackupError(
                f"{archive} says it carries {', '.join(missing)} and does not. "
                "Nothing has been changed."
            )

        data_dir.mkdir(parents=True, exist_ok=True)
        for name in carried:
            shutil.move(str(staged / "data" / name), data_dir / name)

        restored_documents = 0
        staged_documents = staged / "documents"
        if staged_documents.is_dir():
            documents_dir.mkdir(parents=True, exist_ok=True)
            for path in sorted(staged_documents.rglob("*")):
                if not path.is_file():
                    continue
                target = documents_dir / path.relative_to(staged_documents)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(path), target)
                restored_documents += 1
    finally:
        shutil.rmtree(staged, ignore_errors=True)

    secrets = _names(manifest, "secrets_not_included")
    return RestoreSummary(
        databases=tuple(carried),
        documents=restored_documents,
        secrets_to_reenter=secrets,
        from_version=str(manifest.get("created_by_version", "")),
    )
