"""Carrying an install somewhere else, and refusing to lose it on the way.

The documents are the easy half. The half that exists nowhere else is what
people decided - the pins somebody wrote by hand, the folder rules somebody
set, the contradictions somebody adjudicated - and until there was a backup
those lived in three SQLite files with no way off the machine.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import zipfile
from pathlib import Path

import pytest

from openknowledge.backup import (
    FORMAT,
    BackupError,
    read_manifest,
    restore_backup,
    write_backup,
)
from openknowledge.cache import AnswerStore
from openknowledge.config import Settings
from openknowledge.knowledge import KnowledgeStore


def _settings(root: Path, **kw) -> Settings:
    (root / "data").mkdir(parents=True, exist_ok=True)
    (root / "documents").mkdir(parents=True, exist_ok=True)
    return Settings(
        data_dir=str(root / "data"),
        documents_dir=str(root / "documents"),
        _env_file=None,  # type: ignore[call-arg]
        **kw,
    )


def test_what_a_person_decided_survives_the_move(tmp_path: Path) -> None:
    """The whole point: a pin written here answers there."""
    here, there = _settings(tmp_path / "here"), _settings(tmp_path / "there")
    (Path(here.documents_dir) / "leave.md").write_text("# Leave\n\n20 weeks.\n")

    with AnswerStore(here.db_path) as store:
        store.pin("how much parental leave", "20 weeks, fully paid.", author="hr")
    with KnowledgeStore(here.knowledge_db_path) as knowledge:
        knowledge.set_folder_access("finance", frozenset({"group:cfo-office"}))

    write_backup(here, tmp_path / "ok.zip")
    done = restore_backup(tmp_path / "ok.zip", there)

    assert done.documents == 1
    with AnswerStore(there.db_path) as store:
        pin = store.get_pin("how much parental leave")
        assert pin is not None and pin.answer == "20 weeks, fully paid."
        assert pin.author == "hr", "who wrote it travels with what they wrote"
    with KnowledgeStore(there.knowledge_db_path) as knowledge:
        assert knowledge.folder_rules()["finance"] == frozenset({"group:cfo-office"})
    assert (Path(there.documents_dir) / "leave.md").is_file()


def test_no_secret_reaches_the_archive(tmp_path: Path) -> None:
    """A backup is a file that gets emailed. It must be safe to email.

    Every byte of the archive is searched, not just the manifest: a secret
    that leaked through a settings dump or a database row would be just as
    gone, and would be found here rather than by whoever received the file.
    """
    settings = _settings(
        tmp_path / "here",
        admin_token="tok-do-not-leak",
        openai_api_key="sk-do-not-leak",
        oidc_client_secret="oidc-do-not-leak",
    )
    made = write_backup(settings, tmp_path / "ok.zip")

    with zipfile.ZipFile(tmp_path / "ok.zip") as zf:
        every_byte = b"".join(zf.read(name) for name in zf.namelist())
    for secret in (b"tok-do-not-leak", b"sk-do-not-leak", b"oidc-do-not-leak"):
        assert secret not in every_byte, secret

    # And the ones that were set are named, so a restore can say what to type
    # back in rather than leaving somebody to find out when answers stop.
    assert set(made.secrets_omitted) == {
        "OK_ADMIN_TOKEN",
        "OK_OPENAI_API_KEY",
        "OK_OIDC_CLIENT_SECRET",
    }


def test_a_secret_that_was_never_set_is_not_advertised(tmp_path: Path) -> None:
    """ "Set these again" is a list of work, so it holds only real work."""
    made = write_backup(_settings(tmp_path / "here"), tmp_path / "ok.zip")
    assert made.secrets_omitted == ()


def test_restore_refuses_to_overwrite_by_accident(tmp_path: Path) -> None:
    """The one irreversible thing here, and it asks first."""
    here = _settings(tmp_path / "here")
    with AnswerStore(here.db_path) as store:
        store.pin("q", "the answer that is already here", author="hr")
    write_backup(here, tmp_path / "ok.zip")

    with AnswerStore(here.db_path) as store:
        store.pin("q", "changed since the backup", author="hr")

    with pytest.raises(BackupError, match="already holds"):
        restore_backup(tmp_path / "ok.zip", here)
    with AnswerStore(here.db_path) as store:
        assert store.get_pin("q").answer == "changed since the backup", "refused means untouched"

    restore_backup(tmp_path / "ok.zip", here, force=True)
    with AnswerStore(here.db_path) as store:
        assert store.get_pin("q").answer == "the answer that is already here"


def test_a_backup_taken_while_the_database_is_being_written(tmp_path: Path) -> None:
    """A server does not stop being asked questions during a backup.

    Copying a SQLite file mid-write produces something that looks like a
    database and is not, so this goes through SQLite's own backup API. The
    test writes continuously while the archive is taken and then opens what
    came out.
    """
    here = _settings(tmp_path / "here")
    with AnswerStore(here.db_path) as store:
        store.pin("q0", "before", author="hr")

        stop = threading.Event()

        def keep_writing() -> None:
            n = 0
            while not stop.is_set():
                store.pin(f"q{n}", f"answer {n}", author="hr")
                n += 1

        writer = threading.Thread(target=keep_writing, daemon=True)
        writer.start()
        try:
            write_backup(here, tmp_path / "ok.zip")
        finally:
            stop.set()
            writer.join(timeout=5)

    there = _settings(tmp_path / "there")
    restore_backup(tmp_path / "ok.zip", there)
    with AnswerStore(there.db_path) as store:
        assert store.get_pin("q0") is not None, "a consistent snapshot, not a torn file"
        # And it is a real database, not a file that merely opens.
        assert store._conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"  # noqa: SLF001


def test_the_vector_index_is_left_out_and_says_so(tmp_path: Path) -> None:
    """Derived from the documents, so carrying it doubles the file for nothing."""
    here = _settings(tmp_path / "here")
    (Path(here.data_dir) / "vectors.db").write_bytes(b"x" * 4096)
    write_backup(here, tmp_path / "ok.zip")

    with zipfile.ZipFile(tmp_path / "ok.zip") as zf:
        assert not [n for n in zf.namelist() if "vectors" in n]
    assert "vectors.db" in read_manifest(tmp_path / "ok.zip")["omitted"]


def test_an_archive_from_a_newer_build_is_refused(tmp_path: Path) -> None:
    """Guessing at a layout this build has never seen is how a restore
    silently drops half of what somebody trusted it with."""
    archive = tmp_path / "future.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("manifest.json", json.dumps({"format": FORMAT + 1}))

    with pytest.raises(BackupError, match="format"):
        restore_backup(archive, _settings(tmp_path / "there"))


def test_something_that_is_not_a_backup_is_refused(tmp_path: Path) -> None:
    not_ours = tmp_path / "holiday-photos.zip"
    with zipfile.ZipFile(not_ours, "w") as zf:
        zf.writestr("beach.jpg", "not a database")

    with pytest.raises(BackupError, match="not an OpenKnowledge backup"):
        restore_backup(not_ours, _settings(tmp_path / "there"))


def test_an_archive_that_lies_about_its_contents_changes_nothing(tmp_path: Path) -> None:
    """Checked before anything moves, so a truncated file leaves the install
    exactly as it was rather than half replaced."""
    there = _settings(tmp_path / "there")
    with AnswerStore(there.db_path) as store:
        store.pin("q", "still here afterwards", author="hr")

    archive = tmp_path / "truncated.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr(
            "manifest.json",
            json.dumps({"format": FORMAT, "databases": {"openknowledge.db": "promised"}}),
        )

    with pytest.raises(BackupError, match="and does not"):
        restore_backup(archive, there, force=True)
    with AnswerStore(there.db_path) as store:
        assert store.get_pin("q").answer == "still here afterwards"


def test_documents_can_be_left_out_for_a_decisions_only_backup(tmp_path: Path) -> None:
    """Somebody whose documents already live in SharePoint wants the pins."""
    here = _settings(tmp_path / "here")
    (Path(here.documents_dir) / "big.md").write_text("# Big\n\nlots of words\n")
    with AnswerStore(here.db_path) as store:
        store.pin("q", "a decision worth keeping", author="hr")

    made = write_backup(here, tmp_path / "ok.zip", include_documents=False)
    assert made.documents == 0

    there = _settings(tmp_path / "there")
    done = restore_backup(tmp_path / "ok.zip", there)
    assert done.documents == 0
    with AnswerStore(there.db_path) as store:
        assert store.get_pin("q") is not None


def test_nested_document_folders_keep_their_shape(tmp_path: Path) -> None:
    """Folders are how access rules are written, so the tree is load-bearing."""
    here = _settings(tmp_path / "here")
    (Path(here.documents_dir) / "finance" / "2026").mkdir(parents=True)
    (Path(here.documents_dir) / "finance" / "2026" / "limits.md").write_text("# Limits\n")

    write_backup(here, tmp_path / "ok.zip")
    there = _settings(tmp_path / "there")
    restore_backup(tmp_path / "ok.zip", there)

    assert (Path(there.documents_dir) / "finance" / "2026" / "limits.md").is_file()


def test_a_database_that_is_not_there_is_not_claimed(tmp_path: Path) -> None:
    """A fresh install has no auth.db, and the manifest should not pretend."""
    made = write_backup(_settings(tmp_path / "here"), tmp_path / "ok.zip")
    assert "auth.db" not in made.databases
    assert "auth.db" not in read_manifest(tmp_path / "ok.zip")["databases"]


def test_the_backup_is_not_left_behind_as_rubble(tmp_path: Path) -> None:
    """The staging directory is cleaned up whether or not it worked."""
    here = _settings(tmp_path / "here")
    write_backup(here, tmp_path / "ok.zip")
    assert not [p for p in tmp_path.iterdir() if p.name.startswith(".")]


def test_a_restored_database_is_usable_not_just_present(tmp_path: Path) -> None:
    """Present and openable are different claims; this makes the second."""
    here = _settings(tmp_path / "here")
    with AnswerStore(here.db_path) as store:
        for i in range(50):
            store.pin(f"question {i}", f"answer {i}", author="hr")
    write_backup(here, tmp_path / "ok.zip")

    there = _settings(tmp_path / "there")
    restore_backup(tmp_path / "ok.zip", there)
    with sqlite3.connect(there.db_path) as conn:
        assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert conn.execute("SELECT COUNT(*) FROM pinned_answers").fetchone()[0] == 50


@pytest.mark.parametrize(
    "databases",
    ["not a mapping", 42, None, ["openknowledge.db"], {"openknowledge.db": "fine"}],
)
def test_a_manifest_of_any_shape_is_handled_not_trusted(tmp_path: Path, databases) -> None:
    """The manifest is a file somebody handed us, so its shape is a claim.

    Anything that is not a collection of names reads as no names, which
    surfaces as a refusal or an empty restore - both true, neither a
    traceback in front of somebody trying to get their install back.
    """
    archive = tmp_path / "odd.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("manifest.json", json.dumps({"format": FORMAT, "databases": databases}))

    there = _settings(tmp_path / "there")
    try:
        restore_backup(archive, there)
    except BackupError as exc:
        assert "and does not" in str(exc)


def test_the_admin_log_travels_and_says_it_was_restored(tmp_path: Path) -> None:
    """Who changed what is part of what an install would hate to lose - and
    the restore itself is the largest change anyone can make in one command,
    so it writes itself into the log it just replaced."""
    from openknowledge.knowledge.store import Actor

    here, there = _settings(tmp_path / "here"), _settings(tmp_path / "there")
    with KnowledgeStore(here.knowledge_db_path) as knowledge:
        knowledge.record_action(
            Actor(id="alice-oid", name="Alice Moreau", kind="person"), "access.set", "hr"
        )

    archive = tmp_path / "carry.zip"
    write_backup(here, archive)
    restore_backup(archive, there)

    with KnowledgeStore(there.knowledge_db_path) as knowledge:
        entries = knowledge.admin_actions()

    assert [(e.actor.name, e.action) for e in entries] == [
        ("the server console", "restore"),
        ("Alice Moreau", "access.set"),
    ]
    assert entries[0].target == "carry.zip"
