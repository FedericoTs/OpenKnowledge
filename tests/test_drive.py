"""The Drive mirror, against a Drive that lives on loopback.

The mapping tests are the ones that matter most: Drive names people by email
and a directory names them by id, so the vocabulary this produces has to be
the one sign-in mints - and a grant it cannot express has to be dropped
rather than widened.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from pathlib import Path

import pytest
from tests.fake_drive import (
    CLIENT_EMAIL,
    DOC_MIME,
    DOMAIN,
    SUBJECT,
    FakeDrive,
    anyone_grant,
    deleted_group_grant,
    domain_grant,
    group_grant,
    private_key_pem,
    user_grant,
)

from openknowledge.connectors.drive import (
    DriveClient,
    DriveConfig,
    DriveSync,
    principals_from,
)
from openknowledge.connectors.mirror import WITHHELD, SyncStore

ALICE = "alice@contoso.com"
HR_GROUP = "hr-team@contoso.com"


# -- mapping -----------------------------------------------------------------------


def test_people_and_groups_map_to_the_vocabulary_sign_in_mints() -> None:
    principals, unmapped = principals_from(
        [user_grant(ALICE), group_grant(HR_GROUP), domain_grant(DOMAIN)], domain=DOMAIN
    )
    assert principals == frozenset({f"user:{ALICE}", f"group:{HR_GROUP}", "authenticated"})
    assert unmapped == 0


def test_an_address_is_matched_however_it_was_typed() -> None:
    principals, _ = principals_from([user_grant("Alice@Contoso.com ")], domain=DOMAIN)
    assert principals == frozenset({f"user:{ALICE}"}), "addresses are not case-sensitive"


def test_another_companys_domain_is_not_this_companys_people() -> None:
    """A file shared with a partner's whole domain grants them nothing here:
    they are not the people this install signs in."""
    principals, unmapped = principals_from(
        [group_grant(HR_GROUP), domain_grant("partner.example")], domain=DOMAIN
    )
    assert principals == frozenset({f"group:{HR_GROUP}"})
    assert unmapped == 1


def test_a_file_with_no_mappable_reader_is_withheld_not_public() -> None:
    assert principals_from([], domain=DOMAIN) == (frozenset({WITHHELD}), 0)
    assert principals_from([domain_grant("partner.example")], domain=DOMAIN) == (
        frozenset({WITHHELD}),
        1,
    )
    assert principals_from([deleted_group_grant(HR_GROUP)], domain=DOMAIN) == (
        frozenset({WITHHELD}),
        0,
    ), "a grant to a deleted group grants nobody anything"


def test_a_public_link_is_everyone_who_can_reach_this_server() -> None:
    principals, unmapped = principals_from([anyone_grant()], domain=DOMAIN)
    assert principals == frozenset({"authenticated"}) and unmapped == 0


def test_every_role_grants_reading() -> None:
    for role in ("reader", "commenter", "writer", "fileOrganizer", "owner"):
        principals, _ = principals_from([user_grant(ALICE, role)], domain=DOMAIN)
        assert principals == frozenset({f"user:{ALICE}"}), role


# -- the sync -------------------------------------------------------------------------


@pytest.fixture
def drive() -> Iterator[FakeDrive]:
    d = FakeDrive()
    d.add_drive("drive-1", "Company")
    d.add_folder("drive-1", "folder-hr", "HR", "drive-1")
    d.add_file(
        "drive-1",
        "f-leave",
        "parental-leave.md",
        b"# Parental Leave\nTwenty weeks.",
        [group_grant(HR_GROUP)],
        parent="folder-hr",
    )
    d.add_file(
        "drive-1",
        "f-expenses",
        "expenses.md",
        b"# Expenses\nEUR 500.",
        [user_grant(ALICE), domain_grant(DOMAIN)],
    )
    d.add_file(
        "drive-1",
        "f-board",
        "board-minutes.md",
        b"# Minutes\nSecret.",
        [domain_grant("partner.example")],
    )
    d.add_file("drive-1", "f-logo", "logo.png", b"\x89PNG", [anyone_grant()])
    try:
        yield d
    finally:
        d.close()


class _Clock:
    """A clock the test moves, starting from now.

    Wall time rather than a small number on purpose: the service account
    signs an assertion whose iat and exp come from this clock, and Google
    checks them against the real one. A clock that starts at zero mints
    tokens that expired in 1970 - which is exactly what it did, and what
    this comment exists to stop happening again.
    """

    def __init__(self, now: float | None = None) -> None:
        self.now = time.time() if now is None else now

    def __call__(self) -> float:
        return self.now


def _sync(drive: FakeDrive, tmp_path: Path, clock: _Clock) -> DriveSync:
    config = DriveConfig(
        client_email=CLIENT_EMAIL,
        private_key=private_key_pem(),
        subject=SUBJECT,
        domain=DOMAIN,
        api_url=f"{drive.base}/v3",
        token_url=f"{drive.base}/token",
    )
    slept: list[float] = []
    client = DriveClient(config, clock=clock, sleep=slept.append)
    client.slept = slept  # type: ignore[attr-defined]
    return DriveSync(
        client,
        documents_dir=tmp_path / "documents",
        store=SyncStore(tmp_path / "drive.db"),
        permissions_refresh_seconds=3600,
        clock=clock,
    )


def _mirrored(tmp_path: Path) -> set[str]:
    root = tmp_path / "documents"
    return {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}


def test_the_first_sync_mirrors_the_drive_with_its_readers(
    drive: FakeDrive, tmp_path: Path
) -> None:
    sync = _sync(drive, tmp_path, _Clock())
    summary = sync.run()

    assert summary.errors == []
    assert (summary.drives, summary.added, summary.skipped) == (1, 3, 1), summary.as_dict()
    assert _mirrored(tmp_path) == {
        "gdrive/Company/HR/parental-leave.md",
        "gdrive/Company/expenses.md",
        "gdrive/Company/board-minutes.md",
    }, "folders are followed, and the png is not a document"

    readers = sync.principals_map()
    assert readers["gdrive/Company/HR/parental-leave.md"] == {f"group:{HR_GROUP}"}
    assert readers["gdrive/Company/expenses.md"] == {f"user:{ALICE}", "authenticated"}
    assert readers["gdrive/Company/board-minutes.md"] == {WITHHELD}
    assert (summary.documents, summary.withheld, summary.unmapped_grants) == (3, 1, 1)
    assert drive.token_calls == 1


def test_the_second_sync_asks_for_changes_only(drive: FakeDrive, tmp_path: Path) -> None:
    clock = _Clock()
    sync = _sync(drive, tmp_path, clock)
    sync.run()
    drive.requests.clear()

    clock.now += 60
    summary = sync.run()
    assert summary.errors == []
    assert (summary.added, summary.updated, summary.removed) == (0, 0, 0)
    assert any("/v3/changes?" in r for r in drive.requests), "the saved token, not a full walk"
    assert not any("/v3/files?" in r for r in drive.requests), "no walk"
    assert not any("alt=media" in r for r in drive.requests), "nothing changed, nothing downloaded"


def test_a_change_a_move_and_a_trashing_arrive_as_exactly_those(
    drive: FakeDrive, tmp_path: Path
) -> None:
    clock = _Clock()
    sync = _sync(drive, tmp_path, clock)
    sync.run()
    drive.change_content("drive-1", "f-expenses", b"# Expenses\nEUR 1,000.")
    drive.set_permissions("drive-1", "f-expenses", [user_grant(ALICE)])
    drive.rename("drive-1", "f-leave", "leave.md", parent="drive-1")
    drive.trash("drive-1", "f-board")
    drive.requests.clear()

    clock.now += 60
    summary = sync.run()
    assert summary.errors == []
    assert (summary.added, summary.updated, summary.removed) == (0, 2, 1), summary.as_dict()
    assert _mirrored(tmp_path) == {"gdrive/Company/leave.md", "gdrive/Company/expenses.md"}
    expenses = tmp_path / "documents/gdrive/Company/expenses.md"
    assert expenses.read_bytes() == b"# Expenses\nEUR 1,000."
    assert not (tmp_path / "documents/gdrive/Company/HR").exists(), "emptied folders go"
    readers = sync.principals_map()
    assert readers["gdrive/Company/expenses.md"] == {f"user:{ALICE}"}, (
        "a changed file has its readers re-read, so the domain grant is gone"
    )
    assert summary.withheld == 0


def test_readers_are_re_read_on_a_clock_even_when_nothing_changed(
    drive: FakeDrive, tmp_path: Path
) -> None:
    clock = _Clock()
    sync = _sync(drive, tmp_path, clock)
    sync.run()
    drive.set_permissions("drive-1", "f-leave", [group_grant("everyone@contoso.com")])
    drive.requests.clear()

    clock.now += 600
    sync.run()
    assert not any("/permissions" in r for r in drive.requests), "ten minutes is inside the hour"

    clock.now += 3600
    summary = sync.run()
    assert summary.permissions_read == 3, "past the hour every file's readers are asked again"
    assert sync.principals_map()["gdrive/Company/HR/parental-leave.md"] == {
        "group:everyone@contoso.com"
    }


def test_a_google_document_is_exported_as_something_the_parsers_read(
    drive: FakeDrive, tmp_path: Path
) -> None:
    """A Doc has no bytes to download. Exported as .docx it keeps headings and
    tables, which is what gives an answer a real locator to cite."""
    drive.add_file(
        "drive-1",
        "f-handbook",
        "Handbook",
        b"PK-a-docx",
        [domain_grant(DOMAIN)],
        mime_type=DOC_MIME,
    )
    sync = _sync(drive, tmp_path, _Clock())
    sync.run()
    assert "gdrive/Company/Handbook.docx" in _mirrored(tmp_path)
    assert any("/export?" in r for r in drive.requests)
    assert sync.principals_map()["gdrive/Company/Handbook.docx"] == {"authenticated"}


def test_a_form_or_a_drawing_is_skipped_rather_than_mirrored_empty(
    drive: FakeDrive, tmp_path: Path
) -> None:
    """Google holds a Form, a Drawing and a shortcut natively and there is no
    document in any of them to export.

    Both names matter. "Survey" has no extension, so the supported-types check
    would turn it away anyway. "Notes.md" would sail straight through that
    check and be mirrored as whatever bytes Drive returns for a thing that is
    not a file - which is why the guard is on the mime type, not the name.
    """
    for file_id, name in (("f-form", "Survey"), ("f-draw", "Notes.md")):
        drive.add_file(
            "drive-1",
            file_id,
            name,
            b"",
            [domain_grant(DOMAIN)],
            mime_type="application/vnd.google-apps.form",
        )
    sync = _sync(drive, tmp_path, _Clock())
    summary = sync.run()
    mirrored = _mirrored(tmp_path)
    assert not any("Survey" in name for name in mirrored)
    assert not any("Notes.md" in name for name in mirrored), (
        "a Form named like a document is still a Form"
    )
    assert summary.skipped == 3, "two Google-native files and the png"


def test_throttling_is_waited_out_and_an_expired_token_is_refreshed(
    drive: FakeDrive, tmp_path: Path
) -> None:
    sync = _sync(drive, tmp_path, _Clock())
    drive.throttle_once = 3
    drive.expire_token_once = True
    summary = sync.run()
    assert summary.errors == [] and summary.added == 3
    assert sync.drive.slept == [3.0]  # type: ignore[attr-defined]
    assert drive.token_calls == 2, "the 401 bought a new token, not a retry of the old one"


def test_a_page_token_drive_has_forgotten_re_reads_the_drive(
    drive: FakeDrive, tmp_path: Path
) -> None:
    clock = _Clock()
    sync = _sync(drive, tmp_path, clock)
    sync.run()
    drive.stale_page_token_once = True
    drive.requests.clear()
    clock.now += 60
    summary = sync.run()
    assert summary.errors == []
    assert any("/v3/files?" in r for r in drive.requests), "410, then a walk from the start"
    assert _mirrored(tmp_path) == {
        "gdrive/Company/HR/parental-leave.md",
        "gdrive/Company/expenses.md",
        "gdrive/Company/board-minutes.md",
    }


def test_a_wrong_key_is_a_wrong_key(drive: FakeDrive, tmp_path: Path) -> None:
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import rsa

    other = (
        rsa.generate_private_key(public_exponent=65537, key_size=2048)
        .private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
        .decode()
    )
    config = DriveConfig(
        client_email=CLIENT_EMAIL,
        private_key=other,
        domain=DOMAIN,
        api_url=f"{drive.base}/v3",
        token_url=f"{drive.base}/token",
    )
    sync = DriveSync(DriveClient(config), documents_dir=tmp_path / "documents", store=SyncStore())
    summary = sync.run()
    assert len(summary.errors) == 1 and "token request failed" in summary.errors[0]
    assert summary.documents == 0


def test_a_refusal_runs_nothing_and_says_why(drive: FakeDrive, tmp_path: Path) -> None:
    sync = _sync(drive, tmp_path, _Clock())
    sync.refusal = "sign-in is off, so no reader can be enforced"
    summary = sync.run()
    assert summary.errors == [sync.refusal]
    assert drive.requests == [] and _mirrored(tmp_path) == set()
    assert sync.status()["refusal"] == sync.refusal
