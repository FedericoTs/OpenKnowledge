"""Leave the machine somewhere to write.

The upload rate limit bounds one caller's *share*; it does not stop a disk
filling, because a hundred megabytes a minute still fills one given a day.
This is the other half, and it is a different kind of thing: not a policy
about people but the space the server needs to keep working after the last
file fits. A disk at zero cannot commit a SQLite transaction, cannot write
the index, and cannot append to the log the operator would read to find out
why - so the first symptom is an answer engine that stopped answering for a
reason nothing on the page explains.

It is on by default, unlike the rate limits, for that reason. And it is a
floor under free space, not a ceiling on the corpus: it holds whatever else
is filling the disk, and it answers "will the machine still work", not "how
big may this corpus grow".
"""

from __future__ import annotations

import io
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from openknowledge import disk
from openknowledge.api.app import create_app
from openknowledge.config import Settings
from openknowledge.desktop.download import DownloadError, ensure_model
from openknowledge.desktop.manifest import ModelFile

MEGABYTE = 1_000_000


def _settings(tmp_path: Path, **changes: object) -> Settings:
    return Settings(
        data_dir=str(tmp_path / "data"),
        documents_dir=str(tmp_path / "documents"),
        upload_enabled=True,
        local_enabled=False,
        embedding_enabled=False,
        _env_file=None,  # type: ignore[call-arg]
        **changes,
    )


def _file(name: str, size: int):
    body = ("# Policy\n\n" + "meals are reimbursed. " * (size // 22)).encode("utf-8")
    return ("files", (name, io.BytesIO(body[:size].ljust(size, b" ")), "text/markdown"))


def _pretend_free(monkeypatch: pytest.MonkeyPatch, bytes_free: int) -> None:
    monkeypatch.setattr(disk, "free_bytes", lambda _path: bytes_free)


# -- the rule itself -----------------------------------------------------------


def test_the_floor_is_free_space_after_the_write(monkeypatch: pytest.MonkeyPatch) -> None:
    """ "40 free and this is 30" is not a reason to proceed.

    The question is what is left afterwards, which is the whole point: the
    file that fits is exactly the one that leaves nothing for the database.
    """
    _pretend_free(monkeypatch, 40 * MEGABYTE)
    assert disk.no_room_for("/anywhere", 30 * MEGABYTE, floor_mb=20) is not None
    assert disk.no_room_for("/anywhere", 10 * MEGABYTE, floor_mb=20) is None


def test_zero_turns_it_off(monkeypatch: pytest.MonkeyPatch) -> None:
    """The escape hatch, for a deployment that means to run to the edge."""
    _pretend_free(monkeypatch, 1)
    assert disk.no_room_for("/anywhere", 10 * MEGABYTE, floor_mb=0) is None


def test_free_space_answers_for_a_directory_not_yet_created(tmp_path: Path) -> None:
    """Callers ask about somewhere they are about to make."""
    assert disk.free_bytes(tmp_path / "not" / "yet" / "here") > 0


def test_the_refusal_names_the_numbers_and_the_lever(monkeypatch: pytest.MonkeyPatch) -> None:
    """A refusal nobody can act on is an outage with better manners."""
    _pretend_free(monkeypatch, 100 * MEGABYTE)
    said = disk.no_room_for("/anywhere", 60 * MEGABYTE, floor_mb=50)
    assert said is not None
    assert "60.0 MB" in said and "100.0 MB free" in said and "50 MB" in said
    assert "OK_DISK_FLOOR_MB" in said


# -- uploads --------------------------------------------------------------------


def test_an_upload_that_would_breach_the_floor_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _pretend_free(monkeypatch, 3 * MEGABYTE)
    settings = _settings(tmp_path, upload_max_mb=5, disk_floor_mb=2)
    with TestClient(create_app(settings)) as client:
        response = client.post("/documents", files=[_file("policy.md", 2 * MEGABYTE)])
    body = response.json()
    assert body["stored"] == []
    assert "keeps spare" in body["skipped"][0]["reason"]
    assert not (tmp_path / "documents" / "policy.md").exists(), "it was written anyway"


def test_uploads_still_work_with_room_to_spare(tmp_path: Path) -> None:
    """The control: the check must not refuse an ordinary upload."""
    with TestClient(create_app(_settings(tmp_path))) as client:
        response = client.post("/documents", files=[_file("policy.md", 1000)])
    assert [entry["name"] for entry in response.json()["stored"]] == ["policy.md"]


def test_metrics_report_the_free_space_and_the_floor(tmp_path: Path) -> None:
    """The number that explains a server which has started refusing."""
    settings = _settings(tmp_path, admin_token="t0ken", disk_floor_mb=500)
    with TestClient(create_app(settings)) as client:
        body = client.get("/metrics", headers={"Authorization": "Bearer t0ken"}).text
    assert "openknowledge_disk_free_bytes" in body
    assert "openknowledge_disk_floor_bytes 500000000" in body


# -- the 2.6 GB one -------------------------------------------------------------


def test_the_model_download_is_refused_before_the_first_byte(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A download that fills the disk and then fails took the space with it.

    The URL here is unreachable on purpose: if the pre-flight did not run,
    this would fail with a network error instead, and the assertion on the
    message would say so.
    """
    _pretend_free(monkeypatch, 100 * MEGABYTE)
    model = ModelFile(
        filename="big.gguf",
        url="https://127.0.0.1:1/big.gguf",
        sha256="0" * 64,
        size_bytes=2_600 * MEGABYTE,
        purpose="chat",
        license="Apache-2.0",
        context_tokens=4096,
    )
    with pytest.raises(DownloadError, match="keeps spare"):
        ensure_model(model, tmp_path / "models", floor_mb=500)
    assert not (tmp_path / "models" / "big.gguf.part").exists()
