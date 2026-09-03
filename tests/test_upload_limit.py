"""How much one person may add per minute, and why it is counted in bytes.

The security review left this open: `/chat` has a per-asker limit and
`/documents` had none, so anybody who may upload could add files as fast as
the network allowed. `upload_max_mb` bounds one file; nothing bounded the
rate, and forty files in one request is one request.

So the count is bytes, not requests. A limit on requests would be a limit on
nothing here.

Two things it deliberately does not claim. It bounds what is **kept**, not
what is received: by the time the endpoint runs, Starlette has already read
and spooled the body, which no handler can undo. And it bounds *speed*, not
*total* - a hundred megabytes a minute still fills a disk given a day, so
this is a limit on one caller's share, not a disk-space guarantee.
"""

from __future__ import annotations

import io
from collections import deque
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from openknowledge.api.app import create_app
from openknowledge.api.runtime_settings import SettingsChangeError, validate_changes
from openknowledge.config import Settings
from openknowledge.limits import AskerLimiter, _when_there_is_room

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
    """A document of a given size that the parser can actually read."""
    body = ("# Policy\n\n" + "meals are reimbursed. " * (size // 22)).encode("utf-8")
    return ("files", (name, io.BytesIO(body[:size].ljust(size, b" ")), "text/markdown"))


# -- the endpoint --------------------------------------------------------------


def test_the_second_file_over_the_minute_is_refused_and_says_when(tmp_path: Path) -> None:
    """The first is stored, the second is not, and the reason is actionable."""
    settings = _settings(tmp_path, upload_max_mb=2, upload_mb_per_minute=3)
    with TestClient(create_app(settings)) as client:
        response = client.post(
            "/documents", files=[_file("first.md", 2 * MEGABYTE), _file("second.md", 2 * MEGABYTE)]
        )
    assert response.status_code == 201
    body = response.json()
    assert [entry["name"] for entry in body["stored"]] == ["first.md"]
    assert [entry["name"] for entry in body["skipped"]] == ["second.md"]
    reason = body["skipped"][0]["reason"]
    assert "3 MB a minute" in reason
    assert "Try again in" in reason
    assert (tmp_path / "documents" / "second.md").exists() is False


def test_one_request_cannot_carry_more_than_a_minute(tmp_path: Path) -> None:
    """Counting requests would have let this through as a single request."""
    settings = _settings(tmp_path, upload_max_mb=1, upload_mb_per_minute=2)
    with TestClient(create_app(settings)) as client:
        response = client.post(
            "/documents", files=[_file(f"note-{n}.md", MEGABYTE) for n in range(5)]
        )
    body = response.json()
    assert len(body["stored"]) == 2, "the allowance is two megabytes, so two files fit"
    assert len(body["skipped"]) == 3


def test_no_limit_by_default(tmp_path: Path) -> None:
    """The desktop install, where the only uploader is whoever owns the laptop."""
    with TestClient(create_app(_settings(tmp_path))) as client:
        response = client.post(
            "/documents", files=[_file(f"note-{n}.md", MEGABYTE) for n in range(4)]
        )
    assert len(response.json()["stored"]) == 4
    assert response.json()["skipped"] == []


# -- the counting ---------------------------------------------------------------


def test_uploaders_are_counted_apart() -> None:
    """One person's flood must not close the door on everybody else."""
    limiter = AskerLimiter(3 * MEGABYTE)
    assert limiter.check("user:ada", cost=3 * MEGABYTE).allowed
    assert not limiter.check("user:ada", cost=MEGABYTE).allowed
    assert limiter.check("user:grace", cost=3 * MEGABYTE).allowed, (
        "Grace was charged for Ada's upload"
    )


def test_the_wait_is_for_the_room_actually_needed() -> None:
    """A caller who spent the minute on one big file waits for that file.

    The old answer was always "when the oldest leaves", which is right only
    when every charge is the same size. Told to wait a second for room that
    will not exist for fifty, a client retries fifty times.
    """
    window = deque([(10.0, 1), (20.0, 1), (30.0, 8)])
    # Spent 10 of 10; needs 8 more. Dropping the two small ones frees 2, which
    # is not enough - it has to wait for the big one at 30.
    assert _when_there_is_room(window, spent=10, cost=8, ceiling=10, cutoff=0.0) == 30.0
    # Needing 1 more, the first small charge leaving is enough.
    assert _when_there_is_room(window, spent=10, cost=1, ceiling=10, cutoff=0.0) == 10.0


def test_a_charge_bigger_than_the_whole_allowance_does_not_promise_a_retry() -> None:
    """Nothing the window can free would fit it, so no wait is honest."""
    window = deque([(10.0, 1)])
    assert _when_there_is_room(window, spent=1, cost=99, ceiling=10, cutoff=0.0) == 0.0


# -- the pair that cannot both be true -------------------------------------------


def test_a_minute_smaller_than_a_file_is_refused_at_startup(tmp_path: Path) -> None:
    """Otherwise the server accepts a file it can never store.

    A 25 MB file against a 10 MB minute exceeds the whole allowance, so no
    amount of waiting makes room and the refusal repeats forever. Better to
    refuse the configuration, where somebody can act on it.
    """
    with pytest.raises(ValueError, match="below upload_max_mb"):
        _settings(tmp_path, upload_max_mb=25, upload_mb_per_minute=10)


def test_the_same_pair_is_refused_on_the_settings_page(tmp_path: Path) -> None:
    """Half a pair arrives alone; the pair is what has to hold."""
    live = _settings(tmp_path, upload_max_mb=25, upload_mb_per_minute=50)
    with pytest.raises(SettingsChangeError, match="below upload_max_mb"):
        validate_changes({"upload_mb_per_minute": 10}, live)
    assert validate_changes({"upload_mb_per_minute": 25}, live) == {"upload_mb_per_minute": 25}
