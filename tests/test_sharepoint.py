"""The SharePoint mirror, against a Graph that lives on loopback.

Every claim the connector makes is checked against the fake's request log
or its file system effects, never against the connector's own summary alone:
"changes only" is the token URL being asked and two files not being fetched,
"fail closed" is a document a restricted viewer cannot see, and so on.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from tests.fake_graph import (
    CLIENT_ID,
    CLIENT_SECRET,
    TENANT,
    FakeGraph,
    application_grant,
    group_grant,
    link_grant,
    site_group_grant,
    user_grant,
)

from openknowledge.connectors.sharepoint import (
    WITHHELD,
    GraphClient,
    GraphConfig,
    SharePointSync,
    SyncStore,
    principals_from,
)

HR = "11111111-1111-1111-1111-111111111111"
FINANCE = "22222222-2222-2222-2222-222222222222"
ALICE = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"


# -- mapping --------------------------------------------------------------------------


def test_users_groups_and_organisation_links_map_onto_the_session_vocabulary() -> None:
    principals, unmapped = principals_from(
        [user_grant(ALICE), group_grant(HR, inherited_from="folder-1"), link_grant("organization")]
    )
    assert principals == frozenset({f"user:{ALICE}", f"group:{HR}", "authenticated"})
    assert unmapped == 0


def test_a_grant_that_cannot_be_named_is_dropped_never_widened() -> None:
    principals, unmapped = principals_from([group_grant(HR), site_group_grant("HR Visitors")])
    assert principals == frozenset({f"group:{HR}"}), "the site group's members are not added"
    assert unmapped == 1


def test_a_file_with_no_mappable_reader_is_withheld_not_public() -> None:
    assert principals_from([site_group_grant("HR Owners")]) == (frozenset({WITHHELD}), 1)
    assert principals_from([]) == (frozenset({WITHHELD}), 0), "no grants is not everyone"
    assert principals_from([application_grant("app-id")]) == (frozenset({WITHHELD}), 0), (
        "this connector's own grant is not a reader"
    )


def test_a_link_to_named_people_names_them() -> None:
    principals, unmapped = principals_from([link_grant("users", users=(ALICE,))])
    assert principals == frozenset({f"user:{ALICE}"}) and unmapped == 0


# -- the sync -------------------------------------------------------------------------


@pytest.fixture
def graph() -> Iterator[FakeGraph]:
    g = FakeGraph()
    g.add_drive("drive-1", "Documents")
    g.add_file(
        "drive-1",
        "i-leave",
        "HR/Policies/parental-leave.md",
        b"# Parental Leave\nTwenty weeks.",
        [group_grant(HR)],
    )
    g.add_file(
        "drive-1",
        "i-expenses",
        "Finance/expenses.md",
        b"# Expenses\nEUR 500.",
        [group_grant(FINANCE), user_grant(ALICE)],
    )
    g.add_file(
        "drive-1",
        "i-secret",
        "HR/board-minutes.md",
        b"# Minutes\nSecret.",
        [site_group_grant("Owners")],
    )
    g.add_file("drive-1", "i-logo", "HR/logo.png", b"\x89PNG", [group_grant(HR)])
    try:
        yield g
    finally:
        g.close()


class _Clock:
    def __init__(self, now: float = 1_000_000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


def _sync(graph: FakeGraph, tmp_path: Path, clock: _Clock) -> SharePointSync:
    config = GraphConfig(
        tenant_id=TENANT,
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
        site="contoso.sharepoint.com:/sites/HR",
        graph_url=f"{graph.base}/v1.0",
        login_url=graph.base,
    )
    slept: list[float] = []
    client = GraphClient(config, clock=clock, sleep=slept.append)
    client.slept = slept  # type: ignore[attr-defined]
    return SharePointSync(
        client,
        documents_dir=tmp_path / "documents",
        store=SyncStore(tmp_path / "sharepoint.db"),
        permissions_refresh_seconds=3600,
        clock=clock,
    )


def _mirrored(tmp_path: Path) -> set[str]:
    root = tmp_path / "documents"
    return {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}


def test_the_first_sync_mirrors_the_library_with_its_readers(
    graph: FakeGraph, tmp_path: Path
) -> None:
    sync = _sync(graph, tmp_path, _Clock())
    summary = sync.run()

    assert summary.errors == []
    assert (summary.drives, summary.added, summary.skipped) == (1, 3, 1), summary.as_dict()
    assert _mirrored(tmp_path) == {
        "sharepoint/Documents/HR/Policies/parental-leave.md",
        "sharepoint/Documents/Finance/expenses.md",
        "sharepoint/Documents/HR/board-minutes.md",
    }, "supported files only, under the library's own folder tree"
    expenses = tmp_path / "documents/sharepoint/Documents/Finance/expenses.md"
    assert expenses.read_bytes() == b"# Expenses\nEUR 500."

    readers = sync.principals_map()
    assert readers["sharepoint/Documents/HR/Policies/parental-leave.md"] == {f"group:{HR}"}
    assert readers["sharepoint/Documents/Finance/expenses.md"] == {
        f"group:{FINANCE}",
        f"user:{ALICE}",
    }
    assert readers["sharepoint/Documents/HR/board-minutes.md"] == {WITHHELD}
    assert (summary.documents, summary.withheld, summary.unmapped_grants) == (3, 1, 1)

    asked = [r for r in graph.requests if r.startswith("GET")]
    assert sum("/root/delta" in r for r in asked) == 2, "four files, two pages"
    assert sum("/content" in r for r in asked) == 3, "downloaded once each, the png never"
    assert sum("/permissions" in r for r in asked) == 3
    assert graph.token_calls == 1


def test_the_second_sync_asks_for_changes_only(graph: FakeGraph, tmp_path: Path) -> None:
    clock = _Clock()
    sync = _sync(graph, tmp_path, clock)
    sync.run()
    graph.requests.clear()

    clock.now += 60
    summary = sync.run()
    assert summary.errors == []
    assert (summary.added, summary.updated, summary.removed) == (0, 0, 0)
    delta = [r for r in graph.requests if "/root/delta" in r]
    assert len(delta) == 1 and "token=" in delta[0], "the saved delta link, not a full walk"
    assert not any("/content" in r for r in graph.requests), "nothing changed, nothing downloaded"
    assert not any("/permissions" in r for r in graph.requests), "permissions are fresh enough"
    assert graph.token_calls == 1, "the token is still good"


def test_a_change_a_rename_and_a_delete_arrive_as_exactly_those(
    graph: FakeGraph, tmp_path: Path
) -> None:
    clock = _Clock()
    sync = _sync(graph, tmp_path, clock)
    sync.run()
    graph.change_content("drive-1", "i-expenses", b"# Expenses\nEUR 1,000.")
    graph.set_permissions("drive-1", "i-expenses", [group_grant(FINANCE)])
    graph.rename("drive-1", "i-leave", "HR/Leave/parental-leave.md")
    graph.delete("drive-1", "i-secret")
    graph.requests.clear()

    clock.now += 60
    summary = sync.run()
    assert summary.errors == []
    assert (summary.added, summary.updated, summary.removed) == (0, 2, 1), summary.as_dict()
    assert _mirrored(tmp_path) == {
        "sharepoint/Documents/HR/Leave/parental-leave.md",
        "sharepoint/Documents/Finance/expenses.md",
    }
    expenses = tmp_path / "documents/sharepoint/Documents/Finance/expenses.md"
    assert expenses.read_bytes() == b"# Expenses\nEUR 1,000."
    emptied = tmp_path / "documents/sharepoint/Documents/HR/Policies"
    assert not emptied.exists(), "emptied folders go"
    readers = sync.principals_map()
    assert readers["sharepoint/Documents/Finance/expenses.md"] == {f"group:{FINANCE}"}, (
        "a changed file has its permissions re-read, so Alice's revoked grant is gone"
    )
    assert "sharepoint/Documents/HR/board-minutes.md" not in readers
    assert sum("/content" in r for r in graph.requests) == 2, "the changed file and the renamed one"
    assert summary.withheld == 0


def test_permissions_are_re_read_on_a_clock_even_when_nothing_changed(
    graph: FakeGraph, tmp_path: Path
) -> None:
    clock = _Clock()
    sync = _sync(graph, tmp_path, clock)
    sync.run()
    graph.set_permissions("drive-1", "i-leave", [group_grant(HR), group_grant(FINANCE)])
    graph.requests.clear()

    clock.now += 600
    sync.run()
    assert not any("/permissions" in r for r in graph.requests), "ten minutes is inside the hour"
    leave = "sharepoint/Documents/HR/Policies/parental-leave.md"
    assert sync.principals_map()[leave] == {f"group:{HR}"}

    clock.now += 3600
    summary = sync.run()
    assert summary.permissions_read == 3, "past the hour every file's readers are asked again"
    assert sync.principals_map()["sharepoint/Documents/HR/Policies/parental-leave.md"] == {
        f"group:{HR}",
        f"group:{FINANCE}",
    }
    assert not any("/content" in r for r in graph.requests), "re-reading readers downloads nothing"


def test_throttling_is_waited_out_and_an_expired_token_is_refreshed(
    graph: FakeGraph, tmp_path: Path
) -> None:
    sync = _sync(graph, tmp_path, _Clock())
    graph.throttle_once = 3
    graph.expire_token_once = True
    summary = sync.run()
    assert summary.errors == [], summary.errors
    assert summary.added == 3
    assert sync.graph.slept == [3.0], "Retry-After was honoured, once"  # type: ignore[attr-defined]
    assert graph.token_calls == 2, "the 401 bought a new token, not a retry of the old one"


def test_an_expired_delta_link_re_reads_the_library(graph: FakeGraph, tmp_path: Path) -> None:
    clock = _Clock()
    sync = _sync(graph, tmp_path, clock)
    sync.run()
    graph.expire_delta_once = True
    graph.requests.clear()
    clock.now += 60
    summary = sync.run()
    assert summary.errors == []
    delta = [r for r in graph.requests if "/root/delta" in r]
    assert "token=" in delta[0] and "token=" not in delta[1], "410, then a walk from the start"
    assert _mirrored(tmp_path) == {
        "sharepoint/Documents/HR/Policies/parental-leave.md",
        "sharepoint/Documents/Finance/expenses.md",
        "sharepoint/Documents/HR/board-minutes.md",
    }


def test_a_refusal_runs_nothing_and_says_why(graph: FakeGraph, tmp_path: Path) -> None:
    sync = _sync(graph, tmp_path, _Clock())
    sync.refusal = "sign-in is off, so no reader can be enforced"
    summary = sync.run()
    assert summary.errors == ["sign-in is off, so no reader can be enforced"]
    assert graph.requests == [] and _mirrored(tmp_path) == set()
    assert sync.status()["last_error"] == "sign-in is off, so no reader can be enforced"


def test_a_dead_graph_is_an_error_in_the_summary_not_an_exception(tmp_path: Path) -> None:
    config = GraphConfig(
        tenant_id=TENANT,
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
        site="contoso.sharepoint.com:/sites/HR",
        graph_url="http://127.0.0.1:9/v1.0",
        login_url="http://127.0.0.1:9",
    )
    sync = SharePointSync(
        GraphClient(config),
        documents_dir=tmp_path / "documents",
        store=SyncStore(),
    )
    summary = sync.run()
    assert len(summary.errors) == 1 and summary.documents == 0
    assert sync.status()["last_error"] is not None
