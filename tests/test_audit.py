"""The zero-configuration front door.

`openknowledge audit` is the one command that has to work for somebody who has
not decided whether they trust this project yet: no API key, no model, no
database, nothing written, nothing sent. These tests hold it to that, and to
producing a report that reads as two of the reader's own sentences rather than
as a tool's opinion.
"""

from __future__ import annotations

from pathlib import Path

from openknowledge.audit import audit_folder, render
from openknowledge.cli import main

POLICY = """# Expenses Policy

Travel above EUR 500 requires prior approval from a line manager.
The meal allowance limit is EUR 45 per day.
"""

CONTRADICTING = """# Travel Guidelines

Travel above EUR 1,000 requires prior approval from a line manager.
"""

AGREEING = """# Travel Summary

Travel above EUR 500 requires prior approval from a line manager.
"""


def write(root: Path, **files: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for name, body in files.items():
        (root / f"{name}.md").write_text(body, encoding="utf-8")
    return root


def test_it_finds_a_moved_figure_and_quotes_both_sides(tmp_path: Path) -> None:
    root = write(tmp_path / "docs", expenses=POLICY, travel=CONTRADICTING)
    report = audit_folder(root)

    assert report.documents == 2
    assert not report.clean
    assert {report.conflicts[0].left.raw, report.conflicts[0].right.raw} == {
        "EUR 500",
        "EUR 1,000",
    }

    text = render(report)
    assert "EUR 500" in text and "EUR 1,000" in text
    # Both sentences, not just the figures: the finding has to be checkable
    # without opening either file.
    assert "requires prior approval" in text


def test_documents_that_agree_produce_nothing(tmp_path: Path) -> None:
    root = write(tmp_path / "docs", expenses=POLICY, summary=AGREEING)
    report = audit_folder(root)

    assert report.clean
    assert "No contradictions found" in render(report)


def test_it_writes_nothing_and_needs_no_configuration(tmp_path: Path, monkeypatch) -> None:
    root = write(tmp_path / "docs", expenses=POLICY, travel=CONTRADICTING)
    before = sorted(p.name for p in root.rglob("*"))

    # Run from an empty directory so "created nothing" is about the audit rather
    # than about whatever else has touched the repository working tree.
    workdir = tmp_path / "cwd"
    workdir.mkdir()
    monkeypatch.chdir(workdir)

    audit_folder(root)

    assert sorted(p.name for p in root.rglob("*")) == before
    assert list(workdir.iterdir()) == [], "the audit must build no store and no data directory"


def test_the_same_folder_produces_the_same_report_twice(tmp_path: Path) -> None:
    """A report an admin cannot reproduce is a report they cannot act on."""
    root = write(tmp_path / "docs", expenses=POLICY, travel=CONTRADICTING)
    assert render(audit_folder(root)) == render(audit_folder(root))


def test_unreadable_files_are_named_with_a_remedy(tmp_path: Path) -> None:
    root = write(tmp_path / "docs", expenses=POLICY)
    (root / "handbook.doc").write_bytes(b"\xd0\xcf\x11\xe0legacy")

    report = audit_folder(root)

    assert [s.path for s in report.unreadable] == ["handbook.doc"]
    assert ".docx" in report.unreadable[0].reason
    assert "handbook.doc" in render(report)


def test_a_report_states_what_it_did_not_check(tmp_path: Path) -> None:
    """Silence about a gap is how a corpus develops a hole nobody knows about."""
    root = write(tmp_path / "docs", expenses=POLICY)
    text = render(audit_folder(root))

    assert "does not check" in text
    assert "OCR" in text
    assert "left this machine" in text


def test_json_output_carries_both_quotes_and_the_duplicate_pairs(tmp_path: Path) -> None:
    root = write(tmp_path / "docs", expenses=POLICY, travel=CONTRADICTING)
    payload = audit_folder(root).as_dict()

    assert payload["documents"] == 2
    assert payload["claims_checked"] > 0
    assert payload["conflicts"][0]["left"]["sentence"]
    assert payload["conflicts"][0]["right"]["sentence"]
    assert payload["duplicates"] == []


def test_the_cli_exits_non_zero_on_findings_so_it_can_gate_ci(tmp_path: Path) -> None:
    root = write(tmp_path / "docs", expenses=POLICY, travel=CONTRADICTING)

    assert main(["audit", str(root)]) == 1
    assert main(["audit", str(root), "--exit-zero"]) == 0
    assert main(["audit", str(root), "--json"]) == 1


def test_the_cli_reports_an_empty_folder_rather_than_passing_it(tmp_path: Path) -> None:
    """Exit 0 on a folder that read as nothing would be a false all-clear."""
    empty = tmp_path / "empty"
    empty.mkdir()
    assert main(["audit", str(empty)]) == 2


def test_a_clean_folder_exits_zero(tmp_path: Path) -> None:
    root = write(tmp_path / "docs", expenses=POLICY, summary=AGREEING)
    assert main(["audit", str(root)]) == 0


def _register(rows: list[tuple[str, str, str]]) -> str:
    lines = ["# Vendor Register", "", "| Service | Uptime | Notice |", "|---|---|---|"]
    lines += [f"| {name} | {uptime} | {notice} |" for name, uptime, notice in rows]
    return "\n".join(lines)


def test_two_copies_of_one_document_read_as_duplication_not_as_forty_findings(
    tmp_path: Path,
) -> None:
    """The failure that 15 real contracts exposed, in miniature.

    Two versions of one register disagree on nearly every row. Enumerating each
    row is technically correct and useless: the reader's problem is that they
    have two of these, and that is one sentence.
    """
    rows_v1 = [(f"service-{n}", f"{40 + n}%", f"{n + 1} days") for n in range(20)]
    rows_v2 = [
        (f"service-{n}", f"{40 + n}%" if n < 8 else f"{70 + n}%", f"{n + 1} days")
        if n < 8
        else (f"service-{n}", f"{70 + n}%", f"{n + 20} days")
        for n in range(20)
    ]
    root = write(tmp_path / "docs", register_v1=_register(rows_v1), register_v2=_register(rows_v2))

    report = audit_folder(root)

    assert len(report.conflicts) > 20, "the raw findings should be many"
    assert report.pairs[0].agreements > 0, "the rows they still agree on must be counted"
    assert len(report.variants) == 1
    assert report.contradicting == ()

    text = render(report)
    assert "two versions of the same document" in text
    assert text.count("register-v1") < len(report.conflicts)
