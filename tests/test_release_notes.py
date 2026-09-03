"""What a release page says, and the promise that adding a section changed nothing.

`tools/release_notes.py` was extracted from a heredoc inside `release.yml` so
that a version could carry its own section. The risk in that move is silent:
a stray space or a lost line would land on the download page of every future
release and nobody would notice, because nobody rereads boilerplate.

So the first test here is not written from the template. `PUBLISHED_0_12_5` is
the body of https://github.com/FedericoTs/OpenKnowledge/releases/tag/v0.12.5
exactly as the releases API returns it, published 2026-09-03T18:02:37Z by the
old heredoc - the last release cut before this change. With no notes file, the
script has to reproduce it byte for byte. That is a fixed point outside this
repository's own opinion of what the text should be.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml
from tools.release_notes import body, install_paragraph, main, version_notes

ROOT = Path(__file__).resolve().parents[1]
NOTES_DIR = ROOT / "docs/release-notes"

UNSIGNED = (
    "Not code-signed. SmartScreen shows 'Windows protected your PC'; More info, "
    "then Run anyway. Check the SHA-256 below instead. See docs/WINDOWS.md."
)
SETUP = "OpenKnowledge-Setup-0.12.5.exe"
SHA = "b145368fa117d0db02ce4050debe70ffbd045e453fdda027019f62b14e53f683"

PUBLISHED_0_12_5 = f"""The Windows desktop app, as one installer.

**Install:** download `{SETUP}` below and run it. Windows SmartScreen
will warn about an unrecognised app because this build is not yet
code-signed - choose **More info → Run anyway**. No administrator
rights are needed; it installs for your user only.

**First launch** (Start menu → OpenKnowledge) downloads the two
models it answers with - about 2.6 GB, once, each verified against a
pinned SHA-256 - starts the local inference servers, and opens the
chatbot in your browser. Add documents by dragging them into the
chat; manage everything at /manage. Nothing leaves your machine.

**Verified before publishing:** this pipeline silent-installed this
artifact on a clean Windows machine, sat through the real first run,
uploaded a document, and asked a real question - the answer had to be
grounded, cost $0, and come back byte-identical when asked twice.
The first recorded run of that proof lives in
`evals/measured/windows-e2e-first-run.json`.

**Code signing:** {UNSIGNED}

SHA-256 of `{SETUP}`:
`{SHA}`
"""


def test_without_a_notes_file_the_body_is_what_v0_12_5_actually_published() -> None:
    assert body(setup=SETUP, sha=SHA, signing=UNSIGNED) == PUBLISHED_0_12_5


def test_a_version_with_notes_carries_them_above_the_standing_text() -> None:
    said = body(
        setup=SETUP,
        sha=SHA,
        signing=UNSIGNED,
        notes="## What changed\n\n57% on a corpus we did not write.",
    )
    assert said.startswith("## What changed\n\n57% on a corpus we did not write.\n\n---\n\n")
    assert said.endswith(PUBLISHED_0_12_5), "the standing paragraphs still follow, unchanged"


def test_the_notes_file_is_found_by_the_tag_and_only_by_the_tag(tmp_path: Path) -> None:
    (tmp_path / "v0.13.0.md").write_text("First light.\n", encoding="utf-8")
    assert version_notes("v0.13.0", tmp_path) == "First light."
    assert version_notes("v0.13.1", tmp_path) is None
    assert version_notes("0.13.0", tmp_path) is None, "the file is named for the tag, with its v"


def test_an_empty_notes_file_does_not_put_a_bare_rule_on_the_page(tmp_path: Path) -> None:
    (tmp_path / "v0.13.0.md").write_text("\n   \n\n", encoding="utf-8")
    assert version_notes("v0.13.0", tmp_path) is None
    assert body(setup=SETUP, sha=SHA, signing=UNSIGNED, notes=None) == PUBLISHED_0_12_5


def test_the_install_advice_follows_the_recorded_signing_state() -> None:
    unsigned = install_paragraph(SETUP, UNSIGNED)
    signed = install_paragraph(SETUP, "Signed by Some Publisher, valid, timestamped.")
    assert "not yet\ncode-signed" in unsigned and "unrecognised app" in unsigned
    assert "The installer is\ncode-signed" in signed
    assert "still warns while the signature is new" in signed, (
        "a signed build can meet SmartScreen while its reputation is young"
    )
    assert SETUP in unsigned and SETUP in signed


def test_the_signing_state_must_be_given_and_is_never_guessed(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """No default: an unstated signing state would put an unmeasured sentence on
    a download page, and the two wordings contradict each other."""
    common = ["--tag", "v0.12.5", "--setup", SETUP, "--sha", SHA, "--notes-dir", str(tmp_path)]
    with pytest.raises(SystemExit) as refused:
        main(common)
    assert refused.value.code != 0
    assert "one of the arguments" in capsys.readouterr().err

    # The literal string is for previewing; the file is what the build wrote.
    # They have to produce the same page from the same state.
    assert main([*common, "--signing", UNSIGNED]) == 0
    from_string = capsys.readouterr().out
    recorded = tmp_path / "SIGNING.txt"
    recorded.write_text(UNSIGNED + "\n", encoding="utf-8")
    assert main([*common, "--signing-file", str(recorded)]) == 0
    assert capsys.readouterr().out == from_string == PUBLISHED_0_12_5


# -- the workflow that calls it ---------------------------------------------------

WORKFLOW = yaml.safe_load((ROOT / ".github/workflows/release.yml").read_text(encoding="utf-8"))
PUBLISH = WORKFLOW["jobs"]["publish"]["steps"]


def test_publish_checks_out_before_it_downloads_the_installer() -> None:
    uses = [step.get("uses", "") for step in PUBLISH]
    checkout = next(i for i, u in enumerate(uses) if u.startswith("actions/checkout@"))
    download = next(i for i, u in enumerate(uses) if u.startswith("actions/download-artifact@"))
    assert checkout < download, (
        "checkout cleans the workspace; downloading first would put the installer "
        "there for checkout to delete"
    )


def test_publish_builds_its_notes_with_the_script_and_nothing_else() -> None:
    script = "\n".join(step.get("run", "") for step in PUBLISH)
    assert "tools/release_notes.py" in script
    assert "--signing-file SIGNING.txt" in script, "the state the build recorded, not a guess"
    assert "> notes.md" in script and "--notes-file notes.md" in script
    assert "cat > notes.md" not in script, "the template lives in one place now"


def test_every_file_in_the_notes_directory_would_actually_be_used() -> None:
    """A misnamed file is silent: the release publishes without it.

    The only way to notice is a bare release page after the fact, so the
    naming is checked here instead.
    """
    assert (NOTES_DIR / "README.md").is_file(), "the convention is written down beside the files"
    stray = [
        p.name
        for p in NOTES_DIR.iterdir()
        if p.name != "README.md" and not re.fullmatch(r"v\d+\.\d+\.\d+\.md", p.name)
    ]
    assert stray == [], f"named for no tag, so no release would ever pick them up: {stray}"
    blank = [p.name for p in NOTES_DIR.glob("v*.md") if not p.read_text(encoding="utf-8").strip()]
    assert blank == [], f"named for a tag but holding nothing to say: {blank}"
