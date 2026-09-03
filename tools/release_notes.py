"""Build the body of a GitHub Release, with an optional per-version section.

Every release page carried the same paragraphs for a year: how to install,
what the first launch does, what the pipeline proved before publishing, and
whether the installer is signed. All of that is true of every version, which
is why it is a template - and it meant a release page could never say what
changed in *this* version. v0.12.5 was cut to retire an accuracy claim the
FTR evaluation disproved, and its release page said nothing about it: it
differed from v0.12.4's only in the version string and the SHA-256.

So: if ``docs/release-notes/<tag>.md`` exists, it goes at the top of the body,
above a rule, above the standing paragraphs. If it does not, the body is
byte-for-byte what this project has always published - a promise
``tests/test_release_notes.py`` holds against the text v0.12.5 actually
shipped, fetched from the releases API rather than remembered.

The workflow calls it; a person writing notes can preview them the same way:

    python3 tools/release_notes.py --tag v0.13.0 \
        --setup OpenKnowledge-Setup-0.13.0.exe --sha $(sha256sum ...) \
        --signing-file SIGNING.txt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

#: Where a version's own notes live. One file per tag, named for the tag.
NOTES_DIR = Path("docs/release-notes")

_SIGNED_INSTALL = """**Install:** download `{setup}` below and run it. The installer is
code-signed; if SmartScreen still warns while the signature is new,
choose **More info → Run anyway**. No administrator rights are needed;
it installs for your user only."""

_UNSIGNED_INSTALL = """**Install:** download `{setup}` below and run it. Windows SmartScreen
will warn about an unrecognised app because this build is not yet
code-signed - choose **More info → Run anyway**. No administrator
rights are needed; it installs for your user only."""

_STANDING = """The Windows desktop app, as one installer.

{install}

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

**Code signing:** {signing}

SHA-256 of `{setup}`:
`{sha}`
"""


def install_paragraph(setup: str, signing: str) -> str:
    """The install advice the recorded signing state makes true.

    A signed installer can still meet SmartScreen while its reputation is
    young; an unsigned one always does. Telling a stranger to expect the
    warning when there will not be one is as wrong as the reverse.
    """
    template = _SIGNED_INSTALL if signing.startswith("Signed by") else _UNSIGNED_INSTALL
    return template.format(setup=setup)


def version_notes(tag: str, notes_dir: Path = NOTES_DIR) -> str | None:
    """This version's own section, or None when the release has nothing to add.

    A file that exists but holds only whitespace counts as nothing to add: it
    must not put a bare rule on the page.
    """
    path = notes_dir / f"{tag}.md"
    if not path.is_file():
        return None
    text = path.read_text(encoding="utf-8").strip()
    return text or None


def body(*, setup: str, sha: str, signing: str, notes: str | None = None) -> str:
    """The whole release body: this version's section, then the standing one."""
    standing = _STANDING.format(
        install=install_paragraph(setup, signing), signing=signing, setup=setup, sha=sha
    )
    if notes is None:
        return standing
    return f"{notes}\n\n---\n\n{standing}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True, help="the release tag, e.g. v0.13.0")
    parser.add_argument("--setup", required=True, help="the installer's filename")
    parser.add_argument("--sha", required=True, help="SHA-256 of the installer")
    # The workflow passes the file the build wrote; a person previewing their
    # own notes has no such file and should not have to fake one on disk.
    # Exactly one, always: guessing the signing state would put a sentence on
    # a download page that nothing measured.
    state = parser.add_mutually_exclusive_group(required=True)
    state.add_argument(
        "--signing-file",
        type=Path,
        help="the signing state the build recorded, travelling with the artifact",
    )
    state.add_argument("--signing", help="that state as a literal string, for previewing")
    parser.add_argument(
        "--notes-dir",
        type=Path,
        default=NOTES_DIR,
        help="where per-version notes live (default: docs/release-notes)",
    )
    args = parser.parse_args(argv)

    notes = version_notes(args.tag, args.notes_dir)
    if notes:
        where = args.notes_dir / f"{args.tag}.md"
        said = f"carrying the section in {where}"
    else:
        said = "no section of its own - publishing the standing text"
    print(f"release notes for {args.tag}: {said}", file=sys.stderr)
    signing = (
        args.signing
        if args.signing is not None
        else args.signing_file.read_text(encoding="utf-8").strip()
    )
    sys.stdout.write(body(setup=args.setup, sha=args.sha, signing=signing, notes=notes))
    return 0


if __name__ == "__main__":
    sys.exit(main())
