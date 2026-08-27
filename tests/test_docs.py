"""The documentation must only tell people to run things that exist.

Every guide in this repository is a promise that a command works. The failure
this file exists to catch is the cheap one: a doc written slightly ahead of the
code, or left behind by a rename, that sends a new user's very first command
into an argparse error. On a project whose argument is "check it yourself rather
than believing us", that is worse than an undocumented feature.

It walks the parser tree rather than running each command: exact, and it proves
a flag *exists* rather than that argparse exited non-zero, which it does both
for an unknown flag and for a known one missing its value.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pytest

from openknowledge.cli import build_parser

ROOT = Path(__file__).resolve().parent.parent

#: `openknowledge <command> [<subcommand>] <rest of the line>`, in prose or in a
#: fenced block. Placeholder forms (`openknowledge model use <name>`) resolve as
#: well as concrete ones, since only the command path is checked.
INVOCATION = re.compile(r"\bopenknowledge ([a-z][a-z-]*(?: [a-z][a-z-]*)?)([^\n`]*)")

#: Words that follow the binary name in prose without being subcommands.
PROSE = {"is", "in", "on", "and", "to", "the", "as", "was", "will", "does", "reads"}


def documents() -> list[Path]:
    found = [ROOT / "README.md", *sorted((ROOT / "docs").rglob("*.md"))]
    return [path for path in found if path.exists()]


def _resolve(parser: argparse.ArgumentParser, words: list[str]) -> argparse.ArgumentParser | None:
    """Walk down the subparser tree, or None where the path does not exist."""
    for word in words:
        actions = [a for a in parser._actions if isinstance(a, argparse._SubParsersAction)]
        if not actions or word not in actions[0].choices:
            return None
        parser = actions[0].choices[word]
    return parser


def _flags(parser: argparse.ArgumentParser) -> set[str]:
    return {option for action in parser._actions for option in action.option_strings}


def _invocations(text: str) -> list[tuple[list[str], str]]:
    """Command paths shown in ``text``, longest path first, with their tails."""
    found: list[tuple[list[str], str]] = []
    for path, tail in INVOCATION.findall(text):
        words = [w for w in path.split() if w not in PROSE]
        if words:
            found.append((words, tail))
    return found


@pytest.mark.parametrize("path", documents(), ids=lambda p: p.name)
def test_every_command_a_document_shows_is_a_real_command(path: Path) -> None:
    parser = build_parser()
    for words, _ in _invocations(path.read_text()):
        # A one-word path always has to resolve. A two-word one may be a command
        # followed by an ordinary word ("openknowledge index re-reads..."), so
        # it is enough that its first word does.
        resolved = _resolve(parser, words) or _resolve(parser, words[:1])
        assert resolved is not None, (
            f"{path.name} tells the reader to run `openknowledge {' '.join(words)}`, "
            "which is not a command"
        )


@pytest.mark.parametrize("path", documents(), ids=lambda p: p.name)
def test_every_flag_a_document_shows_is_a_real_flag(path: Path) -> None:
    """The subtler case: a real command shown with a flag it does not have.

    `openknowledge audit --min-overlap` was written into the setup guide before
    the flag existed. That is what this catches.
    """
    parser = build_parser()
    for words, tail in _invocations(path.read_text()):
        resolved = _resolve(parser, words)
        if resolved is None:
            resolved = _resolve(parser, words[:1])
            tail = f"{' '.join(words[1:])} {tail}"
        if resolved is None:
            continue  # the command test reports this
        available = _flags(resolved)
        for flag in re.findall(r"(?<![\w-])(--[a-z][a-z-]+)", tail):
            assert flag in available, (
                f"{path.name} shows `openknowledge {' '.join(words)} {flag}`, "
                f"which that command does not have (it has: {', '.join(sorted(available))})"
            )


def test_the_setup_guide_points_at_the_installer_that_exists() -> None:
    guide = (ROOT / "docs" / "LOCAL-SETUP.md").read_text()
    assert "install.sh" in guide
    assert (ROOT / "install.sh").exists()

    # The guide promises the whole thing lives in one folder. Two things are
    # excluded before checking, and both for the same reason - the test has to
    # look at what the script *does*, not at what it says.
    #
    #   comments: the header itself says "no sudo, no system packages".
    #   heredocs: the closing message tells the reader how to add the PATH line
    #             to their own .bashrc. Printing that advice is the opposite of
    #             doing it behind their back.
    code, in_heredoc = [], False
    for line in (ROOT / "install.sh").read_text().splitlines():
        stripped = line.strip()
        if in_heredoc:
            in_heredoc = stripped != "EOF"
            continue
        if "<<EOF" in stripped:
            in_heredoc = True
            continue
        if not stripped.startswith("#"):
            code.append(line)
    executed = "\n".join(code)

    for forbidden in ("sudo", "apt-get", "brew install", ".bashrc", ".zshrc", "/usr/local"):
        assert forbidden not in executed, f"install.sh reaches outside its folder: {forbidden}"
