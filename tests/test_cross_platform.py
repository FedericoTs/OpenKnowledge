"""Two assumptions this suite is not allowed to make about the machine.

The Windows job is the last thing to run and the slowest - roughly sixteen
minutes - so a test that is green here and red there costs a cycle every
time. It happened twice in one afternoon, both times in a test rather than
in the product, and both times for the same underlying reason: the test
asserted on a *rendering* of something rather than on the property.

    UnicodeDecodeError: 'charmap' codec can't decode byte 0x9d in position
    3171: character maps to <undefined>

        `Path.read_text()` with no encoding uses the locale's, which is
        cp1252 on the runner. Two source files hold a curly quote.

    AssertionError: '/etc/passwd' became the path 'etc/passwd'

        `str(Path("/corpus") / "etc/passwd")` is 'D:\\corpus\\etc\\passwd'
        there, so a comparison against "/corpus/" failed a path that had
        escaped nothing.

Neither is exotic and neither needed a Windows machine to find - only a
question asked in a form both filesystems answer the same way. These two
tests ask it here, in under a second, over the whole repository.

They are deliberately narrow. A rule that fires on code that is fine
teaches people to route around it, so each one was measured against the
repository before it was written: the encoding rule flagged 80 real call
sites, all since fixed, and the path rule flagged the one line that had
just failed on the runner and nothing else. Rules with no hazard behind
them - hardcoded /tmp, paths glued with a literal separator - were measured
too, found to have one near-miss between them, and left out.
"""

from __future__ import annotations

import ast
from pathlib import Path

ROOTS = ("src/openknowledge", "tests", "tools")
REPOSITORY = Path(__file__).resolve().parent.parent

#: `open` on one of these is not a file, whatever it is called.
NOT_A_FILE = frozenset({"webbrowser", "pdfplumber", "zipfile", "gzip", "tarfile", "io", "shelve"})

#: What a path is being asked about, when it is asked as a string.
PATHISH_ATTRIBUTES = frozenset({"resolve", "parent", "absolute", "parents", "joinpath"})


def python_files() -> list[Path]:
    found = [
        path
        for root in ROOTS
        for path in sorted((REPOSITORY / root).rglob("*.py"))
        if "__pycache__" not in path.parts
    ]
    assert len(found) > 50, "the sweep found almost nothing; the roots are probably wrong"
    return found


def _where(path: Path, node: ast.AST) -> str:
    return f"{path.relative_to(REPOSITORY)}:{node.lineno}"  # type: ignore[attr-defined]


def _mode_of(node: ast.Call) -> str | None:
    """The literal mode this open() was given, or None if it is computed.

    A mode built at runtime is rare (one site in this repository, a resumed
    download choosing between 'ab' and 'wb') and cannot be judged from here,
    so it is left alone rather than guessed at.
    """
    for keyword in node.keywords:
        if keyword.arg == "mode":
            return keyword.value.value if isinstance(keyword.value, ast.Constant) else None
    if isinstance(node.func, ast.Name):  # the builtin: open(path, mode)
        argument = node.args[1] if len(node.args) > 1 else None
    else:  # the method: path.open(mode)
        argument = node.args[0] if node.args else None
    if argument is None:
        return ""
    return argument.value if isinstance(argument, ast.Constant) else None


def _is_a_file_open(node: ast.Call) -> bool:
    if isinstance(node.func, ast.Name):
        return node.func.id == "open"
    if not (isinstance(node.func, ast.Attribute) and node.func.attr == "open"):
        return False
    base = node.func.value
    if isinstance(base, ast.Name):
        return base.id not in NOT_A_FILE
    if isinstance(base, ast.Attribute):
        return base.attr not in NOT_A_FILE
    return True


def test_every_text_read_and_write_says_which_encoding() -> None:
    """Because 'the locale's' is a different answer on the runner.

    ``errors="replace"`` is not a substitute: it stops the exception, not
    the wrong characters.
    """
    unsaid: list[str] = []
    for path in python_files():
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not isinstance(node, ast.Call):
                continue
            said = any(keyword.arg == "encoding" for keyword in node.keywords)
            if said:
                continue
            if isinstance(node.func, ast.Attribute) and node.func.attr in {
                "read_text",
                "write_text",
            }:
                unsaid.append(f"{_where(path, node)} {node.func.attr}()")
            elif _is_a_file_open(node):
                mode = _mode_of(node)
                if mode is not None and "b" not in mode:
                    unsaid.append(f"{_where(path, node)} open({mode!r})")
    assert not unsaid, (
        "these read or write text without saying in which encoding, so they "
        "decode as cp1252 on the Windows runner and as UTF-8 here: " + "; ".join(unsaid)
    )


def _pathish(node: ast.AST) -> bool:
    """Whether this expression is a path rather than an arbitrary string."""
    for inner in ast.walk(node):
        if isinstance(inner, ast.Name) and inner.id in {"Path", "PurePath", "PosixPath"}:
            return True
        if isinstance(inner, ast.Attribute) and inner.attr in PATHISH_ATTRIBUTES:
            return True
        if isinstance(inner, ast.BinOp) and isinstance(inner.op, ast.Div):
            return True  # the `/` operator, which only paths use this way
    return False


def test_no_path_is_compared_as_a_posix_string() -> None:
    """A path's string form is D:\\corpus\\x there and /corpus/x here.

    Ask containment with ``is_relative_to``, equality with ``==`` between
    two Paths, and a suffix with ``.name`` or ``.suffix``. Every one of
    those gives the same answer on both filesystems; ``str()`` does not.
    """
    rendered: list[str] = []
    for path in python_files():
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            call = None
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in {"startswith", "endswith"}
            ):
                call = node.func.value
            elif isinstance(node, ast.Compare) and any(
                isinstance(op, (ast.Eq, ast.NotEq)) for op in node.ops
            ):
                literal = any(
                    isinstance(other, ast.Constant)
                    and isinstance(other.value, str)
                    and "/" in other.value
                    for other in node.comparators
                )
                call = node.left if literal else None
            if (
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Name)
                and call.func.id == "str"
                and call.args
                and _pathish(call.args[0])
            ):
                rendered.append(_where(path, node))
    assert not rendered, (
        "these compare a path's string form against a literal, which is a "
        "POSIX answer to a question Windows spells differently: " + "; ".join(rendered)
    )
