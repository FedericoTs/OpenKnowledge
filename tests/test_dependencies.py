"""Every third-party import in src/ must be a declared dependency.

Hybrid retrieval shipped importing numpy, which no manifest declared. It worked
on the development machine only because llama-cpp-python - a test tool, not a
dependency - happened to pull numpy in, so 626 tests passed while a clean
install crashed the moment an embedding endpoint was reachable. "Works here"
and "is installable" are different properties, and only one of them was tested.

This walks the AST of everything under src/openknowledge and checks each
absolute import against pyproject.toml. Optional imports are allowed only when
they are named in an extra and the module guards for their absence.
"""

from __future__ import annotations

import ast
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src" / "openknowledge"

#: import name -> distribution name, where they differ.
DISTRIBUTION_OF = {
    "yaml": "pyyaml",
    "docx": "python-docx",
    "pptx": "python-pptx",
    "PIL": "Pillow",
    "pydantic_settings": "pydantic-settings",
}

#: Modules whose distribution is a hard dependency of one we declare, pinned
#: by it tightly enough that declaring our own bound would only invite skew.
#: FastAPI is versioned against starlette minor-by-minor; importing what it
#: guarantees is safer than second-guessing its pin.
PROVIDED_BY = {"starlette": "fastapi"}

#: Imports that are deliberately optional: declared in an extra, and imported
#: only behind a guard that degrades cleanly when the package is absent.
OPTIONAL = {
    "anthropic": "anthropic",
    "opendataloader_pdf": "opendataloader-pdf",
    "pystray": "pystray",
    "PIL": "pillow",
}


def _declared() -> set[str]:
    meta = tomllib.loads((ROOT / "pyproject.toml").read_text())
    names = set()
    for req in meta["project"]["dependencies"]:
        names.add(req.split("[")[0].split(">")[0].split("=")[0].split("<")[0].strip().lower())
    return names


def _imported() -> dict[str, list[str]]:
    found: dict[str, list[str]] = {}
    for path in sorted(SRC.rglob("*.py")):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module]
            for name in names:
                top = name.split(".")[0]
                found.setdefault(top, []).append(str(path.relative_to(ROOT)))
    return found


def test_every_import_in_src_is_a_declared_dependency() -> None:
    declared = _declared()
    problems = []
    for module, places in sorted(_imported().items()):
        if module in sys.stdlib_module_names or module == "openknowledge":
            continue
        if module in OPTIONAL:
            continue  # covered by the extras assertions below
        if module in PROVIDED_BY:
            if PROVIDED_BY[module] in declared:
                continue
            problems.append(f"{module} rides on {PROVIDED_BY[module]}, which is no longer declared")
            continue
        distribution = DISTRIBUTION_OF.get(module, module).lower()
        if distribution not in declared:
            problems.append(f"{module} (used in {places[0]}) is not in [project.dependencies]")
    assert not problems, "undeclared runtime dependencies:\n  " + "\n  ".join(problems)


def test_optional_imports_are_declared_in_an_extra() -> None:
    meta = tomllib.loads((ROOT / "pyproject.toml").read_text())
    extras = {
        req.split("[")[0].split(">")[0].split("=")[0].strip().lower()
        for reqs in meta["project"]["optional-dependencies"].values()
        for req in reqs
    }
    for module, distribution in OPTIONAL.items():
        assert distribution.lower() in extras, (
            f"{module} is on the optional allowlist but no extra provides {distribution}"
        )
