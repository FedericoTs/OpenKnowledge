"""The release workflow's promises, read from the YAML rather than remembered.

Three of them are load-bearing. The tag has to equal the version in
pyproject.toml before anything is built. PyPI has to come after the GitHub
release, never instead of it, so a wheel is only ever published for a version
whose installer was proved on a machine. And the wheel has to be installed
somewhere else and run before it is uploaded, because every other test here
runs against the checkout, where the web assets are always found.

The README is part of the same promise: PyPI renders it, and a relative image
path is a broken picture on the one page a stranger reads first.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = yaml.safe_load((ROOT / ".github/workflows/release.yml").read_text(encoding="utf-8"))
JOBS = WORKFLOW["jobs"]


def _steps(job: str) -> list[dict]:
    return JOBS[job]["steps"]


def _scripts(job: str) -> str:
    return "\n".join(step.get("run", "") for step in _steps(job))


def test_the_tag_is_checked_against_pyproject_before_anything_is_built() -> None:
    assert "check" in JOBS
    script = _scripts("check")
    assert "pyproject.toml" in script and "tomllib" in script
    assert "exit 1" in script, "a mismatch has to fail the job, not print a warning"
    assert JOBS["build"]["needs"] == "check", "the installer must not build for a wrong tag"


def test_pypi_comes_after_the_github_release_and_only_then() -> None:
    pypi = JOBS["pypi"]
    assert pypi["needs"] == "publish"
    assert JOBS["publish"]["needs"] == "build"
    # Trusted publishing: an OIDC identity, not a stored token.
    assert pypi["permissions"]["id-token"] == "write"
    assert pypi["environment"] == "pypi"
    uses = [step.get("uses", "") for step in _steps("pypi")]
    assert any(u.startswith("pypa/gh-action-pypi-publish@") for u in uses), uses
    assert not any("password" in step.get("with", {}) for step in _steps("pypi")), (
        "no token: PyPI trusts the workflow's identity"
    )


def test_the_wheel_is_installed_elsewhere_and_run_before_upload() -> None:
    script = _scripts("pypi")
    assert "uv build" in script and "twine check" in script
    assert "cd /tmp" in script, "the smoke run must not see the checkout's web/ folder"
    assert "--version" in script and "exit 1" in script
    assert "find_asset" in script and "widget/index.html" in script
    assert "openknowledge audit" in script
    names = [step.get("name", "") for step in _steps("pypi")]
    assert names.index("Install the wheel somewhere else and run it") < names.index(
        "Publish to PyPI"
    )


def test_the_readme_has_no_relative_images_because_pypi_renders_it() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    images = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", readme)
    assert images, "the first screen is supposed to carry a screenshot"
    relative = [src for src in images if not src.startswith(("https://", "http://"))]
    assert relative == [], f"PyPI cannot resolve these: {relative}"


def test_the_readme_is_what_pypi_shows() -> None:
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'readme = "README.md"' in pyproject
    assert (ROOT / "docs/RELEASING.md").is_file()
