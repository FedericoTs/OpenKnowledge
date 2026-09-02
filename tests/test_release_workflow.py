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
PACKAGE = yaml.safe_load((ROOT / ".github/workflows/package.yml").read_text(encoding="utf-8"))
PACKAGE_JOBS = PACKAGE["jobs"]


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


# -- code signing ----------------------------------------------------------------
#
# The certificate is a decision and a purchase; the pipeline is code, and it
# has to be right before the certificate exists. Signing happens only when
# the repository variables for it are set, never on a pull request, and the
# build refuses an installer whose signature Windows does not call Valid when
# signing was configured. Unsigned or signed, the build records which, the
# artifact carries the record, and the release notes repeat it.

SIGNING_GUARD = "vars.SIGNING_ACCOUNT != '' && github.event_name != 'pull_request'"


def _package_steps(job: str) -> list[dict]:
    return PACKAGE_JOBS[job]["steps"]


def _step(job: str, name: str) -> dict:
    (found,) = [s for s in _package_steps(job) if s.get("name") == name]
    return found


def test_signing_needs_an_identity_the_installer_job_can_prove() -> None:
    job = PACKAGE_JOBS["windows-installer"]
    assert job["permissions"]["id-token"] == "write"
    assert job["environment"] == "signing", "the federated credential is bound to this name"
    assert JOBS["build"]["permissions"]["id-token"] == "write", (
        "a reusable workflow only holds the token its caller grants"
    )


def test_signing_runs_only_when_configured_and_never_on_a_pull_request() -> None:
    names = [s.get("name") for s in _package_steps("windows-installer")]
    login, executables, installer = (
        _step("windows-installer", "Sign in to Azure for signing"),
        _step("windows-installer", "Sign the executables"),
        _step("windows-installer", "Sign the installer"),
    )
    for step in (login, executables, installer):
        assert step["if"] == SIGNING_GUARD, step.get("name")
    assert login["uses"].startswith("azure/login@")
    for step in (executables, installer):
        assert step["uses"].startswith("azure/artifact-signing-action@")
        assert step["with"]["files-folder-filter"] == "exe"
        assert step["with"]["timestamp-rfc3161"].startswith("http://timestamp")
        for key in ("client-id", "client-secret", "azure-client-secret"):
            assert key not in step["with"], "no secret: the job's OIDC identity signs"
    # In order: sign in, sign what was built and tested, build the installer
    # from the signed executables, sign the installer, then record the state.
    order = [
        names.index("The frozen server serves"),
        names.index("Sign in to Azure for signing"),
        names.index("Sign the executables"),
        names.index("Build the installer"),
        names.index("Sign the installer"),
        names.index("Record whether the installer is signed"),
        names.index("The installer installs, the installed app answers"),
    ]
    assert order == sorted(order), names


def test_the_build_records_the_signing_state_and_refuses_a_bad_signature() -> None:
    record = _step("windows-installer", "Record whether the installer is signed")
    assert "if" not in record, "the record is written on every build, signed or not"
    script = record["run"]
    assert "Get-AuthenticodeSignature" in script
    assert '-ne "Valid"' in script and "throw" in script
    assert "openknowledge.exe" in script and "OpenKnowledgeApp.exe" in script, (
        "the executables inside the installer are checked too"
    )
    assert "SIGNING.txt" in script and "Not code-signed" in script and "Signed by" in script
    uploads = [
        s for s in _package_steps("windows-installer") if "upload-artifact" in s.get("uses", "")
    ]
    assert any("SIGNING.txt" in s["with"]["path"] for s in uploads), (
        "the record travels with the installer"
    )


def test_the_downloaded_installer_is_checked_against_the_record() -> None:
    name = "The downloaded installer carries the signature the build recorded"
    check = _step("windows-e2e", name)
    assert "SIGNING.txt" in check["run"] and "Get-AuthenticodeSignature" in check["run"]
    names = [s.get("name") for s in _package_steps("windows-e2e")]
    launch = "Install, launch, download models, ask, ask again"
    assert names.index(name) < names.index(launch), "checked before it is installed"


def test_the_release_notes_say_whether_the_installer_is_signed() -> None:
    script = _scripts("publish")
    assert "SIGNING.txt" in script
    assert "**Code signing:** $signing" in script
    assert '"Signed by"*)' in script, "the install paragraph follows the recorded state"
    assert "not yet" in script and "code-signed; if SmartScreen still warns" in script
