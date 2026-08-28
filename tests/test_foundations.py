"""State location, asset resolution, and the desktop-mode admin token.

These exist because "works from the checkout" was quietly load-bearing
everywhere: state was CWD-relative (meaningless for a double-clicked app),
the web UI was found at Path(__file__).parents[3] (a wheel served "Chat
widget not found" while every test passed), and the admin API was dead on
any install where nobody hand-sets an environment variable.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from openknowledge.assets import find_asset
from openknowledge.config import Settings, load_settings
from openknowledge.paths import state_paths

ROOT = Path(__file__).resolve().parent.parent


# --- where state lives ------------------------------------------------------


def test_a_directory_with_a_deployment_stays_project_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every existing install must behave exactly as before this module."""
    monkeypatch.delenv("OK_STATE_DIR", raising=False)
    (tmp_path / ".env").write_text("")
    monkeypatch.chdir(tmp_path)
    state = state_paths()
    assert state.mode == "project"
    assert state.root == tmp_path


@pytest.mark.parametrize("marker", ["data", ".env.example", "pyproject.toml"])
def test_other_deployment_markers_also_mean_project_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, marker: str
) -> None:
    monkeypatch.delenv("OK_STATE_DIR", raising=False)
    target = tmp_path / marker
    target.mkdir() if marker == "data" else target.write_text("")
    monkeypatch.chdir(tmp_path)
    assert state_paths().mode == "project"


def test_an_empty_directory_means_per_user_app_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The double-clicked case: no deployment here, so state must not scatter
    across whatever folder was current at launch."""
    monkeypatch.delenv("OK_STATE_DIR", raising=False)
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "lad"))  # the Windows spelling
    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.chdir(empty)

    state = state_paths()
    assert state.mode == "app"
    assert str(tmp_path) in str(state.root), "app state must not be CWD-relative"
    assert state.root.name in ("OpenKnowledge", "openknowledge")


def test_ok_state_dir_overrides_everything(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OK_STATE_DIR", str(tmp_path / "forced"))
    (tmp_path / ".env").write_text("")  # would otherwise force project mode
    monkeypatch.chdir(tmp_path)
    state = state_paths()
    assert state.mode == "override"
    assert state.root == tmp_path / "forced"


def test_resolving_paths_creates_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """`audit` promises to write nothing, and it resolves paths too."""
    monkeypatch.setenv("OK_STATE_DIR", str(tmp_path / "never-created"))
    state = state_paths()
    _ = state.env_file, state.data_dir, state.documents_dir
    assert not (tmp_path / "never-created").exists()


def test_load_settings_relocates_only_genuine_defaults(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OK_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.delenv("OK_DATA_DIR", raising=False)
    monkeypatch.delenv("OK_DOCUMENTS_DIR", raising=False)
    monkeypatch.chdir(tmp_path)

    settings = load_settings()
    assert settings.data_dir == str(tmp_path / "state" / "data")
    assert settings.documents_dir == str(tmp_path / "state" / "documents")

    # Anything the operator set always wins over relocation.
    monkeypatch.setenv("OK_DATA_DIR", str(tmp_path / "mine"))
    assert load_settings().data_dir == str(tmp_path / "mine")


def test_load_settings_reads_the_state_directorys_env_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`model use` writes there; the next start has to read the same file."""
    state = tmp_path / "state"
    state.mkdir()
    (state / ".env").write_text("OK_LOCAL_MODEL=written-by-model-use\n")
    monkeypatch.setenv("OK_STATE_DIR", str(state))
    monkeypatch.delenv("OK_LOCAL_MODEL", raising=False)
    monkeypatch.chdir(tmp_path)
    assert load_settings().local_model == "written-by-model-use"


# --- where the web UI is found ----------------------------------------------


def test_assets_resolve_in_a_checkout() -> None:
    widget = find_asset("widget/index.html")
    assert widget is not None and widget.is_file()
    assert find_asset("site/index.html") is not None
    assert find_asset("no/such/asset.html") is None


def test_a_frozen_bundle_wins_over_the_checkout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """PyInstaller sets sys._MEIPASS; a frozen build must serve its own copy."""
    bundle = tmp_path / "bundle" / "web" / "widget"
    bundle.mkdir(parents=True)
    (bundle / "index.html").write_text("<!-- frozen copy -->")
    monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path / "bundle"), raising=False)

    found = find_asset("widget/index.html")
    assert found is not None
    assert found.read_text() == "<!-- frozen copy -->"


@pytest.mark.skipif(shutil.which("uv") is None, reason="uv is not installed")
def test_the_wheel_ships_the_web_ui(tmp_path: Path) -> None:
    """A wheel without the widget serves 'Chat widget not found' - the failure
    every checkout-run test was structurally unable to see."""
    subprocess.run(
        ["uv", "build", "--wheel", "-o", str(tmp_path)],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    names = zipfile.ZipFile(next(tmp_path.glob("*.whl"))).namelist()
    for shipped in (
        "openknowledge/web/widget/index.html",
        "openknowledge/web/site/index.html",
        "openknowledge/pricing.yaml",
    ):
        assert shipped in names, f"the wheel is missing {shipped}"


# --- the desktop admin token ------------------------------------------------


def test_app_mode_mints_and_persists_an_admin_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fastapi.testclient import TestClient

    from openknowledge.api.app import create_app

    monkeypatch.setenv("OK_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.delenv("OK_ADMIN_TOKEN", raising=False)
    monkeypatch.chdir(tmp_path)

    settings = load_settings()
    assert not settings.admin_token
    with TestClient(create_app(settings)) as client:
        token = settings.admin_token
        assert token, "no token was minted in app mode"
        assert (
            client.get("/admin/config", headers={"Authorization": f"Bearer {token}"}).status_code
            == 200
        )
        assert client.get("/admin/config").status_code == 401

    # Persisted: the next start reads the same token rather than minting anew.
    assert f"OK_ADMIN_TOKEN={token}" in (tmp_path / "state" / ".env").read_text()
    assert load_settings().admin_token == token


def test_project_mode_keeps_the_fail_closed_stance(tmp_path: Path) -> None:
    """On a server, admin-disabled-until-configured is a security decision.
    Minting a token there would silently enable a write surface."""
    from fastapi.testclient import TestClient

    from openknowledge.api.app import create_app

    settings = Settings(data_dir=str(tmp_path / "data"), _env_file=None)  # type: ignore[call-arg]
    with TestClient(create_app(settings)) as client:
        assert client.get("/admin/config").status_code == 503
    assert not settings.admin_token


# --- the bind default -------------------------------------------------------


def test_serve_binds_this_machine_only_by_default() -> None:
    """0.0.0.0 was the old default: LAN exposure and a firewall prompt for a
    personal tool. The container, which genuinely serves a network, passes
    0.0.0.0 explicitly in its Dockerfile CMD."""
    from openknowledge.cli import build_parser

    args = build_parser().parse_args(["serve"])
    assert args.host == "127.0.0.1"
    assert 'serve", "--host", "0.0.0.0"' in (ROOT / "Dockerfile").read_text()
