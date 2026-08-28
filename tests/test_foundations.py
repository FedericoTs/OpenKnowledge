"""State location, asset resolution, and the desktop-mode admin token.

These exist because "works from the checkout" was quietly load-bearing
everywhere: state was CWD-relative (meaningless for a double-clicked app),
the web UI was found at Path(__file__).parents[3] (a wheel served "Chat
widget not found" while every test passed), and the admin API was dead on
any install where nobody hand-sets an environment variable.

Several tests here pin scenarios an adversarial review constructed against
the first version of this code - each one says which.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from openknowledge import paths as paths_module
from openknowledge.assets import find_asset
from openknowledge.config import Settings, load_settings
from openknowledge.paths import state_paths

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def app_mode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """An environment that resolves to app mode, portably.

    The first version of these tests monkeypatched LOCALAPPDATA and asserted
    the path moved - which passes on Linux and fails on real Windows, where
    platformdirs resolves the folder through the shell API and only falls back
    to the environment as a last resort. The Windows CI leg this repo added
    would have been red on its very first run, on a test from the same commit.
    So app mode is simulated where it is decided: platformdirs itself.
    """
    state_root = tmp_path / "per-user-appdata" / "OpenKnowledge"
    monkeypatch.delenv("OK_STATE_DIR", raising=False)
    monkeypatch.setattr(paths_module.platformdirs, "user_data_dir", lambda *a, **k: str(state_root))
    empty = tmp_path / "empty-cwd"
    empty.mkdir()
    monkeypatch.chdir(empty)
    return state_root


# --- where state lives ------------------------------------------------------


def test_an_openknowledge_deployment_is_recognised_by_its_own_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("OK_STATE_DIR", raising=False)
    cases = {
        "env-with-ok-keys": lambda d: (d / ".env").write_text("OK_LOCAL_MODEL=qwen3:8b\n"),
        "own-database": lambda d: (
            (d / "data").mkdir(),
            (d / "data" / "openknowledge.db").write_text(""),
        ),
        "paired-install-dirs": lambda d: ((d / "data").mkdir(), (d / "documents").mkdir()),
        "source-checkout": lambda d: (d / "src" / "openknowledge").mkdir(parents=True),
    }
    for name, plant in cases.items():
        cwd = tmp_path / name
        cwd.mkdir()
        plant(cwd)
        monkeypatch.chdir(cwd)
        assert state_paths().mode == "project", name


def test_a_strangers_project_directory_is_not_a_deployment(
    app_mode: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An adversarial-review scenario, pinned.

    The first marker set accepted any `.env`, any `pyproject.toml`, or
    anything named `data` - so cd-ing into an unrelated Python or Node repo
    silently switched which corpus, token and config every command saw, and
    an `openknowledge index` typed there would have scattered state into a
    stranger's project.
    """
    stranger = tmp_path / "some-node-service"
    stranger.mkdir()
    (stranger / ".env").write_text("DATABASE_URL=postgres://x\nNODE_ENV=production\n")
    (stranger / "pyproject.toml").write_text('[project]\nname = "their-tool"\n')
    (stranger / "data").write_text("a plain FILE named data")
    monkeypatch.chdir(stranger)

    assert state_paths().mode == "app", "hijacked by a stranger's project files"


def test_an_empty_directory_means_per_user_app_state(app_mode: Path) -> None:
    """The double-clicked case: no deployment here, so state must not scatter
    across whatever folder was current at launch."""
    state = state_paths()
    assert state.mode == "app"
    assert state.root == app_mode
    assert state.root != Path.cwd()


def test_ok_state_dir_overrides_everything(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OK_STATE_DIR", str(tmp_path / "forced"))
    (tmp_path / ".env").write_text("OK_LOCAL_MODEL=x\n")  # would force project mode
    monkeypatch.chdir(tmp_path)
    state = state_paths()
    assert state.mode == "override"
    assert state.root == tmp_path / "forced"


def test_a_deleted_working_directory_is_app_mode_not_a_traceback(
    app_mode: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Adversarial-review scenario: a removed worktree or cleaned tmp dir made
    every settings-loading command die in Path.cwd()'s FileNotFoundError."""

    def gone() -> Path:
        raise FileNotFoundError(2, "No such file or directory")

    monkeypatch.setattr(paths_module.Path, "cwd", staticmethod(gone))
    state = state_paths()
    assert state.mode == "app"
    assert state.root == app_mode


def test_resolving_paths_creates_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """`audit` promises to write nothing, and it resolves paths too."""
    monkeypatch.setenv("OK_STATE_DIR", str(tmp_path / "never-created"))
    state = state_paths()
    _ = state.env_file, state.data_dir, state.documents_dir
    assert not (tmp_path / "never-created").exists()


def test_load_settings_relocates_only_genuine_defaults(
    app_mode: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("OK_DATA_DIR", raising=False)
    monkeypatch.delenv("OK_DOCUMENTS_DIR", raising=False)

    settings = load_settings()
    assert settings.data_dir == str(app_mode / "data")
    assert settings.documents_dir == str(app_mode / "documents")

    # Anything the operator set always wins over relocation.
    monkeypatch.setenv("OK_DATA_DIR", str(app_mode / "mine"))
    assert load_settings().data_dir == str(app_mode / "mine")


def test_load_settings_reads_the_state_directorys_env_file(
    app_mode: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`model use` writes there; the next start has to read the same file."""
    app_mode.mkdir(parents=True)
    (app_mode / ".env").write_text("OK_LOCAL_MODEL=written-by-model-use\n")
    monkeypatch.delenv("OK_LOCAL_MODEL", raising=False)
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
    assert found.read_text(encoding="utf-8") == "<!-- frozen copy -->"


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
    app_mode: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fastapi.testclient import TestClient

    from openknowledge.api.app import create_app

    monkeypatch.delenv("OK_ADMIN_TOKEN", raising=False)
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
    assert f"OK_ADMIN_TOKEN={token}" in (app_mode / ".env").read_text(encoding="utf-8")
    assert load_settings().admin_token == token


@pytest.mark.skipif(os.name == "nt", reason="POSIX file modes")
def test_the_minted_token_file_is_owner_only(app_mode: Path, monkeypatch) -> None:
    """Adversarial-review scenario: under umask 022 the token file was 0644,
    so any local account could read the bearer token and drive the loopback
    admin API. 127.0.0.1 is shared by every user on the machine."""
    from fastapi.testclient import TestClient

    from openknowledge.api.app import create_app

    monkeypatch.delenv("OK_ADMIN_TOKEN", raising=False)
    settings = load_settings()
    with TestClient(create_app(settings)):
        pass

    mode = (app_mode / ".env").stat().st_mode
    assert mode & 0o077 == 0, f"token file is group/world accessible: {oct(mode)}"


def test_an_existing_token_in_the_state_file_wins_over_minting(
    app_mode: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two processes starting together must converge, not diverge."""
    from fastapi.testclient import TestClient

    from openknowledge.api.app import create_app

    app_mode.mkdir(parents=True)
    (app_mode / ".env").write_text("OK_OTHER=kept\nOK_ADMIN_TOKEN=already-minted\n")
    monkeypatch.delenv("OK_ADMIN_TOKEN", raising=False)

    settings = load_settings()
    with TestClient(create_app(settings)):
        pass
    assert settings.admin_token == "already-minted"
    content = (app_mode / ".env").read_text(encoding="utf-8")
    assert content.count("OK_ADMIN_TOKEN=") == 1
    assert "OK_OTHER=kept" in content


def test_project_mode_keeps_the_fail_closed_stance(tmp_path: Path) -> None:
    """On a server, admin-disabled-until-configured is a security decision.
    Minting a token there would silently enable a write surface."""
    from fastapi.testclient import TestClient

    from openknowledge.api.app import create_app

    settings = Settings(data_dir=str(tmp_path / "data"), _env_file=None)  # type: ignore[call-arg]
    with TestClient(create_app(settings)) as client:
        assert client.get("/admin/config").status_code == 503
    assert not settings.admin_token


def test_an_ok_state_dir_override_also_stays_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Adversarial-review scenario: an operator pointing OK_STATE_DIR at
    /srv/openknowledge is running a server; minting there would silently
    enable every admin write behind their back."""
    from fastapi.testclient import TestClient

    from openknowledge.api.app import create_app

    monkeypatch.setenv("OK_STATE_DIR", str(tmp_path / "srv"))
    monkeypatch.delenv("OK_ADMIN_TOKEN", raising=False)
    monkeypatch.chdir(tmp_path)

    settings = load_settings()
    with TestClient(create_app(settings)) as client:
        assert client.get("/admin/config").status_code == 503
    assert not settings.admin_token
    assert not (tmp_path / "srv" / ".env").exists(), "a token was minted in override mode"


# --- private env writes -----------------------------------------------------


@pytest.mark.skipif(os.name == "nt", reason="POSIX file modes")
def test_a_private_env_write_is_never_world_readable(tmp_path: Path) -> None:
    from openknowledge.models import write_env

    target = tmp_path / ".env"
    write_env(target, {"OK_ADMIN_TOKEN": "secret"}, private=True)
    assert target.stat().st_mode & 0o077 == 0

    # And the write is atomic: no temp litter left beside it.
    write_env(target, {"OK_OTHER": "x"}, private=True)
    assert [p.name for p in tmp_path.iterdir()] == [".env"]
    assert "OK_ADMIN_TOKEN=secret" in target.read_text(encoding="utf-8")


# --- the bind and the Host header -------------------------------------------


def test_tls_is_both_paths_or_neither(tmp_path: Path) -> None:
    """Half a certificate pair is a misconfiguration said plainly at start,
    never a server that came up insecurely."""
    from openknowledge.cli import _tls_kwargs
    from openknowledge.config import Settings

    def settings(**kw: str) -> Settings:
        return Settings(_env_file=None, **kw)  # type: ignore[call-arg]

    assert _tls_kwargs(settings()) == {}

    cert = tmp_path / "server.crt"
    key = tmp_path / "server.key"
    cert.write_text("not really a cert")
    key.write_text("not really a key")
    assert _tls_kwargs(settings(tls_cert=str(cert), tls_key=str(key))) == {
        "ssl_certfile": str(cert),
        "ssl_keyfile": str(key),
    }

    with pytest.raises(SystemExit, match="together"):
        _tls_kwargs(settings(tls_cert=str(cert)))
    with pytest.raises(SystemExit, match="not a file"):
        _tls_kwargs(settings(tls_cert=str(cert), tls_key=str(tmp_path / "missing.key")))


def test_serve_binds_this_machine_only_by_default() -> None:
    """0.0.0.0 was the old default: LAN exposure and a firewall prompt for a
    personal tool. The container, which genuinely serves a network, passes
    0.0.0.0 explicitly in its Dockerfile CMD."""
    from openknowledge.cli import build_parser

    args = build_parser().parse_args(["serve"])
    assert args.host == "127.0.0.1"
    assert 'serve", "--host", "0.0.0.0"' in (ROOT / "Dockerfile").read_text(encoding="utf-8")


def test_a_rebound_hostname_is_rejected_on_loopback(tmp_path: Path) -> None:
    """Adversarial-review scenario: DNS rebinding points an attacker's domain
    at 127.0.0.1 and a webpage then reads the private corpus. The Host header
    survives the trick, so loopback serving accepts only loopback names."""
    from fastapi.testclient import TestClient

    from openknowledge.api.app import create_app

    settings = Settings(data_dir=str(tmp_path / "data"), _env_file=None)  # type: ignore[call-arg]
    with TestClient(create_app(settings)) as client:
        assert client.get("/healthz").status_code == 200  # Host: testserver
        rebound = client.get("/healthz", headers={"Host": "attacker.example"})
        assert rebound.status_code == 400

    # A deliberate network bind keeps full reachability - the middleware is
    # for the personal-machine case, not a new constraint on deployments.
    lan = Settings(
        data_dir=str(tmp_path / "data2"),
        bind_host="0.0.0.0",
        _env_file=None,  # type: ignore[call-arg]
    )
    with TestClient(create_app(lan)) as client:
        assert client.get("/healthz", headers={"Host": "intranet.corp"}).status_code == 200
