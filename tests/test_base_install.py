"""The server must start on a base install, without the optional extras.

The container installs ``.[anthropic,opendataloader]`` and nothing else, so
anything the app imports at module load must be in the core dependencies. The
Teams channel needs PyJWT to validate a Bot Service token, PyJWT is the
``auth`` extra, and importing it at the top of ``api.app`` made the container
exit at startup with an ImportError - which the docker job caught and the
whole unit suite did not, because this machine has every extra installed.

Run in a subprocess with the optional modules blocked, which is the only way
to reproduce a base install from inside a full one.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

#: What the core dependencies do not include. Add to this when an extra is
#: added, not when a test fails.
OPTIONAL = ("jwt", "anthropic", "opendataloader_pdf", "playwright", "uvicorn")

_PROGRAM = """
import sys

class _NoExtras:
    def find_module(self, name, path=None):
        return self.find_spec(name, path)

    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in {blocked!r}:
            raise ImportError(f"{{name}} is not installed in a base install")
        return None

sys.meta_path.insert(0, _NoExtras())

from openknowledge.api.app import create_app
from openknowledge.config import Settings

app = create_app(
    Settings(
        data_dir={data!r},
        documents_dir={documents!r},
        local_enabled=False,
        embedding_enabled=False,
        escalation_enabled=False,
        _env_file=None,
    )
)
print("created", len(app.routes) > 10)
"""


def test_the_app_can_be_built_without_the_optional_extras(tmp_path) -> None:
    documents = tmp_path / "documents"
    documents.mkdir()
    (documents / "handbook.md").write_text(
        "# Handbook\nThe office closes at 18:00.", encoding="utf-8"
    )
    program = textwrap.dedent(_PROGRAM).format(
        blocked=set(OPTIONAL), data=str(tmp_path / "data"), documents=str(documents)
    )
    result = subprocess.run(
        [sys.executable, "-c", program], capture_output=True, text=True, timeout=180
    )
    assert result.returncode == 0, (
        "the app cannot be built on a base install:\n"
        f"{result.stdout[-2000:]}\n{result.stderr[-2000:]}"
    )
    assert "created True" in result.stdout


def test_the_blocker_actually_blocks() -> None:
    """A test that proves nothing is worse than no test: if the import hook
    quietly let everything through, the check above would pass on any code."""
    program = textwrap.dedent(_PROGRAM.split("from openknowledge")[0]).format(
        blocked={"jwt"}, data="", documents=""
    )
    result = subprocess.run(
        [sys.executable, "-c", program + "\nimport jwt\n"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode != 0
    assert "not installed in a base install" in result.stderr
