"""Published links must point at a ref that exists.

The install command on the site said `.../OpenKnowledge/main/install.sh`. The
repository has no `main` branch, so the first person to copy that line got a
404 - and so did every other link on the page, since they all named the same
branch. Nothing in the suite noticed, because nothing was checking the ref.

This does not reach the network. It checks the one property that made those
links wrong: the ref they name. `HEAD` resolves to whatever the default branch
is, which is right before a merge to main and still right after it, so it is
the only ref a published link should carry.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

#: Files whose links are published to somebody else.
PUBLISHED = (
    "web/site/index.html",
    "README.md",
    "install.sh",
    *(str(p.relative_to(ROOT)) for p in sorted((ROOT / "docs").rglob("*.md"))),
)

#: raw.githubusercontent.com/<owner>/<repo>/<ref>/... and github.com/.../blob/<ref>/...
REFS = re.compile(
    r"raw\.githubusercontent\.com/[\w.-]+/[\w.-]+/([\w.-]+)/"
    r"|github\.com/[\w.-]+/[\w.-]+/(?:blob|tree|raw)/([\w.-]+)/"
)


@pytest.mark.parametrize("name", PUBLISHED, ids=lambda n: Path(n).name)
def test_published_links_name_a_ref_that_resolves(name: str) -> None:
    path = ROOT / name
    if not path.exists():
        pytest.skip(f"{name} is not in this checkout")

    named = {a or b for a, b in REFS.findall(path.read_text(encoding="utf-8"))}
    wrong = named - {"HEAD"}
    assert not wrong, (
        f"{name} links to ref(s) {sorted(wrong)}. Use HEAD: it resolves to the "
        "default branch, so the link works before a merge to main and after it."
    )
