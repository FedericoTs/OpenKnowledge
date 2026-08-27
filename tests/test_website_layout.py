"""What the page actually lays out, in a browser, at the widths people use.

Every other test here reads the HTML. That is enough to catch a wrong number or
a missing command, and useless against the failures a landing page actually has:
the install command pushing the whole page sideways on a phone, or a font that
404s so the design silently falls back to system faces. Both of those happened
here, and neither was visible from the source.

So this one opens the file in Chromium and measures. It skips when Playwright or
a browser is not installed, because it must never be the reason a contributor
cannot run the suite - `make check` on a bare checkout should still pass.
"""

from __future__ import annotations

import shutil
from collections.abc import Iterator
from pathlib import Path

import pytest

PAGE = Path(__file__).resolve().parent.parent / "web" / "site" / "index.html"

#: Narrow, common, and tablet. 360 is the floor worth supporting.
WIDTHS = (360, 390, 768, 1024, 1440)

#: Elements outside the viewport, ignoring anything inside a scroll container -
#: a wide table that scrolls inside its own box is correct, a wide table that
#: scrolls the page is not.
OVERFLOW = """() => {
  const vw = document.documentElement.clientWidth;
  const scrolls = el => {
    for (let n = el.parentElement; n; n = n.parentElement) {
      const ox = getComputedStyle(n).overflowX;
      if (ox === 'auto' || ox === 'scroll' || ox === 'hidden') return true;
    }
    return false;
  };
  const label = el => el.tagName.toLowerCase() +
    (typeof el.className === 'string' && el.className.trim()
      ? '.' + el.className.trim().split(/\\s+/).join('.') : '');
  const wide = [];
  for (const el of document.querySelectorAll('body *')) {
    if (el.getBoundingClientRect().right > vw + 1 && !scrolls(el)) wide.push(label(el));
  }
  return {
    documentWidth: document.documentElement.scrollWidth,
    viewport: vw,
    culprits: [...new Set(wide)].slice(0, 6),
  };
}"""


def _chromium() -> str | None:
    for candidate in Path("/opt/pw-browsers").glob("chromium-*/chrome-linux/chrome"):
        return str(candidate)
    return shutil.which("chromium") or shutil.which("google-chrome")


@pytest.fixture(scope="module")
def browser() -> Iterator[object]:
    sync_playwright = pytest.importorskip(
        "playwright.sync_api", reason="playwright is not installed"
    ).sync_playwright
    executable = _chromium()
    if executable is None:
        pytest.skip("no chromium available")

    with sync_playwright() as pw:
        launched = pw.chromium.launch(executable_path=executable, args=["--no-sandbox"])
        try:
            yield launched
        finally:
            launched.close()


@pytest.mark.parametrize("width", WIDTHS)
def test_the_page_never_scrolls_sideways(browser: object, width: int) -> None:
    """A page that scrolls sideways on a phone clips its own first paragraph.

    It did: the install command is one unbreakable line, and a grid item's
    min-width defaults to auto, so its track grew to fit and took the document
    with it. Nothing in the HTML said so.
    """
    page = browser.new_page(viewport={"width": width, "height": 900})  # type: ignore[attr-defined]
    try:
        page.goto(PAGE.as_uri(), wait_until="networkidle")
        result = page.evaluate(OVERFLOW)
    finally:
        page.close()

    assert result["documentWidth"] <= result["viewport"] + 1, (
        f"the page is {result['documentWidth']}px wide in a {width}px viewport; "
        f"widened by: {', '.join(result['culprits']) or 'unknown'}"
    )


def test_the_pages_own_typefaces_load(browser: object) -> None:
    """They 404'd once and the page still looked plausible in fallback faces."""
    page = browser.new_page(viewport={"width": 1280, "height": 900})  # type: ignore[attr-defined]
    try:
        page.goto(PAGE.as_uri(), wait_until="networkidle")
        page.wait_for_timeout(300)
        statuses = page.evaluate("() => [...document.fonts].map(f => [f.family, f.status])")
    finally:
        page.close()

    assert statuses, "the page declares no local font"
    failed = [family for family, status in statuses if status != "loaded"]
    assert not failed, f"declared but never loaded: {', '.join(failed)}"


@pytest.mark.parametrize("scheme", ["dark", "light"])
def test_text_and_background_never_collapse_into_each_other(browser: object, scheme: str) -> None:
    """Both themes have to be readable, including inside the terminal.

    The terminal stays dark in both, so it does not get the light theme's
    darker accents - which is easy to forget when adding a colour, and shows up
    as near-black text on black.
    """
    page = browser.new_page(  # type: ignore[attr-defined]
        viewport={"width": 1280, "height": 900}, color_scheme=scheme
    )
    try:
        page.goto(PAGE.as_uri(), wait_until="networkidle")
        samples = page.evaluate(
            """() => {
                const rgb = s => s.match(/\\d+(\\.\\d+)?/g).slice(0, 3).map(Number);
                const lum = ([r, g, b]) => {
                    const f = c => {
                        c /= 255;
                        return c <= .03928 ? c / 12.92 : ((c + .055) / 1.055) ** 2.4;
                    };
                    return .2126*f(r) + .7152*f(g) + .0722*f(b);
                };
                const bg = el => {
                    for (let n = el; n; n = n.parentElement) {
                        const c = getComputedStyle(n).backgroundColor;
                        if (c && !c.startsWith('rgba(0, 0, 0, 0)')) return c;
                    }
                    return 'rgb(255,255,255)';
                };
                const out = {};
                for (const sel of ['h1', '.lede', '.terminal .cmd', '.terminal .hit',
                                   '.terminal pre', 'nav .nav-links a', 'footer p']) {
                    const el = document.querySelector(sel);
                    if (!el) continue;
                    const a = lum(rgb(getComputedStyle(el).color));
                    const b = lum(rgb(bg(el)));
                    out[sel] = (Math.max(a, b) + .05) / (Math.min(a, b) + .05);
                }
                return out;
            }"""
        )
    finally:
        page.close()

    # 3.0 is WCAG's large-text floor. Deliberately not 4.5: several of these are
    # intentionally quiet, and a threshold that fails honest muted text would
    # get raised rather than obeyed.
    poor = {sel: round(ratio, 2) for sel, ratio in samples.items() if ratio < 3.0}
    assert not poor, f"unreadable in {scheme} theme: {poor}"
