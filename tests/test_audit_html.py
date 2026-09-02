"""The audit as a page: the same findings, forwardable, and nothing else in it.

Held to four things. It says what the text report says - both documents, both
sentences, the figure marked. Everything from the documents is escaped, since
the page will be opened by the person the finding is about. It fetches
nothing: no script, no stylesheet, no image. And the same folder gives the
same bytes, because the audit promised that and a page with a date on it
would break the promise.
"""

from __future__ import annotations

import re
from html.parser import HTMLParser
from pathlib import Path

from tests.test_audit import AGREEING, CONTRADICTING, POLICY, write

from openknowledge.audit import audit_folder
from openknowledge.audit_html import render_html
from openknowledge.cli import main

ROOT = Path(__file__).resolve().parents[1]


class _Parses(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.tags: list[str] = []
        self.srcs: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.tags.append(tag)
        for name, value in attrs:
            if name in ("src", "href") and value:
                self.srcs.append(value)
            if tag == "link" and name == "rel" and value == "stylesheet":
                self.srcs.append("stylesheet")


def _parsed(page: str) -> _Parses:
    parser = _Parses()
    parser.feed(page)
    return parser


def test_a_finding_shows_both_sentences_with_the_figures_marked(tmp_path: Path) -> None:
    root = write(tmp_path / "docs", expenses=POLICY, travel=CONTRADICTING)
    page = render_html(audit_folder(root))

    assert "1 contradiction, in 1 document pair." in page
    assert "expenses vs travel" in page or "travel vs expenses" in page
    assert "<mark>EUR 500</mark>" in page and "<mark>EUR 1,000</mark>" in page
    # The whole sentence, so the finding is checkable without opening a file -
    # and the figure marked inside it, not only in the "says" line above it.
    assert page.count("requires prior approval from a line manager") >= 2
    assert "above <mark>EUR 500</mark> requires" in page
    assert "above <mark>EUR 1,000</mark> requires" in page
    assert "figure · " in page, "the kind is named the way the text report names it"
    assert "<h2>Duplicated documents</h2>" not in page


def test_everything_from_a_document_is_escaped(tmp_path: Path) -> None:
    """The page is opened by the people the finding is about; a document that
    says <script> must not become a script in their browser."""
    hostile = POLICY.replace(
        "Travel above EUR 500", "Travel <script>alert(1)</script> above EUR 500"
    )
    root = write(tmp_path / "docs", expenses=hostile, travel=CONTRADICTING)
    page = render_html(audit_folder(root))
    assert "<script>alert(1)</script>" not in page
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in page
    assert "<script" not in page.lower().replace("<style>", "")


def test_the_page_fetches_nothing(tmp_path: Path) -> None:
    root = write(tmp_path / "docs", expenses=POLICY, travel=CONTRADICTING)
    page = render_html(audit_folder(root))
    parsed = _parsed(page)
    assert "script" not in parsed.tags and "img" not in parsed.tags and "link" not in parsed.tags
    external = [s for s in parsed.srcs if s.startswith(("http://", "https://", "//"))]
    assert external == ["https://github.com/FedericoTs/OpenKnowledge"], external
    assert "style" in parsed.tags, "the styling is inline, or there is none"


def test_a_clean_folder_says_so(tmp_path: Path) -> None:
    root = write(tmp_path / "docs", expenses=POLICY, summary=AGREEING)
    page = render_html(audit_folder(root))
    assert 'class="verdict clean"' in page
    assert "No contradictions found between these 2 documents." in page
    assert "Where the documents disagree" not in page


def test_files_that_contributed_nothing_are_listed_with_the_reason(tmp_path: Path) -> None:
    root = write(tmp_path / "docs", expenses=POLICY, travel=CONTRADICTING)
    (root / "logo.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    page = render_html(audit_folder(root))
    assert "1 file contributed nothing" in page
    assert "<code>logo.png</code>" in page and "no parser" in page


def test_the_same_folder_gives_the_same_bytes(tmp_path: Path) -> None:
    root = write(tmp_path / "docs", expenses=POLICY, travel=CONTRADICTING)
    first = render_html(audit_folder(root))
    second = render_html(audit_folder(root))
    assert first == second
    # No date anywhere: a page stamped with the day it was made is a page
    # that differs from yesterday's for no reason a reader would care about.
    assert not re.search(r"\b20\d\d-\d\d-\d\d\b", first)
    assert "enerated" not in first


def test_the_contract_corpus_shows_a_duplicated_pair() -> None:
    """Real data: the aveline set carries two versions of one expenses policy,
    and the page has to say so rather than list twenty-four contradictions."""
    page = render_html(audit_folder(ROOT / "evals" / "corpus" / "aveline"))
    assert "<h2>Duplicated documents</h2>" in page
    assert "look like two versions of the same document" in page
    assert 'class="verdict found"' in page


def test_the_cli_writes_the_page_and_keeps_its_exit_code(tmp_path: Path, capsys) -> None:
    root = write(tmp_path / "docs", expenses=POLICY, travel=CONTRADICTING)
    out = tmp_path / "report.html"

    code = main(["audit", str(root), "--html", str(out)])
    assert code == 1, "findings still exit 1 so the audit can gate CI"
    assert out.is_file()
    assert "<mark>EUR 500</mark>" in out.read_text(encoding="utf-8")
    printed = capsys.readouterr().out
    assert "OpenKnowledge audit -" in printed, "the text report is still printed"
    assert f"Wrote {out}" in printed

    assert main(["audit", str(root), "--html", str(out), "--exit-zero"]) == 0


def test_the_cli_can_put_the_page_on_stdout_and_nothing_else(tmp_path: Path, capsys) -> None:
    root = write(tmp_path / "docs", expenses=POLICY, travel=CONTRADICTING)
    main(["audit", str(root), "--html", "-", "--exit-zero"])
    printed = capsys.readouterr().out
    assert printed.startswith("<!doctype html>")
    assert "OpenKnowledge audit -" not in printed, "nothing but the page, so it can be redirected"
    assert "Wrote" not in printed
