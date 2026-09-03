"""The map: documents and what connects them, drawn from facts and nothing else.

Every edge has to come from something another part of the product already
established - a contradiction the claims pipeline found, a supersession a
document declared, two documents cited by one answer - and a document the
viewer could not open must not appear, nor any line to it. The layout has
to be the same every time, and the SVG has to be safe to inline: escaped,
self-contained, nothing fetched.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from fastapi.testclient import TestClient
from tests.test_knowledge_gaps import _boot_paths

from openknowledge.api.app import create_app
from openknowledge.audit import audit_folder
from openknowledge.audit_html import render_html
from openknowledge.config import Settings
from openknowledge.graph import (
    HEIGHT,
    WIDTH,
    Edge,
    Graph,
    Node,
    folder_of,
    from_audit,
    from_engine,
    layout,
    render_svg,
)
from openknowledge.retrieval.base import Document

ROOT = Path(__file__).resolve().parents[1]


def _doc(
    ident: str,
    title: str,
    folder: str = "",
    *,
    text: str = "Some words.",
    principals: frozenset[str] = frozenset(),
    superseded: bool = False,
) -> Document:
    where = f"/corpus/{folder}/{ident}.md" if folder else f"/corpus/{ident}.md"
    return Document(
        document_id=ident,
        title=title,
        text=text,
        url=f"file://{where}",
        allowed_principals=principals,
        superseded=superseded,
    )


@dataclass(frozen=True)
class _Pair:
    """What from_audit reads off an audit DocumentPair."""

    left: str
    right: str
    conflicts: tuple[int, ...]
    is_variant: bool


@dataclass(frozen=True)
class _Conflict:
    left_document: str
    right_document: str


# -- building -------------------------------------------------------------------


def test_the_audit_map_joins_disagreeing_documents_and_marks_versions() -> None:
    docs = [
        _doc("hr-expenses", "Expenses Policy", "hr"),
        _doc("hr-travel", "Travel Guidelines", "hr"),
        _doc("archive-expenses-2023", "Expenses Policy (2023)", "archive"),
        _doc("security-vpn", "VPN Access", "security"),
    ]
    pairs = [
        _Pair("hr-expenses", "hr-travel", conflicts=(1,), is_variant=False),
        _Pair("archive-expenses-2023", "hr-expenses", conflicts=(1,) * 24, is_variant=True),
    ]
    graph = from_audit(docs, pairs, root="/corpus", claims={"hr-expenses": 30, "hr-travel": 4})

    assert [n.id for n in graph.nodes] == [d.document_id for d in docs]
    by_id = {n.id: n for n in graph.nodes}
    assert by_id["hr-expenses"].folder == "hr" and by_id["hr-expenses"].weight == 30
    assert by_id["security-vpn"].weight == 0, "a document with no claims is drawn, small"
    kinds = {(e.source, e.target): (e.kind, e.weight) for e in graph.edges}
    assert kinds[("hr-expenses", "hr-travel")] == ("contradiction", 1)
    assert kinds[("archive-expenses-2023", "hr-expenses")] == ("duplicate", 24)
    assert len(graph.edges) == 2, "nothing joins the VPN document to anything"


def test_supersession_declared_in_a_header_becomes_an_arrow() -> None:
    current = _doc(
        "hr-expenses",
        "Expenses Policy",
        "hr",
        text="# Expenses Policy\n\nSupersedes: Expenses Policy 2023\n\nTravel above EUR 500...",
    )
    retired = _doc("archive-expenses-2023", "Expenses Policy 2023", "archive", superseded=True)
    graph = from_audit([current, retired], [], root="/corpus", claims={})
    (edge,) = graph.edges
    assert edge == Edge("hr-expenses", "archive-expenses-2023", kind="supersession")
    assert {n.id: n.superseded for n in graph.nodes} == {
        "hr-expenses": False,
        "archive-expenses-2023": True,
    }


def test_the_server_map_sizes_by_citations_and_joins_co_cited_documents() -> None:
    docs = [_doc("a", "A"), _doc("b", "B"), _doc("c", "C")]
    graph = from_engine(
        docs,
        root="/corpus",
        conflicts=[_Conflict("b", "c"), _Conflict("c", "b")],
        citations=[("a", "b"), ("b", "a"), ("a",), ("zzz-not-a-document", "a")],
        gaps=[{"question": "how many sick days", "asked": 7}],
        viewer=None,
    )
    weights = {n.id: n.weight for n in graph.nodes}
    assert weights["a"] == 4 and weights["b"] == 2 and weights["c"] == 0
    assert weights["gap:how many sick days"] == 7
    assert [n.kind for n in graph.nodes].count("gap") == 1
    edges = {(e.source, e.target, e.kind): e.weight for e in graph.edges}
    assert edges[("a", "b", "cocitation")] == 2, "cited together twice, whatever the order"
    assert edges[("b", "c", "contradiction")] == 2, "two open conflicts, one line, weight two"
    assert len(edges) == 2


def test_a_viewer_sees_only_their_documents_and_no_line_out_of_them() -> None:
    docs = [
        _doc("public", "Handbook"),
        _doc("hr-secret", "Redundancy Plan", "hr", principals=frozenset({"group:hr"})),
    ]
    everyone = from_engine(
        docs,
        root="/corpus",
        conflicts=[_Conflict("public", "hr-secret")],
        citations=[("public", "hr-secret")],
        gaps=[{"question": "q", "asked": 1}],
        viewer=None,
    )
    assert {n.id for n in everyone.nodes} == {"public", "hr-secret", "gap:q"}
    assert len(everyone.edges) == 2

    outsider = from_engine(
        docs,
        root="/corpus",
        conflicts=[_Conflict("public", "hr-secret")],
        citations=[("public", "hr-secret")],
        gaps=[{"question": "q", "asked": 1}],
        viewer=frozenset({"group:sales"}),
    )
    assert [n.id for n in outsider.nodes] == ["public"], "no title leaks, and no gaps either"
    assert outsider.edges == (), "a line to a hidden document would name it"

    member = from_engine(
        docs,
        root="/corpus",
        conflicts=[],
        citations=[],
        gaps=[],
        viewer=frozenset({"group:hr"}),
    )
    assert {n.id for n in member.nodes} == {"public", "hr-secret"}


def test_folder_of_is_relative_to_the_corpus_root() -> None:
    assert folder_of("file:///corpus/hr/leave.md", "/corpus") == "hr"
    assert folder_of("file:///corpus/hr/2024/leave.md", "/corpus") == "hr/2024"
    assert folder_of("file:///corpus/leave.md", "/corpus") == ""
    assert folder_of("file:///corpus/hr/a%20b/leave.md", "/corpus") == "hr/a b"
    assert folder_of("https://sharepoint.example/x", "/corpus") == ""
    assert folder_of(None, "/corpus") == ""


# -- layout -----------------------------------------------------------------------


def test_the_layout_is_the_same_every_time_and_pulls_connected_documents_together() -> None:
    """Thirty documents in four folders, six of them joined in pairs across
    folders. Repulsion alone leaves a joined pair as far apart as any other;
    only the attraction along the edge brings them in, so the joined pairs
    have to end up much closer than the crowd."""
    folders = ("hr", "finance", "security", "legal")
    nodes = tuple(
        Node(f"d{i}", f"Document {i}", "document", folder=folders[i % 4]) for i in range(30)
    )
    joined = [("d0", "d1"), ("d2", "d3"), ("d4", "d5"), ("d6", "d7"), ("d8", "d9"), ("d10", "d11")]
    graph = Graph(nodes=nodes, edges=tuple(Edge(a, b, kind="contradiction") for a, b in joined))

    first = layout(graph)
    assert first == layout(graph)
    for x, y in first.values():
        assert 0 <= x <= WIDTH and 0 <= y <= HEIGHT

    def dist(a: str, b: str) -> float:
        (x1, y1), (x2, y2) = first[a], first[b]
        return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

    ids = [n.id for n in nodes]
    crowd = [(a, b) for i, a in enumerate(ids) for b in ids[i + 1 :] if (a, b) not in joined]
    crowd_mean = sum(dist(a, b) for a, b in crowd) / len(crowd)
    joined_mean = sum(dist(a, b) for a, b in joined) / len(joined)
    assert joined_mean < crowd_mean / 3, (joined_mean, crowd_mean)


def test_the_passes_fall_as_the_corpus_grows() -> None:
    from openknowledge.graph import iterations_for

    assert iterations_for(12) == 300
    assert iterations_for(1000) < iterations_for(100) < 400
    assert iterations_for(1000) >= 60, "too few passes and nothing settles"


def test_an_empty_graph_lays_out_to_nothing_and_draws_a_valid_svg() -> None:
    graph = Graph(nodes=(), edges=())
    assert layout(graph) == {}
    svg = render_svg(graph, {})
    assert svg.startswith("<svg") and svg.rstrip().endswith("</svg>")


# -- drawing ------------------------------------------------------------------------


def _small_graph() -> Graph:
    nodes = (
        Node("a", "Expenses <b>Policy</b>", "document", folder="hr", weight=5),
        Node("b", "Travel", "document", folder="hr", weight=1),
        Node("old", "Expenses 2023", "document", folder="archive", weight=0, superseded=True),
        Node("gap:sick days", "how many sick days", "gap", weight=3),
    )
    edges = (
        Edge("a", "b", kind="contradiction", weight=1),
        Edge("a", "old", kind="supersession"),
        Edge("a", "b", kind="cocitation", weight=4),
    )
    return Graph(nodes=nodes, edges=edges)


def test_the_svg_escapes_marks_every_kind_and_fetches_nothing() -> None:
    graph = _small_graph()
    svg = render_svg(graph, layout(graph))

    assert "<b>Policy</b>" not in svg and "Expenses &lt;b&gt;Policy&lt;/b&gt;" in svg
    assert 'class="edge contradiction"' in svg and 'class="edge supersession"' in svg
    assert 'class="edge cocitation"' in svg
    assert 'class="node superseded"' in svg, "a retired document is drawn hollow"
    assert 'class="node gap"' in svg and "asked 3 time(s), not answered" in svg
    assert "cited in 5" in svg, "the tooltip says what the size means"
    assert "documents that disagree" in svg, "the legend names the red line"
    assert re.search(r"(src|href)=", svg) is None, "nothing is fetched from anywhere"
    assert "<script" not in svg
    assert svg == render_svg(graph, layout(graph))


def test_the_audit_page_carries_the_map_of_the_contract_corpus() -> None:
    report = audit_folder(ROOT / "evals" / "corpus" / "aveline")
    assert report.graph is not None
    assert len(report.graph.nodes) == report.documents
    page = render_html(report)
    assert "<h2>The documents, and what connects them</h2>" in page
    assert page.count("<svg") == 1
    assert 'class="edge contradiction"' in page and 'class="edge duplicate"' in page
    assert "states " in page, "the audit sizes by what a document states"
    assert render_html(report) == page


# -- the route and the page ---------------------------------------------------------

TOKEN = "t0ken"
AUTH = {"Authorization": f"Bearer {TOKEN}"}


def _client(tmp_path: Path) -> TestClient:
    docs = tmp_path / "documents"
    (docs / "hr").mkdir(parents=True)
    (docs / "hr" / "leave.md").write_text(
        "# Parental Leave\nEmployees get 20 weeks fully paid.", encoding="utf-8"
    )
    (docs / "vpn.md").write_text("# VPN\nConnect to vpn.internal with MFA.", encoding="utf-8")
    settings = Settings(
        data_dir=str(tmp_path / "data"),
        documents_dir=str(docs),
        admin_token=TOKEN,
        local_enabled=False,
        escalation_enabled=False,
        _env_file=None,  # type: ignore[call-arg]
    )
    return TestClient(create_app(settings))


def test_the_route_draws_the_indexed_corpus_as_svg(tmp_path: Path) -> None:
    with _client(tmp_path) as c:
        c.post(
            "/admin/pins",
            headers=AUTH,
            json={"question": "how much leave?", "answer": "20 weeks.", "cite": ["hr-leave"]},
        )
        response = c.get("/admin/graph.svg", headers=AUTH)
    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("image/svg+xml")
    assert response.headers["cache-control"] == "no-store"
    svg = response.text
    assert svg.startswith("<svg") and "Parental Leave" in svg and "VPN" in svg
    assert "cited in 1" in svg, "the pin's citation sizes the document it cites"
    assert 'class="legend"' in svg and ">hr<" in svg, "folders are in the legend"


def test_the_picture_is_drawn_once_until_something_changes(tmp_path: Path, monkeypatch) -> None:
    """The layout is the cost. Two looks at an unchanged corpus lay it out
    once; a new citation is a change, and gets a new picture."""
    from openknowledge.api import app as app_module

    drawn: list[int] = []
    real = app_module.knowledge_graph.layout

    def counting(graph, **kw):  # type: ignore[no-untyped-def]
        drawn.append(len(graph.nodes))
        return real(graph, **kw)

    monkeypatch.setattr(app_module.knowledge_graph, "layout", counting)
    with _client(tmp_path) as c:
        first = c.get("/admin/graph.svg", headers=AUTH).text
        second = c.get("/admin/graph.svg", headers=AUTH).text
        assert first == second and drawn == [2]
        c.post(
            "/admin/pins",
            headers=AUTH,
            json={"question": "vpn?", "answer": "MFA.", "cite": ["vpn"]},
        )
        third = c.get("/admin/graph.svg", headers=AUTH).text
        assert third != first and len(drawn) == 2, "a new citation is a new picture"


def test_every_way_into_the_page_loads_the_map(tmp_path: Path) -> None:
    with _client(tmp_path) as c:
        page = c.get("/manage").text
    assert "async function refreshMap()" in page
    for path, loaded in _boot_paths(page).items():
        assert "refreshMap" in loaded, f"the {path} path does not load the map"
    assert "<h2>The map</h2>" in page and 'id="map"' in page
    assert "Nothing here is inferred" in page
