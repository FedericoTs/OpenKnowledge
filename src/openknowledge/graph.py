"""The documents as a map, drawn from facts the stores already hold.

Obsidian's graph draws the links a person typed. This one draws nothing a
person did not already establish elsewhere in this product: a red line is an
open contradiction the claims pipeline found, a dashed red line two files
that are versions of one document, a grey arrow a document retiring the one
it names as superseded, a green line two documents that answered the same
question together, an orange square a question people asked that the
documents could not answer. Circles are documents, sized by how often real
answers cited them (or, in the audit, by how many figures and rules they
state), coloured by folder, hollow when the document says it is retired.

Nothing is inferred, so nothing here can be wrong about the documents; it
can only be a worse or better picture of the same facts. The layout is a
plain force-directed one, seeded, computed here rather than in the browser,
so the same corpus gives the same picture every time and the page needs no
script to show it.
"""

from __future__ import annotations

import html
import math
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urlparse

import numpy as np

from .knowledge.supersession import announced_by
from .retrieval.base import Document

WIDTH = 960
HEIGHT = 600
#: Folder colours, muted so the red of a contradiction stays the loudest thing.
PALETTE = ("#2f6b4f", "#3f6fa3", "#8a6d3b", "#6c5b9c", "#b0623c", "#2f8a8a", "#7a7a3a", "#9c4f6c")
GAP_COLOUR = "#c47a1c"
EDGE_ORDER = ("cocitation", "supersession", "duplicate", "contradiction")


@dataclass(frozen=True, slots=True)
class Node:
    id: str
    label: str
    kind: str  # "document" | "gap"
    folder: str = ""
    weight: int = 0
    superseded: bool = False


@dataclass(frozen=True, slots=True)
class Edge:
    source: str
    target: str
    kind: str  # "contradiction" | "duplicate" | "supersession" | "cocitation"
    weight: int = 1


@dataclass(frozen=True, slots=True)
class Graph:
    nodes: tuple[Node, ...]
    edges: tuple[Edge, ...]

    @property
    def documents(self) -> tuple[Node, ...]:
        return tuple(n for n in self.nodes if n.kind == "document")


def folder_of(url: str | None, root: str | Path | None) -> str:
    """The folder a document is filed under, relative to the corpus root.

    Documents carry their path as a file URI; the folder is what the map
    colours by. Empty for the root itself, or when the path cannot be placed
    under the root (a document supplied as bare text, say).
    """
    if not url:
        return ""
    parsed = urlparse(url)
    if parsed.scheme not in ("file", ""):
        return ""
    path = Path(unquote(parsed.path))
    if root is not None:
        try:
            return path.parent.resolve().relative_to(Path(root).resolve()).as_posix().strip(".")
        except (ValueError, OSError):
            pass
    return path.parent.name


def _pair(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a <= b else (b, a)


def _supersession_edges(documents: Sequence[Document], present: set[str]) -> list[Edge]:
    return [
        Edge(source=announcer, target=target, kind="supersession")
        for target, announcer in sorted(announced_by(documents).items())
        if target in present and announcer in present
    ]


def from_audit(
    documents: Sequence[Document],
    pairs: Iterable[object],
    *,
    root: str | Path | None,
    claims: Mapping[str, int],
) -> Graph:
    """The audit's map: every readable document, joined where it disagrees.

    ``pairs`` are the audit's DocumentPair groupings - a variant pair becomes
    one dashed line, any other pair one red line weighted by how many figures
    disagree. Supersession comes from the documents' own headers.
    """
    nodes = tuple(
        Node(
            id=d.document_id,
            label=d.title or d.document_id,
            kind="document",
            folder=folder_of(d.url, root),
            weight=int(claims.get(d.document_id, 0)),
            superseded=d.superseded,
        )
        for d in documents
    )
    present = {n.id for n in nodes}
    edges: list[Edge] = []
    for pair in pairs:
        left, right = pair.left, pair.right  # type: ignore[attr-defined]
        if left not in present or right not in present:
            continue
        kind = "duplicate" if pair.is_variant else "contradiction"  # type: ignore[attr-defined]
        edges.append(Edge(*_pair(left, right), kind=kind, weight=len(pair.conflicts)))  # type: ignore[attr-defined]
    edges.extend(_supersession_edges(documents, present))
    return Graph(nodes=nodes, edges=tuple(edges))


def from_engine(
    documents: Sequence[Document],
    *,
    root: str | Path | None,
    conflicts: Iterable[object],
    citations: Iterable[Sequence[str]],
    gaps: Iterable[Mapping[str, object]],
    viewer: frozenset[str] | None,
) -> Graph:
    """The running server's map, as one viewer may see it.

    A document the viewer could not open is not drawn, and neither is any
    line to it: the same rule retrieval applies, so the map cannot leak a
    title the answers would withhold. ``conflicts`` are the open ones the
    knowledge store holds; ``citations`` are the document-id sets of cached
    and pinned answers; ``gaps`` are the unanswered questions.
    """
    visible = [
        d
        for d in documents
        if viewer is None or not d.allowed_principals or d.allowed_principals & viewer
    ]
    present = {d.document_id for d in visible}

    cited: Counter[str] = Counter()
    together: Counter[tuple[str, str]] = Counter()
    for answer in citations:
        ids = sorted({i for i in answer if i in present})
        cited.update(ids)
        for a_index, a in enumerate(ids):
            for b in ids[a_index + 1 :]:
                together[(a, b)] += 1

    nodes: list[Node] = [
        Node(
            id=d.document_id,
            label=d.title or d.document_id,
            kind="document",
            folder=folder_of(d.url, root),
            weight=cited.get(d.document_id, 0),
            superseded=d.superseded,
        )
        for d in visible
    ]
    if viewer is None:
        # Questions are nobody's document: shown only on the unrestricted
        # view, since a curator seeing everything is the audience for them.
        for gap in gaps:
            question = str(gap.get("question", ""))
            asked = gap.get("asked", 0)
            nodes.append(
                Node(
                    id=f"gap:{question}",
                    label=question,
                    kind="gap",
                    weight=int(asked) if isinstance(asked, int | float) else 0,
                )
            )

    disagree: Counter[tuple[str, str]] = Counter()
    for conflict in conflicts:
        left = conflict.left_document  # type: ignore[attr-defined]
        right = conflict.right_document  # type: ignore[attr-defined]
        if left in present and right in present and left != right:
            disagree[_pair(left, right)] += 1

    edges: list[Edge] = []
    for (a, b), n in sorted(together.items()):
        edges.append(Edge(a, b, kind="cocitation", weight=n))
    edges.extend(_supersession_edges(visible, present))
    for (a, b), n in sorted(disagree.items()):
        edges.append(Edge(a, b, kind="contradiction", weight=n))
    return Graph(nodes=tuple(nodes), edges=tuple(edges))


def iterations_for(n: int) -> int:
    """How many passes the simulation gets: fewer as the corpus grows.

    Each pass is an n-by-n numpy sweep, so the cost is quadratic in documents
    and linear in passes. Measured here: a thousand documents at 300 passes
    took 17 s, which is longer than a page should wait. Small corpora keep
    the full run, large ones settle in fewer passes and look the same from a
    distance; the count is a function of n alone, so one corpus always gets
    one picture.
    """
    if n <= 150:
        return 300
    if n <= 600:
        return 200
    return 120


def layout(
    graph: Graph,
    *,
    width: int = WIDTH,
    height: int = HEIGHT,
    seed: int = 0,
    iterations: int | None = None,
) -> dict[str, tuple[float, float]]:
    """Positions for every node, in pixels, from a seeded force-directed run.

    Fruchterman-Reingold in the unit square: every pair repels, every edge
    attracts, a weak pull to the centre keeps islands on the page, and the
    temperature falls so the picture settles. Folders start on a circle so
    a corpus with few edges still clusters by where its files live; gaps
    start along the bottom and are held there. Same seed, same graph, same
    picture - on the same machine; floating point across platforms can move
    a node by less than a pixel.
    """
    nodes = graph.nodes
    n = len(nodes)
    if n == 0:
        return {}
    if iterations is None:
        iterations = iterations_for(n)
    rng = np.random.default_rng(seed)
    index = {node.id: i for i, node in enumerate(nodes)}
    folders = sorted({node.folder for node in nodes if node.kind == "document"})
    angle = {f: 2 * math.pi * k / max(len(folders), 1) for k, f in enumerate(folders)}

    # float32 halves the memory the n-by-n sweep moves each pass, and a
    # position rounded to a tenth of a pixel does not know the difference.
    pos = np.zeros((n, 2), dtype=np.float32)
    is_gap = np.zeros(n, dtype=bool)
    for i, node in enumerate(nodes):
        if node.kind == "gap":
            is_gap[i] = True
            pos[i] = (rng.uniform(0.1, 0.9), rng.uniform(0.86, 0.94))
        else:
            a = angle[node.folder]
            centre = (0.5 + 0.28 * math.cos(a), 0.45 + 0.28 * math.sin(a))
            pos[i] = np.asarray(centre) + rng.normal(0.0, 0.05, 2)

    k = np.float32(0.75 / math.sqrt(n))
    centre_pull = np.array([0.5, 0.45], dtype=np.float32)
    gap_band = np.array([0.0, 0.9], dtype=np.float32)
    gap_strength = np.array([0.0, 0.4], dtype=np.float32)
    src = np.array([index[e.source] for e in graph.edges if e.kind != "gap"], dtype=int)
    dst = np.array([index[e.target] for e in graph.edges], dtype=int)
    temperature = 0.1
    eye = np.eye(n, dtype=bool)
    for _ in range(iterations):
        delta = pos[:, None, :] - pos[None, :, :]
        dist = np.sqrt((delta**2).sum(axis=2))
        dist[eye] = np.inf
        dist = np.maximum(dist, 1e-6)
        repulse = (k * k) / dist
        disp = (delta / dist[:, :, None] * repulse[:, :, None]).sum(axis=1)
        if len(src):
            d = pos[src] - pos[dst]
            length = np.maximum(np.sqrt((d**2).sum(axis=1)), 1e-6)
            pull = (d / length[:, None]) * ((length**2) / k)[:, None]
            np.add.at(disp, src, -pull)
            np.add.at(disp, dst, pull)
        disp += (centre_pull - pos) * np.float32(0.08)
        disp[is_gap] += (gap_band - pos[is_gap]) * gap_strength
        step = np.maximum(np.sqrt((disp**2).sum(axis=1)), 1e-9)
        pos += disp / step[:, None] * np.minimum(step, temperature)[:, None]
        pos = np.clip(pos, 0.03, 0.97).astype(np.float32)
        temperature = max(temperature * 0.96, 0.002)

    # Room for a label on either side, and the legend's rows at the bottom,
    # so no document is laid out under the key that explains it.
    margin_x, margin_top = 110.0, 30.0
    margin_bottom = 30.0 + 14.0 * _legend_rows(graph)
    xs = margin_x + pos[:, 0] * (width - 2 * margin_x)
    ys = margin_top + pos[:, 1] * (height - margin_top - margin_bottom)
    return {
        node.id: (round(float(xs[i]), 1), round(float(ys[i]), 1)) for i, node in enumerate(nodes)
    }


def _radius(weight: int) -> float:
    return min(4.0 + 3.0 * math.sqrt(max(weight, 0)), 18.0)


def _legend_rows(graph: Graph) -> int:
    folders = {n.folder for n in graph.documents}
    kinds = {e.kind for e in graph.edges}
    gaps = any(n.kind == "gap" for n in graph.nodes)
    return min(len(folders), 8) + len(kinds) + (1 if gaps else 0)


def render_svg(
    graph: Graph,
    positions: Mapping[str, tuple[float, float]],
    *,
    width: int = WIDTH,
    height: int = HEIGHT,
    weight_word: str = "cited in",
) -> str:
    """One SVG, self-contained, every string from a document escaped.

    ``weight_word`` is how a node's size is explained in its tooltip: the
    audit sizes by claims ("states"), the server by citations ("cited in").
    """
    e = html.escape
    folders = sorted({n.folder for n in graph.documents})
    colour = {f: PALETTE[i % len(PALETTE)] for i, f in enumerate(folders)}
    kinds_present = [k for k in EDGE_ORDER if any(edge.kind == k for edge in graph.edges)]

    out: list[str] = []
    out.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="100%" role="img" aria-label="The documents and what connects them" '
        f'class="knowledge-map">'
    )
    out.append(
        "<style>"
        ".knowledge-map .edge{fill:none;stroke-linecap:round}"
        ".knowledge-map .edge.cocitation{stroke:#2f6b4f;stroke-opacity:.45}"
        ".knowledge-map .edge.supersession{stroke:#7a8a84;stroke-width:1.5}"
        ".knowledge-map .edge.duplicate{stroke:#a23b2c;stroke-width:2;stroke-dasharray:5 4}"
        ".knowledge-map .edge.contradiction{stroke:#a23b2c;stroke-width:2.5}"
        ".knowledge-map .node.superseded{fill:none;stroke-dasharray:3 2;stroke-width:1.5}"
        ".knowledge-map .label{font:11px system-ui,sans-serif;fill:#3d4a45}"
        ".knowledge-map .legend{font:11px system-ui,sans-serif;fill:#5f6f69}"
        "</style>"
    )
    out.append(
        '<defs><marker id="knowledge-map-arrow" viewBox="0 0 10 10" refX="9" refY="5" '
        'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
        '<path d="M 0 0 L 10 5 L 0 10 z" fill="#7a8a84"/></marker></defs>'
    )

    for kind in EDGE_ORDER:  # quietest first, so a contradiction is drawn on top
        for edge in graph.edges:
            if edge.kind != kind or edge.source not in positions or edge.target not in positions:
                continue
            (x1, y1), (x2, y2) = positions[edge.source], positions[edge.target]
            marker = ' marker-end="url(#knowledge-map-arrow)"' if kind == "supersession" else ""
            words = {
                "contradiction": f"{edge.weight} open contradiction(s)",
                "duplicate": f"two versions of one document, {edge.weight} figures apart",
                "supersession": "retired by",
                "cocitation": f"answered {edge.weight} question(s) together",
            }[kind]
            out.append(
                f'<line class="edge {kind}" x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}"{marker}>'
                f"<title>{e(edge.source)} — {e(edge.target)}: {e(words)}</title></line>"
            )

    show_labels = len(graph.nodes) <= 40
    threshold = 0
    if not show_labels and graph.documents:
        weights = sorted(n.weight for n in graph.documents)
        threshold = weights[int(len(weights) * 0.85)]
    for node in graph.nodes:
        if node.id not in positions:
            continue
        x, y = positions[node.id]
        if node.kind == "gap":
            side = _radius(node.weight) * 1.6
            tip = f"asked {node.weight} time(s), not answered: {node.label}"
            out.append(
                f'<rect class="node gap" x="{x - side / 2:.1f}" y="{y - side / 2:.1f}" '
                f'width="{side:.1f}" height="{side:.1f}" rx="2" fill="{GAP_COLOUR}">'
                f"<title>{e(tip)}</title></rect>"
            )
            continue
        r = _radius(node.weight)
        fill = colour.get(node.folder, PALETTE[0])
        cls = "node superseded" if node.superseded else "node"
        where = f" · in {node.folder}" if node.folder else ""
        retired = " · says it is superseded" if node.superseded else ""
        tip = f"{node.label}{where} · {weight_word} {node.weight}{retired}"
        out.append(
            f'<circle class="{cls}" cx="{x}" cy="{y}" r="{r:.1f}" fill="{fill}" stroke="{fill}">'
            f"<title>{e(tip)}</title></circle>"
        )
        if show_labels or node.weight >= threshold and node.weight > 0:
            label = node.label if len(node.label) <= 28 else node.label[:27] + "…"
            if x > width * 0.62:
                anchor = f'x="{x - r - 3:.1f}" text-anchor="end"'
            else:
                anchor = f'x="{x + r + 3:.1f}"'
            out.append(f'<text class="label" {anchor} y="{y + 4:.1f}">{e(label)}</text>')

    has_gaps = any(n.kind == "gap" for n in graph.nodes)
    y = height - 12 - 14 * _legend_rows(graph)
    for folder in folders[:8]:
        out.append(f'<circle cx="18" cy="{y}" r="5" fill="{colour[folder]}"/>')
        out.append(f'<text class="legend" x="30" y="{y + 4}">{e(folder or "top folder")}</text>')
        y += 14
    words_for = {
        "contradiction": "documents that disagree",
        "duplicate": "two versions of one document",
        "supersession": "retired by (arrow points at the retired one)",
        "cocitation": "answered a question together",
    }
    for kind in kinds_present:
        out.append(f'<line class="edge {kind}" x1="10" y1="{y}" x2="26" y2="{y}"/>')
        out.append(f'<text class="legend" x="30" y="{y + 4}">{words_for[kind]}</text>')
        y += 14
    if has_gaps:
        out.append(f'<rect x="13" y="{y - 5}" width="10" height="10" rx="2" fill="{GAP_COLOUR}"/>')
        out.append(f'<text class="legend" x="30" y="{y + 4}">asked, not answered</text>')
    out.append("</svg>")
    return "\n".join(out)
