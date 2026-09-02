"""The audit as one page, for forwarding.

The text report is written to be pasted into an email. This is the same
findings laid out for the other thing people do with a finding: send the
file, or put it on a screen. Every contradiction shows both documents and
both sentences with the disagreeing figure marked, because the point is not
that a tool flagged something - it is that two of the reader's own sentences
cannot both be true.

Self-contained on purpose: no script, no stylesheet fetched from anywhere, no
image. The page is proof about documents that never left the machine, and it
should be able to say so without phoning anywhere itself. It carries no
timestamp either: the audit promises byte-identical output on the same
folder, and this keeps the promise.
"""

from __future__ import annotations

import html

from . import __version__
from . import graph as knowledge_graph
from .audit import _KIND_LABEL, AuditReport

PROJECT_URL = "https://github.com/FedericoTs/OpenKnowledge"
#: Past this many, a list of unreadable files stops being read.
MAX_UNREADABLE = 200

_STYLE = """
:root { color-scheme: light dark; --ink:#1d2b26; --muted:#5f6f69; --line:#d9e1dd; --bg:#f7f8f6;
        --card:#ffffff; --good:#2f6b4f; --bad:#a23b2c; --mark:#fff2a8; }
@media (prefers-color-scheme: dark) {
  :root { --ink:#e6ebe8; --muted:#9aa8a2; --line:#2f3b36; --bg:#141a17; --card:#1c2420;
          --good:#7fc7a4; --bad:#f0907f; --mark:#5a4d12; } }
* { box-sizing: border-box; }
body { margin:0; background:var(--bg); color:var(--ink); font:15px/1.5 system-ui, -apple-system,
       "Segoe UI", Roboto, sans-serif; }
main { max-width: 880px; margin: 0 auto; padding: 36px 20px 60px; }
.kicker { text-transform: uppercase; letter-spacing: .08em; font-size: 12px; color: var(--muted);
          margin: 0 0 6px; }
h1 { font-size: 22px; margin: 0 0 4px; word-break: break-all; }
h2 { font-size: 16px; margin: 34px 0 12px; }
h3 { font-size: 15px; margin: 0 0 10px; display:flex; gap:10px; align-items:baseline;
     flex-wrap:wrap; }
.counts { color: var(--muted); margin: 0 0 22px; }
.verdict { border-radius: 10px; padding: 16px 18px; font-size: 18px; font-weight: 650;
           background: var(--card); border: 1px solid var(--line); }
.verdict.found { border-left: 6px solid var(--bad); }
.verdict.clean { border-left: 6px solid var(--good); }
.verdict small { display:block; font-size: 13.5px; font-weight: 400; color: var(--muted);
                 margin-top: 4px; }
article.finding { background: var(--card); border: 1px solid var(--line); border-radius: 10px;
                  padding: 16px 18px; margin: 0 0 14px; }
.num { display:inline-block; min-width: 26px; height: 26px; line-height: 26px; text-align:center;
       border-radius: 999px; background: var(--bad); color: #fff; font-size: 13px;
       font-weight: 700; }
.kind { color: var(--muted); font-weight: 400; font-size: 13px; }
.sides { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 14px; }
.side { border-top: 1px solid var(--line); padding-top: 10px; }
.doc { margin: 0 0 4px; font-weight: 650; }
.doc code { font-weight: 400; color: var(--muted); font-size: 12.5px; }
.says { margin: 0 0 8px; }
blockquote { margin: 0; padding: 8px 12px; border-left: 3px solid var(--line); color: var(--ink);
             background: var(--bg); border-radius: 0 8px 8px 0; }
mark { background: var(--mark); color: inherit; padding: 0 3px; border-radius: 3px; }
code { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 13px; }
ul { padding-left: 20px; } li { margin: 4px 0; }
.note, footer { color: var(--muted); font-size: 13.5px; }
footer { margin-top: 40px; border-top: 1px solid var(--line); padding-top: 14px; }
footer a { color: inherit; }
@media print { body { background: #fff; } article.finding, .verdict { break-inside: avoid; } }
"""

_NOT_CHECKED = (
    "a document contradicting itself: two figures in one file are usually a rule and its "
    "exception, and flagging those trains you to ignore this",
    "rules stated without a must / may / must not marker, or not in English",
    "scanned pages: there is no OCR, so an image of a policy reads as nothing",
)


def _n(count: int, noun: str, plural: str | None = None) -> str:
    return f"{count} {noun if count == 1 else (plural or noun + 's')}"


def _quote(sentence: str, raw: str) -> str:
    """The sentence, escaped, with the first occurrence of the figure marked."""
    text = html.escape(" ".join(sentence.split()))
    figure = html.escape(raw)
    if figure and figure in text:
        text = text.replace(figure, f"<mark>{figure}</mark>", 1)
    return text


def render_html(report: AuditReport) -> str:
    e = html.escape
    contradicting = report.contradicting
    found = sum(len(p.conflicts) for p in contradicting)
    out: list[str] = []
    out.append("<!doctype html>")
    out.append('<html lang="en"><head><meta charset="utf-8">')
    out.append('<meta name="viewport" content="width=device-width, initial-scale=1">')
    out.append(f"<title>Knowledge audit - {e(report.root)}</title>")
    out.append(f"<style>{_STYLE}</style></head><body><main>")
    out.append('<p class="kicker">OpenKnowledge audit</p>')
    out.append(f"<h1>{e(report.root)}</h1>")
    out.append(
        f'<p class="counts">{_n(report.documents, "document")} · '
        f"{_n(report.claims_checked, 'claim')} checked · 0 model calls · $0.00</p>"
    )

    if report.clean:
        out.append(
            '<div class="verdict clean">No contradictions found between these '
            f"{_n(report.documents, 'document')}."
            "<small>Every figure and every stated rule that two documents share agrees.</small>"
            "</div>"
        )
    else:
        pairs = _n(len(contradicting), "document pair")
        extra = (
            f" {_n(len(report.variants), 'pair')} of documents look like two versions of one file."
            if report.variants
            else ""
        )
        headline = (
            f"{_n(found, 'contradiction')}, in {pairs}."
            if found
            else "No contradictions between distinct documents."
        )
        out.append(
            f'<div class="verdict found">{headline}'
            f"<small>Each one below is two of these documents' own sentences that cannot both be "
            f"true.{e(extra)}</small></div>"
        )

    if found:
        out.append("<h2>Where the documents disagree</h2>")
        number = 0
        for pair in contradicting:
            for conflict in pair.conflicts:
                number += 1
                kind = _KIND_LABEL.get(conflict.kind, conflict.kind)
                out.append('<article class="finding">')
                versus = f"{e(conflict.left.document_id)} vs {e(conflict.right.document_id)}"
                match = f"{e(kind)} · {conflict.overlap:.0%} context match"
                out.append(
                    f'<h3><span class="num">{number}</span><span>{versus}</span>'
                    f'<span class="kind">{match}</span></h3>'
                )
                out.append('<div class="sides">')
                for side in (conflict.left, conflict.right):
                    out.append('<div class="side">')
                    title, ident = e(side.document_title), e(side.document_id)
                    out.append(f'<p class="doc">{title} <code>{ident}</code></p>')
                    out.append(f'<p class="says">says <mark>{e(side.raw)}</mark></p>')
                    out.append(f"<blockquote>{_quote(side.sentence, side.raw)}</blockquote>")
                    out.append("</div>")
                out.append("</div></article>")

    if report.graph is not None and report.graph.nodes:
        out.append("<h2>The documents, and what connects them</h2>")
        out.append(
            '<p class="note">Every readable document is a circle, sized by how many figures and '
            "rules it states and coloured by folder; a hollow one says it is retired. A red line "
            "is a contradiction above, a dashed red line two versions of one file, a grey arrow "
            "a document retiring the one it names. Islands are documents nothing else touches. "
            "Nothing here is inferred.</p>"
        )
        positions = knowledge_graph.layout(report.graph)
        out.append(knowledge_graph.render_svg(report.graph, positions, weight_word="states"))

    if report.variants:
        out.append("<h2>Duplicated documents</h2>")
        for pair in report.variants:
            out.append(f"<p>{e(pair.describe())}</p>")
        out.append(
            '<p class="note">Listed separately because reconciling them one figure at a time is '
            "the wrong job. The right one is deciding which copy stands.</p>"
        )

    if report.unreadable:
        out.append(f"<h2>{_n(len(report.unreadable), 'file')} contributed nothing</h2><ul>")
        for skipped in report.unreadable[:MAX_UNREADABLE]:
            out.append(f"<li><code>{e(skipped.path)}</code>: {e(skipped.reason)}</li>")
        if len(report.unreadable) > MAX_UNREADABLE:
            out.append(f"<li>and {len(report.unreadable) - MAX_UNREADABLE} more</li>")
        out.append("</ul>")

    out.append("<h2>What this does not check</h2><ul>")
    for item in _NOT_CHECKED:
        out.append(f"<li>{e(item)}</li>")
    out.append("</ul>")

    out.append(
        f"<footer>Produced by OpenKnowledge {e(__version__)} with "
        "<code>openknowledge audit</code>. Nothing was written and no document left this "
        "machine; the same folder gives this same page, byte for byte, every time. "
        f'<a href="{PROJECT_URL}">{PROJECT_URL}</a></footer>'
    )
    out.append("</main></body></html>")
    return "\n".join(out) + "\n"
