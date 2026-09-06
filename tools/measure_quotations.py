"""How models quote, and how often the quotation is real.

v0.12.10 shipped a correct answer carrying a fabricated citation. Asked who
approves a contract of exactly EUR 25,000, the model answered "Head of
Department" - right - and supported it with what reads as a row of the policy
table::

    "Contract value (annual): Above EUR 25,000 | Approver: Chief Financial
     Officer | Additional requirement: Three competitive quotes"

No such row exists. "Above EUR 25,000" occurs in the document as prose about
competitive quotes and is never paired with an approver; the answer welded it
to a different row. The grounding gate measures word and number overlap, so
every element being real somewhere in the document is all it can see.

The obvious fix - reject an answer whose quoted span is not in the evidence -
carries an obvious risk: models quote loosely. They collapse whitespace, drop
markdown, elide with "...", and join table cells with pipes or commas. A check
that demands byte equality would reject honest answers, and this project has
reverted a change for costing one refusal before.

So this measures first. Over every cached answer this repository has kept from
its own evaluation runs, paired with the documents that answer actually cited:

  - how often an answer quotes at all;
  - how many quoted spans are found verbatim, and under how much normalisation;
  - which spans are found under no normalisation at all - the candidates for
    fabrication, printed in full so they can be read rather than counted.

The normalisation ladder is the output that matters. Each rung is a claim
about how a faithful quotation may differ from its source, and the number of
spans it recovers is the evidence for putting it in the gate.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sqlite3
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from openknowledge.connectors.local_files import document_id_for  # noqa: E402
from openknowledge.documents import parse_file  # noqa: E402

#: A quoted span: straight or curly double quotes, not spanning a blank line.
#: Single quotes are excluded - apostrophes make them ambiguous, and no model
#: in the sample used them to quote a source.
_QUOTED = re.compile(r'["“]([^"“”]{1,600}?)["”]', re.DOTALL)

#: Below this a "quotation" is a word or a phrase being mentioned, not a claim
#: about what a document says. Measured across the sample rather than assumed:
#: the tool reports the distribution so the floor can be argued about.
_MIN_WORDS = 4


def quoted_spans(answer: str, min_words: int = _MIN_WORDS) -> list[str]:
    out = []
    for raw in _QUOTED.findall(answer):
        span = raw.strip()
        if len(span.split()) >= min_words and "\n\n" not in span:
            out.append(span)
    return out


# --- the normalisation ladder ------------------------------------------------
# Each rung is cumulative and each is a separate claim about faithful quoting.


def rung_exact(s: str) -> str:
    return s


def rung_space(s: str) -> str:
    """Whitespace collapsed. A quotation wrapped across lines is still a
    quotation, and every model in the sample rewraps."""
    return re.sub(r"\s+", " ", s).strip()


def rung_markdown(s: str) -> str:
    """Emphasis markers dropped. Sources are Markdown; answers quote the words
    and routinely keep or drop the ``**`` around them independently."""
    return rung_space(re.sub(r"[*_`]+", "", s))


def rung_case(s: str) -> str:
    return rung_markdown(s).lower()


def rung_punct(s: str) -> str:
    """Curly quotes, dashes and the separators that join table cells folded
    together. A Markdown table row reads ``| a | b |``; an answer quoting it
    writes ``a | b``, ``a, b`` or ``a - b``."""
    s = rung_case(s)
    s = s.translate(str.maketrans({"‘": "'", "’": "'", "–": "-", "—": "-"}))
    s = re.sub(r"\s*[|,;:]\s*", " ", s)
    s = re.sub(r"\s*-\s*", " ", s)
    return re.sub(r"\s+", " ", s).strip()


RUNGS = [
    ("exact", rung_exact),
    ("whitespace", rung_space),
    ("markdown", rung_markdown),
    ("case", rung_case),
    ("separators", rung_punct),
]


def load_corpus(root: pathlib.Path) -> dict[str, str]:
    """document_id -> text, through the product's own parser."""
    out: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name.startswith("."):
            continue
        try:
            parsed = parse_file(path)
        except Exception:
            continue
        out[document_id_for(path.relative_to(root))] = parsed.text
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--min-words", type=int, default=_MIN_WORDS)
    ap.add_argument("--show", type=int, default=25, help="unmatched spans to print")
    args = ap.parse_args()

    repo = pathlib.Path(__file__).resolve().parents[1]
    scratch = pathlib.Path(
        "/tmp/claude-0/-home-user-OpenKnowledge/0e4a2357-a370-57ae-8cd3-5f947fe4a419/scratchpad"
    )
    # Data directories whose corpus is known, so an answer can be paired with
    # the documents it was actually allowed to see.
    pairs = [
        ("*rules*", repo / "evals/corpus/aveline"),
        ("*aveline*", repo / "evals/corpus/aveline"),
        ("ftr-*", repo / "evals/golden-ftr/documents"),
        ("inj-state", repo / "evals/golden-injection/documents"),
        ("*golden-data", repo / "documents"),
    ]

    corpora = {}
    for _, root in pairs:
        if root not in corpora:
            corpora[root] = load_corpus(root)
            if not corpora[root]:
                print(f"refusing to run: no documents parsed under {root}", file=sys.stderr)
                return 2

    answers = 0
    with_quotes = 0
    spans_total = 0
    found_at: dict[str, int] = {name: 0 for name, _ in RUNGS}
    unmatched: list[tuple[str, str]] = []

    seen: set[tuple[str, str]] = set()
    for pattern, root in pairs:
        for db in sorted(scratch.glob(f"**/{pattern}/openknowledge.db")):
            try:
                con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
                rows = list(con.execute("select answer, citations from answer_cache"))
            except sqlite3.Error:
                continue
            for answer, citations in rows:
                if not answer:
                    continue
                key = (str(db), answer[:120])
                if key in seen:
                    continue
                seen.add(key)
                answers += 1
                spans = quoted_spans(answer, args.min_words)
                if not spans:
                    continue
                with_quotes += 1
                try:
                    cited = {c["document_id"] for c in json.loads(citations or "[]")}
                except (ValueError, TypeError, KeyError):
                    cited = set()
                evidence = "\n".join(
                    text for did, text in corpora[root].items() if not cited or did in cited
                )
                prepared = {name: fn(evidence) for name, fn in RUNGS}
                for span in spans:
                    spans_total += 1
                    hit = None
                    for name, fn in RUNGS:
                        if fn(span) and fn(span) in prepared[name]:
                            hit = name
                            break
                    if hit:
                        found_at[hit] += 1
                    else:
                        unmatched.append((span, ",".join(sorted(cited)) or "(uncited)"))

    print("How models quote, measured over this repository's own evaluation runs")
    print()
    print(f"  answers examined                {answers}")
    print(
        f"  answers containing a quotation  {with_quotes}  ({with_quotes / answers:.1%})"
        if answers
        else ""
    )
    print(f"  quoted spans (>= {args.min_words} words)         {spans_total}")
    print()
    if spans_total:
        print("  first rung that finds the span in the cited documents:")
        running = 0
        for name, _ in RUNGS:
            running += found_at[name]
            print(f"    {name:12s} {found_at[name]:5d}   cumulative {running / spans_total:6.1%}")
        print(f"    {'NOT FOUND':12s} {len(unmatched):5d}   {len(unmatched) / spans_total:6.1%}")
    print()
    if unmatched:
        print(f"  spans found at no rung - read these, do not count them ({args.show} shown):")
        for span, cited in unmatched[: args.show]:
            print(f"    [{cited}] {rung_space(span)[:200]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
