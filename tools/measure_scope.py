#!/usr/bin/env python3
"""How much of a whole-document answer can retrieval even show the model?

Every other measurement in this repository asks whether the system finds the
right fact. This one asks whether it can find *all* of something. "What are
the priorities?", "who are the characters?", "summarise this document" - the
answer is not a sentence sitting in one passage, it is the whole document, and
retrieval hands the model `retrieval_k` chunks of it.

No model is called here, deliberately. This measures the ceiling: of the
passages that carry the answer, how many does search() return? A model cannot
list what it was never shown, so whatever this reports is the best any prompt
or any model could do. Separating the two matters - a 31% ceiling is a
retrieval problem, and no amount of prompt work will move it.

The corpus is evals/golden-ftr/documents: the US Federal Travel Regulation,
public domain, already committed, and written by nobody here. That last part
is the point. An enumeration corpus I wrote myself would be one where I had
unconsciously kept the answer inside the budget.

Run:  uv run python tools/measure_scope.py
      uv run python tools/measure_scope.py --k 6 12 25 50
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from openknowledge.cascade.scope import TERM_PATTERN  # noqa: E402
from openknowledge.retrieval.base import Document  # noqa: E402
from openknowledge.retrieval.bm25 import BM25Retriever  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
DOCUMENTS = ROOT / "evals" / "golden-ftr" / "documents"

#: The one definition of what a glossary entry looks like lives in the code
#: that answers from it, so this harness cannot drift from the product.
_TERM = TERM_PATTERN
#: Sub-clauses - "(i) A military department" - are numbered and excluded.
_SUBCLAUSE = re.compile(r"^\(?[ivx0-9]+\)")


def glossary_terms() -> list[str]:
    """Every term § 300-1.1 defines, read out of the regulation itself.

    This list *is* the reference answer, so it is derived from the source
    rather than typed out here: a hand-copied list would drift from the
    corpus the moment the corpus was refreshed, and would be wrong in
    exactly the direction that flatters the score.
    """
    text = (DOCUMENTS / "ftr-300-1.md").read_text(encoding="utf-8")
    body = text.split("## § 300-1.1 Glossary of terms.")[1].split("## § 300-1.2")[0]
    terms = []
    for paragraph in body.split("\n\n"):
        paragraph = paragraph.strip()
        if not paragraph or paragraph.startswith(("(", "*", "-")):
            continue
        if _SUBCLAUSE.match(paragraph):
            continue
        found = _TERM.match(paragraph)
        if found and len(found.group(1).split()) <= 6:
            terms.append(found.group(1).strip())
    return terms


def transportation_methods() -> list[str]:
    """The four methods § 301-10.2 authorises - a list inside one chunk."""
    return ["common carrier", "government vehicle", "privately owned", "special conveyance"]


def load_corpus() -> list[Document]:
    return [
        Document(document_id=f.stem, title=f.stem, text=f.read_text(encoding="utf-8"))
        for f in sorted(DOCUMENTS.iterdir())
        if f.suffix == ".md"
    ]


#: Each case names what a complete answer contains. ``spans`` is how many
#: chunks of its home document the answer is spread over - the whole point of
#: the set is that one case is inside the budget and one is far outside it.
CASES = [
    {
        "id": "scope-01-glossary",
        "questions": [
            "what terms does the glossary define",
            "list all the terms defined in the glossary of terms",
            "what are all the definitions in part 300-1",
        ],
        "home": "ftr-300-1",
        "expected": glossary_terms,
        "note": "82 terms over 22 chunks - the shape of 'what are the priorities?'",
    },
    {
        "id": "scope-02-transport-methods",
        "questions": [
            "what transportation methods are authorized",
            "list the authorized methods of transportation",
        ],
        "home": "ftr-301-10",
        "expected": transportation_methods,
        "note": "the control: a closed list inside a single chunk, must stay at 100%",
    },
]


def measure(ks: list[int]) -> dict:
    corpus = load_corpus()
    retriever = BM25Retriever()
    retriever.index(corpus)
    chunks = retriever._index.chunks  # noqa: SLF001 - the harness may look inside
    report: dict = {"documents": len(corpus), "chunks": len(chunks), "cases": []}

    print(f"corpus: {len(corpus)} documents, {len(chunks)} chunks\n")
    for case in CASES:
        expected = case["expected"]()
        home = sum(1 for c in chunks if c.document_id == case["home"])
        print(f"{case['id']}  -  {case['note']}")
        print(f"  {len(expected)} items to find, spread over {home} chunks of {case['home']}")
        print(f"  {'question':<50} {'k':>3} {'chunks':>10} {'found':>14}")
        rows = []
        for question in case["questions"]:
            for k in ks:
                got = retriever.search(question, k=k)
                seen = " ".join(s.chunk.text for s in got).lower()
                found = sum(1 for item in expected if item.lower() in seen)
                from_home = sum(1 for s in got if s.chunk.document_id == case["home"])
                pct = round(100 * found / len(expected))
                print(
                    f"  {question:<50} {k:>3} {from_home:>4}/{home:<5} "
                    f"{found:>4}/{len(expected):<4} ({pct:>3}%)"
                )
                rows.append(
                    {
                        "question": question,
                        "k": k,
                        "chunks_from_home": from_home,
                        "chunks_in_home": home,
                        "found": found,
                        "expected": len(expected),
                        "percent": pct,
                    }
                )
            print()
        report["cases"].append({"id": case["id"], "note": case["note"], "rows": rows})
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, nargs="+", default=[6, 12, 25, 50])
    parser.add_argument("--json", type=pathlib.Path, help="write the report here")
    args = parser.parse_args()

    report = measure(args.k)
    if args.json:
        args.json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"written to {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
