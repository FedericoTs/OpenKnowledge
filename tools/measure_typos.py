#!/usr/bin/env python3
"""What a typo costs, before deciding whether to spend anything fixing it.

Two of the sixteen questions a real install could not answer were misspelled -
"summaryze the second document", "what is the list of priroities". BM25 matches
terms exactly, so a misspelled content word contributes nothing, and nobody
here had measured how much that costs or how often it decides an answer.

No model is called. The measurement is at the retrieval layer, where the damage
would be done: for each golden case that names the document it must cite, is
that document still among the passages retrieved after the question is
misspelled? A fact the model never sees is a fact it cannot state, so this is
the ceiling on what a typo can cost, in the same sense as tools/measure_scope.py.

The edits are deterministic, not sampled: the longest content word of each
question - the most distinctive one, and the one a typo hurts most - is
mangled four ways that between them cover what people actually mistype. Same
input, same number, every time.

    uv run python tools/measure_typos.py
    uv run python tools/measure_typos.py --set evals/golden-ftr/ftr.yaml
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from openknowledge.connectors.local_files import document_id_for  # noqa: E402
from openknowledge.documents import is_supported, parse_file  # noqa: E402
from openknowledge.evaluation.dataset import load_cases  # noqa: E402
from openknowledge.retrieval.base import Document  # noqa: E402
from openknowledge.retrieval.bm25 import BM25Retriever  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]

#: Words too common to be worth mangling: a typo in one costs nothing because
#: BM25 gives them almost no weight anyway.
_STOP = frozenset(
    [
        "what",
        "when",
        "where",
        "which",
        "who",
        "whom",
        "whose",
        "why",
        "how",
        "the",
        "a",
        "an",
        "and",
        "or",
        "of",
        "for",
        "to",
        "in",
        "on",
        "at",
        "is",
        "are",
        "was",
        "were",
        "do",
        "does",
        "did",
        "can",
        "could",
        "may",
        "might",
        "must",
        "shall",
        "should",
        "will",
        "would",
        "have",
        "has",
        "had",
        "i",
        "you",
        "it",
        "my",
        "your",
        "our",
        "their",
        "this",
        "that",
        "these",
        "those",
        "there",
        "here",
        "about",
        "with",
        "from",
        "by",
        "as",
        "if",
        "not",
        "no",
        "yes",
    ]
)


def _content_word(question: str) -> str | None:
    """The longest word worth mangling, or None if there is none."""
    words = [w for w in re.findall(r"[A-Za-z]{4,}", question) if w.lower() not in _STOP]
    return max(words, key=len) if words else None


def variants(word: str) -> dict[str, str]:
    """Four edits of ``word``, each a thing people really do.

    Deterministic and middle-anchored: the first and last letters of a word
    are the ones typists get right, and mangling them would overstate the
    damage relative to a real misspelling.
    """
    mid = len(word) // 2
    return {
        "dropped": word[:mid] + word[mid + 1 :],
        "doubled": word[:mid] + word[mid] + word[mid:],
        "transposed": word[: mid - 1] + word[mid] + word[mid - 1] + word[mid + 1 :],
        "substituted": word[:mid] + "x" + word[mid + 1 :],
    }


def load_corpus(documents_dir: pathlib.Path) -> list[Document]:
    """Every readable document under ``documents_dir``, with the ids the app uses.

    Recursive, and the ids come from the connector rather than from the file
    stem: evals/corpus/aveline is a tree of folders and its cases cite
    "hr-expenses-policy", not "expenses-policy". A flat listing found no files
    there at all and the harness reported 0% rather than saying so, which is
    the failure mode a measurement must never have.
    """
    documents = []
    for f in sorted(documents_dir.rglob("*")):
        # Through the real parser, so a corpus with a spreadsheet or a PDF in
        # it - aveline has both, and two of its cases cite them - is the same
        # corpus here as in the product.
        if not f.is_file() or not is_supported(f):
            continue
        parsed = parse_file(f)
        documents.append(
            Document(
                document_id=document_id_for(f.relative_to(documents_dir)),
                title=parsed.title or f.stem,
                text=parsed.text,
                blocks=parsed.blocks,
            )
        )
    if not documents:
        raise SystemExit(f"no readable documents under {documents_dir}")
    return documents


def measure(set_path: pathlib.Path, documents_dir: pathlib.Path, k: int) -> dict:
    corpus = load_corpus(documents_dir)
    retriever = BM25Retriever()
    retriever.index(corpus)
    cases = [c for c in load_cases(set_path) if c.must_cite]

    def finds(question: str, wanted: tuple[str, ...]) -> bool:
        got = {h.chunk.document_id for h in retriever.search(question, k=k)}
        return all(doc in got for doc in wanted)

    known = {d.document_id for d in corpus}
    unknown = sorted({d for c in cases for d in c.must_cite} - known)
    if unknown:
        raise SystemExit(
            f"{len(unknown)} cited document(s) are not in {documents_dir}: {', '.join(unknown[:5])}"
        )

    rows = []
    for case in cases:
        word = _content_word(case.question)
        if word is None:
            continue
        clean = finds(case.question, case.must_cite)
        edits = {}
        for name, typo in variants(word).items():
            edits[name] = finds(case.question.replace(word, typo), case.must_cite)
        rows.append({"id": case.id, "word": word, "clean": clean, **edits})

    kinds = ("dropped", "doubled", "transposed", "substituted")
    scored = [r for r in rows if r["clean"]]
    report = {
        "set": str(set_path.relative_to(ROOT)),
        "documents": str(documents_dir.relative_to(ROOT)),
        "k": k,
        "cases_with_a_citation": len(rows),
        "found_when_spelled_correctly": len(scored),
        "found_after_one_typo": {kind: sum(1 for r in scored if r[kind]) for kind in kinds},
        "rows": rows,
    }
    print(f"{report['set']}  -  {len(rows)} cases naming a document, k={k}\n")
    print(f"  spelled correctly            {len(scored)}/{len(rows)} find it")
    for kind in kinds:
        found = report["found_after_one_typo"][kind]
        share = round(100 * found / len(scored)) if scored else 0
        print(f"  one letter {kind:<12}      {found}/{len(scored)} find it  ({share}%)")
    worst = [r["id"] for r in scored if not all(r[kind] for kind in kinds)]
    if worst:
        print(f"\n  lost by at least one edit: {', '.join(worst)}")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--set", type=pathlib.Path, default=ROOT / "evals/golden-ftr/ftr.yaml")
    parser.add_argument("--documents", type=pathlib.Path, default=None)
    parser.add_argument("--k", type=int, default=6)
    parser.add_argument("--json", type=pathlib.Path)
    args = parser.parse_args()

    # Resolved, because a path typed relative to the working directory cannot
    # be made relative to the repository root without it.
    set_path = args.set.resolve()
    documents = (args.documents or set_path.parent / "documents").resolve()
    report = measure(set_path, documents, args.k)
    if args.json:
        args.json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"\nwritten to {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
