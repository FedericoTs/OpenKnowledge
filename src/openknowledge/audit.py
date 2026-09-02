"""Point at a folder, get back where your documents disagree with each other.

This is the smallest useful thing OpenKnowledge does, and deliberately the
cheapest to try: no API key, no model, no GPU, no database, no configuration.
It reads a directory, extracts every figure and every stated rule, and reports
the pairs that cannot both be true.

It exists as its own entry point because the value lands before any of the
answer engine does. A company that is not ready to run a chatbot over its
policies is still interested in being told that its expense policy says EUR 500
and its travel guidelines say EUR 1,000 - and that finding costs nothing to
produce and nothing to trust, because it is two quotes from their own files.

Nothing is written and nothing leaves the machine. The audit builds no store,
touches no data directory, and calls no provider; re-running it on the same
folder produces byte-identical output.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from . import graph as knowledge_graph
from .connectors.local_files import LocalFilesConnector, SkippedFile
from .documents import SUPPORTED_SUFFIXES
from .knowledge.claims import Conflict, compare_documents, extract_claims
from .knowledge.deontic import extract_deontic_claims
from .knowledge.variants import DocumentPair, group_by_document_pair

#: How a conflict kind reads to somebody who has not read the source.
_KIND_LABEL = {"numeric": "figure", "deontic": "rule"}


@dataclass(frozen=True, slots=True)
class AuditReport:
    """What one audit run found. Free to produce, so free to re-run."""

    root: str
    documents: int
    claims_checked: int
    conflicts: tuple[Conflict, ...]
    unreadable: tuple[SkippedFile, ...]
    #: The same findings grouped by the two documents involved, so that a pair
    #: of near-duplicate files reads as one filing problem rather than as
    #: ninety-eight contradictions.
    pairs: tuple[DocumentPair, ...] = ()
    #: Every readable document and what connects them - see graph.py. Drawn
    #: into the HTML report; absent from the text and the JSON.
    graph: knowledge_graph.Graph | None = None

    @property
    def clean(self) -> bool:
        return not self.conflicts

    @property
    def contradicting(self) -> tuple[DocumentPair, ...]:
        """Pairs that genuinely disagree, as opposed to duplicated documents."""
        return tuple(p for p in self.pairs if not p.is_variant)

    @property
    def variants(self) -> tuple[DocumentPair, ...]:
        return tuple(p for p in self.pairs if p.is_variant)

    def as_dict(self) -> dict[str, object]:
        """Machine-readable form, for a CI step or a ticket generator."""
        return {
            "root": self.root,
            "documents": self.documents,
            "claims_checked": self.claims_checked,
            "conflicts": [
                {
                    "key": c.key,
                    "kind": c.kind,
                    "overlap": c.overlap,
                    "left": {
                        "document": c.left.document_id,
                        "title": c.left.document_title,
                        "says": c.left.raw,
                        "sentence": c.left.sentence.strip(),
                    },
                    "right": {
                        "document": c.right.document_id,
                        "title": c.right.document_title,
                        "says": c.right.raw,
                        "sentence": c.right.sentence.strip(),
                    },
                }
                for c in self.conflicts
            ],
            "duplicates": [
                {
                    "left": p.left,
                    "right": p.right,
                    "differing_figures": len(p.conflicts),
                    "compared": p.compared,
                }
                for p in self.variants
            ],
            "unreadable": [{"path": s.path, "reason": s.reason} for s in self.unreadable],
        }


def audit_folder(
    root: str | Path,
    *,
    min_overlap: float = 0.34,
    deontic_strictness: float = 1.0,
    pdf_backend: str = "auto",
) -> AuditReport:
    """Read a folder and return every contradiction we can find without a model."""
    connector = LocalFilesConnector(root, suffixes=SUPPORTED_SUFFIXES, pdf_backend=pdf_backend)
    documents = connector.fetch()

    conflicts, agreements = compare_documents(
        documents,
        min_overlap=min_overlap,
        deontic_strictness=deontic_strictness,
    )
    pairs = group_by_document_pair(conflicts, agreements)

    # Reported so the run is legible: "0 conflicts" out of 4,000 claims is a
    # clean corpus, out of 3 claims it means nothing was read properly - most
    # likely a folder of scans, which otherwise passes silently.
    per_document = {
        doc.document_id: len(extract_claims(doc)) + len(extract_deontic_claims(doc))
        for doc in documents
    }

    return AuditReport(
        root=str(connector.root),
        documents=len(documents),
        claims_checked=sum(per_document.values()),
        conflicts=tuple(conflicts),
        unreadable=tuple(connector.skipped),
        pairs=tuple(pairs),
        graph=knowledge_graph.from_audit(
            documents, pairs, root=connector.root, claims=per_document
        ),
    )


def render(report: AuditReport, *, width: int = 96) -> str:
    """The human-readable report.

    Written to be pasteable into an email to whoever owns the documents, so
    every finding carries both quotes: the point is not that a tool flagged
    something, it is that two of their own sentences disagree.
    """
    lines: list[str] = []
    lines.append(f"OpenKnowledge audit - {report.root}")
    lines.append(
        f"{report.documents} document(s), {report.claims_checked} claim(s) checked, "
        "0 model calls, $0.00"
    )
    lines.append("")

    if report.clean:
        lines.append("No contradictions found between these documents.")
        lines.append("")
    else:
        contradicting = report.contradicting
        found = sum(len(p.conflicts) for p in contradicting)
        if found:
            lines.append(f"{found} contradiction(s), in {len(contradicting)} document pair(s):")
            lines.append("")
        number = 0
        for pair in contradicting:
            for conflict in pair.conflicts:
                number += 1
                kind = _KIND_LABEL.get(conflict.kind, conflict.kind)
                lines.append(
                    f"{number:>3}. {conflict.left.document_id} vs {conflict.right.document_id}"
                    f"   ({kind}, {conflict.overlap:.0%} context match)"
                )
                for side in (conflict.left, conflict.right):
                    lines.append(f"     [{side.document_id}] says {side.raw}")
                    lines.append(f'       "{_clip(side.sentence, width)}"')
                lines.append("")

        if report.variants:
            lines.append(f"{len(report.variants)} pair(s) look like duplicated documents:")
            lines.append("")
            for pair in report.variants:
                lines.append(f"     {pair.describe()}")
            lines.append("")
            lines.append("     Listed separately because reconciling them one figure at a time is")
            lines.append("     the wrong job - the right one is deciding which copy stands.")
            lines.append("")

    if report.unreadable:
        lines.append(f"{len(report.unreadable)} file(s) contributed nothing:")
        for skipped in report.unreadable[:20]:
            lines.append(f"  {skipped.path}: {skipped.reason}")
        if len(report.unreadable) > 20:
            lines.append(f"  ... and {len(report.unreadable) - 20} more")
        lines.append("")

    lines.append("What this does not check:")
    lines.append("  - a document contradicting itself (two figures in one file are usually")
    lines.append("    a rule and its exception, and flagging those trains you to ignore this)")
    lines.append("  - rules stated without a must/may/must not marker, or not in English")
    lines.append("  - scanned pages - there is no OCR, so an image of a policy reads as nothing")
    lines.append("")
    lines.append("Nothing was written and no document left this machine.")
    return "\n".join(lines)


def _clip(sentence: str, width: int) -> str:
    text = " ".join(sentence.split())
    return text if len(text) <= width else text[: width - 1].rstrip() + "…"
