"""Measuring contradiction detection.

Detection is an accuracy component now, so it needs its own numbers in both
directions - and the two directions fail differently.

**Recall** protects the employee. A missed contradiction means somebody is told
the superseded policy, confidently, with a citation attached.

**Precision** protects the feature. A false flag blocks a question that could
have been answered, and an admin who sees three bogus flags stops reading the
fourth - which costs every real flag after that. A detector at 100% recall and
40% precision gets switched off within a week, and then its recall is zero.

Neither number alone tells you anything, so this reports both, refuses to run
without `clean` cases, and fails on a regression in either.

Runs with no model, deterministically, so it belongs in CI.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from ..knowledge.claims import find_conflicts
from ..retrieval.base import Document


class ConflictSetError(ValueError):
    """The labelled set is malformed - fail loudly rather than scoring less."""


@dataclass(frozen=True, slots=True)
class ConflictCase:
    id: str
    documents: tuple[Document, ...]
    expect_conflict: bool
    #: Optional: which detector should have caught it ("numeric" / "deontic").
    kind: str | None = None
    notes: str = ""


@dataclass(frozen=True, slots=True)
class CaseOutcome:
    case: ConflictCase
    detected: bool
    kinds: tuple[str, ...]
    detail: str = ""

    @property
    def correct(self) -> bool:
        return self.detected == self.case.expect_conflict

    @property
    def is_false_positive(self) -> bool:
        return self.detected and not self.case.expect_conflict

    @property
    def is_miss(self) -> bool:
        return not self.detected and self.case.expect_conflict

    @property
    def wrong_detector(self) -> bool:
        """Detected, correctly, but by the pass we did not expect.

        Not a failure - the outcome is right - but worth surfacing, because it
        usually means one detector is doing another's job by accident and will
        stop when a threshold moves.
        """
        return (
            self.detected
            and self.case.expect_conflict
            and self.case.kind is not None
            and self.case.kind not in self.kinds
        )


@dataclass
class ConflictReport:
    outcomes: list[CaseOutcome] = field(default_factory=list)

    @property
    def true_positives(self) -> int:
        return sum(o.detected and o.case.expect_conflict for o in self.outcomes)

    @property
    def false_positives(self) -> int:
        return sum(o.is_false_positive for o in self.outcomes)

    @property
    def misses(self) -> int:
        return sum(o.is_miss for o in self.outcomes)

    @property
    def true_negatives(self) -> int:
        return sum(not o.detected and not o.case.expect_conflict for o in self.outcomes)

    @property
    def precision(self) -> float:
        flagged = self.true_positives + self.false_positives
        return self.true_positives / flagged if flagged else 1.0

    @property
    def recall(self) -> float:
        real = self.true_positives + self.misses
        return self.true_positives / real if real else 1.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    @property
    def passed(self) -> bool:
        return not self.false_positives and not self.misses

    def to_dict(self) -> dict[str, Any]:
        return {
            "cases": len(self.outcomes),
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "misses": self.misses,
            "true_negatives": self.true_negatives,
            "failures": [
                {
                    "id": o.case.id,
                    "kind": "false positive" if o.is_false_positive else "missed",
                    "detail": o.detail,
                }
                for o in self.outcomes
                if not o.correct
            ],
            "wrong_detector": [o.case.id for o in self.outcomes if o.wrong_detector],
        }


def parse_conflict_cases(
    raw: Any, *, source: str = "<memory>", require_clean: bool = True
) -> list[ConflictCase]:
    if not isinstance(raw, list):
        raise ConflictSetError(f"{source}: expected a list of cases")

    cases: list[ConflictCase] = []
    seen: set[str] = set()
    for i, entry in enumerate(raw):
        if not isinstance(entry, dict):
            raise ConflictSetError(f"{source}: case {i} is not a mapping")
        case_id = str(entry.get("id") or f"case-{i}")
        if case_id in seen:
            raise ConflictSetError(f"{source}: duplicate case id {case_id!r}")
        seen.add(case_id)

        expect = entry.get("expect")
        if expect not in ("conflict", "clean"):
            raise ConflictSetError(f"case {case_id!r}: expect must be 'conflict' or 'clean'")

        documents = entry.get("documents")
        if not isinstance(documents, list) or not documents:
            raise ConflictSetError(f"case {case_id!r}: at least one document is required")

        parsed: list[Document] = []
        for doc in documents:
            if not isinstance(doc, dict) or not doc.get("id") or not doc.get("text"):
                raise ConflictSetError(f"case {case_id!r}: each document needs an id and text")
            parsed.append(
                Document(
                    document_id=str(doc["id"]),
                    title=str(doc.get("title") or doc["id"]),
                    text=str(doc["text"]),
                )
            )

        cases.append(
            ConflictCase(
                id=case_id,
                documents=tuple(parsed),
                expect_conflict=expect == "conflict",
                kind=entry.get("kind"),
                notes=str(entry.get("notes", "")),
            )
        )

    if not cases:
        raise ConflictSetError(f"{source}: contains no cases")
    if require_clean and not any(not c.expect_conflict for c in cases):
        raise ConflictSetError(
            f"{source}: has no 'clean' cases. A set that only contains real "
            "contradictions measures recall and cannot see a false positive, "
            "which is the failure that gets the detector switched off."
        )
    return cases


def load_conflict_cases(path: str | Path) -> list[ConflictCase]:
    """Load from a YAML file, or every ``*.yaml`` in a directory."""
    p = Path(path)
    if p.is_dir():
        files = sorted([*p.glob("*.yaml"), *p.glob("*.yml")])
        if not files:
            raise ConflictSetError(f"{p}: no .yaml files found")
        cases: list[ConflictCase] = []
        for f in files:
            # Each file is parsed leniently for the clean-case rule, which is
            # then enforced across the whole set - one file may legitimately
            # hold only contradictions as long as another holds the negatives.
            cases.extend(
                parse_conflict_cases(
                    yaml.safe_load(f.read_text(encoding="utf-8")),
                    source=str(f),
                    require_clean=False,
                )
            )
        if not any(not c.expect_conflict for c in cases):
            raise ConflictSetError(
                f"{p}: no 'clean' cases anywhere. A set of only real contradictions "
                "measures recall and cannot see a false positive."
            )
        return cases

    if not p.is_file():
        raise ConflictSetError(f"{p}: not found")
    return parse_conflict_cases(yaml.safe_load(p.read_text(encoding="utf-8")), source=str(p))


def run_conflict_eval(
    cases: list[ConflictCase], *, deontic_strictness: float = 1.0
) -> ConflictReport:
    report = ConflictReport()
    for case in cases:
        conflicts = find_conflicts(list(case.documents), deontic_strictness=deontic_strictness)
        detail = ""
        if conflicts:
            top = conflicts[0]
            detail = f"{top.kind}: {top.describe()[:160]}"
        elif case.expect_conflict:
            detail = "no detector fired"

        report.outcomes.append(
            CaseOutcome(
                case=case,
                detected=bool(conflicts),
                kinds=tuple(dict.fromkeys(c.kind for c in conflicts)),
                detail=detail,
            )
        )
    return report


def format_conflict_report(report: ConflictReport) -> str:
    lines = [
        f"{len(report.outcomes)} cases  "
        f"({report.true_positives + report.misses} real contradictions, "
        f"{report.false_positives + report.true_negatives} that must stay quiet)",
        "",
        "Detection",
        f"  precision   {report.precision:>7.1%}  (of what it flagged, how much was real)",
        f"  recall      {report.recall:>7.1%}  (of what was real, how much it flagged)",
        f"  F1          {report.f1:>7.1%}",
        "",
        f"  {report.true_positives} caught   {report.misses} missed   "
        f"{report.false_positives} false   {report.true_negatives} correctly quiet",
    ]

    failures = [o for o in report.outcomes if not o.correct]
    if failures:
        lines += ["", f"Failures ({len(failures)})"]
        for outcome in failures:
            label = "FALSE FLAG" if outcome.is_false_positive else "MISSED"
            lines.append(f"  [{label}] {outcome.case.id}")
            if outcome.detail:
                lines.append(f"      {outcome.detail}")
            if outcome.case.notes:
                lines.append(f"      note: {outcome.case.notes.strip()[:160]}")

    mismatched = [o for o in report.outcomes if o.wrong_detector]
    if mismatched:
        lines += ["", "Caught by an unexpected detector (not a failure, but brittle):"]
        lines += [
            f"  {o.case.id}: expected {o.case.kind}, got {', '.join(o.kinds)}" for o in mismatched
        ]

    lines.append("")
    if report.passed:
        lines.append("PASSED - every contradiction caught, nothing else flagged.")
    else:
        lines.append(f"FAILED - {report.misses} missed, {report.false_positives} falsely flagged.")
    return "\n".join(lines)
