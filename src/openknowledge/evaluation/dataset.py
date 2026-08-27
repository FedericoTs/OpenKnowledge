"""The golden set.

A case is a question plus what a correct answer has to contain. Two kinds:

``answerable``
    The corpus covers this. The answer must cite the right documents and state
    the right facts - and must not state the wrong ones, which is why cases
    carry ``must_not_say``. "20 weeks" appearing is only half the check; "26
    weeks" not appearing is the other half.

``refusal``
    The corpus does *not* cover this, and the only correct behaviour is to say
    so. These are the most important cases in the file. A bot that answers 95%
    of questions correctly and confidently invents the other 5% is unusable in a
    company, because nobody can tell which kind they are looking at.

Cases may list ``paraphrases``: the same question asked differently. They must
produce the same facts, which is what stops a cheap cache from being the only
thing holding answer consistency together.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

CaseKind = Literal["answerable", "refusal"]


@dataclass(frozen=True, slots=True)
class Case:
    id: str
    question: str
    kind: CaseKind = "answerable"
    #: Document ids the answer must cite. Empty means "any citation will do".
    must_cite: tuple[str, ...] = ()
    #: Substrings that must appear (matched on casefolded, whitespace-collapsed text).
    must_say: tuple[str, ...] = ()
    #: Substrings that must NOT appear - usually the plausible wrong answer.
    must_not_say: tuple[str, ...] = ()
    #: Same question, different words. Must yield the same facts.
    paraphrases: tuple[str, ...] = ()
    #: Groups the asker belongs to, for access-control cases.
    principals: tuple[str, ...] | None = None
    notes: str = ""
    tags: tuple[str, ...] = field(default_factory=tuple)

    @property
    def all_phrasings(self) -> tuple[str, ...]:
        return (self.question, *self.paraphrases)


class DatasetError(ValueError):
    """The golden set is malformed - fail loudly rather than silently scoring less."""


def _as_tuple(value: Any, field_name: str, case_id: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, list) and all(isinstance(v, str) for v in value):
        return tuple(value)
    raise DatasetError(f"case {case_id!r}: {field_name} must be a string or list of strings")


def parse_cases(raw: Any, *, source: str = "<memory>") -> list[Case]:
    if not isinstance(raw, list):
        raise DatasetError(f"{source}: expected a list of cases")

    cases: list[Case] = []
    seen: set[str] = set()
    for i, entry in enumerate(raw):
        if not isinstance(entry, dict):
            raise DatasetError(f"{source}: case {i} is not a mapping")
        case_id = str(entry.get("id") or f"case-{i}")
        if case_id in seen:
            raise DatasetError(f"{source}: duplicate case id {case_id!r}")
        seen.add(case_id)

        question = entry.get("question")
        if not isinstance(question, str) or not question.strip():
            raise DatasetError(f"case {case_id!r}: 'question' is required")

        kind = entry.get("kind", "answerable")
        if kind not in ("answerable", "refusal"):
            raise DatasetError(f"case {case_id!r}: kind must be 'answerable' or 'refusal'")

        principals = entry.get("principals")
        cases.append(
            Case(
                id=case_id,
                question=question,
                kind=kind,
                must_cite=_as_tuple(entry.get("must_cite"), "must_cite", case_id),
                must_say=_as_tuple(entry.get("must_say"), "must_say", case_id),
                must_not_say=_as_tuple(entry.get("must_not_say"), "must_not_say", case_id),
                paraphrases=_as_tuple(entry.get("paraphrases"), "paraphrases", case_id),
                principals=(
                    tuple(_as_tuple(principals, "principals", case_id))
                    if principals is not None
                    else None
                ),
                notes=str(entry.get("notes", "")),
                tags=_as_tuple(entry.get("tags"), "tags", case_id),
            )
        )

    if not cases:
        raise DatasetError(f"{source}: contains no cases")
    return cases


def load_cases(path: str | Path) -> list[Case]:
    """Load a golden set from a YAML file, or every ``*.yaml`` in a directory."""
    p = Path(path)
    if p.is_dir():
        cases: list[Case] = []
        files = sorted([*p.glob("*.yaml"), *p.glob("*.yml")])
        if not files:
            raise DatasetError(f"{p}: no .yaml files found")
        for f in files:
            cases.extend(load_cases(f))
        return cases

    if not p.is_file():
        raise DatasetError(f"{p}: not found")
    return parse_cases(yaml.safe_load(p.read_text(encoding="utf-8")), source=str(p))


def filter_cases(
    cases: list[Case], *, kind: CaseKind | None = None, tags: tuple[str, ...] = ()
) -> list[Case]:
    """Narrow a golden set by kind and/or tag.

    ``--only refusal`` is the useful one: it runs just the safety set, which is
    the subset you want on a tight loop while changing retrieval or the
    grounding gate.
    """
    selected = cases
    if kind is not None:
        selected = [c for c in selected if c.kind == kind]
    if tags:
        wanted = set(tags)
        selected = [c for c in selected if wanted & set(c.tags)]
    return selected
