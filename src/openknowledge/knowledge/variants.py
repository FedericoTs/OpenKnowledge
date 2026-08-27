"""Telling a contradiction apart from a duplicate.

Two documents that disagree about one figure are contradicting each other, and
somebody needs to decide which is right. Two documents that disagree about
ninety-eight figures are not contradicting each other in any useful sense: one
is a stale copy, a different vendor's version, or last year's numbers, and the
finding a human needs is *"you have two of these"*, said once.

Unweighted or weighted, no per-claim threshold can separate those two cases,
because each individual finding looks identical in both. What separates them is
the shape of the pair as a whole. Measured on 15 real vendor contracts, which
contain two parallel copies of seven documents, the detector emitted 287
findings; 281 of them were six duplicate pairs restated dozens of times.

The signal is **how much subject matter the two documents share**, not what
fraction of it disagrees. On that corpus the duplicate pairs compared 46 to 189
figures each; every genuinely distinct pair compared nine or fewer. The
disagreement *ratio* was tried first and does not work: the duplicates ran from
0.27 to 0.64 and the distinct pairs from 0.22 to 0.67, straight through each
other. It is not shipped as a tunable, because a knob that does not discriminate
is worse than no knob.

Two documents asserting ninety of the same figures are the same document twice.
Two documents asserting three of the same figures are two documents, and if they
disagree on one, somebody needs to know.

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar, runtime_checkable

from .claims import Conflict


@runtime_checkable
class HasDocumentPair(Protocol):
    """The two ends of a stored disagreement.

    A protocol rather than an import of `StoredConflict`: grouping only needs to
    know which documents a finding joins, and depending on the store's row type
    for that would tie this module to where conflicts happen to be persisted.
    """

    left_document: str
    right_document: str


#: Kept generic so grouping does not force callers to give up the rest of their
#: row. The CLI groups by document pair and then prints each finding's sentences;
#: narrowing to the protocol here would mean looking every one of them up again.
ConflictT = TypeVar("ConflictT", bound=HasDocumentPair)


#: Below this many disagreements a pair is always reported claim by claim. It is
#: also what protects the case the shared-subject test would otherwise get wrong:
#: two long handbooks that genuinely contradict each other on three points share
#: a lot of figures, and three findings is a list a human can read.
_MIN_DISAGREEMENTS = 6

#: How many figures two documents must both assert, on subjects that matched,
#: before the pair itself reads as duplication. The real corpus separated cleanly
#: at 46 against 9; 20 sits between them with room on both sides, which is the
#: most that can be claimed from one corpus.
_MIN_SHARED_FIGURES = 20


@dataclass(frozen=True, slots=True)
class DocumentPair:
    """Every disagreement between two documents, plus what they agreed on."""

    left: str
    right: str
    conflicts: tuple[Conflict, ...]
    agreements: int = 0

    @property
    def compared(self) -> int:
        return len(self.conflicts) + self.agreements

    @property
    def disagreement(self) -> float:
        """Share of the figures these two both assert that they disagree on.

        Reported because it is worth seeing, not used to classify - see the
        module docstring for the measurement that ruled it out.
        """
        return len(self.conflicts) / self.compared if self.compared else 0.0

    @property
    def is_variant(self) -> bool:
        """Two versions of one document rather than two documents in conflict."""
        return len(self.conflicts) >= _MIN_DISAGREEMENTS and self.compared >= _MIN_SHARED_FIGURES

    def describe(self) -> str:
        if self.is_variant:
            return (
                f"{self.left} and {self.right} look like two versions of the same document: "
                f"{len(self.conflicts)} of the {self.compared} figures they share disagree. "
                "Retire one rather than reconciling them line by line."
            )
        return f"{self.left} vs {self.right}: {len(self.conflicts)} disagreement(s)"


def group_by_document_pair(
    conflicts: list[Conflict] | tuple[Conflict, ...],
    agreements: dict[tuple[str, str], int] | None = None,
) -> list[DocumentPair]:
    """Collect findings by the two documents involved, worst pair first.

    Deontic conflicts carry no agreement count - there is no "same rule, same
    force" tally to compare against - so a pair found only through prose never
    reaches the shared-figures threshold and is always listed claim by claim.
    That is the right default: prose findings are rarer and each one is a
    judgement a human has to make anyway.
    """
    agreements = agreements or {}
    grouped: dict[tuple[str, str], list[Conflict]] = {}
    for conflict in conflicts:
        key = _pair(conflict.left.document_id, conflict.right.document_id)
        grouped.setdefault(key, []).append(conflict)

    pairs = [
        DocumentPair(
            left=left,
            right=right,
            conflicts=tuple(found),
            agreements=agreements.get((left, right), agreements.get((right, left), 0)),
        )
        for (left, right), found in grouped.items()
    ]
    # Variants last: they are a filing problem, not an answer that is at risk.
    pairs.sort(key=lambda p: (p.is_variant, -len(p.conflicts), p.left, p.right))
    return pairs


@dataclass(frozen=True)
class StoredPair(Generic[ConflictT]):
    """The same grouping over conflicts read back from the store.

    A stored conflict has no agreement count - the store keeps disagreements,
    not the figures two documents agreed on - so duplication can only be judged
    from the disagreements themselves. That makes this a coarser test than
    `DocumentPair`, and it is applied where its being coarse is safe: naming a
    pair as duplicated in a review list, rather than deciding what to compare.
    """

    left: str
    right: str
    conflicts: tuple[ConflictT, ...]

    @property
    def is_variant(self) -> bool:
        return len(self.conflicts) >= _MIN_SHARED_FIGURES

    def describe(self) -> str:
        if self.is_variant:
            return (
                f"{self.left} and {self.right} disagree on {len(self.conflicts)} figures. "
                "That is a versioning problem, not a contradiction: retire one rather "
                "than reconciling them line by line."
            )
        return f"{self.left} vs {self.right}: {len(self.conflicts)} disagreement(s)"


def group_stored(conflicts: list[ConflictT]) -> list[StoredPair[ConflictT]]:
    """Group stored conflicts by the two documents involved, worst pair first."""
    grouped: dict[tuple[str, str], list[ConflictT]] = {}
    for conflict in conflicts:
        key = _pair(conflict.left_document, conflict.right_document)
        grouped.setdefault(key, []).append(conflict)

    pairs: list[StoredPair[ConflictT]] = [
        StoredPair(left=left, right=right, conflicts=tuple(found))
        for (left, right), found in grouped.items()
    ]
    pairs.sort(key=lambda p: (p.is_variant, -len(p.conflicts), p.left, p.right))
    return pairs


def _pair(left: str, right: str) -> tuple[str, str]:
    return (left, right) if left <= right else (right, left)
