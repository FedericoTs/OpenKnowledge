"""Knowledge lifecycle: draft at ingest, review once, flag contradictions."""

from .claims import (
    Claim,
    Conflict,
    compare_documents,
    compare_numeric_claims,
    extract_claims,
    find_conflicts,
    find_numeric_conflicts,
)
from .crosscheck import CrossCheckFinding, crosscheck_answers
from .deontic import DeonticClaim, Force, conflicts_between, extract_deontic_claims
from .generate import DraftedAnswer, DraftResult, draft_from_document
from .pipeline import (
    IngestReport,
    draft_for_documents,
    ingest_documents,
    rank_by_demand,
    scan_documents,
)
from .reverify import Revision, figure_changes, reverify_changed_documents
from .salience import Salience, salience_from
from .store import (
    KnowledgeStore,
    Proposal,
    ProposalStatus,
    StoredConflict,
    proposal_id,
)
from .variants import DocumentPair, group_by_document_pair

__all__ = [
    "Claim",
    "Conflict",
    "CrossCheckFinding",
    "DeonticClaim",
    "DraftResult",
    "DraftedAnswer",
    "IngestReport",
    "KnowledgeStore",
    "Proposal",
    "ProposalStatus",
    "DocumentPair",
    "Revision",
    "Salience",
    "StoredConflict",
    "compare_documents",
    "compare_numeric_claims",
    "draft_for_documents",
    "draft_from_document",
    "extract_claims",
    "figure_changes",
    "Force",
    "conflicts_between",
    "crosscheck_answers",
    "extract_deontic_claims",
    "find_conflicts",
    "find_numeric_conflicts",
    "group_by_document_pair",
    "ingest_documents",
    "proposal_id",
    "rank_by_demand",
    "scan_documents",
    "reverify_changed_documents",
    "salience_from",
]
