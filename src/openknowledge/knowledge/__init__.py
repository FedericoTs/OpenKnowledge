"""Knowledge lifecycle: draft at ingest, review once, flag contradictions."""

from .claims import Claim, Conflict, extract_claims, find_conflicts
from .generate import DraftedAnswer, DraftResult, draft_from_document
from .pipeline import (
    IngestReport,
    draft_for_documents,
    ingest_documents,
    rank_by_demand,
    scan_documents,
)
from .reverify import Revision, figure_changes, reverify_changed_documents
from .store import (
    KnowledgeStore,
    Proposal,
    ProposalStatus,
    StoredConflict,
    proposal_id,
)

__all__ = [
    "Claim",
    "Conflict",
    "DraftResult",
    "DraftedAnswer",
    "IngestReport",
    "KnowledgeStore",
    "Proposal",
    "ProposalStatus",
    "Revision",
    "StoredConflict",
    "draft_for_documents",
    "draft_from_document",
    "extract_claims",
    "figure_changes",
    "find_conflicts",
    "ingest_documents",
    "proposal_id",
    "rank_by_demand",
    "scan_documents",
    "reverify_changed_documents",
]
