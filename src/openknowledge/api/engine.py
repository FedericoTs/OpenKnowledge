"""Assembles the running system from settings."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from ..cache import AnswerStore
from ..cascade import Cascade
from ..config import Settings
from ..connectors import LocalFilesConnector
from ..knowledge import IngestReport, KnowledgeStore, draft_for_documents, scan_documents
from ..knowledge.reverify import reverify_changed_documents
from ..providers.anthropic_provider import AnthropicProvider
from ..providers.base import ChatProvider
from ..providers.openai_compat import OpenAICompatProvider
from ..retrieval import BM25Retriever
from ..retrieval.base import Document

log = logging.getLogger(__name__)


@dataclass
class Engine:
    settings: Settings
    store: AnswerStore
    retriever: BM25Retriever
    cascade: Cascade
    connector: LocalFilesConnector
    knowledge: KnowledgeStore
    local: ChatProvider | None = None
    frontier: ChatProvider | None = None
    #: Last fetched corpus, so `learn` does not re-read every file from disk.
    documents: list[Document] = field(default_factory=list)
    #: What the last reindex reported as new or changed.
    last_scan: IngestReport | None = None

    def reindex(self) -> tuple[int, int, str, int]:
        """Re-read the corpus. Free: this never calls a model.

        Runs conflict detection and retires drafts built on text that moved,
        both of which cost nothing. Drafting new answers is `learn`, which is
        separate precisely so that re-indexing cannot spend money by surprise.
        """
        self.documents = self.connector.fetch()
        self.retriever.index(self.documents)
        evicted = self.store.evict_other_corpus_versions(self.retriever.corpus_version)
        self.last_scan = scan_documents(
            self.documents,
            store=self.knowledge,
            retriever=self.retriever,
            min_conflict_overlap=self.settings.conflict_min_overlap,
            deontic_strictness=self.settings.deontic_strictness,
        )
        log.info(
            "indexed %d documents into %d chunks (corpus %s); evicted %d stale answers; "
            "%d conflicts open",
            len(self.documents),
            len(self.retriever),
            self.retriever.corpus_version,
            evicted,
            self.last_scan.conflicts_open,
        )
        return len(self.documents), len(self.retriever), self.retriever.corpus_version, evicted

    @property
    def drafting_provider(self) -> ChatProvider | None:
        """Prefer the local model for drafting.

        Drafting reads every changed document in full, so it is the most
        token-hungry thing the system does. Doing it on a model with no
        per-token invoice is the difference between a one-off cost and a
        genuinely free one.
        """
        return self.local or self.frontier

    async def learn(self, *, max_documents: int | None = None) -> IngestReport:
        """The paid pass: draft answers for changed documents, re-check approvals."""
        if self.last_scan is None or not self.documents:
            self.reindex()
        assert self.last_scan is not None

        report = self.last_scan
        provider = self.drafting_provider
        if provider is None:
            report.notes.append("no model configured; nothing drafted")
            return report

        touched = frozenset(report.added) | frozenset(report.changed)
        if touched:
            await draft_for_documents(
                self.documents,
                store=self.knowledge,
                provider=provider,
                corpus_version=self.retriever.corpus_version,
                document_ids=touched,
                report=report,
                min_support_ratio=self.settings.min_support_ratio,
                max_documents=max_documents or self.settings.max_documents_per_ingest,
            )

        if self.settings.reverify_on_change and report.changed:
            revisions = await reverify_changed_documents(
                frozenset(report.changed),
                store=self.knowledge,
                retriever=self.retriever,
                provider=provider,
                corpus_version=self.retriever.corpus_version,
                k=self.settings.retrieval_k,
                min_support_ratio=self.settings.min_support_ratio,
                max_tokens=self.settings.max_answer_tokens,
            )
            material = [r for r in revisions if r.is_material]
            report.revisions_raised = len(material)
            report.cost_usd += sum(r.cost_usd for r in revisions)
            for revision in material:
                report.notes.append(f"figure changed - {revision.describe()}")

        return report

    def approve(self, proposal_id: str, *, reviewer: str | None = None) -> bool:
        """Approve a drafted answer and write it as a pin.

        Approval is the moment a machine draft becomes a human decision, so it
        is also the moment it stops being revocable-by-reindex and starts
        behaving like an answer someone wrote.
        """
        proposal = self.knowledge.approve(proposal_id, reviewer=reviewer)
        if proposal is None:
            return False
        self.store.pin(
            proposal.canonical_query,
            proposal.answer,
            citations=proposal.citations,
            author=reviewer or "approved-draft",
        )
        if proposal.supersedes:
            log.info("proposal %s replaces approved answer %s", proposal.id, proposal.supersedes)
        return True


def _build_local(settings: Settings) -> ChatProvider | None:
    if not settings.local_enabled:
        return None
    return OpenAICompatProvider(
        model_id=settings.local_model,
        base_url=settings.local_base_url,
        api_key=settings.local_api_key,
        tier="local",
    )


def _build_frontier(settings: Settings) -> ChatProvider | None:
    """Only built when an operator has explicitly enabled escalation."""
    if not settings.escalation_enabled:
        return None

    if settings.escalation_provider == "anthropic":
        if not settings.anthropic_api_key:
            log.warning("escalation is enabled but OK_ANTHROPIC_API_KEY is unset; staying local")
            return None
        return AnthropicProvider(
            model_id=settings.escalation_model,
            api_key=settings.anthropic_api_key,
            effort=settings.escalation_effort,
        )

    if not settings.openai_api_key:
        log.warning("escalation is enabled but OK_OPENAI_API_KEY is unset; staying local")
        return None
    return OpenAICompatProvider(
        model_id=settings.escalation_model,
        base_url=settings.escalation_base_url,
        api_key=settings.openai_api_key,
        tier="frontier",
    )


def build_engine(settings: Settings) -> Engine:
    store = AnswerStore(settings.db_path)
    knowledge = KnowledgeStore(settings.knowledge_db_path)
    retriever = BM25Retriever(
        target_words=settings.chunk_target_words,
        overlap_words=settings.chunk_overlap_words,
    )
    connector = LocalFilesConnector(settings.documents_dir)
    local = _build_local(settings)
    frontier = _build_frontier(settings)
    engine = Engine(
        settings=settings,
        store=store,
        retriever=retriever,
        cascade=Cascade(
            store=store,
            retriever=retriever,
            settings=settings,
            local=local,
            frontier=frontier,
            knowledge=knowledge,
        ),
        connector=connector,
        knowledge=knowledge,
        local=local,
        frontier=frontier,
    )
    engine.reindex()
    return engine
