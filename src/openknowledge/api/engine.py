"""Assembles the running system from settings."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from ..cache import AnswerStore
from ..cache.semantic import SemanticIndex
from ..cascade import Cascade
from ..cascade.budget import Budget
from ..cascade.ladder import Ladder, Rung
from ..config import Settings
from ..connectors import LocalFilesConnector
from ..knowledge import IngestReport, KnowledgeStore, draft_for_documents, scan_documents
from ..knowledge.reverify import reverify_changed_documents
from ..providers.anthropic_provider import AnthropicProvider
from ..providers.base import ChatProvider
from ..providers.openai_compat import OpenAICompatProvider
from ..retrieval import BM25Retriever
from ..retrieval.base import Document, Retriever
from ..retrieval.embed import Embedder
from ..retrieval.hybrid import HybridRetriever
from ..retrieval.vectorstore import VectorCache
from ..types import Tier

log = logging.getLogger(__name__)


@dataclass
class Engine:
    settings: Settings
    store: AnswerStore
    retriever: Retriever
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
        if self.cascade.semantic is not None:
            # Question vectors describe cached answers; when the answers for a
            # superseded corpus go, their vectors go with them, or a stale
            # vector would keep nominating an answer that no longer exists.
            self.cascade.semantic.evict_other_corpus_versions(self.retriever.corpus_version)
        self.last_scan = scan_documents(
            self.documents,
            store=self.knowledge,
            retriever=self.retriever,
            min_conflict_overlap=self.settings.conflict_min_overlap,
            deontic_strictness=self.settings.deontic_strictness,
        )
        for skipped in self.connector.skipped:
            self.last_scan.notes.append(f"skipped {skipped.path}: {skipped.reason}")
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
        context_tokens=settings.local_context_tokens,
        timeout=settings.local_timeout_seconds,
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


def _build_ladder(settings: Settings, local: ChatProvider | None) -> Ladder:
    """Assemble the rungs, cheapest first.

    The cheap rung is whatever `local_*` points at - a box you own or an
    open-weight endpoint, the adapter is the same. Above it come the `ladder`
    rungs in the order given, and the frontier last. Every rung answers from the
    same passages under the same prompt and is graded by the same gate, so the
    only thing a rung changes is the price of trying again.
    """
    rungs: list[Rung] = []
    if local is not None:
        rungs.append(Rung(name=settings.local_model, provider=local, tier=Tier.LOCAL))

    for spec in settings.ladder:
        model_id, _, base_url = spec.partition("@")
        model_id = model_id.strip()
        if not model_id:
            log.warning("ignoring empty ladder entry %r", spec)
            continue
        rungs.append(
            Rung(
                name=model_id,
                provider=OpenAICompatProvider(
                    model_id=model_id,
                    base_url=base_url.strip() or settings.escalation_base_url,
                    api_key=settings.ladder_api_key or settings.openai_api_key,
                    tier="frontier",
                ),
                tier=Tier.FRONTIER,
            )
        )

    frontier = _build_frontier(settings)
    if frontier is not None:
        rungs.append(Rung(name=settings.escalation_model, provider=frontier, tier=Tier.FRONTIER))
    return Ladder(tuple(rungs))


def _build_retriever(settings: Settings) -> Retriever:
    """BM25 alone, or BM25 with a dense half fused onto it.

    Always wrapped the same way round: lexical search is the thing that works
    with nothing installed, and the dense half is an addition that is allowed
    to be missing. An unreachable embedding endpoint costs quality, never
    service.
    """
    lexical = BM25Retriever(
        target_words=settings.chunk_target_words,
        overlap_words=settings.chunk_overlap_words,
    )
    if not settings.embedding_enabled:
        return lexical

    embedder = Embedder(
        model=settings.embedding_model,
        base_url=settings.embedding_base_url or settings.local_base_url,
        api_key=settings.local_api_key,
        timeout=settings.local_timeout_seconds,
    )
    cache = VectorCache(Path(settings.data_dir) / settings.vectors_db)
    return HybridRetriever(lexical=lexical, embedder=embedder, cache=cache.load(), store=cache)


def _build_semantic(
    settings: Settings, store: AnswerStore, retriever: Retriever
) -> SemanticIndex | None:
    """The semantic cache, when there is an embedder to power it.

    It reuses the hybrid retriever's own embedder - same model, same endpoint,
    same fingerprint - so a question and the corpus always live in one vector
    space. No embedder (embeddings off, or a BM25-only install) simply means
    no semantic cache, which is the correct degradation: the exact cache and
    every other tier are untouched.
    """
    if not (settings.embedding_enabled and settings.semantic_cache_enabled):
        return None
    embedder = getattr(retriever, "embedder", None)
    if embedder is None:
        return None
    return SemanticIndex(store, embedder)


def build_engine(settings: Settings) -> Engine:
    store = AnswerStore(settings.db_path)
    knowledge = KnowledgeStore(settings.knowledge_db_path)
    retriever = _build_retriever(settings)
    connector = LocalFilesConnector(
        settings.documents_dir,
        pdf_backend=settings.pdf_backend,
        # Bound to the store, not copied from it: folder access rules are
        # admin decisions that change at runtime, and each re-index reads
        # the ones in force.
        folder_rules=knowledge.folder_rules,
    )
    local = _build_local(settings)
    ladder = _build_ladder(settings, local)
    frontier = ladder.rungs[-1].provider if len(ladder) > 1 else None
    log.info("escalation ladder: %s", ladder.describe())
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
            ladder=ladder,
            budget=Budget(
                daily_usd=settings.budget_daily_usd,
                expected_questions_per_day=settings.budget_expected_questions_per_day,
            ),
            semantic=_build_semantic(settings, store, retriever),
        ),
        connector=connector,
        knowledge=knowledge,
        local=local,
        frontier=frontier,
    )
    engine.reindex()
    return engine
