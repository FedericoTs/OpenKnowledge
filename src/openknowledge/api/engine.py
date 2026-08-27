"""Assembles the running system from settings."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from ..cache import AnswerStore
from ..cascade import Cascade
from ..config import Settings
from ..connectors import LocalFilesConnector
from ..providers.anthropic_provider import AnthropicProvider
from ..providers.base import ChatProvider
from ..providers.openai_compat import OpenAICompatProvider
from ..retrieval import BM25Retriever

log = logging.getLogger(__name__)


@dataclass
class Engine:
    settings: Settings
    store: AnswerStore
    retriever: BM25Retriever
    cascade: Cascade
    connector: LocalFilesConnector

    def reindex(self) -> tuple[int, int, str, int]:
        """Re-read the corpus and drop answers derived from the old one."""
        documents = self.connector.fetch()
        self.retriever.index(documents)
        evicted = self.store.evict_other_corpus_versions(self.retriever.corpus_version)
        log.info(
            "indexed %d documents into %d chunks (corpus %s); evicted %d stale answers",
            len(documents),
            len(self.retriever),
            self.retriever.corpus_version,
            evicted,
        )
        return len(documents), len(self.retriever), self.retriever.corpus_version, evicted


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
    retriever = BM25Retriever(
        target_words=settings.chunk_target_words,
        overlap_words=settings.chunk_overlap_words,
    )
    connector = LocalFilesConnector(settings.documents_dir)
    engine = Engine(
        settings=settings,
        store=store,
        retriever=retriever,
        cascade=Cascade(
            store=store,
            retriever=retriever,
            settings=settings,
            local=_build_local(settings),
            frontier=_build_frontier(settings),
        ),
        connector=connector,
    )
    engine.reindex()
    return engine
