"""Assembles the running system from settings."""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field, replace
from pathlib import Path

from ..cache import AnswerStore
from ..cache.semantic import SemanticIndex
from ..cascade import Cascade
from ..cascade.budget import Budget
from ..cascade.ladder import Ladder, Rung
from ..config import Settings
from ..connectors import LocalFilesConnector
from ..connectors.drive import DriveClient, DriveConfig, DriveSync
from ..connectors.sharepoint import (
    GraphClient,
    GraphConfig,
    SharePointSync,
    SyncStore,
    SyncSummary,
)
from ..documents.cache import ParseCache
from ..knowledge import (
    IngestReport,
    KnowledgeStore,
    draft_for_documents,
    scan_documents,
    supersession,
)
from ..knowledge.claims import ClaimCache
from ..knowledge.reverify import reverify_changed_documents
from ..providers.anthropic_provider import AnthropicProvider
from ..providers.azure_openai import AzureOpenAIProvider
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
    #: The SharePoint mirror, when one is configured. It writes files into the
    #: documents folder and stamps their readers; the local connector reads
    #: them like any other file.
    sharepoint: SharePointSync | None = None
    #: The Google Drive mirror, when one is configured. Same contract as the
    #: SharePoint one: it writes files into the documents folder and stamps
    #: their readers, and the local connector reads them like any other file.
    drive: DriveSync | None = None
    #: Last fetched corpus, so `learn` does not re-read every file from disk.
    documents: list[Document] = field(default_factory=list)
    #: What the last reindex reported as new or changed.
    last_scan: IngestReport | None = None
    #: The folder stamp as of the last read; see reindex_if_documents_changed.
    _documents_stamp: str = ""
    #: Claims already pulled out of documents whose text has not changed. Lives
    #: on the engine rather than inside the scan so it survives between
    #: rebuilds, which is the whole point: a rebuild happens on every upload,
    #: every delete and every access rule, and re-reading claims out of a
    #: thousand unchanged documents each time was half its clock.
    claims: ClaimCache = field(default_factory=ClaimCache)

    def documents_fingerprint(self) -> str:
        """A cheap stamp of the corpus folder: names, sizes, modification times.

        Deliberately not the content hash the corpus version uses. That one
        answers "is this the same corpus?" and has to read every byte to say
        so; this one answers "is it worth looking?" and only stats. On a
        thousand documents it is a few milliseconds, which is what lets it run
        on a timer without anyone noticing.
        """
        from ..documents import is_supported

        root = Path(self.settings.documents_dir)
        if not root.is_dir():
            return "no-folder"
        digest = hashlib.sha256()
        for path in sorted(root.rglob("*")):
            if not path.is_file() or not is_supported(path):
                continue
            try:
                stat = path.stat()
            except OSError:  # a file being written as we walk past it
                continue
            digest.update(str(path.relative_to(root)).encode("utf-8"))
            digest.update(f"{stat.st_mtime_ns}:{stat.st_size}".encode())
        return digest.hexdigest()[:16]

    def reindex_if_documents_changed(self) -> bool:
        """Re-read the corpus when the folder has moved under us.

        Uploads and deletes re-index themselves, so this exists for the other
        way documents arrive on a shared server: dropped into the folder,
        synced from SharePoint, corrected in place by whoever owns them.
        Measured before this existed - a policy edited on disk left the index
        holding the old text, and the answer cited last year's figure with
        every appearance of being current. The cache is careful never to serve
        a stale answer, but nothing could save it while the index itself had
        never re-read the file.
        """
        stamp = self.documents_fingerprint()
        if stamp == self._documents_stamp:
            return False
        self.reindex()
        return True

    def sync_sharepoint(self) -> SyncSummary | None:
        """Mirror the configured libraries now, and re-index if anything moved.

        None when no mirror is configured. Free apart from the downloads: the
        sync calls Graph, the re-index calls no model.
        """
        return self._sync(self.sharepoint)

    def sync_drive(self) -> SyncSummary | None:
        """The same, for the Google Drive mirror."""
        return self._sync(self.drive)

    def _sync(self, mirror: SharePointSync | DriveSync | None) -> SyncSummary | None:
        if mirror is None:
            return None
        summary = mirror.run()
        if summary.changed:
            self.reindex()
        return summary

    def mirror_principals(self) -> dict[str, frozenset[str]]:
        """Readers every configured mirror stamped, merged.

        Two mirrors cannot claim the same path - each owns its own folder
        under the documents root - so a merge is a union and never a choice
        between two answers about who may read something.
        """
        found: dict[str, frozenset[str]] = {}
        for mirror in (self.sharepoint, self.drive):
            if mirror is not None:
                found.update(mirror.principals_map())
        return found

    def mirror_owns(self, relative_path: str) -> str | None:
        """Which mirror, if any, put this file here - by the name a person knows."""
        for mirror in (self.sharepoint, self.drive):
            if mirror is not None and mirror.owns(relative_path):
                return mirror.label
        return None

    def reindex(self) -> tuple[int, int, str, int]:
        """Re-read the corpus. Free: this never calls a model.

        Runs conflict detection and retires drafts built on text that moved,
        both of which cost nothing. Drafting new answers is `learn`, which is
        separate precisely so that re-indexing cannot spend money by surprise.
        """
        self.documents = self.connector.fetch()
        # A document is retired by whoever replaced it, and in practice the
        # statement is written in the new file rather than added to the old
        # one. Applied here rather than in the connector because it needs the
        # whole corpus to know which document was named.
        self.documents, retired = supersession.apply(self.documents)
        self._documents_stamp = self.documents_fingerprint()
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
            claims=self.claims,
        )
        self.claims.keep_only(self.documents)
        for document_id, announcer in sorted(retired.items()):
            # Said out loud: retrieval excludes a superseded document whenever
            # anything current matches, so an operator should be able to see
            # that this happened and who said it.
            self.last_scan.notes.append(
                f"{document_id} treated as superseded: {announcer} says it replaced it"
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

    def reapply_access(self) -> int:
        """Re-stamp the corpus with the folder rules as they now stand.

        An access change alters who may read a document and nothing else about
        it, so this is the whole job: no file re-read, no passage re-tokenised,
        no contradiction re-detected, and ``corpus_version`` untouched because
        it hashes content. A full rebuild for this was nine seconds on 1,200
        documents and produced a byte-identical index but for one field.

        It stays synchronous, and that is the point rather than an oversight:
        doing it inside the request is what leaves no window in which a rule
        is stored and the index is still serving the old audience.
        """
        mapping = self.connector.access_map()
        self.documents = [
            replace(doc, allowed_principals=mapping.get(doc.document_id, doc.allowed_principals))
            for doc in self.documents
        ]
        return self.retriever.restamp(mapping)

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
                min_support_ratio_cited=self.settings.min_support_ratio_cited,
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
                min_support_ratio_cited=self.settings.min_support_ratio_cited,
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
        parallel=settings.local_parallel,
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

    if settings.escalation_provider == "azure":
        api_key = settings.azure_openai_api_key
        missing = [
            name
            for name, value in (
                ("OK_AZURE_OPENAI_ENDPOINT", settings.azure_openai_endpoint),
                ("OK_AZURE_OPENAI_DEPLOYMENT", settings.azure_openai_deployment),
                ("OK_AZURE_OPENAI_API_KEY", api_key),
            )
            if not value
        ]
        if missing or api_key is None:
            log.warning(
                "escalation is enabled but %s is unset; staying local", " and ".join(missing)
            )
            return None
        if (settings.azure_openai_input_per_mtok is None) != (
            settings.azure_openai_output_per_mtok is None
        ):
            log.warning(
                "only one of OK_AZURE_OPENAI_INPUT_PER_MTOK/OK_AZURE_OPENAI_OUTPUT_PER_MTOK "
                "is set; ignoring both - costs will be flagged as uncounted, not guessed"
            )
        return AzureOpenAIProvider(
            endpoint=settings.azure_openai_endpoint,
            deployment=settings.azure_openai_deployment,
            api_key=api_key,
            api_version=settings.azure_openai_api_version,
            input_per_mtok=settings.azure_openai_input_per_mtok,
            output_per_mtok=settings.azure_openai_output_per_mtok,
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
        # The provider knows its own name; for Azure that is the deployment,
        # not escalation_model, and the ladder description should say so.
        rungs.append(Rung(name=frontier.model_id, provider=frontier, tier=Tier.FRONTIER))
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
        tag_routing=settings.tag_routing,
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


def _build_sharepoint(settings: Settings) -> SharePointSync | None:
    """The mirror, when it is switched on - refusing to run when it could leak.

    Missing settings and sign-in being off are both recorded as a refusal on
    the sync rather than raised: the server still starts and answers from the
    folder it has, and the status on /manage says exactly why the mirror is
    not running.
    """
    if not settings.sharepoint_enabled:
        return None
    missing = [
        name
        for name, value in (
            ("OK_SHAREPOINT_TENANT_ID", settings.sharepoint_tenant_id),
            ("OK_SHAREPOINT_CLIENT_ID", settings.sharepoint_client_id),
            ("OK_SHAREPOINT_CLIENT_SECRET", settings.sharepoint_client_secret),
            ("OK_SHAREPOINT_SITE", settings.sharepoint_site),
        )
        if not value
    ]
    refusal: str | None = None
    if missing:
        refusal = f"SharePoint sync is on but {' and '.join(missing)} is unset"
    elif settings.auth_mode != "oidc" and settings.sharepoint_require_signin:
        refusal = (
            "sign-in is off, so no reader could be enforced and every mirrored document "
            "would be readable by whoever reaches the widget; turn on OK_AUTH_MODE=oidc, "
            "or set OK_SHAREPOINT_REQUIRE_SIGNIN=false to mirror anyway"
        )
    config = GraphConfig(
        tenant_id=settings.sharepoint_tenant_id,
        client_id=settings.sharepoint_client_id,
        client_secret=settings.sharepoint_client_secret or "",
        site=settings.sharepoint_site,
        drives=tuple(settings.sharepoint_drives),
        graph_url=settings.sharepoint_graph_url,
        login_url=settings.sharepoint_login_url,
    )
    Path(settings.data_dir).mkdir(parents=True, exist_ok=True)
    return SharePointSync(
        GraphClient(config),
        documents_dir=settings.documents_dir,
        store=SyncStore(Path(settings.data_dir) / "sharepoint.db"),
        permissions_refresh_seconds=settings.sharepoint_permissions_refresh_seconds,
        refusal=refusal,
    )


def _mirror_refusal(settings: Settings, missing: list[str], require_signin: bool) -> str | None:
    """Why a mirror must not run, in the words an operator can act on."""
    if missing:
        return f"the mirror is on but {' and '.join(missing)} is unset"
    if settings.auth_mode != "oidc" and require_signin:
        return (
            "sign-in is off, so no reader could be enforced and every mirrored document "
            "would be readable by whoever reaches the widget; turn on OK_AUTH_MODE=oidc, "
            "or turn the require-sign-in setting off to mirror anyway"
        )
    return None


def _build_drive(settings: Settings) -> DriveSync | None:
    """The Drive mirror, when it is switched on - refusing when it could leak."""
    if not settings.drive_enabled:
        return None
    missing = [
        name
        for name, value in (
            ("OK_DRIVE_CLIENT_EMAIL", settings.drive_client_email),
            ("OK_DRIVE_PRIVATE_KEY", settings.drive_private_key),
            ("OK_DRIVE_DOMAIN", settings.drive_domain),
        )
        if not value
    ]
    config = DriveConfig(
        client_email=settings.drive_client_email,
        private_key=settings.drive_private_key or "",
        subject=settings.drive_subject,
        domain=settings.drive_domain,
        drive_ids=tuple(settings.drive_ids),
        api_url=settings.drive_api_url,
        token_url=settings.drive_token_url,
    )
    Path(settings.data_dir).mkdir(parents=True, exist_ok=True)
    return DriveSync(
        DriveClient(config),
        documents_dir=settings.documents_dir,
        store=SyncStore(Path(settings.data_dir) / "drive.db"),
        permissions_refresh_seconds=settings.drive_permissions_refresh_seconds,
        refusal=_mirror_refusal(settings, missing, settings.drive_require_signin),
    )


def build_engine(settings: Settings) -> Engine:
    store = AnswerStore(settings.db_path)
    knowledge = KnowledgeStore(settings.knowledge_db_path)
    retriever = _build_retriever(settings)
    sharepoint = _build_sharepoint(settings)
    drive = _build_drive(settings)
    connector = LocalFilesConnector(
        settings.documents_dir,
        pdf_backend=settings.pdf_backend,
        # Bound to the store, not copied from it: folder access rules are
        # admin decisions that change at runtime, and each re-index reads
        # the ones in force.
        folder_rules=knowledge.folder_rules,
        # Likewise the readers each mirror stamped on the files it wrote.
        file_principals=None,
        # Parsing dominates every scan on the formats a company actually has:
        # 780ms for one small PDF against 6ms for the same words in markdown,
        # almost all of it a Java process starting up. Persisted, so a restart
        # does not re-pay the first build.
        parses=ParseCache(settings.parse_cache_path),
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
        sharepoint=sharepoint,
        drive=drive,
    )
    # Bound after the engine exists so one callable covers every mirror, and
    # so a mirror added later is read without rebuilding the connector.
    connector.file_principals = engine.mirror_principals
    engine.reindex()
    return engine
