"""The cascade: try cheap, verify, escalate only when verification fails.

The ordering is the product. Each tier is asked to answer from the same
retrieved passages under the same rules, and its answer is graded by the same
grounding gate. The only difference between tiers is the price. That is what
makes "use the cheap model" a cost decision rather than a quality gamble - a
cheap answer that cannot pass the gate never reaches the user, it just triggers
the next tier.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, replace

from ..cache import AnswerStore, KeyContext, answer_key
from ..canonical import canonicalize_query
from ..config import Settings
from ..costs import PricingError, Usage, cost_usd, get_price
from ..knowledge.relevance import (
    DEFAULT_MIN_OVERLAP,
    describe_for_user,
    relevant_conflicts,
)
from ..knowledge.store import KnowledgeStore, StoredConflict
from ..prompts import (
    PROMPT_VERSION,
    REFUSAL_TEXT,
    SYSTEM_PROMPT,
    UNAVAILABLE_TEXT,
    format_context,
)
from ..providers.base import ChatProvider, ProviderError
from ..retrieval.base import Chunk
from ..retrieval.bm25 import BM25Retriever
from ..retrieval.grounding import check_grounding
from ..retrieval.rerank import Reranker, StructuralReranker
from ..types import Answer, Citation, Tier
from .budget import Budget, BudgetGovernor
from .corpus import describe, recognise
from .ladder import Ladder, Rung

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class _Attempt:
    """One provider call: its answer if it passed the gate, and what it cost.

    The cost is recorded either way. A call whose answer failed grounding still
    consumed tokens, and a ledger that forgets those understates the very number
    this project is judged on.
    """

    answer: Answer | None
    usage: Usage
    cost_usd: float
    notes: tuple[str, ...] = ()
    #: Whether the model actually produced text. False means the call never
    #: happened - unreachable endpoint, missing model, a prompt that would not
    #: fit - and so the sources were never assessed. True with `answer` None
    #: means it read them and its answer failed the gate. The two look identical
    #: from here and mean opposite things to whoever asked.
    reached_model: bool = False


#: How much of a chunk to keep as the citation snippet shown to the user.
_SNIPPET_CHARS = 280


def _default_reranker(settings: Settings) -> Reranker | None:
    """On unless an operator turns it off. It is free and it lowers escalation."""
    if settings.rerank_candidates <= 0:
        return None
    return StructuralReranker(max_per_document=settings.rerank_max_per_document)


class Cascade:
    """Routes one question through the tiers, cheapest first."""

    def __init__(
        self,
        *,
        store: AnswerStore,
        retriever: BM25Retriever,
        settings: Settings,
        local: ChatProvider | None = None,
        frontier: ChatProvider | None = None,
        knowledge: KnowledgeStore | None = None,
        ladder: Ladder | None = None,
        budget: Budget | None = None,
        reranker: Reranker | None = None,
    ) -> None:
        self.store = store
        self.retriever = retriever
        self.settings = settings
        self.local = local
        self.frontier = frontier
        self.knowledge = knowledge
        # `local` and `frontier` remain the two-rung shorthand, because most
        # deployments have exactly those and should not have to build a ladder
        # to say so. An explicit ladder replaces them entirely.
        self.ladder = ladder if ladder is not None else self._default_ladder()
        self.governor = BudgetGovernor(store=store, budget=budget or Budget())
        self.reranker = reranker if reranker is not None else _default_reranker(settings)

    def _default_ladder(self) -> Ladder:
        rungs: list[Rung] = []
        if self.settings.local_enabled and self.local is not None:
            rungs.append(Rung(name="local", provider=self.local, tier=Tier.LOCAL))
        if self.settings.escalation_enabled and self.frontier is not None:
            rungs.append(Rung(name="frontier", provider=self.frontier, tier=Tier.FRONTIER))
        return Ladder(tuple(rungs))

    # -- keying -----------------------------------------------------------
    @property
    def route_id(self) -> str:
        """Fingerprint of the models this cascade can use.

        The cache key needs *a* model identity, but an answer here may come from
        either tier, so pinning the key to one model would be wrong in both
        directions: swap the local model and stale answers survive; resolve at a
        different tier and an identical question misses. Hashing the whole route
        fixes both - any model change invalidates, and the tier that happened to
        answer does not affect the key.
        """
        rungs = "|".join(f"{r.name}:{r.model_id}:{r.k or '-'}" for r in self.ladder)
        raw = f"{rungs}|{self.settings.escalation_effort}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]

    def _key_context(self) -> KeyContext:
        return KeyContext(
            corpus_version=self.retriever.corpus_version,
            prompt_version=f"{PROMPT_VERSION}:{_suffix_hash(self.settings.system_prompt_suffix)}",
            policy_version=(
                f"k{self.settings.retrieval_k}"
                f":s{self.settings.min_support_ratio}"
                f":c{int(self.settings.require_citations)}"
                f":r{self.settings.rerank_candidates}"
                f":d{self.settings.rerank_max_per_document}"
            ),
            model_id=self.route_id,
        )

    def _system_prompt(self) -> str:
        suffix = self.settings.system_prompt_suffix.strip()
        return (
            f"{SYSTEM_PROMPT}\n\nAdditional instructions from your administrator:\n{suffix}"
            if suffix
            else SYSTEM_PROMPT
        )

    # -- main entry point -------------------------------------------------
    async def answer(
        self,
        question: str,
        *,
        principals: frozenset[str] | None = None,
        channel: str | None = None,
    ) -> Answer:
        canonical = canonicalize_query(question)
        key = answer_key(question, self._key_context())

        result = await self._resolve(question, canonical, key, principals)
        self.store.record(canonical, result, channel=channel)
        return result

    async def _resolve(
        self,
        question: str,
        canonical: str,
        key: str,
        principals: frozenset[str] | None,
    ) -> Answer:
        # Before anything else, and before any cost: a question about the
        # collection rather than from it. No retriever can answer "what
        # documents do you have" - it has no subject to match - so it used to
        # come back as "that isn't covered by the documents I have", which is a
        # system that plainly does know, saying it does not. Free, instant, and
        # correct by construction rather than by a model reading a passage.
        asking_about_the_corpus = recognise(question)
        if asking_about_the_corpus is not None:
            titles, hidden = self.retriever.documents_visible_to(principals)
            return Answer(
                text=describe(titles, chunks=len(self.retriever), hidden=hidden),
                tier=Tier.CORPUS,
                model_id="none",
                cache_key=key,
                grounded=True,
                notes=("answered from the index; no model was called",),
            )

        # A contested claim is where the bot would otherwise be confidently
        # wrong, so an unresolved disagreement outranks everything except a pin
        # that has already accounted for it.
        contested: list[StoredConflict] = []
        near_misses = 0
        if self.knowledge is not None and self.settings.block_on_conflict:
            open_conflicts = self.knowledge.open_conflicts()
            contested = relevant_conflicts(question, open_conflicts)
            if not contested:
                # Disagreements that were relevant but scored under the bar that
                # would have refused. The decision was closest here, and saying
                # so on the answer costs nothing.
                near_misses = len(
                    relevant_conflicts(
                        question, open_conflicts, min_overlap=DEFAULT_MIN_OVERLAP * 0.6
                    )
                )

        # L0 - a human already wrote this answer.
        pin = self.store.get_pin(canonical)
        if pin is not None:
            # A pin written *before* the disagreement appeared was written by
            # someone who had not seen the new document. Serving it is exactly
            # the failure this project exists to prevent - the answer looks
            # human-authored and authoritative, and it is out of date. A pin
            # written after the conflict was detected is a decision about it,
            # and wins.
            unaccounted = [c for c in contested if c.detected_at > pin.updated_at]
            cited = {c.document_id for c in pin.citations}
            if not unaccounted and self.retriever.visible_to(cited, principals):
                return Answer(
                    text=pin.answer,
                    tier=Tier.PINNED,
                    model_id="pinned",
                    cache_key=key,
                    citations=pin.citations,
                )
            if unaccounted:
                log.info(
                    "pin for %r withheld: it predates %d unresolved conflict(s)",
                    canonical,
                    len(unaccounted),
                )

        if contested:
            return Answer(
                text=describe_for_user(contested),
                tier=Tier.CONTESTED,
                model_id="none",
                cache_key=key,
                grounded=True,
                notes=tuple(c.describe() for c in contested[:3]),
            )

        # L1 - we answered this exact question, under this exact corpus.
        cached = self.store.get(key)
        if cached is not None:
            cited = {c.document_id for c in cached.citations}
            if self.retriever.visible_to(cited, principals):
                return Answer(
                    text=cached.answer,
                    tier=Tier.EXACT_CACHE,
                    model_id=cached.model_id,
                    cache_key=key,
                    citations=cached.citations,
                    notes=(f"served from cache (hit {cached.hits})",),
                )
            log.info("cache hit withheld: asker cannot access all cited sources")

        # L2 - an answer drafted from the documents at upload time. It passed
        # the same grounding gate a live answer does, so it is a precomputed
        # cache entry rather than a pin: served, marked as unreviewed, and
        # revocable. Human approval is what promotes it to a pin.
        if self.knowledge is not None and self.settings.serve_drafts:
            draft = self.knowledge.draft_for(canonical)
            if draft is not None:
                cited = {c.document_id for c in draft.citations}
                if self.retriever.visible_to(cited, principals):
                    return Answer(
                        text=draft.answer,
                        tier=Tier.DRAFT,
                        model_id="drafted",
                        cache_key=key,
                        citations=draft.citations,
                        notes=(
                            "auto-drafted from "
                            f"{', '.join(draft.origin_documents)} and not yet reviewed "
                            "by a person",
                        ),
                    )

        # The semantic cache tier is not implemented yet - see ROADMAP.

        # Retrieve once, wide enough for the widest rung; every rung reads a
        # prefix of the same ranked list. Searching per rung would let two rungs
        # answer from different evidence, which is the property that makes
        # climbing the ladder safe rather than a quality gamble.
        wanted = self.ladder.widest_k(self.settings.retrieval_k)
        # Retrieve wider than needed when a reranker can use the slack. BM25
        # ranks chunks independently, so its top-k can be six views of one
        # paragraph; the extra candidates cost nothing and are what the reranker
        # picks from.
        candidates = max(wanted, self.settings.rerank_candidates) if self.reranker else wanted
        hits = self.retriever.search(question, k=candidates, principals=principals)
        if self.reranker is not None:
            hits = self.reranker.rerank(question, hits, k=wanted)
        chunks = [h.chunk for h in hits]
        if not chunks:
            return Answer(
                text=REFUSAL_TEXT,
                tier=Tier.REFUSED,
                model_id="none",
                cache_key=key,
                notes=("no documents matched this question",),
            )

        system = self._system_prompt()
        notes: list[str] = []
        # Spend that has already happened on this question, whatever rung
        # eventually answers it - or none.
        spent_usd = 0.0
        spent_usage = Usage()

        affordable, budget_state, withheld = self.governor.allowed(
            self.ladder,
            prompt_chars=len(system) + sum(len(c.text) for c in chunks) + len(question),
            max_tokens=self.settings.max_answer_tokens,
        )
        notes.extend(withheld)

        climbed_from: Tier | None = None
        #: Did any rung actually read the sources? Decides which of the two
        #: refusals is true at the end of this loop.
        any_rung_ran = False
        for rung in affordable:
            rung_chunks = chunks[: rung.k] if rung.k is not None else chunks
            attempt = await self._try_provider(
                rung.provider,
                rung.tier,
                system,
                format_context(rung_chunks),
                question,
                rung_chunks,
                key,
                max_tokens=rung.max_tokens or self.settings.max_answer_tokens,
                near_misses=near_misses,
            )
            spent_usd += attempt.cost_usd
            spent_usage += attempt.usage
            notes.extend(attempt.notes)
            any_rung_ran = any_rung_ran or attempt.reached_model

            if attempt.answer is not None:
                # Bill the answer for every failed attempt below it too, so the
                # ledger reflects what the question really cost rather than what
                # the winning call cost.
                answer = replace(
                    attempt.answer,
                    cost_usd=spent_usd,
                    usage=spent_usage,
                    escalated_from=climbed_from,
                    notes=(*notes, *attempt.answer.notes),
                )
                self.store.put(key, canonical, answer, self.retriever.corpus_version)
                return answer
            climbed_from = rung.tier

        if not self.ladder:
            notes.append("no model is configured, so nothing could be answered from the documents")
        elif len(self.ladder) == 1 and not self.settings.escalation_enabled:
            # Worth naming: the single most common reason a deployment refuses
            # more than it should is that nobody turned escalation on.
            notes.append(
                "escalation is disabled, so there was no rung above "
                f"{self.ladder.rungs[0].name!r} to fall back to"
            )
        elif not affordable:
            notes.append("every rung was withheld by the budget ceiling")
        elif len(affordable) < len(self.ladder):
            notes.append(
                "the rungs that might have grounded this were withheld by the budget ceiling; "
                "it recovers as spending falls back on pace, and this refusal is not cached"
            )
        elif budget_state.enabled:
            notes.append(f"budget: {budget_state.describe()}")

        # Two different refusals, and saying the wrong one is itself a wrong
        # answer. If some rung read the sources and could not ground an answer,
        # "not covered by the documents" is true and the sources are worth
        # showing. If no rung ever ran - nothing reachable, or the budget
        # withheld them all - the documents were never assessed, so claiming
        # they do not cover it is false, and listing them as "sources" under
        # that claim is worse: it implies something read them.
        #
        # Not cached either way, so a recovered endpoint or a later corpus
        # update gets a fresh attempt. The cost is still reported: rejected
        # answers are not free.
        if any_rung_ran:
            return Answer(
                text=REFUSAL_TEXT,
                tier=Tier.REFUSED,
                model_id="none",
                cache_key=key,
                citations=_citations(chunks),
                usage=spent_usage,
                cost_usd=spent_usd,
                grounded=False,
                notes=tuple(notes) or ("no rung produced a grounded answer",),
            )

        notes.append(
            f"retrieval found {len(chunks)} passage(s); nothing read them, so the documents "
            "have not been ruled out"
        )
        return Answer(
            text=UNAVAILABLE_TEXT,
            tier=Tier.REFUSED,
            model_id="none",
            cache_key=key,
            usage=spent_usage,
            cost_usd=spent_usd,
            grounded=False,
            notes=tuple(notes),
        )

    async def _try_provider(
        self,
        provider: ChatProvider,
        tier: Tier,
        system: str,
        context: str,
        question: str,
        chunks: list[Chunk],
        key: str,
        max_tokens: int | None = None,
        near_misses: int = 0,
    ) -> _Attempt:
        """Call one provider, returning its answer only if it passes the gate."""
        try:
            completion = await provider.complete(
                system=system,
                context=context,
                question=question,
                max_tokens=max_tokens or self.settings.max_answer_tokens,
            )
        except ProviderError as exc:
            log.warning("%s tier failed: %s", tier.value, exc)
            # Nothing was generated, so nothing was billed.
            return _Attempt(
                None,
                Usage(),
                0.0,
                (
                    f"{tier.value} tier unavailable: {exc}",
                    "`openknowledge model status` checks whether that endpoint is up",
                ),
                reached_model=False,
            )

        cost, cost_notes = _price(completion.usage, provider)

        report = check_grounding(
            completion.text,
            chunks,
            min_support_ratio=self.settings.min_support_ratio,
            require_citations=self.settings.require_citations,
        )
        if not report.passed:
            reasons = "; ".join(report.reasons)
            log.info("%s answer rejected by grounding gate: %s", tier.value, reasons)
            # The tokens were spent even though the answer is being discarded.
            return _Attempt(
                None,
                completion.usage,
                cost,
                (*cost_notes, f"{tier.value} answer rejected: {reasons}"),
                reached_model=True,
            )

        cited = set(report.cited_ids)
        answer = Answer(
            text=completion.text,
            tier=tier,
            model_id=completion.model_id,
            cache_key=key,
            citations=_citations([c for c in chunks if c.document_id in cited] or chunks),
            usage=completion.usage,
            cost_usd=cost,
            grounded=True,
            notes=cost_notes,
            support=round(report.support_ratio, 3),
        )
        return _Attempt(answer, completion.usage, cost, cost_notes, reached_model=True)


def _suffix_hash(suffix: str) -> str:
    return hashlib.sha256(suffix.strip().encode("utf-8")).hexdigest()[:8]


def _citations(chunks: list[Chunk]) -> tuple[Citation, ...]:
    """One citation per document, keeping the first chunk that matched."""
    seen: dict[str, Citation] = {}
    for chunk in chunks:
        if chunk.document_id in seen:
            continue
        snippet = chunk.text[:_SNIPPET_CHARS]
        if len(chunk.text) > _SNIPPET_CHARS:
            snippet += "..."
        seen[chunk.document_id] = Citation(
            document_id=chunk.document_id,
            document_title=chunk.document_title,
            snippet=snippet,
            locator=chunk.locator,
            url=chunk.url,
        )
    return tuple(seen.values())


def _price(usage: Usage, provider: ChatProvider) -> tuple[float, tuple[str, ...]]:
    """Cost of a call, with a note when we cannot price it honestly.

    The local *tier* is not the same thing as a free call. An open-weight model
    on Together or Groq reaches the same tier through the same adapter and bills
    per token, so what decides the price is whether there is an invoice behind
    the endpoint - `self_hosted` - not what the tier is named. Pricing every
    local-tier call at zero would understate the bill by exactly the amount an
    operator most needs to see.
    """
    tier = getattr(provider, "tier", "")
    self_hosted = getattr(provider, "self_hosted", True)
    model_id = "local" if tier == "local" and self_hosted else provider.model_id
    try:
        return cost_usd(usage, get_price(model_id)), ()
    except PricingError:
        # Reporting $0 for a call that really cost money would corrupt the whole
        # ledger, so flag it instead of guessing.
        return 0.0, (
            f"cost not counted: no verified price for {provider.model_id!r} in pricing.yaml",
        )
