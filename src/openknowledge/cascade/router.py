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
from ..knowledge.relevance import describe_for_user, relevant_conflicts
from ..knowledge.store import KnowledgeStore, StoredConflict
from ..prompts import PROMPT_VERSION, REFUSAL_TEXT, SYSTEM_PROMPT, format_context
from ..providers.base import ChatProvider, ProviderError
from ..retrieval.base import Chunk
from ..retrieval.bm25 import BM25Retriever
from ..retrieval.grounding import check_grounding
from ..types import Answer, Citation, Tier

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


#: How much of a chunk to keep as the citation snippet shown to the user.
_SNIPPET_CHARS = 280


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
    ) -> None:
        self.store = store
        self.retriever = retriever
        self.settings = settings
        self.local = local
        self.frontier = frontier
        self.knowledge = knowledge

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
        local = getattr(self.local, "model_id", "-")
        frontier = getattr(self.frontier, "model_id", "-")
        raw = f"{local}|{frontier}|{self.settings.escalation_effort}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]

    def _key_context(self) -> KeyContext:
        return KeyContext(
            corpus_version=self.retriever.corpus_version,
            prompt_version=f"{PROMPT_VERSION}:{_suffix_hash(self.settings.system_prompt_suffix)}",
            policy_version=(
                f"k{self.settings.retrieval_k}"
                f":s{self.settings.min_support_ratio}"
                f":c{int(self.settings.require_citations)}"
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
        # A contested claim is where the bot would otherwise be confidently
        # wrong, so an unresolved disagreement outranks everything except a pin
        # that has already accounted for it.
        contested: list[StoredConflict] = []
        if self.knowledge is not None and self.settings.block_on_conflict:
            contested = relevant_conflicts(question, self.knowledge.open_conflicts())

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

        # Retrieve once; every model tier below answers from this same evidence.
        hits = self.retriever.search(question, k=self.settings.retrieval_k, principals=principals)
        chunks = [h.chunk for h in hits]
        if not chunks:
            return Answer(
                text=REFUSAL_TEXT,
                tier=Tier.REFUSED,
                model_id="none",
                cache_key=key,
                notes=("no documents matched this question",),
            )

        context = format_context(chunks)
        system = self._system_prompt()
        notes: list[str] = []
        # Spend that has already happened on this question, whatever tier
        # eventually answers it - or none.
        spent_usd = 0.0
        spent_usage = Usage()

        # L3 - the local model. No per-token invoice.
        if self.settings.local_enabled and self.local is not None:
            attempt = await self._try_provider(
                self.local, Tier.LOCAL, system, context, question, chunks, key
            )
            spent_usd += attempt.cost_usd
            spent_usage += attempt.usage
            notes.extend(attempt.notes)
            if attempt.answer is not None:
                answer = attempt.answer
                self.store.put(key, canonical, answer, self.retriever.corpus_version)
                return answer

        # L4 - the paid tier, reached only because the cheap one could not be verified.
        if self.settings.escalation_enabled and self.frontier is not None:
            attempt = await self._try_provider(
                self.frontier, Tier.FRONTIER, system, context, question, chunks, key
            )
            spent_usage += attempt.usage
            if attempt.answer is not None:
                # Bill the escalated answer for the failed cheap attempt too, so
                # the ledger reflects what the question really cost.
                answer = replace(
                    attempt.answer,
                    cost_usd=attempt.cost_usd + spent_usd,
                    usage=spent_usage,
                    escalated_from=Tier.LOCAL if self.local is not None else None,
                    notes=(*notes, *attempt.answer.notes),
                )
                self.store.put(key, canonical, answer, self.retriever.corpus_version)
                return answer
            spent_usd += attempt.cost_usd
            notes.extend(attempt.notes)
        elif notes:
            notes.append("escalation is disabled, so there was nowhere cheaper to fall back to")

        # Nothing could be grounded. Say so rather than passing along a guess -
        # and do not cache it, so a later corpus update gets a fresh attempt.
        # The cost is still reported: rejected answers are not free.
        return Answer(
            text=REFUSAL_TEXT,
            tier=Tier.REFUSED,
            model_id="none",
            cache_key=key,
            citations=_citations(chunks),
            usage=spent_usage,
            cost_usd=spent_usd,
            grounded=False,
            notes=tuple(notes) or ("no tier produced a grounded answer",),
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
    ) -> _Attempt:
        """Call one provider, returning its answer only if it passes the gate."""
        try:
            completion = await provider.complete(
                system=system,
                context=context,
                question=question,
                max_tokens=self.settings.max_answer_tokens,
            )
        except ProviderError as exc:
            log.warning("%s tier failed: %s", tier.value, exc)
            # Nothing was generated, so nothing was billed.
            return _Attempt(None, Usage(), 0.0, (f"{tier.value} tier unavailable: {exc}",))

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
        )
        return _Attempt(answer, completion.usage, cost, cost_notes)


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
