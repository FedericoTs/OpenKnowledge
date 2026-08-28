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
from collections.abc import AsyncIterator
from dataclasses import dataclass, replace
from typing import Any

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
from ..providers.base import ChatProvider, Completion, Message, ProviderError
from ..retrieval.base import Chunk, Retriever
from ..retrieval.grounding import check_grounding
from ..retrieval.rerank import Reranker, StructuralReranker
from ..types import Answer, Citation, Tier
from .budget import Budget, BudgetGovernor
from .corpus import describe, recognise
from .followup import resolve
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
        retriever: Retriever,
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
        history: tuple[Message, ...] = (),
    ) -> Answer:
        result: Answer | None = None
        async for event in self.answer_stream(
            question, principals=principals, channel=channel, history=history
        ):
            if event["type"] == "final":
                result = event["answer"]
        assert result is not None  # the event stream always ends in a final
        return result

    async def answer_stream(
        self,
        question: str,
        *,
        principals: frozenset[str] | None = None,
        channel: str | None = None,
        history: tuple[Message, ...] = (),
    ) -> AsyncIterator[dict[str, Any]]:
        """The same resolution as :meth:`answer`, narrated while it happens.

        One code path produces both: :meth:`answer` drains this stream and keeps
        only the final. Anything else - a separate streaming resolver - would
        let the two disagree about caching, notes or tier, and the whole
        determinism argument rests on there being exactly one way a question is
        answered.

        Events, in order of appearance: ``status`` (retrieval is done, a model
        is about to run), ``provisional`` (a self-hosted rung is streaming; text
        after this is ungated), ``delta`` (a piece of provisional text),
        ``retract`` (the gate rejected what was just streamed - the reader must
        see it withdrawn, because showing it was the price of streaming and
        withdrawing it is the honesty), and exactly one terminal ``final``
        carrying the Answer, which is byte-identical to what :meth:`answer`
        would have returned.
        """
        # A follow-up is rewritten into the standalone question it means BEFORE
        # anything is keyed, so the cache, the canonical form and the ledger all
        # see a real question. That is what keeps "same question, same answer"
        # true in a conversation: the raw fragment never becomes a key.
        resolution = await resolve(question, history, self._resolver_provider())
        if resolution.rewritten:
            yield {"type": "resolved", "question": resolution.question}

        canonical = canonicalize_query(resolution.question)
        key = answer_key(resolution.question, self._key_context())
        async for event in self._events(resolution.question, canonical, key, principals):
            if event["type"] == "final":
                answer = event["answer"]
                if resolution.note:
                    answer = replace(answer, notes=(resolution.note, *answer.notes))
                if resolution.usage.input_tokens or resolution.usage.output_tokens:
                    # The interpretation call is part of what this question
                    # cost; a ledger that forgets it understates.
                    answer = replace(answer, usage=answer.usage + resolution.usage)
                event = {"type": "final", "answer": answer}
                self.store.record(canonical, answer, channel=channel)
            yield event

    def _resolver_provider(self) -> ChatProvider | None:
        """Who interprets follow-ups: the cheapest self-hosted rung, or nobody.

        Follow-up interpretation is overhead on every dependent question, so it
        runs where tokens are free. With no self-hosted rung it is skipped and
        the question is answered as asked - degraded, said so, never billed.
        """
        for rung in self.ladder.rungs:
            if getattr(rung.provider, "self_hosted", False):
                return rung.provider
        return None

    async def _events(
        self,
        question: str,
        canonical: str,
        key: str,
        principals: frozenset[str] | None,
    ) -> AsyncIterator[dict[str, Any]]:
        # Before anything else, and before any cost: a question about the
        # collection rather than from it. No retriever can answer "what
        # documents do you have" - it has no subject to match - so it used to
        # come back as "that isn't covered by the documents I have", which is a
        # system that plainly does know, saying it does not. Free, instant, and
        # correct by construction rather than by a model reading a passage.
        asking_about_the_corpus = recognise(question)
        if asking_about_the_corpus is not None:
            titles, hidden = self.retriever.documents_visible_to(principals)
            yield _final(
                Answer(
                    text=describe(titles, chunks=len(self.retriever), hidden=hidden),
                    tier=Tier.CORPUS,
                    model_id="none",
                    cache_key=key,
                    grounded=True,
                    notes=("answered from the index; no model was called",),
                )
            )
            return

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
                yield _final(
                    Answer(
                        text=pin.answer,
                        tier=Tier.PINNED,
                        model_id="pinned",
                        cache_key=key,
                        citations=pin.citations,
                    )
                )
                return
            if unaccounted:
                log.info(
                    "pin for %r withheld: it predates %d unresolved conflict(s)",
                    canonical,
                    len(unaccounted),
                )

        if contested:
            yield _final(
                Answer(
                    text=describe_for_user(contested),
                    tier=Tier.CONTESTED,
                    model_id="none",
                    cache_key=key,
                    grounded=True,
                    notes=tuple(c.describe() for c in contested[:3]),
                )
            )
            return

        # L1 - we answered this exact question, under this exact corpus.
        cached = self.store.get(key)
        if cached is not None:
            cited = {c.document_id for c in cached.citations}
            if self.retriever.visible_to(cited, principals):
                yield _final(
                    Answer(
                        text=cached.answer,
                        tier=Tier.EXACT_CACHE,
                        model_id=cached.model_id,
                        cache_key=key,
                        citations=cached.citations,
                        notes=(f"served from cache (hit {cached.hits})",),
                    )
                )
                return
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
                    yield _final(
                        Answer(
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
                    )
                    return

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
            yield _final(
                Answer(
                    text=REFUSAL_TEXT,
                    tier=Tier.REFUSED,
                    model_id="none",
                    cache_key=key,
                    notes=("no documents matched this question",),
                )
            )
            return

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

        yield {"type": "status", "stage": "answering", "passages": len(chunks)}

        climbed_from: Tier | None = None
        #: Did any rung actually read the sources? Decides which of the two
        #: refusals is true at the end of this loop.
        any_rung_ran = False
        for index, rung in enumerate(affordable):
            rung_chunks = chunks[: rung.k] if rung.k is not None else chunks
            max_tokens = rung.max_tokens or self.settings.max_answer_tokens

            # Stream only the first rung, and only when it is self-hosted. The
            # first rung is the slow one - a local model at CPU speed - and the
            # one whose silence reads as a hang. Billed rungs stay unstreamed
            # because some runtimes report no usage on a stream, and a zero in
            # the ledger for a billed call is worse than a spinner.
            streamable = getattr(rung.provider, "stream", None)
            if (
                index == 0
                and streamable is not None
                and getattr(rung.provider, "self_hosted", False)
            ):
                yield {"type": "provisional", "model": rung.name, "tier": rung.tier.value}
                attempt, deltas = (
                    None,
                    streamable(
                        system=system,
                        context=format_context(rung_chunks),
                        question=question,
                        max_tokens=max_tokens,
                    ),
                )
                completion: Completion | None = None
                try:
                    async for item in deltas:
                        if isinstance(item, Completion):
                            completion = item
                        else:
                            yield {"type": "delta", "text": item}
                except ProviderError as exc:
                    log.warning("%s tier failed mid-stream: %s", rung.tier.value, exc)
                    yield {"type": "retract", "reason": "the model became unavailable"}
                    attempt = _Attempt(
                        None,
                        Usage(),
                        0.0,
                        (
                            f"{rung.tier.value} tier unavailable: {exc}",
                            "`openknowledge model status` checks whether that endpoint is up",
                        ),
                        reached_model=False,
                    )
                if attempt is None:
                    assert completion is not None  # the stream ends with one
                    attempt = self._gate(
                        rung.provider,
                        completion,
                        rung.tier,
                        rung_chunks,
                        key,
                        near_misses=near_misses,
                    )
                    if not completion.usage.input_tokens and not completion.usage.output_tokens:
                        notes.append("streamed; the runtime reported no token counts")
                    if attempt.answer is None:
                        # What was just streamed did not survive the gate. The
                        # reader saw it, so the reader must see it withdrawn -
                        # that is the honesty half of the streaming bargain.
                        yield {
                            "type": "retract",
                            "reason": next(iter(attempt.notes), "rejected by the grounding gate"),
                        }
            else:
                attempt = await self._try_provider(
                    rung.provider,
                    rung.tier,
                    system,
                    format_context(rung_chunks),
                    question,
                    rung_chunks,
                    key,
                    max_tokens=max_tokens,
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
                yield _final(answer)
                return
            climbed_from = rung.tier
            if index + 1 < len(affordable):
                yield {"type": "status", "stage": "escalating", "to": affordable[index + 1].name}

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
            yield _final(
                Answer(
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
            )
            return

        notes.append(
            f"retrieval found {len(chunks)} passage(s); nothing read them, so the documents "
            "have not been ruled out"
        )
        yield _final(
            Answer(
                text=UNAVAILABLE_TEXT,
                tier=Tier.REFUSED,
                model_id="none",
                cache_key=key,
                usage=spent_usage,
                cost_usd=spent_usd,
                grounded=False,
                notes=tuple(notes),
            )
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

        return self._gate(provider, completion, tier, chunks, key, near_misses=near_misses)

    def _gate(
        self,
        provider: ChatProvider,
        completion: Completion,
        tier: Tier,
        chunks: list[Chunk],
        key: str,
        *,
        near_misses: int = 0,
    ) -> _Attempt:
        """Judge one completion, however it arrived - streamed or whole.

        Extracted so the streamed rung and the plain one are judged by literally
        the same code. Two gates that merely agree today is how they disagree
        tomorrow.
        """
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


def _final(answer: Answer) -> dict[str, Any]:
    """The one terminal event every resolution ends with."""
    return {"type": "final", "answer": answer}


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
