"""Follow-up questions, resolved into standalone ones before retrieval.

"What about contractors?" has no meaning without the turn before it, and no
retriever can be handed meaning it wasn't given: BM25 sees two content words,
the dense half embeds a fragment, and the model gets passages about nothing in
particular. The chat log had the context all along - the system just never
looked at it.

The design constraint is the cache. Answers are keyed on the question alone,
and that is what makes "same question, same answer, forever" true. Folding raw
history into the key would quietly break it - identical questions diverging on
what happened to be said earlier. So instead the follow-up is rewritten into
the standalone question it means ("Are contractors eligible for parental
leave?"), and THAT is what flows through canonicalisation, retrieval, the
cache and the ledger. The rewritten question is a real question: asked again
tomorrow, phrased by someone else, it hits the same cache entry. Determinism
survives because the rewrite itself is deterministic - temperature 0, seed 0,
same history in, same standalone question out.

The rewrite costs one small model call, so it must not tax people who ask
complete questions: it runs only when there is history AND the question shows
its dependence - a leading pronoun, a "what about", a fragment. A standalone
question with history pays nothing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..costs import Usage
from ..providers.base import ChatProvider, Message, ProviderError

#: Words that reach backwards. A question starting with one of these cannot be
#: understood without the previous turn.
_ANAPHORA = re.compile(
    r"^(and|also|but|what about|how about|what if|why not|same|ok but|okay but)\b"
    r"|^(what|how|why|when|where|who|does|do|is|are|can|could|will|would)\s+"
    r"(about\s+)?(it|that|this|those|these|they|them|he|she|there)\b",
    re.IGNORECASE,
)

#: Pronouns anywhere in a short question also read backwards ("is it paid?").
_PRONOUN = re.compile(r"\b(it|that|this|those|these|they|them|its|their)\b", re.IGNORECASE)

RESOLVE_PROMPT = """\
You rewrite a follow-up question as a standalone question, using the \
conversation before it. Output only the rewritten question - one line, no \
quotes, no explanation. Preserve the asker's intent exactly; take names and \
subjects from the conversation; never answer the question, never add facts. \
If the question is already standalone, output it unchanged."""


def looks_dependent(question: str) -> bool:
    """Whether ``question`` leans on the previous turn to mean anything.

    Lexical and deterministic on purpose. The costs are asymmetric: a missed
    follow-up retrieves as badly as it always did, while a false positive
    spends one small model call whose instruction is "if already standalone,
    output it unchanged" - so this leans permissive, and the length guard
    exists because a six-word question with a pronoun is usually a follow-up
    while a twenty-word one usually carries its own subject.
    """
    stripped = question.strip()
    if _ANAPHORA.search(stripped):
        return True
    return len(stripped.split()) <= 8 and bool(_PRONOUN.search(stripped))


@dataclass(frozen=True)
class Resolution:
    """What the follow-up meant, and what finding that out cost."""

    question: str
    rewritten: bool
    usage: Usage = Usage()
    note: str = ""


def render_history(history: tuple[Message, ...], *, max_turns: int = 8) -> str:
    """The recent conversation, oldest first, one labelled line per turn."""
    recent = history[-max_turns:]
    labels = {"user": "Asked", "assistant": "Answered"}
    return "\n".join(f"{labels.get(m.role, m.role)}: {m.content[:800]}" for m in recent)


async def resolve(
    question: str,
    history: tuple[Message, ...],
    provider: ChatProvider | None,
) -> Resolution:
    """The standalone question ``question`` means, given ``history``.

    Falls back to the question as asked - never fails the whole answer - when
    there is no provider to ask, the provider is down, or the rewrite comes
    back degenerate. A follow-up answered badly is the pre-existing behaviour;
    a follow-up that errors would be a regression.
    """
    if not history or not looks_dependent(question):
        return Resolution(question=question, rewritten=False)
    if provider is None:
        return Resolution(
            question=question,
            rewritten=False,
            note="this looks like a follow-up, but no local model was reachable to "
            "interpret it, so it was answered as asked",
        )

    try:
        completion = await provider.complete(
            system=RESOLVE_PROMPT,
            context=f"The conversation so far:\n{render_history(history)}",
            question=question,
            max_tokens=96,
        )
    except ProviderError:
        return Resolution(
            question=question,
            rewritten=False,
            note="this looks like a follow-up, but the local model did not answer the "
            "interpretation call, so it was answered as asked",
        )

    rewritten = completion.text.strip().strip('"').strip()
    rewritten = rewritten.splitlines()[0].strip() if rewritten else ""
    # A rewrite that dwarfs the input is the model explaining, not rewriting.
    if not rewritten or len(rewritten) > max(240, len(question) * 6):
        return Resolution(question=question, rewritten=False, usage=completion.usage)
    if rewritten.lower() == question.strip().lower():
        return Resolution(question=question, rewritten=False, usage=completion.usage)
    return Resolution(
        question=rewritten,
        rewritten=True,
        usage=completion.usage,
        note=f'interpreted as: "{rewritten}"',
    )
