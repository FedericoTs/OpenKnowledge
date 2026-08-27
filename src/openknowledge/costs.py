"""Token accounting and price lookup.

Every answer OpenKnowledge produces carries a :class:`Usage` record, even the
ones that cost nothing - a cache hit reports zero tokens rather than reporting
nothing. That is deliberate: "what does this bot actually cost us per question"
should be answerable from the ledger alone, without anyone estimating.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, replace
from functools import lru_cache
from importlib import resources
from typing import Any

import yaml

_MILLION = 1_000_000


class PricingError(RuntimeError):
    """Raised when a price is asked for that we do not have a real number for."""


@dataclass(frozen=True, slots=True)
class ModelPrice:
    """USD-per-million-token rates for one model."""

    model_id: str
    provider: str
    tier: str
    input_per_mtok: float | None
    output_per_mtok: float | None
    context_tokens: int | None = None
    cache_min_tokens: int | None = None
    cache_read_multiplier: float = 0.1
    cache_write_multiplier: float = 1.25
    batch_multiplier: float = 0.5
    verified: _dt.date | None = None

    @property
    def is_priced(self) -> bool:
        return self.input_per_mtok is not None and self.output_per_mtok is not None

    def require_priced(self) -> ModelPrice:
        if not self.is_priced:
            raise PricingError(
                f"No verified price for {self.model_id!r}. OpenKnowledge ships this "
                f"slot empty rather than guessing. Add input_per_mtok/output_per_mtok "
                f"to pricing.yaml from the {self.provider} pricing page."
            )
        return self


@dataclass(frozen=True, slots=True)
class Usage:
    """Tokens consumed by a single model call.

    ``input_tokens`` is the *uncached* remainder only, matching how the
    Anthropic API reports it: total prompt size is
    ``input_tokens + cache_read_tokens + cache_write_tokens``.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    batch: bool = False

    @property
    def total_prompt_tokens(self) -> int:
        return self.input_tokens + self.cache_read_tokens + self.cache_write_tokens

    def __add__(self, other: Usage) -> Usage:
        if other.batch != self.batch:
            # Mixing batch and interactive calls would make the multiplier
            # ambiguous; price them separately and add the dollars instead.
            raise ValueError("cannot add batch and non-batch Usage; sum their costs")
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_read_tokens=self.cache_read_tokens + other.cache_read_tokens,
            cache_write_tokens=self.cache_write_tokens + other.cache_write_tokens,
            batch=self.batch,
        )


def cost_usd(usage: Usage, price: ModelPrice) -> float:
    """Dollar cost of ``usage`` at ``price``.

    Raises :class:`PricingError` if the model has no verified rate, rather than
    silently reporting $0.00 for a call that really did cost money.
    """
    p = price.require_priced()
    assert p.input_per_mtok is not None and p.output_per_mtok is not None

    dollars = (
        usage.input_tokens * p.input_per_mtok
        + usage.output_tokens * p.output_per_mtok
        + usage.cache_read_tokens * p.input_per_mtok * p.cache_read_multiplier
        + usage.cache_write_tokens * p.input_per_mtok * p.cache_write_multiplier
    ) / _MILLION

    if usage.batch:
        dollars *= p.batch_multiplier
    return dollars


def self_hosted_cost_usd(
    *,
    hourly_rate_usd: float,
    questions_per_hour: float,
) -> float:
    """Amortised per-question cost of a model you host yourself.

    A self-hosted model has no per-token invoice, but it does occupy hardware.
    That cost is fixed per hour and divided across however many questions
    arrive, which is why self-hosting wins on volume and loses on a bot nobody
    uses. ``hourly_rate_usd`` is whatever the box costs you per hour - a cloud
    GPU instance rate, or amortised capex plus power for on-prem.
    """
    if questions_per_hour <= 0:
        raise ValueError("questions_per_hour must be > 0")
    if hourly_rate_usd < 0:
        raise ValueError("hourly_rate_usd must be >= 0")
    return hourly_rate_usd / questions_per_hour


def _coerce_date(value: Any) -> _dt.date | None:
    if value is None:
        return None
    if isinstance(value, _dt.date):
        return value
    return _dt.date.fromisoformat(str(value))


@lru_cache(maxsize=1)
def load_price_table() -> dict[str, ModelPrice]:
    """Parse the bundled ``pricing.yaml`` into ``{model_id: ModelPrice}``."""
    raw = yaml.safe_load(resources.files("openknowledge").joinpath("pricing.yaml").read_text())
    defaults: dict[str, Any] = raw.get("defaults") or {}

    table: dict[str, ModelPrice] = {}
    for entry in raw.get("models") or []:
        fields = {**defaults, **entry}
        fields["verified"] = _coerce_date(fields.get("verified"))
        price = ModelPrice(**fields)
        table[price.model_id] = price
    return table


def get_price(model_id: str) -> ModelPrice:
    """Look up a model's price, falling back to an unpriced placeholder.

    An unknown model returns an unpriced :class:`ModelPrice` rather than raising,
    so a self-hosted or newly released model still flows through the system; the
    error surfaces later at :func:`cost_usd`, where the number would actually be
    used.
    """
    table = load_price_table()
    if model_id in table:
        return table[model_id]
    if model_id.startswith("local/") or model_id.startswith("ollama/"):
        return replace(table["local"], model_id=model_id)
    return ModelPrice(
        model_id=model_id,
        provider="unknown",
        tier="unknown",
        input_per_mtok=None,
        output_per_mtok=None,
    )
