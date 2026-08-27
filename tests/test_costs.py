"""The arithmetic behind the whole project."""

from __future__ import annotations

import pytest

from openknowledge.costs import (
    PricingError,
    Usage,
    cost_usd,
    get_price,
    load_price_table,
    self_hosted_cost_usd,
)


def test_the_ten_cent_call() -> None:
    """The bill this project exists to eliminate: a fat uncached frontier call."""
    naive = Usage(input_tokens=15_000, output_tokens=1_000)
    assert cost_usd(naive, get_price("claude-opus-5")) == pytest.approx(0.10)


def test_caching_and_tight_retrieval_cut_the_same_call() -> None:
    """Cache the fixed prompt, retrieve less, and the same question gets far cheaper."""
    tuned = Usage(input_tokens=2_000, cache_read_tokens=13_000, output_tokens=400)
    cost = cost_usd(tuned, get_price("claude-opus-5"))
    assert cost == pytest.approx(0.0265)
    assert cost < 0.10 / 3


def test_cache_read_is_a_tenth_of_input() -> None:
    price = get_price("claude-opus-5")
    read = cost_usd(Usage(cache_read_tokens=1_000_000), price)
    fresh = cost_usd(Usage(input_tokens=1_000_000), price)
    assert read == pytest.approx(fresh * 0.1)


def test_cache_write_carries_a_premium() -> None:
    price = get_price("claude-opus-5")
    write = cost_usd(Usage(cache_write_tokens=1_000_000), price)
    assert write == pytest.approx(cost_usd(Usage(input_tokens=1_000_000), price) * 1.25)


def test_batch_halves_the_bill() -> None:
    price = get_price("claude-opus-5")
    live = Usage(input_tokens=10_000, output_tokens=500)
    batched = Usage(input_tokens=10_000, output_tokens=500, batch=True)
    assert cost_usd(batched, price) == pytest.approx(cost_usd(live, price) * 0.5)


def test_self_hosted_has_no_per_token_price() -> None:
    assert cost_usd(Usage(input_tokens=10**7, output_tokens=10**6), get_price("local")) == 0.0


def test_self_hosted_fixed_cost_amortises_over_volume() -> None:
    # A GPU box at $1.20/h is expensive per question when idle and trivial when busy.
    assert self_hosted_cost_usd(hourly_rate_usd=1.20, questions_per_hour=10) == pytest.approx(0.12)
    assert self_hosted_cost_usd(hourly_rate_usd=1.20, questions_per_hour=2000) == pytest.approx(
        0.0006
    )


def test_unpriced_model_refuses_to_invent_a_number() -> None:
    """Better a loud error than a ledger quietly reporting $0 for real spend."""
    with pytest.raises(PricingError, match="No verified price"):
        cost_usd(Usage(input_tokens=1000), get_price("openai-frontier"))


def test_shipped_prices_carry_a_verification_date() -> None:
    for price in load_price_table().values():
        if price.is_priced:
            assert price.verified is not None, f"{price.model_id} has a price but no verified date"


def test_local_prefixed_models_resolve_to_the_local_rate() -> None:
    assert get_price("ollama/qwen3:8b").input_per_mtok == 0.0
    assert cost_usd(Usage(input_tokens=99_999), get_price("local/mistral")) == 0.0


def test_usage_addition_rejects_mixed_batch_modes() -> None:
    with pytest.raises(ValueError, match="batch"):
        Usage(input_tokens=1) + Usage(input_tokens=1, batch=True)


def test_total_prompt_tokens_sums_all_three_buckets() -> None:
    u = Usage(input_tokens=100, cache_read_tokens=900, cache_write_tokens=50)
    assert u.total_prompt_tokens == 1050
