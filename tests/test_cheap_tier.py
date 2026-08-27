"""The cheap tier, and the ledger telling the truth about it.

The cheapest way to answer a grounded question is an open-weight model on a
serverless provider - measured at 116x cheaper than the frontier tier on the
same prompt. It reaches OpenKnowledge through the same OpenAI-compatible adapter
as a self-hosted model, which is what makes it configuration rather than code.

It is also where the ledger is easiest to corrupt: same adapter, same tier name,
but one has no invoice behind it and the other bills per token. Reporting $0 for
a call that cost money is the failure `costs.py` exists to prevent, so it is
tested here directly.
"""

from __future__ import annotations

import pytest
from tools.cost_model import cascade_cost

from openknowledge.cascade.router import _price
from openknowledge.costs import Usage, cost_usd, get_price
from openknowledge.providers.openai_compat import OpenAICompatProvider, is_self_hosted

USAGE = Usage(input_tokens=2_313, output_tokens=1_000)


@pytest.mark.parametrize(
    "url",
    [
        "http://localhost:11434/v1",
        "http://127.0.0.1:8000/v1",
        "http://192.168.1.40:11434/v1",
        "http://10.0.0.5/v1",
        "http://ollama:11434/v1",  # a sibling container
        "http://gpu-box.local:8080/v1",
    ],
)
def test_a_box_you_own_is_self_hosted(url: str) -> None:
    assert is_self_hosted(url) is True


@pytest.mark.parametrize(
    "url",
    [
        "https://api.together.xyz/v1",
        "https://api.groq.com/openai/v1",
        "https://api.deepinfra.com/v1/openai",
        "https://api.openai.com/v1",
        "https://my-inference.example.com/v1",
    ],
)
def test_a_vendor_endpoint_is_not(url: str) -> None:
    assert is_self_hosted(url) is False


def test_an_unparseable_url_is_treated_as_billed() -> None:
    """Failing closed: a cost note is cheap, a silently wrong ledger is not."""
    assert is_self_hosted("") is False
    assert is_self_hosted("not a url") is False


def test_a_self_hosted_local_model_is_priced_at_zero() -> None:
    provider = OpenAICompatProvider(model_id="qwen3:8b", base_url="http://localhost:11434/v1")
    cost, notes = _price(USAGE, provider)

    assert cost == 0.0
    assert notes == ()


def test_an_open_weight_endpoint_in_the_local_tier_is_billed() -> None:
    """The bug this test exists for: same tier, same adapter, real invoice."""
    provider = OpenAICompatProvider(model_id="gpt-oss-20b", base_url="https://api.together.xyz/v1")
    cost, notes = _price(USAGE, provider)

    assert cost == pytest.approx(cost_usd(USAGE, get_price("gpt-oss-20b")))
    assert cost > 0.0
    assert notes == ()


def test_an_unpriced_billed_endpoint_is_flagged_not_guessed() -> None:
    provider = OpenAICompatProvider(
        model_id="some-model-we-have-no-rate-for", base_url="https://api.example.com/v1"
    )
    cost, notes = _price(USAGE, provider)

    assert cost == 0.0
    assert notes and "no verified price" in notes[0]


def test_an_operator_can_override_the_guess() -> None:
    """A private endpoint behind a public hostname is a real deployment shape."""
    provider = OpenAICompatProvider(
        model_id="qwen3:8b", base_url="https://llm.internal.example.com/v1", self_hosted=True
    )
    assert _price(USAGE, provider) == (0.0, ())


# -- what the cascade cost actually turns on -------------------------------


def test_the_open_weight_tier_is_two_orders_cheaper_than_the_frontier() -> None:
    frontier = cost_usd(USAGE, get_price("claude-opus-5"))
    open_weight = cost_usd(USAGE, get_price("gpt-oss-20b"))
    assert frontier / open_weight > 100


def test_escalation_dominates_once_the_cheap_tier_is_nearly_free() -> None:
    """The finding that reordered the roadmap.

    With a $0.0003 cheap tier, halving the escalation rate saves an order of
    magnitude more than raising the free share by forty points.
    """
    free_share_gain = cascade_cost("gpt-oss-20b", "claude-sonnet-5", 0.10) - cascade_cost(
        "gpt-oss-20b", "claude-sonnet-5", 0.10, free=0.85
    )
    escalation_gain = cascade_cost("gpt-oss-20b", "claude-opus-5", 0.10) - cascade_cost(
        "gpt-oss-20b", "claude-opus-5", 0.05
    )
    assert escalation_gain > 10 * free_share_gain


def test_what_you_escalate_to_matters_more_than_what_you_escalate_from() -> None:
    to_frontier = cascade_cost("gpt-oss-20b", "claude-opus-5", 0.10)
    to_mid = cascade_cost("gpt-oss-20b", "claude-sonnet-5", 0.10)
    cheaper_floor = cascade_cost("claude-haiku-4-5", "claude-opus-5", 0.10)

    assert to_mid < to_frontier < cheaper_floor
