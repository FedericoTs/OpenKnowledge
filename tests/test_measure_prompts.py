"""The measured cost model.

`tools/cost_model.py` prices the architecture from assumed token counts;
`tools/measure_prompts.py` prices it from the prompts a real corpus actually
produces. These tests hold the second one to the property that makes it worth
having: it must not report a saving that a deployment would never receive.
"""

from __future__ import annotations

from pathlib import Path

from tools.measure_prompts import _levers, count_tokens, main, prompt_for

from openknowledge.prompts import SYSTEM_PROMPT
from openknowledge.providers.anthropic_provider import CACHE_MIN_TOKENS

POLICY = """# Expenses Policy

## Travel
Travel above EUR 500 requires prior approval from a line manager.
Claims must be submitted within 30 days of the expense being incurred.

## Meals
The meal allowance limit is EUR 45 per day.
"""


def levers(system_tokens: int) -> list[dict[str, object]]:
    return _levers(
        measured_input=3_000,
        system_tokens=system_tokens,
        output_tokens=1_000,
        mid_tier="claude-sonnet-5",
        frontier="claude-opus-5",
        per_day=2_000,
        free_shares=(0.45,),
    )


def test_caching_is_priced_at_zero_when_the_prompt_is_under_the_floor() -> None:
    """The failure this tool exists to catch: a lever claimed and never received."""
    rows = levers(CACHE_MIN_TOKENS - 1)

    assert rows[0]["usd_per_question"] == rows[1]["usd_per_question"]
    assert "inert" in str(rows[1]["name"])


def test_caching_is_priced_when_the_prompt_clears_the_floor() -> None:
    rows = levers(CACHE_MIN_TOKENS + 1)

    assert rows[1]["usd_per_question"] < rows[0]["usd_per_question"]
    assert "inert" not in str(rows[1]["name"])


def test_this_projects_own_system_prompt_is_measured_not_assumed() -> None:
    """If the prompt is ever rewritten past the floor, the tool must notice.

    Pinned as an inequality rather than a number so that editing the prompt for
    good reasons does not fail the build - only the *claim* has to stay honest.
    """
    measured = count_tokens(SYSTEM_PROMPT)
    rows = levers(measured)
    inert = "inert" in str(rows[1]["name"])
    assert inert == (measured < CACHE_MIN_TOKENS)


def test_each_lever_costs_no_more_than_the_one_above_it() -> None:
    rows = levers(CACHE_MIN_TOKENS + 1)
    costs = [float(r["usd_per_question"]) for r in rows]  # type: ignore[arg-type]
    assert costs == sorted(costs, reverse=True)


def test_the_prompt_measured_is_the_prompt_that_would_be_sent() -> None:
    """A measurement of some other string would be worthless."""
    text = prompt_for("What is the travel limit?", [])

    assert SYSTEM_PROMPT in text
    assert "SOURCES:" in text
    assert "What is the travel limit?" in text


def test_it_runs_end_to_end_on_a_real_folder(tmp_path: Path, capsys) -> None:
    corpus = tmp_path / "docs"
    corpus.mkdir()
    (corpus / "expenses.md").write_text(POLICY, encoding="utf-8")
    questions = tmp_path / "q.txt"
    questions.write_text("# a comment\nWhat is the travel approval limit?\n", encoding="utf-8")

    assert main(["--corpus", str(corpus), "--questions", str(questions), "--json"]) == 0

    import json

    payload = json.loads(capsys.readouterr().out)
    assert payload["documents"] == 1
    assert payload["questions"] == 1
    # Every row carries real token counts, including the whole-corpus ceiling.
    assert all(row["input_tokens"] > 0 for row in payload["rows"])
    assert payload["rows"][0]["input_tokens"] >= payload["rows"][-1]["input_tokens"]
