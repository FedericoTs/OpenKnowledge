"""Running one golden set against several configurations.

The tool's job is a comparison, and the thing that decides whether it ever gets
run is what it does when a key is missing. Somebody with only an Anthropic key
must still get a useful answer from the same command as somebody with both, or
this becomes a script only its author executes.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from tools.compare_configs import Profile, _expand, load_profiles, render

REPO = Path(__file__).resolve().parent.parent


def profile(**changes: object) -> Profile:
    base = {"name": "p", "description": "d", "requires": (), "env": {}}
    return Profile(**{**base, **changes})  # type: ignore[arg-type]


# -- environment expansion --------------------------------------------------


def test_a_literal_is_left_alone(monkeypatch) -> None:
    assert _expand("true") == "true"
    assert _expand("http://localhost:11434/v1") == "http://localhost:11434/v1"


def test_a_variable_is_read_from_the_environment(monkeypatch) -> None:
    monkeypatch.setenv("SOME_KEY", "sk-abc")
    assert _expand("$SOME_KEY") == "sk-abc"


def test_a_fallback_is_used_when_the_variable_is_unset(monkeypatch) -> None:
    """What keeps the local profile usable on Ollama, LM Studio or llama.cpp
    without editing the file."""
    monkeypatch.delenv("LOCAL_BASE_URL", raising=False)
    assert _expand("$LOCAL_BASE_URL:-http://localhost:11434/v1") == "http://localhost:11434/v1"

    monkeypatch.setenv("LOCAL_BASE_URL", "http://127.0.0.1:8081/v1")
    assert _expand("$LOCAL_BASE_URL:-http://localhost:11434/v1") == "http://127.0.0.1:8081/v1"


def test_a_variable_with_no_fallback_resolves_empty_rather_than_literal(monkeypatch) -> None:
    """Passing the literal string "$ANTHROPIC_API_KEY" as a key would produce a
    401 that looks like a wrong key rather than a missing one."""
    monkeypatch.delenv("NOT_SET_ANYWHERE", raising=False)
    assert _expand("$NOT_SET_ANYWHERE") == ""


# -- skipping ---------------------------------------------------------------


def test_a_profile_needing_no_keys_always_runs(monkeypatch) -> None:
    assert profile(requires=()).missing_keys() == ()


def test_every_missing_key_is_named_not_just_the_first(monkeypatch) -> None:
    monkeypatch.delenv("TOGETHER_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    p = profile(requires=("TOGETHER_API_KEY", "ANTHROPIC_API_KEY"))
    assert p.missing_keys() == ("TOGETHER_API_KEY", "ANTHROPIC_API_KEY")


def test_an_empty_key_counts_as_missing(monkeypatch) -> None:
    """An exported-but-blank variable is the commonest way a .env goes wrong."""
    monkeypatch.setenv("TOGETHER_API_KEY", "")
    assert profile(requires=("TOGETHER_API_KEY",)).missing_keys() == ("TOGETHER_API_KEY",)


# -- reporting --------------------------------------------------------------


def test_the_report_says_what_was_skipped_and_why() -> None:
    text = render([], [("ladder", ("TOGETHER_API_KEY", "ANTHROPIC_API_KEY"))])
    assert "No profile could run" in text


def test_one_run_reports_and_names_what_is_still_missing() -> None:
    rows = [
        {
            "_profile": "self-hosted",
            "_seconds": 900,
            "accuracy": 0.82,
            "false_answers": 0,
            "determinism": 1.0,
            "cost_per_question_usd": 0.0,
            "free_share": 0.35,
            "tiers": {"local": 12, "exact": 5, "refused": 9},
        }
    ]
    text = render(rows, [("frontier", ("ANTHROPIC_API_KEY",))])

    assert "self-hosted" in text and "82.0%" in text
    assert "Not run:" in text and "ANTHROPIC_API_KEY" in text
    assert "local 12" in text, "the tier spread is the escalation rate"


def test_false_answers_are_explained_before_accuracy() -> None:
    """It outranks every other column and the report has to say so."""
    text = render(
        [{"_profile": "x", "_seconds": 1, "accuracy": 1.0, "false_answers": 0}],
        [],
    )
    assert text.index("false") < text.index("accuracy   Of the answerable")


def test_a_missing_metric_renders_as_a_dash_rather_than_a_zero() -> None:
    """A run that produced no cost figure has not proved the cost was zero."""
    text = render([{"_profile": "x", "_seconds": 1}], [])
    assert "-" in text


# -- the shipped profiles ---------------------------------------------------


def test_the_shipped_profiles_load_and_name_their_keys() -> None:
    profiles = load_profiles(REPO / "evals" / "profiles.yaml")
    by_name = {p.name: p for p in profiles}

    assert by_name["self-hosted"].requires == (), "the local profile must need no key"
    assert "TOGETHER_API_KEY" in by_name["open-weight"].requires
    assert set(by_name["ladder"].requires) == {"TOGETHER_API_KEY", "ANTHROPIC_API_KEY"}


def test_no_shipped_profile_contains_a_secret() -> None:
    """Keys are named, never held. This file is committed."""
    raw = yaml.safe_load((REPO / "evals" / "profiles.yaml").read_text(encoding="utf-8"))
    for entry in raw:
        for key, value in (entry.get("env") or {}).items():
            if "KEY" in key.upper():
                assert str(value).startswith("$"), f"{entry['name']}.{key} holds a literal value"


@pytest.mark.parametrize("name", ["self-hosted", "open-weight", "ladder", "frontier"])
def test_every_profile_pins_every_tier_it_is_not_using(name: str) -> None:
    """A profile that leaves escalation at whatever .env happens to say is not
    measuring the configuration it claims to."""
    profiles = {p.name: p for p in load_profiles(REPO / "evals" / "profiles.yaml")}
    env = profiles[name].env
    assert "OK_LADDER" in env
    assert "OK_ESCALATION_ENABLED" in env
    assert "OK_LOCAL_ENABLED" in env
