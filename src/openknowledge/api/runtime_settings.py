"""The settings an admin may change while the server runs, and what each costs.

Not every setting belongs here, and the exclusions are the design. Paths
(data_dir, documents_dir) relocate state and belong to the operator's shell,
not an HTTP endpoint. The admin token guards this very surface. The website
and contact switches are a hosting decision. Everything editable is listed
explicitly with how it takes effect, because "restart required" discovered by
experiment is the worst kind of setting.

Two application modes, chosen per field:

* ``live`` - the running objects read the settings instance on every request,
  so mutating it is the whole change. Retrieval depth, gate thresholds, the
  semantic cache switch.
* ``rebuild`` - providers, ladder, budget and retriever are constructed from
  settings once, so these swap in a freshly built engine. A few seconds, in
  the request, visible in the response.

Every change is persisted to the same dotenv the next start will read, in the
resolved state directory - a runtime change that silently evaporated on
restart would teach people not to trust the page.
"""

from __future__ import annotations

from typing import Annotated, Any

from pydantic import TypeAdapter

from ..config import Settings

#: field name -> how it takes effect. The single source for the endpoint, the
#: page, and the tests.
EDITABLE: dict[str, str] = {
    # retrieval and grounding
    "retrieval_k": "live",
    "min_support_ratio": "live",
    "min_support_ratio_cited": "live",
    "require_citations": "live",
    "block_on_conflict": "live",
    "serve_drafts": "live",
    "conflict_min_overlap": "live",
    "deontic_strictness": "live",
    "max_answer_tokens": "live",
    # the caches
    "semantic_cache_enabled": "live",
    "semantic_cache_threshold": "live",
    # uploads
    "upload_enabled": "live",
    "upload_max_mb": "live",
    # the local model
    "local_enabled": "rebuild",
    "local_model": "rebuild",
    "local_base_url": "rebuild",
    "local_context_tokens": "rebuild",
    "local_timeout_seconds": "rebuild",
    "local_keep_alive": "rebuild",
    # embeddings
    "embedding_enabled": "rebuild",
    "embedding_model": "rebuild",
    "embedding_base_url": "rebuild",
    # escalation and budget
    "escalation_enabled": "rebuild",
    "escalation_provider": "rebuild",
    "escalation_model": "rebuild",
    "escalation_effort": "rebuild",
    "azure_openai_deployment": "rebuild",
    "budget_daily_usd": "rebuild",
    "budget_expected_questions_per_day": "rebuild",
    # Live on purpose: this is the lever an operator reaches for while a
    # caller is looping, and a rebuild would make them wait for it.
    "asker_questions_per_minute": "live",
}


class SettingsChangeError(ValueError):
    """A change that cannot be applied, with the reason a person can act on."""


def validate_changes(changes: dict[str, Any]) -> dict[str, Any]:
    """Type-check ``changes`` against the Settings model, editable fields only."""
    if not changes:
        raise SettingsChangeError("no settings were given")
    validated: dict[str, Any] = {}
    for key, raw in changes.items():
        if key not in EDITABLE:
            editable = ", ".join(sorted(EDITABLE))
            raise SettingsChangeError(f"{key!r} is not editable at runtime (editable: {editable})")
        field = Settings.model_fields[key]
        # The annotation alone is not the contract - the Field's own metadata
        # (patterns, bounds) is. Validating without it accepted
        # escalation_effort="extreme" and retrieval_k=999, which the model
        # would have refused at startup.
        contract = (
            Annotated[field.annotation, *field.metadata] if field.metadata else field.annotation
        )
        try:
            validated[key] = TypeAdapter(contract).validate_python(raw)
        except Exception as exc:
            raise SettingsChangeError(f"{key}: {exc}") from exc
    return validated


def to_env_value(value: Any) -> str:
    """How pydantic-settings expects the value spelled in a dotenv file."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return ""
    return str(value)


def needs_rebuild(changes: dict[str, Any]) -> bool:
    return any(EDITABLE[key] == "rebuild" for key in changes)
