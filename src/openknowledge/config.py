"""Runtime settings.

Defaults are chosen so that an operator who configures nothing gets the private,
free configuration: local retrieval, no outbound calls, no escalation. Reaching a
paid API is something you switch on, never something that happens because you
forgot to switch it off.
"""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="OK_", env_file=".env", extra="ignore", protected_namespaces=()
    )

    # -- storage ---------------------------------------------------------
    data_dir: str = Field(default="./data", description="Where the SQLite store lives.")
    documents_dir: str = Field(
        default="./documents", description="Folder the local-files connector reads."
    )

    # -- retrieval -------------------------------------------------------
    retrieval_k: int = Field(default=6, ge=1, le=50)
    chunk_target_words: int = Field(default=350, ge=50)
    chunk_overlap_words: int = Field(default=60, ge=0)

    # -- grounding -------------------------------------------------------
    min_support_ratio: float = Field(default=0.45, ge=0.0, le=1.0)
    require_citations: bool = True

    # -- knowledge lifecycle ---------------------------------------------
    #: Draft FAQ answers from documents when they are uploaded or change. This
    #: is the one-off cost that makes the recurring one disappear.
    draft_on_ingest: bool = True
    #: Serve gate-passed drafts that no human has reviewed yet, marked as such.
    #: Turn this off for a compliance posture where nothing machine-written may
    #: reach an employee before sign-off; the cost benefit then waits on review.
    serve_drafts: bool = True
    #: Refuse to answer when two documents disagree about the claim being asked
    #: for. Turning this off makes the bot answer from whichever document
    #: retrieval preferred, which is where confident wrong answers come from.
    block_on_conflict: bool = True
    #: Re-check approved answers whose cited documents changed.
    reverify_on_change: bool = True
    #: Cap on how many changed documents one ingest run will draft for, so a
    #: first import of ten thousand files cannot spend without warning.
    max_documents_per_ingest: int = 200
    conflict_min_overlap: float = Field(default=0.34, ge=0.0, le=1.0)

    # -- local tier ------------------------------------------------------
    local_enabled: bool = True
    local_model: str = "qwen3:8b"
    local_base_url: str = "http://localhost:11434/v1"
    local_api_key: str | None = None

    # -- escalation tier -------------------------------------------------
    #: Off by default. Nothing leaves the machine until an operator opts in.
    escalation_enabled: bool = False
    escalation_provider: str = Field(default="anthropic", pattern="^(anthropic|openai_compat)$")
    escalation_model: str = "claude-opus-5"
    escalation_base_url: str = "https://api.openai.com/v1"
    escalation_effort: str = Field(default="low", pattern="^(low|medium|high|xhigh|max)$")
    anthropic_api_key: str | None = None
    openai_api_key: str | None = None

    # -- generation ------------------------------------------------------
    max_answer_tokens: int = Field(default=1500, ge=64)

    # -- admin -----------------------------------------------------------
    #: Custom text appended to the built-in system prompt. Appended, never
    #: substituted: the grounding rules are what make the cheap tiers safe, so
    #: they are not editable from the admin UI.
    system_prompt_suffix: str = ""
    admin_token: str | None = None

    @property
    def db_path(self) -> str:
        return f"{self.data_dir.rstrip('/')}/openknowledge.db"

    @property
    def knowledge_db_path(self) -> str:
        """Proposals and conflicts, kept separate from the answer cache.

        Different lifecycles: the answer cache is disposable and rebuilt on
        any corpus change, while approvals and conflict resolutions are
        human decisions that must survive one.
        """
        return f"{self.data_dir.rstrip('/')}/knowledge.db"


def load_settings() -> Settings:
    return Settings()
