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
    #: Which PDF backend to use: "auto", "opendataloader" or "pdfplumber".
    #:
    #: OpenDataLoader reports a document's real structure - explicit heading
    #: levels, native table cells, PDF/UA tags - where pdfplumber infers it
    #: from geometry. It needs a JVM, which is unremarkable in a container and
    #: absent from a bare pip install, so "auto" picks whichever can run.
    #:
    #: The two extract slightly different text, so a corpus fingerprints
    #: differently under each. Cached answers regenerate rather than going
    #: stale, but moving between a machine with Java and one without
    #: invalidates all of them. Pin this if that matters.
    pdf_backend: str = Field(default="auto", pattern="^(auto|opendataloader|pdfplumber)$")
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
    #: Scales the prose-contradiction thresholds. Above 1.0 flags less.
    #: Measured by `openknowledge eval-conflicts` - move it with evidence.
    deontic_strictness: float = Field(default=1.0, gt=0.0, le=3.0)

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

    # -- reranking --------------------------------------------------------
    #: Retrieve this many candidates, then rerank down to `retrieval_k`. Free and
    #: model-less: it fixes one document taking every slot, near-duplicate
    #: windows wasting slots, and heading matches scored as ordinary text. Set to
    #: 0 to send BM25's own top-k straight through.
    rerank_candidates: int = Field(default=30, ge=0, le=200)
    rerank_max_per_document: int = Field(default=2, ge=1)

    # -- escalation ladder ------------------------------------------------
    #: Rungs between the cheap tier and the frontier, cheapest first, as
    #: `model_id@base_url` or just `model_id` to reuse `escalation_base_url`.
    #: A gate failure that a mid-size open-weight model can ground costs about
    #: $0.0009 there against $0.037 at the frontier, and escalation is where
    #: almost all of a tuned deployment's remaining spend goes.
    #:
    #:     OK_LADDER='gpt-oss-120b@https://api.together.xyz/v1'
    ladder: list[str] = Field(default_factory=list)
    #: API key for the ladder rungs, when they need one distinct from the others.
    ladder_api_key: str | None = None

    # -- budget -----------------------------------------------------------
    #: Cap on spending over a rolling 24 hours. Unset means no governor.
    #: It limits *escalation*, never service: the cheapest rung is always tried,
    #: and a question the ceiling blocks is refused with the reason rather than
    #: answered from an ungrounded attempt.
    budget_daily_usd: float | None = Field(default=None, ge=0.0)
    #: Only sets the pace for spreading the cap across the day. It does not have
    #: to be accurate - the ceiling self-corrects as real traffic arrives.
    budget_expected_questions_per_day: int = Field(default=2_000, ge=1)

    # -- generation ------------------------------------------------------
    max_answer_tokens: int = Field(default=1500, ge=64)

    # -- website -----------------------------------------------------------
    #: Serve the marketing page at /site and accept its contact form at
    #: /api/contact. **Off by default**: a running answer engine has no business
    #: accepting public writes unless somebody asked it to, and most deployments
    #: serve the widget internally and never need this.
    website_enabled: bool = False
    #: Where submissions go. Its own file, not the answer store: one holds
    #: questions employees asked and the other holds people who want an email,
    #: and they have different retention rules and different readers.
    contacts_db: str = "contacts.db"
    #: Ceiling on submissions accepted per hour. A public write endpoint without
    #: one is an invitation.
    contact_max_per_hour: int = Field(default=60, ge=1)

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
