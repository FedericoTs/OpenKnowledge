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
        env_prefix="OK_",
        env_file=".env",
        # The state env is written as UTF-8 by write_env; reading it with the
        # machine locale (cp1252 on Windows) would corrupt any non-ASCII path.
        env_file_encoding="utf-8",
        extra="ignore",
        protected_namespaces=(),
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
    #: The floor for answers that earn it: every substantive claim cited,
    #: every citation resolving, every figure verified. A faithful summary
    #: compresses and rephrases - a correct, fully cited six-bullet summary
    #: measured 42% in the field and was withdrawn at the 45% floor. The
    #: relaxation is per answer, judged on that answer's own discipline.
    min_support_ratio_cited: float = Field(default=0.30, ge=0.0, le=1.0)
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
    #: The context window the local runtime will actually run with, in tokens.
    #:
    #: Set by `openknowledge model use`, which reads it from the runtime rather
    #: than guessing. Known, a prompt that would not fit is refused here instead
    #: of remotely - runtimes disagree about whether that is an error or a quiet
    #: trim of the prompt's front, where the grounding rules are, and the API
    #: does not say which you have. Zero means unknown, and nothing is checked.
    local_context_tokens: int = Field(default=0, ge=0)
    #: How long to wait for the local model, in seconds.
    #:
    #: Much longer than the paid tiers get, and deliberately: a cold call to a
    #: self-hosted model loads it into memory first, which on a laptop CPU is
    #: minutes before a single token is generated. A cloud API that has not
    #: answered in two minutes has failed; a local one has probably just
    #: finished reading eight gigabytes off disk.
    local_timeout_seconds: float = Field(default=600.0, gt=0)
    #: How many questions the local model server answers at once.
    #:
    #: One number for two things that must never disagree: llama-server is
    #: started with this many slots, and this many requests are allowed
    #: through to it. A request arriving with no slot free does not queue at
    #: the server - its stream is severed mid-answer, and the cascade reports
    #: a model it could not reach. Measured on a shared install: four
    #: simultaneous questions, one answered, three refused while telling the
    #: asker their configuration was broken.
    #:
    #: One slot is right for a laptop, where four full-context KV buffers are
    #: what an integrated GPU refused. A company server with memory to spare
    #: should raise this - both halves move together, so raising it here is
    #: the whole change.
    local_parallel: int = Field(default=1, ge=1)
    #: How often to notice documents that changed in the folder, in seconds.
    #:
    #: Uploads and deletes re-index themselves. This is for the other way
    #: documents arrive on a shared server - dropped into the folder, synced
    #: from SharePoint, corrected in place - where nothing tells the app at
    #: all. Measured before this existed: a policy edited on disk left the
    #: index holding the old text, and the answer cited last year's figure
    #: while looking entirely current, which is the one kind of wrong answer
    #: this product exists to prevent.
    #:
    #: The check is a stat of each file, a few milliseconds on a thousand
    #: documents, and re-reads only when something actually moved. Zero turns
    #: it off for a corpus that only ever changes through the app.
    documents_rescan_seconds: int = Field(default=60, ge=0)
    #: Load the model at server start, in the background, instead of letting
    #: the first question absorb the load time. Costs nothing when the model
    #: is already resident.
    local_warmup: bool = True
    #: How long Ollama keeps the model resident after use ("30m", "-1" for
    #: always, "" to leave Ollama's own five-minute default alone). Only
    #: Ollama honours this; llama.cpp and vLLM never unload.
    local_keep_alive: str = "30m"

    # -- dense retrieval ---------------------------------------------------
    #: Add semantic search alongside BM25, fused by reciprocal rank.
    #:
    #: BM25 matches words; someone asking "how much can I spend on dinner" gets
    #: nothing from a document that says "meals are reimbursed up to EUR 45",
    #: which is exactly the casual phrasing a chat box invites. Embeddings close
    #: that gap and are worse at the things BM25 is best at - EUR 500, form
    #: RA-14 - so both run and the ranks are fused.
    #:
    #: On by default, and harmless when it cannot run: no embedding model means
    #: BM25 alone, reported rather than failed.
    embedding_enabled: bool = True
    #: Runs on the same endpoint as the local chat model unless set otherwise.
    embedding_model: str = "nomic-embed-text"
    embedding_base_url: str = ""
    #: Where chunk vectors live. Derived and disposable: deleting it costs one
    #: re-embed. Kept out of the answer store, which holds human decisions.
    vectors_db: str = "vectors.db"
    #: Serve a cached answer for a differently-phrased question - but only
    #: after the grounding gate re-verifies it against the new question's own
    #: retrieval. Similarity nominates; the gate decides. Measured: cosine
    #: alone cannot tell "parental leave weeks" from "annual leave days".
    semantic_cache_enabled: bool = True
    #: How similar a cached question must be to be worth showing to the gate.
    #: A candidate-finder, not a safety threshold - rejected nominees cost
    #: microseconds. Genuine paraphrases measured 0.727-0.849 on nomic-embed.
    semantic_cache_threshold: float = Field(default=0.70, ge=0.5, le=1.0)

    # -- escalation tier -------------------------------------------------
    #: Off by default. Nothing leaves the machine until an operator opts in.
    escalation_enabled: bool = False
    escalation_provider: str = Field(
        default="anthropic", pattern="^(anthropic|openai_compat|azure)$"
    )
    escalation_model: str = "claude-opus-5"
    escalation_base_url: str = "https://api.openai.com/v1"
    escalation_effort: str = Field(default="low", pattern="^(low|medium|high|xhigh|max)$")
    anthropic_api_key: str | None = None
    openai_api_key: str | None = None

    # -- escalation via Azure OpenAI (docs/AZURE-OPENAI.md) ----------------
    #: The company's own tenant: same models, same compliance boundary,
    #: per-token billing on the company's agreement. This is what "use our
    #: Copilot subscription" translates to - a Copilot seat itself is not a
    #: callable model API. Used when escalation_provider is "azure".
    azure_openai_endpoint: str = ""  # https://<resource>.openai.azure.com
    azure_openai_deployment: str = ""  # the company's name for the model it provisioned
    azure_openai_api_key: str | None = None
    azure_openai_api_version: str = "2024-06-01"
    #: Your prices, from your own Azure price sheet - they vary by region and
    #: agreement, so shipping a number here would be inventing one. Unset,
    #: every escalated call's cost is flagged as uncounted, never guessed.
    azure_openai_input_per_mtok: float | None = Field(default=None, ge=0.0)
    azure_openai_output_per_mtok: float | None = Field(default=None, ge=0.0)

    # -- updates ----------------------------------------------------------
    #: Ask github.com once a day whether a newer release exists, so the
    #: desktop app can offer its one-click verified update. This is an
    #: outbound call and is documented as one; turn it off for air-gapped or
    #: IT-managed installs and nothing ever phones anywhere.
    update_check: bool = True

    # -- tag routing ------------------------------------------------------
    #: Use per-document tags, derived free at index time, to guarantee that a
    #: question naming its documents finds them among its retrieval candidates
    #: - rescued from below the cut when a large corpus buries them. Never a
    #: filter and never a reordering: both were measured against the golden
    #: sets and both made the local model worse. Any ambiguity changes
    #: nothing, and off restores pre-tag retrieval exactly.
    tag_routing: bool = True

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
    #: How many questions one asker may ask per minute. The budget governor
    #: already stops a flood becoming an invoice - what it cannot do is decide
    #: whose questions those were, so one looping caller drags the shared
    #: ceiling down for everybody. 0 disables it, which is right for a desktop
    #: install where the only asker is the person whose laptop it is. Counted
    #: in memory and keyed by a salted hash: enforcing a limit needs to know
    #: this caller asked twelve times, never who they are.
    asker_questions_per_minute: int = Field(default=0, ge=0)

    # -- generation ------------------------------------------------------
    max_answer_tokens: int = Field(default=1500, ge=64)

    # -- serving ------------------------------------------------------------
    #: The interface `serve` bound, recorded by the CLI so the app can decide
    #: whether the loopback Host allowlist applies. Not a way to choose the
    #: bind - that is serve's --host flag.
    bind_host: str = "127.0.0.1"
    #: Host headers accepted when serving loopback. "testserver" is what test
    #: clients send; the rest are the names a local browser can legitimately
    #: use. Extend it if you put a local reverse proxy in front.
    trusted_hosts: list[str] = Field(
        default_factory=lambda: ["127.0.0.1", "localhost", "::1", "testserver"]
    )

    # -- uploads ------------------------------------------------------------
    #: Accept documents over HTTP (the widget's drag-and-drop, POST /documents).
    #: **Off by default** for the same reason the website is: a running answer
    #: engine has no business accepting writes unless somebody asked it to.
    #: A desktop (app-mode) install turns it on at first serve and records the
    #: choice, because drag-and-drop IS how documents arrive there and there is
    #: nobody to set variables; an explicit false is always respected.
    upload_enabled: bool = False
    #: Per-file ceiling. A corpus document larger than this is almost always a
    #: scan, which the parser cannot read anyway.
    upload_max_mb: int = Field(default=25, ge=1, le=500)

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

    # -- sign-in ---------------------------------------------------------
    #: "off" (default) keeps today's behaviour everywhere, including the
    #: trusted-caller mode where a request may assert its own principals.
    #: "oidc" puts sign-in in front of everything but /auth/* and /healthz,
    #: and principals are then minted from the session, never from the wire.
    #: Needs the auth extra (`pip install 'openknowledge[auth]'`) and the
    #: OIDC settings below. Design and limits: docs/ENTRA-SIGNIN.md.
    auth_mode: str = Field(default="off", pattern="^(off|oidc)$")
    #: For Entra, the tenant-id (GUID) form:
    #: https://login.microsoftonline.com/<tenant-id>/v2.0
    oidc_issuer: str = ""
    oidc_client_id: str = ""
    #: Optional for providers that accept public PKCE clients; Entra's Web
    #: platform wants one.
    oidc_client_secret: str = ""
    #: Which ID-token claim carries group ids. Entra calls it "groups".
    oidc_groups_claim: str = "groups"
    #: Members of this group (an Entra group object id) get the admin
    #: surface without the shared token - grant and revoke admins in the
    #: directory, like everything else about them. Empty means token-only.
    oidc_admin_group: str = ""
    #: Members of this group curate knowledge - documents, pins, drafts,
    #: conflicts - without holding governance: they cannot change who may
    #: read a folder, edit settings, apply an update, or read the admin log.
    #: The split exists because the people who know the answers are rarely
    #: the people who should hold the access rules.
    oidc_curator_group: str = ""
    #: The URL people reach this server at, for building the OAuth redirect
    #: URI behind proxies. Empty derives it from each request, which is fine
    #: for localhost testing; Entra refuses http:// redirects anywhere else.
    public_url: str = ""
    session_hours: float = Field(default=8.0, gt=0)
    #: Serve HTTPS directly (both paths, or neither). Sign-in makes TLS
    #: load-bearing: Entra refuses http:// redirect URIs beyond localhost,
    #: and a session cookie on a plain-http LAN is readable in flight. A
    #: reverse proxy works too - these are for the deployment that wants no
    #: second service.
    tls_cert: str = ""
    tls_key: str = ""

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

    @property
    def auth_db_path(self) -> str:
        """Sessions and pending logins. Its own file: sign-in state is
        disposable and personal, and wiping it must not touch answers."""
        return f"{self.data_dir.rstrip('/')}/auth.db"


def load_settings() -> Settings:
    """Settings, with state located by how the process is being run.

    Field defaults stay CWD-relative because that is correct for the audience
    that has a CWD worth speaking of - a checkout, a server directory, the
    container. When the working directory carries no deployment (the
    double-clicked-app case), state must not scatter across whatever folder
    was current at launch, so the unset paths are pointed at the platform's
    per-user data directory instead. Anything the operator set - environment,
    dotenv, either - always wins; only genuine defaults are relocated.
    """
    from .paths import state_paths

    state = state_paths()
    env_file = state.env_file if state.env_file.is_file() else None
    settings = Settings(_env_file=env_file)  # type: ignore[call-arg]
    if state.mode != "project":
        provided = settings.model_fields_set
        if "data_dir" not in provided:
            settings.data_dir = str(state.data_dir)
        if "documents_dir" not in provided:
            settings.documents_dir = str(state.documents_dir)
    return settings
