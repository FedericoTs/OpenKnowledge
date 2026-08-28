# Roadmap

Honest status. "Built" means implemented and covered by tests in this repository.

## Built

- **Cost accounting** — per-call token accounting, rates with verification dates, an
  unpriced model raises rather than reporting $0, ledger and blended cost report.
- **Escalation ladder** — `OK_LADDER` puts as many rungs as you like between the cheap tier
  and the frontier, cheapest first, each answering from the same passages under the same
  gate. A grounding failure a mid-size open-weight model can fix costs $0.0008 instead of
  $0.021. See [COST-STRATEGIES.md](COST-STRATEGIES.md).
- **Budget governor** — a declared daily cap becomes a ceiling on what one question may
  cost, recomputed from the ledger: budget remaining ÷ questions still expected. It limits
  escalation, never service; the cheapest rung is always tried; a blocked question is
  refused with the model it could not afford, and that refusal is not cached.
- **Reranking** — free, deterministic, no model, on by default. Caps how many slots one
  document takes, drops near-duplicate windows, and counts a matching heading trail for more
  than an incidental mention. Measured on 15 real contracts: distinct documents in the
  context 3.83 → 4.42, dominant-document slots 2.75 → 1.92, near-duplicates eliminated.
- **Open-weight cheap tier** — verified serverless rates in `pricing.yaml`, reached through
  the existing OpenAI-compatible adapter, at $0.000316 per measured question (116× cheaper
  than the frontier tier). The cascade prices by whether the endpoint bills rather than by
  tier name, so a hosted open-weight model can no longer be silently recorded as free.
- **Measured cost model** — `tools/measure_prompts.py` assembles the exact prompt the
  running system would send over a real corpus and counts it, so the cost tables are fed by
  measurement rather than assumption. It found that the static system prompt is under the
  API's minimum cacheable prefix, making prompt caching inert, and corrected the headline
  from 19× to 11×. See [COST-MODEL.md](COST-MODEL.md) and `evals/measured/`.
- **Determinism layer** — conservative canonicalisation, five-part cache key, corpus
  fingerprinting, pinned answers, corpus-version eviction.
- **Grounding gate** — citation presence, invented-source detection, unsupported-number
  detection, content-word overlap, abstention handling.
- **Cascade** — pins → exact cache → local → frontier → refuse, with escalation driven by
  the grounding gate, and per-answer cost attribution.
- **Retrieval** — BM25 with overlapping chunks, deterministic tie-breaking, ACL-aware
  scoring, content-addressed corpus versions.
- **Access control** — ACL filtering at retrieval and re-checking at cache read, failing
  closed on unknown documents.
- **Providers** — `ChatProvider` protocol; Anthropic with correct prompt-cache placement;
  OpenAI-compatible covering OpenAI, Ollama, vLLM, LM Studio, llama.cpp.
- **Surfaces** — FastAPI chat endpoint, fail-closed admin API, web chat widget, CLI,
  Docker Compose stack, local-folder connector.
- **Knowledge lifecycle** — FAQ drafting at ingest with gate-checked drafts served as
  precomputed cache entries, a review queue ranked by value, free numeric conflict
  detection, citation-anchored re-verification when documents change, and refusal on
  contested claims including the stale-pin case. See [KNOWLEDGE.md](KNOWLEDGE.md).
- **Document parsing** — PDF, Word, Excel, PowerPoint and Markdown into structured blocks
  with citable locators, plus structure-aware chunking that never splits a rule from its
  condition. PDFs get two backends: OpenDataLoader where a JVM exists, which reports
  heading levels and table cells rather than inferring them, and pdfplumber everywhere
  else. See [DOCUMENTS.md](DOCUMENTS.md), ADR 0007 and ADR 0008.
- **Prose contradiction detection** — deontic claim extraction (must / may / must not) with
  predicate families and hard-versus-soft force pairs, plus a free FAQ cross-check that
  catches a newly uploaded document disagreeing with an existing answer. Measured by
  `eval-conflicts` on a labelled set that is majority near-misses, **and** on 15 real
  contracts, which is where salience weighting and duplicate-pair grouping came from.
  See ADR 0006 and ADR 0009.
- **Standalone audit** — `openknowledge audit ./folder` reports where a folder's documents
  disagree with each other with no API key, no model, no database and nothing written. Exits
  non-zero on findings so it can gate CI.
- **Hybrid retrieval** — BM25 fused with local dense retrieval (`nomic-embed-text`
  via any OpenAI-compatible or Ollama endpoint), interleaved rather than rank-fused:
  measured on real documents, reciprocal rank fusion scored 11/13 where dense alone
  scored 13/13, because BM25's near-noise ranking on a paraphrase outvotes the dense
  half. Task prefixes applied per model family (measured: 7/8 → 8/8 without/with).
  Chunk vectors cached in `vectors.db`, keyed on text and model, so re-indexing
  re-embeds only what changed. Degrades to BM25 alone, with a note, when no
  embedding endpoint exists. Gated by the golden set, which caught the first
  version at 23.5% accuracy — fused scores from two scales had re-sorted the
  reranker — before it shipped.
- **Streaming answers** — the same resolution, narrated: `answer()` and
  `answer_stream()` drain one event generator, so tier, caching, notes and cost
  cannot fork between them. Only the first self-hosted rung streams (billed rungs
  report usage unreliably on streams, and a zero in the ledger for a billed call is
  worse than a spinner). Text that fails the gate is retracted in front of the
  reader - struck through with the reason, the real outcome rendered beneath.
  Measured over a real socket: first status event at 0.2s, tokens progressive,
  where the widget previously showed a static "Looking…" for 15-30s.
- **Follow-up questions** — "what about contractors?" is rewritten into the
  standalone question it means, using the conversation the client sends, before
  anything is keyed - so the cache stays keyed on real questions, the same
  follow-up twice hits the exact cache, and asking the standalone question cold
  hits the same entry. The rewrite is deterministic (temperature 0, seed 0), runs
  only on questions that show their dependence, costs ~10 output tokens on the
  cheapest self-hosted rung, and is billed onto the answer. No self-hosted rung
  means it is skipped with a note, never billed.
- **Semantic cache** — a cached answer for a differently-phrased question, served
  only after three arbiters: similarity nominates (measured: cosine alone cannot
  tell "parental leave weeks" from "annual leave days" - 0.810, inside the
  paraphrase band), retrieval's top-ranked document must be one the cached answer
  cites, every content word of the new question must appear in the cached
  question — the question alone, because matching against the answer's text let a
  sentence the model had volunteered serve "20 weeks" to a contractors question;
  an entry vouches for its question, never for its answer's ramblings — and the
  grounding gate re-verifies against the new question's own retrieval. Each
  arbiter beyond similarity exists because a defeat demanded it: a unit trap, a
  live run, then the golden gate itself at 94.1%. Measured live: a paraphrase
  serves in 1.4s where the model path took 27s; the trap falls through to the
  model, which refuses it.
- **Model management and keep-warm** — `openknowledge model list/use/status`
  reads what is installed and its real context window from the runtime, pins
  windows via derived models on Ollama, and the server warms the model at start
  in the background so the first question stops absorbing a silent multi-minute
  load.
- **Desktop foundations** — state resolves to the platform's per-user data
  directory when run outside a deployment (`openknowledge paths` explains every
  location and why); the web UI ships inside the wheel and resolves across
  checkout, wheel, PyInstaller bundle and container; serve binds 127.0.0.1 by
  default with a loopback Host allowlist (DNS-rebinding defence); app-mode
  serving mints an owner-only admin token; windows-latest CI leg; tracked
  lockfile. Ten findings from an adversarial review of the first version are
  fixed and pinned as tests.
- **Management surface** — knowledge managed from the browser instead of the
  server's filesystem. Uploads land through the chat widget (drag-and-drop) or
  the manage page, filenames flattened and whitelisted because they are
  attacker-controlled strings, one re-index per batch; the reply is a report,
  not a toast: stored, replaced, refused-with-reason, corpus size, and any
  contradiction the new document just opened. A whitelist of settings is
  editable at runtime — validated with their full field constraints (a
  constraint dropped in validation accepted `retrieval_k=999` in the first
  version), **live** ones apply on the next question, **rebuild** ones build
  the new engine first and only then swap, so a bad change leaves the old
  engine serving and nothing persisted. `/manage` puts documents, the review
  queue, contradiction resolution, settings and the cost report behind one
  pasted admin token; a test pins every endpoint the page calls to a route the
  app actually serves. Driven live in a real browser: the settings save
  correctly rejected a whole batch over one bad null before the fix, and the
  re-drive persisted `OK_RETRIEVAL_K=5` to the state `.env`.
- **Desktop app** — `openknowledge desktop` goes from a double-click to a
  serving chatbot: the two measured models downloaded once with resume and
  pinned SHA-256s (the exact bytes the accuracy numbers were produced on — a
  mismatch is refused, not warned about), two bundled llama.cpp servers
  spawned on loopback, the same FastAPI app on 127.0.0.1, a tray icon that
  degrades to a console loop where no tray exists. The launch plan is a pure
  function over the persisted state: an endpoint someone re-pointed at their
  own Ollama is never overwritten and never spawned over, and a dead
  llama-server reports the tail of its own log instead of "timeout".
  PyInstaller builds one folder holding both executables (windowed launcher,
  console CLI); the bundle is built and its frozen `serve`, `/`, `/manage`
  and `paths` verified on Linux in development, and the Package CI workflow
  does the same on windows-latest — plus `fetch-llama.ps1` (llama.cpp
  win-vulkan: GPU via Vulkan, runtime-dispatched CPU otherwise), the Inno
  Setup per-user installer, a silent install, and a run of the installed
  app — gating every packaging change. The first green run needed three
  attempts, each failure a real bug the run itself caught: a
  case-insensitive executable-name collision invisible on Linux, then
  llama.cpp's release layout change — and produced a 63 MB
  `OpenKnowledge-Setup-0.1.0.exe`, silent-installed it on windows-latest,
  and ran the installed app. A dispatch-only e2e job then runs the whole
  product from that installer on a real Windows machine — true first run,
  nothing overridden — and its first green run is recorded in
  `evals/measured/windows-e2e-first-run.json`: serving 24s after launch
  (2.6 GB downloaded and hash-verified inside that), an uploaded document
  answered by the real model in 28.4s at $0 — "The monthly pet cleaning
  fee is EUR 15 [pet-policy]." — and the same question again byte-identical
  from the exact cache. Getting there took two more real Windows bugs: the
  frozen Tk progress window can die natively (Tcl_Panic aborts below
  Python's reach), and `os.fchmod` does not exist on Windows (guarded).
  The first hands-on install then reshaped first run entirely: the app now
  serves before any model exists, the browser asks consent for the 2.6 GB
  download and shows live progress, stalls retry and resume themselves
  (the field laptop's connection died every ~190 MB and each relaunch
  banked the progress), a dead connection ends in a Resume button rather
  than a native dialog, and the engine is swapped in live once the models
  answer - tkinter left the bundle entirely, deleting the native-crash
  class with it. Unsigned so far: SmartScreen consequences and the
  signing plan are in [WINDOWS.md](WINDOWS.md).
- **Website** — a single self-contained page (`web/site/`) leading with the free audit,
  quoting only numbers this repository produces, and stating what the project cannot do yet.
  It fetches nothing from any other host, asserted by a test. Its contact form posts to the
  same container that served it, storing submissions in the operator's own SQLite file, read
  with `openknowledge contacts`. Off by default.
- **First live-model run** — the self-hosted tier measured end to end, not simulated.
  Qwen3-4B (Q4_K_M) on four CPU cores over 10 documents in Markdown, Word, Excel and PDF:
  **100% accuracy, 0 false answers, 100% determinism, 100% paraphrase consistency, $0.00000
  per question**, with the live contradiction correctly refused and eight unanswerable
  questions declined. The golden set is checked to be capable of failing — every answerable
  case rejects its own forbidden answer, asserted by a test. Numbers and their caveats in
  `evals/measured/first-live-run.json`; five runs before it found four real bugs.
- **Configuration comparison** — `tools/compare_configs.py` runs one golden set against
  every configuration in `evals/profiles.yaml` (self-hosted, open-weight, ladder, frontier)
  and prints them side by side, each in its own data directory so a warm cache from one
  cannot flatter the next. A profile whose keys are absent is skipped with a reason rather
  than failing, so the same command works with no keys, one, or both.
- **Test corpus and pre-flight** — eleven synthetic documents across Markdown, Word, Excel
  and PDF built around traps (conditions, negations, a live contradiction, a superseded
  copy, near-misses that must stay quiet), a 26-case golden set over them, and
  `eval --dry-run`, which checks with no model and no cost that every answerable case has
  its evidence in the retrieved context. About half of a new golden set's failures are the
  set's own, and all of them are free to find.
- **Evaluation harness** — golden set with a first-class safety set, scoring for accuracy,
  false answers, determinism and paraphrase consistency, cost reported alongside, and
  baseline comparison that fails CI on regressions. See [EVALUATION.md](EVALUATION.md).

## Next — makes it useful in a real company

1. **A real run against a paid tier.** The self-hosted tier is now measured — see below —
   but the open-weight and frontier rungs are not, so the escalation rate the whole cost
   model turns on is still assumed. That needs an API key.

   Everything else is ready: `tools/compare_configs.py` runs all four configurations from
   one command and skips the ones whose keys are missing. See [TEST-RUN.md](TEST-RUN.md).

   ~~Every cost lever is now built and measured; not one *answer-side* number
   is, because no model has been called.~~ Draft yield, gate pass rate, the escalation rate
   the whole cost model now turns on, and whether an open-weight rung actually grounds
   answers are all assumed. **Everything needed to do it now ships**: a synthetic corpus
   built around traps (`evals/corpus/aveline`), a 26-case golden set with a nine-case safety
   set (`evals/golden-aveline`), a free `eval --dry-run` that proves the set is answerable
   before you spend anything, and a step-by-step guide for all three tiers in
   [TEST-RUN.md](TEST-RUN.md). What is missing is an API key. **This outranks everything
   below it, by a distance.**
2. **A cross-encoder reranker** behind the `Reranker` protocol that already exists. The
   shipped one is free and model-less and fixes three specific BM25 failures; a real
   cross-encoder (`bge-reranker-v2-m3`, Apache-2.0, 80–200 ms for 100 documents on CPU)
   typically adds 5–15 NDCG@10 points on top. Worth the dependency only once a live run
   shows what the free one leaves on the table.
3. **A labelled real corpus for contradictions.** The contract run gave the audit an
   output somebody can read; it did not give it a precision figure, because that corpus
   has no true contradictions in it. What is needed is one company's own policy folder
   with the real disagreements marked. Until that exists, every claim about detection
   accuracy rests on 21 curated cases, and the contract run is the proof that this is
   not enough. **This now outranks everything below it.**
4. **Scope.** Per-vendor and per-country documents are not contradicting each other, and
   the detector cannot tell. Two candidate signals, both free: the named counterparty in
   each document, and the folder a document sits in. Neither is built.
5. **Grow the golden set.** The harness is built; the shipped set covers the sample
   documents only. Real corpus, real questions, and above all more safety cases — they
   are the cheapest insurance in the project.
6. **Contextual chunk embedding.** Hybrid retrieval shipped; the remaining half of
   the contextual-retrieval idea is prepending each chunk's heading trail before
   embedding it - the chunker already carries the trail for a different reason.
   Worth measuring against the golden set like everything else.
7. **Signing the Windows installer.** The installer itself now builds and
   smoke-tests in CI — what remains is identity. Azure Trusted Signing
   (~$10/month, reputation attaches to the verified identity) or a classic
   OV certificate (~$200–400/year, reputation builds per-certificate), then
   one `signtool` step in `package.yml`. Until then SmartScreen shows
   "unrecognized app" and some AV products distrust young unsigned
   PyInstaller binaries on reputation alone. See [WINDOWS.md](WINDOWS.md).
8. **SharePoint connector.** Microsoft Graph enumeration via `delta` (changes only, never
   a full rescan), `Sites.Selected` for least privilege, text extraction, and — the real
   work — mapping item permissions, including group expansion and inheritance, onto
   `allowed_principals`. Graph itself is free to call; the cost is the M365 licences the
   company already has.
9. **Google Drive connector.** Same shape: service account with domain-wide delegation,
   `files.list`, and permission mapping including inherited folder ACLs.
10. **Teams channel.** Written against the **Microsoft 365 Agents SDK** — the Bot Framework
   SDK is archived — with the asker's tenant groups supplying `principals` so access control
   works from the identity Teams already has. Teams is a standard channel, so messages are
   free and unmetered.

11. **The company shape: one server, every laptop a browser.** The topology
   works today and [COMPANY-SERVER.md](COMPANY-SERVER.md) documents it;
   these three make it enterprise-grade, in order:
   - **Sign in with Entra ID (Microsoft company credentials).** OIDC login
     on the server; the session's user and group ids become the
     ``principals`` the retrieval and cache ACL machinery already enforces -
     the lock exists, this adds the badge reader. Needs a real tenant to
     test against.
   - **Folders as categories.** Subfolders already index (the connector
     walks recursively); make them visible structure - grouped sidebar,
     upload-into-folder, admin organisation in /manage - then scoping
     ("answer from HR/ only") and finally per-folder ACLs mapped from
     Entra groups.
   - **Escalate on the company's Azure OpenAI tenant.** Slots into the
     existing ladder as configuration plus Azure's auth header. Stated
     honestly: a Microsoft 365 Copilot seat is not a callable completion
     API, so "use our Copilot subscription" translates to Azure OpenAI in
     the same tenant - same models, same data boundary, per-token billing.
     The provider seam is ready if Microsoft ever opens a Copilot
     inference API.

12. **Let llama.cpp size the GPU offload.** The desktop app now runs one
   slot (`--parallel 1`) and falls back to CPU (`-ngl 0`) when a GPU cannot
   hold the model - the field fix for an integrated GPU refusing the KV
   allocation. llama.cpp ships its own memory-fitting machinery
   (`llama-fit-params`, and newer servers can size offload against the
   device heap); adopting it would replace the binary GPU-or-CPU choice
   with "as many layers as actually fit". Worth doing when it stabilizes;
   the fallback stays as the floor either way.

### Known gaps in document parsing

- **No OCR.** A scanned PDF is reported and indexed as nothing.
- **PDF headings are inferred from type size** on the pdfplumber backend, so a document
  that styles headings at body size reads as one flat section. OpenDataLoader reports
  levels explicitly and does not have this limitation.
- **Borderless tables are missed by both PDF backends**, so a purely visual table with no
  ruling lines reads as prose. A text-alignment fallback was built and removed: on 15 real
  contracts it found no genuine borderless table and fabricated 2,983 rows out of prose.
  OpenDataLoader's `--hybrid` backend would close this, but it needs a running Docling or
  Hancom server, which breaks the no-external-calls promise.
- **Spreadsheet formulas are read as last-saved values**, which can be stale.

### Withdrawn after measurement

- **A scored answer confidence.** Built from free signals, then measured against
  degraded retrieval: 13 of 17 cases got *more* confident on *less* evidence, and at
  `k=2` no penalty fired at all. Every signal was a property of the retrieval setting
  rather than of the answer. Replaced by `Answer.support`, the grounding gate's own
  figure, which is a fact rather than a prediction. See
  [EVALUATION.md](EVALUATION.md) and `retrieval/confidence.py`.

### Known gaps in contradiction detection

Worth stating plainly, and the first one is the largest thing wrong with this project:

- **Precision on a real corpus is unmeasured, and was recently catastrophic.** On 15 real
  vendor contracts the detector emitted 320 findings and no useful ones. Salience weighting
  and duplicate-pair grouping took that to 6 listed findings plus 6 correctly identified
  duplicate pairs — but those 6 listed findings are still all false, and that corpus contains
  no true contradiction to find. The labelled set is at 100/100 and always was, throughout.
  See [KNOWLEDGE.md](KNOWLEDGE.md#what-a-labelled-set-could-not-tell-us).
- **No notion of scope.** Fifteen contracts with fifteen different counterparties have no
  business agreeing with each other, and nothing here knows that. This is the single biggest
  correctness gap: the detector assumes every document in the folder speaks for the same
  authority about the same world. Per-vendor, per-country and per-client corpora break that.
- **No deontic marker, no detection.** "The policy was withdrawn in March" against a document
  still stating the policy is invisible to the pattern passes.
- **English-only.** The marker vocabulary is English; another language gets numeric
  detection only.
- **Retrieved-window bound.** The cross-check compares against the passage that matched, so a
  contradiction stated elsewhere in a long document is missed.
- **Validated on 21 curated cases.** Enough to catch a regression, and demonstrably not
  enough to characterise a real corpus — that is exactly what the contract run showed.

## Later

- Admin web UI (pins, costs, prompt, connectors) — the API exists, the UI does not
- Slack channel adapter
- Per-document prompt caching for hot documents — the only caching lever that pays here,
  since the system prompt is measured at 476 tokens and cannot cache at all
- Per-rung retrieval width, for a rung whose context window cannot take the full set
- Incremental re-indexing (today's full rebuild is correct but O(corpus))
- Postgres + pgvector backend for multi-instance deployments
- Batch pre-warming: answer the top questions overnight at the 50% batch rate
- Conversation follow-ups ("what about contractors?") — needs care against a single-question
  cache key
- Structured document handling: tables, spreadsheets, diagrams
- OpenTelemetry traces per tier

## Explicitly not planned

- **A hosted SaaS.** The premise is that documents stay on your infrastructure.
- **Fine-tuning.** Gives up determinism and pins you to a model version, for something
  retrieval already does.
- **An agent loop.** A large cost multiplier that should be earned with evidence.
- **Telemetry.** No usage data leaves the deployment. Ever.
