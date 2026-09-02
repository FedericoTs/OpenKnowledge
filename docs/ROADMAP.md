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
- **Document tags and routed retrieval** — every indexed document gets a derived,
  readable tag set (folder taxonomy, title, headings, top tf-idf terms — free, no
  model, shown in the listing), and a question that names its documents decisively
  is guaranteed to find them among its retrieval candidates — rescued from below
  the cut when a large corpus buries them, never filtered, never reordered. Both
  stronger designs were caught by the golden sets within a run each (a filter
  starved a one-chunk document's context; routed-first ordering filled the
  context with same-topic tables and dropped aveline to 0.88), which is what the
  sets are for. Two folded tag hits to match, a small matched share to fire, any
  ambiguity changes nothing. `OK_TAG_ROUTING=false` restores pre-tag retrieval
  exactly; both corpus-and-set preflights run in the unit suite. See ADR 0011.
- **One-click verified updates** — the desktop app checks the pinned GitHub repo
  once a day (`OK_UPDATE_CHECK=false` disables; the check is documented as the
  outbound call it is) and offers new releases as a sidebar button: download,
  verify against the release asset's own SHA-256 digest, clean shutdown through
  the launcher's quit path, silent install, relaunch. Never silent-by-default and
  never unverified: a release without a digest is not an update, a tampered
  download is deleted and refused, a server install tells you its operator
  updates it. Admin-only when sign-in is on.
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
4. **Scope — the counterparty half is built** (v0.8.0). Two agreements with different
   companies are not disagreeing, and the detector now knows it: parties are read from
   the positions that define them, the party common to the most documents is dropped as
   your own, and a pair is compared unless both name parties and share none. On a corpus
   of the shape that broke it, 534 findings across 136 pairs became 3 across 3.
   **The folder signal is deliberately not built**: two folders are as often a topic
   split as a scope split, and suppressing across them would hide a real
   HR-versus-Finance contradiction. What remains is per-country and per-client corpora
   that do not name parties in contract language — and a precision figure, which needs
   item 3.
5. **Grow the golden set.** The harness is built; the shipped set covers the sample
   documents only. Real corpus, real questions, and above all more safety cases — they
   are the cheapest insurance in the project.
6. ~~**Contextual chunk embedding.**~~ **Already built, and now measured.** The
   chunker states each passage's heading trail once at the top of the passage
   (`_passage` in `retrieval/base.py`), so it is already part of what gets
   embedded - and of what BM25 indexes, which is more than this item asked
   for. It arrived as a side effect of the "heading said once" work; this
   entry simply predated it. Checked rather than assumed: **68 of 68** chunks
   on the aveline corpus begin with their trail.

   What was never established is whether it earns the space. `tools/measure_context.py`
   answers that - the rank of each labelled case's required citation, with the
   trail in the embedded text and without, by cosine alone so BM25 cannot
   answer for it. On aveline the trail is **13.7% of every embedded passage**
   and **not one case changes rank**: median 1.0, mean 1.14, top-1 12/14 both
   ways.

   That is a fact about the corpus at least as much as about the trail. Eleven
   documents whose questions name their subject leave nothing to disambiguate,
   and a required document already at rank 1 cannot move up. The measurement
   is worth re-running on a corpus that can discriminate - which is item 3
   above, and needs somebody's real folder.
7. **Signing the Windows installer — the pipeline is built, the identity is not.**
   `package.yml` signs the two executables and the installer with Azure Artifact
   Signing (formerly Trusted Signing, ~$10/month, reputation attaches to the
   verified identity) the moment six repository variables exist, authenticating
   with the job's own OIDC token; it verifies all three signatures, records the
   state in the artifact, and the release notes say "Signed by …" or "Not
   code-signed". What remains is the Azure account and identity validation,
   which only the project's owner can do — the steps are in
   [WINDOWS.md](WINDOWS.md). Until then SmartScreen shows "unrecognized app"
   and some AV products distrust young unsigned PyInstaller binaries.
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
     the lock exists, this adds the badge reader. Designed in
     [ENTRA-SIGNIN.md](ENTRA-SIGNIN.md): generic OIDC tested against a
     fake IdP in CI, Entra documented first; only Microsoft's half of the
     flow needs a real tenant to verify.
   - **Folders as categories.** Built, both halves. The sidebar groups
     documents under their folders with counts (loose files under
     "Unfiled"), uploads pick a destination folder - or name a new one -
     from the widget and from /manage, deleting a nested document
     addresses its full path (flattening once made `HR/x.md` delete a
     root-level `x.md`), and an emptied folder stays listed, because a
     category an admin made is a decision. And folders now carry access:
     admins rule them in /manage (`group:<id>`, `user:<id>`,
     `authenticated` - the ids sign-in mints), the deepest rule wins for
     its subtree, documents are stamped at index time, and the one rule
     holds everywhere at once - answers, every cache tier, the corpus
     listing, the sidebar, uploads and deletes. Still ahead: scoping
     ("answer from HR/ only").
   - **Escalate on the company's Azure OpenAI tenant.** Built - and it
     was exactly the size claimed: a thin dialect over the existing
     adapter (deployments URL, api-version, api-key header) behind
     `OK_ESCALATION_PROVIDER=azure`, setup documented in
     [AZURE-OPENAI.md](AZURE-OPENAI.md). The ledger uses the price the
     operator states from their own Azure agreement, and flags calls
     "cost not counted" rather than guessing when none is stated.
     Stated honestly, as before: a Microsoft 365 Copilot seat is not a
     callable completion API, so "use our Copilot subscription"
     translates to Azure OpenAI in the same tenant - same models, same
     data boundary, per-token billing. The provider seam is ready if
     Microsoft ever opens a Copilot inference API.

12. **Let llama.cpp size the GPU offload.** The desktop app now runs one
   slot (`--parallel 1`) and falls back to CPU (`-ngl 0`) when a GPU cannot
   hold the model - the field fix for an integrated GPU refusing the KV
   allocation. llama.cpp ships its own memory-fitting machinery
   (`llama-fit-params`, and newer servers can size offload against the
   device heap); adopting it would replace the binary GPU-or-CPU choice
   with "as many layers as actually fit". Worth doing when it stabilizes;
   the fallback stays as the floor either way.

13. **A grounding gate that knows a summary from an extraction.** Built,
   as the citation-conditioned floor the field case asked for. The
   finding, measured on a real corpus: "what does the document cover?"
   produced a faithful six-bullet summary, every bullet cited, withdrawn
   at 42% content-word support against the 45% floor - a good summary
   compresses and rephrases, which is exactly what one global ratio
   penalises. Now an answer that earns it - every substantive claim
   carries a resolving citation, no unknown ids, no unverified figures -
   is graded against `min_support_ratio_cited` (default 0.30, live in
   /manage) instead of `min_support_ratio` (0.45, unchanged). The
   relaxation is judged per answer on that answer's own discipline;
   uncited prose, invented ids and wrong numbers keep exactly the old
   treatment, and fully-cited invention still dies at the lower floor.
   The prompt (v2) teaches the shape the gate rewards: overview and list
   questions are answerable by enumerating what the sources state even
   when the question's word ("priorities", "steps") never appears in the
   document, summaries stay in the sources' own vocabulary, and every
   bullet and sentence is cited. Gated the house way: golden set and
   safety set re-run against the same local model before and after.

### The refusal, kept

A refusal is the most useful thing this product says, and it was the only
thing it said that nobody kept. Every other tier leaves a trace - an answer, a
cache entry, a line in the cost report - while "I don't know - that isn't
covered by the documents I have" was said once and forgotten, so the person
who owns the corpus never learned that eleven colleagues had asked the same
unanswerable question that month.

`openknowledge gaps` and `GET /admin/gaps` now rank those questions by how
many people asked. Each line is a document worth writing, or a question worth
pinning, in the order worth doing it. A system that guesses cannot produce
this report at all: it has no refusals to count.

The report is aggregate by construction. The ledger it reads has no identity
column, so it can say a question was asked forty times and never who asked it
- a knowledge base that reports what its people are looking for should not
also be a log of who looked. A test asserts that column stays absent.

### Who changed what, and who may

Every other decision this product records says *what* was decided. The
`folder_access` table held a folder, its readers and a timestamp, and no
column for the person who set it. Nothing anywhere recorded who deleted a
document, approved a draft, resolved a contradiction or changed a setting.
"Somebody walled off the HR folder last Tuesday" had no answer.

Thirteen mutating admin actions now write to an admin log kept beside the
other human decisions, and read back at `openknowledge admin-log`, `GET
/admin/log`, or the *Admin log* panel on `/manage`. A signed-in person is
named by their directory subject id, so a row survives a rename and points
at an account somebody can disable. A change made with the shared admin
token is recorded as naming nobody, and the log says so on every surface
that shows it: no amount of logging recovers an identity a shared secret
never carried, and pretending otherwise would be worse than the gap.

`OK_OIDC_CURATOR_GROUP` splits the admin surface in two, because the people
who know the answers are rarely the people who should hold the access
rules. Curators shape what the assistant says — pins, drafts,
contradictions, documents, re-indexing. Admins additionally decide who may
read a folder, what the settings are, when to update, and who has been
doing all of the above. Every admin is a curator; an install that sets no
curator group behaves exactly as it did.

Building it found a hole. Driving attribution end to end meant signing in
as somebody who was *not* an admin — and their `DELETE /documents/...`
returned 200 with the file gone, because the documents endpoints gated on
the uploads switch rather than on who was asking, so readable meant
deletable for everyone in the company. Uploading over an existing filename
was the same hole by another route. Both now need the curator role when
sign-in is on; contributing a *new* document still does not, because
contribution is what that switch is for. With sign-in off nothing changed:
reaching the port was always full control there, and a role check against
an identity that does not exist is theatre.

The asymmetry with the gaps report above is deliberate and now holds from
both ends. Who governs is recorded by name. Who is curious is not.

### A PDF is not a markdown file

The plan was to cache document parses, listed as second-tier work behind
things that needed somebody else. Measuring the number it rested on moved
it to the front.

Per document, on this project's own parsers: markdown **5.9 ms**, docx
**56.7 ms**, PDF **780 ms**. A hundred and thirty-two times, for the same
words. Profiling put **99.2%** of a PDF rebuild in `opendataloader`, which
spawns a Java process once per file — the cost is JVM startup and pipe
traffic, not reading the document. Sixty small PDFs rebuilt in 46 seconds;
a thousand policy PDFs, which is an ordinary corpus for the company this is
built for, is minutes, paid again on every upload and every delete. PDF is
the format a company's policies actually live in, so this was never
second-tier.

None of that work is new each time, so it is now remembered — keyed on the
**content**, not the clock. `mtime` and size are the obvious key and the
wrong one: `rsync -t`, `git checkout` and every restore-from-backup put old
timestamps on new bytes, and a cache that believed them would serve last
year's policy for ever. A test rewrites a document to the same length,
rolls its mtime backwards, and asserts the new figure reaches the corpus.

Rebuild: 46 s → **0.23 s**, and 0.74 s across a restart because it is on
disk. A warm cache and no cache at all produce the same `corpus_version`
and the same passages, chunk for chunk.

It is deliberately not in the backup — a backup already carries the
documents, and a parse of every one would double the archive to save a
rebuild — and deliberately not used by `openknowledge audit`, which
promises no database and nothing written.

What it did **not** fix was the first index, and its own record said so: a
cold corpus of a thousand PDFs was still minutes, because each one still
started a JVM. That is now fixed too. Measured directly rather than
inferred: of the 656 ms a four-page PDF cost, about **640 ms was the
process starting up** and roughly 51 ms was parsing. The parser accepts a
batch, so it is handed one — 64 documents per invocation, which takes ~13x
of a possible ~16x while bounding both the files staged on disk and what
the parser holds at once.

A first index of 120 PDFs: **65.7 s → 3.1 s**, twenty-one times faster, and
the documents that come out are asserted identical — blocks, title, pages
and warnings, per document and across the whole corpus. A thousand policy
PDFs go from about nine minutes to about twenty-five seconds. With a warm
parse cache the same scan starts no JVM at all.

Worth being exact about **who was waiting nine minutes**, because it was
never everybody. The Java parser is an optional extra: the Docker image
installs it and a JVM, the Windows installer does not and never has. So a
Windows install has always used pdfplumber, measured at **59.7 ms** a
document — a minute for those same thousand PDFs, and no JVM problem to
have. What changed is the ranking: OpenDataLoader was eleven times slower
than pdfplumber and chosen anyway, for structure it reports rather than
infers. Batched it is **20.9 ms** a document, so it is now about three
times *faster* as well. The better parser is no longer the slow one.

The reading-ahead is driven by the walk and stays **one group ahead**
rather than parsing everything first: a group holds every document in it in
memory, and reading a thousand large PDFs before indexing any of them would
trade a JVM problem for a memory one. Doing it eagerly measured 22x against
this 21x — inside the noise, and not worth holding a second copy of a
corpus for.

Two things would have been silent. The parser names each output after its
input's **basename**, so batching `HR/policy.pdf` and `Finance/policy.pdf`
by their real paths writes one `policy.json` — verified, not feared: two
different PDFs with the same name produced exactly one output file. That is
a document lost, or one policy answered out of another's text. Every file
is staged under a generated name instead. And the CLI exits non-zero when
any file in a batch is unreadable, *having already written every good
document*, so the failure is read rather than raised and anything missing
is re-parsed alone — where it produces the same sentence it would have
produced had it never been batched.

That last test found a defect it was not looking for. The sentence an
operator saw for an unreadable PDF was the whole `java` command line and an
exit code. It now reads `broken.pdf: OpenDataLoader: this file is not a
valid PDF file (corrupted or truncated content).`

### An upload is one document, not the whole corpus

With parses cached and PDFs batched, what an upload still paid for was
re-indexing everything. Measured before touching it: **1.04 s** at 400
documents, **3.42 s** at 1,200, **7.23 s** at 2,400 — linear, about 3 ms a
document, so five thousand policies is fifteen seconds of waiting after
dragging in one file. Profiling put 79% of that in the BM25 index build and
a third of *that* in deriving tags, almost all of it tokenising the same
unchanged documents again.

Chunking a document, tokenising its passages and counting its words depend
on **that document alone**, so they are remembered. The tf-idf ranking
behind its tags is **not**, and that is the point rather than an omission:
adding one document changes what every other document's words are
distinctive against, and a cached tag set would be right on the day it was
computed and quietly drift from then on. It is also the cheap half — the
tokenising is what costs.

Index build **2.30 s → 0.13 s** on 1,200 documents; an upload end to end
3.42 s → 1.22 s. Memory went from 68.0 MB to 77.8 MB, the cache accounting
for 9.8 MB of that.

It is still a full rebuild — what is reused is the work, not the result.
`corpus_version`, the chunks and every tag are rebuilt from the current
corpus, so a deleted document really does disappear.

The key is **everything a chunk is made of**, not the content hash.
`content_hash` was the obvious choice and is wrong twice: a chunk carries
`allowed_principals`, so a text-only key could hand back a passage stamped
with a folder's *previous* audience; and chunking reads `blocks`, so a
heading rewritten as a paragraph leaves the text byte-identical and the
passages different. Hashing all of it costs 22 ms per 1,200 documents
against the 2.3 s it saves, so nothing was traded off.

Held to the same standard as the rest: on a 150-document corpus of mixed
formats in nested folders, the corpus document frequency, every document's
tags, and the whole index — chunks, term frequencies, lengths, document
frequencies, average length, `corpus_version`, principals and both tag maps
— are asserted identical to what the previous implementation produced.

### The same words, flattened once

With parses cached, PDFs batched and per-document index work reused, what an
upload still spent its time on was re-reading the folder: `connector.fetch()`
was **47%** of it. Profiling put 1.09 of those 1.54 seconds in
`ParsedDocument.text` — `normalise` makes six full passes over a document,
and 22 MB of text at roughly 21 MB/s a pass is a second, paid on every upload
and every delete, for documents where nothing changed.

It is a pure function of the blocks, so the parse cache stores it beside
them. Re-reading 1,200 documents: **1.55 s → 0.42 s**; one upload end to end
3.15 s → 1.95 s.

The cost is disk. `parses.db` goes from 1.1x to **2.1x** the size of the
corpus text. That is the right way round here — the file is derived data that
can be deleted for the price of one rebuild, and what it buys is a wait
somebody is sitting through.

The field is excluded from equality, which matters more than it sounds: a
document read from the cache carries the text and a freshly parsed one does
not, and those two **are the same document**. Several tests assert exactly
that, and they are asserting something true.

Worth recording next to the numbers: measured again hours later, the same
unchanged code took 3.15 s where the v0.7.1 record had said 1.22 s. Nothing
regressed — this container had since downloaded 2.6 GB of models and run two
eval suites, and page cache and host contention are not controlled. **This
box is not a stable timing reference between sessions.** Every figure here is
a same-session A/B with the shipped code restored from HEAD between runs, and
the older records should be read as ratios rather than seconds.

### An access change is not a corpus change

Both access endpoints re-indexed the whole corpus, synchronously, inside the
request. Measured through `PUT /admin/access/hr` on 1,200 documents:
**9.6 seconds** to change who may read a folder — and it grows, so past a
proxy's timeout the admin sees the request fail while the change actually
applied, which invites a retry on something that looked like a checkbox.

What an access change actually changes was measured before anything was
written: `corpus_version` unchanged, zero answers evicted, and the only
difference anywhere is `allowed_principals` on the passages. That follows
from the design — `corpus_version` hashes content, and a rule is not
content; the answer cache re-checks a cached answer's sources against
whoever is asking at read time, so an access change invalidates nothing and
*must not*, or the hit rate the cost model depends on goes for nothing.

So a rule change now re-stamps instead of rebuilding: **93 ms**. The index
is swapped as one frozen object in a single assignment, the same atomicity
a rebuild has, so nobody is served through half of an access change. A
document the rules say nothing about keeps what it had — silence is not
permission, and reading an absent entry as "open to everyone" would be a
way to widen access by omission.

The obvious move was to make the rebuild asynchronous. That would have
broken a property the endpoint's own comment states: no window may exist in
which a rule is stored and the index is still serving the old audience. The
fix was to make the work small, not to defer it — and a test signs somebody
in, confirms they can see the folder, changes the rule and asserts their
very next request cannot.

### The rebuild, measured before it was optimised

The plan was incremental indexing. Measuring first changed what got built.

Embeddings turned out to be cached by chunk text already, so the expensive
part was incremental before anyone touched it. Profiling a rebuild found
the cost somewhere else entirely: contradiction detection was **57%** of
it, and pulling claims back out of documents alone was **48%**. The
retrieval index everybody would optimise first — BM25 at 29%, parsing at
14% — was the smaller half.

The comment that made it so said *"Conflicts are free, so re-run over the
whole corpus every time."* True of money, which is why it was written and
why re-running was affordable. Not true of the clock — and a rebuild runs
on every upload, every delete and every access rule, inside the request
that is waiting for it. At 1,200 markdown documents that was 27.5 seconds
to change who may read a folder.

So claims are now remembered per document, keyed by its text, while every
pair is still compared on every scan. Which conflicts are found does not
change — that is asserted directly, cached against uncached — and a
1,200-document rebuild went from 27.5 to 9.4 seconds.

**The bug it found on the way.** The test written to prove the cache
faithful compared a warm engine against a fresh one, and they disagreed.
The obvious reading was a stale cache; running the same check against the
unmodified code disagreed identically. A contradiction that had been
*corrected in the documents* stayed open forever, because the only thing
that ever cleared one was a document leaving the corpus. With
`block_on_conflict` on — the default — the questions it gated stayed
refused after the corpus was already right, and the only way out was an
admin resolving something that no longer existed. The scan now reconciles
what it found against what is still flagged, leaving resolved rows alone
because those are decisions somebody made.

### One asker, everybody's ceiling

The budget governor turns a declared budget into a ceiling on what one
question may cost, recomputed from the ledger every time. It is a good
design and it has a blind spot: it cannot see *whose* questions moved the
ceiling. A looping bot integration and forty colleagues asking one question
each look identical to it, so the first is paid for by the second.

`OK_ASKER_QUESTIONS_PER_MINUTE` closes that: over the limit, one caller
gets a 429 and a sentence explaining it, and everyone else is served
normally in the same breath. It is off by default, because a desktop
install should not meet a limit it never asked for, and it is a live
setting, because it is the lever an operator reaches for while the looping
is happening.

The counters keep no record of who asked what. They live in this process's
memory, are keyed by a salted per-process hash of the asker rather than by
the asker, and are gone at restart — the same promise the gaps report and
the reported-answers table make, made the same way: not by policy but by
there being nowhere for the data to go.

`/metrics` is the other half. `/healthz` says the server is up; it does not
say spend tripled at eleven, that a third of today's questions were refused,
or that one caller has been rate-limited four hundred times. Prometheus text
exposition of the ledger, the index and the limiter, admin-only, with no
question text and no identity in it — a metric with the question in it is a
log of what people asked, published to whatever scrapes it.

Writing that renderer produced a bug worth keeping in mind: emitting samples
in the order they were composed looks right and is not. The format requires
a metric family's lines to be contiguous, and the moment a second window was
added the two families interleaved — which a strict scraper rejects as a
duplicate. It groups by family now, and a test asserts the blocks rather
than the bytes.

### The wrong answer nobody heard about

A refusal was already the best-instrumented thing this product does: it is
counted, ranked by how many people asked, and reported as a list of
documents worth writing. An answer that was *wrong* had none of that,
because it looked exactly like an answer that was right. There was no
button, no table and no signal — somebody noticed, told a colleague, and
the documents never heard.

The answer card now carries **This is wrong**, and one sentence from
somebody who already knows the right answer is the whole fix. It lands in
*Answered wrong* on `/manage` (and `openknowledge reports`), ranked by how
many people said so, with the answer they were shown and the notes they
left; one box pins the correction and closes the report. Closing it is an
admin action, so it is attributed in the admin log like every other.

Three deliberate limits. Only questions this install actually answered can
be reported, so the table holds real answers rather than whatever anybody
posts. The same wrong answer to the same question is one row with a count,
because a hundred colleagues agreeing is one thing to act on. And a report
raised before a re-index is marked stale rather than deleted — the
documents changed underneath it, so it may already be fixed, and "we fixed
that" is a claim somebody should be able to check.

It records nothing about who reported it. That is the same promise the
gaps report makes about who asked, and it is checked the same way it is
made: by dumping the database and asserting the reporter's name and
subject id appear nowhere in the bytes.

### The update button, cause two

"I still don't see the update button after several releases" had two
independent causes, and fixing either alone would have left the report
standing. The first was the installer: Inno removes nothing a new build
stopped shipping, so PyInstaller's version-named `.dist-info` directories
piled up and an upgraded install reported the old version — eighteen
releases published, none ever received.

The second was in the page. `refreshUpdateChip()` and the fetch that fills
the version label had drifted inside `removeDocument`'s try block — still
at column zero, but inside the function — so a fresh load showed a blank
footer and an empty chip, and the version appeared only after somebody
deleted a document. Both are fixed.

The guard against the second is worth reading. CI installs no browser, so
the browserless check is the one that has to hold, and its first version
asserted the boot call appears at column zero — which was *true of the
bug*. Scope was what was wrong, so it now computes brace nesting depth
over source with strings and comments blanked, and asserts an occurrence
runs at depth zero. Confirmed by putting the bug back and watching it go
red.

### The relaunch, finally tested

The v0.5.0 release went green with a PyInstaller traceback inside its
upgrade job. The traceback was the harness — a `--version` probe that
launched the frozen app while the installer was overwriting its
`base_library.zip`, absorbed by the retry loop as "not yet" — but two
defects were underneath it.

The check could pass mid-install. Inno's log says an install takes 6.57
seconds and the version-named `.dist-info` that `--version` reads lands
2.96 seconds in, with 400 files still to write. That run declared the
upgrade landed 1.9 seconds before the installer finished, then ended; the
runner found no app alive at cleanup, where the previous release's run had
left one. The check now waits for the app to answer `/healthz` reporting
the new version — the running process, which no half-written directory can
satisfy.

And the hop it exists to prove had never run. The job started
`openknowledge.exe serve` and told the handoff to relaunch
`openknowledge.exe` — the CLI, which with no subcommand prints a usage
error and exits. It has been relaunching nothing since it was written, and
passing, because it only read a version string off disk. It now drives
`OpenKnowledgeApp.exe`, the windowed entry the shortcut runs. Measured
after the fix: 15.2 seconds from handoff to the reopened app serving as
the new version, and no traceback anywhere in the log.

The product had a smaller version of the same mistake. `spawn_installer`
relaunched `sys.executable`, which is the windowed build when the shortcut
started it and the CLI when `openknowledge desktop` did — reachable from
any terminal, and what the installer's PATH option is for. That update
closed the app and never brought it back. `relaunch_target` now hands off
to the windowed build when the CLI is running, with the name pinned to the
one the bundle builds.

### Known gaps in the Windows upgrade

- ~~**Inno removes nothing the new build dropped.**~~ **Built.** An upgrade used to
  replace every file the new installer carries and leave everything else exactly
  where it was. That was fatal once: the frozen app reads its version from bundled
  `.dist-info`, whose directory name carries the version, so an upgrade left the old
  one beside the new and the app went on reporting the version it had replaced —
  which is why an install could never be seen to update itself.

  Clearing that one directory fixed the instance and left the class open. The
  installer now stops the app (`PrepareToInstall` → `taskkill` on the windowed app,
  the CLI and `llama-server`) and then clears `{app}\_internal` **wholesale**, so
  `[Files]` lays down exactly what the build carries and nothing else survives.
  The person's state is untouched: documents, database, models and settings live
  under `%LOCALAPPDATA%\OpenKnowledge`.

  Measured rather than declared: CI plants a file the new build does not ship into
  the old install's runtime and fails if it survives — once on a quiet install, and
  once while the app is **running and holding those very files**, which is the case
  the wholesale delete exists for. A source guard also pins that the update helper
  is PowerShell and so is not one of the images the installer kills; a helper on
  that list would be shot dead partway through its own install.

  Residual risk, stated plainly: a failure between the delete and the copy (disk
  full, antivirus, a cancelled install) leaves a runtime that re-running the
  installer repairs. That is the ordinary trade every replacing installer makes,
  and it is the price of not shipping a directory that accumulates every file the
  project has ever shipped.

### Known gaps in the golden sets

- **`must_not_say` is a flat substring list**, so it cannot express "not *as the answer
  to this question*". Two aveline cases forbade figures that a correct, fully attributed
  answer legitimately states - the allowed payment range beside the standard term, the
  annual training deadline beside the new joiner one - and failed answers that gave both
  with their labels. They now forbid only figures that cannot belong in a correct answer
  at all: another row's value, another rule's duration, a section heading read as a
  deadline. That is the right narrowing and it is still a proxy. A case that wants to
  say "30 days must not be given *as the new joiner deadline*" has no way to say it, so
  the check has to be aimed at what a wrong answer would contain instead of at what it
  would mean.

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
- **No timeout on a PDF parse.** `TIMEOUT_SECONDS` in `documents/opendataloader.py`
  says what a bound would be and is not enforced: the wrapper package offers no
  timeout and runs the jar with a blocking `subprocess.run`, so nothing on this
  side can interrupt a parse that never returns. A pathological or hostile PDF
  hangs an index, and now that PDFs are parsed in groups it takes its whole
  group with it. Fixing it properly means a `timeout` in the wrapper or
  spawning the jar here, which would duplicate the wrapper's job and drift with
  it. Not attempted; stated rather than left as a constant that reads like a
  protection nobody has.

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
- ~~**No notion of scope.**~~ **Built.** Two agreements with different companies are not
  disagreeing, and the detector now knows it. Scope comes from the parties an agreement
  declares itself to be *between* — never from a mention, so a policy that names a booking
  agent is not thereby scoped to it — read from the document's opening, with the party
  common to the most documents dropped, because your own company is on every contract you
  hold and therefore tells two of them apart from nothing. On a corpus of the shape that
  broke it: **534 findings across 136 pairs → 3 across 3**, none cross-vendor, with the
  same-vendor and policy contradictions both still caught. The rule only suppresses when
  *both* documents name parties and share none: silence is not a scope. Pairs left
  uncompared are reported, so "no contradictions" cannot quietly mean "never compared".
  Still open: per-country and per-client corpora that do not name parties in contract
  language. The folder signal is deliberately **not** used — two folders are as often a
  topic split as a scope split, and suppressing across them would hide a real
  HR-versus-Finance contradiction.
- **Supersession announced by another document is now read** — the half of this that
  is safe to act on. `declares_superseded` needs somebody to have gone back and edited
  the *old* file, and in practice nobody does; the statement that gets written is in
  the new document's header, `**Supersedes:** Expenses Policy v3.0 (January 2023)`.
  That line is in this project's own sample corpus and was ignored, because the
  self-declaration matcher deliberately skips "Supersedes:" — there it is the current
  copy talking about a *different* document, which is exactly what makes it useful
  here. See `knowledge/supersession.py`.
- **A withdrawal announced in prose is still invisible**, and that is now a decision
  rather than an omission. "The policy was withdrawn in March" would have to be
  *inferred*, and retrieval does not downrank a superseded document — it **excludes**
  it whenever any current document matches (`demote_superseded`). So a wrong inference
  takes a live policy out of almost every answer. A header field written by whoever
  replaced the document is a statement; a sentence in the body is a guess, and a guess
  is not worth that. The resolution is conservative for the same reason: the announcer
  is never its own target, every significant word of the target's title must appear in
  the phrase, and a tie retires nothing.
- **English-only.** The marker vocabulary is English; another language gets numeric
  detection only.
- **Retrieved-window bound.** The cross-check compares against the passage that matched, so a
  contradiction stated elsewhere in a long document is missed.
- **Validated on 21 curated cases.** Enough to catch a regression, and demonstrably not
  enough to characterise a real corpus — that is exactly what the contract run showed.

## Later

- Admin web UI — `/manage` has costs, the most-asked list, gaps, wrong answers, pinned
  answers, access rules, the review queue, contradictions, live settings and the admin
  log, whether each model endpoint answers, every setting in force with its default
  and its source, a backup button, and the map (documents joined by the contradictions,
  supersessions and co-citations the stores hold; also on the audit page). Deliberately not a UI: prompt editing (a
  prompt change goes through both golden sets, not a text box), connector setup,
  and restore (it overwrites the store, so it stays a command on the server).
- Slack channel adapter
- Per-document prompt caching for hot documents — the only caching lever that pays here,
  since the system prompt is measured at 476 tokens and cannot cache at all
- Per-rung retrieval width, for a rung whose context window cannot take the full set
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
