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
6. **Semantic cache.** Local embeddings over canonical questions, similarity threshold,
   and — the part the literature gets wrong for this use case — a **synchronous** check
   before serving. Published designs verify asynchronously: serve now, verify later, demote
   if wrong. A wrong policy answer served now has already been acted on. We can verify
   synchronously for free by retrieving for the *new* question and running the existing
   grounding gate on the *cached* answer against those fresh chunks. Production hit rates
   are 20–45%, not 95%; ~0.92 similarity is the usual balance point, and the golden set's
   paraphrase pairs are already the labelled data needed to calibrate it.
7. **Hybrid retrieval + contextual chunks.** BM25 fused with local dense retrieval, plus
   Anthropic's contextual-retrieval trick of prepending each chunk's place in its document
   before embedding — measured at 49% fewer retrieval failures, 67% with reranking. Our
   chunker already carries a heading trail on every chunk, which is most of that idea built
   for a different reason. Local embeddings are effectively free: `nomic-embed-text` does
   ~580 chunks/sec on a laptop CPU, and no vector database is needed at corpus scale.
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
