# Roadmap

Honest status. "Built" means implemented and covered by tests in this repository.

## Built

- **Cost accounting** — per-call token accounting, rates with verification dates, an
  unpriced model raises rather than reporting $0, ledger and blended cost report.
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
- **Evaluation harness** — golden set with a first-class safety set, scoring for accuracy,
  false answers, determinism and paraphrase consistency, cost reported alongside, and
  baseline comparison that fails CI on regressions. See [EVALUATION.md](EVALUATION.md).

## Next — makes it useful in a real company

1. **A labelled real corpus for contradictions.** The contract run gave the audit an
   output somebody can read; it did not give it a precision figure, because that corpus
   has no true contradictions in it. What is needed is one company's own policy folder
   with the real disagreements marked. Until that exists, every claim about detection
   accuracy rests on 21 curated cases, and the contract run is the proof that this is
   not enough. **This now outranks everything below it.**
2. **Scope.** Per-vendor and per-country documents are not contradicting each other, and
   the detector cannot tell. Two candidate signals, both free: the named counterparty in
   each document, and the folder a document sits in. Neither is built.
3. **A real run.** Nothing here has been executed against a live model: every answer-side
   quality number is measured with scripted providers, and the draft yield, gate pass rate
   and local-tier competence are all assumed. Point it at a folder of real policies with a
   model configured and the assumptions become measurements.
4. **Grow the golden set.** The harness is built; the shipped set covers the sample
   documents only. Real corpus, real questions, and above all more safety cases — they
   are the cheapest insurance in the project.
5. **Semantic cache.** Local embeddings over canonical questions, similarity
   threshold, and a citation check before serving a near-match. This is where the free share
   grows, and it needs to be careful: a hash cannot decide two sentences mean the same thing,
   so a bad match must be catchable.
6. **Hybrid retrieval + reranking.** BM25 fused with local dense retrieval, then a
   cross-encoder rerank. The biggest single cost lever is sending fewer, better chunks — this
   is cheaper *and* more accurate, which is a rare combination.
7. **SharePoint connector.** Microsoft Graph enumeration, text extraction, and — the real
   work — mapping item permissions, including group expansion and inheritance, onto
   `allowed_principals`.
8. **Google Drive connector.** Same shape: service account with domain-wide delegation,
   `files.list`, and permission mapping including inherited folder ACLs.
9. **Teams channel.** Bot Framework adapter, with the asker's tenant groups supplying
   `principals` so access control works from the identity Teams already has.

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
