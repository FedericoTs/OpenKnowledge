# Roadmap

Honest status. "Built" means implemented and covered by tests in this repository.

## Built

- **Cost accounting** — per-call token accounting, rates with verification dates, an
  unpriced model raises rather than reporting $0, ledger and blended cost report.
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
- **Document parsing** — PDF (headings from type size, ruled and unruled tables),
  Word, Excel, PowerPoint and Markdown into structured blocks with citable locators,
  plus structure-aware chunking that never splits a rule from its condition. See
  [DOCUMENTS.md](DOCUMENTS.md) and ADR 0007.
- **Prose contradiction detection** — deontic claim extraction (must / may / must not) with
  predicate families and hard-versus-soft force pairs, plus a free FAQ cross-check that
  catches a newly uploaded document disagreeing with an existing answer. Measured by
  `eval-conflicts` on a labelled set that is majority near-misses. See ADR 0006.
- **Evaluation harness** — golden set with a first-class safety set, scoring for accuracy,
  false answers, determinism and paraphrase consistency, cost reported alongside, and
  baseline comparison that fails CI on regressions. See [EVALUATION.md](EVALUATION.md).

## Next — makes it useful in a real company

1. **A real run.** Nothing here has been executed against a live model on a real
   corpus: every quality number is measured with scripted providers, and the draft
   yield, gate pass rate and local-tier competence are all assumed. Point it at a
   folder of real policies with a model configured and the assumptions become
   measurements. This now outranks everything below it.
2. **Grow the golden set.** The harness is built; the shipped set covers the sample
   documents only. Real corpus, real questions, and above all more safety cases — they
   are the cheapest insurance in the project.
3. **Semantic cache.** Local embeddings over canonical questions, similarity
   threshold, and a citation check before serving a near-match. This is where the free share
   grows, and it needs to be careful: a hash cannot decide two sentences mean the same thing,
   so a bad match must be catchable.
4. **Hybrid retrieval + reranking.** BM25 fused with local dense retrieval, then a
   cross-encoder rerank. The biggest single cost lever is sending fewer, better chunks — this
   is cheaper *and* more accurate, which is a rare combination.
5. **SharePoint connector.** Microsoft Graph enumeration, text extraction, and — the real
   work — mapping item permissions, including group expansion and inheritance, onto
   `allowed_principals`.
6. **Google Drive connector.** Same shape: service account with domain-wide delegation,
   `files.list`, and permission mapping including inherited folder ACLs.
7. **Teams channel.** Bot Framework adapter, with the asker's tenant groups supplying
   `principals` so access control works from the identity Teams already has.

### Known gaps in document parsing

- **No OCR.** A scanned PDF is reported and indexed as nothing.
- **PDF headings are inferred from type size**, so a document that styles headings at body
  size reads as one flat section.
- **Unruled tables need a numeric column**, so a purely textual table reads as prose.
- **Spreadsheet formulas are read as last-saved values**, which can be stale.

### Known gaps in contradiction detection

Worth stating plainly, because the measurement only covers what the set contains:

- **No deontic marker, no detection.** "The policy was withdrawn in March" against a document
  still stating the policy is invisible to the pattern passes.
- **English-only.** The marker vocabulary is English; another language gets numeric
  detection only.
- **Retrieved-window bound.** The cross-check compares against the passage that matched, so a
  contradiction stated elsewhere in a long document is missed.
- **Validated on 21 cases.** Enough to catch a regression, not enough to characterise a real
  corpus. Growing that set is the cheapest accuracy work available.

## Later

- Admin web UI (pins, costs, prompt, connectors) — the API exists, the UI does not
- Slack channel adapter
- Per-document prompt caching for hot documents
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
