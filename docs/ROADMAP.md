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

## Next — makes it useful in a real company

1. **Semantic cache (tier L2).** Local embeddings over canonical questions, similarity
   threshold, and a citation check before serving a near-match. This is where the free share
   grows, and it needs to be careful: a hash cannot decide two sentences mean the same thing,
   so a bad match must be catchable.
2. **Hybrid retrieval + reranking.** BM25 fused with local dense retrieval, then a
   cross-encoder rerank. The biggest single cost lever is sending fewer, better chunks — this
   is cheaper *and* more accurate, which is a rare combination.
3. **SharePoint connector.** Microsoft Graph enumeration, text extraction, and — the real
   work — mapping item permissions, including group expansion and inheritance, onto
   `allowed_principals`.
4. **Google Drive connector.** Same shape: service account with domain-wide delegation,
   `files.list`, and permission mapping including inherited folder ACLs.
5. **Teams channel.** Bot Framework adapter, with the asker's tenant groups supplying
   `principals` so access control works from the identity Teams already has.
6. **Evaluation harness.** A golden set of question/answer/citation triples, run on every
   change, reporting accuracy *and* blended cost. Without this, "the local model is good
   enough" is an opinion. This should arguably outrank items 3–5.

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
