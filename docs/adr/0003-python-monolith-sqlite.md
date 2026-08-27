# ADR 0003 — One Python service, SQLite by default

**Status:** accepted · **Date:** 2026-08-27

## Context

The unit of delivery is a package an IT department installs on a spare VM. The people
deploying it are not necessarily the people who wrote it, and often not developers at all.
Meanwhile the domain — retrieval, embeddings, rerankers, document parsing, local inference —
is overwhelmingly Python.

## Decision

A single Python service (FastAPI), with SQLite as the default datastore and a plain browser
page for the chat widget. No separate frontend build, no message broker, no external database
required to start.

## Consequences

**Good.** `docker compose up` is the whole installation. The retrieval ecosystem is native.
One process means one place to look when something breaks, which matters when the operator
did not write the code. SQLite persists the cache, pins, and ledger across restarts with no
configuration — and a cold cache is precisely when this tool is at its most expensive, so
"persistence by default" is a cost feature as much as a convenience one.

**Bad.** SQLite means one writer, so horizontal scaling needs the Postgres backend on the
roadmap. A single process couples the indexing path to the serving path: a large re-index
competes with live traffic. Python's concurrency story puts a ceiling on throughput that a
Go or Rust service would not have.

**Mitigation.** Storage access is confined to `cache/store.py`, so the Postgres swap is one
module rather than a refactor. Nothing outside it knows the database is SQLite.

## Alternatives considered

- **Python core + TypeScript admin/marketing app.** Better admin UI ergonomics. Rejected for
  v0: two toolchains, two containers, and a build step, for a UI that is currently four
  endpoints.
- **Postgres + pgvector from the start.** Where this ends up, and required for multi-instance
  deployments. Rejected as the *default* because it makes the first-run experience a database
  setup, and pgvector is not needed until hybrid retrieval exists.
