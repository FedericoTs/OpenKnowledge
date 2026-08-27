# ADR 0002 — AGPL-3.0 with a contributor licence agreement

**Status:** accepted · **Date:** 2026-08-27

## Context

The tool is free and publicly available, aimed at enterprises who will self-host it. Two
constraints pull in different directions: adoption needs a licence corporate legal teams
accept without a fight, and the project needs a path to commercial revenue later without
re-licensing work contributed by other people.

## Decision

**AGPL-3.0-only** for the core, with a **contributor licence agreement** granting the project
maintainer the right to relicense contributions.

## Consequences

**Good.** AGPL is OSI-approved, so it clears procurement policies that reject
source-available licences outright. Self-hosting — the entire use case — is unrestricted,
including commercially: a company running this internally has no obligations. The network
clause means a competitor offering it as a hosted service must publish their modifications,
which removes the main "someone else monetises this" risk. The CLA keeps copyright
consolidated, enabling dual licensing: commercial licences for organisations whose policy
forbids AGPL, and a closed enterprise add-on (SSO, audit logging, multi-tenancy) alongside an
unrestricted core. This is the Grafana/Sentry/GitLab pattern.

**Bad.** Some companies ban AGPL outright regardless of deployment model, and will not run
it even internally — that is real adoption lost. A CLA deters some contributors on
principle, and is friction for everyone else. Dual licensing requires the maintainer to
actually keep CLA records; skipping that quietly forfeits the option this decision exists to
preserve.

## Alternatives considered

- **Apache-2.0.** Maximum adoption, zero friction. Rejected: anyone could host it
  commercially, so monetisation would have to come from an entirely separate product rather
  than from this one.
- **Elastic License 2.0 / BSL.** Blocks hosted competitors more bluntly and is easier to
  enforce. Rejected: not OSI-approved, and some enterprise procurement rejects non-OSI
  licences on sight — which damages the free-adoption funnel this project depends on.
