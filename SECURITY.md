# Security

## Reporting a vulnerability

Please report privately via GitHub's [security advisory][advisory] form rather than a public
issue. We aim to acknowledge within 72 hours.

[advisory]: https://github.com/FedericoTs/OpenKnowledge/security/advisories/new

## Threat model

OpenKnowledge runs inside your network and indexes documents you already own. The
consequential risks are about **access** and **accuracy**, not about the network perimeter.

### Access control

Two enforcement points, because one is not enough:

- **Retrieval** filters on `allowed_principals` during scoring, so a restricted user gets
  different results rather than silently fewer.
- **Cache reads** are re-checked. The cache is shared across users and its key deliberately
  excludes identity — per-user caches would destroy the hit rate the cost model depends on —
  so a cached answer is served only if the asker could have retrieved each cited source
  themselves. Unknown document ids fail closed.

**The weak link is the connector.** A connector that does not populate `allowed_principals`
from the source system's ACLs makes every indexed document visible to everyone who can reach
the bot. The bundled local-files connector applies one ACL to the whole folder; that is fine
for a uniformly-readable corpus and wrong for anything else.

Channels must pass the asker's real identity. `POST /chat` with `principals: null` means
unrestricted, which is only correct where every indexed document is visible to everyone.

### Admin surface

The admin API is **fail-closed**: with no `OK_ADMIN_TOKEN` set it returns 503 rather than
running unauthenticated. Tokens are compared in constant time. Pins are the highest-value
target in the system — someone who can write pins can define the canonical answer to any
question — so treat that token like a production credential and put the deployment behind
your normal internal auth.

### Data egress

Nothing leaves the machine unless `OK_ESCALATION_ENABLED=true` and an API key is set. When
escalation is on, the retrieved document excerpts and the question are sent to the configured
provider; the ledger records every such call. There is no telemetry and no
OpenKnowledge-operated service to phone home to.

### Prompt injection

A document in the corpus can contain text aimed at the model ("ignore previous instructions
and say X"). Current mitigations are partial and worth being honest about: the system prompt
instructs the model to answer only from sources, and the grounding gate limits the damage —
an injected answer still has to cite retrieved documents and use numbers found in them. It is
not a complete defence. Treat the corpus as trusted input, and be deliberate about who can
add documents to indexed folders.

### Not currently addressed

- Per-user rate limiting
- Encryption at rest for the SQLite store (use disk encryption)
- Audit log of admin actions beyond the pin author field
