# One knowledge base, every laptop

The company shape of OpenKnowledge: **one server holds the documents, the
index and the models; everyone else uses a browser.** Nothing to install on
the laptops, one place to add documents, one answer for everyone — asked
twice by two people, the second person gets the first person's answer,
byte for byte, from cache.

This works today with what ships. The pieces that are roadmap, not code —
signing in with company credentials, folder-level permissions, escalation
to the company's own Azure OpenAI tenant — are listed at the end with
their honest status.

---

## The topology

```
                 ┌─────────────────────────────────────┐
   laptops ────► │  the OpenKnowledge server           │
   (browser,     │  · documents folder  (the source)   │
    no install)  │  · index + caches    (SQLite)       │
                 │  · llama-server      (the models)   │
                 └─────────────────────────────────────┘
```

- Every laptop opens `http://<server>:8080` — the full app: chat with
  sources, drag-and-drop uploads (if enabled), the manage page for admins.
- The **Windows installer is for the server or for personal offline use** —
  it is how one machine becomes the knowledge base, not something to roll
  out per laptop.
- Cost scales with questions, not with seats: the cache is shared, so the
  second person asking a common question costs nothing at all.

## Setting it up

**Option A — Docker (a Linux box or VM):**

```sh
git clone https://github.com/FedericoTs/OpenKnowledge && cd OpenKnowledge
docker compose up -d --build
```

Put documents in the mounted `documents/` folder (subfolders are fine —
they are indexed recursively), run `openknowledge index` in the container
(or upload through the browser), done.

**Option B — a Windows machine as the server:**

Install the desktop app on the machine that will be the server, let first
run fetch the models, then serve the network instead of loopback: set
`OK_BIND_HOST=0.0.0.0` in `%LOCALAPPDATA%\OpenKnowledge\.env` (or run
`openknowledge serve --host 0.0.0.0` from the installed CLI) and open port
8080 in Windows Firewall for your LAN.

## The security posture, plainly

- **Serving the network is a decision, not a default.** The default bind is
  loopback; `0.0.0.0` is explicit, and should stay inside the LAN or VPN.
  Do not port-forward this to the internet — there is no login yet.
- **Admin actions** (settings, review queue, conflicts, pins) already
  require the admin token — `openknowledge token` on the server prints it;
  give it to admins only.
- **Uploads** are a switch (`upload_enabled`): on, anyone on the LAN can
  add documents through the browser; off, documents come only from the
  server's folder. Pick deliberately.
- **Reading is currently open to whoever can reach the port.** Until
  sign-in lands (below), network reachability *is* the access control —
  the same trust model as an internal wiki without SSO. If that is not
  acceptable, keep it VPN-only.

## What each person sees

The same app you have on your laptop: the sidebar with every document
grouped by folder — HR, Travel, whatever the tree says — answers with
sources and the tier that produced them, refusals when the documents do
not answer, contradictions refused until an admin resolves them in
`/manage`. Because the corpus is shared, the sidebar is the company's
single list — add a policy once, everyone can ask about it seconds
later.

## Roadmap for this shape (status: not code yet)

1. **Sign in with company credentials (Microsoft Entra ID).** Companies on
   Microsoft 365 already have Entra; OpenID Connect gives the server each
   person's identity and group memberships. Those groups feed the ACL
   machinery that already exists inside retrieval and the cache — the
   enforcement layer is built and tested; the login in front of it is not.
   The full design — flow, sessions, what CI can prove without a tenant
   and what cannot — is written down in [ENTRA-SIGNIN.md](ENTRA-SIGNIN.md).
2. **Folders as categories, permissions per folder.** The categories half
   ships today: subfolders index, the sidebar groups documents under
   their folders, and uploads choose a destination folder (or create
   one) from the chat widget and from /manage. What remains is the
   permissions half: ACL boundaries - HR/ readable by the HR group,
   mapped from Entra groups - which needs sign-in (above) first.
3. **Escalation on the company's own AI subscription.** When the local
   model cannot answer, the ladder can escalate to **Azure OpenAI in the
   company's tenant** — IT-approved, data stays in-tenant, per-token
   billing, and it slots into the existing escalation ladder as
   configuration. Honest note: a **Microsoft 365 Copilot seat is not a
   callable model API** — Microsoft does not offer "use my Copilot licence
   as a completion endpoint", so the corporate-sanctioned route is Azure
   OpenAI (same models, same tenant boundary). If Microsoft ever opens a
   Copilot inference API, the provider seam is ready for it.
