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
  loopback; `0.0.0.0` is explicit. Without sign-in it should stay inside
  the LAN or VPN.
- **Sign-in is the line everything else follows.** With `OK_AUTH_MODE=oidc`
  (see [ENTRA-SIGNIN.md](ENTRA-SIGNIN.md)) the server knows who is asking,
  and folder rules, roles and the admin log all become real. With sign-in
  off, network reachability *is* the access control — the same trust model
  as an internal wiki without SSO — and everything below about roles does
  not apply, because there is no identity to apply it to.
- **Admin actions** (settings, access rules, updates) require the admin
  token — `openknowledge token` on the server prints it — or membership of
  `OK_OIDC_ADMIN_GROUP`. A second group, `OK_OIDC_CURATOR_GROUP`, curates
  knowledge without holding governance.
- **Uploads** are a switch (`upload_enabled`): on, anyone signed in may
  contribute a *new* document. Deleting one, or uploading over an existing
  name, needs the curator or admin role — those take away something people
  were relying on, and cannot be undone from the app.
- **Every admin change is logged**, with who made it: `openknowledge
  admin-log`, or the *Admin log* panel on `/manage`. Changes made with the
  shared token name nobody, which is what sign-in fixes.

## What each person sees

The same app you have on your laptop: the sidebar with every document
grouped by folder — HR, Travel, whatever the tree says — answers with
sources and the tier that produced them, refusals when the documents do
not answer, contradictions refused until an admin resolves them in
`/manage`. Because the corpus is shared, the sidebar is the company's
single list — add a policy once, everyone can ask about it seconds
later.

## Backing it up

```
openknowledge backup --out /backups/openknowledge-$(date +%F).zip
```

One file. It carries what exists nowhere else — the pinned answers somebody
wrote by hand, the folder access rules somebody decided, the resolved
contradictions, and the ledger that knows what has been asked and what it cost
— plus the documents, unless you pass `--no-documents` because they already
live somewhere you back up separately.

The databases are copied through SQLite's own backup API, so this is safe to
run against a server that is still answering questions. Copying the files with
`cp` while it serves is not: you get something that looks like a database and
is not.

**Secrets are deliberately not in the archive.** A backup is a file that gets
emailed and dropped in shared storage, and one carrying an API key is a leak
waiting for somebody to be helpful with it. The backup prints the names of the
settings that were set, and the restore prints them again — those are typed
back in by hand.

The vector index is left out too: it is derived from the documents and rebuilt
by the first `openknowledge index`, so carrying it would double the file to
save a few minutes of CPU.

## Putting it back

```
openknowledge restore /backups/openknowledge-2026-09-01.zip
openknowledge index
```

Restore refuses to run over an install that already has databases — that is
the one irreversible thing here — unless you pass `--force`. It checks the
whole archive before it moves anything, so a truncated or foreign file is
refused with your install untouched. Once it starts moving files it goes one
at a time, so a machine that loses power halfway leaves a half-restored
directory; that is the reason for the refusal rather than a silent overwrite.

It will name the secrets that were set when the backup was taken. Questions
are refused until those are set again, so do that before telling anyone the
server is back.

## Roadmap for this shape (status: not code yet)

1. **Sign in with company credentials (Microsoft Entra ID).** Companies on
   Microsoft 365 already have Entra; OpenID Connect gives the server each
   person's identity and group memberships. Those groups feed the ACL
   machinery that already exists inside retrieval and the cache — the
   enforcement layer is built and tested; the login in front of it is not.
   The full design — flow, sessions, what CI can prove without a tenant
   and what cannot — is written down in [ENTRA-SIGNIN.md](ENTRA-SIGNIN.md),
   and the sign-in machinery itself now ships: the click-by-click tenant
   walkthrough for admins is [ENTRA-SETUP.md](ENTRA-SETUP.md). What no
   test of ours can reach — Microsoft's half of the flow against a real
   tenant — is the remaining verification step.
2. **Folders as categories, permissions per folder.** Both halves ship.
   Subfolders index, the sidebar groups documents under their folders,
   and uploads choose a destination folder (or create one) from the chat
   widget and from /manage. On top of that, admins rule folders in
   /manage's **Access** section: `HR/` readable by `group:<id>` means
   non-members get no HR answers, no HR titles in the corpus listing, no
   HR entries in the sidebar, and no writes into the folder - one rule,
   enforced everywhere the documents surface. The deepest rule wins for
   its subtree; an unruled folder stays open to everyone.
3. **Escalation on the company's own AI subscription.** Ships: when the
   local model cannot ground an answer, the ladder escalates to **Azure
   OpenAI in the company's tenant** — IT-approved, data stays in-tenant,
   per-token billing on the company's own agreement, graded by the same
   grounding gate and cached like every other answer. Setup is five
   settings and a deployment name: [AZURE-OPENAI.md](AZURE-OPENAI.md).
   Costs use the price *you* state from your own Azure price sheet;
   unstated, every call is flagged "cost not counted" rather than
   guessed. Honest note, still: a **Microsoft 365 Copilot seat is not a
   callable model API** — Microsoft does not offer "use my Copilot
   licence as a completion endpoint", so the corporate-sanctioned route
   is Azure OpenAI (same models, same tenant boundary). If Microsoft
   ever opens a Copilot inference API, the provider seam is ready for it.
