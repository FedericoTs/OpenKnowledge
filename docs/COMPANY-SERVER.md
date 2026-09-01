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

## PDFs, and how long the first index takes

Two PDF parsers ship, and which one you get depends on how you installed:

| | parser | per document | a thousand policy PDFs |
|---|---|---|---|
| **Docker** | OpenDataLoader (Java) | **21 ms** | about 25 seconds |
| **Windows installer** | pdfplumber (pure Python) | **60 ms** | about a minute |

The Docker image installs OpenDataLoader and a JVM, because it reads a
document's real structure — explicit heading levels, table rows and cells,
a page number on every element — rather than inferring it from type size
and ruling lines. The Windows installer bundles no Java, so it uses
pdfplumber, and always has.

The Java parser used to cost **656 ms** a document, because it started a
JVM for each one: about 640 ms of process start-up around 51 ms of actual
parsing. That made a thousand PDFs nine minutes on Docker, and it is why
the better parser was also the slow one. It is now handed 64 documents per
invocation instead of one, which makes it about **three times faster than
pdfplumber** rather than eleven times slower — better structure and less
waiting, with nothing to configure. The documents that come out are
asserted identical to parsing each one alone.

Every index after the first is faster still: a file whose bytes have not
changed is never re-read, and a corpus that has not changed at all starts
no Java process whatever. The parses live in `parses.db` beside the other
state, survive a restart, and are keyed on the content of each file rather
than its timestamp — so a restore-from-backup or an `rsync -t`, which put
old timestamps on new bytes, cannot make it serve you the previous version
of a policy.

It is pure derived data. Deleting `parses.db` costs one rebuild and nothing
else, which is why it is a separate file and why it is not in the backup:
the backup already carries the documents themselves.

A PDF the parser cannot read is **named** rather than skipped in silence,
with the reason whichever parser you have gave — `handbook.pdf:
OpenDataLoader: this file is not a valid PDF file (corrupted or truncated
content).` One unreadable file never costs you the rest of the batch.

## Changing who may read what

Rules are set per folder on `/manage`, and the deepest rule wins for its
subtree. A change is in force the moment the response returns — there is no
window in which a rule is stored and the index is still answering to the
old audience — and it costs milliseconds rather than a rebuild, because a
rule decides a document's audience and nothing else about it.

Cached answers are deliberately not thrown away when a rule changes: the
cache keys on the corpus, not on who is asking, and a cached answer is only
served to somebody who could have retrieved each of its sources themselves.
Access is enforced when the answer is read, not when it was written.

## Keeping one caller from spending everybody's day

The budget governor already stops a flood becoming an invoice: the ceiling
it computes is *remaining budget ÷ questions still expected*, so a thousand
questions lower what any one question may cost rather than running the bill
up. What it cannot do is decide **whose** questions those were — so a
looping bot integration, or a colleague who found they can paste a
spreadsheet into the chat, drags that shared ceiling down for everyone.

```sh
OK_ASKER_QUESTIONS_PER_MINUTE=30   # 0 (the default) is off
```

Over the limit, that caller gets a `429` and a sentence saying so; everyone
else is unaffected. It is a live setting — change it on `/manage` while a
caller is looping and it applies to their next question, no restart.

The counters live in memory, are keyed by a salted hash of the asker rather
than by the asker, and are gone when the process restarts. Enforcing a limit
needs to know that *this* caller has asked twelve times in the last minute;
it never needs to know who they are.

Two things to know before setting it. With sign-in **on**, the bucket is the
person. With sign-in **off** it is the address the request came from — which
on a desktop install is the one person using it, and behind a reverse proxy
is *everybody*, so a proxied deployment should turn sign-in on rather than
rely on this to tell its people apart. And the default is off, because a
desktop install should not meet a limit it never asked for.

The counting is per process. Two servers behind a load balancer each
enforce their own, so the effective limit is doubled — set it accordingly,
or keep the deployment to one server, which is the shape this guide
describes.

## Watching it

```sh
curl -H "Authorization: Bearer $(openknowledge token)" http://server:8080/metrics
```

Prometheus text exposition: build version, documents and passages indexed,
questions and spend per tier for both today and all time, questions refused
by the rate limiter, open contradictions, open wrong-answer reports. Nothing
in it is new data — it is the ledger, the index and the limiter, formatted so
a graph can be drawn without anybody writing a parser.

It is admin-only, unlike `/healthz`: spend and volume are not everybody's
business, and a scraper carries the admin token as easily as any other
header. It carries no question text and no identity — a metric with the
question in it is a log of what people asked, published to whatever scrapes
it.

## When it gets one wrong

A refusal is easy to learn from: it is counted, ranked and reported
(`openknowledge gaps`). A *wrong* answer is the hard one, because it looks
exactly like a right one — so it used to reach a colleague and never the
documents.

The answer card carries a **This is wrong** button. A reader clicks it,
types one sentence — "the figure changed in April" is the whole fix — and
it lands in *Answered wrong* on `/manage`, ranked by how many people said
so, with the answer they were shown and the notes they left. From there
one box pins the right answer and closes the report.

```sh
openknowledge reports          # the same list, on the server
```

Two things it will not do. It records **nothing about who reported it** —
the same rule the gaps report follows, and checked against the bytes of
the database, not just the API: what is useful is which answer is wrong
and why, and a knowledge base that reports what its people got wrong
should not also be a record of who complained. And it only accepts
reports for questions this install actually answered, so the list holds
real answers rather than whatever anybody posts.

A report raised before a re-index is marked **stale**: the documents
changed underneath it, so it may already be fixed. It is flagged rather
than deleted, because "we fixed that" is a claim somebody should be able
to check.

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
