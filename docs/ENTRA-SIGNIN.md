# Sign in with company credentials — the design

The company server today trusts the network: whoever can reach the port
can read, and `/chat` accepts the asker's principals **from the request
body** — fine when the caller is a trusted bot backend, useless as access
control for browsers. This document is the plan for closing that gap with
OpenID Connect sign-in, Microsoft Entra ID first.

It is a design first and a status report second. **Built and tested:
increments 1 through 5** of the build order at the end - the OIDC client
and session store (`src/openknowledge/auth/`), the gate and
server-minted principals, the sign-in UI in the widget and /manage,
admin-by-group, and the TLS settings - proven against the fake IdP
described below (`tests/test_auth.py`, `tests/test_auth_api.py`), in a
real browser riding the whole redirect chain
(`tests/test_auth_browser.py`), and including the end-to-end test where
an HR-restricted answer serves an HR member and refuses everyone else.
The admin's click-by-click walkthrough is [ENTRA-SETUP.md](ENTRA-SETUP.md).
What remains is increment 6 alone: verifying Microsoft's half of the
flow against a real tenant, which only a human with one can do.

## The lock already exists

Enforcement is not the work. Every path that can serve an answer already
checks visibility, and the checks are tested:

- Retrieval filters chunks by principals in both halves of the hybrid
  search (`retrieval/hybrid.py`, `retrieval/bm25.py`).
- Every serving tier re-checks before answering from stored state:
  pinned answers, the exact cache, drafts and the semantic cache all
  refuse when any cited document is not visible to the asker
  (`cascade/router.py`, the `visible_to` calls).
- "What documents do you have?" lists only what the asker may see
  (`documents_visible_to`).
- A document with an **empty ACL is visible to everyone** — which is why
  today's corpora keep working unchanged when sign-in arrives.

Because cache hits re-check visibility, the shared cache stays safe by
construction: two people asking the same question share one answer only
when both may read every document it cites.

What is missing is the badge reader: turning *who is asking* into
`principals` on the server, instead of believing whatever the request
asserts.

## Decisions

**Generic OIDC, Entra documented first.** The flow is standard
authorization-code + PKCE, endpoints read from the issuer's discovery
document. Entra is the documented, first-class path; Keycloak or any
other OIDC provider works by pointing the same three settings elsewhere.
This is also what makes the feature testable: CI runs the full flow
against a small fake IdP signing with a test key — no tenant required.

**Small, auditable client — not a framework.** PyJWT (with its JWKS
client) plus the httpx we already ship: validate signature, issuer,
audience, expiry and nonce; reject everything else. MSAL becomes worth
its weight when we need Graph or on-behalf-of flows; a login redirect
does not.

**Sessions live server-side, in the state SQLite.** Entra group claims
are lists of GUIDs that do not fit in a cookie. The browser holds an
opaque, HttpOnly, SameSite=Lax session id; the session row holds the
user id, display name, groups and expiry. Sign-out deletes the row.

**Principals are minted by the server, never accepted from the wire.**
A session becomes `user:{oid}`, one `group:{gid}` per group, and
`authenticated`. When sign-in is on, a request body that carries its own
`principals` is refused with 400 — an escalation attempt should fail
loudly, not be silently ignored. With sign-in off, today's
trusted-caller behaviour is unchanged.

**Admins become a group.** `OK_OIDC_ADMIN_GROUP=<group-id>` grants the
admin surface to that group's members — organising folders "directly by
admins" stops needing a shared token. The token keeps working for
automation and for deployments without sign-in.

**Honesty about group claims.** Entra puts groups in the ID token only
when the app registration asks for them, and silently switches to an
"overage" pointer when a user is in more than ~200 groups. The setup
guide will say: configure the app to emit **groups assigned to the
application** (the clean, bounded option). If a token arrives with an
overage marker anyway, sign-in fails with a message naming the fix —
resolving overage via Graph is roadmap, not silently-wrong.

**Honesty about TLS.** Entra refuses `http://` redirect URIs except on
`localhost`. Testing on the server machine itself works plain; serving a
LAN under sign-in requires HTTPS — either the reverse proxy IT already
runs, or `OK_TLS_CERT`/`OK_TLS_KEY` handed straight to uvicorn. The
session cookie is marked Secure exactly when the public URL is https.

## Configuration

```sh
OK_AUTH_MODE=oidc                       # default: off (today's behaviour)
OK_OIDC_ISSUER=https://login.microsoftonline.com/<tenant-id>/v2.0
OK_OIDC_CLIENT_ID=<app registration id>
OK_OIDC_CLIENT_SECRET=<client secret>
OK_PUBLIC_URL=https://knowledge.example.com   # builds the redirect URI
OK_OIDC_ADMIN_GROUP=<group object id>   # optional: this group is admin
OK_OIDC_CURATOR_GROUP=<group object id> # optional: this group curates only
OK_OIDC_GROUPS_CLAIM=groups             # default
OK_SESSION_HOURS=8                      # default
```

With `OK_AUTH_MODE=off` nothing changes anywhere — the desktop app and
personal servers never see any of this.

## What is gated

| Route | Signed out |
|---|---|
| `/auth/login`, `/auth/callback`, `/auth/logout` | open (they are the door) |
| `/healthz` | open — monitoring needs no identity |
| everything else — widget, `/chat`, streaming, documents, `/manage`, `/setup` | 302 to sign-in (pages) / 401 (API) |
| admin API | admin group in session, or the admin token as today |

### Two roles, because they are two jobs

The people who know the answers are rarely the people who should hold the
access rules. `OK_OIDC_CURATOR_GROUP` splits them:

| | admin | curator | everyone else |
|---|---|---|---|
| pin an answer, approve a draft, resolve a contradiction | yes | yes | no |
| re-index, run `learn`, read costs and gaps | yes | yes | no |
| delete or replace a document | yes | yes | no |
| upload a *new* document | yes | yes | if uploads are on |
| change who may read a folder | yes | no | no |
| change settings, apply an update | yes | no | no |
| read the admin log | yes | no | no |

Every admin is a curator. Leave `OK_OIDC_CURATOR_GROUP` empty and the
curator surface *is* the admin surface — an install that never sets one
behaves exactly as it did before the split existed.

Two things are worth being precise about. **Deleting is not the mirror of
uploading**: an upload adds something the corpus did not have, a delete —
or an upload over an existing name — takes away something people were
relying on, and cannot be undone from the app. So with sign-in on, both
need a role, while contributing a new document stays governed by the
uploads switch alone. And **with sign-in off, none of this applies**:
there is no identity to check against, reaching the port was always full
control, and a role check against an identity that does not exist would be
theatre. The desktop app is unchanged.

### The admin log

Every admin and curator change is recorded: what, when, and who — by their
directory subject id, so the entry survives a rename and points at an
account someone can disable. Read it at `GET /admin/log`, in the *Admin
log* panel on `/manage`, or on the server with:

```sh
openknowledge admin-log --days 30
```

Changes made with the shared admin token are recorded as
`shared admin token` and name nobody. That is not a gap the log can close
— a shared secret never carried an identity — and it is the plainest
argument for turning sign-in on. Entries record *which* settings changed
and never what they were set to, because a base URL can carry a password
and the log travels in every backup.

The log is append-only through the app: nothing in the API deletes from
it, so an admin cannot quietly undo their own change. That is a property
of the surface, not of the file — anyone with the server's disk can edit
the table like any other. It is the threat model a company server has,
stated rather than overclaimed.

The widget's only change: show who is signed in, offer sign-out, and
treat a 401 as "go sign in". The marketing site keeps its own
`OK_WEBSITE_ENABLED` switch; a public site and a signed-in app on one
process is the operator's decision, not a default.

## What proves it, without a tenant

The fake IdP is the heart of the test suite: a loopback server with a
discovery document, a JWKS endpoint holding a test RSA key, and a token
endpoint that signs whatever identity a test asks for. Against it:

- the full round trip — redirect, state echoed, code exchanged, session
  minted, widget loads signed-in;
- the end-to-end ACL truth: index a document restricted to `group:hr`,
  sign in a user without that group, and the answer, the cache hit and
  the corpus listing must all deny it — sign in an HR member and all
  three serve;
- the refusals: wrong issuer, wrong audience, expired token, replayed
  nonce, tampered state, an overage marker, body-supplied `principals`
  while sign-in is on;
- session lifecycle: expiry, sign-out, cookie flags;
- a Playwright drive of the whole flow in a real browser.

## What needs the real tenant

Only a human with a tenant can verify Microsoft's half: the app
registration screens, consent, real group claims, clock skew against
real tokens. That is the recorded verification step, and it needs from
the company: a tenant id, an app registration
([ENTRA-SETUP.md](ENTRA-SETUP.md) lists the exact clicks), one test
group with two test users, and the hostname the server will live at.
Every surprise the live tenant produces becomes a pinned test, as usual.

## Build order

1. `auth/oidc.py` + `auth/sessions.py`: discovery, JWKS validation,
   code flow, session store in the state DB — with the fake-IdP rig and
   unit tests in the same commit.
2. Settings + route gating + server-minted principals (and the 400 on
   asserted ones). The ACL end-to-end test lands here.
3. Widget and /manage: signed-in chip, sign-out, 401 handling.
4. Admin-by-group beside the token.
5. TLS settings, `docs/ENTRA-SETUP.md` (the clicks), and the live drive.
6. Real-tenant verification with the company's values; field findings
   become tests.

Per-folder ACLs - the feature this unblocks - now ship as well: admins
rule folders in /manage's Access section, and the rule holds everywhere
the documents surface. See [COMPANY-SERVER.md](COMPANY-SERVER.md).
