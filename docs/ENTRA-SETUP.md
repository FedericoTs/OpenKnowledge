# Connecting Microsoft Entra ID, click by click

What a company admin does once so that everyone signs into OpenKnowledge
with their normal Microsoft account. Fifteen minutes, no code. The design
behind it — and what is enforced once sign-in is on — is in
[ENTRA-SIGNIN.md](ENTRA-SIGNIN.md).

You need: rights to create an app registration in your tenant (or someone
who has them), and the URL your server will live at.

## No company yet? Test on a personal tenant

You do not need a registered company to test any of this. An individual
Azure account (sign up at azure.microsoft.com/free with any Microsoft
account — "individual" is a normal choice at signup; the card is for
identity verification) comes with its own **Microsoft Entra ID tenant**,
and the free tier of Entra covers everything OpenKnowledge uses: app
registrations, users, security groups, the groups claim. Create test
users (`alice@<yourtenant>.onmicrosoft.com`, Users → New user), put them
in test groups, and follow the steps below unchanged — the tokens, the
claims and the consent screens are the same machinery a company tenant
produces, which is exactly why a personal tenant is a valid dress
rehearsal.

Two personal-tenant papercuts worth knowing in advance: new tenants ship
with **security defaults** on, so a test user's first sign-in asks them
to register the Authenticator app — register it (or turn security
defaults off under Entra ID → Overview → Properties, a reasonable call
for a throwaway test tenant, not for production). And after registering
the app, **API permissions → Grant admin consent** spares every test
user an individual consent prompt.

---

## 1. Register the application

In the [Entra admin center](https://entra.microsoft.com) →
**App registrations** → **New registration**:

- **Name:** `OpenKnowledge`
- **Supported account types:** *Accounts in this organizational directory
  only* (single tenant)
- **Redirect URI:** platform **Web**, value
  `https://<your-server>/auth/callback`
  — for a first test on the server machine itself,
  `http://localhost:8080/auth/callback` also works: localhost is the one
  place Entra accepts plain `http://`.

Register, then from the **Overview** page copy two values:

- **Application (client) ID** → this is `OK_OIDC_CLIENT_ID`
- **Directory (tenant) ID** → goes into the issuer URL below

## 2. Create the client secret

**Certificates & secrets** → **New client secret** → pick an expiry your
policy allows → **copy the Value immediately** (it is shown once). This is
`OK_OIDC_CLIENT_SECRET`. Put a reminder wherever you track renewals: when
it expires, sign-in stops with a code-exchange error until you mint a new
one.

## 3. Put groups in the token

**Token configuration** → **Add groups claim** → select **Groups assigned
to the application** → under ID, emit as **Group ID** → save.

"Assigned to the application" is deliberate — not "all security groups".
It keeps the token bounded (Entra silently switches to an overage pointer
past ~200 groups, which OpenKnowledge refuses rather than half-trusts),
and it makes intent explicit: the groups that matter here are the ones you
assign.

Then assign them: **Enterprise applications** → **OpenKnowledge** →
**Users and groups** → **Add user/group** — add each group that should be
able to use or administer the knowledge base. A group's **Object ID**
(visible on the group's page in Entra) is what OpenKnowledge sees.

## 4. Configure the server

In the server's `.env` (`%LOCALAPPDATA%\OpenKnowledge\.env` for a Windows
install; the checkout's `.env` otherwise):

```sh
OK_AUTH_MODE=oidc
OK_OIDC_ISSUER=https://login.microsoftonline.com/<tenant-id>/v2.0
OK_OIDC_CLIENT_ID=<application (client) id>
OK_OIDC_CLIENT_SECRET=<the secret value>
OK_PUBLIC_URL=https://<your-server>
OK_OIDC_ADMIN_GROUP=<object id of your admins group>   # optional
```

Two details that bite:

- **The issuer must use the tenant ID (the GUID), not your domain name.**
  OpenKnowledge checks the discovery document's issuer against this value
  and refuses a mismatch, with an error that says exactly this.
- **`OK_OIDC_ADMIN_GROUP` members get `/manage` with no token.** Everyone
  else who needs admin automation keeps using `OK_ADMIN_TOKEN`. Grant and
  revoke admins in the directory, like everything else about them.

Restart the server. From now on every browser is redirected to Microsoft,
signs in with the company account, and comes back known: their groups are
the `principals` the retrieval and cache ACLs enforce.

## 5. HTTPS, because Entra insists

Entra refuses `http://` redirect URIs anywhere except localhost, so a
server on the LAN needs TLS. Either terminate it in front (the reverse
proxy IT already runs — with [Caddy](https://caddyserver.com), the entire
config is `knowledge.example.com { reverse_proxy 127.0.0.1:8080 }`), or
hand the certificate straight to the server:

```sh
OK_TLS_CERT=/etc/ssl/knowledge.crt
OK_TLS_KEY=/etc/ssl/knowledge.key
```

Both or neither — half a pair refuses to start rather than serving
insecurely. The session cookie is marked `Secure` exactly when
`OK_PUBLIC_URL` is https.

## 6. Give folders to groups

Once sign-in works, folder permissions are an admin screen, not a config
file: open **/manage → Access**, and next to each folder enter who may
read it — `group:<object-id>` (the group's Object ID from Entra),
`user:<object-id>`, or `authenticated` for anyone signed in. Save
re-indexes immediately. One rule holds everywhere at once: answers, the
shared cache, the corpus listing, the chat sidebar, uploads and deletes.
The deepest rule wins for its subtree, and a folder without a rule stays
open to everyone.

---

## When it does not work

| You see | It means | Fix |
|---|---|---|
| `AADSTS50011` (redirect URI mismatch) | the registered URI differs from `OK_PUBLIC_URL` + `/auth/callback` | make them byte-identical, scheme included |
| "names issuer ... not the configured" | issuer uses your domain, or the wrong tenant | use the tenant **ID** form from step 4 |
| "too many groups for the token to list them" | groups claim is set to all groups and the account is in ~200+ | step 3: emit **assigned** groups only |
| Signed in, but no group-restricted answers | groups never reached the token | step 3 both halves: add the claim **and** assign the groups |
| "the identity provider refused the code exchange" | expired or wrong client secret | mint a new secret, update `.env`, restart |
| Sign-in works on the server, not from laptops | redirect URI is the localhost one | register the real `https://` URI and set `OK_PUBLIC_URL` |

Every sign-in failure OpenKnowledge itself detects is shown on the error
page in plain words — the table above is for the failures Microsoft
reports in its own vocabulary.
