# SharePoint

OpenKnowledge can mirror the document libraries of one SharePoint site into its
documents folder and answer from them, with each file readable in the chat by
exactly the people SharePoint lets read it. Built against Microsoft Graph's
documented shapes and a loopback fake of them (`tests/fake_graph.py`); **not yet
run against a real tenant** — see the last section.

## What it does

- Reads each library through Graph's **delta** feed: the first sync walks the
  library, every later one asks only what changed and downloads only that.
- Mirrors files into `<documents>/sharepoint/<library>/…` as ordinary files,
  where the same parsers, parse cache and re-index handle them like anything
  dropped into the folder. Unsupported types (images, say) are skipped and
  counted.
- Asks Graph for each file's **permissions** when the file is new or changed,
  and again on a clock for unchanged files, and stamps the readers onto the
  document in the vocabulary sign-in already uses: Entra users become
  `user:<object-id>`, Entra groups `group:<object-id>`, a sharing link scoped to
  the organisation `authenticated`.
- **Fails closed.** A grant it cannot express — a SharePoint site group, a
  device, a shape Graph has not documented — is dropped, never widened into
  "everyone". A file left with no mappable reader is stamped with a principal
  nobody holds: indexed, listed under *Documents* with the count of such files,
  and shown to no one but an administrator. Modern team sites back their
  Members group with an Entra group, which maps; classic site groups do not.

The mirror is the sync's to change: uploads into it and deletions from it are
refused in the app with a pointer to SharePoint, where the change belongs.

## Setting it up

1. **An app registration** in Entra ID — the one from [ENTRA-SETUP.md](ENTRA-SETUP.md)
   can be reused — with an **application** (not delegated) permission on
   Microsoft Graph. Least privilege is `Sites.Selected`, after which an
   administrator grants the app read access to the one site
   (`POST /sites/{site-id}/permissions` with the app's id and the `read`
   role); the broader `Sites.Read.All` with admin consent reads every site and
   needs no per-site grant. A client secret for the app.
2. **Sign-in on** (`OK_AUTH_MODE=oidc`). With sign-in off no reader can be
   enforced and every mirrored file would be readable by whoever reaches the
   widget, so the sync refuses to run until `OK_SHAREPOINT_REQUIRE_SIGNIN=false`
   says that is intended.
3. **Settings:**

   | setting | value |
   |---|---|
   | `OK_SHAREPOINT_ENABLED` | `true` |
   | `OK_SHAREPOINT_TENANT_ID` | the directory (tenant) id |
   | `OK_SHAREPOINT_CLIENT_ID` | the app registration's client id |
   | `OK_SHAREPOINT_CLIENT_SECRET` | its client secret |
   | `OK_SHAREPOINT_SITE` | the site, path-addressed: `contoso.sharepoint.com:/sites/HR` |
   | `OK_SHAREPOINT_DRIVES` | library names to mirror, comma-separated; empty means all |
   | `OK_SHAREPOINT_POLL_SECONDS` | how often to ask what changed (default 300; 0 turns the timer off) |
   | `OK_SHAREPOINT_PERMISSIONS_REFRESH_SECONDS` | how long readers are trusted before being re-asked (default 3600) |

4. Run `openknowledge sharepoint sync` once from the server to see the first
   mirror happen and read its summary, then `openknowledge sharepoint status`
   any time. The server syncs on the timer; *Documents* on `/manage` says when
   the last sync ran, how many files are mirrored and how many are withheld;
   an administrator can also ask for a sync now (`POST /admin/sharepoint/sync`).

## What to expect, and its bounds

- A new or changed file appears within one poll interval.
- A **revoked** grant is honoured within the permissions refresh interval:
  Graph's delta feed does not report permission changes on their own, so
  readers are re-asked on a clock. Shorten the interval if that bound is too
  loose for your documents; each refresh is one Graph call per file.
- A deleted file leaves the mirror and the index at the next sync.
- If Graph says a delta link has expired (410), the library is re-read from
  the start; the mirror ends up identical.
- Throttling (429, 503) is waited out per `Retry-After`; an expired token is
  refreshed once; anything else is recorded as the sync's last error and shown
  on `/manage`.

## Honesty about what this is

Every call, payload and edge case here was written against Microsoft's
documentation and proved against `tests/fake_graph.py`, which answers like
Graph does for sites, drives, delta paging and tokens, content redirects and
permissions. It has **not** been run against a real tenant, because none was
available where this was built. The first real run is the measurement this
connector still owes: what a tenant's permission shapes actually look like in
practice, how many grants the mapping cannot express, and how Graph throttles
a library of a few thousand files. When you run it, the numbers to keep are the
sync summary's `withheld` and `unmapped_grants`, and the log's first errors.
