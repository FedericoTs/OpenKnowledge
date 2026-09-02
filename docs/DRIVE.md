# Google Drive

OpenKnowledge can mirror Google Drive shared drives into its documents folder
and answer from them, with each file readable in the chat by the people Drive
lets read it. Built against Google's documented API shapes and a loopback fake
of them (`tests/fake_drive.py`); **not yet run against a real Workspace** — see
the last section.

## The part to understand before turning it on

**Drive names people by email; a directory names them by id.** SharePoint hands
back the same Entra object ids sign-in already puts in a session, so the two
vocabularies meet by themselves. Drive hands back `alice@contoso.com` and
`hr-team@contoso.com`.

So this maps a user grant to `user:<email>` and a group grant to
`group:<group-email>`, and sign-in now mints `user:<verified email>` alongside
`user:<subject>`. That is what lets a person match their own files. It is minted
only from an email the identity provider vouches for: an address the person
typed would let anyone read someone else's documents by claiming it.

Two consequences worth knowing before you rely on it:

- **A file granted to a person by address works out of the box**, as does a
  file shared with your whole Workspace domain (it becomes "anyone signed in").
- **A file granted to a Google group works only if your sign-in emits that
  group's email** as a group claim. Entra emits group ids, not Google group
  emails; Google's own OIDC tokens carry no groups at all. Where the emails are
  not emitted, those files are simply invisible — which is the direction this
  fails in, deliberately. Grant by domain or by person where that matters.

Anything else — a domain that is not yours, a grant shape Google has not
documented — is dropped, never widened. A file left with no reader this can
express is stamped as withheld: indexed, counted on `/manage`, shown to nobody.

## Setting it up

1. **A service account** in Google Cloud with the Drive API enabled. Download
   its JSON key; you need `client_email` and `private_key` from it.
2. **Domain-wide delegation** (recommended): in the Workspace admin console,
   authorise the service account's client id for the scope
   `https://www.googleapis.com/auth/drive.readonly`, and set
   `OK_DRIVE_SUBJECT` to a person whose view of the shared drives is the one
   to mirror. Without delegation the account sees only what has been shared
   with its own address, which is a workable but fiddlier setup.
3. **Settings on the server:**

   | setting | value |
   |---|---|
   | `OK_DRIVE_ENABLED` | `true` |
   | `OK_DRIVE_CLIENT_EMAIL` | the service account's address |
   | `OK_DRIVE_PRIVATE_KEY` | its PEM private key (keep the `\n` escapes intact) |
   | `OK_DRIVE_SUBJECT` | the person to impersonate, with delegation |
   | `OK_DRIVE_DOMAIN` | your Workspace domain, e.g. `contoso.com` |
   | `OK_DRIVE_IDS` | shared drive ids to mirror; empty means all it can see |
   | `OK_DRIVE_POLL_SECONDS` | how often to ask what changed (default 300; 0 turns the timer off) |
   | `OK_DRIVE_PERMISSIONS_REFRESH_SECONDS` | how long readers are trusted (default 3600) |

4. Run `openknowledge drive sync` once from the server to watch the first
   mirror and read its summary, then `openknowledge drive status` any time.
   The server syncs on the timer; *Documents* on `/manage` shows when the last
   sync ran, how many files are mirrored and how many are withheld; an
   administrator can also ask for a sync now (`POST /admin/drive/sync`).

## What to expect, and its bounds

- **Google-native files are exported**: a Doc as `.docx`, a Sheet as `.xlsx`,
  a Slide deck as `.pptx`, which is what gives the parsers real headings,
  tables and locators. A Form, a Drawing or a shortcut holds no document and
  is skipped.
- **A revoked grant takes up to `OK_DRIVE_PERMISSIONS_REFRESH_SECONDS` to
  bite.** The changes feed does not report permission changes on their own, so
  readers are re-asked on a clock; that interval is the bound.
- **The mirror is the sync's to change.** Uploading into it or deleting from
  it through the app is refused with a pointer to Drive, because the next sync
  would undo it.
- **My Drive is not mirrored**, only shared drives. A company's documents that
  live in one person's My Drive are one resignation from being unreachable,
  and that is a filing problem this tool should not paper over.

## Honesty about what this is

Every call and mapping here follows Google's documentation and is proved
against `tests/fake_drive.py`, which verifies the signed assertion the service
account presents, serves the shared drive list, the start page token, the first
walk, the changes feed with paging and removals, permissions, downloads and
exports. It has **not** been run against a real Workspace, because none was
available where this was built. The first real run is what it owes: which grant
shapes actually appear, how many the mapping cannot express, and whether Drive
throttles a large shared drive harder than the backoff here expects. The
numbers to keep from that run are the sync summary's `withheld` and
`unmapped_grants`.
