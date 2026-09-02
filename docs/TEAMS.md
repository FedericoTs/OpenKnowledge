# Teams

OpenKnowledge can answer in Microsoft Teams as a bot: ask about a policy in a
chat or a channel, get the answer with the passage it came from, and get it
from exactly the documents you are allowed to open. Built against Microsoft's
documented Bot Framework and Graph shapes and a loopback fake of them
(`tests/fake_botframework.py`); **not yet run against a real tenant** — see the
last section.

## What it does, and what it refuses

A bot endpoint is a public URL. Anybody who finds `/teams/messages` can post to
it, so nothing in a request is believed until it has been proved:

- **The token.** Every inbound activity carries a JWT the Bot Service signed.
  Its signature is checked against the keys Microsoft publishes, along with the
  issuer, the audience (this bot's app id) and the expiry. Its `serviceUrl`
  claim must match the activity's own — which is what stops a valid token being
  replayed with the reply address rewritten to somebody else's server.
- **The tenant.** A bot registration can be added by any tenant that finds it.
  The tenant id on the activity must be the one this install serves; anything
  else is refused, because the documents behind it are not theirs.
- **Who is asking.** The activity names the asker's Entra object id. Their
  group memberships come from Graph (`transitiveMemberOf`, so nesting counts),
  cached for fifteen minutes, and become the `group:<id>` principals retrieval
  already enforces. **If the lookup fails the bot does not answer as
  everybody**: it answers from what every employee may read and says in the
  reply that it did.

Neither refusal explains itself to the sender. Both are one line in the log.

The reply goes back through the connector API rather than in the HTTP response,
because a self-hosted model can take a minute and the Bot Service stops waiting
long before that. The asker sees a typing indicator, then the answer with its
sources. The per-person rate limit applies here as it does on the web, and a
question that hits it is told so rather than dropped.

## Setting it up

1. **An Azure Bot registration** (Azure portal → Create a resource → Azure Bot),
   single-tenant, with a new or existing Entra app registration. Note the app
   id and create a client secret. Set the messaging endpoint to
   `https://<your-server>/teams/messages` and enable the Microsoft Teams channel.
2. **Graph permission** on that app registration: `GroupMember.Read.All`
   (application), with admin consent. That is what the group lookup needs and
   the least that will do it. `User.Read.All` is not required.
3. **Settings on the server:**

   | setting | value |
   |---|---|
   | `OK_TEAMS_ENABLED` | `true` |
   | `OK_TEAMS_APP_ID` | the bot registration's app id |
   | `OK_TEAMS_APP_PASSWORD` | its client secret |
   | `OK_TEAMS_TENANT_ID` | the one tenant this bot serves |
   | `OK_TEAMS_GROUPS_TTL_SECONDS` | how long a person's groups are trusted (default 900) |

   With any of the first four missing and `OK_TEAMS_ENABLED` on, the server
   refuses to start and names what is unset: an endpoint that accepts
   activities it cannot validate is worse than no endpoint.
4. **HTTPS.** The Bot Service will not deliver to a plain-HTTP endpoint. See
   the TLS settings in [COMPANY-SERVER.md](COMPANY-SERVER.md).
5. **The app package.** `packaging/teams/manifest.json` is a manifest to fill
   in — the app id in two places, your company's name and URLs — zipped with a
   192×192 `color.png` and a 32×32 transparent `outline.png` and uploaded by an
   administrator (Teams admin centre → Manage apps → Upload) or sideloaded for
   a pilot.

## What to expect, and its bounds

- **A removal from a group takes up to `OK_TEAMS_GROUPS_TTL_SECONDS` to bite.**
  Shorten it if that is too loose; each expiry costs one Graph call per person
  who asks.
- **Group nesting counts, sharing-by-link does not.** The principals come from
  directory membership, which is what the folder rules and the SharePoint
  mirror are written in.
- **A lost reply is lost.** If the connector refuses the delivery, the answer
  was still produced and is in the ledger; there is no retry queue. The log
  says so.
- **Threads are not conversations.** Each question is answered on its own, as
  on the web. Follow-ups ("what about contractors?") are roadmap, not code.

## Honesty about what this is

Every call and check here follows Microsoft's documentation and is proved
against `tests/fake_botframework.py`, which signs inbound tokens (correct ones
and deliberately wrong ones), serves the metadata and keys they are validated
against, answers Graph's transitive group lookup, and records what the bot
said. It has **not** been run against a real tenant, because none was available
where this was built. The first real run is what it still owes: whether the
issuer and audience hold for a single-tenant registration as documented, what
Teams actually puts in `channelData.tenant.id` for guests and federated users,
and how long the whole round trip takes with a real model behind it.
