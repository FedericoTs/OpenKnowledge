"""Microsoft Teams, as a bot that answers from the company's own documents.

Three things make this different from putting the widget behind a link.

*The request must be proved.* A bot endpoint is a public URL, so every inbound
activity carries a JWT the Bot Service signed, and this validates it against
the keys Microsoft publishes: signature, issuer, audience (this bot's app id),
expiry, and the ``serviceUrl`` claim against the activity's own - which is what
stops a valid token being replayed with a rewritten reply address. Nothing in
the body is believed until the token says so.

*The tenant must be the company's.* A bot registration can be reached by any
tenant that adds it. The tenant id on the activity is checked against the
configured one, so another organisation adding this bot gets a refusal rather
than answers from documents that are not theirs.

*Who is asking decides what they may read.* The activity names the asker's
Entra object id; their group memberships come from Graph, cached briefly, and
become the ``group:<id>`` principals retrieval already enforces. When the
lookup fails the bot does not fall back to answering as everybody: it answers
from what every employee may read and says that is what it did.

The reply goes back through the connector API rather than in the HTTP
response, because a self-hosted model can take a minute and the Bot Service
stops waiting long before that. The asker sees a typing indicator, then the
answer with its sources.

Built against ``tests/fake_botframework.py``, whose endpoints follow
Microsoft's documented shapes. It has not yet been run against a real tenant -
see docs/TEAMS.md.
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import httpx
import jwt

from ..types import Answer, Tier
from .base import InboundMessage

log = logging.getLogger(__name__)

#: Where the Bot Service publishes the keys it signs inbound tokens with.
BOT_METADATA_URL = "https://login.botframework.com/v1/.well-known/openidconfiguration"
#: What those tokens name themselves as.
BOT_ISSUER = "https://api.botframework.com"
#: The scope a bot asks for to call the connector back.
CONNECTOR_SCOPE = "https://api.botframework.com/.default"
GRAPH_URL = "https://graph.microsoft.com/v1.0"
LOGIN_URL = "https://login.microsoftonline.com"
#: Clock skew allowed on an inbound token, in seconds.
LEEWAY_SECONDS = 300
#: How long a fetched key set is trusted before it is fetched again.
KEYS_MAX_AGE_SECONDS = 3600.0
#: How long an asker's group list is trusted. Short enough that a removal from
#: a group takes effect within the hour, long enough that a busy channel does
#: not make a Graph call per message.
GROUPS_TTL_SECONDS = 900.0
#: Said when the asker's groups could not be read, so a thin answer is never
#: mistaken for the whole truth.
LIMITED_NOTE = (
    "I could not check which groups you are in just now, so this answers only from "
    "documents every employee may read."
)


def _without_markers(text: str, document_ids: frozenset[str]) -> str:
    """The answer as a person should read it, with its citation markers removed.

    A model cites by writing ``[hr-leave]`` after a claim; that marker is how
    the grounding gate checks the answer against its sources, and it is
    machinery, not prose. The widget strips it in the browser, so until now a
    Teams reader was the only one who saw it.

    Only markers naming a document this answer actually cited are removed. A
    bracket the document itself wrote - "clause [7] applies" - is the
    document's own text and stays, which is why this matches the ids rather
    than anything in brackets.
    """
    for document_id in sorted(document_ids, key=len, reverse=True):
        text = re.sub(rf"[ \t]*\[{re.escape(document_id)}\]", "", text)
    return re.sub(r"[ \t]{2,}", " ", text).strip()


class TeamsError(RuntimeError):
    """An activity this bot will not act on, with the reason to log."""


@dataclass(frozen=True, slots=True)
class TeamsConfig:
    app_id: str
    app_password: str
    tenant_id: str
    metadata_url: str = BOT_METADATA_URL
    issuer: str = BOT_ISSUER
    graph_url: str = GRAPH_URL
    login_url: str = LOGIN_URL
    timeout: float = 30.0


@dataclass(frozen=True, slots=True)
class Conversation:
    """Where a reply goes. Every field comes from a validated activity."""

    service_url: str
    conversation_id: str
    activity_id: str | None = None


def _text_of(activity: dict) -> str:
    """The question, with the @mention of the bot removed.

    In a channel every message names the bot, and Teams leaves that name in
    the text: "@OpenKnowledge how much parental leave" would otherwise be
    canonicalised as a different question from the same one asked in a chat.
    """
    text = str(activity.get("text") or "")
    for entity in activity.get("entities") or []:
        if entity.get("type") != "mention":
            continue
        mentioned = str((entity.get("mentioned") or {}).get("name") or "")
        raw = str(entity.get("text") or (f"<at>{mentioned}</at>" if mentioned else ""))
        if raw:
            text = text.replace(raw, " ")
        if mentioned:
            text = text.replace(f"<at>{mentioned}</at>", " ").replace(f"@{mentioned}", " ")
    return " ".join(text.split())


class TokenValidator:
    """Inbound Bot Service tokens, checked against the keys Microsoft publishes.

    The metadata document and the key set are fetched once and kept for an
    hour; an unknown ``kid`` refetches once, for key rollover, and then fails.
    """

    def __init__(
        self,
        config: TeamsConfig,
        *,
        http: httpx.AsyncClient | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.config = config
        self._http = http or httpx.AsyncClient(timeout=config.timeout)
        self._clock = clock
        self._keys: dict[str, jwt.PyJWK] = {}
        self._jwks_uri = ""
        self._fetched_at = 0.0

    async def aclose(self) -> None:
        await self._http.aclose()

    async def claims(self, authorization: str | None, activity: dict) -> dict:
        """The token's claims, or :class:`TeamsError` saying which check failed."""
        if not authorization or not authorization.lower().startswith("bearer "):
            raise TeamsError("the activity carried no bearer token")
        raw = authorization[7:].strip()
        key = await self._signing_key(raw)
        try:
            claims = jwt.decode(
                raw,
                key=key,
                algorithms=["RS256"],
                audience=self.config.app_id,
                issuer=self.config.issuer,
                leeway=LEEWAY_SECONDS,
                options={"require": ["exp", "iat", "aud", "iss"]},
            )
        except jwt.PyJWTError as exc:
            raise TeamsError(f"the inbound token failed validation: {exc}") from exc
        # The claim binds a token to the address its replies go to. Without
        # this check a token minted for one bot could be posted here with a
        # serviceUrl pointing anywhere, and the answer would follow it.
        claimed = str(claims.get("serviceurl") or claims.get("serviceUrl") or "")
        actual = str(activity.get("serviceUrl") or "")
        if claimed and claimed.rstrip("/") != actual.rstrip("/"):
            raise TeamsError(
                "the token was issued for a different serviceUrl than the activity names"
            )
        return claims

    async def _signing_key(self, raw: str) -> jwt.PyJWK:
        try:
            kid = jwt.get_unverified_header(raw).get("kid")
        except jwt.PyJWTError as exc:
            raise TeamsError(f"the inbound token is not a JWT: {exc}") from exc
        if not kid:
            raise TeamsError("the inbound token names no signing key (missing kid)")
        stale = self._clock() - self._fetched_at > KEYS_MAX_AGE_SECONDS
        if kid not in self._keys or stale:
            await self._fetch_keys()
        if kid not in self._keys:
            raise TeamsError(f"the Bot Service publishes no key {kid!r}")
        return self._keys[kid]

    async def _fetch_keys(self) -> None:
        try:
            if not self._jwks_uri or self._clock() - self._fetched_at > KEYS_MAX_AGE_SECONDS:
                metadata = await self._http.get(self.config.metadata_url)
                metadata.raise_for_status()
                document = metadata.json()
                uri = document.get("jwks_uri")
                if not uri:
                    raise TeamsError("the Bot Service metadata names no jwks_uri")
                self._jwks_uri = str(uri)
            response = await self._http.get(self._jwks_uri)
            response.raise_for_status()
            keys = response.json().get("keys", [])
        except httpx.HTTPError as exc:
            raise TeamsError(f"could not read the Bot Service signing keys: {exc}") from exc
        self._keys = {
            entry["kid"]: jwt.PyJWK(entry, algorithm="RS256")
            for entry in keys
            if entry.get("kty") == "RSA" and entry.get("kid")
        }
        self._fetched_at = self._clock()


class Connector:
    """The two calls a bot makes outward: a typing indicator and a reply.

    Both need an app-only token for the connector scope, cached until a minute
    before it expires. A failure to reply is logged and swallowed: the answer
    was already produced and paid for, and raising here would only turn one
    lost message into a stack trace nobody reads.
    """

    def __init__(
        self,
        config: TeamsConfig,
        *,
        http: httpx.AsyncClient | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.config = config
        self._http = http or httpx.AsyncClient(timeout=config.timeout)
        self._clock = clock
        self._token = ""
        self._expires_at = 0.0
        #: Bumped per token fetch, so a test can prove the cache is used.
        self.token_fetches = 0

    async def aclose(self) -> None:
        await self._http.aclose()

    async def token(self) -> str:
        if not self._token or self._clock() >= self._expires_at:
            url = f"{self.config.login_url.rstrip('/')}/{self.config.tenant_id}/oauth2/v2.0/token"
            response = await self._http.post(
                url,
                data={
                    "grant_type": "client_credentials",
                    "client_id": self.config.app_id,
                    "client_secret": self.config.app_password,
                    "scope": CONNECTOR_SCOPE,
                },
            )
            self.token_fetches += 1
            if response.status_code != 200:
                raise TeamsError(f"the connector token request failed: HTTP {response.status_code}")
            body = response.json()
            self._token = str(body["access_token"])
            self._expires_at = self._clock() + float(body.get("expires_in", 3600)) - 60.0
        return self._token

    async def send(self, where: Conversation, activity: dict) -> None:
        url = f"{where.service_url.rstrip('/')}/v3/conversations/{where.conversation_id}/activities"
        if where.activity_id:
            url = f"{url}/{where.activity_id}"
        try:
            token = await self.token()
            response = await self._http.post(
                url, json=activity, headers={"Authorization": f"Bearer {token}"}
            )
            if response.status_code >= 400:
                log.warning(
                    "teams: the connector refused a reply: HTTP %s %s",
                    response.status_code,
                    response.text[:200],
                )
        except (TeamsError, httpx.HTTPError) as exc:
            log.warning("teams: could not deliver a reply: %s", exc)

    async def typing(self, where: Conversation) -> None:
        await self.send(Conversation(where.service_url, where.conversation_id), {"type": "typing"})


class GroupLookup:
    """An asker's Entra groups, from Graph, cached briefly.

    ``transitiveMemberOf`` rather than ``memberOf``: a person in a group that
    is a member of the group a folder names should be able to read that
    folder, and nesting is how directories are actually organised. Only the
    group ids are kept.
    """

    def __init__(
        self,
        config: TeamsConfig,
        *,
        http: httpx.AsyncClient | None = None,
        clock: Callable[[], float] = time.time,
        ttl: float = GROUPS_TTL_SECONDS,
    ) -> None:
        self.config = config
        self._http = http or httpx.AsyncClient(timeout=config.timeout)
        self._clock = clock
        self._ttl = ttl
        self._cache: dict[str, tuple[float, frozenset[str]]] = {}
        self._token = ""
        self._expires_at = 0.0
        self.lookups = 0

    async def aclose(self) -> None:
        await self._http.aclose()

    async def _token_for_graph(self) -> str:
        if not self._token or self._clock() >= self._expires_at:
            url = f"{self.config.login_url.rstrip('/')}/{self.config.tenant_id}/oauth2/v2.0/token"
            response = await self._http.post(
                url,
                data={
                    "grant_type": "client_credentials",
                    "client_id": self.config.app_id,
                    "client_secret": self.config.app_password,
                    "scope": "https://graph.microsoft.com/.default",
                },
            )
            if response.status_code != 200:
                raise TeamsError(f"the Graph token request failed: HTTP {response.status_code}")
            body = response.json()
            self._token = str(body["access_token"])
            self._expires_at = self._clock() + float(body.get("expires_in", 3600)) - 60.0
        return self._token

    async def groups(self, object_id: str) -> frozenset[str]:
        """Group ids this person belongs to. Raises :class:`TeamsError` when unknown."""
        cached = self._cache.get(object_id)
        now = self._clock()
        if cached is not None and now - cached[0] < self._ttl:
            return cached[1]
        url = (
            f"{self.config.graph_url.rstrip('/')}/users/{object_id}/transitiveMemberOf"
            "/microsoft.graph.group?$select=id&$top=999"
        )
        found: set[str] = set()
        try:
            token = await self._token_for_graph()
            while url:
                response = await self._http.get(url, headers={"Authorization": f"Bearer {token}"})
                if response.status_code >= 400:
                    raise TeamsError(f"Graph refused the group lookup: HTTP {response.status_code}")
                body = response.json()
                found.update(str(row["id"]) for row in body.get("value") or [] if row.get("id"))
                url = str(body.get("@odata.nextLink") or "")
        except httpx.HTTPError as exc:
            raise TeamsError(f"could not reach Graph for the group lookup: {exc}") from exc
        self.lookups += 1
        groups = frozenset(found)
        self._cache[object_id] = (now, groups)
        return groups


@dataclass
class TeamsChannel:
    """Bot Framework activities in, grounded answers out."""

    name: str = "teams"
    config: TeamsConfig = field(default_factory=lambda: TeamsConfig("", "", ""))

    def parse(self, payload: dict) -> InboundMessage:
        """The question and who asked it, from an activity already validated.

        Principals are deliberately absent here: they need a Graph call, so
        they are added by :meth:`principals`, which the route awaits. Parsing
        a payload must never be the place a permission is decided.
        """
        if str(payload.get("type") or "") != "message":
            raise TeamsError(f"not a message activity: {payload.get('type')!r}")
        sender = payload.get("from") or {}
        user_id = str(sender.get("aadObjectId") or sender.get("id") or "")
        if not user_id:
            raise TeamsError("the activity names no sender")
        text = _text_of(payload)
        if not text:
            raise TeamsError("the activity carries no text")
        conversation = payload.get("conversation") or {}
        return InboundMessage(
            text=text,
            user_id=user_id,
            channel=self.name,
            thread_id=str(conversation.get("id") or "") or None,
        )

    def tenant_of(self, payload: dict) -> str:
        channel_data = payload.get("channelData") or {}
        return str((channel_data.get("tenant") or {}).get("id") or "")

    def conversation_of(self, payload: dict) -> Conversation:
        conversation = payload.get("conversation") or {}
        return Conversation(
            service_url=str(payload.get("serviceUrl") or ""),
            conversation_id=str(conversation.get("id") or ""),
        )

    async def principals(
        self, message: InboundMessage, lookup: GroupLookup
    ) -> tuple[frozenset[str], bool]:
        """Who this asker is, and whether their groups are actually known.

        Fails closed: a lookup that cannot answer yields the principals every
        signed-in employee holds and nothing more, and the caller says so in
        the reply rather than quietly answering from less than it should.
        """
        base = {"authenticated", f"user:{message.user_id}"}
        try:
            groups = await lookup.groups(message.user_id)
        except TeamsError as exc:
            log.warning("teams: group lookup failed for %s: %s", message.user_id, exc)
            return frozenset(base), False
        return frozenset(base | {f"group:{g}" for g in groups}), True

    def reply(self, answer: Answer, *, limited: bool = False) -> dict:
        """The answer as an activity: the text, then where it came from.

        Markdown rather than an Adaptive Card: a card renders the same
        sentences with more moving parts, and the one thing that must survive
        every Teams client is the citation - a reply nobody can check is the
        thing this product exists not to produce.
        """
        cited = frozenset(c.document_id for c in answer.citations)
        lines = [_without_markers(answer.text, cited)]
        if limited:
            lines.append(f"\n\n_{LIMITED_NOTE}_")
        if answer.citations:
            lines.append("\n\n**Sources**")
            for citation in answer.citations:
                where = f" ({citation.locator})" if citation.locator else ""
                title = citation.document_title or citation.document_id
                if citation.url and citation.url.startswith(("http://", "https://")):
                    lines.append(f"\n- [{title}]({citation.url}){where}")
                else:
                    lines.append(f"\n- {title}{where}")
        elif answer.tier not in (Tier.REFUSED, Tier.CONTESTED):
            lines.append("\n\n_No source was cited for this answer._")
        return {
            "type": "message",
            "textFormat": "markdown",
            "text": "".join(lines),
        }
