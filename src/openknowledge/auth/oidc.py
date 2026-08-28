"""OpenID Connect, small enough to audit.

The company server's sign-in: authorization-code flow with PKCE, endpoints
read from the issuer's discovery document, ID tokens validated against the
issuer's published keys. Entra ID is the documented first-class provider;
anything that speaks OIDC works by pointing the same three settings at it -
which is also what makes this testable, because the test suite runs the whole
flow against a loopback IdP signing with a throwaway key.

Deliberately not a framework. The one flow this server needs is a redirect,
an exchange, and a signature check; PyJWT does the cryptography and the claim
checks, httpx does the two HTTP calls, and everything rejected says why in a
sentence a person can act on. MSAL earns its weight when Graph or
on-behalf-of flows appear; a login does not.

Every failure raises :class:`OidcError` with a human-readable reason - these
strings end up on the browser's error page, so they name fixes, not internals.
"""

from __future__ import annotations

import base64
import hashlib
import secrets
import time
from dataclasses import dataclass

import httpx
import jwt

#: Clock skew tolerated when validating token timestamps, in seconds. Entra's
#: own guidance; enough for a server whose NTP drifted a little, small enough
#: that an expired token stays expired.
_LEEWAY_SECONDS = 60

#: How long fetched signing keys are trusted before refetching. Providers
#: rotate keys rarely and publish ahead of time; an unknown ``kid`` also
#: forces a refetch, so this only bounds staleness, not correctness.
_KEYS_MAX_AGE_SECONDS = 3600.0


class OidcError(Exception):
    """Sign-in failed, with a reason fit for the person who saw it fail."""


@dataclass(frozen=True, slots=True)
class ProviderConfig:
    """The three endpoints this flow needs, from the discovery document."""

    issuer: str
    authorization_endpoint: str
    token_endpoint: str
    jwks_uri: str


@dataclass(frozen=True, slots=True)
class PendingLogin:
    """What must survive between the redirect out and the callback in."""

    state: str
    nonce: str
    code_verifier: str
    created_at: float


@dataclass(frozen=True, slots=True)
class Identity:
    """Who signed in, reduced to what the ACL machinery needs."""

    subject: str
    name: str
    groups: tuple[str, ...]


class OidcClient:
    """One provider, one client registration, the whole flow."""

    def __init__(
        self,
        *,
        issuer: str,
        client_id: str,
        client_secret: str = "",
        groups_claim: str = "groups",
    ) -> None:
        self.issuer = issuer.rstrip("/")
        self.client_id = client_id
        self.client_secret = client_secret
        self.groups_claim = groups_claim
        self._http = httpx.AsyncClient(timeout=10.0)
        self._provider: ProviderConfig | None = None
        self._keys: dict[str, jwt.PyJWK] = {}
        self._keys_fetched_at = 0.0

    async def aclose(self) -> None:
        await self._http.aclose()

    # -- discovery ---------------------------------------------------------

    async def provider(self) -> ProviderConfig:
        """The provider's endpoints, discovered once and kept.

        The document's ``issuer`` must equal the configured one - that check
        is what stops a mistyped tenant from silently trusting a different
        authority. For Entra, use the tenant-id (GUID) form of the issuer:
        ``https://login.microsoftonline.com/<tenant-id>/v2.0``.
        """
        if self._provider is not None:
            return self._provider
        url = f"{self.issuer}/.well-known/openid-configuration"
        try:
            response = await self._http.get(url)
            response.raise_for_status()
            document = response.json()
        except httpx.HTTPError as exc:
            raise OidcError(
                f"could not read the identity provider's discovery document: {exc}"
            ) from exc
        if document.get("issuer", "").rstrip("/") != self.issuer:
            raise OidcError(
                f"the discovery document at {url} names issuer {document.get('issuer')!r}, "
                f"not the configured {self.issuer!r} - for Entra, use the tenant-id form "
                "of the issuer, not a domain name"
            )
        try:
            self._provider = ProviderConfig(
                issuer=self.issuer,
                authorization_endpoint=document["authorization_endpoint"],
                token_endpoint=document["token_endpoint"],
                jwks_uri=document["jwks_uri"],
            )
        except KeyError as exc:
            raise OidcError(f"the discovery document is missing {exc}") from exc
        return self._provider

    # -- the flow ----------------------------------------------------------

    async def begin_login(self, redirect_uri: str) -> tuple[str, PendingLogin]:
        """The URL to send the browser to, and what to remember until it returns."""
        provider = await self.provider()
        pending = PendingLogin(
            state=secrets.token_urlsafe(32),
            nonce=secrets.token_urlsafe(32),
            code_verifier=secrets.token_urlsafe(48),
            created_at=time.time(),
        )
        challenge = _b64url(hashlib.sha256(pending.code_verifier.encode("ascii")).digest())
        params = httpx.QueryParams(
            response_type="code",
            client_id=self.client_id,
            redirect_uri=redirect_uri,
            scope="openid profile",
            state=pending.state,
            nonce=pending.nonce,
            code_challenge=challenge,
            code_challenge_method="S256",
        )
        return f"{provider.authorization_endpoint}?{params}", pending

    async def complete_login(self, code: str, pending: PendingLogin, redirect_uri: str) -> Identity:
        """Exchange the code, validate the ID token, return who signed in."""
        provider = await self.provider()
        form = {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": redirect_uri,
            "client_id": self.client_id,
            "code_verifier": pending.code_verifier,
        }
        if self.client_secret:
            form["client_secret"] = self.client_secret
        try:
            response = await self._http.post(provider.token_endpoint, data=form)
            response.raise_for_status()
            payload = response.json()
        except httpx.HTTPError as exc:
            raise OidcError(f"the identity provider refused the code exchange: {exc}") from exc
        raw_token = payload.get("id_token")
        if not raw_token:
            raise OidcError("the token response carried no id_token")
        claims = await self._validated_claims(raw_token, provider)
        if claims.get("nonce") != pending.nonce:
            raise OidcError("the token's nonce does not match this sign-in attempt")
        return self._identity_from(claims)

    # -- validation --------------------------------------------------------

    async def _validated_claims(self, raw_token: str, provider: ProviderConfig) -> dict:
        key = await self._signing_key(raw_token, provider)
        try:
            return jwt.decode(
                raw_token,
                key=key,
                algorithms=["RS256"],
                audience=self.client_id,
                issuer=self.issuer,
                leeway=_LEEWAY_SECONDS,
                options={"require": ["exp", "iat", "aud", "iss"]},
            )
        except jwt.PyJWTError as exc:
            raise OidcError(f"the ID token failed validation: {exc}") from exc

    async def _signing_key(self, raw_token: str, provider: ProviderConfig) -> jwt.PyJWK:
        """The provider key that signed this token, by ``kid``.

        Fetched with the same httpx stack as everything else - a second HTTP
        client with its own proxy rules is how sign-in works on the laptop
        and fails behind the corporate egress. An unknown ``kid`` refetches
        once (key rollover), then fails plainly.
        """
        try:
            kid = jwt.get_unverified_header(raw_token).get("kid")
        except jwt.PyJWTError as exc:
            raise OidcError(f"the ID token is not a JWT: {exc}") from exc
        if not kid:
            raise OidcError("the ID token names no signing key (missing kid)")
        stale = time.time() - self._keys_fetched_at > _KEYS_MAX_AGE_SECONDS
        if kid not in self._keys or stale:
            await self._fetch_keys(provider)
        if kid not in self._keys:
            raise OidcError(f"the identity provider publishes no key {kid!r}")
        return self._keys[kid]

    async def _fetch_keys(self, provider: ProviderConfig) -> None:
        try:
            response = await self._http.get(provider.jwks_uri)
            response.raise_for_status()
            document = response.json()
        except httpx.HTTPError as exc:
            raise OidcError(f"could not read the identity provider's signing keys: {exc}") from exc
        keys: dict[str, jwt.PyJWK] = {}
        for entry in document.get("keys", []):
            if entry.get("kty") != "RSA" or not entry.get("kid"):
                continue
            keys[entry["kid"]] = jwt.PyJWK(entry, algorithm="RS256")
        self._keys = keys
        self._keys_fetched_at = time.time()

    def _identity_from(self, claims: dict) -> Identity:
        # Entra's `oid` is the person's immutable directory id; `sub` is
        # pairwise per app registration. Prefer the one that stays stable
        # when the app registration is recreated.
        subject = claims.get("oid") or claims.get("sub")
        if not subject:
            raise OidcError("the ID token carries neither oid nor sub")
        overage = claims.get("_claim_names", {})
        if self.groups_claim in overage:
            raise OidcError(
                "this account is in too many groups for the token to list them "
                "(Entra's groups overage). Configure the app registration to emit "
                "'groups assigned to the application' and assign the groups that "
                "matter, so the token stays bounded."
            )
        raw_groups = claims.get(self.groups_claim, [])
        if not isinstance(raw_groups, list):
            raise OidcError(f"the {self.groups_claim!r} claim is not a list")
        groups = tuple(str(g) for g in raw_groups)
        name = str(claims.get("name") or claims.get("preferred_username") or subject)
        return Identity(subject=str(subject), name=name, groups=groups)


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")
