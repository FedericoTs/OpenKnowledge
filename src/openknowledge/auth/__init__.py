"""Sign-in for the company server: OIDC in front of the ACLs that exist.

The retrieval and cache layers already enforce ``principals`` on every path
that can serve an answer; this package adds the badge reader that mints those
principals from a real identity instead of believing the request body. Design
and honest limits: ``docs/ENTRA-SIGNIN.md``.

Imported only when ``OK_AUTH_MODE=oidc`` - it needs the ``auth`` extra
(``pip install 'openknowledge[auth]'``), and a deployment with sign-in off
never touches it.
"""

from .oidc import Identity, OidcClient, OidcError, PendingLogin, ProviderConfig
from .sessions import Session, SessionStore

__all__ = [
    "Identity",
    "OidcClient",
    "OidcError",
    "PendingLogin",
    "ProviderConfig",
    "Session",
    "SessionStore",
]
