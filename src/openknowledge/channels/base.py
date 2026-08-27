"""Chat surfaces.

A channel translates between a messaging platform and one call to
:meth:`~openknowledge.cascade.router.Cascade.answer`. The interface is small on
purpose - the cascade does not know or care where a question came from.

The part that is not cosmetic is ``principals``. Every channel already knows who
is asking (SSO claims, Teams tenant groups, Slack workspace membership), and that
identity is what makes access control work. A channel that drops it turns a
permission-aware system into an open one, so the protocol requires it explicitly
rather than leaving it to be forgotten.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class InboundMessage:
    """A question, normalised out of whatever the platform sent."""

    text: str
    user_id: str
    channel: str
    #: Groups/roles the asker belongs to, from the platform's own identity.
    #: ``None`` means unrestricted, which is only correct for a deployment where
    #: every document is visible to everyone.
    principals: frozenset[str] | None = field(default=None)
    thread_id: str | None = None


@runtime_checkable
class Channel(Protocol):
    """A chat surface."""

    name: str

    def parse(self, payload: dict) -> InboundMessage:
        """Turn a platform webhook payload into an :class:`InboundMessage`."""
        ...
