"""Microsoft Teams channel - not implemented yet.

Sketched rather than omitted so the shape of the work is visible. Teams needs an
Azure Bot Service registration, an Entra ID app, a signed-JWT check on every
inbound activity, and a manifest packaged for the tenant admin to sideload or
publish.

The piece worth designing carefully is the identity mapping: an inbound Teams
activity carries ``from.aadObjectId``, and turning that into the group
memberships that populate ``principals`` needs a Graph lookup plus caching. Get
that wrong and the bot answers from documents the asker cannot open - which is
the failure this project cares most about avoiding. See ROADMAP.md.
"""

from __future__ import annotations

from .base import InboundMessage


class TeamsChannel:
    """Placeholder for the Bot Framework adapter."""

    name = "teams"

    def parse(self, payload: dict) -> InboundMessage:
        raise NotImplementedError(
            "The Teams channel is not implemented yet. It needs an Azure Bot Service "
            "registration, JWT validation on inbound activities, and a Graph lookup to "
            "map from.aadObjectId onto group principals. Use the web widget, or POST "
            "to /chat directly with the asker's principals."
        )
