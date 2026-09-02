"""The Google Drive connector - not implemented yet.

Placed here deliberately rather than left out, so the shape of the work is
visible: an OAuth registration plus a paged listing call plus a text extraction
step, and it must populate ``allowed_principals`` from the source's own ACLs.
Doing that properly - group expansion, inherited permissions, sharing links - is
most of the work, and doing it wrong leaks documents. SharePoint has been built
along exactly those lines (``sharepoint.py``); Drive has not. See ROADMAP.md.
"""

from __future__ import annotations

from ..retrieval.base import Document


class NotImplementedConnector:
    """Base for connectors that are specified but not built."""

    name = "unimplemented"
    setup_hint = ""

    def fetch(self) -> list[Document]:
        raise NotImplementedError(
            f"The {self.name} connector is not implemented yet. {self.setup_hint} "
            "Use LocalFilesConnector, or export the library to a folder for now."
        )


class GoogleDriveConnector(NotImplementedConnector):
    """Drive v3: ``files.list`` with ``supportsAllDrives``, plus ``permissions.list`` per file."""

    name = "google-drive"
    setup_hint = (
        "It needs a Google Cloud service account with domain-wide delegation, and must map "
        "each file's permissions - including inherited folder ACLs - onto allowed_principals."
    )
