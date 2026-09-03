"""Every setting this install is running with, and whether anybody set it.

The settings a server runs with are read once, at start, from a dotenv in the
resolved state directory and from ``OK_*`` environment variables. Three things
go wrong with that in practice: the file was edited after the start and the
change is not in force; a variable set in one shell is not set in the
service's; and a value somebody assumed was the default was set two admins
ago. Each of them is answered by the same view - what the process actually
holds, next to what the default would have been, with the file it read named.

Secrets are the one thing this never shows. A field whose name ends in
``_key``, ``_secret`` or ``_token`` is reported as set or not set, and a test
plants a sentinel in every one of them and looks for it in the output.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from pydantic import SecretStr

from .api.runtime_settings import EDITABLE
from .config import Settings
from .paths import state_paths

#: A name that ends in one of these holds a credential, or a path to one.
#: ``_tokens`` does not match: ``max_answer_tokens`` counts words, not access.
SECRET_NAME = re.compile(r"(_key|_secret|_token|password)$")

#: Where each setting is filed on the page: first match wins, by prefix or by
#: name. A setting no rule claims lands in "Other", which a test refuses -
#: every setting has a home or the page does not ship.
GROUPS: tuple[tuple[str, tuple[str, ...], tuple[str, ...]], ...] = (
    (
        "Where things live",
        ("documents_", "vectors_", "contacts_", "pdf_"),
        ("data_dir", "update_check"),
    ),
    (
        "Retrieval and grounding",
        ("retrieval_", "chunk_", "min_support_", "rerank_"),
        ("require_citations", "tag_routing", "max_answer_tokens", "system_prompt_suffix"),
    ),
    (
        "Knowledge",
        ("draft_", "conflict_", "deontic_"),
        ("serve_drafts", "block_on_conflict", "reverify_on_change", "max_documents_per_ingest"),
    ),
    ("Local model", ("local_",), ()),
    ("Embeddings and the semantic cache", ("embedding_", "semantic_cache_"), ()),
    (
        "Escalation and budget",
        ("escalation_", "azure_openai_", "ladder", "budget_"),
        ("anthropic_api_key", "openai_api_key"),
    ),
    ("SharePoint", ("sharepoint_",), ()),
    ("Teams", ("teams_",), ()),
    ("Google Drive", ("drive_",), ()),
    (
        "Serving",
        ("upload_", "asker_", "tls_", "contact_", "trusted_", "disk_"),
        ("bind_host", "public_url", "website_enabled"),
    ),
    ("Sign-in and admin", ("oidc_",), ("auth_mode", "admin_token", "session_hours")),
)


def group_of(name: str) -> str:
    for label, prefixes, names in GROUPS:
        if name in names or (prefixes and name.startswith(prefixes)):
            return label
    return "Other"


def is_secret(name: str, value: object) -> bool:
    return bool(SECRET_NAME.search(name)) or isinstance(value, SecretStr)


def _value(value: object, *, secret: bool) -> Any:
    """What the page may show: a secret only as set or not set, the rest as is."""
    if secret:
        return "set" if value not in (None, "") else "not set"
    return _shown(value)


def _shown(value: object) -> Any:
    """A JSON-able rendering of a non-secret value."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, list | tuple):
        return [_shown(v) for v in value]
    return value


def describe(settings: Settings) -> dict[str, Any]:
    """The effective configuration, grouped, with every secret redacted."""
    grouped: dict[str, list[dict[str, Any]]] = {label: [] for label, _, _ in GROUPS}
    grouped["Other"] = []
    for name, field in Settings.model_fields.items():
        value = getattr(settings, name)
        default = field.get_default(call_default_factory=True)
        secret = is_secret(name, value)
        grouped[group_of(name)].append(
            {
                "name": name,
                "env": f"OK_{name.upper()}",
                "value": _value(value, secret=secret),
                "redacted": secret,
                "is_default": value == default,
                # How a change would take effect, when the page can make one.
                "live": EDITABLE.get(name),
            }
        )
    state = state_paths()
    return {
        "state": {
            "mode": state.mode,
            "root": str(state.root),
            "env_file": str(state.env_file),
            "env_file_exists": state.env_file.exists(),
        },
        "groups": [
            {
                "name": label,
                "settings": rows,
                "set": sum(1 for r in rows if not r["is_default"]),
            }
            for label, rows in grouped.items()
            if rows
        ],
    }
