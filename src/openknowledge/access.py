"""Folder access rules: which principals may read which part of the tree.

Folders are the corpus's categories; a rule names who may read a category.
The rule store lives with the other human decisions (the knowledge store),
the stamping happens at index time in the connector, and enforcement is the
ACL machinery that already guards retrieval, every cache tier and the
corpus listing - this module is only the shared vocabulary and the lookup.

Two deliberate simplicities:

- **The deepest rule wins, alone.** A rule on ``HR/Payroll`` replaces the
  rule on ``HR`` for that subtree rather than merging with it. Merging
  reads as convenient until a broad ancestor rule silently widens a
  narrow one; replacement means every folder's audience is one rule you
  can point at.
- **No rule means everyone.** Unruled folders and loose root files keep
  today's behaviour, so turning the feature on restricts exactly what an
  admin ruled and nothing else.
"""

from __future__ import annotations

import re
from collections.abc import Mapping

#: What a principal may look like. The vocabulary is what sign-in mints
#: (`auth/sessions.py`): a typo'd principal would not fail - it would
#: never match anyone, which reads as "nobody can see HR/" three weeks
#: later. Refusing anything outside the vocabulary keeps rules honest.
PRINCIPAL_PATTERN = re.compile(r"^(authenticated|group:\S+|user:\S+)$")


def validate_principals(raw: list[str]) -> frozenset[str] | str:
    """The principals as a set, or a sentence saying what was wrong."""
    cleaned = [p.strip() for p in raw if p.strip()]
    if not cleaned:
        return "a rule needs at least one principal; delete the rule to open the folder"
    bad = [p for p in cleaned if not PRINCIPAL_PATTERN.fullmatch(p)]
    if bad:
        return (
            f"unknown principal form: {', '.join(sorted(bad))} - use "
            "'group:<object-id>', 'user:<object-id>' or 'authenticated'"
        )
    return frozenset(cleaned)


def effective_principals(folder: str, rules: Mapping[str, frozenset[str]]) -> frozenset[str]:
    """Who may read a document filed under ``folder``.

    Walks from the folder up towards the root and returns the first rule
    found - the deepest one. Empty means unrestricted. ``folder`` is the
    posix-relative folder path ('' or '.' for the root).
    """
    current = "" if folder == "." else folder
    while current:
        rule = rules.get(current)
        if rule:
            return rule
        current, _, _ = current.rpartition("/")
    return rules.get("", frozenset()) or frozenset()
