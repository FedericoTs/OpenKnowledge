"""Findings from a whole-surface security review, each pinned to the code.

Three of these were leaks when they were written, reproduced here first and
then fixed; the rest are properties that already held and are pinned because
they are the kind that regress silently.

The shape the two access leaks shared is worth naming, because it will
recur: **the check ran over the wrong set, or over nothing at all.**
``visible_to`` loops over the documents it is handed and returns True when it
runs out, so a check over an empty set passes. A pin a curator wrote without
listing its sources cites nothing. A draft cites what the answer quotes, not
everything it was written from. In both cases the code had a control, and the
control was asking a question whose answer was "no documents, so no problem".
"""

from __future__ import annotations

import json
import re
import zipfile
from pathlib import Path

import pytest

from openknowledge import graph as knowledge_graph
from openknowledge.api.app import _safe_document_name, _safe_document_path
from openknowledge.api.runtime_settings import EDITABLE, SettingsChangeError, validate_changes
from openknowledge.backup import FORMAT, MANIFEST, _is_secret, restore_backup
from openknowledge.cascade import Cascade
from openknowledge.config import Settings
from openknowledge.configview import SECRET_NAME
from openknowledge.knowledge.store import KnowledgeStore
from openknowledge.models import write_env
from openknowledge.types import Citation, Tier

SECRET = "Executive bands run from EUR 180000 to EUR 240000."
QUESTION = "What are the executive salary bands?"
CANONICAL = "what are the executive salary bands"

# The shared corpus restricts board-comp to `board`, so `staff` is an asker
# with something hidden from them and `board` is one with nothing hidden.
STAFF = frozenset({"staff"})
BOARD = frozenset({"board"})


# -- what a pin is allowed to say, and to whom ---------------------------------


async def test_an_uncited_pin_is_withheld_where_anything_is_hidden(
    store, retriever, settings
) -> None:
    """The leak: a curator's pin, restricted content, no citation, everyone.

    ``POST /admin/pins`` defaults ``cite`` to an empty list, and a curator is
    explicitly not the person who decides who may read what. Before the fix
    this returned tier PINNED with the text below to an asker holding only
    ``staff``; the identical pin *with* a citation was correctly refused.
    """
    store.pin(CANONICAL, SECRET, citations=(), author="curator")
    answer = await Cascade(store=store, retriever=retriever, settings=settings).answer(
        QUESTION, principals=STAFF
    )
    assert answer.tier is not Tier.PINNED
    assert SECRET not in answer.text


async def test_an_uncited_pin_still_answers_when_nothing_is_hidden(
    store, retriever, settings
) -> None:
    """The other half: the fix must not silence ordinary pins.

    An asker who can reach every document in the corpus cannot learn anything
    from an unattributed answer that they could not read for themselves, so
    the pin is served - which is what every desktop install and every
    deployment without folder rules looks like.
    """
    store.pin(CANONICAL, SECRET, citations=(), author="curator")
    answer = await Cascade(store=store, retriever=retriever, settings=settings).answer(
        QUESTION, principals=BOARD
    )
    assert answer.tier is Tier.PINNED
    assert answer.text == SECRET


async def test_a_cited_pin_is_still_checked_against_its_citation(
    store, retriever, settings
) -> None:
    """The control that already worked, kept working."""
    store.pin(
        CANONICAL,
        SECRET,
        citations=(Citation(document_id="board-comp", document_title="Board", snippet="x"),),
        author="curator",
    )
    answer = await Cascade(store=store, retriever=retriever, settings=settings).answer(
        QUESTION, principals=STAFF
    )
    assert answer.tier is not Tier.PINNED


# -- what a draft was written from, not only what it quotes --------------------


async def test_a_draft_is_checked_against_what_it_was_drafted_from(
    store, retriever, settings, tmp_path: Path
) -> None:
    """A draft cites what it quotes; it is *about* everything it read.

    Before the fix this served the restricted text to `staff` and, in the
    note explaining where the draft came from, named the restricted document
    to someone not allowed to know it exists.
    """
    knowledge = KnowledgeStore(tmp_path / "knowledge.db")
    knowledge.propose(
        canonical_query=CANONICAL,
        question=QUESTION,
        answer=SECRET,
        citations=(Citation(document_id="hr-handbook", document_title="HR", snippet="x"),),
        origin_documents=("board-comp",),
        corpus_version=retriever.corpus_version,
        source="ingest",
    )
    cascade = Cascade(
        store=store,
        retriever=retriever,
        settings=settings.model_copy(update={"serve_drafts": True}),
        knowledge=knowledge,
    )
    try:
        answer = await cascade.answer(QUESTION, principals=STAFF)
        assert answer.tier is not Tier.DRAFT
        assert SECRET not in answer.text
        assert not any("board-comp" in note for note in answer.notes), (
            "the note named a document this asker may not know exists"
        )
    finally:
        knowledge.close()


# -- a setting cannot write outside its own line -------------------------------


def test_a_setting_cannot_carry_a_second_dotenv_line() -> None:
    """Every applied change is written as ``KEY=value``, one per line.

    So a value holding a newline writes a second line - a key that is not in
    EDITABLE. Measured before the fix: azure_openai_deployment accepted
    "quiet\\nOK_ADMIN_TOKEN=..." and the dotenv came back carrying exactly
    that, minting a bearer token the next start honours and which outlives
    its author's removal from the admin group.
    """
    payload = "quiet\nOK_ADMIN_TOKEN=attacker-chosen"
    accepted = []
    for key in sorted(EDITABLE):
        try:
            validate_changes({key: payload})
        except SettingsChangeError:
            continue
        accepted.append(key)
    assert not accepted, f"these settings still accept a two-line value: {accepted}"


def test_write_env_refuses_a_line_break(tmp_path: Path) -> None:
    """The same refusal at the file, because write_env has other callers."""
    with pytest.raises(ValueError, match="line break"):
        write_env(tmp_path / ".env", {"OK_ANYTHING": "quiet\nOK_ADMIN_TOKEN=x"})


# -- properties that already held ----------------------------------------------

HOSTILE_NAMES = (
    "../../etc/passwd",
    "..\\..\\windows\\system32\\x.txt",
    "/etc/passwd",
    "C:\\Windows\\x.txt",
    "....//....//etc/passwd",
    "..%c0%af..%c0%afetc/passwd",
    "HR/../../../etc/passwd",
    "a/./../../b",
    "\uff0e\uff0e/etc/passwd",
    "\uff0fetc\uff0fpasswd",
    "x\x00.md",
    ".ssh/authorized_keys",
    "",
)


@pytest.mark.parametrize("raw", HOSTILE_NAMES)
def test_no_hostile_name_leaves_the_documents_folder(raw: str) -> None:
    name = _safe_document_name(raw)
    assert name is None or ("/" not in name and "\\" not in name and not name.startswith(".")), (
        f"{raw!r} became the file name {name!r}"
    )
    path = _safe_document_path(raw)
    if path is not None:
        # Containment, asked in a way both filesystems answer the same. The
        # first version compared strings against "/corpus/", which is a POSIX
        # answer to a question Windows spells D:\corpus\ - green here, red on
        # the runner, over a path ('etc/passwd') that never escaped anything.
        root = Path("/corpus").resolve()
        assert (root / path).resolve().is_relative_to(root), f"{raw!r} became the path {path!r}"


def test_a_hostile_archive_writes_only_where_it_is_told(tmp_path: Path) -> None:
    """Zip slip, straight at restore.

    ``ZipFile.extractall`` drops ``..`` components from member names, so this
    holds today for a reason nothing in this repository states. Pinned
    because the obvious refactor - a loop that opens each member and writes
    it where the archive says - removes the only thing stopping it.

    The assertion is "nothing outside data/ and documents/", not "nothing
    outside the state directory". The first version of this test used the
    looser one and a hand-rolled extract loop passed it: the escape landed
    one level up, in the state directory itself, which is where the dotenv
    holding OK_ADMIN_TOKEN lives. A backup archive that can rewrite that file
    is a backup archive that can grant itself the admin API.
    """
    state = tmp_path / "state"
    data_dir, documents_dir = state / "data", state / "documents"
    archive = tmp_path / "hostile.okbackup"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr(MANIFEST, json.dumps({"format": FORMAT, "databases": {}, "documents": 1}))
        zf.writestr("documents/../../escaped.txt", "pwned")
        zf.writestr("documents/../../../escaped2.txt", "pwned")
        zf.writestr("../escaped3.txt", "pwned")
        zf.writestr("/escaped4.txt", "pwned")

    restore_backup(archive, Settings(data_dir=str(data_dir), documents_dir=str(documents_dir)))
    allowed = {archive.resolve()}
    written = [
        path.resolve()
        for path in tmp_path.rglob("*")
        if path.is_file()
        and path.resolve() not in allowed
        and data_dir.resolve() not in path.resolve().parents
        and documents_dir.resolve() not in path.resolve().parents
    ]
    assert not written, f"the archive wrote outside data/ and documents/: {written}"


def test_the_backup_and_the_config_page_agree_on_what_a_secret_is() -> None:
    """Two rules, one meaning: what must never be shown or shipped.

    ``backup._is_secret`` decides what a backup archive withholds;
    ``configview.SECRET_NAME`` decides what /manage prints. They are separate
    implementations of one idea, so they can drift - and the way they drift
    is that a credential added later matches one and not the other, and ends
    up in every backup taken afterwards.
    """
    disagree = [
        name
        for name in sorted(Settings.model_fields)
        if _is_secret(name) != bool(SECRET_NAME.search(name))
    ]
    assert not disagree, f"these are a secret to one and not the other: {disagree}"


def test_no_setting_that_reads_like_a_credential_is_shown_in_the_clear() -> None:
    """A new API key must not need someone to remember to redact it."""
    credential = re.compile(r"key$|secret$|token$|password$|credential", re.I)
    exposed = [
        name
        for name in sorted(Settings.model_fields)
        if credential.search(name) and not SECRET_NAME.search(name)
    ]
    assert not exposed, f"these would be printed in full on /manage: {exposed}"


def test_the_map_escapes_everything_a_stranger_can_write() -> None:
    """The map's gap nodes are questions anyone who may ask has written.

    It renders inside /manage via innerHTML, so an unescaped label would be
    script running with an admin's session. Pinned rather than assumed.
    """
    payload = '"><script>alert(1)</script>'
    built = knowledge_graph.Graph(
        nodes=(
            knowledge_graph.Node(id="d1", label=payload, kind="document", folder=payload, weight=2),
            knowledge_graph.Node(id="d2", label="ok.md", kind="document", folder="HR", weight=1),
            knowledge_graph.Node(id="q1", label=payload, kind="gap", weight=3),
        ),
        edges=(knowledge_graph.Edge(source="d1", target="d2", kind="contradiction"),),
    )
    svg = knowledge_graph.render_svg(built, knowledge_graph.layout(built))
    assert "<script>" not in svg
    assert payload not in svg
    assert "&lt;script&gt;" in svg, "the label vanished rather than being escaped"
