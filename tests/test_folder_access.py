"""Folder access rules: vocabulary, lookup, stamping, and the open default.

The signed-in end-to-end - an HR folder walled off from non-members through
real sessions - lives in test_auth_api.py. These tests hold the pieces:
the principal vocabulary refuses typos instead of matching nobody, the
deepest rule wins alone, the connector stamps documents with the rules in
force at fetch time, and with no rule everything behaves exactly as before.
"""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from openknowledge.access import effective_principals, validate_principals
from openknowledge.api.app import create_app
from openknowledge.config import Settings
from openknowledge.connectors.local_files import LocalFilesConnector
from openknowledge.knowledge.store import KnowledgeStore

# -- the vocabulary ---------------------------------------------------------


def test_the_vocabulary_is_what_sign_in_mints() -> None:
    assert validate_principals(["group:abc-123", "user:o-1", "authenticated"]) == frozenset(
        {"group:abc-123", "user:o-1", "authenticated"}
    )
    assert validate_principals([" group:x ", ""]) == frozenset({"group:x"})


def test_a_typo_is_refused_not_matched_against_nobody() -> None:
    verdict = validate_principals(["hr-group"])
    assert isinstance(verdict, str) and "group:<object-id>" in verdict
    assert isinstance(validate_principals([]), str)
    assert isinstance(validate_principals(["group: spaced id"]), str)


# -- the lookup -------------------------------------------------------------

RULES = {
    "HR": frozenset({"group:hr"}),
    "HR/Payroll": frozenset({"group:payroll"}),
}


def test_the_deepest_rule_wins_alone() -> None:
    assert effective_principals("HR", RULES) == {"group:hr"}
    assert effective_principals("HR/Policies", RULES) == {"group:hr"}
    assert effective_principals("HR/Payroll", RULES) == {"group:payroll"}
    assert effective_principals("HR/Payroll/2026", RULES) == {"group:payroll"}
    assert effective_principals("Travel", RULES) == frozenset()
    assert effective_principals("", RULES) == frozenset()
    assert effective_principals(".", RULES) == frozenset()


def test_a_root_rule_covers_loose_files() -> None:
    rules = {"": frozenset({"authenticated"}), "HR": frozenset({"group:hr"})}
    assert effective_principals("", rules) == {"authenticated"}
    assert effective_principals("Travel", rules) == {"authenticated"}
    assert effective_principals("HR", rules) == {"group:hr"}


# -- the store --------------------------------------------------------------


def test_rules_survive_like_the_decisions_they_are(tmp_path: Path) -> None:
    with KnowledgeStore(tmp_path / "knowledge.db") as store:
        store.set_folder_access("HR", frozenset({"group:hr"}))
        store.set_folder_access("HR", frozenset({"group:hr", "user:cfo"}))  # upsert
        assert store.folder_rules() == {"HR": frozenset({"group:hr", "user:cfo"})}
    with KnowledgeStore(tmp_path / "knowledge.db") as store:
        assert store.folder_rules() == {"HR": frozenset({"group:hr", "user:cfo"})}
        assert store.clear_folder_access("HR") is True
        assert store.clear_folder_access("HR") is False
        assert store.folder_rules() == {}


# -- the connector ----------------------------------------------------------


def _tree(root: Path) -> None:
    (root / "HR" / "Payroll").mkdir(parents=True)
    (root / "HR" / "leave.md").write_text("# Leave\n\n25 days.", encoding="utf-8")
    (root / "HR" / "Payroll" / "bands.md").write_text("# Bands\n\nC: 70k.", encoding="utf-8")
    (root / "handbook.md").write_text("# Handbook\n\nCloses 18:00.", encoding="utf-8")


def test_documents_are_stamped_with_the_rules_in_force(tmp_path: Path) -> None:
    _tree(tmp_path)
    rules: dict[str, frozenset[str]] = {}
    connector = LocalFilesConnector(tmp_path, folder_rules=lambda: rules)

    open_corpus = {d.document_id: d.allowed_principals for d in connector.fetch()}
    assert all(acl == frozenset() for acl in open_corpus.values())

    # The provider is consulted per fetch: new rules, new stamps, no rebuild.
    rules.update(RULES)
    ruled = {d.document_id: d.allowed_principals for d in connector.fetch()}
    assert ruled["HR-leave"] == {"group:hr"}
    assert ruled["HR-Payroll-bands"] == {"group:payroll"}
    assert ruled["handbook"] == frozenset()


def test_a_ruled_folder_still_falls_back_to_the_connector_default(tmp_path: Path) -> None:
    """The whole-corpus default (cloud connectors use it) keeps meaning
    "everything", and a folder rule overrides it for its subtree."""
    _tree(tmp_path)
    connector = LocalFilesConnector(
        tmp_path,
        allowed_principals=frozenset({"board"}),
        folder_rules=lambda: {"HR": frozenset({"group:hr"})},
    )
    stamped = {d.document_id: d.allowed_principals for d in connector.fetch()}
    assert stamped["handbook"] == {"board"}
    assert stamped["HR-leave"] == {"group:hr"}


# -- with sign-in off, rules bite only asserted principals ------------------


def test_rules_apply_to_asserted_principals_even_without_sign_in(tmp_path: Path) -> None:
    """Trusted-caller mode: no principals means the caller is the system
    itself and sees everything; asserted principals are honoured against
    the rules. Exactly today's semantics, now fed by folder rules."""
    docs = tmp_path / "documents"
    _tree(docs)
    settings = Settings(
        data_dir=str(tmp_path / "data"),
        documents_dir=str(docs),
        admin_token="t0ken",
        upload_enabled=True,
        local_enabled=False,
        embedding_enabled=False,
        _env_file=None,  # type: ignore[call-arg]
    )
    with TestClient(create_app(settings)) as client:
        headers = {"Authorization": "Bearer t0ken"}
        put = client.put("/admin/access/HR", json={"principals": ["group:hr"]}, headers=headers)
        assert put.status_code == 200

        bad = client.put("/admin/access/HR", json={"principals": ["hr"]}, headers=headers)
        assert bad.status_code == 422 and "group:<object-id>" in bad.json()["detail"]

        everything = client.post("/chat", json={"question": "What documents do you have?"})
        assert "Leave" in everything.json()["answer"]

        outsider = client.post(
            "/chat",
            json={"question": "What documents do you have?", "principals": ["group:other"]},
        )
        assert "Leave" not in outsider.json()["answer"]
        assert "Handbook" in outsider.json()["answer"]

        # Clearing the rule opens the folder again, immediately.
        assert client.delete("/admin/access/HR", headers=headers).status_code == 200
        again = client.post(
            "/chat",
            json={"question": "What documents do you have?", "principals": ["group:other"]},
        )
        assert "Leave" in again.json()["answer"]


# -- a superseded copy is one decision, not two dozen contradictions ---------


def test_a_duplicated_pair_scans_as_one_versions_conflict() -> None:
    """The aveline corpus's own spec: the free passes find two contradictions
    and one duplicated pair. The audit always said so; the server scan used
    to open every shared figure of the archive copy as its own blocking
    conflict - 26 open conflicts on an 11-document corpus, and the whole
    expenses domain refused."""
    from openknowledge.connectors.local_files import LocalFilesConnector
    from openknowledge.knowledge import scan_documents
    from openknowledge.knowledge.store import KnowledgeStore
    from openknowledge.retrieval import BM25Retriever

    docs = LocalFilesConnector("evals/corpus/aveline").fetch()
    retriever = BM25Retriever()
    retriever.index(docs)
    store = KnowledgeStore(":memory:")
    report = scan_documents(docs, store=store, retriever=retriever)
    open_conflicts = store.open_conflicts()
    kinds = sorted(c.kind for c in open_conflicts)
    assert kinds == ["numeric", "numeric", "versions"], kinds
    versions = next(c for c in open_conflicts if c.kind == "versions")
    assert versions.documents == frozenset({"archive-expenses-policy-2023", "hr-expenses-policy"})
    assert any("two versions of the same document" in n for n in report.notes)
    store.close()


def test_a_versions_conflict_never_gates_an_answer() -> None:
    from dataclasses import replace as dc_replace

    from openknowledge.knowledge.relevance import relevant_conflicts
    from openknowledge.knowledge.store import KnowledgeStore

    store = KnowledgeStore(":memory:")
    from openknowledge.knowledge.claims import find_conflicts
    from openknowledge.retrieval.base import Document

    left = Document(
        document_id="current",
        title="Expenses",
        text="Meal allowance for domestic travel is EUR 45 per day.",
    )
    right = Document(
        document_id="stale",
        title="Expenses (old)",
        text="Meal allowance for domestic travel is EUR 35 per day.",
    )
    (conflict,) = find_conflicts([left, right])
    real = store.record_conflict(conflict)
    versions = dc_replace(real, key=real.key + ":v", kind="versions")

    question = "What is the meal allowance for domestic travel?"
    assert relevant_conflicts(question, [real]) == [real]
    assert relevant_conflicts(question, [versions]) == [], (
        "a duplicated-pair record must not block the question"
    )
    store.close()


def test_the_cited_floor_threads_through_every_gate_caller() -> None:
    """engine.learn passes min_support_ratio_cited into drafting and
    re-verification; a hop that silently lacks the parameter is a TypeError
    at the first paid ingest, which no free test would meet."""
    import inspect

    from openknowledge.knowledge.generate import draft_from_document
    from openknowledge.knowledge.pipeline import draft_for_documents
    from openknowledge.knowledge.reverify import reverify_changed_documents

    for fn in (draft_for_documents, draft_from_document, reverify_changed_documents):
        assert "min_support_ratio_cited" in inspect.signature(fn).parameters, fn.__name__
