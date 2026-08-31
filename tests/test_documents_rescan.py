"""Documents that change in the folder rather than through the app.

Uploads and deletes re-index themselves. On a shared server documents also
arrive the other way: dropped into the folder, synced from SharePoint,
corrected in place by whoever owns them - and nothing tells the app at all.

Measured before this existed: a policy edited on disk left the index holding
the old text. The answer then cited last year's figure with every appearance
of being current, which is the one kind of wrong answer this product exists
to prevent - and the careful cache-key design could not help, because the
cache is only asked after the index has already decided what the corpus says.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from openknowledge.api.engine import build_engine
from openknowledge.config import Settings

POLICY = """# Expenses Policy

Meals are reimbursed up to EUR 40 per day when travelling on company business.
Receipts must accompany every expense claim.
"""


@pytest.fixture
def engine(tmp_path: Path):
    docs = tmp_path / "documents"
    docs.mkdir()
    (docs / "expenses.md").write_text(POLICY, encoding="utf-8")
    settings = Settings(
        data_dir=str(tmp_path / "data"),
        documents_dir=str(docs),
        local_enabled=False,
        embedding_enabled=False,
        _env_file=None,  # type: ignore[call-arg]
    )
    built = build_engine(settings)
    yield built, docs
    built.store.close()
    built.knowledge.close()


def test_a_policy_corrected_on_disk_is_re_read(engine) -> None:
    """The field case. Finance fixes the number in the shared folder; nobody
    uploads anything; the answer must follow the file."""
    built, docs = engine
    before = built.retriever.corpus_version
    assert "EUR 40" in " ".join(d.text for d in built.documents)

    policy = docs / "expenses.md"
    policy.write_text(POLICY.replace("EUR 40", "EUR 75"), encoding="utf-8")

    assert built.reindex_if_documents_changed() is True
    assert built.retriever.corpus_version != before
    text = " ".join(d.text for d in built.documents)
    assert "EUR 75" in text and "EUR 40" not in text


def test_an_untouched_folder_is_not_re_read(engine) -> None:
    """The check runs on a timer, so its cost when nothing changed is the
    cost of the feature. It must decline to work."""
    built, _ = engine
    assert built.reindex_if_documents_changed() is False
    assert built.reindex_if_documents_changed() is False


def test_a_new_document_is_noticed(engine) -> None:
    built, docs = engine
    (docs / "handbook.md").write_text("# Handbook\n\nNotice is three months.\n", encoding="utf-8")
    assert built.reindex_if_documents_changed() is True
    assert built.retriever.document_count == 2


def test_a_removed_document_is_noticed(engine) -> None:
    built, docs = engine
    (docs / "expenses.md").unlink()
    assert built.reindex_if_documents_changed() is True
    assert built.retriever.document_count == 0


def test_an_edit_that_keeps_the_size_is_still_noticed(engine) -> None:
    """Same length, different meaning - "EUR 40" to "EUR 90". Size alone
    would miss it; the stamp carries the modification time too."""
    built, docs = engine
    policy = docs / "expenses.md"
    time.sleep(0.01)
    policy.write_text(POLICY.replace("EUR 40", "EUR 90"), encoding="utf-8")
    assert built.reindex_if_documents_changed() is True
    assert "EUR 90" in " ".join(d.text for d in built.documents)


def test_the_stamp_ignores_files_no_parser_can_read(engine) -> None:
    """A colleague's stray .zip in the folder must not cause a re-index every
    minute for the life of the server."""
    built, docs = engine
    (docs / "archive.zip").write_bytes(b"PK\x03\x04 not a document")
    assert built.reindex_if_documents_changed() is False
