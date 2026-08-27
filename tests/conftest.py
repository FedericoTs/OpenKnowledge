from __future__ import annotations

import pytest

from openknowledge.cache import AnswerStore
from openknowledge.config import Settings
from openknowledge.knowledge import KnowledgeStore
from openknowledge.retrieval import BM25Retriever, Document


@pytest.fixture
def documents() -> list[Document]:
    return [
        Document(
            "hr-handbook",
            "HR Handbook",
            "Parental leave. Employees with at least 12 months of continuous service are "
            "entitled to 20 weeks of fully paid parental leave. Requests must be submitted "
            "at least 30 days in advance through the HR portal. Unused leave does not "
            "carry over into the following year.",
        ),
        Document(
            "expenses",
            "Expenses Policy",
            "Travel expenses require prior written approval for any amount above EUR 500. "
            "Meals are reimbursed up to EUR 45 per day. Alcohol is not reimbursable under "
            "any circumstances.",
        ),
        Document(
            "board-comp",
            "Board Compensation",
            "Executive salary bands for the coming year are set between EUR 180000 and "
            "EUR 240000 pending board approval.",
            allowed_principals=frozenset({"board"}),
        ),
    ]


@pytest.fixture
def retriever(documents: list[Document]) -> BM25Retriever:
    r = BM25Retriever()
    r.index(documents)
    return r


@pytest.fixture
def store() -> AnswerStore:
    with AnswerStore() as s:
        yield s


@pytest.fixture
def knowledge() -> KnowledgeStore:
    with KnowledgeStore() as store:
        yield store


@pytest.fixture
def settings() -> Settings:
    return Settings(
        local_enabled=True,
        escalation_enabled=False,
        _env_file=None,  # type: ignore[call-arg]
    )
