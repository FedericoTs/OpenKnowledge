"""One server, several colleagues, all asking at once.

The desktop app serialises by hand: one person, one question. A company
server does not, and the code's only defence was a comment in llama.py
saying the app "serializes its requests through the cascade anyway" - an
assumption about behaviour, not a property of anything.

Measured before this gate existed, against a real llama-server with one
slot: four simultaneous questions answered one and refused three, each
refusal telling the asker that no model was reachable and their
configuration was at fault. Nothing was misconfigured. The server severs
the streams it has no slot for.

So the provider queues to the slot count instead. These tests use an
endpoint that behaves the way llama-server does - it refuses to be in two
places at once - so a regression fails here rather than in someone's
office.
"""

from __future__ import annotations

import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from openknowledge.providers.base import ProviderError
from openknowledge.providers.openai_compat import OpenAICompatProvider


class _SlottedServer:
    """An endpoint with a fixed number of slots, which severs the overflow.

    Deliberately not a queue: llama-server does not politely wait, and a
    fake that queued would pass whether or not the provider did its job.
    """

    def __init__(self, slots: int) -> None:
        serving = self
        self.slots = slots
        self.busy = 0
        self.high_water = 0
        self.severed = 0
        self._lock = threading.Lock()

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args: object) -> None:  # noqa: D102
                pass

            def do_POST(self) -> None:  # noqa: N802 - http.server API
                with serving._lock:
                    serving.busy += 1
                    serving.high_water = max(serving.high_water, serving.busy)
                    over = serving.busy > serving.slots
                    if over:
                        serving.severed += 1
                try:
                    length = int(self.headers.get("Content-Length", 0))
                    self.rfile.read(length)
                    if over:
                        # No slot: accept, then cut the connection - exactly
                        # what the field failure looked like.
                        self.close_connection = True
                        return
                    body = json.dumps(
                        {
                            "choices": [{"message": {"content": "20 weeks [handbook]"}}],
                            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
                        }
                    ).encode()
                    import time

                    time.sleep(0.25)  # long enough for the others to pile up
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                finally:
                    with serving._lock:
                        serving.busy -= 1

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_port}/v1"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


async def _ask_together(provider: OpenAICompatProvider, n: int) -> list[object]:
    async def one(i: int) -> object:
        try:
            return await provider.complete(
                system="s", context="c", question=f"question {i}", max_tokens=32
            )
        except ProviderError as exc:
            return exc

    return await asyncio.gather(*(one(i) for i in range(n)))


@pytest.mark.parametrize("slots", [1, 2])
def test_every_simultaneous_question_is_answered(slots: int) -> None:
    """The bar: N people asking at once get N answers. Not N-1, and never a
    refusal that blames their configuration."""
    endpoint = _SlottedServer(slots)
    try:
        provider = OpenAICompatProvider(
            model_id="local", base_url=endpoint.url, tier="local", parallel=slots
        )
        results = asyncio.run(_ask_together(provider, 6))
    finally:
        endpoint.close()

    failures = [r for r in results if isinstance(r, ProviderError)]
    assert not failures, f"{len(failures)} of 6 failed under load: {failures[:2]}"
    assert endpoint.severed == 0, "the gate let more through than the endpoint had slots"
    assert endpoint.high_water <= slots, (
        f"{endpoint.high_water} requests were in flight against {slots} slot(s)"
    )


def test_without_the_gate_the_endpoint_severs_streams() -> None:
    """The failure being fixed, reproduced: with no limit the overflow is
    cut off. This is what the field measurement looked like, and it is what
    proves the fake is not simply forgiving."""
    endpoint = _SlottedServer(1)
    try:
        provider = OpenAICompatProvider(
            model_id="local", base_url=endpoint.url, tier="local", parallel=0
        )
        results = asyncio.run(_ask_together(provider, 6))
    finally:
        endpoint.close()

    assert any(isinstance(r, ProviderError) for r in results), (
        "the fake endpoint must sever overflow, or the gate test proves nothing"
    )


def test_a_severed_stream_names_contention_not_a_dead_endpoint() -> None:
    """The refusal that sent an operator to check a healthy server. The
    message must name the slots, because that is the thing to change."""
    endpoint = _SlottedServer(1)
    try:
        provider = OpenAICompatProvider(
            model_id="local", base_url=endpoint.url, tier="local", parallel=0
        )
        results = asyncio.run(_ask_together(provider, 6))
    finally:
        endpoint.close()

    errors = [r for r in results if isinstance(r, ProviderError)]
    assert errors
    assert any("OK_LOCAL_PARALLEL" in str(e) for e in errors), errors[0]


# -- the index is replaced, never edited in place -----------------------------


def test_searching_while_the_corpus_is_rebuilt_never_sees_a_seam() -> None:
    """A rebuild used to clear the retriever's parallel arrays and refill
    them, so a search arriving mid-rebuild could read a short chunk list - a
    wrong "not covered" - or, worse, a chunk whose statistics belonged to a
    different chunk, which is a citation naming a document the text never
    came from.

    The state is one frozen snapshot now, swapped by a single assignment, so
    a reader sees wholly the old corpus or wholly the new one. This hammers
    the two against each other; against the old code it fails.
    """
    import threading

    from openknowledge.retrieval import BM25Retriever
    from openknowledge.retrieval.base import Document

    def corpus(marker: str, size: int) -> list[Document]:
        return [
            Document(
                f"policy-{i}",
                f"Policy {i}",
                f"Expenses are reimbursed up to EUR {marker} per day. "
                f"Clause {i} applies to travel and subsistence.",
            )
            for i in range(size)
        ]

    retriever = BM25Retriever()
    retriever.index(corpus("40", 40))
    stop = threading.Event()
    seams: list[str] = []

    def rebuild() -> None:
        marker = 40
        while not stop.is_set():
            marker = 75 if marker == 40 else 40
            retriever.index(corpus(str(marker), 40))

    def read() -> None:
        while not stop.is_set():
            # A reader that dies takes its evidence with it. Against the old
            # in-place rebuild these threads raised IndexError and vanished,
            # leaving the assertion below with an empty list and the test
            # passing for the worst possible reason.
            try:
                hits = retriever.search("expenses reimbursed per day", k=5)
            except Exception as exc:  # noqa: BLE001 - the failure under test
                seams.append(f"{type(exc).__name__} during search: {exc}")
                continue
            for hit in hits:
                # Every chunk must be a whole one from some corpus, and its
                # own text - a mismatch here is the citation bug.
                if "Expenses are reimbursed" not in hit.chunk.text:
                    seams.append(f"torn chunk: {hit.chunk.text[:40]!r}")
                if (
                    hit.chunk.document_id.removeprefix("policy-")
                    not in hit.chunk.text.split("Clause ")[-1]
                ):
                    seams.append(f"chunk scored under another's id: {hit.chunk.document_id}")

    writer = threading.Thread(target=rebuild, daemon=True)
    readers = [threading.Thread(target=read, daemon=True) for _ in range(3)]
    writer.start()
    for r in readers:
        r.start()
    threading.Event().wait(2.0)
    stop.set()
    writer.join(timeout=10)
    for r in readers:
        r.join(timeout=10)

    assert not seams, f"{len(seams)} inconsistent reads during rebuild: {seams[:3]}"
