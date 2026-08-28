"""Wall-clock timestamps that are safe to compare for order.

``time.time()`` on Windows (CPython 3.12) advances in ~15.6 ms ticks, so two
writes made back to back routinely land on the same timestamp - and the
stale-pin check compares exactly such neighbours: a pin's ``updated_at``
against a conflict's ``detected_at``. On Linux the clock resolves finely
enough that the tie never showed; the Windows CI leg caught it as a test
that passed twice and then failed, which is a coin flip on where inside a
clock tick two calls fall.

Anything whose timestamp participates in an ordering decision takes its
time from here: real wall-clock time, bumped by a microsecond whenever the
clock has not visibly moved since the previous call, so timestamps issued
by one process never tie. Ties across processes remain physically possible
and are handled where the comparison happens - the router treats a tie as
"cannot prove the pin was written with the conflict in view" and withholds.

Timestamps that are only ever displayed or summed (the cost ledger, contact
submissions) keep plain ``time.time()``; a synthetic microsecond would be a
lie there and prevents nothing.
"""

from __future__ import annotations

import threading
import time

_lock = threading.Lock()
_last = 0.0


def ordered_now() -> float:
    """The current time, strictly greater than any this process returned before."""
    global _last
    with _lock:
        now = time.time()
        if now <= _last:
            now = _last + 1e-6
        _last = now
        return now
