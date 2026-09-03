"""One asker cannot spend everybody else's day.

The budget governor already stops a flood becoming an invoice: the ceiling it
computes is *remaining budget / questions still expected*, so a thousand
questions in an hour lower what any one question may cost rather than running
the bill up. What it cannot do is decide **whose** questions those were.

That is the gap this closes. A looping bot integration, a colleague who
discovered they can paste a spreadsheet into the chat, a retry storm behind a
proxy - each of them drags the shared ceiling down for everyone, and the first
anybody notices is that answers got worse for the whole company at eleven on a
Tuesday. A per-asker limit turns "everyone is degraded" into "one caller is
told to slow down".

**It keeps no record of who asked what.** The counters live in this process's
memory, are keyed by a salted hash of the asker rather than by the asker, and
are gone when the process restarts. That is the same promise the gaps report
and the reported-answers table make, and it is made the same way: not by
policy but by there being nowhere for the data to go. Enforcing a limit needs
to know that *this* caller has asked twelve times in the last minute; it never
needs to know who they are, and it must not become the log that says so.

The salt is per-process and random, so the keys are not reversible from a heap
dump and do not correlate across a restart.
"""

from __future__ import annotations

import hashlib
import secrets
import threading
import time
from collections import deque
from dataclasses import dataclass

#: Above this many distinct askers in the window, sweep the stale ones. Chosen
#: to be far past any real deployment's concurrent askers, so the sweep is
#: amortised to approximately never rather than run on every question.
_SWEEP_ABOVE = 4096


@dataclass(frozen=True, slots=True)
class Decision:
    """Whether this may proceed, and what to tell whoever asked."""

    allowed: bool
    #: What this caller has spent inside the window, including this request.
    #: Questions, for the asker limit; bytes, for the upload limit.
    asked: int
    #: Seconds until enough falls out of the window to make room. 0 when allowed.
    retry_after: float = 0.0


class AskerLimiter:
    """How much one caller may spend per window.

    A sliding window of timestamps rather than a fixed bucket: a fixed bucket
    lets a caller spend the whole limit at 10:59:59 and the whole limit again
    at 11:00:00, which is the burst the limit exists to stop.

    What is counted is whatever ``cost`` says. Questions cost one each, which
    is what this class was written for. Uploaded bytes cost their own size,
    because a limit on upload *requests* would be a limit on nothing: one
    request carries as many files as the client cares to attach.

    ``per_minute`` of zero disables it entirely, which is the right default for
    a desktop install where the only asker is the person whose laptop it is.
    """

    def __init__(self, per_minute: int, *, window_seconds: float = 60.0) -> None:
        self.per_minute = max(0, int(per_minute))
        self.window_seconds = window_seconds
        self._salt = secrets.token_bytes(16)
        self._seen: dict[str, deque[tuple[float, int]]] = {}
        self._lock = threading.Lock()
        #: Questions refused by this limiter since the process started. The one
        #: number an operator wants when answers went strange for an hour.
        self.refused = 0

    @property
    def enabled(self) -> bool:
        return self.per_minute > 0

    def key(self, who: str) -> str:
        """The asker, as something this process can count but not read back."""
        return hashlib.blake2b(who.encode("utf-8"), key=self._salt, digest_size=16).hexdigest()

    def check(self, who: str, *, cost: int = 1, now: float | None = None) -> Decision:
        """Charge ``cost`` to this caller and say whether it may proceed."""
        if not self.enabled or cost <= 0:
            return Decision(allowed=True, asked=0)

        moment = time.monotonic() if now is None else now
        cutoff = moment - self.window_seconds
        key = self.key(who)
        with self._lock:
            if len(self._seen) > _SWEEP_ABOVE:
                self._sweep(cutoff)
            window = self._seen.setdefault(key, deque())
            while window and window[0][0] <= cutoff:
                window.popleft()
            spent = sum(charge for _, charge in window)
            if spent + cost > self.per_minute:
                self.refused += 1
                return Decision(
                    allowed=False,
                    asked=spent,
                    retry_after=_when_there_is_room(window, spent, cost, self.per_minute, cutoff),
                )
            window.append((moment, cost))
            return Decision(allowed=True, asked=spent + cost)

    def _sweep(self, cutoff: float) -> None:
        """Forget askers with nothing left in the window. Called under the lock."""
        stale = [key for key, window in self._seen.items() if not window or window[-1][0] <= cutoff]
        for key in stale:
            del self._seen[key]

    def forget_everyone(self) -> None:
        """Drop every counter. What a restart would do, without one."""
        with self._lock:
            self._seen.clear()


def _when_there_is_room(
    window: deque[tuple[float, int]], spent: int, cost: int, ceiling: int, cutoff: float
) -> float:
    """Seconds until enough of the window expires to fit ``cost``.

    With every charge worth one, this is when the oldest leaves - which is
    what it used to say. With charges of different sizes it is not: a caller
    who spent their minute on one large file waits for that file, and a
    caller who spent it on forty small ones waits only for as many as it
    takes to make room. Saying "try again in a second" when the room will
    not exist for fifty is worse than saying nothing.
    """
    freed = 0
    for moment, charge in window:
        freed += charge
        if spent - freed + cost <= ceiling:
            return max(0.0, moment - cutoff)
    # Nothing this window can free is enough: the request is larger than the
    # whole allowance. The caller needs a bigger limit, not a longer wait.
    return 0.0
