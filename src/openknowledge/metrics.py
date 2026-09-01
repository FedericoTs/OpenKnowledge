"""What this install is doing, in the format a monitoring system already reads.

`/healthz` says the server is up. It does not say that spend tripled at eleven,
that a third of today's questions were refused, or that one caller has been
rate-limited four hundred times - which are the three things an operator needs
before they can act, and which until now could only be reconstructed by hand
from the cost report.

Prometheus text exposition, because it is the format every monitoring system
can scrape and a person can read with `curl`. Nothing here is new data: it is
the ledger, the index and the limiter, formatted so a graph can be drawn
without anybody writing a parser.

Two things it deliberately does not carry. No per-question labels - a metric
with the question in it is a log of what people asked, published to whatever
scrapes it. And no identity anywhere, for the same reason the gaps report and
the reported-answers table have none: the rate limiter counts askers without
naming them, and this reports the count.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

#: What a Prometheus scraper expects to be told it is reading.
CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"


@dataclass(frozen=True, slots=True)
class Sample:
    """One number, with the words a reader needs to know what it counts."""

    name: str
    help: str
    #: ``counter`` only ever goes up; ``gauge`` is whatever it is right now.
    kind: str
    value: float
    labels: tuple[tuple[str, str], ...] = field(default=())


def _escaped(value: str) -> str:
    """A label value as the exposition format requires it."""
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _line(sample: Sample) -> str:
    if not sample.labels:
        return f"{sample.name} {sample.value:g}"
    labels = ",".join(f'{key}="{_escaped(value)}"' for key, value in sample.labels)
    return f"{sample.name}{{{labels}}} {sample.value:g}"


def render(samples: Iterable[Sample]) -> str:
    """The exposition text for these samples.

    Samples sharing a name are one metric family with several label sets, and
    the format requires a family's lines to be **contiguous**: HELP, TYPE, then
    every one of its samples, before any other family starts. So this groups by
    name rather than writing samples out in the order it was handed them.

    Writing them in arrival order looks right and is not: interleaving two
    families - which is what happens the moment the caller adds a second window
    or a second label - makes a strict scraper reject the whole page as a
    duplicated family. Insertion order of names is kept, so the page still
    reads top to bottom the way the caller composed it.
    """
    families: dict[str, list[Sample]] = {}
    for sample in samples:
        families.setdefault(sample.name, []).append(sample)

    lines: list[str] = []
    for name, group in families.items():
        lines.append(f"# HELP {name} {group[0].help}")
        lines.append(f"# TYPE {name} {group[0].kind}")
        lines.extend(_line(sample) for sample in group)
    return "\n".join(lines) + "\n"


def _number(value: object) -> float:
    """A number out of a report entry, whatever it actually holds.

    The cost report is typed as ``dict[str, object]`` because it is assembled
    from SQL, and a metrics endpoint that raises on one odd row publishes
    nothing at all - which is worse than publishing a zero next to everything
    that is fine.
    """
    return float(value) if isinstance(value, int | float) else 0.0


def from_cost_report(report: dict[str, object], *, window: str) -> list[Sample]:
    """Questions and spend, per tier, out of what the ledger already knows.

    ``window`` labels the period so "today" and "since this install began" can
    be graphed against each other - the pair that answers "is this normal?".
    """
    by_tier = report.get("by_tier")
    tiers: Sequence[tuple[str, dict[str, object]]] = (
        sorted(by_tier.items()) if isinstance(by_tier, dict) else ()
    )
    samples: list[Sample] = []
    for tier, counts in tiers:
        if not isinstance(counts, dict):
            continue
        samples.append(
            Sample(
                "openknowledge_questions_total",
                "Questions answered, by the tier that answered them.",
                "counter",
                _number(counts.get("questions")),
                (("tier", tier), ("window", window)),
            )
        )
    for tier, counts in tiers:
        if not isinstance(counts, dict):
            continue
        samples.append(
            Sample(
                "openknowledge_spend_usd_total",
                "What was actually spent, by tier. Read from the ledger, not estimated.",
                "counter",
                _number(counts.get("spend_usd")),
                (("tier", tier), ("window", window)),
            )
        )
    return samples
