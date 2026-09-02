"""The unit of parsed content.

Every format - PDF, Word, Excel, PowerPoint, Markdown - reduces to an ordered
list of blocks, each carrying three things beyond its text:

``kind``
    Whether it is a heading, a paragraph, a list item, or a table row. Chunking
    uses this to avoid splitting things that only mean something whole.

``heading_path``
    The trail of headings above it: ``("Expenses Policy", "Approval thresholds")``.
    This is what turns a bare sentence into a locatable claim. "Above EUR 500"
    is ambiguous; "Expenses Policy > Approval thresholds: above EUR 500" is not,
    and retrieval can match on the heading words too.

``locator``
    Where it physically came from - ``p. 3``, ``Sheet1!A7``, ``slide 4``. This
    ends up in the citation, so an employee can open the source and check. A
    citation that cannot be checked is decoration.

Table rows are the case that justifies the whole model. Internal policy keeps
its thresholds in tables, and a table flattened to prose gives you
``Grade 3 500 60`` - which the numeric claim extractor will happily read as
three unrelated figures. Carrying the header with every row keeps the row a
sentence.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import StrEnum

_WHITESPACE = re.compile(r"[ \t ]+")
_BLANK_LINES = re.compile(r"\n{3,}")


class BlockKind(StrEnum):
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    LIST_ITEM = "list_item"
    TABLE_ROW = "table_row"
    CAPTION = "caption"

    @property
    def is_atomic(self) -> bool:
        """Whether splitting this block would destroy its meaning.

        A table row split down the middle is worse than useless - it produces a
        number with no label attached to it.
        """
        return self in (BlockKind.TABLE_ROW, BlockKind.HEADING, BlockKind.LIST_ITEM)


@dataclass(frozen=True, slots=True)
class Block:
    kind: BlockKind
    text: str
    heading_path: tuple[str, ...] = ()
    locator: str | None = None
    #: Heading depth, 1 being the document title. Zero for non-headings.
    level: int = 0

    @property
    def contextual_text(self) -> str:
        """The block with its heading trail, as retrieval and the model see it.

        Headings are not repeated onto themselves, and a table row keeps its own
        already-labelled form.
        """
        if self.kind is BlockKind.HEADING or not self.heading_path:
            return self.text
        return f"{' > '.join(self.heading_path)}: {self.text}"


@dataclass(frozen=True, slots=True)
class ParsedDocument:
    """What a parser returns."""

    blocks: tuple[Block, ...] = ()
    title: str | None = None
    pages: int = 0
    #: Problems worth telling an operator about - a scanned PDF, a password, a
    #: sheet that could not be read. Surfaced rather than swallowed: a document
    #: that silently contributes nothing is how a corpus develops holes nobody
    #: notices until an answer is wrong.
    warnings: tuple[str, ...] = field(default_factory=tuple)
    #: The flattened text, when somebody has already computed it.
    #:
    #: Flattening is not free: ``normalise`` makes six full passes over the
    #: document, and on a corpus of 1,200 files that was 1.09 of the 1.54
    #: seconds every upload and every delete spent re-reading a folder in
    #: which one file had changed. It is a pure function of the blocks, so the
    #: parse cache stores it beside them and hands it back here.
    #:
    #: Excluded from equality on purpose. A document read from the cache
    #: carries it and a freshly parsed one does not, and those two are the
    #: same document - the tests that assert a cached parse equals an uncached
    #: one are asserting something true, and comparing a memo would make them
    #: fail for a difference that is not one.
    flattened: str | None = field(default=None, compare=False, repr=False)

    @property
    def text(self) -> str:
        """Flattened plain text, for anything that just wants the words."""
        if self.flattened is not None:
            return self.flattened
        return normalise("\n\n".join(b.contextual_text for b in self.blocks))

    @property
    def is_empty(self) -> bool:
        return not any(b.text.strip() for b in self.blocks)


#: How far into a document a self-declaration of supersession is looked for.
#: Status lines live in the header block; a document whose *body* discusses a
#: superseded policy is talking about some other document, not itself.
_SUPERSEDED_HEAD_CHARS = 800

#: Ways documents say "do not use me". Deliberately past-tense and passive:
#: "Supersedes: v3.0" is what the *current* copy says and must not match,
#: while "SUPERSEDED by v4.1", "Status: superseded" and "Retained for audit
#: only" are all the retired copy speaking. The status form tolerates the
#: markdown bold and colons metadata lines are written with.
_SUPERSEDED_RE = re.compile(
    r"""
    \bsuperseded\s+by\b
    | \bstatus\b[\s:*_~-]{0,8}(?:superseded|obsolete|withdrawn|archived)\b
    | \bretained\s+for\s+(?:audit|reference|historical)\b
    | \bno\s+longer\s+in\s+(?:force|effect)\b
    | \b(?:this|the)\s+(?:document|policy|procedure|version)\s+
      (?:is|was|has\s+been)\s+(?:superseded|withdrawn|replaced|retired)\b
    """,
    re.IGNORECASE | re.VERBOSE,
)


def declares_superseded(text: str) -> bool:
    """Whether this document says, near its top, that it has been retired.

    An archived policy that opens with "SUPERSEDED by Expenses Policy v4.1.
    Retained for audit only." has already made the versioning decision - no
    heuristic about dates or folder names is needed, and none is used. Only
    the head of the document is examined, so a current policy that *mentions*
    its superseded predecessor further down is not caught, and "Supersedes:"
    - the current copy naming what it replaced - does not match.
    """
    return bool(_SUPERSEDED_RE.search(text[:_SUPERSEDED_HEAD_CHARS]))


def normalise(text: str) -> str:
    """Collapse the whitespace damage that extraction leaves behind.

    PDF text layers in particular arrive with runs of spaces where the original
    had columns, and non-breaking spaces where it had none.
    """
    text = text.replace(" ", " ").replace("\r\n", "\n").replace("\r", "\n")
    text = _WHITESPACE.sub(" ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    return _BLANK_LINES.sub("\n\n", text).strip()


def table_row_text(headers: list[str], cells: list[str]) -> str:
    """Render one table row so it still means something on its own.

    ``["Grade", "Limit", "Notice"]`` + ``["3", "EUR 500", "60 days"]`` becomes
    ``Grade: 3 | Limit: EUR 500 | Notice: 60 days``.

    Without the labels this row reads as three unrelated numbers, and the claim
    extractors - which are how contradictions and invented figures get caught -
    have nothing to attach them to. Unlabelled columns fall back to the bare
    value rather than inventing a header.
    """
    parts: list[str] = []
    for i, cell in enumerate(cells):
        value = normalise(str(cell)) if cell is not None else ""
        if not value:
            continue
        header = normalise(str(headers[i])) if i < len(headers) and headers[i] else ""
        parts.append(f"{header}: {value}" if header else value)
    return " | ".join(parts)


def looks_like_header_row(cells: list[str]) -> bool:
    """Whether a row reads as column labels rather than data.

    Deliberately conservative: a wrong guess mislabels every row beneath it, so
    it takes a row of short, non-numeric, mostly-populated cells.
    """
    values = [normalise(str(c)) for c in cells if c is not None and str(c).strip()]
    if len(values) < 2:
        return False
    if any(any(ch.isdigit() for ch in v) for v in values):
        return False
    return all(len(v) <= 40 for v in values)
