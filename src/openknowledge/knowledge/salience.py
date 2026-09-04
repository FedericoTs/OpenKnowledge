"""How much evidence a shared word actually is.

Contradiction detection asks whether two claims are about the same subject, and
answers it by measuring how much context they share. Counting words treats every
word as equal evidence, which is fine on a tidy policy corpus and wrong on real
documents: contracts, procedures and policies share an enormous boilerplate
vocabulary - *party*, *agreement*, *notice*, *written*, *service* - and two
sentences overlapping on nothing but that are not about the same subject.

Measured on 15 real vendor contracts, unweighted overlap produced 320 findings
and no true ones. Almost every pair was two different counterparties' boilerplate
scoring a high Jaccard on words that appear in every document in the folder.

So words are weighted by how rare they are across the corpus's own claims, in the
usual inverse-document-frequency shape, then normalised so the mean weight is
1.0. That last step is what keeps the thresholds interpretable: shared mass is
still measured in words, they are just no longer all worth one word each. Three
shared boilerplate words come to less than one; three shared topic words come to
several.

A corpus of two documents has no useful frequency statistics, so weights stay
near 1.0 there and the behaviour is what it always was. This is deliberate:
the correction should arrive with the evidence that justifies it.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class Salience:
    """Per-word evidence weights, normalised so the mean weight is 1.0."""

    weights: dict[str, float] = field(default_factory=dict)
    #: Remembered totals, per claim context. Not part of the value: two Salience
    #: objects with the same weights are the same object as far as anyone
    #: outside is concerned, whatever either has happened to compute.
    _totals: dict[frozenset[str], float] = field(default_factory=dict, repr=False, compare=False)

    def weight(self, word: str) -> float:
        """Unseen words are neutral: a word we have no statistics for is one word."""
        return self.weights.get(word, 1.0)

    def mass(self, words: Iterable[str]) -> float:
        """Total evidence in a set of words, in units of average words.

        Measured: this was 82% of the time it took to index a corpus, because
        contradiction detection compares every pair of claims and every
        comparison re-added the weights of both sides from scratch. A claim's
        own total does not depend on what it is being compared against, so it is
        worth remembering - 8.9 million calls at 300 documents, over a few
        thousand distinct claims.

        Two shapes are handled separately. With no weights at all every word is
        worth exactly one, so the total is the number of words and no addition
        is needed; that is the ``UNIFORM`` case, which every caller that
        supplies no corpus statistics uses. With weights, a context is a
        frozenset and hashes in constant time after the first look, so it can
        key a cache. Anything else - a generator, a list - is added up as
        before, since remembering it would mean consuming it.
        """
        if not self.weights:
            if isinstance(words, (frozenset, set)):
                return float(len(words))
            return float(sum(1 for _ in words))
        return sum(self.weight(w) for w in words)

    def total(self, context: frozenset[str]) -> float:
        """``mass``, remembered - for a claim's own context and nothing else.

        The distinction is the whole point. A claim's context is one of a few
        thousand in a corpus and is asked for once per comparison, so
        remembering it turns 8.9 million additions into a few thousand. An
        intersection or a union belongs to a *pair*, of which there are
        millions, and remembering those would trade the time for a dictionary
        the size of the comparison itself. The first version of this did
        exactly that, inside ``mass``, and would have grown to 2.9 million
        entries on a 300-document corpus.
        """
        if not self.weights:
            return float(len(context))
        total = self._totals.get(context)
        if total is None:
            total = sum(self.weight(w) for w in context)
            self._totals[context] = total
        return total

    def shared_mass(self, a: frozenset[str], b: frozenset[str]) -> float:
        return self.mass(a & b)

    def jaccard(self, a: frozenset[str], b: frozenset[str]) -> float:
        """Weighted Jaccard: shared evidence over total evidence."""
        union = self.mass(a | b)
        if union <= 0.0:
            return 0.0
        return self.mass(a & b) / union

    def coefficient(self, a: frozenset[str], b: frozenset[str]) -> float:
        """Weighted overlap coefficient: shared evidence over the smaller side.

        The asymmetric form, used where one sentence may legitimately be much
        longer than the other - a rule and its restatement in a summary.
        """
        smaller = min(self.total(a), self.total(b))
        if smaller <= 0.0:
            return 0.0
        return min(1.0, self.mass(a & b) / smaller)


#: Every word worth exactly one word. What the detectors did before, and what
#: they still do when nobody supplies corpus statistics.
UNIFORM = Salience()


def salience_from(contexts: Iterable[frozenset[str]]) -> Salience:
    """Build weights from the context windows of every claim in a corpus.

    Frequency is counted over *claims* rather than documents because that is the
    unit being compared. A word in every claim of a 200-page contract is
    boilerplate whether or not it also appears in the file next to it.
    """
    windows = [c for c in contexts if c]
    total = len(windows)
    if total < 4:
        # Too few samples for a frequency to mean anything. Staying uniform is
        # more honest than weighting on noise.
        return UNIFORM

    document_frequency: dict[str, int] = {}
    for window in windows:
        for word in window:
            document_frequency[word] = document_frequency.get(word, 0) + 1

    raw = {word: math.log(total / df) + 1.0 for word, df in document_frequency.items()}
    mean = sum(raw.values()) / len(raw)
    if mean <= 0.0:  # pragma: no cover - log(total/df) + 1 is always positive
        return UNIFORM
    return Salience({word: value / mean for word, value in raw.items()})
