# Contributing

## Getting set up

```bash
git clone https://github.com/FedericoTs/OpenKnowledge
cd OpenKnowledge
make install
make check     # lint, typecheck, tests
```

## Before opening a pull request

- `make check` passes.
- New behaviour has a test. New *claims* have a test — if a change makes something cheaper
  or more accurate, the test should demonstrate it.
- Commits are signed off (`git commit -s`), per [CLA.md](CLA.md).

## Things worth knowing before you change them

**`canonical.py` is conservative on purpose.** Stemming, lemmatisation, or a stopword list
would raise the cache hit rate and introduce a correctness bug: "which expenses are *not*
reimbursable" must never collapse into "which expenses are reimbursable". `NEVER_STRIP` and
`tests/test_canonical.py` exist to keep that a deliberate decision. If you want to close the
paraphrase gap, the semantic cache tier is the place — it can check citations before serving
a near-match, which a hash cannot.

**The grounding gate is what makes the cheap tiers safe.** Loosening it to raise the local
hit rate reverses the project's central trade-off. If a threshold is wrong, change it with
evidence — a failing case and a test.

**Prices are facts, not estimates.** Entries in `pricing.yaml` carry a `verified` date.
Do not add a number you have not read off the vendor's pricing page; leave the slot empty
and let `cost_usd()` raise. A ledger that quietly reports $0 for real spend is worse than one
that refuses to compute.

**Connectors must populate `allowed_principals`.** A connector that returns documents without
their ACLs turns a permission-aware system into a leak.

## Where help is most useful

See [ROADMAP.md](docs/ROADMAP.md). The highest-leverage items right now:

1. **Evaluation harness** — a golden set of question/answer/citation triples, run on every
   change, reporting accuracy *and* blended cost. Without it, "the local model is good
   enough" is an opinion.
2. **Hybrid retrieval + reranking** — the single biggest cost lever, and it improves accuracy
   at the same time.
3. **SharePoint / Google Drive connectors** — mostly ACL mapping, and it has to be right.
