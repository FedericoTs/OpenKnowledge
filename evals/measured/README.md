# Measured token counts

`tools/cost_model.py` used to price this architecture entirely from assumed token
counts. The assumptions turned out to be wrong in three compounding ways, and the
only way to find that out was to measure a real corpus.

`real-contracts.json` is the output of

```bash
uv run python tools/measure_prompts.py --corpus <folder> --questions <file> --json
```

run against **15 real third-party vendor contracts, SLAs, DPAs and registers** —
around 100 pages, plus a parallel copy of seven of them. The documents are
third-party and are not redistributed here; what is committed is the derived
token counts, so the cost model can be fed by measurement instead of by
guesswork and anybody can see which numbers are which.

## What it corrected

| assumption | was | measured |
|---|---:|---:|
| Cacheable system prompt | 2,000 tokens | **476 tokens, under the 512 floor — caches nothing** |
| Prompt at 6 chunks | 4,500 tokens | **2,313 tokens** |
| Answer length after tightening retrieval | 400 tokens | held at 1,000 — retrieval does not shorten answers |

The first two partly cancelled; the third was a methodology error, giving the
retrieval row credit for a 60% cut in output tokens that retrieval does not
cause. Corrected, the headline goes from 19× to **11×**, which is lower and
true.

## Replacing it with your own

Point it at your own folder and your own questions. The numbers here describe a
corpus of vendor contracts; a corpus of HR policies will chunk differently and
retrieve differently, and yours is the one your invoice will be based on.

`tools/cost_model.py` reads this file if it is present and falls back to
documented assumptions if it is not — and says which it used, on every run.
