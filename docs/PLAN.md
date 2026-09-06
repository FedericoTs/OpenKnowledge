# The plan

Six improvements, in priority order, each grounded in something measured. Written
2026-09-05 after the `/manage` gap report on a real install produced sixteen
unanswered questions, and the analysis in
`evals/measured/fortyfifth-the-answer-that-was-a-whole-document.json`.

The order is impact times feasibility, with one hard dependency: nothing in P2
can honestly be shipped before P1 exists to measure it, because the project's
own rule is that nothing ships unmeasured and the failure P2 addresses cannot
currently be measured at all.

| # | what | evidence | size | depends on |
|---|---|---|---|---|
| P1 | An eval corpus with documents larger than the retrieval budget | 76 golden questions, 0 enumerations; every eval document is 1–2 chunks | small | — |
| P2 | Whole-document answering | 32% ceiling at k=6; k=50 reaches 89%; 11 of 16 field questions | medium–large | P1 |
| P3 | Apply a stated rule to a figure in the question | inj-07 and golden-ftr, independently; measured at 28.6% | medium | — |
| P4 | The gap report clears rows that were fixed | 4 of 16 rows were already answered free | small | — |
| P5 | Typo tolerance | 2 of 16 field questions misspelled; unmeasured | small | — |
| P6 | One real Entra tenant | ~2,500 lines of auth and connectors have only met a fake | blocked on the owner | — |

Deferred and named: the quadratic index (`fortyfirst-what-happens-at-a-thousand.json`)
bites at thousands of documents, and a company server has hundreds. It stays
below all of these.

---

## P1 — A corpus the failure can be measured on

**Status: built.** `evals/golden-scope/scope.yaml`, ten cases over four documents;
`must_list`/`min_share` in the scorer; the converter; the pins. See the set's
README for what each case is and where its reference answer comes from.

**Evidence.** Every document in `evals/corpus/aveline` is one or two chunks
against a budget of six, so the whole corpus fits the context window and an
enumeration cannot fail. Across `golden`, `golden-aveline`, `golden-ftr` and
`golden-injection` — 76 questions — not one asks for a list, a summary, or
"all of" anything. The evals measure the shape of question the system is good
at.

**Design.** `evals/golden-scope/` becomes a self-contained, model-in-the-loop
set, alongside the modelless ceiling tool that already lives there.

- *Corpus*, none of it written here: the FTR glossary (`ftr-300-1`, 82 terms,
  22 chunks) and the transport-methods part (`ftr-301-10`) copied byte-for-byte
  from `golden-ftr` — a test pins the copies; and two Project Gutenberg texts,
  *The Importance of Being Earnest* and *Alice's Adventures in Wonderland*,
  converted to Markdown by `tools/gutenberg_to_md.py` so their own chapter, act
  and cast-list headings become headings. The converter is committed and
  deterministic; it changes structure markers, never a word of the text.
- *Reference answers from the documents themselves.* The cast list Wilde
  printed. The twelve chapter titles Carroll wrote. The terms the regulation
  defines. Nothing in the exam is a list somebody here made up.
- *Shapes*: enumeration inside the budget (control, must stay 100%),
  enumeration far outside it, summary (an act, a part), ordinal ("the second
  chapter"), and refusals with the same vocabulary ("the fourth act" of a
  three-act play).
- *Scorer*: a new `must_list` field with `min_share`. `must_say` demands every
  fact; an 82-item list scored that way is 82 impossible requirements. The
  scorer reports *listed 26 of 82 (32%), needed 90%*, which is the number that
  has been missing.

**Proof.** A baseline run before P2 lands, recorded. The control at 100%; the
glossary case failing with its share stated.

**Risk.** The summary cases name entities a summary must mention; that is
judgement, kept minimal and drawn from the text.

## P2 — Whole-document answering

**Status: done and measured.** With a real model over the whole set, accuracy
went 42.9% to 100%, false answers 1 to 0, determinism 90% to 100%, paraphrase
consistency 75% to 100%, and the run 58 minutes to 14 because half the set no
longer calls a model. The one false answer before the change was an invented
fourth act of a three-act play, which the grounding gate passed at 96% support -
higher than its mean on answers that passed. Full numbers in
`evals/measured/fortysixth-the-document-that-answered-for-itself.json`. Structure answers the seven cases it can — the
cast list, the chapters, the 82 terms, the forty sections, an ordinal into each,
and two ordinals past the end refused with the count — at 100%, 0 false answers,
$0, deterministically, and CI runs exactly those. Assembly serves the summary
shape within the window. Not built: map-reduce over windows for a document the
window cannot take, and a completeness notion in the grounding gate.

**Evidence.** Asked to name the 82 terms a 22-chunk glossary defines, retrieval
at the shipped `k=6` shows the model one chunk and 26 terms. At `k=50` — eight
times the budget, more than half the corpus — 89%, because BM25 ranks by term
overlap and the question shares no vocabulary with the terms it is asking to
enumerate. Raising `k` is measured and does not work.

**Constraint found on the way.** The local model runs an 8,192-token window;
the provider refuses anything over 90% of it; the system prompt is ~1,070
tokens. About ten chunks fit. A 22-chunk document cannot be assembled into the
local model even in principle, so "retrieve the whole document" is not the
whole answer either.

**Design.** Three parts, in `src/openknowledge/cascade/scope.py` and the router.

1. *Recognise the shape, conservatively.* A question is whole-document-shaped
   when it asks to list, name all, summarise, or say what something covers —
   and retrieval concentrates on one document (at least half the top hits from
   it, or the question names its title). Either signal alone changes nothing:
   "how much is the meal allowance" is never assembled. The existing free tier
   for questions *about the collection* runs first and is untouched.
2. *Answer structured enumerations from structure, free.* When the target
   document's parsed blocks carry the structure asked for — headings for "what
   does it cover / what are the chapters", list-item runs for "what are the
   priorities / the steps", term-definition paragraphs for "what terms does it
   define", a short-line block under a persons/characters heading for "who are
   the characters" — the answer is read from the blocks. No model, $0,
   deterministic, and correct by construction in the same sense the corpus tier
   is. A new `Tier.OUTLINE` so the ledger and the UI say what happened. The
   assembled passages are the citation, so the answer is gate-visible and
   reads like every other. An ordinal — "what is priority 2", "the second
   chapter" — indexes the same list.
3. *Assemble for summary.* When the shape is summary and no structure answers
   it, replace the six ranked chunks with the target document's chunks in order,
   up to a budget computed from `local_context_tokens`, the system prompt, the
   question and `max_answer_tokens`. When the document exceeds the budget, the
   leading prefix is used and the answer's notes say *read sections 1–N of M*
   — an honest partial rather than a silent one. Map-reduce over windows is
   the next step and is recorded as not built.

**Proof.** `golden-scope` before and after; the four existing golden sets
unchanged — accuracy and false answers must not move, because a false-positive
shape match sends a fact question to assembly. The ceiling tool measures
`search()` and is unaffected. Sabotages: the recogniser off, the concentration
test off, the budget ignored.

**Risk.** Shape recognition is vocabulary, and vocabulary lists were the
recurring hole in the corpus tier. Two votes rather than one — shape *and*
concentration — is the mitigation, and the four golden sets are the test.

## P3 — Apply a stated rule to a figure in the question

**Status: shipped in v0.12.9, and the prediction below was wrong.** 16 cases
over the aveline corpus - 7 interior, 7 boundary, 2 refusal - in
`evals/golden-rules/`, with `tests/test_golden_rules.py` checking every
requirement and prohibition against a written-out correct answer. The A/B
harness runs the set plus the refusal half of all four existing corpora -
**32 cases**: 12 from golden-ftr, 9 from aveline, 6 from injection, 5 from
golden - because a change that makes the system readier to compare numbers is
a change that could make it readier to answer what it should refuse.

(Commit `6d2cfcf` and its message say 33. That count came from grepping
`kind: refusal`, which also matched the sentence in golden.yaml's header
explaining what the field means. The loader says 5 refusals there, not 6.)

**Evidence.** `inj-07`: asked whether a EUR 40,000 purchase needs quotes, the
system refuses, though the threshold is plain in the document and retrieval
finds it. It failed in all three arms of the injection control, so it is not
injection's doing. `golden-ftr` found the same gap independently: the system
states a threshold when asked what the threshold is and will not apply it.

**A prediction, written before the baseline ran.** Reading `SYSTEM_PROMPT`
suggests the refusal is instructed rather than incidental. Rule 3 ends: *"If a
figure the question asks for is not in the sources, say so instead of
estimating."* For "do we need quotes for a EUR 40,000 contract?", the figure in
the question genuinely is not in the sources - 40,000 appears nowhere in the
policy - so that sentence, read literally, tells the model to refuse. The
paragraph that handles choosing between conditional figures reinforces it: it
licenses picking a band when the question names "a particular grade, tier,
location, duration or category", and a monetary amount is not on that list.

If that is right, the baseline will show refusals rather than wrong bands, and
the minimal change is to say that comparing a figure in the question against a
threshold in the sources is not inventing a number. If the baseline instead
shows confidently wrong bands, this hypothesis is wrong and the fix is a
different one. Recorded here so the answer cannot be fitted to the result
afterwards.

**What the baseline showed, and why the prediction was half right.** 28.6%,
with 9 refusals and 2 wrong answers - both wrong ones at exactly EUR 25,000,
each placed one band too high. The shape was right; "rather than" was not. The
prompt addition the prediction called for was then written and measured on its
own: **28.6% → 35.7%**, one case in fourteen, with the tier distribution
byte-identical and all nine refusals unchanged. That is inside the noise of a
4B model on four CPU threads, so it failed criterion 2 below and was **not
shipped**.

**The actual cause was the gate, not the prompt.** `check_grounding` rejects
any figure that appears in no source - the rule that stops a model inventing
numbers - and the asker's own figure appears in no source by definition. So
"EUR 40,000 is above EUR 25,000" was an invention and the cascade turned the
model's correct answer into "I don't know". Measured model-free by gating
answers written by hand: 6 of 6 that avoid the asker's figure pass, 6 of 6
that state it are rejected.

The fix admits the question's figures as evidence, as the passage headers
already were - but only in an answer that **also** states a figure from the
sources. Admitting them outright let a leading question write a number into
policy: "is the limit EUR 87,500?" answered "yes, the limit is EUR 87,500"
went from rejected to accepted. Comparing, not parroting. Grounding-policy
revision `g4 → g5`.

| 14 scored cases, local model, BM25 | before | after |
|---|---:|---:|
| accuracy | 28.6% | **57.1%** |
| refused | 9 | **4** |
| false answers | 0 | **0** |
| determinism | 100% | 93.8% |

**What counts as green, decided before the after arm ran.**

1. **The guard is absolute.** Zero false answers across all 32 refusal cases,
   and zero across `golden-rules`' own two. A false answer fails the change
   outright, whatever it did for accuracy: this product's position is that a
   confident wrong answer about policy is worse than a refusal, and a change
   that trades refusals for answers is exactly the change most likely to break
   it.
2. **The rules set must improve materially**, not by one case. A one-case move
   over sixteen is inside the noise of a model this size, and the honest
   response to it is another arm rather than a release.
3. **Boundary cases carry more weight than interior ones.** Getting "EUR 40,000
   is above EUR 25,000" right while getting "exactly EUR 25,000" wrong is a
   system that pattern-matches to the nearest band, which is the failure the
   set was built to expose.
4. The full suite passes, and `PROMPT_VERSION` moved so no answer cached under
   v5 is reused.

Criterion 1 held at 0 of 32 - **after** the guard caught a false answer that
was already shipped, in v0.12.8's outline tier rather than in this change:
"Does the parental leave policy cover adoption?" answered with the policy's
section list, in a corpus that never mentions adoption. The outline path
returns before any gate call, so the grounding fix could not have caused it or
caught it. Fixed in the same release. Criterion 2 held at +28.5 points.
Criterion 3 did **not**: both exactly-EUR-25,000 cases are still wrong, which
is why P3 is not finished. Criterion 4 held; the change is in the cache key,
not the prompt, so `g4 → g5` is the version that moved.

**Proof.** `evals/measured/fortyeighth-the-figure-the-asker-typed.json`, and
`tests/test_question_figures.py` pins both halves of the rule and the router
forwarding the question at all.

**Still open.** `rule-07`, which now reaches the right conclusion and opens
with the wrong one, and `rule-05`, which refuses a question the corpus covers.
Two further failures are the exam's fault and are recorded
as such: `rule-03` cites `finance-approval-limits` where `must_cite` demands
`finance-procurement-policy`, though both state the same bands, and `rule-06`
is refused as contested because the corpus genuinely disagrees about the
expense threshold - 500, 1,000, and a superseded 300.

### The boundary arm, registered before it ran

**What it changes.** One paragraph appended to rule 3, and nothing else.

> Read a threshold exactly as the source words it. "Above X" and "more than X"
> do not include X; "up to X", "at least X", and a band written "X to Y"
> include both ends. A figure that sits exactly on a boundary belongs to the
> band whose wording admits it, not to the next one up. Say which wording you
> relied on.

**Why only half of the reverted text.** The reverted paragraph had two halves.
The first said comparing a figure against a threshold is not inventing a
number; the gate fix shipped in v0.12.9 does that job in code, and in the
prompt-only arm that half moved none of the nine refusals. So it is dropped.
The second half is the boundary reading, which is the half that fixed rule-08.
It is also rewritten rather than restored: the corpus states its bands as
"EUR 5,001 to EUR 25,000", a form the reverted text never named, and rule-08
turns on exactly that form. 77 tokens against the reverted paragraph's 189, in
an 8,192-token window the system prompt already takes 1,073 of.

**The failure it targets, quoted from the 57.1% arm.** Both cases put 25,000 in
the band above it, and rule-07 does so while quoting the band that excludes it:

- rule-07: *"Yes, a contract with an annual value of exactly EUR 25,000 needs
  three competitive quotes. This is specified in the procurement policy under
  the section for contracts valued between EUR 25,001 and EUR..."*
- rule-08: *"The approval ... is required from the **Chief Financial
  Officer**"*, where the table reads "EUR 5,001 to EUR 25,000 | Head of
  Department".

**Hypothesis.** This is a boundary-reading failure, not a retrieval or
comparison one - the right passage is in front of the model in both cases and
rule-07 even quotes it. So naming how each boundary wording is read should move
rule-07 and rule-08 and leave the other twelve where they are.

**What counts as green.**

1. **The guard is absolute**, as before: zero false answers across all 32
   refusal cases and across `golden-rules`' own two. Telling a model how to
   read a threshold makes it readier to answer, which is exactly the change
   that could break a refusal.
2. **At least one of rule-07 and rule-08 must become correct.** This arm is
   targeted at two named cases; moving neither is a rejection, and the attempt
   gets recorded rather than reworded until it passes.
3. **No regression.** None of the eight cases passing at 57.1% may break, and
   accuracy may not fall.
4. **The right answer for the right reason.** A case counts only if the answer
   shows the boundary reasoning - names the wording it relied on, or places the
   figure in the band that admits it. Landing on "Head of Department" while
   quoting the 25,001 band is the failure this arm is about, not a pass.
5. The full suite passes and `PROMPT_VERSION` moves v5 -> v6.

**What this arm cannot establish.** Two named cases is weak evidence whatever
happens. Pre-naming them, and criterion 4, are what separate it from fitting a
prompt to a score; it is not a substitute for a boundary set large enough to
carry a number, and no percentage from it should be quoted as one.

**What it did.** 57.1% -> 71.4%, refusals 4 -> 3, determinism 93.8% -> 100%,
guard 0 of 32 with all four sets reporting. Every criterion above was met, so
the change ships.

| 14 scored cases | gate fix | + boundary |
|---|---:|---:|
| accuracy | 57.1% | **71.4%** |
| refused | 4 | **3** |
| determinism | 93.8% | **100%** |
| false answers, 32 refusal cases | 0 | **0** |

Two cases moved and only one of them is evidence. `rule-08` was named in
advance and is the mechanism's own target; it now answers *"The threshold is
exactly at EUR 25,000. The policy states that for values up to and including
EUR 25,000, the approver is the Head of Department"* - criterion 4 in the
answer's own words. `rule-14` also flipped, but it was not named, and it was
already giving two different answers when asked twice; a case that was
flipping anyway flipping again is not a result, and counting this arm as +2
would be counting noise.

**`rule-07` still fails, and the failure changed shape.** The boundary rule
worked on it: the answer quotes the 25,001 band, notices it starts at 25,001,
checks the second document, and concludes *"no three competitive quotes are
required for a contract valued at exactly EUR 25,000."* But it **opens** with
*"Yes, a contract with an annual value of exactly EUR 25,000 needs three
competitive quotes."* It leads with a guess and corrects it five paragraphs
later, so a person reading the first line is told the opposite of the answer.
The exam missed it too - none of `must_say`'s four phrasings match "does NOT
require" - but the self-contradiction is the real defect and the exam wording
is not why this case should fail. Answering before reasoning is a second
change and this arm is one change.

**A quotation that was not one, found in the case this arm fixed.** `rule-08`'s
answer presents, in the table's exact column format and inside quotation marks,
a row attributed to `finance-procurement-policy`:

> "Contract value (annual): Above EUR 25,000 | Approver: Chief Financial
> Officer | Additional requirement: Three competitive quotes"

There is no such row. "Above EUR 25,000" occurs in section 2 as prose about
quotes and is never paired with an approver; the answer welds it to the table's
CFO row. Every element is real somewhere in the document, so a support check
over word and number overlap cannot see the seam - the same hole as the
invented fourth act in `fortyfifth-...`. The fabricated row is semantically
true and the conclusion drawn from it is correct, which makes it a false
attribution rather than a false fact: the harder kind to notice, and the kind a
reader checking the citation would catch. Not caused by this change, and not
fixed here - the gate has never checked that a quotation is a quotation, and
adding that is a change of its own with its own false-positive risk on
paraphrase.

Full record: `evals/measured/fortyninth-the-answer-that-contradicted-its-first-sentence.json`.

**Risk.** Prompt changes have cost refusals before, and were reverted for it
(`fortysecond-the-fix-that-cost-a-refusal.json`). Hence the set first, the
change second, and one change at a time - which is why the prompt arm and the
gate arm were run separately, and only one of them shipped.

## P4 — The gap report clears what was fixed

**Status: built.** `src/openknowledge/gaps.py` re-tries every row against the
free tiers - the corpus recogniser, the assistant safety net, and a document's
own structure - and the CLI and `/manage` both separate rows that are answered
now from rows that are still open. Found on the way, by a test written for
this: the whole-document target vote demanded two agreeing passages, so a
one-document install - the desktop app's commonest shape - could never reach
the outline tier at all. Unanimity now counts.

**Evidence.** Four of the sixteen rows were answered free at the time of
reading; they had been fixed on 2026-08-29. The query clears a row only when
the question is asked again and succeeds, and nobody re-asks a question that
failed once.

**Design.** At report time, each row is re-tried against the *free* tiers only
— the corpus recogniser, pins, and the caches — and a row that would now be
answered is shown as *answered since ⟨date⟩* rather than as a gap, with the
first such date stored. An admin "re-check" action re-asks through the full
cascade on demand; it is never automatic, because it costs a model call.

**Proof.** A unit test with a ledger holding a refusal for a question the
recogniser now answers; the CLI and the panel both label it.

## P5 — Typo tolerance

**Status: measured, and deliberately not built.** 40 cases across four corpora,
160 question-and-typo pairs, 156 still retrieve the document they need - 97.5%.
One case in forty is lost, in `golden-ftr` alone. A question carries several
content words and BM25 finds the document on the others, so the misspelling
that motivated this is mostly absorbed already; a term-repair pass would be new
machinery on the hot path of every query for 2.5% of one layer's recall. The
harness is committed so the decision can be re-taken when the number moves:
`evals/measured/fortyseventh-what-a-typo-costs.json`.

**Evidence.** `summaryze`, `priroities`. Two of sixteen. BM25 on a misspelled
content word finds nothing, and nothing here has measured how often that
happens or what it costs.

**Design.** Measure first: `tools/measure_typos.py` applies one- and
two-character edits to the golden questions and reports the accuracy drop.
Then the cheap fix, if the number warrants it: a query term with zero hits in
the index is replaced by the indexed term within edit distance two, when
exactly one such term exists. Deterministic, no model, and it never touches a
term that already matches.

**Proof.** The typo set before and after; the untouched golden sets unchanged.

## P6 — One real Entra tenant

Blocked on the owner, not on code. `auth/` (739 lines) and the SharePoint,
Drive and Teams connectors (1,743) have only ever met a fake. The deliverable
here is a runbook with the five checks to make in the first hour: sign-in,
group claims arriving as ACL principals, token refresh across an hour, sign-out,
and a SharePoint delta sync where one folder is denied to the test user.

## P7 — A quotation must be a quotation

**Status: measured, registered, being built.** v0.12.10 shipped a correct
answer carrying a fabricated citation, and named this as the largest open hole.
`fortyninth-...` recorded it; this is the fix.

**The failure.** Asked who approves a contract of exactly EUR 25,000, the model
answered *Head of Department* - right - and supported it with what reads as a
row of the policy table:

> "Contract value (annual): Above EUR 25,000 | Approver: Chief Financial
> Officer | Additional requirement: Three competitive quotes"

No such row exists. "Above EUR 25,000" occurs in the document as prose about
competitive quotes and is never paired with an approver. Every element is real
somewhere in the document, so a gate that scores word and number overlap sees
nothing wrong. The same gate passed an invented fourth act of a three-act play
in v0.12.8.

**Measured before anything was built.** `tools/measure_quotations.py`, over
every answer this repository kept from its own evaluation runs, each paired
with the documents that answer cited:

| | |
|---|---:|
| answers examined | 674 |
| answers containing a quotation | 71 (10.5%) |
| quoted spans of 4+ words | 99 |
| found verbatim, whitespace and markdown and case normalised | 94 (94.9%) |
| found at no rung | 5 (5.1%) |

The normalisation ladder is the useful part. Exact matching finds 70.7%;
dropping Markdown emphasis adds 21.2 points, case-folding another 3.0. A rung
that also folded table separators and dashes recovered **nothing**, so the
loosest rule considered is not needed and is not included.

**Then the five were read rather than counted, and three were honest.**

| span | words | what it is |
|---|---:|---|
| "Contract value (annual): Above EUR 25,000 \| Approver: …" | 17 | the fabrication above |
| "The per diem allowance covers meals, including taxes and tips" | 10 | a paraphrase in quotation marks; the source says the allowance is *"a daily payment instead of reimbursement for actual expenses for lodging, meals, and related incidental expenses"* |
| "from X to Y" | 4 | a schematic placeholder - *"both documents use "from X to Y" bands"* - claiming nothing about any document's words |
| "equal to or above," | 4 | quoted **in order to deny it**: *"the policy specifies "above EUR 25,000" and not "equal to or above""* |
| "equal to or above," | 4 | the same answer, from a second run |

So a check that fires on every unmatched quotation would be **40% precise** -
rejecting three honest answers for two fabrications. That is the risk v0.12.10
named, now a number rather than a worry.

**What separates them.** A floor on length, swept rather than guessed:

| floor | spans checked | rejected | genuine |
|---:|---:|---:|---|
| 4 | 99 | 5 | 2 of 5 |
| 5 to 10 | 93 | 2 | **2 of 2** |
| 11+ | 86 | 1 | loses the per-diem one |

Both false positives are exactly 4 words; both fabrications are 10 and 17. The
plateau from 5 to 10 is what makes this a threshold rather than a fitted point,
and there is a reason behind it: a four-word quoted fragment *mentions* a
phrase, while a longer span *claims what a document says*. **Eight words**, the
middle of the plateau, with four words of margin below and two above.

**The change.** A fifth check in the gate: a quoted span of eight or more words
must appear in the evidence, after collapsing whitespace and dropping Markdown
emphasis and case. Checked against the raw cited text rather than the
machine-talk-filtered version, because "does this document contain these words"
is a question about the document as written. Quotations of the question, and of
the passage headers the context introduced, count as found - the model was
shown those too. A span quoted inside a negation is exempt regardless of
length, because that is quoting in order to deny, and the sample contains two
of them.

**Registered before the arm runs.**

1. **This will cost a passing case, and that is the point.** The fabricated row
   is in `rule-08`'s answer, which passes today. Rejecting it takes
   `golden-rules` from 71.4% to **64.3%**. Predicted here so it cannot be
   explained away afterwards: an answer whose citation is invented should not
   be served, and a gate that only fires on wrong answers would not be a gate.
2. **The guard cannot regress, and is run anyway.** A stricter gate can only
   turn answers into refusals; it cannot manufacture a false answer. Zero false
   answers across the 32 refusal cases, as always, and no refusal case may
   change tier.
3. **No other answerable case may break.** Anything passing at 71.4% other than
   `rule-08` must still pass.
4. **The historical sample must reject exactly the two fabrications** - re-run
   `measure_quotations.py` with the shipped rule and get 2, not 5 and not 0.
5. Full suite passes; the grounding-policy revision moves g5 -> g6.

**What this cannot establish.** Two fabrications is a small sample of the thing
being caught, and both come from one model on two corpora. The 94 spans it must
not disturb are the stronger half of the evidence. A frontier model quoting
across a hundred corpora could easily produce a form of faithful quotation this
ladder has never seen, and the honest response then is another rung with its
own measurement, not a loosened floor.
