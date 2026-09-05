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
| P3 | Apply a stated rule to a figure in the question | inj-07 and golden-ftr, independently | medium | — |
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

**Status: set written and pre-flighted; baseline running.** 16 cases over the
aveline corpus - 7 interior, 7 boundary, 2 refusal - in `evals/golden-rules/`,
with `tests/test_golden_rules.py` checking every requirement and prohibition
against a written-out correct answer. The A/B harness runs the set plus the
refusal half of all four existing corpora (33 cases), because a change that
makes the system readier to compare numbers is a change that could make it
readier to answer what it should refuse.

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

**Design.** A targeted set first, `evals/golden-rules`: ten to fifteen
"does *this* qualify" cases over the aveline and Northwind corpora, each with
a numeric threshold in the document and a figure in the question, on both
sides of the line. Then one prompt addition, measured alone — when a source
states a threshold and the question states a figure, compare them and say
which side it falls — with `PROMPT_VERSION` bumped. Accepted only if the four
existing sets keep zero false answers.

**Proof.** `golden-rules` before and after; the other sets unmoved.

**Risk.** Prompt changes have cost refusals before, and were reverted for it
(`fortysecond-the-fix-that-cost-a-refusal.json`). Hence the set first, the
change second, and one change at a time.

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
