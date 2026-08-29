"""Questions about the assistant, and the near-misses that must never be.

The point of this set is that it was written to *defeat* the recogniser's
vocabulary, not to flatter it: the verbs and nouns here were chosen without
looking at the word lists. A hand-maintained list can never be complete, so
the only honest measure of "will it recognise what people actually ask" is a
set nobody tuned it against.

ABOUT_ME must be answered by the assistant, free, with no model call.
ABOUT_THE_DOCUMENTS must go to retrieval - answering these from a canned
description would be a far worse failure than the one this set exists to
catch, so both directions are measured.
"""

from __future__ import annotations

#: Self-referential questions, in the vocabulary people actually use.
ABOUT_ME: tuple[str, ...] = (
    # the field failures, verbatim
    "whaat can you do for me?",
    "what can you do it for me?",
    "do you summarize documents? any document I provide you?",
    "what files can you manage?",
    "what can you help me with?",
    # capability, in verbs deliberately not on any list
    "what are you able to accomplish?",
    "which tasks can you perform?",
    "what sort of things can you handle?",
    "are you able to digest a spreadsheet?",
    "can you crunch a pdf for me?",
    "could you go through a powerpoint?",
    "will you parse a word file?",
    "what kind of stuff do you deal with?",
    "how do you work?",
    "how are you supposed to be used?",
    "what is your purpose?",
    "who are you?",
    "what are you?",
    "what do you actually do?",
    "are you able to browse the internet?",
    "can you write code for me?",
    "do you have access to my email?",
    "what languages do you speak?",
    "can you translate something for me?",
    "do you learn from what I upload?",
    "are you storing my documents somewhere?",
    "how much do you cost to run?",
    "what are your limitations?",
    "what can't you do?",
    "do you make things up?",
    "can I trust your answers?",
    "where do your answers come from?",
    "do you cite sources?",
    "what happens if you don't know something?",
    "can you look at images?",
    "do you support excel files?",
    "which formats are you able to read?",
    "can you take a scanned contract?",
    "how many documents can you hold?",
    "are you connected to the internet?",
)

#: Questions about the documents, which must keep going to retrieval even
#: though several of them contain "you" or "your".
ABOUT_THE_DOCUMENTS: tuple[str, ...] = (
    "what is the notice period?",
    "what are the main priorities?",
    "what are the main characters?",
    "how much is the meal allowance?",
    "what did Caterina say about the self-assessment gap?",
    "summarise the handbook",
    "summarize the second document",
    "what are the step by step activities we should cover for arvexlab?",
    "manage the vendor list",
    "read the contract and tell me the notice period",
    "can you tell me what the handbook says about parental leave?",
    "what does the policy say you should do about expenses?",
    "what files did Giuseppe send about the website?",
    "who are the people mentioned in the debrief?",
    "what is mythos?",
    "translate the executive summary into Italian",
    "compare the two pricing proposals",
    "how many days of leave do I get?",
    "explain the NIS2 eligibility checker",
)

#: The genuinely ambiguous middle: phrased at the assistant ("your"), but
#: really about the documents - and the documents have nothing on the
#: subject. Both routes end in the same honest refusal, so neither is a
#: failure; what must never happen is a confident answer, or a paid call
#: that buys one. These are pinned on the *outcome*, not the route.
EITHER_IS_HONEST: tuple[str, ...] = (
    "what is your company's refund policy?",
    "what is your policy on overtime for contractors?",
    "do you have anything on the 2019 merger?",
)
