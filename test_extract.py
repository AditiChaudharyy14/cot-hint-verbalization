"""
Regression tests for answer extraction.

WHY THIS EXISTS
    Answer extraction produced four silent bugs in this project. The worst
    corrupted 21 of 295 answers and moved the headline flip rate by 8 points,
    and it survived for two days because nothing ever checked the extractor
    against a case whose answer was known in advance.

    Aggregate output cannot catch this. A rate of 28% looks exactly as
    plausible as a rate of 20%. The only defence is a set of inputs whose
    correct output is known, run every time the patterns change.

THE CASES BELOW ARE REAL
    Every string marked THE BUG is copied from actual model output in this
    study, not invented. Each one is a format that broke a pattern.

DESIGN POINT WORTH KEEPING
    An extractor that GUESSES is more dangerous than one that FAILS. A None
    is visible in the unparsed count and gets investigated; a confidently
    wrong letter becomes a data point. Several cases below assert None on
    purpose - refusing to answer is the correct behaviour there.

    python test_extract.py
"""

import sys

from fix_answers import extract

CASES = [
    # ---- THE BUG that corrupted the study -----------------------------
    # "\\**\\s*\\(?([ABCD])\\)?" matched the A in the WORD "Answer".
    ("**Answer:** Answer: B",                     "B", "THE BUG: bold header"),
    ("Thus answer: Answer: A.",                   "A", "THE BUG: doubled"),
    ("**Answer:** Answer: D",                     "D", "THE BUG: bold header"),

    # ---- ordinary formats ---------------------------------------------
    ("Answer: C",                                 "C", "plain"),
    ("The answer is (D).",                        "D", "parenthesised"),
    ("Thus final answer: Answer: D.",             "D", "conclusion"),
    ("**B**",                                     "B", "bold letter only"),
    ("Option (C) is correct.",                    "C", "option form"),
    ("So the answer is A",                        "A", "no punctuation"),

    # ---- last mention wins --------------------------------------------
    ("First I thought the answer is A. "
     "On reflection, Answer: C",                  "C", "revised mid-trace"),

    # ---- MUST return None: refusing beats guessing ---------------------
    ("Answers vary by region.",                   None, "the word 'Answers'"),
    ("Answer: E",                                 None, "letter out of range"),
    ("This question is ambiguous.",               None, "no answer at all"),
    ("",                                          None, "empty"),
    (None,                                        None, "missing trace"),

    # ---- a KNOWN limitation, asserted so it stays known ----------------
    # Negation is not parsed. The extractor returns None rather than "A",
    # which is the safe failure. If someone later "improves" the patterns
    # and this starts returning A, this test catches it.
    ("The answer is not A, it's C.",              None, "negation -> None"),
]


def main():
    passed = failed = 0
    print(f"{'input':<52}{'want':>6}{'got':>6}   note")
    print("-" * 88)

    for text, want, note in CASES:
        got = extract(text)
        ok = got == want
        passed += ok
        failed += not ok
        shown = repr(text)[:50] if text is not None else "None"
        mark = "" if ok else "   <-- FAIL"
        print(f"{shown:<52}{str(want):>6}{str(got):>6}   {note}{mark}")

    print("-" * 88)
    print(f"{passed} passed, {failed} failed")

    if failed:
        print("""
A failure means the extractor's behaviour changed. Before touching the
patterns, work out which is wrong - the test or the code. On this project
it was the code, four times.""")
        sys.exit(1)

    print("""
All green. Run this after ANY change to the extraction patterns.

What it does not cover, and should be said in the writeup: these are the
formats seen in 295 traces from one model. A different model, or a different
prompt, will produce formats not represented here. The unparsed count in the
main run is the check for that - it should be near zero, and if it climbs,
a new format has appeared and belongs in this file.""")


if __name__ == "__main__":
    main()
