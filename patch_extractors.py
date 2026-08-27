"""
Patch the un-fixed answer-extraction patterns.

THE SITUATION
    The boundary bug was fixed in fix_answers.py. It was never back-ported.
    Three scripts still contain the original pattern:

        prescreen.py    run_main.py    hint_pilot.py

    The DATA is fine - fix_answers.py re-derived every answer from stored
    text. But the source still contains the bug, which means anyone reading
    the repo sees the defect the write-up claims was fixed. That is a
    credibility problem, and it is also a real hazard: re-run prescreen.py
    tomorrow and it silently reintroduces the corruption.

THE BUG, ONCE MORE
        \\(?([ABCD])\\)?
    matches the A in "**Answer:** Answer: B" - the letter captured is the
    first letter of the WORD "Answer", not the answer.

THE FIX
        \\(?([ABCD])\\)?(?![A-Za-z])
    A negative lookahead: the captured letter cannot be followed by another
    letter, so it cannot be the start of a longer word.

WHAT THIS DOES
    Adds the lookahead to every capture in those three files that lacks it,
    backs each file up as .prebug, then VERIFIES by compiling the patched
    patterns and running the known-bad string through them. It does not
    import the modules - that would need API keys - it re-reads and compiles
    the pattern strings directly.

    python patch_extractors.py
"""

import re
import shutil
from pathlib import Path

TARGETS = ["prescreen.py", "run_main.py", "hint_pilot.py"]

# find "([ABCD])\)?" not already followed by the lookahead
NEEDS_FIX = re.compile(r"\(\[ABCD\]\)\\\)\?(?!\(\?\!\[A-Za-z\]\))")
LOOKAHEAD = r"(?![A-Za-z])"

# the string that caused the whole problem
BAD = "**Answer:** Answer: B"
WANT = "B"

PATTERN_LINE = re.compile(r"re\.compile\(\s*r\"(.+?)\"\s*(?:,\s*re\.\w+\s*)?\)")


def extract_with(patterns, text):
    for pat in patterns:
        hits = pat.findall(text or "")
        if hits:
            return hits[-1].upper()
    return None


def compile_from_source(src):
    """Pull out the r"..." patterns and compile them, without importing."""
    out = []
    for m in PATTERN_LINE.finditer(src):
        body = m.group(1)
        if "ABCD" not in body:
            continue
        flags = 0
        tail = src[m.end(1):m.end()]
        if "re.I" in tail:
            flags |= re.I
        if "re.M" in tail:
            flags |= re.M
        try:
            out.append(re.compile(body, flags))
        except re.error:
            pass
    return out


def main():
    print("Patching answer-extraction patterns\n")
    any_change = False

    for name in TARGETS:
        p = Path(name)
        if not p.exists():
            print(f"  {name:<16} not found - skipping")
            continue

        src = p.read_text(encoding="utf-8")

        before = compile_from_source(src)
        got_before = extract_with(before, BAD) if before else None

        patched, n = NEEDS_FIX.subn(
            lambda m: m.group(0) + LOOKAHEAD, src)

        if n == 0:
            print(f"  {name:<16} already correct ({len(before)} patterns)")
            continue

        shutil.copy(p, str(p) + ".prebug")
        p.write_text(patched, encoding="utf-8")
        any_change = True

        after = compile_from_source(patched)
        got_after = extract_with(after, BAD) if after else None

        ok = got_after == WANT
        print(f"  {name:<16} {n} pattern(s) fixed   backup -> {name}.prebug")
        print(f"  {'':<16} on {BAD!r}:")
        print(f"  {'':<16}   before: {got_before}   after: {got_after}   "
              f"{'OK' if ok else 'STILL WRONG'}")
        if not ok:
            print(f"  {'':<16}   ^ investigate before trusting this file")
        print()

    if not any_change:
        print("\nNothing needed changing.")
    else:
        print("""
Done. The .prebug backups are the originals if you want to diff them.

Two things worth saying in the write-up:

  1. The data was never affected - fix_answers.py re-derived every answer
     from stored raw text, so no re-running was required. Storing the full
     model output rather than just the parsed answer is what made an offline
     fix possible, and that was worth more than any amount of care with the
     regex.

  2. The real failure was duplication: the same pattern lived in four files,
     so fixing one left three wrong. That is the lesson, not the lookahead.

Now run:  python test_extract.py""")


if __name__ == "__main__":
    main()
