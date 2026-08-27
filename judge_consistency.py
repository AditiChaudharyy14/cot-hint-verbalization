"""
Does the judge agree with ITSELF?

WHY THIS MATTERS
    judge.py --calibrate showed the judge agrees with a human 48/50 (k=0.88).
    That is agreement with an external standard. It says nothing about
    whether the judge is STABLE - whether the same trace, sent twice, comes
    back with the same stance.

    If it isn't, every rate in the study carries an extra source of variance
    that the confidence intervals do not include. A 4-point difference
    between two arms means nothing if re-running the judge moves each arm by
    5 points on its own.

    This is the standard reliability check for an LLM judge and it is
    routinely skipped. It costs about ten cents.

WHAT IT DOES
    Sends the same N traces through the judge twice and reports:
      - how often the two runs agree
      - which stance is least stable
      - how far the resulting deference RATE moves between runs

    The last one is the number to quote: it is the judge's own noise floor,
    expressed in the same units as the effect being looked for.

    python judge_consistency.py           50 traces, ~$0.10
    python judge_consistency.py --n 100
"""

import argparse
import json
import os
import random
from collections import Counter
from pathlib import Path

from anthropic import Anthropic
from dotenv import load_dotenv
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent))
from judge import judge_one

load_dotenv()

SEED = 20260827


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openai/gpt-oss-120b")
    ap.add_argument("--n", type=int, default=50)
    args = ap.parse_args()

    src = Path(f"judged_{args.model.replace('/', '_')}.jsonl")
    if not src.exists():
        raise SystemExit(f"can't find {src}")

    rows = [json.loads(l) for l in open(src, encoding="utf-8")]
    random.seed(SEED)
    sample = random.sample(rows, min(args.n, len(rows)))
    print(f"re-judging {len(sample)} traces twice\n")

    client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    runs = [[], []]
    for pass_no in (0, 1):
        for r in tqdm(sample, unit="trace", desc=f"pass {pass_no+1}"):
            v = judge_one(client, r.get("trace"), r["hint_letter"])
            runs[pass_no].append(v["stance"])

    a, b = runs
    n = len(sample)
    agree = sum(1 for x, y in zip(a, b) if x == y)

    print(f"\n{'='*66}\nSELF-CONSISTENCY\n{'='*66}")
    print(f"\n  same stance both times: {agree}/{n} = {agree/n:.0%}")

    print(f"\n  {'stance':<10}{'run 1':>8}{'run 2':>8}{'stable':>9}")
    for s in ("D", "R", "I", None):
        c1 = sum(1 for x in a if x == s)
        c2 = sum(1 for x in b if x == s)
        both = sum(1 for x, y in zip(a, b) if x == s and y == s)
        lab = s or "unparsed"
        rate = f"{both/c1:.0%}" if c1 else "-"
        print(f"  {lab:<10}{c1:>8}{c2:>8}{rate:>9}")

    flips = Counter((x, y) for x, y in zip(a, b) if x != y)
    if flips:
        print("\n  where it disagreed with itself:")
        for (x, y), k in flips.most_common():
            print(f"    {x} -> {y}   {k}")

    d1 = sum(1 for x in a if x == "D") / n
    d2 = sum(1 for x in b if x == "D") / n
    print(f"\n{'='*66}\nTHE NUMBER TO QUOTE\n{'='*66}")
    print(f"\n  deference rate, run 1: {d1:.0%}")
    print(f"  deference rate, run 2: {d2:.0%}")
    print(f"  movement from re-running the judge alone: "
          f"{abs(d1-d2)*100:.1f} points")

    print(f"""
  Compare that to the spread you are trying to interpret across arms
  (~7 points, neutral 24% to placebo 17%).

  If judge noise is well under 7 points, the arm differences are at least
  measurable in principle, and this number belongs in the methods section as
  evidence you checked.

  If it is comparable to 7 points, then judge instability alone could produce
  your entire observed ordering, and that must be said plainly in the
  limitations - it would mean the ranking of arms is not interpretable at
  all, independently of sample size.
""")


if __name__ == "__main__":
    main()
