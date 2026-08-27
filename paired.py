"""
The analysis the design actually calls for.

WHY THIS EXISTS
    The same 59 stable questions went through all five arms. That is a
    REPEATED-MEASURES design: every question is its own control. analyse.py
    treated the arms as five independent samples, which is the wrong test for
    this data - it discards the pairing and inflates the variance with
    question-level differences that the design already controls for.

    Some questions are simply stickier than others. A question the model
    never defers on contributes nothing to a between-arm comparison except
    noise. Pairing removes that.

WHAT IT RUNS
    Cochran's Q      - the paired generalization of the omnibus chi-square:
                       k related binary treatments measured on the same
                       blocks. This is the correct omnibus for this design.

    McNemar (exact)  - the paired two-arm test. Only DISCORDANT questions
                       carry information: those where the model deferred
                       under one framing and not the other. Questions that
                       behave identically under both are uninformative by
                       construction, and the exact binomial is used rather
                       than the chi-square approximation because those
                       counts are small.

    Effective n      - the number that actually matters, and the reason this
                       script may be the most useful thing in the project.
                       A question that never defers in any arm, or defers in
                       every arm, cannot show a framing effect. Only the ones
                       in between can. Nominal n is 59 per arm; the effective
                       n is much smaller, and reporting it is more honest
                       than reporting the nominal one.

    python paired.py
"""

import argparse
import json
import math
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

ARMS = ["neutral", "evaluation", "deployment", "training", "placebo"]


def chi2_sf(x, df):
    try:
        from scipy.stats import chi2
        return float(chi2.sf(x, df))
    except Exception:
        return None


def binom_two_sided(k, n):
    """Exact two-sided binomial test against p = 0.5."""
    if n == 0:
        return 1.0
    def c(n_, r):
        return math.comb(n_, r)
    tail = sum(c(n, i) for i in range(0, min(k, n - k) + 1))
    p = 2 * tail / (2 ** n)
    return min(1.0, p)


def load(path):
    rows = []
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        r["stance"] = (r.get("judge") or {}).get("stance")
        rows.append(r)
    return rows


def build_matrix(rows):
    """{question_id: {arm: 1 if deferred else 0}} for complete questions."""
    m = defaultdict(dict)
    for r in rows:
        if r["stance"] in ("D", "R", "I"):
            m[r["id"]][r["arm"]] = int(r["stance"] == "D")
    return {q: a for q, a in m.items() if all(k in a for k in ARMS)}


def cochran_q(matrix, arms):
    k = len(arms)
    rows = [[matrix[q][a] for a in arms] for q in matrix]
    G = [sum(r[j] for r in rows) for j in range(k)]        # per-arm totals
    L = [sum(r) for r in rows]                             # per-question totals
    num = (k - 1) * (k * sum(g * g for g in G) - sum(G) ** 2)
    den = k * sum(L) - sum(l * l for l in L)
    if den == 0:
        return None
    q = num / den
    return q, k - 1, chi2_sf(q, k - 1)


def mcnemar(matrix, a, b):
    b01 = sum(1 for q in matrix if not matrix[q][a] and matrix[q][b])
    b10 = sum(1 for q in matrix if matrix[q][a] and not matrix[q][b])
    n = b01 + b10
    p = binom_two_sided(min(b01, b10), n)
    return b01, b10, n, p


def block(t):
    print(f"\n{'='*70}\n{t}\n{'='*70}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openai/gpt-oss-120b")
    args = ap.parse_args()

    src = Path(f"judged_{args.model.replace('/', '_')}.jsonl")
    if not src.exists():
        raise SystemExit(f"can't find {src}")

    rows = load(src)
    matrix = build_matrix(rows)
    n_q = len(matrix)

    block("DESIGN")
    print(f"\n  {len(rows)} traces, {n_q} questions with all {len(ARMS)} arms")
    print( "  Every question appears once per arm - this is repeated measures,")
    print( "  not five independent samples.")

    # ------------------------------------------------- effective sample size
    block("EFFECTIVE SAMPLE SIZE  -  read this before any p-value")

    per_q = Counter(sum(matrix[q][a] for a in ARMS) for q in matrix)
    print("\n  how many of the 5 arms each question deferred in:\n")
    print(f"    {'deferred in':<14}{'questions':>10}")
    for i in range(len(ARMS) + 1):
        c = per_q.get(i, 0)
        bar = "#" * c
        note = ""
        if i == 0:
            note = "   never - carries no information"
        elif i == len(ARMS):
            note = "   always - carries no information"
        print(f"    {i} of 5{'':<8}{c:>10}  {bar}{note}")

    dead = per_q.get(0, 0) + per_q.get(len(ARMS), 0)
    live = n_q - dead
    print(f"\n  questions that could show a framing effect: {live} of {n_q}")
    print(f"  the other {dead} answer the same way under every framing.")
    print(f"""
  This is the honest sample size. Nominal n is 59 per arm; the number of
  questions where framing could possibly matter is {live}. Any power
  calculation based on 59 overstates what this study could detect.""")

    # ------------------------------------------------------------- omnibus
    block("OMNIBUS  -  Cochran's Q  (the paired chi-square)")
    res = cochran_q(matrix, ARMS)
    if res:
        q, df, p = res
        ptxt = f"p = {p:.3f}" if p is not None else "(scipy needed for p)"
        print(f"\n  Q = {q:.3f}, df = {df}, {ptxt}")
        print(f"\n  per-arm deference counts out of {n_q} questions:")
        for a in ARMS:
            k = sum(matrix[q_][a] for q_ in matrix)
            print(f"    {a:<12}{k:>4}/{n_q}   {k/n_q:>4.0%}")
        print("""
  Cochran's Q is the right omnibus here: k related binary treatments on the
  same blocks. Compare it to the unpaired chi-square in analyse.py and report
  this one, noting that the unpaired version ignores the design.""")

    # ------------------------------------------------------------ pairwise
    block("PAIRWISE  -  McNemar, exact")
    print("""
  Only DISCORDANT questions count - the ones that deferred under one framing
  and not the other. Everything else is uninformative by construction.

  b01 = deferred under the SECOND arm only
  b10 = deferred under the FIRST arm only
""")
    print(f"  {'comparison':<28}{'b01':>5}{'b10':>5}{'disc':>6}{'p':>9}")
    print("  " + "-" * 53)

    planned = [("evaluation", "neutral"), ("neutral", "placebo")]
    for a, b in planned + [p_ for p_ in combinations(ARMS, 2)
                           if p_ not in planned and p_[::-1] not in planned]:
        b01, b10, n, p = mcnemar(matrix, a, b)
        tag = ""
        if (a, b) == ("evaluation", "neutral"):
            tag = "  <- planned"
        elif (a, b) == ("neutral", "placebo"):
            tag = "  <- validity check"
        star = "*" if p < .05 else " "
        print(f"  {a[:12]+' vs '+b[:12]:<28}{b01:>5}{b10:>5}{n:>6}"
              f"{p:>9.3f}{star}{tag}")

    print("""
  A comparison with very few discordant questions cannot detect anything,
  whatever its p-value. Quote the discordant count next to every p.""")

    block("WHAT TO WRITE")
    print("""
  1. Report Cochran's Q as the omnibus, and say why: the design is repeated
     measures and the unpaired chi-square ignores that.
  2. Report McNemar with its discordant counts for the planned contrast and
     the placebo check.
  3. Lead the power discussion with the effective sample size above, not with
     n=59. "Only N of 59 questions varied across framings at all" is a much
     stronger and more honest statement than a minimum detectable effect
     computed from the nominal n.
""")


if __name__ == "__main__":
    main()
