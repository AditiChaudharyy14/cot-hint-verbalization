"""
Tests for the statistics. Read this before you run it.

WHY THIS FILE EXISTS
    Every number in the write-up comes from a function neither of us checked
    against a value known from outside the code. A wrong confidence interval
    or a mis-implemented McNemar would be completely invisible - the output
    would look exactly as plausible as it does now.

    It is also the honest answer to "did you understand the analysis or just
    run it?" You cannot write a test like this without knowing what the
    right answer is and why.

HOW TO USE IT PROPERLY
    Before running: take each EXPECTED value below and work out where it
    comes from. The derivations are written out. If one doesn't make sense,
    that is the thing to ask about - it is a gap in the analysis you would
    have to defend.

    Then run it and see them pass.

    python test_stats.py
"""

import math
import sys

from analyse import wilson
from paired import binom_two_sided, cochran_q, mcnemar
from explain import fisher_exact

TOL = 1e-4
results = []


def check(name, got, want, tol=TOL, why=""):
    ok = abs(got - want) < tol
    results.append(ok)
    mark = "ok  " if ok else "FAIL"
    print(f"  [{mark}] {name}")
    print(f"         got {got:.6f}   want {want:.6f}")
    if why:
        for line in why.strip().split("\n"):
            print(f"         {line}")
    print()


def head(t):
    print(f"\n{'='*70}\n{t}\n{'='*70}\n")


# ===================================================================== 1
head("1  WILSON INTERVAL  -  against published values")

print("""  The Wilson score interval is used instead of the normal approximation
  because at n~59 with p~0.2 the normal interval is visibly wrong and can
  run below zero. These two cases are standard textbook values.
""")

_, lo, hi = wilson(0, 10)
check("wilson(0, 10) lower", lo, 0.0, 1e-3,
      why="""0 successes. The normal approximation gives [0, 0] - it claims
zero uncertainty from ten observations, which is absurd. Wilson
does not collapse.""")
check("wilson(0, 10) upper", hi, 0.27753, 1e-4,
      why="""Published value 0.2775. Ten observations with no successes are
still consistent with a true rate as high as 28%.""")

_, lo, hi = wilson(1, 10)
check("wilson(1, 10) lower", lo, 0.01791, 1e-4, why="Published value 0.0179.")
check("wilson(1, 10) upper", hi, 0.40415, 1e-4, why="Published value 0.4041.")

p, lo, hi = wilson(14, 59)
check("wilson(14, 59) point", p, 14 / 59, 1e-9,
      why="This is the neutral arm's deference rate: 24%.")
check("wilson(14, 59) lower", lo, 0.14694, 1e-4,
      why="""The interval reported in the results is [15%, 36%]. If this
fails, every CI in the write-up is wrong.""")
check("wilson(14, 59) upper", hi, 0.35975, 1e-4)


# ===================================================================== 2
head("2  EXACT BINOMIAL  -  the engine inside McNemar")

print("""  McNemar's exact test asks: given n discordant pairs, how surprising is
  a split this lopsided if each pair were a coin flip? So it is a two-sided
  binomial test against p = 0.5.
""")

check("binom_two_sided(0, 6)", binom_two_sided(0, 6), 2 / 64, 1e-9,
      why="""6 discordant, all one way. One arrangement out of 2^6 = 64 gives
6-0, doubled for two-sidedness: 2/64 = 0.03125.

THIS IS THE NUMBER FROM THE LIMITATIONS SECTION. With only 6
discordant questions the smallest achievable p is 0.031 - and it
requires a PERFECT split. The planned test could not have reached
significance on anything less.""")

check("binom_two_sided(2, 6)", binom_two_sided(2, 6), 44 / 64, 1e-9,
      why="""The observed 4-2 split. Tail = C(6,0)+C(6,1)+C(6,2) = 1+6+15 = 22.
Doubled: 44/64 = 0.6875, which rounds to the 0.688 in the output.""")

check("binom_two_sided(3, 10)", binom_two_sided(3, 10), 352 / 1024, 1e-9,
      why="""The placebo check, 3-7. Tail = 1+10+45+120 = 176 out of 1024,
doubled = 0.34375 -> the 0.344 in the output.""")

check("binom_two_sided(0, 0)", binom_two_sided(0, 0), 1.0, 1e-9,
      why="No discordant pairs at all: no evidence either way, p = 1.")


# ===================================================================== 3
head("3  FISHER EXACT  -  against Fisher's own example")

print("""  Fisher's tea-tasting experiment: 8 cups, 4 of each kind, the taster gets
  3 of 4 right. The canonical two-sided answer is 0.4857.
""")

check("fisher_exact(3,1,1,3)", fisher_exact(3, 1, 1, 3), 0.485714, 1e-5,
      why="""Hypergeometric. P(a=3) = C(4,3)C(4,1)/C(8,4) = 16/70.
Two-sided sums every arrangement at most as likely as observed:
P(0)+P(1)+P(3)+P(4) = (1+16+16+1)/70 = 34/70 = 0.4857.
P(a=2) = 36/70 is MORE likely than observed, so it is excluded.""")

check("fisher_exact(5,5,5,5)", fisher_exact(5, 5, 5, 5), 1.0, 1e-9,
      why="Perfectly balanced table: no evidence of association at all.")


# ===================================================================== 4
head("4  COCHRAN'S Q  -  proved against McNemar")

print("""  This is the important one, because Cochran's Q is the omnibus the whole
  paired analysis rests on and there is no textbook value to hand.

  Instead, use an identity: with k = 2 treatments, Cochran's Q reduces
  EXACTLY to McNemar's chi-square statistic,

        Q  =  (b01 - b10)^2 / (b01 + b10)

  because the concordant rows contribute nothing to either the numerator
  or the denominator. If the implementation is right, it must reproduce
  that. If it does not, the omnibus in the results is wrong.
""")

# 3 questions defer under B only, 7 under A only, plus concordant filler
matrix = {}
i = 0
for _ in range(3):
    matrix[i] = {"A": 0, "B": 1}; i += 1
for _ in range(7):
    matrix[i] = {"A": 1, "B": 0}; i += 1
for _ in range(5):
    matrix[i] = {"A": 1, "B": 1}; i += 1
for _ in range(5):
    matrix[i] = {"A": 0, "B": 0}; i += 1

b01, b10, n_disc, _ = mcnemar(matrix, "A", "B")
check("mcnemar b01", b01, 3, 1e-9)
check("mcnemar b10", b10, 7, 1e-9)
check("mcnemar discordant", n_disc, 10, 1e-9,
      why="""The 10 concordant questions are invisible to the test. Only
questions that behaved DIFFERENTLY between arms carry information -
which is exactly why effective sample size matters so much here.""")

q, df, _ = cochran_q(matrix, ["A", "B"])
check("cochran_q equals McNemar chi2", q, (3 - 7) ** 2 / 10, 1e-9,
      why="""(b01-b10)^2 / (b01+b10) = 16/10 = 1.6.
Matching this is real evidence the implementation is correct.""")
check("cochran_q df", df, 1, 1e-9, why="k - 1 = 1 for two arms.")

# --- a case where Q is UNDEFINED, not zero -------------------------------
# Caught a wrong expectation on the first run of this file. Worth keeping
# for that reason: the test was wrong and the code was right.
flat = {i: {a: 1 for a in "ABC"} for i in range(8)}
res = cochran_q(flat, list("ABC"))
ok = res is None
results.append(ok)
print(f"  [{'ok  ' if ok else 'FAIL'}] cochran_q, every question constant")
print(f"         got {res}   want None")
print("""         Denominator is k*sum(L) - sum(L^2). With every row constant
         at L = k, that is k*(n*k) - n*k^2 = 0. Q is 0/0 - UNDEFINED,
         not zero. Returning None is right; returning 0 would assert
         "no difference" where there is simply no information.

         Same principle as the 44 questions in this study that behave
         identically under all five framings: they contribute nothing
         to Cochran's denominator, exactly as they contribute nothing
         to McNemar's discordant count. That is what "effective sample
         size 15, not 59" means arithmetically.
""")

# --- Q genuinely equal to zero, which needs variation to exist -----------
balanced = {}
i = 0
for _ in range(5):
    balanced[i] = {"A": 1, "B": 0}; i += 1
for _ in range(5):
    balanced[i] = {"A": 0, "B": 1}; i += 1
qz, _, _ = cochran_q(balanced, ["A", "B"])
check("cochran_q, arms differ but totals equal", qz, 0.0, 1e-9,
      why="""Ten questions vary, but each arm defers on exactly 5 - so the
arms are indistinguishable in aggregate. (b01-b10)^2/(b01+b10)
= 0/10 = 0. THIS is a real zero: information exists and says
the arms are the same. Compare with the case above, where no
information exists at all. A statistic must tell those apart.""")


# ===================================================================== end
head("RESULT")
n_pass, n_fail = sum(results), len(results) - sum(results)
print(f"  {n_pass} passed, {n_fail} failed\n")

if n_fail:
    print("""  A failure means a number in the write-up is wrong. Find out which
  before doing anything else.""")
    sys.exit(1)

print("""  All green. Every statistic in the results has now been checked against
  a value derived independently of the code:

    Wilson    published textbook intervals
    binomial  counted by hand from Pascal's triangle
    Fisher    Fisher's own tea-tasting example
    Cochran   the k=2 identity with McNemar

  Worth one line in the methods section: the analysis functions were
  verified against known values rather than trusted. That is a different
  claim from "I ran the code", and it is the one that answers the question
  about understanding what an AI assistant wrote.
""")
