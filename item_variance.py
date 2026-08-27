"""
What distinguishes the questions the model always defers on?

THE OPEN QUESTION
    39 of 59 questions never defer under any framing. 5 always do. Framing
    explains none of it, and baseline correctness cannot explain it either -
    the stability screen left only 2 questions with an incorrect baseline,
    so there is no variance in that variable to work with.

    So the item effect is real and unexplained. This tests what is left in
    the data.

FOUR CANDIDATES

  1  TRACE LENGTH as a proxy for uncertainty.
     No logprobs are available through this API, so length is the only
     confidence signal recorded. If the model reasons longer on the
     questions it defers on, deference tracks difficulty. Weak proxy -
     length also tracks how much there is to say - so treat it as
     suggestive at best.

  2  SUBJECT. Some domains may invite deference more than others.

  3  HINT LETTER. If deference depends on WHICH letter was suggested,
     that is position bias, not authority - and it would be a problem
     for the measure rather than a finding.

  4  BASELINE LETTER. Same logic from the other side.

EXPLORATORY - SAY SO IN THE WRITE-UP
    These four tests were chosen after seeing the main result. At alpha=.05,
    four tests carry roughly a 1-in-5 chance of one false positive. Nothing
    here is confirmatory, and anything that looks significant is a
    hypothesis for a future study, not a finding from this one.

    python item_variance.py
"""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

ARMS = ["neutral", "evaluation", "deployment", "training", "placebo"]


def read(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]


def block(t):
    print(f"\n{'='*70}\n{t}\n{'='*70}")


def mannwhitney(a, b):
    """Returns (U, p) or (None, None). Rank-based: no normality assumed,
    which matters because trace lengths are heavily skewed."""
    try:
        from scipy.stats import mannwhitneyu
        u, p = mannwhitneyu(a, b, alternative="two-sided")
        return float(u), float(p)
    except Exception:
        return None, None


def spearman(x, y):
    try:
        from scipy.stats import spearmanr
        r, p = spearmanr(x, y)
        return float(r), float(p)
    except Exception:
        return None, None


def chi2_table(counts):
    rows = [(k, n - k) for k, n in counts if n > 0]
    if len(rows) < 2:
        return None
    tot = sum(a + b for a, b in rows)
    col = [sum(r[0] for r in rows), sum(r[1] for r in rows)]
    chi2 = 0.0
    for a, b in rows:
        rt = a + b
        for obs, ct in ((a, col[0]), (b, col[1])):
            exp = rt * ct / tot
            if exp > 0:
                chi2 += (obs - exp) ** 2 / exp
    df = len(rows) - 1
    try:
        from scipy.stats import chi2 as _c
        return chi2, df, float(_c.sf(chi2, df))
    except Exception:
        return chi2, df, None


def median(v):
    s = sorted(v)
    n = len(s)
    if not n:
        return 0
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


def rate_block(title, groups, note=""):
    """groups: list of (label, deferred, total)"""
    print(f"\n  {title}\n")
    print(f"    {'group':<26}{'deferred':>12}{'rate':>8}")
    for lab, k, n in groups:
        if n:
            print(f"    {lab:<26}{f'{k}/{n}':>12}{k/n:>7.0%}")
    res = chi2_table([(k, n) for _, k, n in groups if n])
    if res:
        c, df, p = res
        ptxt = f"p = {p:.3f}" if p is not None else "(scipy needed)"
        star = "  *" if p is not None and p < .05 else ""
        print(f"\n    chi2 = {c:.2f}, df = {df}, {ptxt}{star}")
    if note:
        print(f"    {note}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openai/gpt-oss-120b")
    args = ap.parse_args()

    slug = args.model.replace("/", "_")
    jp, sp = Path(f"judged_{slug}.jsonl"), Path(f"stable_{slug}.jsonl")
    for p in (jp, sp):
        if not p.exists():
            raise SystemExit(f"can't find {p}")

    qs = {q["id"]: q for q in read(sp)}
    rows = []
    for r in read(jp):
        st = (r.get("judge") or {}).get("stance")
        if st not in ("D", "R", "I"):
            continue
        q = qs.get(r["id"], {})
        r["stance"] = st
        r["deferred"] = st == "D"
        r["subject"] = q.get("subject", "unknown")
        r["length"] = len(r.get("trace") or "")
        rows.append(r)

    print(f"{len(rows)} scored traces")

    # ------------------------------------------------------- 1 trace length
    block("1  -  TRACE LENGTH  (proxy for uncertainty)")

    d = [r["length"] for r in rows if r["deferred"]]
    n_ = [r["length"] for r in rows if not r["deferred"]]
    print(f"\n  median trace length")
    print(f"    deferred      {median(d):>8,.0f} chars   (n={len(d)})")
    print(f"    did not       {median(n_):>8,.0f} chars   (n={len(n_)})")
    u, p = mannwhitney(d, n_)
    if p is not None:
        print(f"\n    Mann-Whitney U = {u:.0f}, p = {p:.4f}"
              f"{'  *' if p < .05 else ''}")
    print("""
    Shorter traces on deferred answers would mean the model stops
    reasoning once it accepts the claim - deference as a shortcut.
    Longer would mean it deliberates and yields anyway.""")

    # per question
    by_q = defaultdict(lambda: {"defer": 0, "len": []})
    for r in rows:
        by_q[r["id"]]["defer"] += int(r["deferred"])
        by_q[r["id"]]["len"].append(r["length"])

    qs_ = list(by_q)
    xs = [by_q[q]["defer"] for q in qs_]
    ys = [median(by_q[q]["len"]) for q in qs_]
    rho, pr = spearman(xs, ys)
    print(f"\n  per QUESTION: deference count (0-5) vs median trace length")
    if rho is not None:
        print(f"    Spearman rho = {rho:+.3f}, p = {pr:.4f}"
              f"{'  *' if pr < .05 else ''}   (n={len(qs_)} questions)")
        print("\n    This is the cleaner test - it asks whether the questions")
        print("    the model defers on are the ones it thinks longest about.")

    # ------------------------------------------------------------ 2 subject
    block("2  -  SUBJECT")
    subs = defaultdict(lambda: [0, 0])
    for r in rows:
        subs[r["subject"]][1] += 1
        subs[r["subject"]][0] += int(r["deferred"])
    groups = sorted(((s, k, n) for s, (k, n) in subs.items()),
                    key=lambda t: -t[1] / max(t[2], 1))
    rate_block("deference by MMLU subject", groups,
               note="With ~8 subjects this is badly underpowered per cell.")

    # -------------------------------------------------------- 3 hint letter
    block("3  -  HINT LETTER  (position bias check)")
    hl = defaultdict(lambda: [0, 0])
    for r in rows:
        hl[r["hint_letter"]][1] += 1
        hl[r["hint_letter"]][0] += int(r["deferred"])
    rate_block("deference by which letter the hint named",
               [(f"hint = ({L})", hl[L][0], hl[L][1]) for L in "ABCD"],
               note=("A difference here is a PROBLEM, not a finding: it would\n"
                     "    mean the measure partly tracks letter position rather\n"
                     "    than the authority claim."))

    # ---------------------------------------------------- 4 baseline letter
    block("4  -  BASELINE LETTER")
    bl = defaultdict(lambda: [0, 0])
    for r in rows:
        b = r.get("baseline_answer")
        if b:
            bl[b][1] += 1
            bl[b][0] += int(r["deferred"])
    rate_block("deference by the model's own original answer",
               [(f"baseline = {L}", bl[L][0], bl[L][1]) for L in "ABCD"])

    # ------------------------------------------------------------- closing
    block("HOW TO REPORT THIS")
    print("""
  All four tests were chosen AFTER seeing the main result. Four tests at
  alpha=.05 means roughly a 1-in-5 chance that one looks significant by
  chance alone. Say that in the write-up and label the section exploratory.

  If nothing reaches significance, the honest sentence is that the item
  effect is robust but its source was not identified with the variables
  recorded - and that logprobs, which this API does not expose, are the
  obvious next thing to look at.

  A clearly stated unexplained effect is a better contribution than a
  post-hoc explanation that would not replicate.
""")


if __name__ == "__main__":
    main()
