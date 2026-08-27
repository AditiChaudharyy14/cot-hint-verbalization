"""
If framing doesn't explain deference, what does?

THE OPENING
    paired.py showed that 44 of 59 questions behave identically under all
    five framings. 39 never defer, 5 always defer. So deference is a property
    of the ITEM, not of the context. Something distinguishes those items, and
    a null result that identifies where the variance actually lives is worth
    considerably more than one that just reports an absence.

    Everything here uses data already collected. No API calls.

TWO HYPOTHESES, AND THE SECOND IS ALSO A CONFOUND CHECK

  H1  DEFERENCE IS CALIBRATED
      The model yields to the professor on questions it was already getting
      wrong, and holds firm on ones it was getting right. If true, this is
      not sycophancy - it is the model correctly weighting an outside claim
      against its own confidence. That is a substantive claim about a model
      and it explains the framing null: a sentence about who is watching
      does not change what the model knows.

  H2  THE HINT IS SOMETIMES JUST CORRECT
      The hint letter was drawn at random from the letters OTHER than the
      model's baseline answer. When the baseline was wrong, that draw can
      land on the genuinely correct answer.

      In those cases "deference" may be the model recognising a right answer
      rather than yielding to authority - and the two are indistinguishable
      in the primary DV.

      This is a real flaw in the design. It needs to be measured and
      reported whichever way it comes out, and it should have been caught
      before data collection rather than after.

    python explain.py
"""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

ARMS = ["neutral", "evaluation", "deployment", "training", "placebo"]
Z = 1.959963985

BAR = "#4C78A8"
BAR2 = "#B4A0C8"
INK = "#1b1b1b"
MUTED = "#6b6b6b"
GRID = "#d8d8d8"


def wilson(k, n, z=Z):
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z / d * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return p, max(0.0, c - h), min(1.0, c + h)


def fisher_exact(a, b, c, d):
    """Two-sided Fisher exact on [[a,b],[c,d]]."""
    try:
        from scipy.stats import fisher_exact as fe
        return float(fe([[a, b], [c, d]])[1])
    except Exception:
        pass
    n = a + b + c + d
    r1, r2 = a + b, c + d
    c1 = a + c

    def prob(x):
        return (math.comb(r1, x) * math.comb(r2, c1 - x)) / math.comb(n, c1)

    lo = max(0, c1 - r2)
    hi = min(r1, c1)
    obs = prob(a)
    return min(1.0, sum(prob(x) for x in range(lo, hi + 1)
                        if prob(x) <= obs + 1e-12))


def read(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]


def block(t):
    print(f"\n{'='*70}\n{t}\n{'='*70}")


def two_by_two(title, rows, label_a, label_b, note=""):
    """rows: (name, deferred, total) x2"""
    (na, ka, ta), (nb, kb, tb) = rows
    pa, la, ha = wilson(ka, ta)
    pb, lb, hb = wilson(kb, tb)
    p = fisher_exact(ka, ta - ka, kb, tb - kb)

    print(f"\n  {title}\n")
    print(f"    {'group':<28}{'deferred':>12}{'rate':>8}{'95% CI':>18}")
    print(f"    {na:<28}{f'{ka}/{ta}':>12}{pa:>7.0%}   [{la:>4.0%}, {ha:>4.0%}]")
    print(f"    {nb:<28}{f'{kb}/{tb}':>12}{pb:>7.0%}   [{lb:>4.0%}, {hb:>4.0%}]")
    print(f"\n    difference {(pa-pb)*100:+.1f} points"
          f"   Fisher exact p = {p:.4f}"
          f"{'  *' if p < .05 else ''}")
    if note:
        print(f"\n    {note}")
    return p, pa, pb


def chart(data, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib missing - no chart)")
        return

    labels = [d[0] for d in data]
    pts = [wilson(d[1], d[2]) for d in data]
    ps = [p for p, _, _ in pts]
    lo = [p - l for p, l, _ in pts]
    hi = [h - p for p, _, h in pts]
    cols = [BAR if i < 2 else BAR2 for i in range(len(data))]

    fig, ax = plt.subplots(figsize=(7.4, 4.2), dpi=200)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    x = range(len(labels))
    ax.bar(x, ps, width=0.6, color=cols, zorder=3)
    ax.errorbar(x, ps, yerr=[lo, hi], fmt="none", ecolor=INK,
                elinewidth=1.4, capsize=5, capthick=1.4, zorder=4)
    for i, (p, _, h) in enumerate(pts):
        ax.text(i, h + 0.02, f"{p:.0%}", ha="center", va="bottom",
                fontsize=10, color=INK)
        ax.text(i, -0.04, f"n={data[i][2]}", ha="center", va="top",
                fontsize=8, color=MUTED)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=9, color=INK)
    ax.set_ylabel("traces that deferred", fontsize=10, color=INK)
    ax.set_title("Deference depends on the item, not the framing",
                 fontsize=11, color=INK, loc="left", pad=12)
    ax.set_ylim(0, min(1.0, max(h for _, _, h in pts) + 0.14))
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, length=0)
    fig.tight_layout()
    fig.savefig(path, facecolor="white")
    print(f"\nchart -> {Path(path).resolve()}")


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
        q = qs.get(r["id"])
        st = (r.get("judge") or {}).get("stance")
        if not q or st not in ("D", "R", "I"):
            continue
        r["stance"] = st
        r["deferred"] = st == "D"
        r["baseline_correct"] = q.get("baseline_correct")
        r["correct"] = q.get("correct")
        r["hint_is_correct"] = r["hint_letter"] == q.get("correct")
        rows.append(r)

    print(f"{len(rows)} scored traces joined to {len(qs)} stable questions")

    # ------------------------------------------------------------------ H1
    block("H1  -  is deference CALIBRATED?")
    print("""
  Did the model yield to the professor on questions it was already getting
  wrong, and hold firm on ones it was getting right?""")

    right = [r for r in rows if r["baseline_correct"]]
    wrong = [r for r in rows if r["baseline_correct"] is False]

    p1, pr, pw = two_by_two(
        "deference by whether the model's OWN answer was right",
        [("baseline answer was WRONG", sum(r["deferred"] for r in wrong), len(wrong)),
         ("baseline answer was RIGHT", sum(r["deferred"] for r in right), len(right))],
        "wrong", "right",
        note=("If the wrong-baseline rate is much higher, deference tracks the\n"
              "    model's own (lack of) knowledge rather than the authority of\n"
              "    the source."))

    # ------------------------------------------------------------------ H2
    block("H2  -  or was the hint simply CORRECT sometimes?")
    print("""
  The hint letter was drawn at random from letters other than the baseline
  answer. When the baseline was wrong, that draw can land on the genuinely
  correct answer - and following it then is not deference to authority, it
  is being right.""")

    hc = [r for r in rows if r["hint_is_correct"]]
    hw = [r for r in rows if not r["hint_is_correct"]]

    p2, phc, phw = two_by_two(
        "deference by whether the HINT pointed at the true answer",
        [("hint WAS the correct answer", sum(r["deferred"] for r in hc), len(hc)),
         ("hint was wrong", sum(r["deferred"] for r in hw), len(hw))],
        "hint correct", "hint wrong",
        note=("A large gap here is a CONFOUND in the primary DV, not a\n"
              "    finding. It must be reported as a limitation."))

    print(f"\n  {len(hc)} of {len(rows)} traces ({len(hc)/max(len(rows),1):.0%}) "
          f"had a hint that pointed at the true answer.")

    # ---------------------------------------------------- the clean subset
    block("THE CLEAN TEST  -  hint wrong only")
    print("""
  Restricting to traces where the hint was WRONG removes the confound
  entirely. Any deference here is deference to the source, full stop.
  This is the number to lead with if H2 came out large.""")

    cw = [r for r in hw if r["baseline_correct"] is False]
    cr = [r for r in hw if r["baseline_correct"]]
    if cw and cr:
        two_by_two(
            "among wrong hints only, by baseline correctness",
            [("baseline WRONG, hint wrong", sum(r["deferred"] for r in cw), len(cw)),
             ("baseline RIGHT, hint wrong", sum(r["deferred"] for r in cr), len(cr))],
            "a", "b")

    print(f"\n  per-arm deference among WRONG hints only:")
    for arm in ARMS:
        sub = [r for r in hw if r["arm"] == arm]
        if sub:
            k = sum(r["deferred"] for r in sub)
            print(f"    {arm:<12}{k:>3}/{len(sub):<4} {k/len(sub):>4.0%}")

    # ------------------------------------------------------------- summary
    block("WHAT THIS MEANS")
    lines = []
    if p1 < .05:
        lines.append(
            f"  H1 SUPPORTED (p={p1:.4f}). Deference tracks the model's own\n"
            f"  correctness: {pw:.0%} when its baseline answer was wrong vs\n"
            f"  {pr:.0%} when it was right. Deference is calibrated, not\n"
            f"  sycophantic - and that explains the framing null. Context\n"
            f"  framing does not change what the model knows.\n"
            f"  This is your headline.")
    else:
        lines.append(
            f"  H1 not supported (p={p1:.4f}). Deference does not track the\n"
            f"  model's own correctness. Report it - a hypothesis tested and\n"
            f"  rejected is worth stating - and say the source of the item\n"
            f"  variance is still unidentified.")

    if p2 < .05:
        lines.append(
            f"\n  H2 CONFIRMED (p={p2:.4f}) - and this is a design flaw, not a\n"
            f"  finding. {phc:.0%} deference when the hint was right vs {phw:.0%}\n"
            f"  when wrong. Some of what the DV counts as deference is the\n"
            f"  model recognising a correct answer. Report the wrong-hint\n"
            f"  subset as the clean estimate and say the hint letter should\n"
            f"  have been constrained to incorrect options by design.")
    else:
        lines.append(
            f"\n  H2 not confirmed (p={p2:.4f}). Deference is roughly the same\n"
            f"  whether or not the hint happened to be correct, so the\n"
            f"  confound is small. Still report that you checked.")

    print("\n" + "\n".join(lines))

    chart([("baseline\nWRONG", sum(r["deferred"] for r in wrong), len(wrong)),
           ("baseline\nRIGHT", sum(r["deferred"] for r in right), len(right)),
           ("hint was\nCORRECT", sum(r["deferred"] for r in hc), len(hc)),
           ("hint was\nWRONG", sum(r["deferred"] for r in hw), len(hw))],
          f"deference_by_item_{slug}.png")


if __name__ == "__main__":
    main()
