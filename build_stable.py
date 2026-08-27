"""
Rebuild stable_<model>.jsonl from the raw pre-screen output.

Needed because yesterday's pre-screen died on the daily token limit before it
reached the line that writes the stable file. All the data is in the raw log;
this just re-derives the summary. No API calls.

    python build_stable.py
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openai/gpt-oss-120b")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--questions", default="questions.jsonl")
    args = ap.parse_args()

    slug = args.model.replace("/", "_")
    raw = Path(f"prescreen_raw_{slug}.jsonl")
    out = Path(f"stable_{slug}.jsonl")

    if not raw.exists():
        print(f"can't find {raw}")
        print("check what's there with:  Get-ChildItem *.jsonl")
        return

    qs = {json.loads(l)["id"]: json.loads(l)
          for l in open(args.questions, encoding="utf-8")}

    by_id = defaultdict(list)
    for line in open(raw, encoding="utf-8"):
        row = json.loads(line)
        by_id[row["id"]].append(row)

    stable, unstable, incomplete, unparsed = [], 0, 0, 0

    for qid, rows in by_id.items():
        if len(rows) < args.runs:
            incomplete += 1
            continue
        answers = [r["answer"] for r in rows[:args.runs]]
        if any(a is None for a in answers):
            unparsed += 1
        elif len(set(answers)) == 1:
            q = qs[qid]
            stable.append({**q,
                           "baseline_answer": answers[0],
                           "baseline_correct": answers[0] == q["correct"]})
        else:
            unstable += 1

    with out.open("w", encoding="utf-8") as f:
        for q in stable:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")

    seen = len(stable) + unstable + unparsed
    print(f"questions with data : {len(by_id)}")
    print(f"fully screened      : {seen}")
    print(f"  stable            : {len(stable)}  ({len(stable)/max(seen,1):.0%})")
    print(f"  unstable          : {unstable}")
    print(f"  unparsed          : {unparsed}")
    print(f"  incomplete (<{args.runs} runs) : {incomplete}")

    n_right = sum(q["baseline_correct"] for q in stable)
    print(f"\nof the {len(stable)} stable, {n_right} correct "
          f"({n_right/max(len(stable),1):.0%}), "
          f"{len(stable)-n_right} wrong")
    print("\nThe wrong ones matter: a hint has the most room to move an answer")
    print("the model got wrong but held to consistently.")
    print(f"\nwrote {out.resolve()}")


if __name__ == "__main__":
    main()
