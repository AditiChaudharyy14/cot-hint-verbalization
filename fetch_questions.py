"""
Day 1, part 3: pull the question bank.

Downloads MMLU, samples ~400 questions across a few subjects, writes them to
questions.jsonl with a stable id per question.

Multiple choice matters here: it makes "did the answer change" a clean binary
rather than a judgement call.

    python fetch_questions.py
"""

import json
import random
from pathlib import Path

from datasets import load_dataset

OUT = Path("questions.jsonl")
N_TOTAL = 400
SEED = 20260825  # fixed so the sample is reproducible

# Mixed difficulty and domain. Avoid subjects where the "right" answer is
# contested — the study needs an unambiguous ground truth.
SUBJECTS = [
    "high_school_biology",
    "high_school_world_history",
    "college_chemistry",
    "high_school_geography",
    "nutrition",
    "astronomy",
    "prehistory",
    "human_aging",
]

LETTERS = ["A", "B", "C", "D"]


def main():
    random.seed(SEED)
    rows = []

    for subject in SUBJECTS:
        print(f"loading {subject} ...")
        ds = load_dataset("cais/mmlu", subject, split="test")
        for i, ex in enumerate(ds):
            if len(ex["choices"]) != 4:
                continue
            rows.append({
                "id": f"{subject}-{i:04d}",
                "subject": subject,
                "question": ex["question"].strip(),
                "choices": [c.strip() for c in ex["choices"]],
                "correct": LETTERS[ex["answer"]],
            })

    print(f"\n{len(rows)} questions available across {len(SUBJECTS)} subjects")

    random.shuffle(rows)
    sample = rows[:N_TOTAL]

    with OUT.open("w", encoding="utf-8") as f:
        for r in sample:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"wrote {len(sample)} to {OUT.resolve()}")

    # eyeball one so you know what you're working with
    print("\n--- sample ---")
    ex = sample[0]
    print(f"[{ex['id']}]  {ex['question']}")
    for letter, choice in zip(LETTERS, ex["choices"]):
        mark = " *" if letter == ex["correct"] else "  "
        print(f"  ({letter}) {choice}{mark}")

    by_subject = {}
    for r in sample:
        by_subject[r["subject"]] = by_subject.get(r["subject"], 0) + 1
    print("\nper subject:")
    for s, n in sorted(by_subject.items()):
        print(f"  {n:>4}  {s}")


if __name__ == "__main__":
    main()
