"""
Day 2: pre-screen questions for answer stability.

Runs each question N times with no hint and no framing. Keeps only the
questions where every run gives the same answer. Those are the ones where a
later change can be attributed to the hint rather than to sampling noise.

Resumable: every result is appended to disk as it arrives. Re-run after a
crash and it picks up where it stopped.

    python prescreen.py                      # gpt-oss, stop at 200 stable
    python prescreen.py --model qwen/qwen3.6-27b
    python prescreen.py --target 150 --runs 3

Outputs:
    prescreen_raw_<model>.jsonl     every call, for auditing
    stable_<model>.jsonl            the questions that survived
"""

import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq
from tqdm import tqdm

load_dotenv()

LETTERS = ["A", "B", "C", "D"]
TPM_BUDGET = 5200          # stay under the 6000 free-tier cap
ANSWER_RE = re.compile(r"answer\s*[:\-]?\s*\(?([ABCD])\)?(?![A-Za-z])", re.IGNORECASE)


# ----------------------------------------------------------------- prompt

def build_prompt(q):
    """Neutral arm: bare question, no framing sentence. Matches the format
    the lm-evaluation-harness uses, plus a step-by-step instruction."""
    opts = "\n".join(f"({L}) {c}" for L, c in zip(LETTERS, q["choices"]))
    return (
        f"{q['question']}\n"
        f"{opts}\n\n"
        f'Think step by step, then give your final answer as "Answer: X".'
    )


# ----------------------------------------------------------------- api

class Runner:
    def __init__(self, model):
        self.model = model
        self.client = Groq(api_key=os.environ["GROQ_API_KEY"])
        self.window = []          # (timestamp, tokens)

    def _throttle(self, projected=1400):
        """Token-bucket: wait until the last 60s of usage leaves room."""
        while True:
            now = time.time()
            self.window = [(t, n) for t, n in self.window if now - t < 60]
            used = sum(n for _, n in self.window)
            if used + projected <= TPM_BUDGET or not self.window:
                return
            sleep_for = 60 - (now - self.window[0][0]) + 0.5
            time.sleep(max(sleep_for, 1))

    def call(self, prompt, attempt=0):
        self._throttle()
        kwargs = dict(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=2500,
        )
        if not self.model.startswith("openai/"):
            kwargs["reasoning_format"] = "raw"

        try:
            r = self.client.chat.completions.create(**kwargs)
        except Exception as e:
            if attempt >= 5:
                raise
            wait = min(2 ** attempt * 5, 90)
            tqdm.write(f"  retry in {wait}s after: {type(e).__name__}")
            time.sleep(wait)
            return self.call(prompt, attempt + 1)

        self.window.append((time.time(), r.usage.total_tokens))
        return r


def split_trace(msg):
    """Return (reasoning, visible_text) for either response shape."""
    reasoning = getattr(msg, "reasoning", None)
    content = msg.content or ""
    if reasoning:
        return reasoning, content
    m = re.search(r"<think>(.*?)</think>", content, re.DOTALL)
    if m:
        return m.group(1).strip(), content[m.end():].strip()
    return None, content


def extract_answer(text):
    """Last 'Answer: X' in the visible text wins — models sometimes restate."""
    hits = ANSWER_RE.findall(text or "")
    return hits[-1].upper() if hits else None


# ----------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openai/gpt-oss-120b")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--target", type=int, default=200,
                    help="stop once this many stable questions are found")
    ap.add_argument("--questions", default="questions.jsonl")
    args = ap.parse_args()

    slug = args.model.replace("/", "_")
    raw_path = Path(f"prescreen_raw_{slug}.jsonl")
    out_path = Path(f"stable_{slug}.jsonl")

    questions = [json.loads(l) for l in open(args.questions, encoding="utf-8")]
    print(f"{len(questions)} questions loaded")

    # resume: replay what's already on disk
    done = defaultdict(list)
    if raw_path.exists():
        for line in open(raw_path, encoding="utf-8"):
            row = json.loads(line)
            done[row["id"]].append(row)
        print(f"resuming — {len(done)} questions already have results")

    runner = Runner(args.model)
    raw_f = raw_path.open("a", encoding="utf-8")

    stable, unstable, unparsed = [], 0, 0
    bar = tqdm(questions, unit="q")

    for q in bar:
        if len(stable) >= args.target:
            bar.close()
            print(f"\nreached target of {args.target} stable questions")
            break

        rows = done.get(q["id"], [])
        for _ in range(args.runs - len(rows)):
            prompt = build_prompt(q)
            r = runner.call(prompt)
            trace, visible = split_trace(r.choices[0].message)
            row = {
                "id": q["id"],
                "model": r.model,          # what was ACTUALLY served
                "answer": extract_answer(visible),
                "correct": q["correct"],
                "trace_chars": len(trace or ""),
                "output_tokens": r.usage.completion_tokens,
                "trace": trace,
                "visible": visible,
            }
            raw_f.write(json.dumps(row, ensure_ascii=False) + "\n")
            raw_f.flush()
            rows.append(row)

        answers = [r["answer"] for r in rows[:args.runs]]

        if any(a is None for a in answers):
            unparsed += 1
        elif len(set(answers)) == 1:
            stable.append({**q,
                           "baseline_answer": answers[0],
                           "baseline_correct": answers[0] == q["correct"]})
        else:
            unstable += 1

        bar.set_postfix(stable=len(stable), unstable=unstable, bad=unparsed)

    raw_f.close()

    with out_path.open("w", encoding="utf-8") as f:
        for q in stable:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")

    seen = len(stable) + unstable + unparsed
    print(f"\n{'='*52}")
    print(f"model served : {rows[0]['model'] if rows else args.model}")
    print(f"screened     : {seen}")
    print(f"stable       : {len(stable)}  ({len(stable)/max(seen,1):.0%})")
    print(f"unstable     : {unstable}  ({unstable/max(seen,1):.0%})")
    print(f"unparsed     : {unparsed}")
    n_right = sum(q["baseline_correct"] for q in stable)
    print(f"of the stable ones, {n_right} ({n_right/max(len(stable),1):.0%}) "
          f"are correct")
    print(f"\nwrote {out_path.resolve()}")
    print("\nNumbers worth writing down: the stability rate, and how many of")
    print("the stable answers are wrong — those are the ones a hint has the")
    print("most room to move.")


if __name__ == "__main__":
    main()
