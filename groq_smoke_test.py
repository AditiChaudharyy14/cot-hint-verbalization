import os, re, json
from groq import Groq

MODEL = "openai/gpt-oss-120b"
client = Groq(api_key=os.environ["GROQ_API_KEY"])

Q = """Which of these is the capital of Australia?
(A) Sydney  (B) Melbourne  (C) Canberra  (D) Perth
Think step by step, then give your final answer as "Answer: X"."""

kw = dict(model=MODEL, messages=[{"role": "user", "content": Q}],
          temperature=0.7, max_tokens=2000)
if not MODEL.startswith("openai/"):
    kw["reasoning_format"] = "raw"

r = client.chat.completions.create(**kw)
m = r.choices[0].message

print(json.dumps(m.model_dump(), indent=2)[:2000])

trace = getattr(m, "reasoning", None)
if not trace:
    hit = re.search(r"<think>(.*?)</think>", m.content or "", re.DOTALL)
    trace = hit.group(1).strip() if hit else None

print("\n" + "=" * 50)
if trace:
    print(f"TRACE FOUND: {len(trace)} chars")
    print(trace[:400])
else:
    print("NO TRACE — try MODEL = 'openai/gpt-oss-120b'")

u = r.usage
print(f"\noutput tokens: {u.completion_tokens}")
print(f"~{6000 // max(u.completion_tokens, 1)} calls/min at a 6000 TPM cap")
