# LLM usage log — CoT faithfulness project

One line per session, written as I go rather than reconstructed at the end.

Feeds the application question: *"How did you use LLMs in this research task and
write-up? Which LLMs? How exactly did you make sure they weren't just giving you
slop?"*

Structure for each entry: **what it did · what I verified myself · what I did not
verify, and how surprised I'd be to find an error there · what it got wrong.**

---

## Day 1 — Sun 24 Aug · setup, feasibility, background reading

**Model used:** Claude (Opus) in the Claude app, conversationally, for the whole
session. No other assistant.

### What it did

**Provider and model selection.** Compared options on two axes: cost, and whether
the API exposes a raw reasoning trace. Found in Anthropic's docs that no current
Claude model returns raw chain of thought — only `"summarized"` or `"omitted"`.
That ruled Claude out as the subject model: a summarizer sitting between me and
the trace would contaminate the exact quantity I am measuring. Settled on Groq's
free tier, which returns raw traces.

**Code.** Wrote `groq_smoke_test.py` (single-call feasibility test) and
`fetch_questions.py` (MMLU download and sampling).

**Background.** Pointed me to Turpin et al. 2023, Needham et al. 2025, and the
LessWrong review of evaluation awareness. Summarised the relevant findings and
identified four published hint wordings I could adopt rather than invent.

**Design changes it argued for, which I accepted:**
- Adding a second, larger model (`openai/gpt-oss-120b`) because the literature
  reports small models barely react to eval-vs-deployment framing. Running only
  a 27B risks a null I couldn't interpret.
- A second pilot gate on day 3: base verbalization rate must fall between roughly
  5% and 90%, or there's a floor/ceiling effect and the whole design fails.

### What I verified myself

- **Ran the smoke test.** Confirmed a raw trace comes back and parses. The
  central feasibility claim I checked directly rather than trusting.
- **Measured my own throughput** rather than using its estimates: Qwen 3.6 27B
  ≈ 980 output tokens/call, ~5 calls/min; gpt-oss-120b ≈ 190 tokens, ~31/min.
  All scheduling is based on my measured numbers.
- **Built the question bank and read the output.** 400 MMLU questions across
  8 subjects, seed 20260825. Inspected a sample question and the per-subject
  counts.
- **Caught a wrong citation.** It told me Turpin reviewed 234 and 192
  generations. The paper says 426 explanations reviewed, exactly 1 mentioning
  the bias. It had been reading a summary, not the paper. I found this by
  opening the PDF.

### What I did NOT verify

- **The Anthropic summarized-thinking claim.** I read the docs page but did not
  test a Claude call to confirm the trace is genuinely summarized.
  *Surprise if wrong: low-moderate.* The documentation is explicit, but this is
  documentation rather than observation, and it drove a major design decision.
- **The reported verbalization rates** (Claude 3.7 at 25%, R1 at 39%). Taken
  from Anthropic's summary page, not from the paper's tables.
  *Surprise if wrong: moderate.* These set my expected base rate, so if they're
  off my power calculation is off — but day 3's pilot measures my own base rate
  directly, which makes this self-correcting.
- **Third-party pricing figures** used for budgeting. *Surprise if wrong: moderate.*
  No effect on results.

### What went wrong

**A script silently ran the wrong model three times.** I asked for
`openai/gpt-oss-120b` on the command line, but the script I was running had the
model hardcoded and ignored the argument. It returned Qwen traces three runs in a
row while I believed I was testing gpt-oss.

Caught by reading the *shape* of the output rather than trusting the command —
the traces kept opening with the same phrase and using `<think>` tags, which
gpt-oss doesn't produce. Confirmed by grepping the file for the model line.

Consequence for the design: the main runner must write the model name from the
**API response** (`response.model`) into every output row, not the variable set
at the top of the script. Otherwise a mid-run deprecation or silent reroute would
produce a results file that quietly mixes two models.

**I leaked an API key** in a screenshot. Revoked it, issued a new one, moved both
keys into a `.env` file.

### Observations worth keeping

- **Run-to-run variance is large.** Same model, same prompt, 980 tokens one run
  and 631 the next — about 35%. Trace length is noisy, so any comparison of trace
  lengths across arms needs a decent sample.
- **The two models produce completely different objects.** Qwen writes
  3,400–4,800 characters of structured essay with numbered headings and a literal
  "Draft:" section — it composes for a reader. gpt-oss writes ~320 characters of
  terse working note. My dependent variable is whether the trace mentions the
  hint, and a 320-character note has far less room to mention anything.
- **This weakens the size comparison.** The two models differ in scale, trace
  style, and training regime at once, so a difference between them can't be
  cleanly attributed to size. Claude initially presented the two-model design as
  a clean test of the size finding and walked that back when the traces came
  back. Goes in limitations.
- **gpt-oss wrote "No policy violation." unprompted** — it has a trained-in
  self-check step in its reasoning. That is close to what my framings probe, and
  could make it either unusually sensitive to being watched, or invariant because
  the check runs regardless. Worth stating as a prior before I see data.

---

## Day 2 — Mon 25 Aug · prompts and framings

*(fill in tonight)*

**Model used:**

**What it did**

**What I verified myself**

**What I did NOT verify** — and how surprised I'd be by an error there

**What went wrong**

**Observations worth keeping**

---
