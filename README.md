# reasoning_vs_nonreasoning_llm_comparison

Do two sibling LLMs — one reasoning-tuned, one not — look at the same question's
words in the same way?

This repo compares **prompt-token attention** between a reasoning-tuned model and
its non-reasoning sibling on an identical base, and reports the difference.

- `Scale-or-Reason/Qwen2.5-1.5B-ift` (non-reasoning / instruction-tuned)
- `Scale-or-Reason/Qwen2.5-1.5B-reasoning`

Same base, same tokenizer, so "reasoning vs not" is the only variable.

---

## The question, and why it needed reframing

The naive version of this experiment compares "how much attention each model puts
on the prompt". That comparison does not work, and the reason is structural, not
a bug:

- The reasoning model **deliberates before answering**. On a one-fact question it
  generated 301 tokens, of which 232 were thinking and 68 were answering.
- The IFT model **does not deliberate at all**. It generated 10 tokens, all of
  them the answer.

Averaging attention over those two generations compares *deliberation* against
*answering*. Worse, comparing "attention on the prompt" as a total mostly
measures how many tokens each model spent doing which job — a length artifact.

So the project drops the "how much" question and asks only about **shape**: the
distribution of attention across the prompt's own words, renormalized to sum to
1. Renormalizing cancels the unequal totals; slicing to a single phase cancels the
unequal generation lengths.

One prompt so far, chosen deliberately: *"Which country has Paris as its
capital?"* — the answer word `Paris` sits in the **middle**, so "attention follows
meaning" and "attention follows recency" predict opposite things for it.

---

## Results

One prompt, one greedy run, IFT vs Reasoning, Qwen2.5-1.5B, T4.

Both models answered correctly (`France`). Both stopped on their own turn
terminator, so no truncation and no repetition loop contaminates either window.

### Headline

| comparison | what it holds fixed | distance |
|---|---|---|
| **C1** Reasoning thinking vs Reasoning answering | the model | **0.0945** |
| **C2** IFT vs Reasoning, both answering | the phase | **0.0663** |

Distance is total variation: **0** = the two attention patterns are identical,
**1** = they have nothing in common.

**Reading these in plain terms:** C2 = 0.066 means about 6.6% of the attention
mass would have to move to make the two models' patterns identical — roughly
**1.7 percentage points per word**, on a prompt where an average word's share is
12.5 points. **The two models distribute attention over the question's words
almost identically.**

C1 = 0.094 is the same-model control. It says: switching from *thinking* to
*answering* changes a model's attention shape by **more** than switching from one
*model* to the other. Deliberating and answering are more different from each
other than the two models are from each other.

### Where the attention goes

Renormalized share of attention per prompt word:

| token | role | IFT | Reasoning (answering) | diff |
|---|---|---|---|---|
| `Which` | question | 5.1% | 3.8% | −1.3 |
| `country` | question | 14.3% | 11.5% | −2.8 |
| `has` | filler | 8.8% | 7.8% | −1.0 |
| `Paris` | **answer** | 13.5% | 12.0% | −1.5 |
| `as` | filler | 8.8% | 8.9% | +0.2 |
| `its` | filler | 5.9% | 6.5% | +0.6 |
| `capital` | answer-attr | 22.0% | 26.9% | **+4.9** |
| `?` | punct | 21.6% | 22.6% | +1.0 |

Both models are dominated by **recency**: the two last words (`capital`, `?`)
take ~44–50% of the mass between them, and the first word (`Which`) takes the
least. Their rank correlation between attention and position is **identical:
+0.524 for both**.

Neither model ignores meaning, though. In IFT, `country` sits at attention rank 3
but position rank 7 — it is attended **four ranks more than its position
explains**, and it is a content word sitting at the front. That is a meaning
effect, not a recency effect.

### The phase effect (C1), in detail

What changes inside the reasoning model when it stops thinking and starts
answering:

| token | thinking | answering | change |
|---|---|---|---|
| `capital` | 20.1% | 26.9% | **+6.9** |
| `?` | 26.6% | 22.6% | −4.0 |
| `Which` | 6.3% | 3.8% | −2.5 |
| `Paris` | 13.9% | 12.0% | −2.0 |

While deliberating, the model leans a little on the question word and the
punctuation — it is recapping the question. While answering, it concentrates on
`capital`. That is what you would expect from a model that has to name the
*country that has Paris as its capital*.

---

## What this establishes, and what it does not

**Established:** on a simple factual prompt, the two models differ very little in
how they distribute attention over the question's words. The attention *shape* is
nearly the same; both are mostly recency-driven with the same modest content
weighting. And the largest shape effect measured is not model identity but
generation phase.

**Not established — the "attention mass" number.** IFT keeps 0.72 of its
attention on the prompt tokens; Reasoning keeps 0.44. That looks like a big
difference, and an earlier version of this analysis reported it as a 6.2×–8.7×
ratio ("IFT reads the prompt N times more"). **That ratio is not a model
property and must not be quoted.** It swings 6.16 → 8.75 just by changing a head
threshold, and it is dominated by the fact that IFT's entire window is
early-generation answer-writing while Reasoning's is mostly deliberation. Shape
says the two models distribute attention almost identically; the mass ratio says
only that they spend different amounts of time in different phases.

**Not established — the attention sink.** Most heads park their attention on
prompt token 0, the `<|im_start|>` format marker, which carries no meaning. The
reasoning model looks sink-dominant at **every** layer 2–26 (mean sink share
0.75–0.97 vs IFT's 0.54–0.86, unanimous direction across 26 layers). Whether
that is model character or just deliberation parking mass on a safe token is
**open** — and it is a *mass* question, which this analysis deliberately drops.

**Not established — anything across prompts.** This is one prompt, greedy
decoding, no variance estimate. A single-run result.

---

## How to run

No GPU locally — all compute runs on Google Colab.

```python
!git clone https://github.com/MuhammadZakiAhmad/reasoning_vs_nonreasoning_llm_comparison.git
%cd reasoning_vs_nonreasoning_llm_comparison
%run attention_visualization_for_non_reasoning_vs_reasoning_llms.py
```

Use `%run`, not `!python`. `%run` executes inside the notebook kernel, so
matplotlib's inline backend renders and models stay cached between runs. `!python`
spawns a subprocess where figures vanish and `!pip` raises `SyntaxError`.

Set `TEST_PROMPT_INDEX` in the experiment file to pick a prompt.

Costs: ~11.5 GiB VRAM for both models in float32 on a 14.56 GiB T4. Attention is
collected one forward pass per generated token, so runtime scales with response
length. The first run in a kernel prints `Collecting ... attention`; every later
run reads the tensor back from a per-VM cache in under a second.

---

## Repo layout

| file | what it is |
|---|---|
| `attention_visualization_for_non_reasoning_vs_reasoning_llms.py` | the experiment. Prompt specs, generation, attention collection, and the Stage A/B/C analyses |
| `models.py` | model loading, cached once per kernel, with an off-GPU check |
| `project_history/` | append-only lab notebook: decisions, errors, dead ends, per-session logs |

The local unit-test harness lives outside the repo (`%TEMP%`) — it `ast`-extracts
the real function source rather than importing the experiment, which would
pip-install and load 12 GB of weights.

---

## Status

Working on the simple factual prompt, with the pipeline verified end to end:
shapes sum to 1, windows add up, generation stops on the models' own terminator,
and the greedy-run values reproduce exactly across Colab sessions.

Next: the same Stage C comparison across a dozen prompt types, to find out whether
the IFT-vs-Reasoning shape distance stays near zero or blows up somewhere.
