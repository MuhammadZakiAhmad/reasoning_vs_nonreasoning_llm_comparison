# Project: Reasoning Vs Non-Reasoning LLM Models

## Overview
- **Goal**: Exploratory. No fixed definition of done — scope expands as interest dictates. Core thread: compare attention behavior of a reasoning LLM against its non-reasoning (IFT) sibling on the same base model, and visualize the difference.
- **Stack**: Python, PyTorch, HuggingFace Transformers (AutoTokenizer / AutoModelForCausalLM), pandas, numpy, matplotlib. Original work in Google Colab (GPU).
- **Started**: 2026-10-05 14:10
- **Project Status**: ACTIVE

## Constraints
None specified on the work itself.

Hard environmental constraint — **there is no GPU on this machine**. All compute runs on Google Colab. This dictates the whole workflow (see Decisions Made):
- Code lives here as `.py` files in a git repo, pushed to GitHub.
- Colab clones/pulls the repo and executes the file with `%run`.
- The script must stay Colab-shaped: `!pip` shell escapes and inline matplotlib are fine and expected. Do NOT strip them.
- Execution must use `%run`, not `!python file.py`. `%run` shares the notebook kernel, so shell escapes and the inline plotting backend work. `!python` spawns a separate process where `!pip` raises SyntaxError and figures never render.

Working note: existing code is a Colab notebook export. Local runs on Windows require manual adaptation (no `!pip` lines, GPU may be absent).

---

## Project Log

### Achievements
<!-- Append-only. Format: [SN-YYYY-MM-DD-HHMM] {what} — {how} — {where} -->
[S1-2026-10-05-1410] Project history initialized — /start-project interview — project_history/

### All Errors Encountered
<!-- Append-only. Format: [SN-YYYY-MM-DD-HHMM] `{error}` — where: {location} — status: {resolved/partial/open} — tried: {what} -->
[S1-2026-10-05-1410] `NameError: name 'load_model_fixed' is not defined` — where: attention_visualization_for_non_reasoning_vs_reasoning_llms.py:64 — status: resolved — tried: renamed the call to `load_model()`. Note the broad `try/except Exception` (62-68) had been swallowing the NameError and printing "Error: ..." while execution continued with both models unbound; the later failure looked unrelated. Prefer narrow exception handling here in future.
[S1-2026-10-05-1410] `NameError: name 'display_attention' is not defined` — where: attention_visualization_for_non_reasoning_vs_reasoning_llms.py:367 (also referenced at 299) — status: resolved — tried: uncommented the function definition (was lines 280-296, present only as comments).
[S1-2026-10-05-1410] Bare DataFrame expressions would not render under `%run` — where: attention_visualization_for_non_reasoning_vs_reasoning_llms.py:373, 398, 541 — status: resolved — tried: wrapped each in `display()`. `%run` does not give cell-style auto-display of the trailing expression, so the results were invisible.
[S1-2026-10-05-1410] `!pip -q install ...` shell escape — where: attention_visualization_for_non_reasoning_vs_reasoning_llms.py:10 — status: resolved (by design, not removed) — tried: nothing needed. Colab/IPython syntax executed under `%run`. Blocked only if the file is ever run as `python file.py`. Cost: reinstalls transformers/accelerate/sentencepiece on every `%run`, adding ~20-30s per execution. Acceptable for now; could be conditioned on a cached check later.
[S1-2026-10-05-1410] Config silently ignored — where: attention_visualization_for_non_reasoning_vs_reasoning_llms.py:38-40 vs 104 — status: open — tried: nothing yet. `DO_SAMPLE = True`, `TEMPERATURE = 0.7`, `TOP_P = 0.9` are declared but `model.generate()` hardcodes `do_sample=False`. Config is dead code; decoding is always greedy. Harmless — greedy is arguably the right choice for a reproducible attention comparison — but misleading.
[S1-2026-10-05-1410] Reasoning model generated twice for identical inputs — where: attention_visualization_for_non_reasoning_vs_reasoning_llms.py:317-332 and 334-349 — status: open — tried: nothing yet. Duplicated block; second call discards the first result. Pure wasted compute. Greedy decode means results are identical, so removing it is behavior-preserving. Costs one extra full generation pass per run.

### Dead Ends (do not retry)
<!-- Append-only. Format: [SN-YYYY-MM-DD-HHMM] {what tried} — why: {reason} — retry if: {condition or "never"} -->

### Decisions Made
<!-- Append-only. Format: [SN-YYYY-MM-DD-HHMM] {decision} — why: {reason} — rejected: {alternatives} — reversible: {yes/no} -->
[S1-2026-10-05-1410] Code-authoring happens locally, compute happens on Colab, repo is the transport — why: this machine has no GPU, so heavy work must run where the GPU is; a git repo is the only handoff that keeps a real version history instead of copy-pasting notebook cells — rejected: editing directly in Colab (no local history, no diffing, no durable record), Google Drive sync (no versioning) — reversible: yes
[S1-2026-10-05-1410] Execution entrypoint is `%run <file>.py` in a Colab cell, not `!python <file>.py` — why: `%run` executes inside the notebook kernel, so `!pip` shell escapes resolve and matplotlib's inline backend renders figures; `!python` is a subprocess where the `!pip` line is a SyntaxError and plots are discarded — rejected: `!python` (breaks both shell escapes and inline display), converting back to .ipynb (loses plain-file diffing) — reversible: yes
[S1-2026-10-05-1410] Script stays Colab-shaped — `!pip` install line and inline `plt.show()` kept rather than made portable — why: Colab is the only target; portability to a standalone host is not a goal and would complicate the file — rejected: making it dual-mode with a `pip install` fallback and `savefig`-only plotting — reversible: yes
[S1-2026-10-05-1410] Remote is https://github.com/MuhammadZakiAhmad/reasoning_vs_nonreasoning_llm_comparison — why: public, so Colab's `!git clone` needs no credential handling — rejected: private repo (requires embedding a PAT in the notebook) — reversible: yes
[S1-2026-10-05-1410] Local commit rebased onto the remote's auto-generated README commit rather than force-pushed — why: preserve upstream history, avoid destroying anything on the remote — rejected: `git push --force` (would have deleted the README commit) — reversible: yes
[S1-2026-10-05-1410] Added `.gitattributes` with `* text=auto eol=lf` — why: Git on Windows wanted to convert the working copy to CRLF (warned on every add); Colab checks out on Linux and LF everywhere removes an entire class of line-ending surprise — rejected: leaving autocrlf unmanaged — reversible: yes
[S1-2026-10-05-1410] Applied only execution-blocking and visibility fixes to the imported script; left the duplicated generation block and per-run pip install alone — why: goal of the first pass is to confirm the Colab pipeline works end to end, so the diff should be small and auditable; unrelated cleanups would muddy that verification — rejected: a full refactor on first import — reversible: yes
[S1-2026-10-05-1410] Model pair = `Scale-or-Reason/Qwen2.5-1.5B-ift` vs `Scale-or-Reason/Qwen2.5-1.5B-reasoning` — why: identical base/tokenizer, so IFT-vs-reasoning is the only variable; same tokenizer removes tokenization alignment work — rejected: comparing across different base families (would confound architecture with training objective) — reversible: yes
[S1-2026-10-05-1410] Load with `torch_dtype=torch.float32` and `attn_implementation="eager"` — why: eager attention is required to get real attention matrices out of `output_attentions=True`; SDPA/flash do not expose them — rejected: default SDPA (fast but returns no attention weights) — reversible: yes
[S1-2026-10-05-1410] Attention collected one generated token at a time against a manually threaded KV cache, taking only the prompt columns (`layer_attn[:, :prompt_length]`) — why: isolates "what did each generated token look at in the prompt", avoiding the self-attention-on-generated-tokens confound — rejected: single forward pass over prompt+generation (drags in generated-token-to-generated-token attention) — reversible: yes
[S1-2026-10-05-1410] Two-stage normalization: mean over (generated_tokens, layers, heads), then renormalize over the user-prompt slice so each model's prompt attention sums to 1 — why: the first mean handles the two models generating different token counts; the second makes the intra-prompt distribution comparable across models — rejected: raw (unnormalized) comparison, unfair when output lengths differ — reversible: yes

### Features
<!-- Append-only. Format: [SN-YYYY-MM-DD-HHMM] {feature name} — status: {planned/in-progress/completed} — {description} -->
[S1-2026-10-05-1410] Dual-model loading — status: completed — loads IFT and reasoning Qwen2.5-1.5B with pad-token fallback and eager attention
[S1-2026-10-05-1410] Greedy response generation — status: completed — chat-template render + generate, returns (prompt_ids, generated_ids, text)
[S1-2026-10-05-1410] Per-generated-token prompt attention collection — status: completed — manual KV-cache loop, shape [generated_tokens, layers, heads, prompt_tokens]
[S1-2026-10-05-1410] Prompt attention aggregation — status: completed — average_prompt_attention() → [prompt_tokens]
[S1-2026-10-05-1410] User-prompt isolation — status: completed — substring match of raw prompt ids inside the chat-template ids, slices attention to real user tokens only
[S1-2026-10-05-1410] Grouped bar chart IFT vs Reasoning — status: completed — matplotlib, per prompt token
[S1-2026-10-05-1410] Colab execution pipeline — status: completed — repo cloned/pulled in Colab, script run via `%run`; script kept Colab-shaped (`!pip` escape + inline plots intact)
[S1-2026-10-05-1410] Scaling/extension — status: planned — more prompts, more layers/heads views, bigger models, head-level analysis

### Important Files
<!-- Append-only. Format: [SN-2026-10-05-1410] {file path} — {description/purpose} -->
[S1-2026-10-05-1410] attention_visualization_for_non_reasoning_vs_reasoning_llms.py — Colab notebook export. Entire pipeline: model loading, generation, KV-cache attention collection, normalization, comparison DataFrame, bar plot. Three defects fixed on import, two left open (see Errors).
[S1-2026-10-05-1410] .gitignore — excludes .claude/, __pycache__, .ipynb_checkpoints, and generated *.png
[S1-2026-10-05-1410] .gitattributes — `* text=auto eol=lf`; keeps the Colab (Linux) checkout free of Windows CRLF
[S1-2026-10-05-1410] project_history/project_details.md — this file. Master append-only log.
[S1-2026-10-05-1410] project_history/sessions/S1-2026-10-05-1410.md — session 1 log.
[S1-2026-10-05-1410] git remote `origin` — https://github.com/MuhammadZakiAhmad/reasoning_vs_nonreasoning_llm_comparison (public, branch main). This is the transport between local authoring and Colab compute.

---

## Session Index
| ID | Date | Status | Working On |
|----|------|--------|------------|
| S1-2026-10-05-1410 | 2026-10-05 | ACTIVE | Understand initial work |
