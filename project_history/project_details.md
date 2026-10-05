# Project: Reasoning Vs Non-Reasoning LLM Models

## Overview
- **Goal**: Exploratory. No fixed definition of done — scope expands as interest dictates. Core thread: compare attention behavior of a reasoning LLM against its non-reasoning (IFT) sibling on the same base model, and visualize the difference.
- **Stack**: Python, PyTorch, HuggingFace Transformers (AutoTokenizer / AutoModelForCausalLM), pandas, numpy, matplotlib. Original work in Google Colab (GPU).
- **Started**: 2026-10-05 14:10
- **Project Status**: ACTIVE

## Constraints
None specified.

Working note: existing code is a Colab notebook export. Local runs on Windows require manual adaptation (no `!pip` lines, GPU may be absent).

---

## Project Log

### Achievements
<!-- Append-only. Format: [SN-YYYY-MM-DD-HHMM] {what} — {how} — {where} -->
[S1-2026-10-05-1410] Project history initialized — /start-project interview — project_history/

### All Errors Encountered
<!-- Append-only. Format: [SN-YYYY-MM-DD-HHMM] `{error}` — where: {location} — status: {resolved/partial/open} — tried: {what} -->
[S1-2026-10-05-1410] `NameError: name 'load_model_fixed' is not defined` — where: attention_visualization_for_non_reasoning_vs_reasoning_llms.py:64 — status: open — tried: nothing yet. Call is inside a broad `try/except Exception` (lines 62-68), so the NameError is swallowed and only printed as "Error: ...". Consequence: `ift_model` / `reasoning_model` never bind, and the failure surfaces far downstream as a second NameError on first use.
[S1-2026-10-05-1410] `NameError: name 'display_attention' is not defined` — where: attention_visualization_for_non_reasoning_vs_reasoning_llms.py:367 (also referenced at line 299) — status: open — tried: nothing yet. The function body (lines 280-296) exists only as a commented-out block, so the live call at 367 cannot resolve.
[S1-2026-10-05-1410] `!pip -q install ...` shell escape — where: attention_visualization_for_non_reasoning_vs_reasoning_llms.py:10 — status: open — tried: nothing yet. Colab/IPython syntax, not valid standalone Python. Blocks any direct `python file.py` run; also fails if pip install is re-run every execution.
[S1-2026-10-05-1410] Config silently ignored — where: attention_visualization_for_non_reasoning_vs_reasoning_llms.py:38-40 vs 104 — status: open — tried: nothing yet. `DO_SAMPLE = True`, `TEMPERATURE = 0.7`, `TOP_P = 0.9` are declared but `model.generate()` hardcodes `do_sample=False`. Config is dead code; decoding is always greedy.
[S1-2026-10-05-1410] Reasoning model generated twice for identical inputs — where: attention_visualization_for_non_reasoning_vs_reasoning_llms.py:317-332 and 334-349 — status: open — tried: nothing yet. Duplicated block; second call discards the first result. Pure wasted compute (not a crash).

### Dead Ends (do not retry)
<!-- Append-only. Format: [SN-YYYY-MM-DD-HHMM] {what tried} — why: {reason} — retry if: {condition or "never"} -->

### Decisions Made
<!-- Append-only. Format: [SN-YYYY-MM-DD-HHMM] {decision} — why: {reason} — rejected: {alternatives} — reversible: {yes/no} -->
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
[S1-2026-10-05-1410] Local/standalone runnability — status: planned — strip Colab-isms, fix broken calls, add device fallback
[S1-2026-10-05-1410] Scaling/extension — status: planned — more prompts, more layers/heads views, bigger models, head-level analysis

### Important Files
<!-- Append-only. Format: [SN-2026-10-05-1410] {file path} — {description/purpose} -->
[S1-2026-10-05-1410] attention_visualization_for_non_reasoning_vs_reasoning_llms.py — Colab notebook export. Entire pipeline: model loading, generation, KV-cache attention collection, normalization, comparison DataFrame, bar plot. Contains 5 known defects (see Errors).
[S1-2026-10-05-1410] project_history/project_details.md — this file. Master append-only log.
[S1-2026-10-05-1410] project_history/sessions/S1-2026-10-05-1410.md — session 1 log.

---

## Session Index
| ID | Date | Status | Working On |
|----|------|--------|------------|
| S1-2026-10-05-1410 | 2026-10-05 | ACTIVE | Understand initial work |
