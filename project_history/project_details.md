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
[S1-2026-10-05-1410] Repo created and pushed — git init, rebase onto remote README commit, push to main — github.com/MuhammadZakiAhmad/reasoning_vs_nonreasoning_llm_comparison
[S1-2026-10-05-1410] Script executes end-to-end up to the reasoning model — Colab T4, `%run` after `!git clone` — got PyTorch 2.11.0+cu130, CUDA True, T4 14.56 GB, both models loaded, IFT generation complete, `Attention tensor shape: torch.Size([256, 28, 12, 60])`
[S1-2026-10-05-1410] First real attention tensor produced — 256 generated tokens x 28 layers x 12 heads x 60 prompt tokens, exactly the designed shape — means load/generate/collect chain is correct
[S1-2026-10-05-1410] Colab execution path proven — `!git clone` -> `%cd` -> `%run <file>.py` — this is now the established loop for all future work
[S1-2026-10-05-1410] First complete comparison produced — both models generated, both attention tensors collected, per-token table emitted for the 31 user-prompt tokens of the chickens/cows prompt — IFT and Reasoning columns each sum to exactly 1.0, and the difference column sums to 0, confirming the two-stage normalization behaves as designed
[S1-2026-10-05-1410] Attention sanity-checked against known-good patterns — content words attract attention (`legs` 0.106, `.` 0.087, `?` 0.061, `cows` 0.053), function words do not (`has` 0.011, `are` 0.013, `in` 0.013, `A` 0.014). This is the expected shape for a word problem, so the extraction path is producing real attention rather than noise or a degenerate tensor

### All Errors Encountered
<!-- Append-only. Format: [SN-YYYY-MM-DD-HHMM] `{error}` — where: {location} — status: {resolved/partial/open} — tried: {what} -->
[S1-2026-10-05-1410] `NameError: name 'load_model_fixed' is not defined` — where: attention_visualization_for_non_reasoning_vs_reasoning_llms.py:64 — status: resolved — tried: renamed the call to `load_model()`. Note the broad `try/except Exception` (62-68) had been swallowing the NameError and printing "Error: ..." while execution continued with both models unbound; the later failure looked unrelated. Prefer narrow exception handling here in future.
[S1-2026-10-05-1410] `NameError: name 'display_attention' is not defined` — where: attention_visualization_for_non_reasoning_vs_reasoning_llms.py:367 (also referenced at 299) — status: resolved — tried: uncommented the function definition (was lines 280-296, present only as comments).
[S1-2026-10-05-1410] Bare DataFrame expressions would not render under `%run` — where: attention_visualization_for_non_reasoning_vs_reasoning_llms.py:373, 398, 541 — status: resolved — tried: wrapped each in `display()`. `%run` does not give cell-style auto-display of the trailing expression, so the results were invisible.
[S1-2026-10-05-1410] `SyntaxError: invalid syntax` at `!pip -q install -U transformers accelerate sentencepiece` — where: attention_visualization_for_non_reasoning_vs_reasoning_llms.py:10, raised in Colab via `%run` — status: resolved — tried: replaced with `subprocess.check_call([sys.executable, "-m", "pip", "install", ...])`. Root cause was a wrong assumption on my part that `%run` transforms `!` shell escapes; it transforms `%magic` lines only. Symptom was total failure to execute — nothing past line 10 ever ran.
[S1-2026-10-05-1410] `!pip -q install ...` shell escape — where: attention_visualization_for_non_reasoning_vs_reasoning_llms.py:10 — status: resolved (line removed, not merely tolerated) — tried: as above. Original note here wrongly labelled it harmless; it was fatal under `%run`. Cost remains: deps reinstall on every run (~20-30s).
[S1-2026-10-05-1410] Config silently ignored — where: attention_visualization_for_non_reasoning_vs_reasoning_llms.py:38-40 vs 104 — status: open — tried: nothing yet. `DO_SAMPLE = True`, `TEMPERATURE = 0.7`, `TOP_P = 0.9` are declared but `model.generate()` hardcodes `do_sample=False`. Config is dead code; decoding is always greedy. Harmless — greedy is arguably the right choice for a reproducible attention comparison — but misleading.
[S1-2026-10-05-1410] `WARNING:accelerate.big_modeling:Some parameters are on the meta device because they were offloaded to the cpu.` then a multi-minute stall — where: attention_visualization_for_non_reasoning_vs_reasoning_llms.py:323 (reasoning reload), observed on Colab T4 14.56 GB — status: resolved — tried: removed the redundant reload. Root cause: both models are loaded near the top of the file, then the reasoning model was loaded a *second* time. With the IFT model still resident, that second float32 copy (~6 GB) exceeded VRAM, so accelerate silently offloaded to CPU. Nothing raised an exception — the run just stalled, which is the nastiest failure mode in this file. Diagnostic rule: any `meta device` / offload warning means VRAM is exhausted, not that the load is slow.
[S1-2026-10-05-1410] Reasoning model generated twice for identical inputs — where: attention_visualization_for_non_reasoning_vs_reasoning_llms.py:317-332 and 334-349 (pre-fix numbering) — status: resolved — tried: deleted the duplicated block. Previously logged as "wasted compute"; after the OOM diagnosis it also costs a full extra 256-token generation pass against exhausted VRAM, so it is no longer merely wasteful.
[S1-2026-10-05-1410] IFT response truncated mid-sentence — where: Colab run, `MAX_NEW_TOKENS = 256` at line 34 — status: resolved — tried: raised to 1024 and added a warning inside `generate_response()` when generation stops at the cap, so truncation can never pass silently again. The IFT reply had been cut at `x = 10 - y = 1` (mid-equation). Because the attention average covers generated tokens, a truncated response means the average covered an incomplete answer.
[S1-2026-10-05-1410] Chart rendered as an empty figure — where: `plt.show()` at the end of the plotting block — status: resolved — tried: replaced the pyplot state-machine with explicit `fig, ax = plt.subplots(...)` plus `display(fig)`. Colab printed `<Figure size 640x480 with 0 Axes>` (default size, no axes) rather than the 14x6 chart. Under `%run` there is no cell output area for the inline backend to draw into, so `plt.show()` does not render.
[S1-2026-10-05-1410] `[transformers] torch_dtype is deprecated! Use dtype instead!` — where: attention_visualization_for_non_reasoning_vs_reasoning_llms.py:54 — status: resolved — tried: renamed the kwarg to `dtype=`. Cosmetic; the installed transformers accepts both.
[S1-2026-10-05-1410] `Access to the secret HF_TOKEN has not been granted` + `You are sending unauthenticated requests to the HF Hub` — where: Colab, model download — status: open (benign) — tried: nothing. Unauthenticated downloads worked; these are Colab/Hub rate-limit notices, not errors affecting the run. Would matter only under heavy repeated downloading.

### Dead Ends (do not retry)
<!-- Append-only. Format: [SN-YYYY-MM-DD-HHMM] {what tried} — why: {reason} — retry if: {condition or "never"} -->

### Decisions Made
<!-- Append-only. Format: [SN-YYYY-MM-DD-HHMM] {decision} — why: {reason} — rejected: {alternatives} — reversible: {yes/no} -->
[S1-2026-10-05-1410] Code-authoring happens locally, compute happens on Colab, repo is the transport — why: this machine has no GPU, so heavy work must run where the GPU is; a git repo is the only handoff that keeps a real version history instead of copy-pasting notebook cells — rejected: editing directly in Colab (no local history, no diffing, no durable record), Google Drive sync (no versioning) — reversible: yes
[S1-2026-10-05-1410] Execution entrypoint is `%run <file>.py` in a Colab cell, not `!python <file>.py` — why: `%run` executes inside the notebook kernel, so matplotlib's inline backend renders figures and kernel state persists; `!python` spawns a subprocess where plots are discarded — rejected: `!python` (loses inline display), converting back to .ipynb (loses plain-file diffing) — reversible: yes

CORRECTION [S1-2026-10-05-1410]: the original entry here also claimed `%run` transforms `!` shell escapes. **That is false.** `%run` transforms `%magic` lines but not `!` escapes — a `!pip` line in a `%run` file raises SyntaxError. The script therefore had to become plain-Python-valid; see the corresponding entry below and the SyntaxError in All Errors Encountered.
[S1-2026-10-05-1410] The script must be valid plain Python, not "Colab-shaped" — `!pip` install replaced by `subprocess.check_call([sys.executable, "-m", "pip", "install", ...])` — why: `%run` does not accept `!` escapes, so notebook-only syntax makes the file unrunnable; the subprocess form installs into the running kernel's environment on Colab while staying valid Python anywhere — rejected: keeping `!pip` and requiring a separate install cell in Colab (works, but leaves a file that cannot be executed as-is), `get_ipython().system(...)` (needs IPython in scope, fails outside notebooks) — reversible: yes
[S1-2026-10-05-1410] Source file verified with `python -m py_compile` before pushing — why: catches exactly this class of notebook-syntax leak locally, before a Colab round-trip reveals it — rejected: relying on Colab runs as the only syntax check — reversible: yes

OPEN QUESTION [S1-2026-10-05-1410]: **the IFT model does not look "non-reasoning".** Its reply to the chickens/cows prompt was a full worked solution — define variables, set up two equations, solve by substitution, in markdown with LaTeX — not a direct answer. That flattens the project's central contrast. Either (a) the "ift" checkpoint was trained on reasoning-style outputs too, in which case "reasoning vs non-reasoning" is the wrong framing and the real axis is something else (length? trace verbosity? self-verification?), or (b) the frozen prompt (`prompts[2]`, a multi-step math word problem) is itself eliciting step-by-step output from both models, and an easier prompt like `prompts[0]` ("capital of France") would separate them properly. Not resolved this session. Verify before building anything on top of the comparison.

OPEN QUESTION [S1-2026-10-05-1410]: **the per-prompt renormalization erases the likeliest signal.** `ift_user_attention / ift_user_attention.sum()` forces each model's user-token attention to sum to 1, so "how much attention stayed on the prompt at all" cannot appear in the table. That share is arguably the first place a reasoning model should differ (locking onto its own generated trace instead of re-reading the question). Mitigation added: a `Prompt attention mass` printout of the pre-normalization sum for both models. Evaluate that number before trusting any redistribution in the table.

OBSERVATION [S1-2026-10-05-1410]: the IFT/Reasoning difference has a coherent positional structure, not scattered noise. On the chickens/cows prompt, reasoning > IFT across prompt tokens 0-8 (`A farmer has chickens and cows . There are`) and IFT > reasoning across tokens 10-27 (the digits, `animals`, `legs`, `.`, `cows`), with the closing question clause near zero. Plausible reading: the reasoning model weights the problem setup and entities, IFT weights the numeric constraints. **Do not conclude anything from this yet** — magnitudes are small (mean |diff| ~0.004, max 0.019 across a 1.0-scale column) and it rests on a single prompt, greedy decoding, and one run with no variance estimate.

OPEN QUESTION [S1-2026-10-05-1410]: **averaging over all generated tokens conflates generation phases.** Early tokens are the model reading the problem; late tokens are it working the algebra. A model that reasons longer spends proportionally more of its average in the solving phase, so the averaged comparison is partly measuring response length rather than attention style. Candidate fix (not yet built): bucket generated tokens by position (e.g. quartiles of the generation) and compare like phase with like phase.
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
