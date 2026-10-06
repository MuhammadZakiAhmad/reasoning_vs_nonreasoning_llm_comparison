# -*- coding: utf-8 -*-
"""Attention Visualization for Non Reasoning Vs Reasoning LLMs

Compare prompt-token attention between a reasoning-tuned LLM and its
non-reasoning (IFT) sibling, both on the same Qwen2.5-1.5B base.

Run in Colab:

    !git clone https://github.com/MuhammadZakiAhmad/reasoning_vs_nonreasoning_llm_comparison.git
    %cd reasoning_vs_nonreasoning_llm_comparison
    %run attention_visualization_for_non_reasoning_vs_reasoning_llms.py

The models are loaded by models.py, which caches them for the life of the
kernel. Editing this file and re-running it does NOT reload them -- only a
runtime restart does.

Original notebook:
    https://colab.research.google.com/drive/1eezvrVvmYXf_veCtAFr6CNu1y9M-w_VG
"""

import hashlib
import os
import sys
import tempfile

# Make the repo root importable when this file is executed via %run in Colab.
if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

# display() renders objects into the Colab output. Imported explicitly rather
# than relying on it being pre-injected into the kernel namespace.
from IPython.display import display

from models import get_model

# Cap on generated tokens. Both models answer these prompts in long form (the
# IFT model writes a full worked solution), so 256 was truncating them
# mid-sentence. 1024 is enough for a complete answer on these prompts.
#
# Cost warning: collect_prompt_attention() runs one forward pass per generated
# token, so runtime is linear in generated length -- 1024 is roughly 4x the
# work of 256, per model.
MAX_NEW_TOKENS = 1024

# Where a completed (generation + attention) run is cached, for the life of the
# Colab VM.
#
# Why this exists: `%run` re-executes this whole file, and the attention
# collection costs one forward pass per generated token -- over 2000 passes for
# the two models at a 1024-token cap. Without a cache, changing a single print
# statement costs a full regeneration. With one, only the first run in a kernel
# pays that, and every later run reads the tensor off disk in well under a
# second.
#
# Deliberately NOT written into the repo: the cache is a throwaway, nothing
# reads it but this script, and a stale file must never be mistaken for source.
# On Colab it lands in /content, which is wiped when the VM dies -- so like the
# model cache in models.py, it is per-kernel-session by construction.
if os.path.isdir("/content"):
    CACHE_DIR = "/content/attn_cache"
else:
    CACHE_DIR = os.path.join(tempfile.gettempdir(), "attn_cache")

# Which entry of PROMPT_SPECS to analyse.
#
# Now 0 ("capital of France"). The chickens/cows prompt (index 2) got a full
# worked solution out of BOTH models, so it could not settle whether the IFT
# checkpoint is really non-reasoning. A trivial factual question is the test:
# if IFT writes a worked solution here too, the reasoning-vs-non-reasoning
# framing is dead and the project needs reframing.
TEST_PROMPT_INDEX = 0

# Declared but NOT currently used: generate_response() hardcodes
# do_sample=False, so decoding is always greedy. See project history.
DO_SAMPLE = True
TEMPERATURE = 0.7
TOP_P = 0.9

# Loads from the Hub on the first run in a kernel; an instant no-op on every
# run after that, so iterating on this file costs no reload.
ift_tokenizer, ift_model = get_model("ift")
reasoning_tokenizer, reasoning_model = get_model("reasoning")

prompts = [
    "What is the capital of France?",

    "If a train travels 60 kilometers in 1 hour, how far will it travel in 3 hours?",

    "A farmer has chickens and cows. There are 10 animals in total and 28 legs. How many chickens and how many cows are there?",

    "Why does ice float on water?",

    "John is older than Mary. Mary is older than Sarah. Who is the youngest?"
]

len(prompts)

# ============================================================
# GROUND TRUTH AND ANSWER-CHECKING
# ============================================================
#
# `must_contain` entries are expected to appear in a correct response.
# The check is deliberately crude: it looks for marker strings, so it will
# catch an obviously wrong answer but NOT a right-answer-wrong-reasoning, a
# subtly wrong number, or a marker that appears incidentally.
#
# Treat the verdict as a flag to go read the response, never as proof.

PROMPT_SPECS = [
    {
        "prompt": "What is the capital of France?",
        "ground_truth": "Paris",
        "must_contain": ["paris"],
    },
    {
        "prompt": "If a train travels 60 kilometers in 1 hour, how far will it travel in 3 hours?",
        "ground_truth": "180 kilometers",
        "must_contain": ["180"],
    },
    {
        "prompt": "A farmer has chickens and cows. There are 10 animals in total and 28 legs. How many chickens and how many cows are there?",
        "ground_truth": "6 chickens and 4 cows",
        "must_contain": ["6", "4"],
    },
    {
        "prompt": "Why does ice float on water?",
        "ground_truth": "Ice is less dense than liquid water",
        "must_contain": ["dens"],
    },
    {
        "prompt": "John is older than Mary. Mary is older than Sarah. Who is the youngest?",
        "ground_truth": "Sarah",
        "must_contain": ["sarah"],
    },
]


# Closing tag that ends the reasoning model's trace. Guessed from the '<think>'
# opening tag seen in output -- VERIFY against a non-truncated reasoning
# response. If the real closing tag differs, final_answer_region() returns None
# and every reasoning verdict reads NO FINAL ANSWER (which is at least honest,
# but wrong about why).
REASONING_END_TAGS = ("</think>", "</thinking>")


def final_answer_region(response_text):
    """Text after the reasoning trace closes, or None if it never closed.

    None means the model was still reasoning when generation stopped. There is
    then no answer to grade, and any marker found in the raw text sits inside
    the working-out, not in an answer.
    """

    for tag in REASONING_END_TAGS:
        idx = response_text.find(tag)

        if idx != -1:
            return response_text[idx + len(tag):]

    return None


def check_answer(response_text, spec, scope="full"):
    """Marker-match verdict over `scope`, reporting WHERE each marker matched.

    scope='full'  -- whole response; right for a model that answers directly
    scope='final' -- only after the reasoning trace closes

    Deliberately crude: it catches an obviously wrong answer, NOT a
    right-answer-wrong-reasoning or a marker that appears incidentally. The
    positions are the point of returning a dict -- an answer found only inside
    a working-out is not an answer, and only the positions reveal that.
    """

    if scope == "final":
        region = final_answer_region(response_text)
    else:
        region = response_text

    if region is None:
        return {
            "verdict": "NO FINAL ANSWER",
            "hits": [],
            "misses": list(spec["must_contain"]),
            "positions": {},
            "scope_chars": 0,
        }

    text = region.lower()

    hits = [m for m in spec["must_contain"] if m in text]
    misses = [m for m in spec["must_contain"] if m not in text]

    positions = {m: text.find(m) for m in hits}

    if not misses:
        verdict = "CORRECT"
    elif hits:
        verdict = "PARTIAL"
    else:
        verdict = "INCORRECT"

    return {
        "verdict": verdict,
        "hits": hits,
        "misses": misses,
        "positions": positions,
        "scope_chars": len(region),
    }


def find_degenerate_tail(generated_ids, max_period=256, min_run_tokens=40):
    """Find a periodic tail -- the model looping after it finished answering.

    Returns (start_index, period), or (None, None) when the generation ends
    normally.

    Detection: for each candidate period p, find the last position where the
    sequence breaks its own p-periodicity (ids[i] != ids[i - p]). Everything
    after that break is an unbroken stretch of p-token repeats. A stretch of at
    least min_run_tokens that also spans at least two full periods is a
    collapse; the smallest such start wins, because if the tail is p-periodic
    from index k then it IS a loop from k for any p that holds.

    Note the off-by-one-period: periodicity is only *detectable* from the second
    copy onward, since the first copy has nothing before it to match against.
    The confirmed start is therefore walked back one period, to the beginning of
    that first copy -- which is where the repetition actually starts.

    Why periodicity rather than the obvious "longest repeated n-gram": the
    observed loops are whole *sentences*. The IFT model repeated a ~30-token
    sentence to the 1024 cap, and a fixed small n-gram window misses that
    entirely -- as the France run demonstrated, reporting "no collapse" while
    ~75% of the generation was a chat loop. Sweeping the period catches loops
    of any length up to max_period, and returning the period says which kind of
    loop it was.

    Known limit: only the TAIL is examined, which matches "model finished, then
    never stopped". A loop that starts mid-generation and then recovers is not
    detected.
    """

    ids = generated_ids.tolist()
    total = len(ids)

    best_start = None
    best_period = None

    for period in range(1, min(max_period, total - 1) + 1):

        last_break = -1

        for i in range(period, total):
            if ids[i] != ids[i - period]:
                last_break = i

        # Back up one period, to the start of the first unbroken copy.
        start = max(0, last_break + 1 - period)
        run = total - start

        if run >= min_run_tokens and run >= 2 * period:
            if best_start is None or start < best_start:
                best_start = start
                best_period = period

    return best_start, best_period

@torch.no_grad()
def generate_response(model, tokenizer, prompt):

    messages = [
        {"role": "user", "content": prompt}
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    ).to(model.device)

    prompt_length = inputs["input_ids"].shape[1]

    output_ids = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        use_cache=True,
        pad_token_id=tokenizer.pad_token_id
    )

    generated_ids = output_ids[0, prompt_length:]

    generated_text = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True
    )

    if generated_ids.shape[0] >= MAX_NEW_TOKENS:
        print(
            f"NOTE: generation reached the {MAX_NEW_TOKENS}-token cap. That means "
            "EITHER the answer is truncated OR the model finished and then fell "
            "into a repetition loop. find_degenerate_tail() below separates the "
            "two; read the response to confirm which."
        )

    return (
        inputs["input_ids"][0].detach().cpu(),
        generated_ids.detach().cpu(),
        generated_text
    )

@torch.no_grad()
def collect_prompt_attention(
    model,
    tokenizer,
    prompt,
    generated_ids
):

    messages = [
        {"role": "user", "content": prompt}
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    ).to(model.device)

    prompt_ids = inputs["input_ids"]

    prompt_length = prompt_ids.shape[1]
    generated_ids = generated_ids.to(model.device)

    # ---------------------------------------------------------
    # Step 1:
    # Run the original prompt once to create the KV cache.
    #
    # We don't need its attention values.
    # ---------------------------------------------------------

    outputs = model(
        input_ids=prompt_ids,
        use_cache=True,
        output_attentions=False,
        return_dict=True
    )

    past_key_values = outputs.past_key_values

    # We will store:
    #
    # generated token
    #       ↓
    # layer
    #       ↓
    # head
    #       ↓
    # original prompt tokens
    #
    attention_rows = []

    # ---------------------------------------------------------
    # Step 2:
    # Feed each generated token through the model.
    # ---------------------------------------------------------

    for token_id in generated_ids:

        token_input = token_id.view(1, 1)

        outputs = model(
            input_ids=token_input,
            past_key_values=past_key_values,
            use_cache=True,
            output_attentions=True,
            return_dict=True
        )

        # outputs.attentions is a tuple:
        #
        # one tensor per layer
        #
        # Each tensor has shape approximately:
        # [batch, heads, query_length, key_length]

        layer_attention = []

        for layer_attn in outputs.attentions:

            # Since we supplied one token, query_length = 1.
            #
            # Shape:
            # [1, heads, 1, total_sequence_length]

            layer_attn = layer_attn[0, :, 0, :]

            # Keep ONLY the columns belonging to the
            # ORIGINAL PROMPT.

            prompt_attn = layer_attn[:, :prompt_length]

            layer_attention.append(
                prompt_attn.detach().float().cpu()
            )

        # [layers, heads, prompt_tokens]
        layer_attention = torch.stack(layer_attention)

        attention_rows.append(layer_attention)

        # Update KV cache for the next generated token.
        past_key_values = outputs.past_key_values

    # [generated_tokens, layers, heads, prompt_tokens]
    attention_tensor = torch.stack(attention_rows)

    return attention_tensor

def average_prompt_attention(attention_tensor):
    """
    Input:
        [generated_tokens, layers, heads, prompt_tokens]

    Output:
        [prompt_tokens]

    Each value is:
        average attention from generated tokens
        to that particular original prompt token.
    """

    return attention_tensor.mean(dim=(0, 1, 2))


def cache_path(which, prompt):
    """Cache file for one (model, prompt, token cap) combination.

    The cap is in the filename, so raising MAX_NEW_TOKENS invalidates the cache
    rather than silently reusing a run that stopped earlier.
    """

    digest = hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:12]

    return os.path.join(CACHE_DIR, f"{which}_{digest}_{MAX_NEW_TOKENS}.npz")


def run_and_cache(which, model, tokenizer, prompt):
    """Generate, collect attention, and cache the result on disk.

    Returns prompt_ids / generated_ids / text / attention / trim_start /
    trim_period.

    The attention is collected over the FULL untrimmed generation. That is the
    point of the cache: a generated token's attention depends only on itself and
    the tokens before it, so row i is identical whether the run stopped at 1024
    tokens or at 400. Paying for the full collection once therefore makes every
    future change to the trim rule free -- including the drift-boundary trim on
    the wish list. Averaging over a prefix of the tensor reproduces exactly what
    collecting only that prefix would have produced.
    """

    path = cache_path(which, prompt)

    if os.path.exists(path):

        cached = np.load(path, allow_pickle=False)

        trim_start = int(cached["trim_start"])

        print(f"{which}: reusing cached run ({os.path.basename(path)})")

        return {
            "prompt_ids": torch.from_numpy(cached["prompt_ids"]),
            "generated_ids": torch.from_numpy(cached["generated_ids"]),
            "text": str(cached["text"]),
            "attention": torch.from_numpy(cached["attention"]),
            "trim_start": None if trim_start < 0 else trim_start,
            "trim_period": None if int(cached["trim_period"]) < 0 else int(cached["trim_period"]),
        }

    prompt_ids, generated_ids, text = generate_response(model, tokenizer, prompt)

    print(f"\nCollecting {which} attention (full untrimmed generation)...")

    attention = collect_prompt_attention(model, tokenizer, prompt, generated_ids)

    print(f"{which} attention tensor shape:", tuple(attention.shape))

    trim_start, trim_period = find_degenerate_tail(generated_ids)

    os.makedirs(CACHE_DIR, exist_ok=True)

    np.savez(
        path,
        prompt_ids=prompt_ids.numpy(),
        generated_ids=generated_ids.numpy(),
        text=np.array(text),
        attention=attention.numpy().astype(np.float32),
        trim_start=-1 if trim_start is None else trim_start,
        trim_period=-1 if trim_period is None else trim_period,
    )

    print(f"{which}: cached run to {os.path.basename(path)}")

    return {
        "prompt_ids": prompt_ids,
        "generated_ids": generated_ids,
        "text": text,
        "attention": attention,
        "trim_start": trim_start,
        "trim_period": trim_period,
    }


# ============================================================
# STAGE A HELPERS: WHO IS A SINK HEAD, AND WHO IS A READER?
# ============================================================
#
# Everything the headline comparison does averages over all 28 layers and all 12
# heads at once -- 336 heads mashed into one number per prompt token. Most of
# those heads turn out to park their attention on prompt token 0 and carry no
# information, so that average is dominated by the parking and the IFT-vs-
# Reasoning gap is largely a gap in sink size.
#
# These two functions exist to break the average apart. Both take a plain numpy
# array so they can be unit-tested locally without torch or the models.

# Position 0 of the rendered chat template: '<|im_start|>'. The classic
# attention sink -- always visible, positionally stable, semantically empty.
SINK_INDEX = 0


def sink_profile(attention, sink_index=SINK_INDEX):
    """Sink share and total prompt attention, per (layer, head).

    Input: [generated_tokens, layers, heads, prompt_tokens].

    share[l, h] is the fraction of head (l, h)'s prompt attention that lands on
    sink_index. mass[l, h] is how much prompt attention that head carried in the
    first place.

    Both numbers matter. A head with share 0.05 and mass 0.0001 is a reader, but
    not a loud one, and a 5% reader holding almost no attention cannot carry a
    comparison on its own.
    """

    per_head = attention.mean(axis=0)

    mass = per_head.sum(axis=-1)

    share = np.divide(
        per_head[..., sink_index],
        mass,
        out=np.zeros_like(mass),
        where=mass > 0
    )

    return share, mass


def keep_reader_heads(attention, share, threshold, sink_index=SINK_INDEX):
    """Average prompt attention using only the heads that are not sink-dominant.

    Returns (prompt_vector, retained_share_of_mass, kept, total_heads).

    retained_share_of_mass is the number to read first. If the readers hold only
    a percent or two of the total, then the comparison built on them is a
    comparison of a percent or two of the signal, and should be reported that
    way rather than as if it were the whole picture.

    Note the sink is zeroed rather than renormalized away: dropping the sink's
    mass and rescaling would hide exactly the quantity we are trying to see.
    """

    keep = share < threshold

    per_head = attention.mean(axis=0) * keep[..., None]

    prompt_vector = per_head.sum(axis=(0, 1))

    total = attention.mean(axis=0).sum()

    retained = float(per_head.sum() / total) if total > 0 else 0.0

    return prompt_vector, retained, int(keep.sum()), int(keep.size)

test_prompt = prompts[TEST_PROMPT_INDEX]
test_spec = PROMPT_SPECS[TEST_PROMPT_INDEX]

print("PROMPT:")
print(test_prompt)

print("\nGenerating with IFT model...")

ift_run = run_and_cache("ift", ift_model, ift_tokenizer, test_prompt)

ift_prompt_ids = ift_run["prompt_ids"]
ift_generated_ids = ift_run["generated_ids"]
ift_text = ift_run["text"]

print("\nIFT RESPONSE:")
print(ift_text)

# Drop any repetition collapse before averaging. Averaging over a loop measures
# the loop, not the solution -- the IFT model answers "capital of France" in one
# line and then chats to the cap, so without this its average is mostly a
# conversation loop.
#
# The old n-gram version of this check MISSED that loop: the repeated unit is a
# ~30-token sentence, well beyond any small fixed n-gram window. It now sweeps
# the period instead, so the loop is caught and the reported period says what
# kind of loop it was.
ift_degenerate_start = ift_run["trim_start"]
ift_degenerate_period = ift_run["trim_period"]

if ift_degenerate_start is None:
    ift_attention_ids = ift_generated_ids
    print(f"\nNo repetition collapse. Using all {len(ift_generated_ids)} generated tokens.")
else:
    ift_attention_ids = ift_generated_ids[:ift_degenerate_start]
    print(
        f"\nREPETITION COLLAPSE at generated token {ift_degenerate_start} of "
        f"{len(ift_generated_ids)} (period {ift_degenerate_period} tokens). "
        f"Attention uses only the first {len(ift_attention_ids)} tokens -- the "
        "loop is excluded."
    )

if len(ift_attention_ids) == 0:
    raise RuntimeError("IFT generation collapsed immediately; nothing to average.")

# Rows of the cached tensor line up with generated tokens, so trimming is a
# slice of the tensor -- no re-collection. Change the slice and re-`%run` and
# the cache serves it back instantly.
ift_attention = ift_run["attention"][:len(ift_attention_ids)]

print("\nAttention tensor shape:")
print(ift_attention.shape)

ift_avg_attention = average_prompt_attention(ift_attention)

print("\nAverage attention shape:")
print(ift_avg_attention.shape)

def display_attention(tokenizer, prompt_ids, average_attention):

    tokens = tokenizer.convert_ids_to_tokens(
        prompt_ids.tolist()
    )

    rows = []

    for token, attention in zip(tokens, average_attention.tolist()):
        rows.append({
            "token": token,
            "average_attention": attention
        })

    df = pd.DataFrame(rows)

    return df


# ift_df = display_attention(
#     ift_tokenizer,
#     ift_prompt_ids,
#     ift_avg_attention
# )

# ift_df

# The reasoning model is ALREADY resident -- it was loaded alongside the IFT
# model near the top of this file. Do not load it again: a second copy costs
# another ~6 GB in float32 on top of the resident IFT model, which exceeds the
# T4's ~14.5 GB and makes accelerate silently offload weights to CPU/meta.
# That offload is what hung the previous run.
print("Reasoning model already loaded.")
print("Dtype:", next(reasoning_model.parameters()).dtype)
print("Device:", next(reasoning_model.parameters()).device)

print("PROMPT:")
print(test_prompt)

print("\nGenerating with reasoning model...")

reasoning_run = run_and_cache(
    "reasoning",
    reasoning_model,
    reasoning_tokenizer,
    test_prompt
)

reasoning_prompt_ids = reasoning_run["prompt_ids"]
reasoning_generated_ids = reasoning_run["generated_ids"]
reasoning_text = reasoning_run["text"]

print("\nREASONING RESPONSE:")
print(reasoning_text)

print("\nGenerated tokens:")
print(len(reasoning_generated_ids))

# ============================================================
# ANSWER CHECK
# ============================================================

print("\n" + "=" * 64)
print("ANSWER CHECK (heuristic marker match -- read the response yourself)")
print("=" * 64)

print("Ground truth:", test_spec["ground_truth"])

# IFT answers directly, so grade its whole response.
ift_check = check_answer(ift_text, test_spec, scope="full")

# The reasoning model must be graded ONLY after its trace closes: a marker found
# inside <think> is working-out, not an answer. On the chickens/cows run the
# reasoning model never emitted a closing tag, and the old whole-text check
# called it CORRECT off '6' and '4' that only ever appeared mid-thought.
reasoning_check = check_answer(reasoning_text, test_spec, scope="final")


def report_check(label, result):

    print(
        f"{label}: {result['verdict']}"
        f"   hits={result['hits']}  misses={result['misses']}"
        f"   graded_chars={result['scope_chars']}"
    )

    for marker, pos in result["positions"].items():
        print(f"           {marker!r} -> first at char {pos}")

    if result["verdict"] == "NO FINAL ANSWER":
        print(
            "           no closing reasoning tag found, so the response ends "
            "inside the trace and emitted no answer to grade"
        )


report_check("IFT      ", ift_check)
report_check("Reasoning", reasoning_check)

ift_verdict = ift_check["verdict"]
reasoning_verdict = reasoning_check["verdict"]

# Generated length matters for the comparison below: average_prompt_attention()
# averages over however many tokens each model produced, so unequal counts mean
# the two averages cover different amounts of each response.
print(
    f"Tokens generated -- IFT: {len(ift_generated_ids)}"
    f"  Reasoning: {len(reasoning_generated_ids)}"
)

print("=" * 64)

reasoning_degenerate_start = reasoning_run["trim_start"]
reasoning_degenerate_period = reasoning_run["trim_period"]

if reasoning_degenerate_start is None:
    reasoning_attention_ids = reasoning_generated_ids
    print(f"\nNo repetition collapse. Using all {len(reasoning_generated_ids)} generated tokens.")
else:
    reasoning_attention_ids = reasoning_generated_ids[:reasoning_degenerate_start]
    print(
        f"\nREPETITION COLLAPSE at generated token {reasoning_degenerate_start} of "
        f"{len(reasoning_generated_ids)} (period {reasoning_degenerate_period} tokens). "
        f"Attention uses only the first {len(reasoning_attention_ids)} tokens -- the "
        "loop is excluded."
    )

if len(reasoning_attention_ids) == 0:
    raise RuntimeError("Reasoning generation collapsed immediately; nothing to average.")

reasoning_attention = reasoning_run["attention"][:len(reasoning_attention_ids)]

reasoning_avg_attention = average_prompt_attention(
    reasoning_attention
)

print("\nTokens used for attention (after collapse trim):")
print(f"  IFT       : {len(ift_attention_ids)} / {len(ift_generated_ids)}")
print(f"  Reasoning : {len(reasoning_attention_ids)} / {len(reasoning_generated_ids)}")

# How much of each model's attention lands on the prompt at all, rather than on
# its own generated tokens. This is the quantity the per-prompt renormalization
# further down divides out, so it is NOT visible in the final comparison table.
#
# It is often the bigger effect: a model that mostly attends to its own
# reasoning trace scores much lower here than one that keeps looking back at
# the question.
print("\nPrompt attention mass (share of attention on prompt tokens):")
print(f"  IFT       : {ift_avg_attention.sum().item():.4f}")
print(f"  Reasoning : {reasoning_avg_attention.sum().item():.4f}")
print("  Scale: 1.0 means all attention stayed on prompt tokens.")

print("\nAttention tensor shape:")
print(reasoning_attention.shape)

reasoning_df = display_attention(
    reasoning_tokenizer,
    reasoning_prompt_ids,
    reasoning_avg_attention
)

display(reasoning_df)

ift_tokens = ift_tokenizer.convert_ids_to_tokens(
    ift_prompt_ids.tolist()
)

reasoning_tokens = reasoning_tokenizer.convert_ids_to_tokens(
    reasoning_prompt_ids.tolist()
)

print("Same tokenization:", ift_tokens == reasoning_tokens)

if ift_tokens != reasoning_tokens:
    print("\nIFT tokens:")
    print(ift_tokens)

    print("\nReasoning tokens:")
    print(reasoning_tokens)

comparison = pd.DataFrame({
    "token": ift_tokens,
    "IFT_attention": ift_avg_attention.numpy(),
    "Reasoning_attention": reasoning_avg_attention.numpy()
})

display(comparison)

x = np.arange(len(comparison))
width = 0.38

fig, ax = plt.subplots(figsize=(14, 6))

ax.bar(
    x - width / 2,
    comparison["IFT_attention"],
    width,
    label="IFT"
)

ax.bar(
    x + width / 2,
    comparison["Reasoning_attention"],
    width,
    label="Reasoning"
)

ax.set_xticks(x)
ax.set_xticklabels(comparison["token"], rotation=60, ha="right")

ax.set_ylabel("Average attention from generated tokens")
ax.set_xlabel("Original prompt token")

ax.set_title(
    "Prompt-token attention: IFT vs Reasoning\n"
    f"IFT: {ift_verdict}   |   Reasoning: {reasoning_verdict}"
    f"   (ground truth: {test_spec['ground_truth']})"
)

ax.legend()

fig.tight_layout()
fig.savefig("ift_vs_reasoning_attention.png", dpi=150, bbox_inches="tight")

# display(fig) rather than plt.show(): under `%run` there is no cell output
# area for the inline backend to draw into, which is why the previous run
# emitted a bare empty figure instead of the chart.
display(fig)

# ============================================================
# TOKEN-BY-TOKEN ATTENTION COMPARISON
# ONLY THE USER'S ACTUAL PROMPT
# ============================================================

# 1. Tokenize ONLY the user's actual prompt
user_prompt_ids = ift_tokenizer.encode(
    test_prompt,
    add_special_tokens=False
)

user_tokens = [
    ift_tokenizer.decode([token_id])
    for token_id in user_prompt_ids
]


# 2. Re-create the FULL chat-template prompt
messages = [
    {"role": "user", "content": test_prompt}
]

full_inputs = ift_tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt"
)

full_ids = full_inputs["input_ids"][0].cpu()


# 3. Find where the actual user prompt occurs
user_ids_tensor = torch.tensor(user_prompt_ids)

start = None

for i in range(len(full_ids) - len(user_ids_tensor) + 1):

    if torch.equal(
        full_ids[i:i + len(user_ids_tensor)],
        user_ids_tensor
    ):
        start = i
        break

if start is None:
    raise ValueError("Could not find the user prompt inside the chat template.")

end = start + len(user_prompt_ids)

print("User prompt starts at full-sequence token:", start)
print("User prompt ends at full-sequence token:", end - 1)
print("Number of user prompt tokens:", len(user_tokens))


# 4. Extract ONLY user-prompt attention
ift_user_attention = ift_avg_attention[start:end]
reasoning_user_attention = reasoning_avg_attention[start:end]


# 5. Normalize within the USER PROMPT
#
# This makes the attention values sum to 1 for each model.
#
# Important:
# Your earlier average_prompt_attention() already handled
# different GENERATED lengths by averaging across generated
# tokens.
#
# This second normalization handles the distribution INSIDE
# the user prompt.

ift_user_attention_norm = (
    ift_user_attention /
    ift_user_attention.sum()
)

reasoning_user_attention_norm = (
    reasoning_user_attention /
    reasoning_user_attention.sum()
)


# 6. Build the final token-by-token comparison
comparison = pd.DataFrame({

    "token": user_tokens,

    "IFT_attention":
        ift_user_attention_norm.numpy(),

    "Reasoning_attention":
        reasoning_user_attention_norm.numpy()

})


# 7. Add difference
comparison["Reasoning_minus_IFT"] = (
    comparison["Reasoning_attention"]
    - comparison["IFT_attention"]
)


# 8. Display
display(comparison)


# ============================================================
# STAGE A: WHERE DOES PROMPT ATTENTION ACTUALLY GO?
# ============================================================
#
# Every number above averages over all 28 layers and all 12 heads at once. The
# reports below break that average apart, in increasing order of how much they
# change what can be said:
#
#   A1  sink share per (layer, head) -- is the sink concentrated in a few heads
#       (a clean split, so dropping them is meaningful) or spread across nearly
#       all of them (no clean split, so any threshold is arbitrary)?
#   A2  sink share per layer -- are the early layers the sink, as the literature
#       reports, leaving the middle layers readable?
#   A3  redo the comparison using only the NON-sink heads. This is the one that
#       decides whether "IFT looks at the prompt more" survives.
#
# Working on the TRIMMED tensor, so a repetition loop cannot skew a head's share
# the way it skewed the whole-model average.

ift_attn_np = ift_attention.numpy()
reasoning_attn_np = reasoning_attention.numpy()

ift_sink_share, ift_sink_mass = sink_profile(ift_attn_np)
reasoning_sink_share, reasoning_sink_mass = sink_profile(reasoning_attn_np)

n_layers, n_heads = ift_sink_share.shape

print("\n" + "=" * 64)
print("DETERMINISM CHECK (this run vs S1 run #2)")
print("=" * 64)
print("Greedy decoding is reproducible, so these must match run #2 exactly. If")
print("they do not, the cache or the pipeline changed and nothing below is")
print("comparable to the earlier numbers.")

print(
    f"  trim          IFT {len(ift_attention_ids)}/{len(ift_generated_ids)}"
    f" (expect 441/1024, period 39)"
    f"   Reasoning {len(reasoning_attention_ids)}/{len(reasoning_generated_ids)}"
    f" (expect 829/1024, period 1)"
)
print(
    f"  token {SINK_INDEX} attention  IFT {float(ift_avg_attention[SINK_INDEX]):.4f}"
    f" (expect 0.3950)"
    f"   Reasoning {float(reasoning_avg_attention[SINK_INDEX]):.6f} (expect 0.359161)"
)
print(
    f"  prompt mass   IFT {float(ift_avg_attention.sum()):.4f} (expect 0.4326)"
    f"   Reasoning {float(reasoning_avg_attention.sum()):.4f} (expect 0.3891)"
)

print("\n" + "=" * 64)
print("STAGE A1: SINK SHARE PER LAYER AND HEAD")
print("=" * 64)

print(f"Sink = prompt token {SINK_INDEX} ({ift_tokens[SINK_INDEX]!r}), the first")
print("token of the rendered chat template. Each cell is the fraction of that")
print("head's prompt attention that lands on the sink.")
print("  1.00 = pure sink head, holds no information about the question")
print("  0.00 = never looks at the sink at all")

ift_share_df = pd.DataFrame(
    ift_sink_share.round(2),
    columns=[f"h{h}" for h in range(n_heads)]
)

ift_share_df.insert(0, "layer", range(n_layers))

display(ift_share_df)

print("\nHow many IFT heads are sink-dominant?")
for threshold in (0.9, 0.75, 0.5, 0.2):
    n = int((ift_sink_share > threshold).sum())
    print(
        f"  share > {threshold:<5}: {n:3d} of {ift_sink_share.size} heads"
        f" ({100 * n / ift_sink_share.size:5.1f}%)"
    )

# The shape of the distribution is the real question, not any one threshold: a
# two-hump distribution means sink heads and reader heads are genuinely distinct
# groups, so a threshold cuts between them. One smear means there is no such
# split and the threshold is picking an arbitrary line.
edges = np.linspace(0.0, 1.0, 11)
counts, _ = np.histogram(ift_sink_share, bins=edges)

print("\nDistribution of IFT sink shares. Two humps = a real split; one fat")
print("hump = any threshold is arbitrary:")

for lo, hi, count in zip(edges[:-1], edges[1:], counts):
    bar = "#" * int(round(count / max(counts.max(), 1) * 40))
    print(f"  {lo:.1f}-{hi:.1f}  {count:3d}  {bar}")

print("\n" + "=" * 64)
print("STAGE A2: SINK SHARE PER LAYER")
print("=" * 64)

layer_df = pd.DataFrame({
    "layer": range(n_layers),
    "IFT_mean_sink_share": ift_sink_share.mean(axis=1).round(3),
    "Reasoning_mean_sink_share": reasoning_sink_share.mean(axis=1).round(3),
    "IFT_heads_above_0.5": (ift_sink_share > 0.5).sum(axis=1),
    "Reasoning_heads_above_0.5": (reasoning_sink_share > 0.5).sum(axis=1),
})

display(layer_df)

print("\nWhere the prompt attention mass actually sits, per layer:")
print("(mass = that layer's total prompt attention, before any normalization)")

layer_mass_df = pd.DataFrame({
    "layer": range(n_layers),
    "IFT_mass": ift_sink_mass.sum(axis=1).round(5),
    "Reasoning_mass": reasoning_sink_mass.sum(axis=1).round(5),
})

display(layer_mass_df)

print("\n" + "=" * 64)
print("STAGE A3: THE COMPARISON WITH SINK HEADS REMOVED")
print("=" * 64)

ift_full_mass = float(ift_avg_attention.sum())
reasoning_full_mass = float(reasoning_avg_attention.sum())

print(
    f"Prompt mass, all heads -- IFT {ift_full_mass:.4f}"
    f"   Reasoning {reasoning_full_mass:.4f}"
)
print(
    f"  of which the sink alone is"
    f" IFT {float(ift_avg_attention[SINK_INDEX]):.4f}"
    f" ({100 * float(ift_avg_attention[SINK_INDEX]) / ift_full_mass:.1f}%)"
    f"   Reasoning {float(reasoning_avg_attention[SINK_INDEX]):.4f}"
    f" ({100 * float(reasoning_avg_attention[SINK_INDEX]) / reasoning_full_mass:.1f}%)"
)

# Two thresholds on purpose. A finding that holds at both is robust; a finding
# that flips between them was a property of the threshold, not of the models.
for threshold in (0.9, 0.5):

    ift_vec, ift_retained, ift_kept, ift_total = keep_reader_heads(
        ift_attn_np, ift_sink_share, threshold
    )

    reasoning_vec, reasoning_retained, reasoning_kept, reasoning_total = (
        keep_reader_heads(reasoning_attn_np, reasoning_sink_share, threshold)
    )

    print("\n" + "-" * 64)
    print(f"Threshold: drop heads whose sink share exceeds {threshold}")

    print(
        f"  IFT       kept {ift_kept:3d}/{ift_total} heads,"
        f" holding {100 * ift_retained:6.2f}% of prompt attention"
    )
    print(
        f"  Reasoning kept {reasoning_kept:3d}/{reasoning_total} heads,"
        f" holding {100 * reasoning_retained:6.2f}% of prompt attention"
    )

    print(
        f"  prompt mass with sinks dropped -- IFT {ift_vec.sum():.4f}"
        f"   Reasoning {reasoning_vec.sum():.4f}"
    )

    # Same user-prompt window as the table at the top of the file: `start` and
    # `end` are the location of the raw user prompt inside the chat template,
    # found above.
    ift_user = ift_vec[start:end]
    reasoning_user = reasoning_vec[start:end]

    reader_df = pd.DataFrame({
        "token": user_tokens,
        "IFT_attention": ift_user,
        "Reasoning_attention": reasoning_user,
    })

    reader_df["Reasoning_minus_IFT"] = (
        reader_df["Reasoning_attention"] - reader_df["IFT_attention"]
    )

    print("\n  reader heads only, raw (unnormalized) attention per user token:")
    display(reader_df)

    mean_abs_raw = float(np.abs(reader_df["Reasoning_minus_IFT"]).mean())

    ift_denom = float(ift_user.sum())
    reasoning_denom = float(reasoning_user.sum())

    if ift_denom > 0 and reasoning_denom > 0:
        diff_norm = reasoning_user / reasoning_denom - ift_user / ift_denom
        mean_abs_norm = float(np.abs(diff_norm).mean())

        print("\n  reader heads only, each column rescaled to sum to 1:")
        display(pd.DataFrame({
            "token": user_tokens,
            "IFT_attention": ift_user / ift_denom,
            "Reasoning_attention": reasoning_user / reasoning_denom,
            "Reasoning_minus_IFT": diff_norm,
        }))
    else:
        mean_abs_norm = float("nan")

    print(
        f"\n  mean |Reasoning - IFT| across user tokens --"
        f" raw {mean_abs_raw:.5f}, rescaled {mean_abs_norm:.5f}"
    )
    print("  (all-heads rescaled figure was 0.0122 on run #2)")

