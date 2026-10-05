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

import os
import sys

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


def find_degenerate_tail(generated_ids, max_ngram=8, min_run_tokens=40):
    """Index where a repeated-n-gram collapse begins, or None if there is none.

    A small model under greedy decoding sometimes finishes its answer and then
    loops on a short n-gram forever ('ifrifrifr...'). Those tokens carry no
    answer content, so averaging attention over them measures the loop, not the
    solution.

    For n in 1..max_ngram, find the longest run of consecutive identical
    n-grams. A run covering >= min_run_tokens tokens counts as a collapse.
    Returns the earliest such start so nothing before the loop is discarded.
    """

    ids = generated_ids.tolist()
    total = len(ids)

    best = None

    for n in range(1, max_ngram + 1):

        i = 0

        while i + n <= total and (best is None or i < best):

            gram = ids[i:i + n]

            reps = 1
            j = i + n

            while j + n <= total and ids[j:j + n] == gram:
                reps += 1
                j += n

            if reps * n >= min_run_tokens and (best is None or i < best):
                best = i

            i += 1

    return best

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

test_prompt = prompts[TEST_PROMPT_INDEX]
test_spec = PROMPT_SPECS[TEST_PROMPT_INDEX]

print("PROMPT:")
print(test_prompt)

print("\nGenerating with IFT model...")

ift_prompt_ids, ift_generated_ids, ift_text = generate_response(
    ift_model,
    ift_tokenizer,
    test_prompt
)

print("\nIFT RESPONSE:")
print(ift_text)

# Drop any repetition collapse before averaging. Averaging over a loop measures
# the loop, not the solution -- on the chickens/cows run the IFT model finished
# its answer and then emitted 'ifr' to the cap, so a majority of its "generated
# tokens" were noise.
ift_degenerate_start = find_degenerate_tail(ift_generated_ids)

if ift_degenerate_start is None:
    ift_attention_ids = ift_generated_ids
    print(f"\nNo repetition collapse. Using all {len(ift_generated_ids)} generated tokens.")
else:
    ift_attention_ids = ift_generated_ids[:ift_degenerate_start]
    print(
        f"\nREPETITION COLLAPSE at generated token {ift_degenerate_start} of "
        f"{len(ift_generated_ids)}. Attention uses only the first "
        f"{len(ift_attention_ids)} tokens -- the loop is excluded."
    )

if len(ift_attention_ids) == 0:
    raise RuntimeError("IFT generation collapsed immediately; nothing to average.")

print("\nCollecting IFT attention...")

ift_attention = collect_prompt_attention(
    ift_model,
    ift_tokenizer,
    test_prompt,
    ift_attention_ids
)

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

reasoning_prompt_ids, reasoning_generated_ids, reasoning_text = generate_response(
    reasoning_model,
    reasoning_tokenizer,
    test_prompt
)

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

reasoning_degenerate_start = find_degenerate_tail(reasoning_generated_ids)

if reasoning_degenerate_start is None:
    reasoning_attention_ids = reasoning_generated_ids
    print(f"\nNo repetition collapse. Using all {len(reasoning_generated_ids)} generated tokens.")
else:
    reasoning_attention_ids = reasoning_generated_ids[:reasoning_degenerate_start]
    print(
        f"\nREPETITION COLLAPSE at generated token {reasoning_degenerate_start} of "
        f"{len(reasoning_generated_ids)}. Attention uses only the first "
        f"{len(reasoning_attention_ids)} tokens -- the loop is excluded."
    )

if len(reasoning_attention_ids) == 0:
    raise RuntimeError("Reasoning generation collapsed immediately; nothing to average.")

print("\nCollecting reasoning attention...")

reasoning_attention = collect_prompt_attention(
    reasoning_model,
    reasoning_tokenizer,
    test_prompt,
    reasoning_attention_ids
)

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

