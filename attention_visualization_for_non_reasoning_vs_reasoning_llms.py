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
# Now 5 ("Which country has Paris as its capital?"). The France prompt (index 0)
# settled the framing -- IFT answers directly, the reasoning model deliberates --
# but its token table cannot be interpreted: the meaningful words ('France', '?')
# are also the LAST words, so "reasoning weights meaningful words" and "reasoning
# weights recent words" predict the same pattern.
#
# Index 5 is the same question asked the other way round. 'Paris' is the answer
# and sits in the MIDDLE, while the last tokens ('capital', '?') are no longer
# the answer. The two hypotheses now predict OPPOSITE things for 'Paris': high if
# meaning drives attention, low if only position does. That one token decides it.
#
# Index 0 keeps its recorded expected values, so flipping back reproduces run #2
# exactly -- and the run cache serves it without regenerating.
TEST_PROMPT_INDEX = 5

# Declared but NOT currently used: generate_response() hardcodes
# do_sample=False, so decoding is always greedy. See project history.
DO_SAMPLE = True
TEMPERATURE = 0.7
TOP_P = 0.9

# Loads from the Hub on the first run in a kernel; an instant no-op on every
# run after that, so iterating on this file costs no reload.
ift_tokenizer, ift_model = get_model("ift")
reasoning_tokenizer, reasoning_model = get_model("reasoning")

# The prompt list is DERIVED from PROMPT_SPECS rather than written out again.
# These used to be two parallel lists that had to be edited in lockstep with
# nothing enforcing it -- so adding a prompt to one and not the other would have
# silently shifted TEST_PROMPT_INDEX onto the wrong question.
#
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
#
# `roles` labels what each user-prompt token DOES (question word, filler, answer,
# answer attribute, punctuation) so two prompts can be compared by role instead
# of by position -- which is the whole point of the position-swap probe. It is
# only applied when its length matches the real tokenization, so a wrong guess
# degrades to a printed note rather than a broken table.
#
# `expected`, when present, holds a previous run's numbers so a later run can
# prove the pipeline and cache reproduce them exactly.

PROMPT_SPECS = [
    {
        "prompt": "What is the capital of France?",
        "ground_truth": "Paris",
        "must_contain": ["paris"],
        "roles": [
            "question", "filler", "filler", "answer-attr", "filler", "ANSWER", "punct",
        ],
        # Recorded from the S1 France run. Reproducing these exactly is what
        # makes the disk cache trustworthy rather than merely convenient.
        "expected": {
            "ift_trim": (441, 1024, 39),
            "reasoning_trim": (829, 1024, 1),
            "ift_token0": 0.3950,
            "reasoning_token0": 0.359161,
            "ift_mass": 0.4326,
            "reasoning_mass": 0.3891,
        },
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
    {
        # The position-swap probe. Same question as index 0, reversed, so the
        # answer word sits in the middle instead of at the end. If attention
        # tracks meaning, 'Paris' is high; if it tracks recency, 'Paris' is low
        # and the tail wins again.
        "prompt": "Which country has Paris as its capital?",
        "ground_truth": "France",
        "must_contain": ["france"],
        "roles": [
            "question", "question", "filler", "ANSWER",
            "filler", "filler", "answer-attr", "punct",
        ],
        # Recorded from the S2 position-swap run, which printed these and asked
        # for them back. Note the numbers are measured under the PERIODIC-TAIL
        # rule alone; see the determinism-check block below for why.
        "expected": {
            "ift_trim": (264, 1024, 1),
            "reasoning_trim": (683, 1024, 1),
            "ift_token0": 0.3812,
            "reasoning_token0": 0.367706,
            "ift_mass": 0.4534,
            "reasoning_mass": 0.4079,
        },
    },
]

prompts = [spec["prompt"] for spec in PROMPT_SPECS]


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


# Chat control tokens that should never appear inside a single-turn answer.
# If the model emits one, it has closed its own reply and opened a new turn --
# it is role-playing the conversation continuing, and everything after that
# point is not the model answering the question.
TURN_BOUNDARY_TOKENS = ("<|im_end|>", "<|im_start|>", "<|endoftext|>")


def turn_boundary_ids(tokenizer):
    """Token ids for the chat turn markers, skipping any the tokenizer lacks.

    Passing the id set in (rather than the tokenizer) keeps find_drift_start()
    a pure function of plain ints, so it can be tested without loading a model.
    """

    ids = set()

    for token in TURN_BOUNDARY_TOKENS:

        token_id = tokenizer.convert_tokens_to_ids(token)

        # A tokenizer that lacks the marker returns the unknown-token id, and on
        # some tokenizers unk_token_id is itself None. Either way, skip it --
        # adding a bogus id would cut every generation at the first real token.
        if token_id is None or token_id == tokenizer.unk_token_id:
            continue

        ids.add(token_id)

    return ids


def find_drift_start(generated_ids, boundary_ids):
    """Index of the first generated turn-boundary token, or None.

    Deliberately separate from find_degenerate_tail(), because the two catch
    different failures and neither implies the other:

      * find_degenerate_tail -- the model finishes, then repeats a fixed phrase
        forever. Caught by periodicity, and only in the TAIL.
      * find_drift_start -- the model finishes, then keeps going by inventing
        further conversation turns. NOT periodic, so the tail detector is blind
        to it by construction.

    The second failure was the big one on the France run: IFT's kept 441 tokens
    held roughly 50 tokens of answer and 390 of chat role-play, and one faked
    block repeated twice inside the kept prefix. Cutting here is what makes the
    average describe the model answering rather than the model chatting.
    """

    ids = (
        generated_ids.tolist()
        if hasattr(generated_ids, "tolist")
        else list(generated_ids)
    )

    boundary_ids = set(boundary_ids)

    for index, token_id in enumerate(ids):

        if token_id in boundary_ids:
            return index

    return None


def earliest_trim(drift_start, degenerate_start):
    """The earlier of the two cut points, treating None as 'no cut'.

    Both detectors can fire on one run, and either can fire alone. The average
    must stop at whichever comes first -- otherwise a chat-drift block sitting
    before a clean loop would still be averaged in.
    """

    cuts = [cut for cut in (drift_start, degenerate_start) if cut is not None]

    return min(cuts) if cuts else None


def resolve_attention_trim(
    label,
    generated_ids,
    boundary_ids,
    degenerate_start,
    degenerate_period,
):
    """How many leading generated tokens the average may use, and why.

    One place decides the trim for both models, so the two cannot silently
    drift apart -- which is how the sink-column bug survived review earlier.
    Returns the count; the caller slices.
    """

    drift_start = find_drift_start(generated_ids, boundary_ids)
    trim_start = earliest_trim(drift_start, degenerate_start)

    n_generated = len(generated_ids)
    n_used = n_generated if trim_start is None else trim_start

    print(f"\n{label} attention window:")

    if drift_start is None:
        print(
            f"  drift : none -- no chat turn markers in {n_generated} tokens, "
            "so the model never started inventing further turns"
        )
    else:
        print(
            f"  drift : assistant-turn drift at generated token {drift_start} of "
            f"{n_generated} -- the model closed its reply and began writing more "
            "conversation turns"
        )

    if degenerate_start is None:
        print("  loop  : none -- generation never became periodic")
    else:
        print(
            f"  loop  : periodic collapse at generated token {degenerate_start} "
            f"of {n_generated} (period {degenerate_period} tokens)"
        )

    if trim_start is None:
        print(f"  -> using all {n_used} generated tokens")
    else:
        print(
            f"  -> using the first {n_used} of {n_generated} tokens; the "
            f"remaining {n_generated - n_used} are excluded from the average"
        )

    return n_used


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
    future change to the trim rule free -- which is exactly what the
    drift-boundary trim added in S2 cost: nothing, on already-cached prompts.
    Averaging over a prefix of the tensor reproduces exactly what collecting
    only that prefix would have produced.

    Only the PERIODIC-TAIL trim is stored in the cache file. Drift cutting is
    derived from generated_ids at read time, so adding or changing it does not
    invalidate an existing .npz.
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


def content_mass(attention, sink_index=SINK_INDEX):
    """Total prompt attention on NON-sink tokens, summed over all heads.

    The sink column is removed before summing, so this is the attention the
    model spends on prompt tokens that are not the format marker. Dividing it by
    the raw prompt mass says how much of "attention to the prompt" was the sink.
    """

    per_head = attention.mean(axis=0).copy()

    per_head[..., sink_index] = 0.0

    return float(per_head.sum())


def keep_reader_heads(attention, share, threshold, sink_index=SINK_INDEX):
    """Average prompt attention using only the heads that are not sink-dominant.

    Returns (prompt_vector, retained_share_of_content, kept, total_heads).

    retained_share_of_content is the number to read first: the fraction of all
    NON-sink prompt attention that the kept heads account for. If the readers
    hold only a few percent of it, then the comparison built on them is a
    comparison of a few percent of the signal, and should be reported that way
    rather than as if it were the whole picture.

    The sink COLUMN is zeroed as well as the sink-dominant heads being dropped.
    Dropping heads alone is not enough: a head kept at a 0.9 threshold may still
    put most of its mass on the sink, so the surviving total would still be
    mostly sink and the "sinks removed" label would simply be false. (This was
    wrong in the first version of this function -- the docstring said the sink
    was zeroed and the code did not do it, which inflated every mass printed
    under it. The per-word tables were unaffected, because they slice away the
    sink token, but the mass numbers were not.)

    Note the surviving mass is reported, never rescaled to 1: rescaling would
    hide exactly the quantity being measured.
    """

    keep = share < threshold

    per_head = attention.mean(axis=0).copy()

    per_head[..., sink_index] = 0.0

    total_content = float(per_head.sum())

    reader_mass = per_head * keep[..., None]

    prompt_vector = reader_mass.sum(axis=(0, 1))

    retained = float(reader_mass.sum() / total_content) if total_content > 0 else 0.0

    return prompt_vector, retained, int(keep.sum()), int(keep.size)


# ============================================================
# STAGE B HELPERS: SEPARATING MEANING FROM POSITION
# ============================================================


def rank_positions_vs_attention(attention_values):
    """Rank tokens by attention and by position, so the two can be compared.

    Input: user-token attention, in prompt order.

    Returns (attn_rank, pos_rank), both 1-based with 1 meaning "most".

    pos_rank counts BACKWARDS from the end: the last token is rank 1, the first
    token is rank n. That is the order pure recency predicts, so a token whose
    attn_rank is a SMALLER number than its pos_rank was attended more than its
    position alone explains -- the signature of a meaning effect. Counting
    pos_rank forwards instead would make the two columns agree almost everywhere
    and hide the very thing being tested.
    """

    values = np.asarray(attention_values, dtype=np.float64)

    n = values.size

    if n == 0:
        return np.zeros(0, dtype=int), np.zeros(0, dtype=int)

    # rank 1 = highest attention; stable sort breaks ties by position.
    order = np.argsort(-values, kind="stable")

    attn_rank = np.empty(n, dtype=int)
    attn_rank[order] = np.arange(1, n + 1)

    pos_rank = np.arange(n, 0, -1)

    return attn_rank, pos_rank


def rank_correlation(a, b):
    """Spearman correlation -- Pearson on the ranks, so no scipy needed.

    Returns nan rather than 0.0 when either side is constant or too short: 0.0
    would read as "measured, and unrelated", whereas the honest answer is "no
    relationship is measurable here".
    """

    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    if a.size < 2 or b.size < 2 or a.std() == 0 or b.std() == 0:
        return float("nan")

    return float(np.corrcoef(a, b)[0, 1])


def repeated_token_groups(tokens, attention_values):
    """Group positions by identical token -- the same word in different places.

    A repeated token is a free control group: it means the same thing every time
    it appears, so any attention difference between its occurrences must be
    positional. The chat template is full of them -- the newline marker alone
    appears five times.

    Returns {token: [(position, attention), ...]}, for tokens appearing at least
    twice.
    """

    groups = {}

    for position, (token, value) in enumerate(zip(tokens, attention_values)):
        groups.setdefault(token, []).append((position, float(value)))

    return {token: occ for token, occ in groups.items() if len(occ) >= 2}

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
# Two independent failures are cut here, because the France run showed that one
# detector cannot see both (see find_drift_start / find_degenerate_tail for the
# full account). resolve_attention_trim() applies both and reports which fired.
ift_degenerate_start = ift_run["trim_start"]
ift_degenerate_period = ift_run["trim_period"]

ift_boundary_ids = turn_boundary_ids(ift_tokenizer)

ift_attention_count = resolve_attention_trim(
    "IFT",
    ift_generated_ids,
    ift_boundary_ids,
    ift_degenerate_start,
    ift_degenerate_period,
)

ift_attention_ids = ift_generated_ids[:ift_attention_count]

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

# Same two cut points as IFT, applied by the same function so the two models
# cannot drift apart. The reasoning model may well have no turn markers at all
# -- it closes its trace and answers -- in which case only the loop trim fires
# and nothing changes for it.
reasoning_boundary_ids = turn_boundary_ids(reasoning_tokenizer)

reasoning_attention_count = resolve_attention_trim(
    "Reasoning",
    reasoning_generated_ids,
    reasoning_boundary_ids,
    reasoning_degenerate_start,
    reasoning_degenerate_period,
)

reasoning_attention_ids = reasoning_generated_ids[:reasoning_attention_count]

if len(reasoning_attention_ids) == 0:
    raise RuntimeError("Reasoning generation collapsed immediately; nothing to average.")

reasoning_attention = reasoning_run["attention"][:len(reasoning_attention_ids)]

reasoning_avg_attention = average_prompt_attention(
    reasoning_attention
)

print("\nTokens used for attention (after drift and loop trims):")
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

# The determinism check compares against baselines recorded BEFORE the
# drift-boundary trim existed, so it recomputes those six values under the old
# PERIODIC-TAIL-ONLY rule. That is not a fudge: this check's job is to prove
# generation, attention collection and tail detection are unchanged, and those
# are exactly the things the tail-only basis measures. The trim POLICY is a
# downstream choice and is not what this check is testing -- without the split,
# adding drift detection would read as a regression on every prompt recorded
# earlier.
ift_tail_only_ids = (
    ift_generated_ids
    if ift_degenerate_start is None
    else ift_generated_ids[:ift_degenerate_start]
)

reasoning_tail_only_ids = (
    reasoning_generated_ids
    if reasoning_degenerate_start is None
    else reasoning_generated_ids[:reasoning_degenerate_start]
)

ift_check_avg = average_prompt_attention(
    ift_run["attention"][:len(ift_tail_only_ids)]
)
reasoning_check_avg = average_prompt_attention(
    reasoning_run["attention"][:len(reasoning_tail_only_ids)]
)

print("\n" + "=" * 64)
print(f"DETERMINISM CHECK -- {test_spec['prompt']}")
print("=" * 64)
print("Measured under the periodic-tail rule alone, so the numbers stay")
print("comparable to baselines recorded before the drift trim was added.")

expected = test_spec.get("expected")

if expected is None:

    # A prompt with no recorded baseline is not a failure -- it is the first
    # measurement. Print the values so they can be pasted back into the spec,
    # which turns the next run into a real check rather than a fresh start.
    print("No baseline recorded for this prompt. Greedy decoding is reproducible,")
    print("so these values should repeat exactly on the next run -- record them in")
    print("PROMPT_SPECS['expected'] to turn that repetition into a real check.")

    print(
        f"  trim          IFT {len(ift_tail_only_ids)}/{len(ift_generated_ids)}"
        f" (period {ift_degenerate_period})"
        f"   Reasoning {len(reasoning_tail_only_ids)}/{len(reasoning_generated_ids)}"
        f" (period {reasoning_degenerate_period})"
    )
    print(
        f"  token {SINK_INDEX} attention  IFT {float(ift_check_avg[SINK_INDEX]):.4f}"
        f"   Reasoning {float(reasoning_check_avg[SINK_INDEX]):.6f}"
    )
    print(
        f"  prompt mass   IFT {float(ift_check_avg.sum()):.4f}"
        f"   Reasoning {float(reasoning_check_avg.sum()):.4f}"
    )

else:

    checks = [
        (
            "IFT trim (n_used, n_generated, period)",
            (len(ift_tail_only_ids), len(ift_generated_ids), ift_degenerate_period),
            expected["ift_trim"],
        ),
        (
            "Reasoning trim (n_used, n_generated, period)",
            (len(reasoning_tail_only_ids), len(reasoning_generated_ids), reasoning_degenerate_period),
            expected["reasoning_trim"],
        ),
        ("IFT token 0 attention", float(ift_check_avg[SINK_INDEX]), expected["ift_token0"]),
        ("Reasoning token 0 attention", float(reasoning_check_avg[SINK_INDEX]), expected["reasoning_token0"]),
        ("IFT prompt mass", float(ift_check_avg.sum()), expected["ift_mass"]),
        ("Reasoning prompt mass", float(reasoning_check_avg.sum()), expected["reasoning_mass"]),
    ]

    all_ok = True

    for label, got, want in checks:

        if isinstance(want, tuple):
            ok = tuple(got) == tuple(want)
        else:
            # Attention goes through float32 on disk, so compare at recorded
            # precision rather than for bit equality.
            ok = abs(got - want) < 1e-4

        all_ok = all_ok and ok

        print(f"  {'ok  ' if ok else 'FAIL'} {label}: got {got}  expected {want}")

    if all_ok:
        print("\nReproduced exactly -- the cache and the pipeline are unchanged.")
    else:
        print("\nMISMATCH. Stop here: nothing below is comparable to the baseline.")

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

# Content mass = prompt attention with the sink column removed. Everything below
# is reported as a share of THIS, not of the raw prompt mass, because a share of
# the raw mass mostly measures the sink.
ift_content_mass = content_mass(ift_attn_np)
reasoning_content_mass = content_mass(reasoning_attn_np)

if_to_mean = ift_sink_mass.size
re_to_mean = reasoning_sink_mass.size

print(
    f"Prompt mass, all heads -- IFT {ift_full_mass:.4f}"
    f"   Reasoning {reasoning_full_mass:.4f}"
)
print(
    f"  the sink alone is"
    f" IFT {float(ift_avg_attention[SINK_INDEX]):.4f}"
    f" ({100 * float(ift_avg_attention[SINK_INDEX]) / ift_full_mass:.1f}%)"
    f"   Reasoning {float(reasoning_avg_attention[SINK_INDEX]):.4f}"
    f" ({100 * float(reasoning_avg_attention[SINK_INDEX]) / reasoning_full_mass:.1f}%)"
)
print(
    f"  non-sink CONTENT mass (the real budget) --"
    f" IFT {ift_content_mass / if_to_mean:.4f}"
    f"   Reasoning {reasoning_content_mass / re_to_mean:.4f}"
)
print("  (same per-head average scale as the line above, sink column removed)")

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
        f" holding {100 * ift_retained:6.2f}% of all non-sink content attention"
    )
    print(
        f"  Reasoning kept {reasoning_kept:3d}/{reasoning_total} heads,"
        f" holding {100 * reasoning_retained:6.2f}% of all non-sink content attention"
    )

    # Content mass with the sink column removed, so this number is comparable
    # across thresholds (unlike the raw prompt mass, which is mostly sink).
    ift_reader_mass = float(ift_vec.sum()) / ift_total
    reasoning_reader_mass = float(reasoning_vec.sum()) / reasoning_total

    print(
        f"  content mass on the prompt -- IFT {ift_reader_mass:.5f}"
        f"   Reasoning {reasoning_reader_mass:.5f}"
    )

    if reasoning_reader_mass > 0:
        print(
            f"  IFT / Reasoning = {ift_reader_mass / reasoning_reader_mass:.3f}"
            "   (1.0 = the two models read the prompt equally)"
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
    print("  (the France prompt, all heads, was 0.0122)")


# ============================================================
# STAGE B: IS ATTENTION JUST POSITION?
# ============================================================
#
# The France prompt could not answer this. Its meaningful tokens ('France', '?')
# were also its LAST tokens, so "meaning" and "recency" predicted the same thing
# and no amount of analysis could pull them apart. The position-swap prompt moves
# the answer word into the middle, where the two hypotheses disagree.
#
# Three reports, strongest evidence last:
#
#   B1  rank each user token by attention and by position. Where the rankings
#       agree, the table is a recency ramp; where they disagree, something else
#       is at work.
#   B2  the same table labelled by ROLE (question word / filler / ANSWER), so two
#       differently-worded prompts can be compared by what a token does rather
#       than where it sits.
#   B3  repeated tokens -- identical meaning at different positions. The cleanest
#       test in the data, and it needs no new generation or new prompt.

print("\n" + "=" * 64)
print("STAGE B: IS ATTENTION JUST POSITION?")
print("=" * 64)

print(f"Prompt: {test_prompt}")

ift_user_raw = ift_avg_attention[start:end].numpy()
reasoning_user_raw = reasoning_avg_attention[start:end].numpy()

ift_attn_rank, pos_rank = rank_positions_vs_attention(ift_user_raw)
reasoning_attn_rank, _ = rank_positions_vs_attention(reasoning_user_raw)

print("\nB1. Attention rank vs position rank  (pos_rank 1 = the LAST token)")
print("    shift = pos_rank - attn_rank")
print("      shift > 0  -> attended MORE than its position explains (meaning)")
print("      shift ~ 0  -> attention is what position alone predicts")
print("      shift < 0  -> attended LESS than its position explains")

stage_b_df = pd.DataFrame({
    "pos": list(range(len(user_tokens))),
    "token": user_tokens,
    "IFT_attn": ift_user_raw,
    "IFT_rank": ift_attn_rank,
    "pos_rank": pos_rank,
    "IFT_shift": pos_rank - ift_attn_rank,
    "R_attn": reasoning_user_raw,
    "R_rank": reasoning_attn_rank,
    "R_shift": pos_rank - reasoning_attn_rank,
})

roles = test_spec.get("roles")

if roles and len(roles) == len(user_tokens):
    stage_b_df.insert(1, "role", roles)
elif roles:
    print(
        f"\nNOTE: the spec lists {len(roles)} roles but this prompt tokenizes to "
        f"{len(user_tokens)} tokens, so the role column is omitted rather than "
        f"guessed. Roles given: {roles}"
    )

display(stage_b_df)

print("\nRank correlation between attention and position, per model:")
print("  1.0 = a pure recency ramp. 0.0 = position explains nothing.")

for label, attn_rank in (("IFT", ift_attn_rank), ("Reasoning", reasoning_attn_rank)):
    print(f"  {label:10s}: {rank_correlation(attn_rank, pos_rank):+.3f}")

print("\nThe token to look at is the ANSWER word. If it sits near the bottom of")
print("the attention ranking while sitting in the MIDDLE of the prompt, then")
print("position is driving the table and meaning is not.")


# ------------------------------------------------------------
# B3. The same token, at several different positions
# ------------------------------------------------------------

print("\n" + "-" * 64)
print("B3. Identical tokens at different positions")
print("-" * 64)
print("A token that means the same thing every time it appears. Any attention")
print("difference between its occurrences is positional by construction, so this")
print("is the cleanest position test in the data -- no new prompt, no new run.")
print("It also says whether the recency reading generalises beyond the question")

ift_attn_all = ift_avg_attention.numpy()
reasoning_attn_all = reasoning_avg_attention.numpy()

ift_groups = repeated_token_groups(ift_tokens, ift_attn_all)
reasoning_groups = repeated_token_groups(reasoning_tokens, reasoning_attn_all)

repeated = sorted(
    (token for token in ift_groups if token in reasoning_groups),
    key=lambda token: -len(ift_groups[token]),
)

tested_any = False

for token in repeated:

    if len(ift_groups[token]) < 3:
        continue

    tested_any = True

    print(f"\n{token!r} appears {len(ift_groups[token])} times")

    for label, groups in (("IFT      ", ift_groups), ("Reasoning", reasoning_groups)):

        positions = [position for position, _ in groups[token]]
        values = [value for _, value in groups[token]]

        shown = "   ".join(f"pos{position}={value:.5f}" for position, value in groups[token])

        print(f"  {label}: {shown}")
        print(
            f"               correlation with position: "
            f"{rank_correlation(positions, values):+.3f}"
        )

    # The sink appears more than once too, and its position-0 occurrence would
    # dominate any correlation -- so flag it rather than let it masquerade as
    # independent evidence.
    if token == ift_tokens[SINK_INDEX]:
        print(
            "  NOTE: one of these occurrences IS the sink (position 0). A high"
            " correlation here restates the sink finding, it does not corroborate"
            " it."
        )

if not tested_any:
    print("\nNo token repeats three or more times, so B3 has nothing to test.")

print("\n" + "-" * 64)
print("HOW TO READ STAGE B")
print("-" * 64)
print("  * ANSWER token ranks high, shift strongly positive -> meaning drives")
print("    attention, and the token-level table in the main report is real.")
print("  * Every shift near 0 and rank correlation near 1.0 -> position alone")
print("    explains the table, and the IFT/Reasoning split is a recency effect")
print("    rather than attention to content.")
print("  * B3 breaks ties: identical tokens, so any spread there is positional by")
print("    construction -- except for the sink, which is flagged.")

