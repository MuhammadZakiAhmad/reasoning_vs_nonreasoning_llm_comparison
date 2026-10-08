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

import contextlib
import hashlib
import io
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
#
# Image is here because figures are displayed as encoded PNGs, not as Figure
# objects -- see show_figure() for why.
#
# Markdown is how a heading gets BOLD in the output. print() writes plain text,
# so `print("**text**")` shows the asterisks rather than emphasising anything --
# the run needs real emphasis on the headings it wants the reader to skip past.
from IPython.display import Image, Markdown, display

from models import get_model

# Cap on generated tokens. Both models answer these prompts in long form (the
# IFT model writes a full worked solution), so 256 was truncating them
# mid-sentence.
#
# NOT a safety net. The S5 run killed that assumption: 5 of the 7 `kind` prompts
# ended at exactly this cap with no </think> and no answer -- opinion,
# counterfactual, creative, math, math-trap. It is a real ceiling that the
# reasoning model runs into, not a bound it never approaches.
#
# Cost warning: collect_prompt_attention() runs one forward pass per generated
# token, so runtime is linear in generated length -- 1024 is roughly 4x the
# work of 256, per model. The cap is part of the cache key, so changing it
# invalidates every cached prompt and pays that cost again.
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

# The single-prompt deep dive is GONE as of S5. It ran one PROMPT_SPECS index
# through the full Stage A/B/C audit and printed that as the verbose reference --
# but `%run` executes top to bottom, so every run led with that block, in a
# different format from the sweep below, and the output read as two experiments
# interleaved. The prompt-type sweep at the end of the file is now the whole run.
#
# Consequence, deliberate: the Stage A (sink heads), Stage B (position vs
# meaning) and single-prompt per-token tables are no longer EXECUTED anywhere.
# Their functions are still defined and still covered by the local harness, so
# reviving any of them is a call, not a rewrite. The sink result also needs its
# own phase-matched MASS test before it can be quoted again -- see project
# history.

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
        # No `expected` block: the stop-condition fix changed generation itself
        # (1024-token loops became a 9-token answer and a 300-token answer), so
        # every baseline recorded before it is void. The old France values --
        # trim (441, 1024, 39), token 0 0.3950, mass 0.4326 -- are in git
        # history at the commit before "stop on the model's own turn
        # terminator". The next run prints fresh values to record.
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
        # No `expected` block, for the same reason as index 0: the stop-condition
        # fix changed generation. The S2 swap values -- trim (264, 1024, 1) and
        # (683, 1024, 1), token 0 0.3812 and 0.367706, mass 0.4534 and 0.4079 --
        # are in git history at the commit before the fix. The next run prints
        # fresh values to record.
    },

    # ----------------------------------------------------------------
    # Prompts 6-12 (below): one per KIND, for the Stage C cross-type run.
    #
    # Why these exist: index 5 is a factual lookup with a single retrievable
    # answer, so both models converge and there is little for them to differ
    # about. If the two models' attention SHAPES diverge anywhere, it should be
    # on a prompt with no single retrievable answer, where the reasoning model's
    # deliberation is doing real work.
    #
    # A `kind` label is what opts a spec into the cross-type loop at the end of
    # this file -- there is no second list to keep in sync.
    #
    # `ungraded` marks an open-ended prompt: there is no marker string that can
    # be "right", and an empty `must_contain` would otherwise read as a vacuous
    # CORRECT. See check_answer().
    # ----------------------------------------------------------------

    {
        "kind": "opinion",
        "prompt": "Is it better to study in the morning or at night? Give your opinion.",
        "ground_truth": None,
        "must_contain": [],
        "ungraded": True,
    },
    {
        "kind": "explain-why",
        "prompt": "Explain why the sky looks blue during the day.",
        "ground_truth": "Short (blue) wavelengths scatter more than long (red) ones",
        "must_contain": ["scatter", "blue"],
    },
    {
        "kind": "counterfactual",
        "prompt": "What would happen to the oceans if the Moon suddenly disappeared?",
        "ground_truth": "Tides would shrink to the small solar-driven component",
        "must_contain": ["tide"],
    },
    {
        "kind": "ambiguous",
        # The interesting question here is not which name it picks but whether
        # it picks one confidently or hedges -- so it is deliberately ungraded.
        "prompt": "Who was the greatest scientist of all time?",
        "ground_truth": None,
        "must_contain": [],
        "ungraded": True,
    },
    {
        "kind": "creative",
        # Highest degeneration risk in the set: "produce a list" is the classic
        # way a 1.5B model starts looping. The per-prompt status line reports it.
        "prompt": "Give five names for a small coffee shop.",
        "ground_truth": None,
        "must_contain": [],
        "ungraded": True,
    },
    {
        "kind": "math",
        # Deliberately a division, not a multiplication. The intuitive move is
        # 25 * 1.2 = 30, which is wrong; the right move is 25 / 0.8 = 31.25. A
        # model that does not deliberate tends to take the intuitive move, so
        # this separates the two models by ANSWER as well as by attention.
        "prompt": "If a jacket costs $25 after a 20% discount, what was its original price?",
        "ground_truth": "31.25",
        "must_contain": ["31.25"],
    },
    {
        "kind": "math-trap",
        "prompt": "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?",
        "ground_truth": "0.05",
        "must_contain": ["0.05"],
        # The famous intuitive-but-wrong answer is 0.10. Correct phrasings the
        # crude marker MISSES: "5 cents", "five cents" -- read the response.
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

    if spec.get("ungraded"):
        # An open-ended prompt has no marker that can be "right", so an empty
        # must_contain would fall through to `not misses` and report a vacuous
        # CORRECT -- a verdict with no basis. Say so instead.
        return {
            "verdict": "UNGRADED (open-ended -- read the response)",
            "hits": [],
            "misses": [],
            "positions": {},
            "scope_chars": len(region),
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


# Tokens that END a reply, passed to generate() as eos_token_id.
#
# Why this exists: both models were generating to the 1024-token cap, and the
# loops that followed ('iziñiziz...' for IFT, 'übübüb...' for Reasoning) looked
# like a model defect. They are not. The diagnostic window showed IFT emitting
# '<|im_end|>' at generated token 9 -- right after 'Paris is the capital of
# **France**.' -- and Reasoning emitting it at token 300, right after
# '**Answer:** France.'. Both models signalled stop at their natural end.
#
# generate() was never told what that signal was. The tokenizer reports
# eos_token_id 151645 ('<|im_end|>'), but generate() reads the MODEL's
# generation_config, which evidently does not include it, so the stop token was
# just another token. Every loop we spent two sessions building detectors for
# was a reply that had already finished.
#
# '<|im_start|>' is deliberately NOT here: it opens a turn rather than ending
# one, and stopping on it would cut off a generation that had legitimately
# begun a new turn. It stays in TURN_BOUNDARY_TOKENS, where it is a drift
# signal rather than a stop.
STOP_TOKEN_STRINGS = ("<|im_end|>", "<|endoftext|>")


# Chat control tokens that should never appear inside a single-turn answer.
# If the model emits one, it has closed its own reply and opened a new turn --
# it is role-playing the conversation continuing, and everything after that
# point is not the model answering the question.
#
# With the stop condition fixed this should now be a guard that never fires.
# It is kept because a guard that never fires is cheap, and because a silent
# regression in the stop set would otherwise be invisible.
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


def stop_token_ids(tokenizer):
    """Ids that should end generation, for generate(eos_token_id=...).

    A DIFFERENT set from turn_boundary_ids(): '<|im_start|>' is excluded here
    because it opens a turn rather than ending one.

    Returns a list, not a set, because generate() also accepts a single int and
    an order-stable list keeps the value easy to print and to fold into the
    cache key. Duplicates are dropped in case a tokenizer maps two of these
    strings to the same id.
    """

    ids = []

    for token in STOP_TOKEN_STRINGS:

        token_id = tokenizer.convert_tokens_to_ids(token)

        if token_id is None or token_id == tokenizer.unk_token_id:
            continue

        if token_id not in ids:
            ids.append(token_id)

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
    quiet=False,
):
    """How many leading generated tokens the average may use, and why.

    One place decides the trim for both models, so the two cannot silently
    drift apart -- which is how the sink-column bug survived review earlier.
    Returns the count; the caller slices.

    quiet=True suppresses the per-model block and returns the count alone. The
    cross-type loop calls this once per model per prompt, where that block would
    be ~6 lines x 14 runs; it prints _guard_status() instead, built from the
    same two detectors so the compact line cannot disagree with the trim.
    """

    drift_start = find_drift_start(generated_ids, boundary_ids)
    trim_start = earliest_trim(drift_start, degenerate_start)

    n_generated = len(generated_ids)
    n_used = n_generated if trim_start is None else trim_start

    if quiet:
        return n_used

    print(f"\n{label} attention window:")

    if drift_start is None:
        print(
            f"  stop  : no turn terminator in {n_generated} tokens, so the model "
            "was still mid-reply when generation ended"
        )
    elif drift_start == n_generated - 1:
        # The terminator is the LAST generated token, so generation stopped on
        # it. That is the model ending its own reply, not drift -- and telling
        # the two apart is the whole reason this branch exists. Post-fix this
        # is the expected case; the drift branch below should be dead.
        print(
            f"  stop  : clean stop on a turn terminator at token {drift_start} of "
            f"{n_generated} -- the model ended its own reply"
        )
    else:
        print(
            f"  drift : the model ended its reply at token {drift_start} of "
            f"{n_generated} and then kept generating -- everything after that "
            "point is not an answer"
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


def verify_drift_cut(tokenizer, generated_ids, boundary_ids, window=4):
    """Print the tokens either side of the drift cut, or say there was none.

    A drift cut is only trustworthy if the token it stopped on really is a chat
    turn marker and not an ordinary token that happens to share its id. The id
    alone cannot settle that; the decoded text can. This window is what makes
    the cut auditable instead of merely plausible.

    Deliberately separate from resolve_attention_trim(): this needs a tokenizer,
    and keeping the tokenizer out of the trim function is what lets the trim be
    unit-tested against plain integer lists.

    Also prints the tokenizer's eos/pad ids, because a turn marker appearing at
    token 9 of a 1024-token generation means generation did NOT stop on it --
    which only makes sense if eos is some other token.
    """

    eos_id = getattr(tokenizer, "eos_token_id", None)
    eos_str = getattr(tokenizer, "eos_token", None)
    pad_id = getattr(tokenizer, "pad_token_id", None)
    pad_str = getattr(tokenizer, "pad_token", None)

    print(f"  tokenizer eos={eos_id} ({eos_str!r})   pad={pad_id} ({pad_str!r})")

    index = find_drift_start(generated_ids, boundary_ids)

    if index is None:
        print("  no drift cut for this model, so there is nothing to verify")
        return

    ids = (
        generated_ids.tolist()
        if hasattr(generated_ids, "tolist")
        else list(generated_ids)
    )

    lo = max(0, index - window)
    hi = min(len(ids), index + window + 1)

    print(f"  cut at generated token {index}; boundary ids {sorted(boundary_ids)}")
    print(f"  decoded tokens {lo}..{hi - 1}:")

    for i in range(lo, hi):

        flag = "   <-- CUT: this token ends the kept window" if i == index else ""

        print(f"    [{i:4d}] id={ids[i]:6d}  {tokenizer.decode([ids[i]])!r}{flag}")


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

    # The model's own eos is printed rather than trusted. It is the reason the
    # previous runs went to the token cap: whatever this says, it was missing
    # the chat turn terminator, so the model's stop signal was ignored.
    configured_eos = getattr(model.generation_config, "eos_token_id", None)

    stop_ids = stop_token_ids(tokenizer)

    print(
        f"Stop condition: tokenizer gives {stop_ids}"
        f"   (model config eos was {configured_eos})"
    )

    output_ids = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        use_cache=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=stop_ids,
    )

    generated_ids = output_ids[0, prompt_length:]

    generated_text = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True
    )

    if generated_ids.shape[0] >= MAX_NEW_TOKENS:
        print(
            f"NOTE: generation reached the {MAX_NEW_TOKENS}-token cap, so the "
            "reply may be TRUNCATED rather than finished -- and on non-factual "
            "prompts the reasoning model does run past 1024 while still inside "
            "its trace. The status line reports whether a loop fired, but a "
            "model can wander or self-repeat inside <think> without ever "
            "becoming periodic, and no detector here would see that. Read the "
            "generation summary printed with each prompt."
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


def cache_path(which, prompt, stop_strings=STOP_TOKEN_STRINGS):
    """Cache file for one (model, prompt, stop set, token cap) combination.

    The cap AND the stop set are both folded into the digest, so changing either
    invalidates the cache instead of silently serving a run produced under
    different conditions.

    That is not hypothetical: adding '<|im_end|>' to the stop set turns a
    1024-token loop into a 9-token answer. Reusing the old file would have made
    the fix look like it did nothing. Folding the stop strings into the hash
    rather than into the filename keeps old files quietly unused rather than
    piling up unreadable names.
    """

    key = prompt + "\x00" + ",".join(sorted(stop_strings))

    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]

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


# ============================================================
# STAGE C HELPERS: PROMPT-ATTENTION SHAPE, PHASE-MATCHED
# ============================================================
#
# Every comparison above averages over the WHOLE generation. That is wrong for
# two models whose generations do different things: the reasoning model spends
# ~85% of its tokens thinking and the IFT model spends 0% of its tokens thinking,
# because it has no thinking phase. So "IFT vs Reasoning" was partly comparing
# answering against deliberating.
#
# The reframe: stop comparing how MUCH attention lands on the prompt (that is a
# length/phase property and it is out of scope now), and compare the SHAPE -- the
# distribution of attention across the prompt's own words, renormalized to sum to
# 1. Shape is a within-model, within-prompt quantity, so the two models' unequal
# generation lengths cancel out of it.
#
# Two comparisons, and the second is what makes the first readable:
#
#   C1  Reasoning thinking vs Reasoning answering -- SAME model, same prompt,
#       same weights. Any shape difference here is a pure PHASE effect and
#       cannot be a model difference. This is the yardstick.
#   C2  IFT vs Reasoning-answering -- both models writing their answer. Matched
#       phase at last: IFT's 9 tokens are all answer, Reasoning's post-</think>
#       tokens are all answer.
#
# If C1's distance is small, the phase worry is dead and C2 is a clean model
# comparison. If C1's distance is large, the phase effect is real and C2 must be
# read with that number beside it. Either answer is a result.


def _as_numpy(attention):
    """Accept a torch tensor or an ndarray; return a float64 ndarray.

    The cache path yields a torch tensor, the pure-helper path yields numpy, and
    the unit tests yield numpy. One converter keeps every caller the same.
    """

    if hasattr(attention, "detach"):
        attention = attention.detach().cpu()

    return np.asarray(attention, dtype=np.float64)


def find_trace_end(generated_ids, tokenizer, end_tags=REASONING_END_TAGS, report=None):
    """Index of the first generated token AFTER the reasoning trace closes.

    Returns None when no closing tag is present in the generation. None means
    "no answer phase exists to compare", and every caller must SKIP rather than
    fall back to the whole window -- falling back is how the chickens/cows run
    graded a marker found inside an unclosed trace.

    Search order, token-identity first and text second:

      1. The tag as an exact ID SEQUENCE, scanned over generated_ids. This is
         the primary path and it never touches decoded text, so there are no
         offsets and no whitespace to get wrong.

      2. Only if (1) finds nothing: concatenate the per-token STRINGS from
         convert_ids_to_tokens and map the tag's end character back to a token
         index.

    `report`, when a list is passed, gets the name of the path that fired
    appended to it ("id-sequence" / "token-strings"). Which path found the
    boundary was invisible in S4 -- the cut was provably right, but which code
    produced it could not be read off the output. A list rather than a changed
    return type, so every existing caller keeps its plain int-or-None.

    Note what is deliberately NOT used: tokenizer.decode(). Its default is
    skip_special_tokens=True, which silently DELETES <|im_end|> / <think> and
    every later offset shifts by the length of what was removed. That is the
    same class of mistake as the eos_token_id bug -- trusting a default instead
    of checking it -- and it is why neither path here calls decode().
    """

    ids = (
        generated_ids.tolist()
        if hasattr(generated_ids, "tolist")
        else list(generated_ids)
    )

    # 1. Exact id-sequence match.
    for tag in end_tags:

        tag_ids = tokenizer.encode(tag, add_special_tokens=False)

        if not tag_ids:
            continue

        n = len(tag_ids)

        for i in range(len(ids) - n + 1):

            if ids[i:i + n] == list(tag_ids):
                if report is not None:
                    report.append("id-sequence")
                return i + n

    # 2. Per-token string fallback.
    token_strings = tokenizer.convert_ids_to_tokens(ids)

    joined = ""
    spans = []

    for index, token_string in enumerate(token_strings):
        spans.append((len(joined), len(joined) + len(token_string), index))
        joined += token_string

    for tag in end_tags:

        position = joined.find(tag)

        if position == -1:
            continue

        tag_end = position + len(tag)

        for _, span_end, index in spans:

            if span_end >= tag_end:
                if report is not None:
                    report.append("token-strings")
                return index + 1

    if report is not None:
        report.append("not-found")

    return None


def prompt_shape(attention, user_start, user_end, lo=0, hi=None):
    """Renormalized attention SHAPE over the user-prompt tokens, one phase.

    attention : [generated_tokens, layers, heads, prompt_tokens]
    lo, hi    : generated-token window, hi exclusive; hi=None means "to the end"

    Returns a 1-D float array of length (user_end - user_start) summing to 1.

    The sink is excluded by construction: user_start/user_end delimit the raw
    user prompt inside the chat template, and the sink sits at prompt index 0,
    outside that range. The sink is reported separately elsewhere; it must not
    be part of a shape, because a shape is a statement about the question's own
    words.

    The normalization is the whole point: it divides out HOW MUCH attention
    reached the prompt, leaving only how that attention was spread across the
    words. Two models with wildly different prompt-attention totals therefore
    become directly comparable.

    Raises rather than returning zeros: an empty window or an all-zero sum means
    the caller asked a question with no answer, and a silent nan would propagate
    into every number below it.
    """

    attention = _as_numpy(attention)

    if hi is None:
        hi = attention.shape[0]

    if lo < 0 or hi > attention.shape[0] or hi <= lo:
        raise ValueError(
            f"empty or out-of-range phase window [{lo}:{hi}] for a generation "
            f"of {attention.shape[0]} tokens"
        )

    # Mean over generated tokens in the window, then over layers and heads.
    per_prompt_token = attention[lo:hi].mean(axis=(0, 1, 2))

    user = per_prompt_token[user_start:user_end]

    total = float(user.sum())

    if total <= 0:
        raise ValueError(
            "attention on the user prompt sums to zero over this window, so no "
            "shape exists"
        )

    return user / total


def total_variation_distance(a, b):
    """Half the L1 distance between two distributions.

    0.0 = identical shapes; 1.0 = no overlap at all. Both inputs must already be
    normalized (prompt_shape does that). Reported instead of a raw mean-|diff|
    because it has a fixed 0-1 range, so two distances can be compared across
    different prompts without knowing the token count.
    """

    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")

    return float(np.abs(a - b).sum() / 2.0)


def verify_phase_cut(
    tokenizer,
    generated_ids,
    cut_index,
    end_tags=REASONING_END_TAGS,
    window=4,
    quiet=False,
):
    """Print what a trace-end cut landed on, three ways, before it is trusted.

    Same discipline as verify_drift_cut, which is what exposed the eos bug: one
    line of decoded output around the boundary killed two sessions of wrong
    theory. An off-by-one here would reshape the answer phase and quietly change
    every Stage C number.

    Prints, for the tokens either side of the cut:
      * the raw ids
      * the per-token strings via convert_ids_to_tokens (shows <|...|> and the
        G/C space markers)
      * decode(..., skip_special_tokens=False) as text
    and cross-checks the cut against final_answer_region() on the full text, so
    the token-space boundary and the text-space boundary can be compared
    directly.

    quiet=True keeps the audit but drops it to one line -- the decoded text
    either side. The cross-type loop calls this 7 times, where the full block
    would be ~12 lines each, and the text either side is the part that has
    actually ever caught anything.
    """

    ids = (
        generated_ids.tolist()
        if hasattr(generated_ids, "tolist")
        else list(generated_ids)
    )

    if cut_index is None:

        print("phase cut: NONE -- no trace-closing tag found in the generation.")
        print(f"  tags searched: {end_tags}")
        print("  the phase comparison must be skipped, not fall back to all tokens")

        return

    before = max(0, cut_index - window)
    after = min(len(ids), cut_index + window)

    if quiet:

        before_text = tokenizer.decode(
            ids[max(0, cut_index - 2):cut_index], skip_special_tokens=False
        )
        after_text = tokenizer.decode(
            ids[cut_index:min(len(ids), cut_index + 2)], skip_special_tokens=False
        )

        print(
            f"  cut @{cut_index} of {len(ids)}:"
            f" {before_text!r} -> {after_text!r}"
        )

        return

    print(
        f"phase cut: answering starts at generated token {cut_index} of {len(ids)}"
    )
    print(
        f"  thinking : {cut_index} tokens"
        f"   answering : {len(ids) - cut_index} tokens"
    )

    print("  raw ids either side:")
    print(f"    before: {ids[before:cut_index]}")
    print(f"    after : {ids[cut_index:after]}")

    print("  per-token strings either side:")
    print(f"    before: {tokenizer.convert_ids_to_tokens(ids[before:cut_index])}")
    print(f"    after : {tokenizer.convert_ids_to_tokens(ids[cut_index:after])}")

    print("  text either side (skip_special_tokens=False):")
    print(
        f"    before: "
        f"{tokenizer.decode(ids[before:cut_index], skip_special_tokens=False)!r}"
    )
    print(
        f"    after : "
        f"{tokenizer.decode(ids[cut_index:after], skip_special_tokens=False)!r}"
    )

    full_text = tokenizer.decode(ids, skip_special_tokens=False)
    region = final_answer_region(full_text)

    print("  text-space cross-check (final_answer_region):")

    if region is None:
        print("    NONE -- the tag was found in id/string space but NOT in the")
        print("    decoded text. Do NOT trust this cut; report it before reading")
        print("    any Stage C number.")
    else:
        print(f"    region starts: {region[:60]!r}")


def locate_user_prompt(tokenizer, prompt):
    """(start, end, labels) for the raw prompt inside its chat template.

    The chat template wraps the prompt in control tokens, so the prompt's own
    tokens sit at an offset that depends on the prompt. Matching the raw
    tokenization as a substring of the templated ids is how that offset is found
    WITHOUT hardcoding a range -- a hardcoded range breaks silently on the next
    prompt, which is exactly the trap the S4 run's "24..31" would have set.

    Returns the half-open [start, end) range in FULL-sequence coordinates, and
    the per-token labels for the plot's y-axis.

    Raises when the raw tokenization does not appear in the template, rather than
    returning a wrong range: a wrong range produces a shape over the wrong tokens
    and every number downstream would be mislabelled but plausible.
    """

    user_ids = tokenizer.encode(prompt, add_special_tokens=False)

    messages = [{"role": "user", "content": prompt}]

    full_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )["input_ids"][0]

    n = len(user_ids)
    needle = torch.tensor(user_ids)

    start = None

    for i in range(len(full_ids) - n + 1):

        if torch.equal(full_ids[i:i + n], needle):
            start = i
            break

    if start is None:
        raise ValueError(
            "the prompt's own tokenization does not appear inside its chat "
            f"template, so its token range cannot be located: {prompt!r}"
        )

    labels = [tokenizer.decode([token_id]) for token_id in user_ids]

    return start, start + n, labels


def per_word_diff(a, b):
    """Per-word difference b - a, in the same normalized units as the shapes.

    Kept separate from the plot because this is where a bug would hide: the SIGN
    is what the chart reads, so the sign is what gets unit-tested. A chart drawn
    from a sign-flipped diff looks entirely plausible.
    """

    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")

    return b - a


def plot_shape_diff(labels, diff, title, up_label, down_label):
    """Diverging horizontal bars: where `b` puts more (or less) weight than `a`.

    Bars right of zero are words `b` weights more than `a`; bars left are words
    `a` weights more. Values are PERCENTAGE POINTS, and the bars' absolute values
    sum to 2 x the total-variation distance, so the chart and the TV number agree
    by construction.

    Returns the figure; the caller displays it. Nothing is saved to disk -- the
    figures are for reading in the notebook.

    Read the SHAPE of the bars, not the level: a long bar means the two windows
    disagree about that word. A word both windows ignore gives a short bar, and a
    word both windows care about equally also gives a short bar -- this view
    cannot tell those apart. The paired table printed above the chart can.
    """

    points = np.asarray(diff, dtype=np.float64) * 100.0

    y = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(9.5, 0.42 * len(labels) + 2.1))

    colors = ["#c0392b" if value >= 0 else "#1f618d" for value in points]

    ax.barh(y, points, color=colors, height=0.68)

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.axvline(0.0, color="black", linewidth=1.0)

    span = max(float(np.abs(points).max()), 1.0)
    offset = span * 0.025

    for yi, value in zip(y, points):

        if value >= 0:
            ax.text(
                value + offset, yi, f"{value:+.1f}",
                va="center", ha="left", fontsize=9,
            )
        else:
            ax.text(
                value - offset, yi, f"{value:+.1f}",
                va="center", ha="right", fontsize=9,
            )

    ax.set_xlim(-span * 1.3, span * 1.3)
    ax.set_xlabel("percentage points of prompt attention")
    ax.set_title(
        f"{title}\nright = {up_label}   |   left = {down_label}",
        fontsize=10,
    )
    ax.grid(axis="x", alpha=0.25)
    ax.set_axisbelow(True)

    fig.tight_layout()

    return fig


def _guard_parts(label, generated_ids, boundary_ids, trim_start, trim_period):
    """How a generation ended -- stop, drift, loop, cap -- as DATA.

    Split out of _guard_status() so the per-prompt RESPONSE heading and the
    guard line can both state the same fate. Two copies of this logic would
    eventually disagree, and a heading that says "ended its own reply" above a
    guard line that says CAP-HIT is worse than no heading at all.

    Built from the SAME two detectors resolve_attention_trim() uses, so nothing
    here can disagree with the trim it describes.
    """

    n = len(generated_ids)
    drift = find_drift_start(generated_ids, boundary_ids)

    if drift is None:
        stop = "no-stop (mid-reply)"
    elif drift == n - 1:
        stop = "clean-stop"
    else:
        stop = f"DRIFT@{drift}"

    if trim_start is None:
        loop = "none"
    else:
        loop = f"LOOP@{trim_start}/period {trim_period}"

    return {
        "label": label,
        "n": n,
        "stop": stop,
        "loop": loop,
        "cap": "CAP-HIT" if n >= MAX_NEW_TOKENS else "under-cap",
    }


def _guard_status(label, generated_ids, boundary_ids, trim_start, trim_period):
    """One line describing how a generation ended: stop, drift, loop, cap.

    Exists because the run spans many prompts: the six-line per-model block is
    right for one prompt and unreadable for fourteen. The line keeps the
    guards visible -- a prompt that degenerated, or one that hit the cap, is
    reported rather than buried.
    """

    p = _guard_parts(label, generated_ids, boundary_ids, trim_start, trim_period)

    return (
        f"{p['label']:10s} {p['n']:5d} tok | {p['stop']:16s} | "
        f"loop: {p['loop']:22s} | {p['cap']}"
    )


def response_heading(label, parts):
    """The heading above a full response: 'IFT RESPONSE (74 tok -- ...)'.

    The fate is read off _guard_parts, the same dict the guard line formats, so
    the heading cannot claim the model finished while the diagnostic line says
    it hit the cap. Cap outranks the stop reason: a generation that ended on the
    final token of the cap has not been shown to have chosen to stop.
    """

    if parts["cap"] == "CAP-HIT":
        fate = "CAPPED mid-reply"
    elif parts["stop"] == "clean-stop":
        fate = "ended its own reply"
    else:
        fate = parts["stop"]

    return f"{label} RESPONSE  ({parts['n']} tokens -- {fate})"


def shape_table(labels, ift_shape, think_shape, answer_shape,
                diff_models, diff_phase):
    """The one per-token table: three normalized shapes and both differences.

    The two diff columns are PASSED IN rather than recomputed here. They are the
    exact arrays the C1/C2 figures draw, so the printed numbers and the bars
    cannot disagree -- not through a second call, and not through a later edit
    that changes one call site and not the other.

    The SUM row is the invariant, not decoration: each shape column must sum to
    1, so each diff column must sum to ~0. If a slice is ever added without
    renormalising, the SUM row says so before any distance is read.
    """

    a = np.asarray(ift_shape, dtype=np.float64)
    t = np.asarray(think_shape, dtype=np.float64)
    r = np.asarray(answer_shape, dtype=np.float64)

    d_models = np.asarray(diff_models, dtype=np.float64)
    d_phase = np.asarray(diff_phase, dtype=np.float64)

    df = pd.DataFrame({
        "token": list(labels),
        "IFT_answering": a,
        "R_thinking": t,
        "R_answering": r,
        "R_answer_minus_IFT": d_models,
        "R_answer_minus_thinking": d_phase,
    })

    df.loc[len(df)] = [
        "SUM",
        a.sum(),
        t.sum(),
        r.sum(),
        d_models.sum(),
        d_phase.sum(),
    ]

    return df


def inference_facts(rows, total_prompts, skipped_indices):
    """Counts from this run, and nothing else.

    Deliberately mechanical. A template cannot tell "the model refused to stop"
    from "the prompt got truncated", so no line here states a cause -- every
    line is a count that can be checked against the blocks above it. The reading
    of those counts is written by hand afterwards.
    """

    traced = [row for row in rows if row.get("C1_phase") is not None]

    lines = [
        f"prompts with a phase cut, i.e. with a C1/C2 number: "
        f"{len(traced)} of {total_prompts}",
        f"prompts skipped, i.e. no answering phase at all: "
        f"{len(skipped_indices)} of {total_prompts}",
    ]

    if traced:

        c1_bigger = [row for row in traced if row["C1_larger"] == "yes"]

        lines.append(
            f"C1 > C2, phase effect bigger than model gap: "
            f"{len(c1_bigger)} of {len(traced)}"
        )
        lines.append(
            "C1 / C2 per prompt: "
            + ", ".join(
                f"p{row['prompt']} {row['C1_phase']:.4f}/{row['C2_models']:.4f}"
                for row in traced
            )
        )

    if skipped_indices:
        lines.append(
            "skipped prompts: " + ", ".join(str(i) for i in skipped_indices)
        )

    return lines


def render_figure_png(fig, dpi=130):
    """PNG bytes for a figure.

    Split out from show_figure() so the bytes themselves are testable -- a
    display path that silently produces an empty image is exactly the S1 failure
    ("the chart has never been seen"), and this is the part that can be checked
    without a browser.
    """

    buffer = io.BytesIO()

    fig.savefig(buffer, format="png", dpi=dpi, bbox_inches="tight")

    data = buffer.getvalue()

    buffer.close()

    return data


def show_figure(fig, dpi=130):
    """Display a figure as an explicit PNG, and free it.

    Deliberately NOT `display(fig)`. That relies on IPython having attached a
    `_repr_png_` to the Figure object, which happens inside a kernel's inline
    setup -- and which could not be confirmed to fire here at all, in or out of a
    kernel. A display call that silently renders nothing is worse than one that
    errors, because the run looks successful.

    Encoding the PNG ourselves is explicit, so the image either appears or the
    failure is visible at the line that caused it.

    Nothing is written to disk; the buffer is discarded after display.
    """

    display(Image(data=render_figure_png(fig, dpi=dpi)))

    plt.close(fig)


def section(title):
    """A BOLD heading in the output.

    print() writes plain text, so `print("**text**")` shows the asterisks rather
    than emphasising anything. Colab renders Markdown, so a heading that is
    meant to be skimmable has to go through display(Markdown(...)).
    """

    display(Markdown(f"**{title}**"))


def claude_block(lines):
    """The diagnostics, under a heading that says who they are for in bold.

    The heading is the contract: everything between it and the next heading is
    for whoever is checking the numbers, not for whoever wants the result. One
    place, always the same place -- so the reader learns one thing to skip
    instead of hunting for which of five scattered debug prints matters.
    """

    display(Markdown("**FOR CLAUDE -- SKIP THIS BLOCK (diagnostics)**"))

    for line in lines:

        # A captured diagnostic can be several lines (verify_phase_cut's
        # no-cut message is three). Indenting only the first would let the rest
        # escape the block visually, so every line is indented.
        parts = str(line).splitlines() or [""]

        for part in parts:
            print("      " + part)


def capture_stdout(func, *args, **kwargs):
    """(return value, captured stdout) for a function that prints its audit.

    verify_phase_cut() in quiet mode prints its one-line audit rather than
    returning it, and the loud path must keep doing exactly that -- the harness
    pins both behaviours. Capturing the line here puts it in reading order
    inside the diagnostics block instead of wherever the call happened to sit.
    """

    buffer = io.StringIO()

    with contextlib.redirect_stdout(buffer):
        result = func(*args, **kwargs)

    return result, buffer.getvalue()


def generation_summary(text, tail=700):
    """Two lines describing a generation: its shape, then its end.

    Returns a LIST of lines, so the caller indents them without the helper
    having to know about prefixes.

    Why this and not the whole text: on non-factual prompts the reasoning model
    runs to the 1024 cap with its trace still open, and the two things that need
    reading are (a) whether the trace opened and closed at all, and (b) what the
    model was doing when generation stopped. Printing several thousand
    characters per prompt would put the report back to being the wall of text
    the plots exist to replace.

    `</think>` is ordinary tokens here, not a special token, so a plain
    skip_special_tokens=True decode keeps it. Only <|im_end|> / <|endoftext|>
    are dropped -- which is why `<|im_end|>` never appears in the text.
    """

    text = "" if text is None else text

    opened = "<think>" in text
    closed = "</think>" in text

    if len(text) <= tail:
        body = text
    else:
        body = "..." + text[-tail:]

    return [
        f"{len(text)} chars | <think> opened: {'yes' if opened else 'no'}"
        f" | </think> closed: {'yes' if closed else 'no'}",
        f"tail: {body!r}",
    ]


# ============================================================
# STAGE C ACROSS PROMPT TYPES  --  THIS IS THE RUN
# ============================================================
#
# The single-prompt deep dive that used to run first (Stage A sink heads, Stage
# B position vs meaning, one prompt's full tables) is gone. `%run` executes top
# to bottom, so it always printed a differently-shaped block BEFORE this sweep
# and the output read as two experiments stapled together. Its functions are
# still defined above and still covered by the local harness.
#
# The question this run answers: does the IFT-vs-Reasoning attention-shape gap
# depend on the KIND of question? The old single prompt was a factual lookup
# with one retrievable answer, so there was little for the two models to differ
# about. This sweeps every spec carrying a `kind` label -- opinion, explain-why,
# counterfactual, ambiguous, creative, math, math-trap.
#
# Per prompt the output is, IN THIS ORDER:
#
#   1. the prompt
#   2. IFT's FULL response, then Reasoning's FULL response
#   3. one table: all three normalized shapes per token + both differences
#   4. two diverging-bar figures -- C2 (model comparison), C1 (phase yardstick)
#   5. a BOLD "FOR CLAUDE -- SKIP" block holding the diagnostics
#
# The bold heading is the contract: everything under it is for checking the
# numbers, not for reading the result. It is bold because print() cannot be --
# print("**x**") shows the asterisks.
#
# The measurement, unchanged from S4:
#   shape = attention renormalized over the prompt's own words, sums to 1
#   C1    = Reasoning thinking vs Reasoning answering, same model
#   C2    = IFT answering vs Reasoning answering, phase-matched
# The sink at full-sequence index 0 sits OUTSIDE the user-token slice, so it is
# excluded structurally and can never leak into a shape.
#
# NEW_PROMPT_INDICES is DERIVED from PROMPT_SPECS, not a second hand-kept list.
# Adding a spec with a `kind` is all it takes; two parallel lists edited in
# lockstep is how the old TEST_PROMPT_INDEX drifted once.

NEW_PROMPT_INDICES = [
    index for index, spec in enumerate(PROMPT_SPECS) if spec.get("kind")
]

section(f"STAGE C ACROSS PROMPT TYPES -- {len(NEW_PROMPT_INDICES)} prompts")
print(
    "   kinds: "
    + ", ".join(f"{i}:{PROMPT_SPECS[i]['kind']}" for i in NEW_PROMPT_INDICES)
)


def stage_c_for_prompt(index):
    """One prompt, one block, in reading order.

    Ordering is the point. Before this, the response text was a 700-char tail
    printed several screens away from the table describing it, with the
    diagnostics wedged in between. Now the prompt, both full responses, the
    table and the two figures are contiguous, and every diagnostic is collected
    and printed ONCE at the end under a heading that says to skip it.

    Returns a row dict, or None when the prompt is not comparable (no
    trace-closing tag, an empty phase, or a shape that is not normalized).

    A skipped prompt still prints BOTH FULL RESPONSES: on the five prompts where
    the reasoning model never left <think>, the response IS the result, and a
    status line saying CAP-HIT is not something a reader can check. What a
    skipped prompt does not get is a table or a figure -- a chart drawn from a
    phase that was never located carries the same false label a number would.
    """

    spec = PROMPT_SPECS[index]
    prompt = spec["prompt"]
    kind = spec["kind"]

    print("\n" + "=" * 72)
    print(f"PROMPT {index}   |   kind: {kind}")
    print("=" * 72)
    print(prompt)

    ift_run = run_and_cache("ift", ift_model, ift_tokenizer, prompt)
    reasoning_run = run_and_cache(
        "reasoning", reasoning_model, reasoning_tokenizer, prompt
    )

    ift_generated = ift_run["generated_ids"]
    reasoning_generated = reasoning_run["generated_ids"]

    ift_boundary_ids = turn_boundary_ids(ift_tokenizer)
    reasoning_boundary_ids = turn_boundary_ids(reasoning_tokenizer)

    # The fate facts, computed once. The response heading AND the guard line are
    # both formatted from these, so a heading cannot say a model ended its own
    # reply while the diagnostic line under it says the model hit the cap.
    ift_parts = _guard_parts(
        "IFT", ift_generated, ift_boundary_ids,
        ift_run["trim_start"], ift_run["trim_period"],
    )
    reasoning_parts = _guard_parts(
        "Reasoning", reasoning_generated, reasoning_boundary_ids,
        reasoning_run["trim_start"], reasoning_run["trim_period"],
    )

    section(response_heading("IFT", ift_parts))
    print(ift_run["text"])

    section(response_heading("REASONING", reasoning_parts))
    print(reasoning_run["text"])

    # Diagnostics accumulate here and print together at the end, so nothing
    # interrupts the prompt -> responses -> table -> figures reading order.
    diag = [
        _guard_status(
            "IFT", ift_generated, ift_boundary_ids,
            ift_run["trim_start"], ift_run["trim_period"],
        ),
        _guard_status(
            "Reasoning", reasoning_generated, reasoning_boundary_ids,
            reasoning_run["trim_start"], reasoning_run["trim_period"],
        ),
        # Line 0 only: the char count and the <think>/</think> flags. The tail is
        # line 1, and the whole response is printed above, so repeating it is
        # noise. Kept in the diagnostics because "did it ever close <think>" is
        # the fact that decides whether this prompt can be measured at all.
        "IFT       " + generation_summary(ift_run["text"])[0],
        "Reasoning " + generation_summary(reasoning_run["text"])[0],
    ]

    # quiet=True: one status line replaces the per-model block. The trim itself
    # still comes from the single shared function, so the two models cannot
    # drift apart on how they were cut.
    ift_end = resolve_attention_trim(
        "IFT",
        ift_generated,
        ift_boundary_ids,
        ift_run["trim_start"],
        ift_run["trim_period"],
        quiet=True,
    )
    reasoning_end = resolve_attention_trim(
        "Reasoning",
        reasoning_generated,
        reasoning_boundary_ids,
        reasoning_run["trim_start"],
        reasoning_run["trim_period"],
        quiet=True,
    )

    start, end, labels = locate_user_prompt(ift_tokenizer, prompt)

    # Which search path found the boundary was an open question after S4 -- the
    # cut was provably right, but which code produced it was invisible. The
    # report list answers it without changing find_trace_end's return type.
    path = []
    reasoning_cut = find_trace_end(
        reasoning_generated, reasoning_tokenizer, report=path
    )

    # verify_phase_cut() prints its one-line audit in quiet mode rather than
    # returning it, and the loud path must keep doing exactly that. Capturing
    # here puts the line inside the diagnostics block in reading order, instead
    # of wherever the call happens to sit.
    _, cut_line = capture_stdout(
        verify_phase_cut,
        reasoning_tokenizer,
        reasoning_generated,
        reasoning_cut,
        quiet=True,
    )

    diag.append(cut_line.strip())
    diag.append(f"trace-end found by: {path[0] if path else 'not-found'}")

    if reasoning_cut is None:

        print()
        print("   RESULT: skipped -- the reasoning response above never closed")
        print("   </think>, so there is no answering phase to compare against.")
        print("   Falling back to the whole generation would be labelled")
        print("   'matched phase' while not being matched. The response above IS")
        print("   the finding for this prompt.")

        claude_block(diag)

        return None

    think_tokens = reasoning_cut
    answer_tokens = reasoning_end - reasoning_cut

    if think_tokens < 1 or answer_tokens < 1:

        print()
        print(
            f"   RESULT: skipped -- the cut leaves an empty phase"
            f" (thinking {think_tokens}, answering {answer_tokens})."
        )

        claude_block(diag)

        return None

    ift_full = _as_numpy(ift_run["attention"])
    reasoning_full = _as_numpy(reasoning_run["attention"])

    if ift_end > ift_full.shape[0] or reasoning_end > reasoning_full.shape[0]:

        print()
        print("   RESULT: skipped -- a window is longer than the tensor it slices.")

        claude_block(diag + [
            f"IFT window {ift_end} / tensor {ift_full.shape[0]}",
            f"Reasoning window {reasoning_end} / tensor {reasoning_full.shape[0]}",
        ])

        return None

    ift_shape = prompt_shape(ift_full, start, end, lo=0, hi=ift_end)
    think_shape = prompt_shape(reasoning_full, start, end, lo=0, hi=reasoning_cut)
    answer_shape = prompt_shape(
        reasoning_full, start, end, lo=reasoning_cut, hi=reasoning_end
    )

    # Checked before any distance is quoted: a shape that does not sum to 1 means
    # the slicing is wrong, and a wrong slice still produces plausible numbers.
    bad = [
        f"{label} {float(np.abs(shape).sum()):.6f}"
        for label, shape in (
            ("IFT", ift_shape),
            ("Reasoning-think", think_shape),
            ("Reasoning-answer", answer_shape),
        )
        if abs(float(np.abs(shape).sum()) - 1.0) >= 1e-6
    ]

    if bad:

        print()
        print("   RESULT: skipped -- a shape does not sum to 1, so the slice is")
        print("   wrong and any distance from it would be wrong too.")

        claude_block(diag + ["shape sums: " + ", ".join(bad)])

        return None

    c2 = total_variation_distance(ift_shape, answer_shape)
    c1 = total_variation_distance(think_shape, answer_shape)

    # Computed ONCE and passed to both the table and the charts, so the printed
    # numbers and the bars cannot disagree through a second call.
    c2_diff = per_word_diff(ift_shape, answer_shape)
    c1_diff = per_word_diff(think_shape, answer_shape)

    diag += [
        f"windows: IFT {ift_end} tok answering |"
        f" Reasoning {think_tokens} thinking + {answer_tokens} answering",
        f"shapes sum to 1: IFT {float(ift_shape.sum()):.6f},"
        f" think {float(think_shape.sum()):.6f},"
        f" answer {float(answer_shape.sum()):.6f}",
        f"C1 (same model, thinking vs answering) = {c1:.4f}",
        f"C2 (IFT vs Reasoning, both answering)  = {c2:.4f}",
        "NOTE: both models are IN the answering phase, but IFT answers in a few",
        "tokens and Reasoning answers at length -- same phase, not same verbosity.",
    ]

    section(
        f"[{index}] {kind} -- attention per prompt word, normalized (sums to 1)"
    )
    print("   R_answer_minus_IFT      = the gap the top figure draws  (C2)")
    print("   R_answer_minus_thinking = the gap the bottom figure draws (C1)")
    display(
        shape_table(labels, ift_shape, think_shape, answer_shape, c2_diff, c1_diff)
    )

    show_figure(plot_shape_diff(
        labels,
        c2_diff,
        f"[{index}] {kind} -- C2: Reasoning answering vs IFT answering"
        f"  (TV {c2:.4f})",
        "Reasoning weights it more",
        "IFT weights it more",
    ))

    show_figure(plot_shape_diff(
        labels,
        c1_diff,
        f"[{index}] {kind} -- C1: answering vs thinking, same model"
        f"  (TV {c1:.4f})",
        "answering weights it more",
        "thinking weights it more",
    ))

    claude_block(diag)

    return {
        "prompt": index,
        "kind": kind,
        "C1_phase": round(c1, 4),
        "C2_models": round(c2, 4),
        "C1_larger": "yes" if c1 > c2 else "no",
        "IFT_tok": int(ift_end),
        "think_tok": int(think_tokens),
        "answer_tok": int(answer_tokens),
    }


stage_c_rows = []

for index in NEW_PROMPT_INDICES:

    # ValueError ONLY, and deliberately not a broader except.
    #
    # locate_user_prompt(), prompt_shape() and per_word_diff() all raise
    # ValueError ON PURPOSE for conditions that belong to a single prompt -- a
    # prompt whose tokenization does not survive templating, a window that is
    # empty, a shape that is all zeros. Without this, one such prompt aborts the
    # whole sweep and every prompt after it is lost, which is the same shape of
    # failure as the S1 broad `except Exception` that hid a NameError for a whole
    # run. ValueError is a type those functions choose; a NameError or a
    # TypeError from a typo would still come through and be seen.
    try:

        row = stage_c_for_prompt(index)

    except ValueError as exc:

        print(f"\n   SKIPPED prompt {index}: {type(exc).__name__}: {exc}")
        print("   (a condition of this prompt, not of the run -- later prompts")
        print("   are unaffected)")

        row = None

    if row is not None:
        stage_c_rows.append(row)


skipped = [
    index for index in NEW_PROMPT_INDICES
    if not any(row["prompt"] == index for row in stage_c_rows)
]

section("C1 / C2 ACROSS PROMPT TYPES")

if stage_c_rows:

    display(pd.DataFrame(stage_c_rows).set_index("prompt"))

else:

    print("   No prompt produced a comparable pair. Every prompt skipped; the")
    print("   block above for each prompt names its own reason.")

claude_block([
    "C1 = same model, thinking vs answering -- the phase yardstick.",
    "C2 = IFT vs Reasoning, both answering -- the model comparison.",
    "Read C2 only against its own row's C1: if C2 is not clearly smaller than",
    "C1, the model gap is not separable from the phase effect.",
])

section("WHAT WE CAN INFER")

for line in inference_facts(stage_c_rows, len(NEW_PROMPT_INDICES), skipped):
    print("   " + line)

print("\n   Counts only. The reading of them is written by hand after the run --")
print("   a template cannot tell 'the model refused to stop' from 'the prompt")
print("   was cut off'.")
