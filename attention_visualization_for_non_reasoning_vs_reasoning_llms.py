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
from IPython.display import Image, display

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


def _guard_status(label, generated_ids, boundary_ids, trim_start, trim_period):
    """One line describing how a generation ended: stop, drift, loop, cap.

    Built from the SAME two detectors resolve_attention_trim() uses, so this
    compact line cannot disagree with the trim it summarises.

    Exists because the run spans many prompts: the six-line per-model block is
    right for one prompt and unreadable for fourteen. The line keeps the
    guards visible -- a prompt that degenerated, or one that hit the cap, is
    reported rather than buried.
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

    cap = "CAP-HIT" if n >= MAX_NEW_TOKENS else "under-cap"

    return f"{label:10s} {n:5d} tok | {stop:16s} | loop: {loop:22s} | {cap}"


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

# Show what the cut actually landed on. A short window is the whole point here:
# IFT's answer is ~9 tokens, so an off-by-one or a false-positive boundary id
# would leave almost nothing to average and every number downstream would be
# measuring the wrong tokens.
verify_drift_cut(ift_tokenizer, ift_generated_ids, ift_boundary_ids)

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

verify_drift_cut(
    reasoning_tokenizer,
    reasoning_generated_ids,
    reasoning_boundary_ids,
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


# ============================================================
# STAGE C: PROMPT-ATTENTION SHAPE, PHASE-MATCHED
# ============================================================
#
# Everything above compares HOW MUCH attention each model puts on the prompt, or
# averages over the whole generation. Both are contaminated by the fact that the
# two models' generations do different jobs: the reasoning model deliberates
# first, the IFT model does not deliberate at all.
#
# Stage C drops the "how much" question entirely and asks only about SHAPE -- the
# distribution of attention across the prompt's own words, renormalized to sum to
# 1. Because it is renormalized, unequal prompt-attention totals cancel out, and
# because it is sliced to a single phase, unequal generation lengths cancel too.
#
#   C1  Reasoning thinking vs Reasoning answering  -> the PHASE yardstick.
#       Same model, so any difference here cannot be a model difference.
#   C2  IFT vs Reasoning answering                 -> the real comparison,
#       both models writing their answer.
#
# Nothing below runs unless the trace-end cut is confirmed by verify_phase_cut.

print("\n" + "=" * 64)
print("STAGE C: PROMPT-ATTENTION SHAPE, PHASE-MATCHED")
print("=" * 64)

print(f"Prompt: {test_prompt}")
print(f"User prompt tokens: {user_tokens}")
print(
    f"User prompt = full-sequence tokens {start}..{end - 1}"
    " (the sink at index 0 is outside this range, so it is never part of a shape)"
)

# ------------------------------------------------------------
# Where the reasoning trace closes, verified before it is used
# ------------------------------------------------------------

reasoning_cut = find_trace_end(reasoning_generated_ids, reasoning_tokenizer)

verify_phase_cut(reasoning_tokenizer, reasoning_generated_ids, reasoning_cut)

reasoning_full = _as_numpy(reasoning_run["attention"])
ift_full = _as_numpy(ift_run["attention"])

# The generation end, after the loop/drift trim. Post-eos this equals the full
# generation; printed either way so a trim that actually removed something is
# visible rather than assumed away.
reasoning_end = len(reasoning_attention_ids)
ift_end = len(ift_attention_ids)

print(
    f"\nWindows available --"
    f"  IFT {ift_end}/{ift_full.shape[0]} tokens"
    f"   Reasoning {reasoning_end}/{reasoning_full.shape[0]} tokens"
    f" (post-trim / generated)"
)

stage_c_ok = True

if reasoning_cut is None:

    stage_c_ok = False

    print("\nSKIPPED: no trace-closing tag in the reasoning generation, so its")
    print("answer phase cannot be located and C1/C2 would not be comparing what")
    print("they claim to. Nothing below is printed rather than printing numbers")
    print("under a label that is false.")

else:

    think_tokens = reasoning_cut
    answer_tokens = reasoning_end - reasoning_cut

    if think_tokens < 1 or answer_tokens < 1:
        stage_c_ok = False
        print(
            f"\nSKIPPED: the cut leaves an empty phase --"
            f" thinking {think_tokens}, answering {answer_tokens}."
        )

    if reasoning_end > reasoning_full.shape[0] or ift_end > ift_full.shape[0]:
        stage_c_ok = False
        print("\nSKIPPED: a window is longer than the tensor it slices.")

if stage_c_ok:

    ift_shape = prompt_shape(ift_full, start, end, lo=0, hi=ift_end)

    reasoning_think_shape = prompt_shape(
        reasoning_full, start, end, lo=0, hi=reasoning_cut
    )

    reasoning_answer_shape = prompt_shape(
        reasoning_full, start, end, lo=reasoning_cut, hi=reasoning_end
    )

    # Structural invariants. A shape that does not sum to 1, or phases that do
    # not add up to the generation, means the slicing is wrong -- so these are
    # checked before any distance is quoted.
    for label, shape in (
        ("IFT", ift_shape),
        ("Reasoning-think", reasoning_think_shape),
        ("Reasoning-answer", reasoning_answer_shape),
    ):
        total = float(np.abs(shape).sum())

        print(
            f"  {label:16s} sum of |weights| = {total:.6f}"
            f"   ({'ok' if abs(total - 1.0) < 1e-6 else 'BAD -- not normalized'})"
        )

    tv_phase = total_variation_distance(reasoning_think_shape, reasoning_answer_shape)
    tv_models = total_variation_distance(ift_shape, reasoning_answer_shape)

    print("\n" + "-" * 64)
    print("C1. THE PHASE YARDSTICK -- same model, thinking vs answering")
    print("-" * 64)
    print(
        f"Reasoning thinking : tokens 0..{reasoning_cut - 1}"
        f" ({think_tokens} tokens)"
    )
    print(
        f"Reasoning answering: tokens {reasoning_cut}..{reasoning_end - 1}"
        f" ({answer_tokens} tokens)"
    )

    display(pd.DataFrame({
        "token": user_tokens,
        "Reasoning_thinking": reasoning_think_shape,
        "Reasoning_answering": reasoning_answer_shape,
        "answering_minus_thinking": reasoning_answer_shape - reasoning_think_shape,
    }))

    print(
        f"\n  TOTAL VARIATION DISTANCE (thinking vs answering): {tv_phase:.4f}"
    )
    print("  0.00 = thinking and answering look at the prompt identically")
    print("  1.00 = they have nothing in common")
    print("  THIS IS THE NUMBER THAT DECIDES WHETHER C2 CAN BE TRUSTED.")

    print("\n" + "-" * 64)
    print("C2. THE COMPARISON -- IFT vs Reasoning, both answering")
    print("-" * 64)
    print(f"IFT window       : tokens 0..{ift_end - 1} ({ift_end} tokens)")
    print(
        f"Reasoning window : tokens {reasoning_cut}..{reasoning_end - 1}"
        f" ({answer_tokens} tokens), after its trace closed"
    )

    display(pd.DataFrame({
        "token": user_tokens,
        "IFT_attention": ift_shape,
        "Reasoning_answering": reasoning_answer_shape,
        "Reasoning_minus_IFT": reasoning_answer_shape - ift_shape,
    }))

    print(
        f"\n  TOTAL VARIATION DISTANCE (IFT vs Reasoning, both answering):"
        f" {tv_models:.4f}"
    )

    print("\n" + "=" * 64)
    print("HOW TO READ STAGE C")
    print("=" * 64)
    print("  * Compare C1's distance to C2's before believing C2.")
    print("  * C1 large (say, well above ~0.3): a model's shape depends heavily on")
    print("    whether it is thinking or answering. C2 is then only meaningful as a")
    print("    phase-matched comparison, and prompt-attention mass is off the table.")
    print("  * C1 small: the phase effect is minor, and C2's gap is closer to a real")
    print("    model difference.")
    print("  * IFT's window is ~a few tokens. A shape from that few samples is")
    print("    noisy -- read the table, not just the distance.")

    print(
        f"\nHEADLINE NUMBERS -- phase yardstick (C1, same model): {tv_phase:.4f}"
        f"   model comparison (C2, matched phase): {tv_models:.4f}"
    )

else:

    print("\nStage C produced no numbers. See the SKIPPED reason above.")


# ============================================================
# STAGE C ACROSS PROMPT TYPES
# ============================================================
#
# Everything above analyses TEST_PROMPT_INDEX alone. That answers "what happens
# on this one question" and cannot answer the question the reframe exists for:
# does the IFT-vs-Reasoning shape gap depend on the KIND of question?
#
# Index 5 is a factual lookup with a single retrievable answer, so both models
# converge and there is little for them to differ about. A prompt with no
# retrievable answer is where a real difference would show -- and a math or logic
# prompt with a trap answer is where deliberation demonstrably does work.
#
# This block runs the same Stage C measurement over every spec carrying a `kind`
# label and prints, per prompt: a status line, the window sizes, C1 and C2, the
# paired per-word table, and TWO diverging-bar figures --
#
#   C2  Reasoning answering vs IFT answering    (the model comparison)
#   C1  Reasoning answering vs Reasoning thinking (the phase yardstick)
#
# The loop is DERIVED from PROMPT_SPECS, not a second hand-kept list. Adding a
# spec with a `kind` is all it takes to include it; two parallel lists that must
# be edited in lockstep is how TEST_PROMPT_INDEX already drifted once.

NEW_PROMPT_INDICES = [
    index for index, spec in enumerate(PROMPT_SPECS) if spec.get("kind")
]

print("\n" + "=" * 64)
print("STAGE C ACROSS PROMPT TYPES")
print("=" * 64)
print(
    f"{len(NEW_PROMPT_INDICES)} prompts carry a `kind` label: "
    + ", ".join(f"{i}:{PROMPT_SPECS[i]['kind']}" for i in NEW_PROMPT_INDICES)
)


def stage_c_for_prompt(index):
    """C1, C2, both tables and both figures for one prompt.

    Returns a row dict, or None when the prompt is not comparable (no
    trace-closing tag, an empty phase, or a shape that is not normalized).

    A skipped prompt is NOT plotted and NOT given numbers: a chart drawn from a
    phase that was never located carries the same false label as a number would.
    """

    spec = PROMPT_SPECS[index]
    prompt = spec["prompt"]
    kind = spec["kind"]

    print("\n" + "#" * 64)
    print(f"# PROMPT {index} -- kind: {kind}")
    print("#" * 64)
    print(prompt)

    ift_run = run_and_cache("ift", ift_model, ift_tokenizer, prompt)
    reasoning_run = run_and_cache(
        "reasoning", reasoning_model, reasoning_tokenizer, prompt
    )

    ift_generated = ift_run["generated_ids"]
    reasoning_generated = reasoning_run["generated_ids"]

    # What each model actually produced. Without this the status line says a
    # prompt was cut at the cap but not WHETHER the model was still reasoning
    # or had quietly gone in circles -- and a non-periodic self-repeat inside
    # <think> fires none of the detectors.
    for label, run in (("IFT", ift_run), ("Reasoning", reasoning_run)):

        print(f"\n  {label} generation:")

        for line in generation_summary(run["text"]):
            print("    " + line)

    ift_boundary_ids = turn_boundary_ids(ift_tokenizer)
    reasoning_boundary_ids = turn_boundary_ids(reasoning_tokenizer)

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

    # The loop/stop guards, one line each. A prompt that degenerated or hit the
    # cap says so here rather than in a wall of text nobody reads.
    print("  " + _guard_status(
        "IFT", ift_generated, ift_boundary_ids,
        ift_run["trim_start"], ift_run["trim_period"],
    ))
    print("  " + _guard_status(
        "Reasoning", reasoning_generated, reasoning_boundary_ids,
        reasoning_run["trim_start"], reasoning_run["trim_period"],
    ))

    start, end, labels = locate_user_prompt(ift_tokenizer, prompt)

    path = []
    reasoning_cut = find_trace_end(
        reasoning_generated, reasoning_tokenizer, report=path
    )

    # quiet=True keeps the audit as one line: the decoded text either side of the
    # cut is the part of the full block that has actually ever caught anything.
    verify_phase_cut(
        reasoning_tokenizer, reasoning_generated, reasoning_cut, quiet=True
    )

    if path:
        # Which search path found the boundary was an open question after S4 --
        # the cut was provably right, but which code produced it was invisible.
        print(f"  trace-end found by: {path[0]}")

    if reasoning_cut is None:
        print("  SKIPPED: no trace-closing tag, so there is no answering phase")
        print("  to compare against -- and a fallback to the whole generation")
        print("  would be labelled 'matched phase' while not being matched.")
        return None

    think_tokens = reasoning_cut
    answer_tokens = reasoning_end - reasoning_cut

    if think_tokens < 1 or answer_tokens < 1:
        print(
            f"  SKIPPED: the cut leaves an empty phase --"
            f" thinking {think_tokens}, answering {answer_tokens}."
        )
        return None

    ift_full = _as_numpy(ift_run["attention"])
    reasoning_full = _as_numpy(reasoning_run["attention"])

    if ift_end > ift_full.shape[0] or reasoning_end > reasoning_full.shape[0]:
        print("  SKIPPED: a window is longer than the tensor it slices.")
        return None

    ift_shape = prompt_shape(ift_full, start, end, lo=0, hi=ift_end)
    think_shape = prompt_shape(reasoning_full, start, end, lo=0, hi=reasoning_cut)
    answer_shape = prompt_shape(
        reasoning_full, start, end, lo=reasoning_cut, hi=reasoning_end
    )

    # Checked before any distance is quoted: a shape that does not sum to 1 means
    # the slicing is wrong, and a wrong slice produces plausible-looking numbers.
    for label, shape in (
        ("IFT", ift_shape),
        ("Reasoning-think", think_shape),
        ("Reasoning-answer", answer_shape),
    ):
        total = float(np.abs(shape).sum())

        if abs(total - 1.0) >= 1e-6:
            print(f"  SKIPPED: {label} shape sums to {total:.6f}, not 1.")
            return None

    c2 = total_variation_distance(ift_shape, answer_shape)
    c1 = total_variation_distance(think_shape, answer_shape)

    # Computed once and shared by the table and the chart, so the printed
    # numbers and the bars cannot disagree through a repeated call.
    c2_diff = per_word_diff(ift_shape, answer_shape)
    c1_diff = per_word_diff(think_shape, answer_shape)

    print(
        f"  windows: IFT {ift_end} tok (answering) |"
        f" Reasoning {think_tokens} thinking + {answer_tokens} answering"
    )
    print(f"  C1 (same model, thinking vs answering) = {c1:.4f}")
    print(f"  C2 (IFT vs Reasoning, both answering)  = {c2:.4f}")

    display(pd.DataFrame({
        "token": labels,
        "IFT_answering": ift_shape,
        "Reasoning_answering": answer_shape,
        "C2_answer_minus_IFT": c2_diff,
    }))

    fig_c2 = plot_shape_diff(
        labels,
        c2_diff,
        f"[{index}] {kind} -- C2: Reasoning answering vs IFT answering"
        f"  (TV {c2:.4f})",
        "Reasoning weights it more",
        "IFT weights it more",
    )
    show_figure(fig_c2)

    fig_c1 = plot_shape_diff(
        labels,
        c1_diff,
        f"[{index}] {kind} -- C1: answering vs thinking, same model"
        f"  (TV {c1:.4f})",
        "answering weights it more",
        "thinking weights it more",
    )
    show_figure(fig_c1)

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
    # failure as the S1 broad `except Exception` that hid a NameError for a
    # whole run. ValueError is a type those functions choose; a NameError or a
    # TypeError from a typo would still come through and be seen.
    try:

        row = stage_c_for_prompt(index)

    except ValueError as exc:

        print(f"\n  SKIPPED prompt {index}: {type(exc).__name__}: {exc}")
        print("  (a condition of this prompt, not of the run -- later prompts")
        print("  are unaffected)")

        row = None

    if row is not None:
        stage_c_rows.append(row)


print("\n" + "=" * 64)
print("C1 / C2 ACROSS PROMPT TYPES")
print("=" * 64)

if stage_c_rows:

    display(pd.DataFrame(stage_c_rows).set_index("prompt"))

    print("\nC1 = same model, thinking vs answering -- the phase yardstick.")
    print("C2 = IFT vs Reasoning, both answering -- the model comparison.")
    print("Read C2 only against its own row's C1: if C2 is not clearly smaller")
    print("than C1, the model gap is not separable from the phase effect.")

else:

    print("No prompt produced a comparable pair. Every prompt skipped; each")
    print("SKIPPED line above names its own reason.")

skipped = [
    index for index in NEW_PROMPT_INDICES
    if not any(row["prompt"] == index for row in stage_c_rows)
]

if skipped:
    print(
        "\nNOTE: prompt(s) " + ", ".join(str(i) for i in skipped)
        + " produced no number. Reported rather than hidden -- a skip is a"
        " result about that prompt, not a failure of the run."
    )

