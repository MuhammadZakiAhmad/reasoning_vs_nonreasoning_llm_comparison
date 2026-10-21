"""
CACHE-ONLY SOLIDIFICATION of the C1 / C2 attention-shape measurement.

WHAT THIS IS
------------
The sweep (attention_visualization_for_non_reasoning_vs_reasoning_llms.py)
prints, per prompt, two numbers:

    C1 = Reasoning THINKING vs Reasoning ANSWERING  (same model, phase varies)
    C2 = IFT ANSWERING vs Reasoning ANSWERING         (model varies, phase fixed)

and a footer that counts how often C1 > C2. That count is a claim about
reality, and right now it has no error bar: nothing says whether a gap of
0.002 is a real difference between two models or the noise of averaging a few
hundred attention rows. "C1 > C2 in 2 of 3" is not a finding until the yardstick
says 0.002 is above the floor.

This script supplies the yardstick, and four other things the sweep cannot
supply from a single number each:

    1. NOISE FLOOR. Split each phase window in half and compare the two
       halves to each other. Two samples of the SAME thing. Any model-to-model
       difference smaller than this is not resolvable by this measurement.

    2. ERROR BARS. Paired bootstrap over layers and over generated tokens, so
       "C1 > C2" comes with a probability, not a vibe.

    3. LENGTH-MATCHED RECOMPUTE. The IFT answer and the reasoning answer are
       different lengths. Truncate both to the shorter length and recompute.
       If C2 barely moves, verbosity is not what is being measured.

    4. SINK MASS, PHASE-MATCHED. The per-layer sink headline was reclassified
       as a MASS question and left unresolved, because every shape excludes
       the sink by construction. Mass is measurable here, per phase: what
       fraction of a head's prompt attention lands on token 0, for the IFT
       answering window, the reasoning thinking window and the reasoning
       answering window -- same phase, same normalisation, no threshold.

    5. PER-HEAD DECOMPOSITION of C2. Is the model gap a few heads, or all of
       them? And are those heads sink heads? If C2 survives dropping the
       sink-dominant heads and zeroing the sink column, it is not a sink
       artifact.

    6. LENGTH-MATCHED ERROR BARS. The unmatched comparison pairs a long
       thinking window with a short answering window, and the length-matched
       block showed that this asymmetry moves C1 by 40%. A significance
       statement computed on the unmatched windows is a statement about
       unequal averaging, not about models. So the bootstrap is also run on
       the matched windows, where both sides carry the same number of
       generated tokens.

    7. SLICE SENSITIVITY. Even at matched length, the phase effect depends on
       WHICH part of the thinking window is averaged: the first 436 thinking
       tokens and the last 436 thinking tokens are different places in the
       reasoning, and they do not sit the same distance from the answer. If
       the spread across slices is as large as the C1-vs-C2 gap, then the
       number is a property of the slice and not quotable as "the phase
       effect".

    8. SHARED READER HEADS. Dropping the sink-dominant heads from each model
       separately leaves the two sides holding DIFFERENT head populations, so
       part of the resulting distance is composition, not shape. This block
       keeps only the heads that are readers in BOTH models, which is the same
       set on both sides, and puts error bars on it.

HOW TO RUN
----------
In the SAME Colab kernel that just ran the sweep, in this order:

    %run solidify.py

DO NOT RESTART THE RUNTIME. The attention tensors live in the .npz cache on
the VM's disk; a restart wipes them and this script has nothing to work on.
(If they are gone, re-run the sweep -- it is deterministic, so the tensors
come back identical, and the ~25 minutes are cheaper than a wrong conclusion.)

No model is loaded here. Only the two tokenizers and the cached tensors, so
this runs in seconds rather than minutes and needs no GPU.

WHY THE CODE IS EXTRACTED, NOT COPIED
-------------------------------------
The window rules -- where a phase starts, where a generation ends, how a shape
is normalised -- are the measurement. A solidification script that
re-implemented them would be measuring something slightly different from the
sweep, and the difference would be invisible in the output. So the functions
are lifted out of the experiment file by AST and executed in a fresh
namespace: not a copy of the measurement, the measurement itself.

The one thing that is NOT extracted is the list of expected C1/C2 values, and
those are stated here as RUN_BASELINES. The script recomputes C1/C2 from the
cache and checks them against those values -- so the script also verifies
that the sweep's numbers are reproducible from the tensors on disk, and it
says so explicitly.
"""

import ast
import hashlib
import os
import tempfile

import numpy as np
import torch


# ============================================================
# THE EXPERIMENT FILE AND THE CODE TO LIFT OUT OF IT
# ============================================================

EXPERIMENT_FILE = "attention_visualization_for_non_reasoning_vs_reasoning_llms.py"

IFT_REPO = "Scale-or-Reason/Qwen2.5-1.5B-ift"
REASONING_REPO = "Scale-or-Reason/Qwen2.5-1.5B-reasoning"

# The functions that define the measurement. Everything the solidification
# computes is built on these, so they are the real ones and not lookalikes.
WANT_FUNCTIONS = (
    "_as_numpy",
    "turn_boundary_ids",
    "find_drift_start",
    "earliest_trim",
    "resolve_attention_trim",
    "find_trace_end",
    "prompt_shape",
    "total_variation_distance",
    "locate_user_prompt",
    "sink_profile",
    "content_mass",
    "keep_reader_heads",
    "cache_path",
)

# Constants the extracted functions read out of their globals. REASONING_END_TAGS
# in particular is a DEFAULT ARGUMENT of find_trace_end, so it has to exist in
# the namespace before that function is defined -- which is why the constants
# are extracted in a first pass and the functions in a second.
WANT_CONSTANTS = (
    "REASONING_END_TAGS",
    "STOP_TOKEN_STRINGS",
    "TURN_BOUNDARY_TOKENS",
    "SINK_INDEX",
    "PROMPT_SPECS",
)

# The C1 / C2 values the sweep printed at the 2048-token cap, so this script
# can check that it reproduces the run from the cached tensors. If these ever
# drift, the drift is in the measurement, and it is better to find it here.
RUN_BASELINES = {
    6: {"kind": "opinion", "C1": 0.0641, "C2": 0.0932,
        "ift": 383, "think": 1028, "answer": 436},
    7: {"kind": "explain-why", "C1": 0.0729, "C2": 0.0560,
        "ift": 503, "think": 845, "answer": 400},
    9: {"kind": "ambiguous", "C1": 0.0680, "C2": 0.0662,
        "ift": 670, "think": 780, "answer": 608},
}

# The prompts the reasoning model never closed </think> on at 2048. Listed so
# the report can say "4 of 7 prompts are not measurable" without the reader
# having to remember which, and so a later run that measures one of them is
# visibly a change rather than a silent addition.
NON_MEASURABLE_AT_2048 = {
    8: "counterfactual -- 2048 tokens, no </think>",
    10: "creative -- 2048 tokens, no </think> (periodic tail from token 80)",
    11: "math -- 2048 tokens, no </think>",
    12: "math-trap -- 2048 tokens, no </think> (periodic tail from token 1975)",
}

# Bootstrap size. 400 replicates puts the 2.5th percentile on a reasonable
# footing without the run taking minutes: every replicate is a small numpy
# op on a precomputed per-layer or per-token table, not a re-read of the
# tensor.
BOOTSTRAP_REPLICATES = 400
BOOTSTRAP_SEED = 20261021


def load_experiment_namespace():
    """Pull the measurement code out of the experiment file WITHOUT running it.

    The experiment file is a script, not a module: importing it would download
    the models, generate, and plot -- the entire sweep. So the wanted
    definitions are extracted by AST and executed in a fresh namespace.

    Two passes on purpose. The constants go first because REASONING_END_TAGS
    is a default argument of find_trace_end: a function whose default names a
    name that does not exist yet fails at DEFINITION time, not at call time,
    and the error message would not say which name was missing.
    """

    if not os.path.exists(EXPERIMENT_FILE):
        raise SystemExit(
            f"{EXPERIMENT_FILE} not found. Run this from the repo root, in the "
            "same directory as the experiment."
        )

    with open(EXPERIMENT_FILE, encoding="utf-8") as handle:
        source = handle.read()

    tree = ast.parse(source, filename=EXPERIMENT_FILE)

    namespace = {
        "np": np,
        "torch": torch,
        "os": os,
        "hashlib": hashlib,
        "__name__": "experiment_extracted",
    }

    found_constants = set()

    for node in tree.body:

        if not isinstance(node, ast.Assign):
            continue

        for target in node.targets:

            if isinstance(target, ast.Name) and target.id in WANT_CONSTANTS:

                exec(
                    compile(
                        ast.Module(body=[node], type_ignores=[]),
                        EXPERIMENT_FILE,
                        "exec",
                    ),
                    namespace,
                )

                found_constants.add(target.id)

    missing_constants = set(WANT_CONSTANTS) - found_constants

    if missing_constants:
        raise SystemExit(
            "could not extract from "
            f"{EXPERIMENT_FILE}: {sorted(missing_constants)}"
        )

    found_functions = set()

    for node in tree.body:

        if isinstance(node, ast.FunctionDef) and node.name in WANT_FUNCTIONS:

            exec(
                compile(
                    ast.Module(body=[node], type_ignores=[]),
                    EXPERIMENT_FILE,
                    "exec",
                ),
                namespace,
            )

            found_functions.add(node.name)

    missing_functions = set(WANT_FUNCTIONS) - found_functions

    if missing_functions:
        raise SystemExit(
            "could not extract from "
            f"{EXPERIMENT_FILE}: {sorted(missing_functions)}"
        )

    return namespace


# ============================================================
# THE CACHE
# ============================================================

def cache_dir_candidates():
    """Where the attention cache might be, most likely first.

    The experiment file picks its cache directory with
    `os.path.isdir("\\content")` -- a backslash path, which is a RELATIVE path
    on Linux, so on Colab that test is False and the cache lands in the
    system temp directory rather than /content. Rather than guess which, this
    script looks in both and uses whichever actually holds the files.

    The cache is per-kernel either way: both directories live on the VM's
    disk, and both die with the VM.
    """

    return [
        "/content/attn_cache",
        os.path.join(tempfile.gettempdir(), "attn_cache"),
        os.path.join(os.getcwd(), "attn_cache"),
    ]


def resolve_cache_dir(namespace):
    """The first candidate directory that actually contains .npz files."""

    found = []

    for candidate in cache_dir_candidates():

        if not os.path.isdir(candidate):
            continue

        npz_files = [name for name in os.listdir(candidate) if name.endswith(".npz")]

        if npz_files:
            found.append((candidate, len(npz_files)))

    if not found:
        raise SystemExit(
            "no attention cache found. Looked in:\n  "
            + "\n  ".join(cache_dir_candidates())
            + "\nThe cache is written by the sweep and lives on the VM's disk. "
            "If the runtime was restarted, it is gone: re-run the sweep in "
            "this kernel, then run this again."
        )

    for candidate, count in found:
        print(f"   cache: {candidate}  ({count} .npz files)")

    return found[0][0]


def load_run(namespace, which, prompt, cap, cache_dir):
    """One cached run, or None when that file does not exist.

    The cap is part of the cache FILENAME, so the 1024 and 2048 sets coexist
    and can be compared directly.
    """

    namespace["CACHE_DIR"] = cache_dir
    namespace["MAX_NEW_TOKENS"] = cap

    path = namespace["cache_path"](which, prompt)

    if not os.path.exists(path):
        return None

    with np.load(path, allow_pickle=False) as cached:

        trim_start = int(cached["trim_start"])
        trim_period = int(cached["trim_period"])

        return {
            "path": path,
            "generated_ids": cached["generated_ids"],
            "text": str(cached["text"]),
            "attention": cached["attention"],
            "trim_start": None if trim_start < 0 else trim_start,
            "trim_period": None if trim_period < 0 else trim_period,
        }


def rebuild_windows(namespace, run, tokenizer, label):
    """Re-derive the windows the sweep used, from the cached run alone.

    Same two steps the sweep takes: the drift cut comes from the generated
    token ids at READ time, and the periodic-tail cut comes from the value
    stored in the cache. resolve_attention_trim() combines them, and it is
    the same function the sweep called, so the two cannot disagree.
    """

    boundary_ids = namespace["turn_boundary_ids"](tokenizer)

    end = namespace["resolve_attention_trim"](
        label,
        run["generated_ids"],
        boundary_ids,
        run["trim_start"],
        run["trim_period"],
        quiet=True,
    )

    return end, boundary_ids


# ============================================================
# THE METRICS
# ============================================================

def normalize(vector):
    """Renormalize a non-negative vector to sum to 1, or None if it is empty."""

    total = float(vector.sum())

    if total <= 0:
        return None

    return vector / total


def split_half_floor(namespace, attention, start, end, lo, hi):
    """How different do two HALVES of the same phase look?

    The sharpest available null for this measurement. Both halves describe the
    same model, the same phase and the same prompt; the only difference is
    which generated tokens went into the average. A model-to-model distance
    below this is not a difference the measurement can see.

    Returns None for a window too short to split.
    """

    length = hi - lo

    if length < 8:
        return None

    middle = lo + length // 2

    first = namespace["prompt_shape"](attention, start, end, lo, middle)
    second = namespace["prompt_shape"](attention, start, end, middle, hi)

    return float(namespace["total_variation_distance"](first, second))


def per_layer_table(attention, lo, hi):
    """Mean prompt attention per layer: [layers, prompt_tokens].

    prompt_shape() averages over generated tokens, layers AND heads and then
    normalizes. Averages commute, so the same number can be rebuilt from a
    per-layer table by averaging the selected layers and normalizing once.
    That is what makes a layer bootstrap cheap: the resample touches a
    [layers, prompt_tokens] table instead of a 60 MB tensor slice.
    """

    return attention[lo:hi].mean(axis=(0, 2))


def per_token_table(attention, lo, hi):
    """Mean prompt attention per generated token: [tokens, prompt_tokens].

    The same trick on the token axis, for the token bootstrap.
    """

    return attention[lo:hi].mean(axis=(1, 2))


def per_head_table(attention, lo, hi):
    """Mean prompt attention per (layer, head): [layers, heads, prompt_tokens].

    Needed for the reader-head work, where the head set differs from the one
    the layer and token tables average over. Averages commute, so this is the
    same number the per-layer table is built from, just with the head axis
    left in.
    """

    return attention[lo:hi].mean(axis=0)


def fixed_length_slices(window_length, slice_length):
    """Every place a slice of `slice_length` can sit inside a window.

    Returns [(label, lo, hi)]. A window no longer than the slice has only one
    possible slice and is reported as "full"; a longer one has three -- the
    start, the middle and the end -- because the point of this helper is to
    find out whether the answer depends on which part of a long window gets
    averaged.
    """

    if window_length <= slice_length:
        return [("full", 0, window_length)]

    options = [
        ("early", 0, slice_length),
        (
            "middle",
            (window_length - slice_length) // 2,
            (window_length - slice_length) // 2 + slice_length,
        ),
        ("late", window_length - slice_length, window_length),
    ]

    unique = []
    seen = set()

    for label, lo, hi in options:

        if (lo, hi) in seen:
            continue

        seen.add((lo, hi))
        unique.append((label, lo, hi))

    return unique


def shape_from_masked_heads(table, mask, start, end):
    """The prompt shape built from a chosen set of (layer, head) pairs.

    table: [layers, heads, prompt_tokens] -- per_head_table()
    mask:  [layers, heads] -- True for the heads to keep

    The sink needs no special handling here: it sits at prompt index 0,
    outside the user-token slice, so slicing [start:end] removes it. That is
    the same reason prompt_shape() does not zero it either.
    """

    vector = (table * mask[..., None]).sum(axis=(0, 1))

    return normalize(vector[start:end])


def shared_reader_mask(namespace, ift_window, reasoning_window, threshold):
    """The heads that are READERS IN BOTH models, at one threshold.

    keep_reader_heads() works out "share < threshold" from the sink profile,
    so a head is a reader on one side and not the other often enough to
    matter. Keeping each model's own readers leaves the two sides holding
    different populations, and part of the distance between them is then
    composition rather than shape. This returns the INTERSECTION, which is
    the same set of heads on both sides.

    The rule is the extracted function's own rule -- share < threshold -- so
    the mask here and the mask inside keep_reader_heads() agree by
    construction, not by two pieces of code happening to match.
    """

    share_ift, _ = namespace["sink_profile"](ift_window)
    share_reasoning, _ = namespace["sink_profile"](reasoning_window)

    keep = (share_ift < threshold) & (share_reasoning < threshold)

    return keep, share_ift, share_reasoning


def retained_with_mask(namespace, window, mask, threshold):
    """How much non-sink attention the masked heads hold, per model.

    keep_reader_heads() computes this from the share array it is given, so
    it is handed a synthetic one: 0.0 on the heads to keep, 1.0 on the heads
    to drop. Its own rule is `share < threshold`, and 1.0 is not below any
    threshold used here, so the mask it derives is exactly the mask passed
    in. The retained fraction is then computed by the extracted code rather
    than restated here.
    """

    synthetic_share = np.where(mask, 0.0, 1.0)

    vector, retained, kept, total = namespace["keep_reader_heads"](
        window, synthetic_share, threshold
    )

    return retained, kept, total


def bootstrap_layers(namespace, tables, start, end, replicates, seed):
    """Paired bootstrap over LAYERS, for the full windows AND the matched ones.

    Paired means the same resampled layer indices are used for every shape in
    a replicate -- the IFT shape, the thinking shape and the answering shape.
    That is the honest pairing: a replicate asks "if the model had put its
    attention in a slightly different set of layers, how much would C1 and C2
    move?" Resampling each model independently would answer a different
    question, and would inflate the spread of the GAP.

    tables: a dict of per-layer tables, keyed by role:
      think_full / answer_full  -- the unmatched comparison
      think_m1  / answer_m1     -- C1 on length-matched windows
      ift_full  / ift_m2        -- the IFT side, matched and unmatched
      answer_m2                 -- the reasoning answering side, matched
    All three of the original roles are needed: C1 compares THINKING with
    ANSWERING inside one model, so a bootstrap that only had the answering
    table would compare the answering phase with itself and report C1 = 0.

    The layer selection is drawn ONCE per replicate and reused for every
    table, which is what makes the matched and unmatched numbers comparable
    with each other AND with the previous run of this script.
    """

    comparisons = (
        ("C1", "think_full", "answer_full"),
        ("C2", "ift_full", "answer_full"),
        ("C1_matched", "think_m1", "answer_m1"),
        ("C2_matched", "ift_m2", "answer_m2"),
    )

    layer_count = tables["ift_full"].shape[0]

    rng = np.random.default_rng(seed)

    results = {name: [] for name, _, _ in comparisons}

    for _ in range(replicates):

        selection = rng.integers(0, layer_count, layer_count)

        shapes = {}
        broken = False

        for key, table in tables.items():

            shape = normalize(table[selection].mean(axis=0)[start:end])

            if shape is None:
                broken = True
                break

            shapes[key] = shape

        if broken:
            continue

        for name, left_key, right_key in comparisons:

            results[name].append(
                float(
                    namespace["total_variation_distance"](
                        shapes[left_key], shapes[right_key]
                    )
                )
            )

    return {name: np.array(values) for name, values in results.items()}


def bootstrap_tokens(namespace, tables, start, end, replicates, seed):
    """Bootstrap over GENERATED TOKENS, the other axis of the same question.

    The layer bootstrap asks how much the answer depends on WHICH LAYERS
    carried the attention. This one asks how much it depends on WHICH
    GENERATED TOKENS were averaged in.

    The axis is NOT paired the way the layer axis is, and the difference
    matters: the three windows are different CONTENT, so there is no honest
    way to say that token 40 of the thinking and token 40 of the answering
    are "the same draw". Each window is therefore resampled on its own, which
    makes the spread WIDER than a paired test would, not narrower. The
    probability printed from this axis is a Monte Carlo over independent
    resamples; the layer axis is the one that supports a paired claim.

    The draw order follows the table dict, and the first three entries are the
    original three windows, so the unmatched C1 and C2 numbers are unchanged
    from the previous run of this script.
    """

    comparisons = (
        ("C1", "think_full", "answer_full"),
        ("C2", "ift_full", "answer_full"),
        ("C1_matched", "think_m1", "answer_m1"),
        ("C2_matched", "ift_m2", "answer_m2"),
    )

    rng = np.random.default_rng(seed)

    results = {name: [] for name, _, _ in comparisons}

    for _ in range(replicates):

        shapes = {}
        broken = False

        for key, table in tables.items():

            picks = rng.integers(0, table.shape[0], table.shape[0])

            shape = normalize(table[picks].mean(axis=0)[start:end])

            if shape is None:
                broken = True
                break

            shapes[key] = shape

        if broken:
            continue

        for name, left_key, right_key in comparisons:

            results[name].append(
                float(
                    namespace["total_variation_distance"](
                        shapes[left_key], shapes[right_key]
                    )
                )
            )

    return {name: np.array(values) for name, values in results.items()}


def bootstrap_reader_layers(namespace, ift_table, reasoning_table, mask,
                            start, end, replicates, seed):
    """Layer bootstrap for the SHARED-reader-heads C2, and the all-heads C2
    computed on the same layer draws.

    Two numbers come out of the same replicate, so they can be compared
    directly: the C2 over the shared reader heads, and the C2 over all heads.
    The head set is fixed at its full-window value rather than recomputed per
    replicate, so the interval reflects layer sampling only -- the mask
    itself is treated as known.
    """

    layer_count = ift_table.shape[0]

    rng = np.random.default_rng(seed)

    reader_values = []
    full_values = []

    for _ in range(replicates):

        selection = rng.integers(0, layer_count, layer_count)

        ift_selected = ift_table[selection]
        reasoning_selected = reasoning_table[selection]
        keep_selected = mask[selection]

        left_reader = shape_from_masked_heads(
            ift_selected, keep_selected, start, end
        )
        right_reader = shape_from_masked_heads(
            reasoning_selected, keep_selected, start, end
        )

        left_full = normalize(ift_selected.sum(axis=(0, 1))[start:end])
        right_full = normalize(reasoning_selected.sum(axis=(0, 1))[start:end])

        if (left_reader is None or right_reader is None
                or left_full is None or right_full is None):
            continue

        reader_values.append(
            float(namespace["total_variation_distance"](left_reader, right_reader))
        )
        full_values.append(
            float(namespace["total_variation_distance"](left_full, right_full))
        )

    return np.array(reader_values), np.array(full_values)


def summarise(values, label):
    """Median and a 95% interval, in one line's worth of numbers."""

    if len(values) == 0:
        return f"      {label}: no replicates"

    low, median, high = np.percentile(values, [2.5, 50.0, 97.5])

    return (
        f"      {label}: median {median:.4f}   "
        f"95% CI [{low:.4f}, {high:.4f}]"
    )


def sink_report(namespace, attention, lo, hi, start, end):
    """How much of a head's prompt attention lands on the sink, per layer.

    The sink is prompt index 0 -- '<|im_start|>' -- and it sits OUTSIDE the
    user-token slice, so no shape in this project can contain it. That is
    correct for a shape, and it is also why the sink headline stayed
    unresolved: it is a statement about MASS, and mass was never measured in
    a phase-matched way.

    This is that measurement. share = the fraction of a head's prompt
    attention sitting on token 0, which is a ratio and therefore does not
    move when the amount of prompt attention moves -- the failure mode that
    retired the raw mass numbers.
    """

    window = attention[lo:hi]

    share, mass = namespace["sink_profile"](window)

    per_layer_share = share.mean(axis=1)

    return {
        "share": share,
        "mass": mass,
        "per_layer_share": per_layer_share,
        "per_layer_sink_heads": (share > 0.5).sum(axis=1),
        "mean_share": float(share.mean()),
        "sink_head_fraction": float((share > 0.5).mean()),
    }


def reader_shape(namespace, attention, lo, hi, start, end, threshold=0.5):
    """The answering shape with the sink column zeroed and the sink-dominant
    heads dropped.

    keep_reader_heads() does the work; this normalizes what it returns into
    a shape over the user tokens. The retained-content fraction it reports
    alongside is the honest caveat: if the surviving heads hold only a small
    share of the non-sink attention, then the comparison built on them is a
    comparison of a small share of the signal, and has to be reported that
    way.
    """

    window = attention[lo:hi]

    share, _ = namespace["sink_profile"](window)

    vector, retained, kept, total = namespace["keep_reader_heads"](
        window, share, threshold
    )

    shape = normalize(vector[start:end])

    return shape, retained, kept, total


def per_head_decomposition(namespace, ift_att, reasoning_att, start, end,
                           ift_lo, ift_hi, r_lo, r_hi):
    """C2 for every (layer, head) pair, not just for the average of all 336.

    Answers two questions at once: is the model gap carried by a handful of
    heads or spread evenly, and are those heads sink heads. A gap that lives
    entirely in sink-dominant heads is a statement about the sink, not about
    the two models reading the question differently.
    """

    ift_heads = ift_att[ift_lo:ift_hi].mean(axis=0)
    reasoning_heads = reasoning_att[r_lo:r_hi].mean(axis=0)

    distances = np.zeros(ift_heads.shape[:2], dtype=np.float64)

    for layer in range(ift_heads.shape[0]):

        for head in range(ift_heads.shape[1]):

            left = normalize(ift_heads[layer, head][start:end])
            right = normalize(reasoning_heads[layer, head][start:end])

            if left is None or right is None:
                continue

            distances[layer, head] = float(
                namespace["total_variation_distance"](left, right)
            )

    return distances


# ============================================================
# THE REPORT
# ============================================================

def rule(char="=", width=74):
    print(char * width)


def print_slice_grid(title, row_name, rows, columns, values):
    """A small grid of numbers, with a spread row underneath.

    values: dict[(row_label, column_label)] -> float
    """

    print(f"      {title}")
    print(f"      {row_name:<10}" + "".join(f"{col:>13}" for col in columns))

    for row in rows:

        line = f"      {row:<10}"

        for col in columns:

            value = values.get((row, col))

            line += f"{value:>13.4f}" if value is not None else f"{'-':>13}"

        print(line)

    line = f"      {'spread':<10}"

    for col in columns:

        column_values = [
            values[(row, col)] for row in rows if (row, col) in values
        ]

        if column_values:
            line += f"{max(column_values) - min(column_values):>13.4f}"
        else:
            line += f"{'-':>13}"

    print(line)


def report_prompt(namespace, index, cap, tokenizers, cache_dir):
    """Everything the cache knows about one prompt, printed as one block.

    Returns a row dict for the summary table, or None when the prompt is not
    measurable (the reasoning model never closed </think>, so there is no
    answering phase to compare against).
    """

    spec = namespace["PROMPT_SPECS"][index]

    prompt = spec["prompt"]
    kind = spec.get("kind", "?")

    ift_run = load_run(namespace, "ift", prompt, cap, cache_dir)
    reasoning_run = load_run(namespace, "reasoning", prompt, cap, cache_dir)

    if ift_run is None or reasoning_run is None:
        missing = "ift" if ift_run is None else "reasoning"

        print()
        print(
            f"      NOT IN CACHE -- no {missing} run at cap {cap}. The cache is "
            "per-kernel: if the runtime was restarted it is gone."
        )

        return {
            "prompt": index,
            "kind": kind,
            "cap": cap,
            "measurable": False,
            "reason": f"no {missing} run cached at cap {cap}",
        }

    ift_tokenizer, reasoning_tokenizer = tokenizers

    ift_end, _ = rebuild_windows(
        namespace, ift_run, ift_tokenizer, "IFT"
    )
    reasoning_end, _ = rebuild_windows(
        namespace, reasoning_run, reasoning_tokenizer, "Reasoning"
    )

    reasoning_cut = namespace["find_trace_end"](
        reasoning_run["generated_ids"], reasoning_tokenizer
    )

    print()
    rule()
    print(f"PROMPT {index}   |   kind: {kind}   |   cap: {cap}")
    rule()
    print(prompt)
    print()
    print(
        f"      IFT generated {len(ift_run['generated_ids'])} tokens, "
        f"answering window {ift_end}"
    )
    print(
        f"      Reasoning generated {len(reasoning_run['generated_ids'])} tokens, "
        f"window {reasoning_end}"
    )

    if reasoning_cut is None:
        print()
        print("      NOT MEASURABLE -- the reasoning generation never closed")
        print("      </think>, so there is no answering phase to compare.")
        print("      Nothing below applies to this prompt, by design.")

        return {
            "prompt": index,
            "kind": kind,
            "cap": cap,
            "measurable": False,
            "reason": "the reasoning generation never closed </think>",
        }

    think_tokens = reasoning_cut
    answer_tokens = reasoning_end - reasoning_cut

    if think_tokens < 1 or answer_tokens < 1:
        print()
        print(
            f"      NOT MEASURABLE -- the cut leaves an empty phase "
            f"(thinking {think_tokens}, answering {answer_tokens})."
        )

        return {
            "prompt": index,
            "kind": kind,
            "cap": cap,
            "measurable": False,
            "reason": (
                f"the cut leaves an empty phase "
                f"(thinking {think_tokens}, answering {answer_tokens})"
            ),
        }

    start, end, labels = namespace["locate_user_prompt"](ift_tokenizer, prompt)

    ift_att = namespace["_as_numpy"](ift_run["attention"])
    reasoning_att = namespace["_as_numpy"](reasoning_run["attention"])

    ift_shape = namespace["prompt_shape"](ift_att, start, end, lo=0, hi=ift_end)
    think_shape = namespace["prompt_shape"](
        reasoning_att, start, end, lo=0, hi=reasoning_cut
    )
    answer_shape = namespace["prompt_shape"](
        reasoning_att, start, end, lo=reasoning_cut, hi=reasoning_end
    )

    c1 = float(namespace["total_variation_distance"](think_shape, answer_shape))
    c2 = float(namespace["total_variation_distance"](ift_shape, answer_shape))

    print(
        f"      windows: IFT answering {ift_end} | "
        f"Reasoning thinking {think_tokens} + answering {answer_tokens}"
    )

    # The two matching lengths. C1 compares thinking with answering, so it
    # matches on the shorter of those two; C2 compares the two answering
    # windows, so it matches on the shorter of those. They are usually
    # DIFFERENT numbers, which is why the two comparisons need separate
    # tables rather than one shared "matched" table.
    matched_c1 = min(think_tokens, answer_tokens)
    matched = min(ift_end, answer_tokens)

    # --- the run cross-check -------------------------------------------------
    # Only at 2048: the 1024 numbers are a DIFFERENT measurement (truncated
    # answering windows), so comparing them against the 2048 baselines would
    # report a mismatch that is not one.
    baseline = RUN_BASELINES.get(index) if cap == 2048 else None

    if baseline:

        c1_ok = abs(round(c1, 4) - baseline["C1"]) < 1e-9
        c2_ok = abs(round(c2, 4) - baseline["C2"]) < 1e-9

        print(
            f"      run cross-check: C1 {c1:.4f} (run {baseline['C1']:.4f}) "
            f"{'OK' if c1_ok else 'MISMATCH'}   "
            f"C2 {c2:.4f} (run {baseline['C2']:.4f}) "
            f"{'OK' if c2_ok else 'MISMATCH'}"
        )

        if not (c1_ok and c2_ok):
            print(
                "      !! The cache does not reproduce the sweep's numbers. "
                "Stop and find out why before reading anything below."
            )

    # --- 1. the noise floor --------------------------------------------------
    floor_ift = split_half_floor(
        namespace, ift_att, start, end, 0, ift_end
    )
    floor_think = split_half_floor(
        namespace, reasoning_att, start, end, 0, reasoning_cut
    )
    floor_answer = split_half_floor(
        namespace, reasoning_att, start, end, reasoning_cut, reasoning_end
    )

    floor_c2 = max(v for v in (floor_ift, floor_answer) if v is not None)
    floor_c1 = max(v for v in (floor_think, floor_answer) if v is not None)

    print()
    print("   --- NOISE FLOOR: two halves of the SAME phase -------------------")
    print(
        f"      IFT answering, halves differ by: "
        f"{floor_ift:.4f}" if floor_ift is not None else "      IFT answering: too short"
    )
    print(
        f"      Reasoning thinking, halves differ by: {floor_think:.4f}"
        if floor_think is not None
        else "      Reasoning thinking: too short"
    )
    print(
        f"      Reasoning answering, halves differ by: {floor_answer:.4f}"
        if floor_answer is not None
        else "      Reasoning answering: too short"
    )
    print(f"      floor for C2 (the larger answering floor): {floor_c2:.4f}")
    print(f"      floor for C1 (the larger of thinking/answering): {floor_c1:.4f}")
    print(
        f"      C1 {c1:.4f} = {c1 / floor_c1:.2f}x its floor   |   "
        f"C2 {c2:.4f} = {c2 / floor_c2:.2f}x its floor"
    )

    # --- 2. error bars --------------------------------------------------------
    # The dict order is deliberate: the first three entries are the three
    # original windows, so the unmatched C1 and C2 come out of the bootstrap
    # stream exactly as they did before the matched tables were added.
    layer_tables = {
        "ift_full": per_layer_table(ift_att, 0, ift_end),
        "think_full": per_layer_table(reasoning_att, 0, reasoning_cut),
        "answer_full": per_layer_table(
            reasoning_att, reasoning_cut, reasoning_end
        ),
        "ift_m2": per_layer_table(ift_att, 0, matched),
        "think_m1": per_layer_table(reasoning_att, 0, matched_c1),
        "answer_m1": per_layer_table(
            reasoning_att, reasoning_cut, reasoning_cut + matched_c1
        ),
        "answer_m2": per_layer_table(
            reasoning_att, reasoning_cut, reasoning_cut + matched
        ),
    }

    layer_boot = bootstrap_layers(
        namespace,
        layer_tables,
        start,
        end,
        BOOTSTRAP_REPLICATES,
        BOOTSTRAP_SEED,
    )

    token_tables = {
        "ift_full": per_token_table(ift_att, 0, ift_end),
        "think_full": per_token_table(reasoning_att, 0, reasoning_cut),
        "answer_full": per_token_table(
            reasoning_att, reasoning_cut, reasoning_end
        ),
        "ift_m2": per_token_table(ift_att, 0, matched),
        "think_m1": per_token_table(reasoning_att, 0, matched_c1),
        "answer_m1": per_token_table(
            reasoning_att, reasoning_cut, reasoning_cut + matched_c1
        ),
        "answer_m2": per_token_table(
            reasoning_att, reasoning_cut, reasoning_cut + matched
        ),
    }

    token_boot = bootstrap_tokens(
        namespace,
        token_tables,
        start,
        end,
        BOOTSTRAP_REPLICATES,
        BOOTSTRAP_SEED,
    )

    c1_layers = layer_boot["C1"]
    c2_layers = layer_boot["C2"]
    c1_tokens = token_boot["C1"]
    c2_tokens = token_boot["C2"]

    print()
    print(f"   --- BOOTSTRAP over LAYERS ({BOOTSTRAP_REPLICATES} replicates) ---")
    print(summarise(c1_layers, "C1"))
    print(summarise(c2_layers, "C2"))

    if len(c1_layers) and len(c2_layers):
        p_phase_larger = float((c1_layers > c2_layers).mean())
        print(
            f"      P(C1 > C2) in a replicate: {p_phase_larger:.3f}"
            f"   ({p_phase_larger:.1%} of replicates have the phase effect larger)"
        )

    print()
    print(f"   --- BOOTSTRAP over GENERATED TOKENS ({BOOTSTRAP_REPLICATES} reps) ---")
    print(summarise(c1_tokens, "C1"))
    print(summarise(c2_tokens, "C2"))

    if len(c1_tokens) and len(c2_tokens):
        p_phase_larger = float((c1_tokens > c2_tokens).mean())
        print(f"      P(C1 > C2) in a replicate: {p_phase_larger:.3f}")
        print(
            "      note: this axis is NOT paired -- the three windows are "
            "different content, so each is resampled on its own"
        )

    # --- 3. length-matched recompute -----------------------------------------
    ift_matched = namespace["prompt_shape"](
        ift_att, start, end, lo=0, hi=matched
    )
    answer_matched = namespace["prompt_shape"](
        reasoning_att, start, end, lo=reasoning_cut, hi=reasoning_cut + matched
    )

    c2_matched = float(
        namespace["total_variation_distance"](ift_matched, answer_matched)
    )

    think_matched = namespace["prompt_shape"](
        reasoning_att, start, end, lo=0, hi=matched_c1
    )
    answer_matched_c1 = namespace["prompt_shape"](
        reasoning_att, start, end, lo=reasoning_cut, hi=reasoning_cut + matched_c1
    )

    c1_matched = float(
        namespace["total_variation_distance"](think_matched, answer_matched_c1)
    )

    print()
    print("   --- LENGTH-MATCHED RECOMPUTE ------------------------------------")
    print(
        f"      C2 at matched length ({matched} tokens both sides): "
        f"{c2_matched:.4f}   (unmatched {c2:.4f}, "
        f"change {c2_matched - c2:+.4f})"
    )
    print(
        f"      C1 at matched length ({matched_c1} tokens both sides): "
        f"{c1_matched:.4f}   (unmatched {c1:.4f}, "
        f"change {c1_matched - c1:+.4f})"
    )

    # --- 3a. error bars on the matched windows --------------------------------
    # The unmatched comparison pairs a long thinking window with a short
    # answering window, and the block above shows that this asymmetry moves C1
    # by tens of percent. A significance statement computed on the unmatched
    # windows is therefore a statement about unequal averaging, not about
    # models. This is the same bootstrap, run on windows that carry the same
    # number of generated tokens on both sides.
    print()
    print(
        f"   --- BOOTSTRAP ON THE LENGTH-MATCHED WINDOWS "
        f"({BOOTSTRAP_REPLICATES} replicates) ---"
    )
    print("      both sides now carry the same number of generated tokens")

    matched_boot = {}

    for axis, values in (("layers", layer_boot), ("tokens", token_boot)):

        c1m = values["C1_matched"]
        c2m = values["C2_matched"]

        matched_boot[axis] = (c1m, c2m)

        print(f"      {axis}:")
        print(summarise(c1m, "   C1 matched"))
        print(summarise(c2m, "   C2 matched"))

        if len(c1m) and len(c2m):

            p_matched = float((c1m > c2m).mean())

            print(f"      P(C1m > C2m): {p_matched:.3f}")

    # --- 3b. slice sensitivity ------------------------------------------------
    # Even at matched length, WHICH part of the thinking window is averaged
    # moves the number. The first 436 thinking tokens and the last 436
    # thinking tokens are different places in the reasoning. If the spread
    # across slices is as large as the gap between C1 and C2, then the number
    # is a property of the slice, not a quotable "phase effect".
    print()
    print("   --- SLICE SENSITIVITY: does the number depend on WHICH part of")
    print("       the window gets averaged? ---")

    think_slices = fixed_length_slices(think_tokens, matched_c1)
    answer_slices = fixed_length_slices(answer_tokens, matched_c1)

    c1_slice_values = {}

    for think_label, think_lo, think_hi in think_slices:

        for answer_label, answer_lo, answer_hi in answer_slices:

            think_slice_shape = namespace["prompt_shape"](
                reasoning_att, start, end, lo=think_lo, hi=think_hi
            )
            answer_slice_shape = namespace["prompt_shape"](
                reasoning_att, start, end,
                lo=reasoning_cut + answer_lo,
                hi=reasoning_cut + answer_hi,
            )

            c1_slice_values[(think_label, answer_label)] = float(
                namespace["total_variation_distance"](
                    think_slice_shape, answer_slice_shape
                )
            )

    print_slice_grid(
        "C1 by slice of the THINKING window (columns: slice of the answering window)",
        "thinking",
        [label for label, _, _ in think_slices],
        [label for label, _, _ in answer_slices],
        c1_slice_values,
    )

    ift_slices = fixed_length_slices(ift_end, matched)
    reasoning_answer_slices = fixed_length_slices(answer_tokens, matched)

    c2_slice_values = {}

    for ift_label, ift_lo, ift_hi in ift_slices:

        for answer_label, answer_lo, answer_hi in reasoning_answer_slices:

            ift_slice_shape = namespace["prompt_shape"](
                ift_att, start, end, lo=ift_lo, hi=ift_hi
            )
            reasoning_slice_shape = namespace["prompt_shape"](
                reasoning_att, start, end,
                lo=reasoning_cut + answer_lo,
                hi=reasoning_cut + answer_hi,
            )

            c2_slice_values[(ift_label, answer_label)] = float(
                namespace["total_variation_distance"](
                    ift_slice_shape, reasoning_slice_shape
                )
            )

    print_slice_grid(
        "C2 by slice (rows: slice of the IFT answer, columns: slice of the reasoning answer)",
        "IFT",
        [label for label, _, _ in ift_slices],
        [label for label, _, _ in reasoning_answer_slices],
        c2_slice_values,
    )

    c1_spread = (
        max(c1_slice_values.values()) - min(c1_slice_values.values())
        if c1_slice_values else 0.0
    )
    c2_spread = (
        max(c2_slice_values.values()) - min(c2_slice_values.values())
        if c2_slice_values else 0.0
    )
    matched_gap = c1_matched - c2_matched

    print()
    print(
        f"      the C1-vs-C2 gap at matched length is {matched_gap:+.4f}; "
        f"moving the slice moves C1 by {c1_spread:.4f} and C2 by {c2_spread:.4f}"
    )

    if max(c1_spread, c2_spread) > abs(matched_gap):
        print(
            "      !! the slice choice moves the number MORE than the gap does, "
            "so the gap does not survive re-cutting the window: it is a "
            "property of where the cut fell, not of the phase or the model"
        )
    else:
        print(
            "      the slice choice moves the numbers less than the gap does, "
            "so the gap survives being re-cut"
        )

    # --- 3c. shared reader heads ----------------------------------------------
    # Dropping the sink-dominant heads from each model SEPARATELY leaves the
    # two sides holding different head populations, so part of the distance
    # between them is composition rather than shape. This block keeps only
    # the heads that are readers in BOTH models: the same set on both sides.
    print()
    print("   --- C2 OVER SHARED READER HEADS (a reader in BOTH models) ---")
    print("       the same head set on both sides, so this compares two SHAPES")
    print("       and not two head populations")

    ift_window = ift_att[0:matched]
    reasoning_window = reasoning_att[reasoning_cut:reasoning_cut + matched]

    ift_head_table = per_head_table(ift_att, 0, matched)
    reasoning_head_table = per_head_table(
        reasoning_att, reasoning_cut, reasoning_cut + matched
    )

    shared_sweep = {}
    shared_masks = {}

    for threshold in (0.3, 0.5, 0.7, 0.9):

        keep, _, _ = shared_reader_mask(
            namespace, ift_window, reasoning_window, threshold
        )

        if not keep.any():

            print(
                f"      threshold {threshold:.1f}: no head is a reader in both "
                "models, so there is nothing to compare"
            )
            shared_sweep[threshold] = None
            continue

        left = shape_from_masked_heads(ift_head_table, keep, start, end)
        right = shape_from_masked_heads(reasoning_head_table, keep, start, end)

        if left is None or right is None:

            print(f"      threshold {threshold:.1f}: no shape survives")
            shared_sweep[threshold] = None
            continue

        distance = float(namespace["total_variation_distance"](left, right))

        retained_ift, kept, total = retained_with_mask(
            namespace, ift_window, keep, threshold
        )
        retained_reasoning, _, _ = retained_with_mask(
            namespace, reasoning_window, keep, threshold
        )

        shared_sweep[threshold] = distance
        shared_masks[threshold] = keep

        print(
            f"      threshold {threshold:.1f}: C2m {c2_matched:.4f} -> "
            f"{distance:.4f} ({distance - c2_matched:+.4f})   "
            f"shared heads {kept}/{total}   holding IFT {retained_ift:.1%} / "
            f"Reasoning {retained_reasoning:.1%} of the non-sink attention"
        )

    # Error bars on the shared-reader C2, from the same layer draws as the
    # all-heads C2, so the two can be compared inside each replicate.
    if 0.5 in shared_masks:

        reader_boot, full_boot = bootstrap_reader_layers(
            namespace,
            ift_head_table,
            reasoning_head_table,
            shared_masks[0.5],
            start,
            end,
            BOOTSTRAP_REPLICATES,
            BOOTSTRAP_SEED,
        )

        print()
        print(
            f"   --- BOOTSTRAP over LAYERS on the SHARED reader heads "
            f"(threshold 0.5, {BOOTSTRAP_REPLICATES} reps) ---"
        )
        print(summarise(reader_boot, "C2 shared-reader"))
        print(summarise(full_boot, "C2 all-heads (matched)"))

        if len(reader_boot) and len(full_boot):

            p_reader = float((reader_boot > full_boot).mean())

            print(
                f"      P(shared-reader C2 > all-heads C2): {p_reader:.3f}   "
                f"(the gap is {'bigger' if p_reader > 0.5 else 'not bigger'} "
                "on the reader heads)"
            )
    else:
        reader_boot = np.array([])
        full_boot = np.array([])

    # --- 4. the sink, phase-matched -------------------------------------------
    sink_ift = sink_report(
        namespace, ift_att, 0, ift_end, start, end
    )
    sink_think = sink_report(
        namespace, reasoning_att, 0, reasoning_cut, start, end
    )
    sink_answer = sink_report(
        namespace, reasoning_att, reasoning_cut, reasoning_end, start, end
    )

    print()
    print("   --- SINK MASS, PHASE-MATCHED (share of prompt attention on token 0)")
    print("      layer   IFT-answer   R-thinking   R-answer   sink heads (R-answer)")
    for layer in range(len(sink_ift["per_layer_share"])):
        print(
            f"      {layer:5d}   {sink_ift['per_layer_share'][layer]:10.3f}   "
            f"{sink_think['per_layer_share'][layer]:10.3f}   "
            f"{sink_answer['per_layer_share'][layer]:9.3f}   "
            f"{sink_answer['per_layer_sink_heads'][layer]:6d}/12"
        )

    print(
        f"      mean over all layers: IFT-answer {sink_ift['mean_share']:.3f} | "
        f"R-thinking {sink_think['mean_share']:.3f} | "
        f"R-answer {sink_answer['mean_share']:.3f}"
    )
    print(
        f"      heads that are sink-dominant (share > 0.5): "
        f"IFT-answer {sink_ift['sink_head_fraction']:.1%} | "
        f"R-thinking {sink_think['sink_head_fraction']:.1%} | "
        f"R-answer {sink_answer['sink_head_fraction']:.1%}"
    )

    # --- 5. per-head decomposition --------------------------------------------
    distances = per_head_decomposition(
        namespace,
        ift_att,
        reasoning_att,
        start,
        end,
        0,
        ift_end,
        reasoning_cut,
        reasoning_end,
    )

    flat = distances.flatten()

    low, median, high = np.percentile(flat, [10, 50, 90])

    print()
    print("   --- PER-HEAD C2 (one distance per layer x head) -------------------")
    print(
        f"      per-head TV: median {median:.4f}   p10 {low:.4f}   p90 {high:.4f}   "
        f"max {flat.max():.4f}"
    )

    order = np.argsort(flat)[::-1][:8]

    print("      largest per-head gaps:")
    for flat_index in order:
        layer, head = divmod(int(flat_index), distances.shape[1])
        sink_share = sink_ift["share"][layer, head]
        print(
            f"        L{layer:02d} H{head:02d}  TV {flat[flat_index]:.4f}   "
            f"IFT sink share {sink_share:.3f}"
        )

    # Reader heads only. The threshold is swept rather than fixed, because a
    # fixed threshold on a sink-dominated model is a number that moves when
    # the cutoff moves -- the same failure that retired the raw mass metric.
    # If the conclusion only holds at one threshold, that is what this block
    # is for.
    print()
    print("   --- C2 OVER EACH MODEL'S OWN READER HEADS (sink column zeroed,")
    print("       sink-dominant heads dropped) -- swept over the threshold,")
    print("       because the conclusion must not depend on where the cutoff")
    print("       is put. CAVEAT: each model keeps its OWN reader heads, so the")
    print("       two sides hold DIFFERENT populations and part of this")
    print("       distance is composition rather than shape. The shared-head")
    print("       block above is the comparable one.")

    c2_reader = None
    reader_sweep = {}

    for threshold in (0.3, 0.5, 0.7, 0.9):

        left, retained_left, kept_left, total_left = reader_shape(
            namespace, ift_att, 0, ift_end, start, end, threshold
        )
        right, retained_right, kept_right, total_right = reader_shape(
            namespace, reasoning_att, reasoning_cut, reasoning_end, start, end,
            threshold,
        )

        if left is None or right is None:

            empty = "IFT" if left is None else "Reasoning"

            print(
                f"      threshold {threshold:.1f}: no shape -- the {empty} model "
                "has no surviving head, so there is no reader signal to compare"
            )
            reader_sweep[threshold] = None
            continue

        distance = float(namespace["total_variation_distance"](right, left))

        reader_sweep[threshold] = distance

        if threshold == 0.5:
            c2_reader = distance

        print(
            f"      threshold {threshold:.1f}: C2 {c2:.4f} -> {distance:.4f} "
            f"({distance - c2:+.4f})   "
            f"kept IFT {kept_left}/{total_left} (holding {retained_left:.1%} of "
            f"non-sink attention), Reasoning {kept_right}/{total_right} "
            f"(holding {retained_right:.1%})"
        )

    return {
        "prompt": index,
        "kind": kind,
        "cap": cap,
        "measurable": True,
        "C1": c1,
        "C2": c2,
        "floor_c1": floor_c1,
        "floor_c2": floor_c2,
        "c1_matched": c1_matched,
        "c2_matched": c2_matched,
        "p_phase_larger": (
            float((c1_layers > c2_layers).mean())
            if len(c1_layers) and len(c2_layers)
            else None
        ),
        "p_phase_larger_matched": (
            float((layer_boot["C1_matched"] > layer_boot["C2_matched"]).mean())
            if len(layer_boot["C1_matched"]) and len(layer_boot["C2_matched"])
            else None
        ),
        "p_phase_larger_matched_tokens": (
            float((token_boot["C1_matched"] > token_boot["C2_matched"]).mean())
            if len(token_boot["C1_matched"]) and len(token_boot["C2_matched"])
            else None
        ),
        "matched_gap": matched_gap,
        "c1_slice_spread": c1_spread,
        "c2_slice_spread": c2_spread,
        "shared_reader_sweep": shared_sweep,
        "c2_shared_reader": shared_sweep.get(0.5),
        "c2_reader": c2_reader,
        "reader_sweep": reader_sweep,
        "ift_tok": ift_end,
        "think_tok": think_tokens,
        "answer_tok": answer_tokens,
    }


def main():
    print()
    rule()
    print("CACHE-ONLY SOLIDIFICATION  --  noise floor, error bars, matched windows,")
    print("phase-matched sink mass, per-head decomposition")
    rule()

    namespace = load_experiment_namespace()

    print("   extracted from the experiment file:")
    print("     " + ", ".join(WANT_FUNCTIONS))

    cache_dir = resolve_cache_dir(namespace)

    from transformers import AutoTokenizer

    print("   loading tokenizers (no models, no GPU)...")
    ift_tokenizer = AutoTokenizer.from_pretrained(IFT_REPO)
    reasoning_tokenizer = AutoTokenizer.from_pretrained(REASONING_REPO)

    tokenizers = (ift_tokenizer, reasoning_tokenizer)

    indices = [
        index
        for index, spec in enumerate(namespace["PROMPT_SPECS"])
        if spec.get("kind")
    ]

    rows = []
    skipped = []

    for cap in (2048, 1024):

        print()
        rule("=")
        print(f"CAP {cap}")
        rule("=")

        for index in indices:

            row = report_prompt(namespace, index, cap, tokenizers, cache_dir)

            if row is None:
                continue

            if row.get("measurable"):
                rows.append(row)
            else:
                skipped.append(row)

    # --- summary ---------------------------------------------------------------
    print()
    rule("=")
    print("SUMMARY")
    rule("=")

    header = (
        f"   {'cap':>5}  {'prompt':<7}{'kind':<14}{'C1':>8}{'C2':>8}"
        f"{'floorC1':>9}{'floorC2':>9}{'C1/floor':>10}{'C2/floor':>10}"
        f"{'P(C1>C2)':>10}"
    )

    print(header)
    print("   " + "-" * (len(header) - 3))

    for row in rows:
        print(
            f"   {row['cap']:>5}  {row['prompt']:<7}{row['kind']:<14}"
            f"{row['C1']:>8.4f}{row['C2']:>8.4f}"
            f"{row['floor_c1']:>9.4f}{row['floor_c2']:>9.4f}"
            f"{row['C1'] / row['floor_c1']:>10.2f}"
            f"{row['C2'] / row['floor_c2']:>10.2f}"
            f"{(row['p_phase_larger'] if row['p_phase_larger'] is not None else float('nan')):>10.3f}"
        )

    matched_rows = [row for row in rows if row["cap"] == 2048]

    if matched_rows:

        print()
        print("   LENGTH-MATCHED (both answering windows truncated to the shorter one)")
        for row in matched_rows:
            print(
                f"   {row['prompt']:<7}{row['kind']:<14}"
                f"C2 {row['C2']:.4f} -> {row['c2_matched']:.4f} "
                f"({row['c2_matched'] - row['C2']:+.4f})   "
                f"C1 {row['C1']:.4f} -> {row['c1_matched']:.4f} "
                f"({row['c1_matched'] - row['C1']:+.4f})"
            )

        print()
        print("   C2 OVER READER HEADS ONLY, swept over the sink-share threshold")
        for row in matched_rows:
            sweep = row.get("reader_sweep", {})
            parts = []
            for threshold in sorted(sweep):
                value = sweep[threshold]
                parts.append(
                    f"{threshold:.1f}->{value:.4f}" if value is not None
                    else f"{threshold:.1f}->none"
                )
            print(
                f"   {row['prompt']:<7}{row['kind']:<14}"
                f"C2 {row['C2']:.4f}   " + "  ".join(parts)
            )

        print()
        print("   LENGTH-MATCHED, WITH ERROR BARS (the number the claim rests on)")
        header = (
            f"   {'prompt':<7}{'kind':<14}{'C1m':>8}{'C2m':>8}{'gap':>9}"
            f"{'P(C1m>C2m)':>12}{'layers P':>10}{'tokens P':>10}"
        )
        print(header)
        print("   " + "-" * (len(header) - 3))

        for row in matched_rows:

            p_layers = row.get("p_phase_larger_matched")
            p_tokens = row.get("p_phase_larger_matched_tokens")

            print(
                f"   {row['prompt']:<7}{row['kind']:<14}"
                f"{row['c1_matched']:>8.4f}{row['c2_matched']:>8.4f}"
                f"{row['matched_gap']:>+9.4f}"
                f"{(p_layers if p_layers is not None else float('nan')):>12.3f}"
                f"{(p_tokens if p_tokens is not None else float('nan')):>10.3f}"
            )

        print()
        print("   SLICE SENSITIVITY (how far the number moves when the window")
        print("   is re-cut) -- compare the spread with the gap above")
        header = (
            f"   {'prompt':<7}{'kind':<14}{'C1 spread':>12}{'C2 spread':>12}"
            f"{'C1m-C2m gap':>13}{'slice moves more?':>20}"
        )
        print(header)
        print("   " + "-" * (len(header) - 3))

        for row in matched_rows:

            c1_spread = row.get("c1_slice_spread", 0.0)
            c2_spread = row.get("c2_slice_spread", 0.0)
            gap = abs(row.get("matched_gap", 0.0))

            verdict = (
                "YES -- gap does not survive" if max(c1_spread, c2_spread) > gap
                else "no -- gap survives"
            )

            print(
                f"   {row['prompt']:<7}{row['kind']:<14}"
                f"{c1_spread:>12.4f}{c2_spread:>12.4f}{gap:>13.4f}{verdict:>20}"
            )

        print()
        print("   C2 OVER SHARED READER HEADS (the same head set on both sides)")
        header = (
            f"   {'prompt':<7}{'kind':<14}{'C2m':>8}{'0.3':>9}{'0.5':>9}"
            f"{'0.7':>9}{'0.9':>9}"
        )
        print(header)
        print("   " + "-" * (len(header) - 3))

        for row in matched_rows:

            sweep = row.get("shared_reader_sweep", {})

            parts = []

            for threshold in (0.3, 0.5, 0.7, 0.9):

                value = sweep.get(threshold)

                parts.append(
                    f"{value:.4f}" if value is not None else "none"
                )

            print(
                f"   {row['prompt']:<7}{row['kind']:<14}"
                f"{row['c2_matched']:>8.4f}" + "".join(f"{p:>9}" for p in parts)
            )

    # --- the cap question, in one place ---------------------------------------
    by_prompt = {}

    for row in rows:
        by_prompt.setdefault(row["prompt"], {})[row["cap"]] = row

    both = [index for index, caps in by_prompt.items() if len(caps) > 1]

    if both:
        print()
        print("   CAP SENSITIVITY -- the same measurement at two horizons")
        for index in sorted(both):
            low_cap = sorted(by_prompt[index])[0]
            high_cap = sorted(by_prompt[index])[-1]
            low = by_prompt[index][low_cap]
            high = by_prompt[index][high_cap]
            print(
                f"   {index:<7}{high['kind']:<14}"
                f"C1 {low['C1']:.4f} -> {high['C1']:.4f}   "
                f"C2 {low['C2']:.4f} -> {high['C2']:.4f}"
            )

    # A prompt that is measurable at one cap and not at the other is the
    # sharpest cap fact there is, so it is called out rather than left to the
    # reader to notice a missing row.
    for index in sorted(by_prompt):

        caps = by_prompt[index]

        if 2048 in caps and 1024 not in caps:
            print(
                f"   {index:<7}{caps[2048]['kind']:<14}"
                f"measurable at 2048 but NOT at 1024 -- the reasoning model "
                f"needed the extra room to close </think>"
            )

        if 1024 in caps and 2048 not in caps:
            print(
                f"   {index:<7}{caps[1024]['kind']:<14}"
                f"measurable at 1024 but NOT at 2048 -- worth a look"
            )

    print()
    print("   Not measurable (no answering phase to compare against):")

    for row in sorted(skipped, key=lambda item: (item["cap"], item["prompt"])):
        print(
            f"     cap {row['cap']:<5} {row['prompt']:<3} {row['kind']:<14} "
            f"{row['reason']}"
        )

    # The skip list at the current cap is a BEHAVIOUR fact, not a pipeline
    # fact, so it is checked against what the last run recorded. A difference
    # is not a bug -- the model's behaviour may have changed -- but it must be
    # read rather than noticed later.
    expected_skips = set(NON_MEASURABLE_AT_2048)
    actual_skips = {
        row["prompt"] for row in skipped if row["cap"] == 2048
    }

    if expected_skips != actual_skips:
        print()
        print("   !! The 2048 skip list differs from the one recorded at the")
        print("      last run. That is a behaviour change, not a bug -- but it")
        print("      has to be read, not noticed later.")
        print(f"      expected: {sorted(expected_skips)}")
        print(f"      actual  : {sorted(actual_skips)}")

    print()
    print("   What this does NOT establish, however good these numbers look:")
    print("     - one prompt per kind, greedy decoding, no seed variation")
    print("     - the bootstrap measures layer and token sampling noise; it")
    print("       does not measure prompt sampling noise, which is the one")
    print("       that a reader will ask about")
    print("     - the token axis is NOT paired: the three windows are different")
    print("       content, so each is resampled on its own. Only the layer")
    print("       axis supports a paired claim")
    print("     - the split-half floor is a PROXY for the noise, not a")
    print("       calibrated null. Its halves are shorter than the windows")
    print("       C1 and C2 are measured on (so it is conservative) AND they")
    print("       come from one trajectory (so it understates the difference")
    print("       between two INDEPENDENT runs). Only two independent runs")
    print("       can calibrate it, and that needs the GPU")
    print("     - the sink SHARE is a ratio, and the raw attention mass is")
    print("       still not quotable")
    print("     - a cap is a horizon, not a proof: 'did not terminate within")
    print("       N tokens' is the strongest claim any of this supports")
    print()
    rule()


if __name__ == "__main__":
    main()
