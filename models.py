# -*- coding: utf-8 -*-
"""Model loading for the reasoning vs non-reasoning attention comparison.

Kept in its own module so the two models are loaded ONCE per Colab kernel
session and then stay resident in GPU memory. Re-running the experiment
script re-imports this module, which is a no-op once it is in sys.modules --
so iterating on the experiment itself costs no reload.

The cache lives in sys.modules, so Runtime -> Restart session clears it and
the next run reloads from scratch.
"""

import functools
import subprocess
import sys

# Install dependencies once, on the first import of this module.
#
# Written as plain Python rather than the `!pip` shell escape: `%run`
# transforms `%magic` lines but not `!` escapes, so a `!pip` line here raises
# SyntaxError.
subprocess.check_call(
    [sys.executable, "-m", "pip", "install", "-q", "-U",
     "transformers", "accelerate", "sentencepiece"]
)

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

IFT_MODEL = "Scale-or-Reason/Qwen2.5-1.5B-ift"
REASONING_MODEL = "Scale-or-Reason/Qwen2.5-1.5B-reasoning"


def _report_environment():
    print("PyTorch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        print(
            "VRAM:",
            round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2),
            "GB",
        )


def _load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float32,
        device_map="auto",
        attn_implementation="eager",
    )

    model.eval()

    return tokenizer, model


def _check_on_gpu(model, which):
    """Warn if accelerate silently offloaded weights off the GPU.

    device_map="auto" does not raise when VRAM runs out -- it moves weights to
    CPU memory and carries on. The only symptom is that everything becomes
    glacially slow, which is easy to misread as "still loading".
    """

    devices = {p.device.type for p in model.parameters()}

    if devices - {"cuda"}:
        print(
            f"WARNING: the {which} model has parameters on {sorted(devices)} "
            "rather than cuda only. VRAM is exhausted, so weights were "
            "offloaded to CPU and generation will be extremely slow. Restart "
            "the session and load one model at a time."
        )


@functools.lru_cache(maxsize=None)
def get_model(which):
    """Return (tokenizer, model) for 'ift' or 'reasoning'.

    Cached for the life of the kernel. The first call loads from the Hub;
    every later call returns the resident objects immediately, so editing and
    re-running the experiment script does not reload anything.
    """

    if which == "ift":
        model_name = IFT_MODEL
    elif which == "reasoning":
        model_name = REASONING_MODEL
    else:
        raise ValueError(f"unknown model {which!r}; expected 'ift' or 'reasoning'")

    print(f"Loading {which} model (cached after this): {model_name} ...")

    tokenizer, model = _load_model(model_name)

    _check_on_gpu(model, which)

    print(f"{which} model ready on {next(model.parameters()).device}.")

    return tokenizer, model


_report_environment()
