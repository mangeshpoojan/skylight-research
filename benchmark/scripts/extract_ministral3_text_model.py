#!/usr/bin/env python3
"""Extract the text-only causal LM from a Ministral-3 multimodal checkpoint.

The Ministral-3 BF16 repos ship ``Mistral3ForConditionalGeneration`` (vision
tower + language model), which ``AutoModelForCausalLM`` refuses to load. The
sparse-attention adapter only needs the language model, so this script copies
the text backbone + lm_head into a standalone ``ministral3`` causal-LM
checkpoint together with the tokenizer and generation config.

Example:
    python extract_ministral3_text_model.py \
        --repo mistralai/Ministral-3-3B-Instruct-2512-BF16 \
        --out /data/prithvi/models/Ministral-3-3B-Instruct-2512-text
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoTokenizer,
)


def main() -> None:
    """Convert one Ministral-3 multimodal checkpoint to a text-only causal LM."""
    parser: argparse.ArgumentParser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True, help="HF repo id of the BF16 model.")
    parser.add_argument("--out", required=True, help="Output directory.")
    args: argparse.Namespace = parser.parse_args()

    out_dir: Path = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading multimodal model {args.repo}...", flush=True)
    full_model = AutoModelForImageTextToText.from_pretrained(
        args.repo, torch_dtype=torch.bfloat16
    )

    text_config = full_model.config.text_config
    text_config.dtype = torch.bfloat16
    print(f"Building text-only {text_config.model_type} causal LM...", flush=True)
    lm = AutoModelForCausalLM.from_config(text_config, torch_dtype=torch.bfloat16)

    missing, unexpected = lm.model.load_state_dict(
        full_model.model.language_model.state_dict(), strict=True
    )
    assert not missing and not unexpected, (missing, unexpected)
    lm.lm_head.load_state_dict(full_model.lm_head.state_dict(), strict=True)
    lm.generation_config = full_model.generation_config

    print(f"Saving to {out_dir}...", flush=True)
    lm.save_pretrained(out_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.repo)
    tokenizer.save_pretrained(out_dir)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
