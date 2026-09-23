#!/usr/bin/env python3
"""Extract the text-only causal LM from a Qwen3.5 multimodal checkpoint.

Qwen3.5 ships ``Qwen3_5ForConditionalGeneration`` (vision tower + language
model + MTP head), which ``AutoModelForCausalLM`` cannot load. The sparse
attention adapter only needs the language model, so this script remaps
``model.language_model.*`` / ``lm_head.*`` tensors into a standalone
``qwen3_5_text`` causal-LM checkpoint together with the tokenizer and
generation config.

Example:
    python extract_qwen35_text_model.py \
        --repo Qwen/Qwen3.5-27B \
        --out /data/prithvi/models/Qwen3.5-27B-text
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

from huggingface_hub import snapshot_download
from safetensors.torch import load_file, save_file
from transformers import AutoConfig, AutoTokenizer


LANG_PREFIX: str = "model.language_model."
LM_HEAD_PREFIX: str = "lm_head."


def remap_key(key: str) -> Optional[str]:
    """Map a multimodal checkpoint key onto the text-only CausalLM layout.

    Args:
        key: Original tensor name in the ConditionalGeneration checkpoint.

    Returns:
        Remapped name, or ``None`` if the tensor should be dropped (vision/MTP).
    """
    if key.startswith(LANG_PREFIX):
        return "model." + key[len(LANG_PREFIX) :]
    if key.startswith(LM_HEAD_PREFIX):
        return key
    return None


def extract_text_checkpoint(src_dir: Path, out_dir: Path) -> None:
    """Rewrite safetensor shards from ``src_dir`` into a text-only dump at ``out_dir``.

    Args:
        src_dir: Snapshot of the multimodal HuggingFace repo.
        out_dir: Destination directory for the text-only causal LM.
    """
    index_path: Path = src_dir / "model.safetensors.index.json"
    index: Dict[str, object] = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map: Dict[str, str] = index["weight_map"]  # type: ignore[assignment]

    shard_to_keys: Dict[str, List[str]] = defaultdict(list)
    remapped_weight_map: Dict[str, str] = {}
    dropped: int = 0
    for key, shard_name in weight_map.items():
        new_key: Optional[str] = remap_key(key)
        if new_key is None:
            dropped += 1
            continue
        shard_to_keys[shard_name].append(key)
        remapped_weight_map[new_key] = shard_name

    print(
        f"Keeping {len(remapped_weight_map)} tensors, dropping {dropped} "
        f"(vision/MTP) across {len(shard_to_keys)} shards.",
        flush=True,
    )

    total_size: int = 0
    for shard_idx, shard_name in enumerate(sorted(shard_to_keys)):
        src_shard: Path = src_dir / shard_name
        print(
            f"[{shard_idx + 1}/{len(shard_to_keys)}] remapping {shard_name}...",
            flush=True,
        )
        tensors = load_file(str(src_shard))
        remapped: Dict[str, object] = {}
        for key in shard_to_keys[shard_name]:
            new_key = remap_key(key)
            assert new_key is not None
            tensor = tensors[key]
            remapped[new_key] = tensor
            total_size += int(tensor.nbytes)
        dest_shard: Path = out_dir / shard_name
        save_file(remapped, str(dest_shard))
        del tensors, remapped

    new_index: Dict[str, object] = {
        "metadata": {"total_size": total_size},
        "weight_map": remapped_weight_map,
    }
    (out_dir / "model.safetensors.index.json").write_text(
        json.dumps(new_index, indent=2) + "\n", encoding="utf-8"
    )

    full_config = AutoConfig.from_pretrained(str(src_dir))
    text_config = full_config.text_config
    text_config.architectures = ["Qwen3_5ForCausalLM"]
    text_config.save_pretrained(str(out_dir))

    tokenizer = AutoTokenizer.from_pretrained(str(src_dir))
    tokenizer.save_pretrained(str(out_dir))

    for extra_name in (
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ):
        src_extra: Path = src_dir / extra_name
        dest_extra: Path = out_dir / extra_name
        if src_extra.exists() and not dest_extra.exists():
            shutil.copy2(src_extra, dest_extra)

    print(f"Wrote text-only checkpoint to {out_dir}", flush=True)


def parse_args() -> argparse.Namespace:
    """Parse repo id and output directory."""
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Extract a text-only Qwen3.5 causal LM checkpoint."
    )
    parser.add_argument(
        "--repo",
        default="Qwen/Qwen3.5-27B",
        help="HuggingFace repo id of the multimodal post-trained model.",
    )
    parser.add_argument(
        "--out",
        default="/data/prithvi/models/Qwen3.5-27B-text",
        help="Output directory for the text-only checkpoint.",
    )
    return parser.parse_args()


def main() -> None:
    """Download (if needed) and extract the text-only Qwen3.5 checkpoint."""
    args: argparse.Namespace = parse_args()
    out_dir: Path = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {args.repo} (cached if already present)...", flush=True)
    src: str = snapshot_download(repo_id=args.repo)
    print(f"Source snapshot: {src}", flush=True)
    extract_text_checkpoint(Path(src), out_dir)


if __name__ == "__main__":
    main()
