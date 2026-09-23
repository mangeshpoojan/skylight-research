#!/usr/bin/env python3
"""Official LongBench v1 sweep on Qwen3.5-27B with Sink(16)+Local(16).

Same eval as the 128/128 v1 sweep (THUDM LongBench, 21 tasks, official
prompts/maxlens, middle truncation at 31500, hybrid GDN, micro-metrics)
but Sink/Local are 16 tokens and the config set is:

* dense
* pq_is_{2,5}          Sink(16)+Local(16)+PQImportance (Ray Tune sidecar)
* pqcache_{2,5}        Sink(16)+Local(16)+PQCache (plain, LOFT PQ hyperparams)
* oracle_topk_{2,5}    Sink(16)+Local(16)+OracleTopK

Heavy sizes spend ``target% - 32/32768`` after Sink+Local.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type

import torch

REPO_ROOT: Path = Path("/data/prithvi/skylight-research")
os.chdir(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT))

from benchmark.longbench import LongBench, OFFICIAL_32K_MAX_LENGTH, OFFICIAL_V1_TASKS
from sparse_attention_hub.adapters import ModelAdapterHF
from sparse_attention_hub.metric_logging.logger import MicroMetricLogger
from sparse_attention_hub.sparse_attention.research_attention import (
    ResearchAttentionConfig,
)
from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
    LocalMaskerConfig,
    OracleTopKConfig,
    PQCacheConfig,
    PQImportanceConfig,
    SinkMaskerConfig,
)
from benchmark.scripts.pq_kmeans_patch import install_memory_efficient_kmeans

VALID_SUBSETS: List[str] = list(OFFICIAL_V1_TASKS)

MODEL_KEY: str = "qwen35-27b"
MODEL_PATH: str = "/data/prithvi/models/Qwen3.5-27B-text"

MAX_CONTEXT_LENGTH: int = OFFICIAL_32K_MAX_LENGTH
MAX_NEW_TOKENS: int = 512
SINK_SIZE: int = 16
WINDOW_SIZE: int = 16
SEQ_LEN: int = 32768
FIXED_FRACTION: float = (SINK_SIZE + WINDOW_SIZE) / float(SEQ_LEN)

RESULT_ROOT: Path = Path("/data/prithvi/pq_importance_runs/longbench_qwen35_27b_v1_sl16")
TUNE_BEST_DIR: Path = Path("/data/prithvi/pq_importance_runs/pq_is_tune_qwen35_27b_sl16")

ALL_CONFIGS: List[str] = [
    "dense",
    "pq_is_2",
    "pq_is_5",
    "pqcache_2",
    "pqcache_5",
    "oracle_topk_2",
    "oracle_topk_5",
]


def remaining_budget(pct: int) -> float:
    """Return the fraction left for the third masker after Sink+Local.

    Args:
        pct: Target density in percent.

    Returns:
        ``pct/100 - 32/32768``.
    """
    remaining: float = pct / 100.0 - FIXED_FRACTION
    if remaining <= 0:
        raise ValueError(f"{pct}% is smaller than Sink+Local ({FIXED_FRACTION:.6f})")
    return remaining


def _config_from_serialized(data: Dict[str, Any]) -> ResearchAttentionConfig:
    """Rebuild a ResearchAttentionConfig from the Ray Tune sidecar JSON.

    Args:
        data: Serialized ``sparse_config`` dict.

    Returns:
        ResearchAttentionConfig with ``apply_to_layer_types`` set for Qwen3.5.
    """
    type_map: Dict[str, Type] = {
        "SinkMaskerConfig": SinkMaskerConfig,
        "LocalMaskerConfig": LocalMaskerConfig,
        "PQImportanceConfig": PQImportanceConfig,
        "PQCacheConfig": PQCacheConfig,
        "OracleTopKConfig": OracleTopKConfig,
    }
    maskers: List[Any] = []
    for item in data["masker_configs"]:
        cls: Type = type_map[item["type"]]
        valid: set = {field.name for field in fields(cls)}
        params: Dict[str, Any] = {
            key: value for key, value in item.get("params", {}).items() if key in valid
        }
        maskers.append(cls(**params))
    apply_raw: Any = data.get("apply_to_layer_types") or ["full_attention"]
    return ResearchAttentionConfig(
        masker_configs=maskers,
        apply_to_layer_types=tuple(apply_raw),
    )


def load_tuned_pq_is(pct: int) -> Optional[ResearchAttentionConfig]:
    """Load the Ray Tune best PQImportance config for a density target.

    Args:
        pct: Target density in percent (2 or 5).

    Returns:
        Tuned config, or None if the sidecar has not been written yet.
    """
    path: Path = TUNE_BEST_DIR / f"best_pq_is_{pct}.json"
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as file:
        payload: Dict[str, Any] = json.load(file)
    config: ResearchAttentionConfig = _config_from_serialized(payload["sparse_config"])
    config.apply_to_layer_types = ("full_attention",)
    return config


def build_config(config_name: str) -> Optional[ResearchAttentionConfig]:
    """Build the sparse attention config for a named sl16 sweep point.

    Args:
        config_name: One of dense, pq_is_P, pqcache_P, oracle_topk_P.

    Returns:
        ResearchAttentionConfig, or None for dense.
    """
    if config_name == "dense":
        return None

    family: str
    pct_str: str
    family, pct_str = config_name.rsplit("_", 1)
    pct: int = int(pct_str)
    leftover: float = remaining_budget(pct)
    base: List[Any] = [
        SinkMaskerConfig(sink_size=SINK_SIZE),
        LocalMaskerConfig(window_size=WINDOW_SIZE),
    ]
    layer_types: Tuple[str, ...] = ("full_attention",)

    if family == "oracle_topk":
        return ResearchAttentionConfig(
            masker_configs=base + [OracleTopKConfig(heavy_size=leftover)],
            apply_to_layer_types=layer_types,
        )
    if family == "pqcache":
        return ResearchAttentionConfig(
            masker_configs=base
            + [
                PQCacheConfig(
                    heavy_size=leftover,
                    pq_group_factor=4,
                    pq_bits=8,
                    kmeans_iter=10,
                    init_offset=SINK_SIZE,
                    metric="euclidean",
                )
            ],
            apply_to_layer_types=layer_types,
        )
    if family == "pq_is":
        tuned: Optional[ResearchAttentionConfig] = load_tuned_pq_is(pct)
        if tuned is not None:
            print(
                f"Using Ray Tune PQ+IS config from {TUNE_BEST_DIR / f'best_pq_is_{pct}.json'}",
                flush=True,
            )
            tuned.apply_to_layer_types = layer_types
            return tuned
        print(
            f"No tuned PQ+IS config for {pct}%; falling back to 50/50 remaining budget.",
            flush=True,
        )
        heavy_size: float = leftover * 0.5
        sample_size: float = leftover - heavy_size
        return ResearchAttentionConfig(
            masker_configs=base
            + [
                PQImportanceConfig(
                    heavy_size=heavy_size,
                    sample_size=sample_size,
                    pq_group_factor=4,
                    pq_bits=8,
                    kmeans_iter=10,
                    init_offset=SINK_SIZE,
                    metric="euclidean",
                )
            ],
            apply_to_layer_types=layer_types,
        )
    raise ValueError(f"unknown config {config_name}")


def subset_is_complete(result_dir: Path) -> bool:
    """Return True if this subset already has raw results and metrics.

    Args:
        result_dir: Per-subset result directory.

    Returns:
        Whether the run can be skipped.
    """
    return (result_dir / "metrics.json").exists() and (
        result_dir / "raw_results.csv"
    ).exists()


def run_subset(adapter: ModelAdapterHF, subset: str, result_dir: Path) -> Dict[str, Any]:
    """Run one LongBench task and return the overall metrics.

    Args:
        adapter: Loaded model adapter reused across subsets of this job.
        subset: LongBench task name, e.g. ``narrativeqa``.
        result_dir: Directory to write raw results and metrics.

    Returns:
        The metrics dict (empty if metrics were not written).
    """
    result_dir.mkdir(parents=True, exist_ok=True)
    metric_logger: MicroMetricLogger = MicroMetricLogger()
    metric_logger.flush()
    metric_logger.configure_logging(
        log_path=str(result_dir),
        enabled_metrics=["research_attention_density", "research_attention_output_error"],
        sampling_factor=0.1,
        max_records=20000,
    )
    benchmark: LongBench = LongBench([subset])
    generation_kwargs: Dict[str, Any] = {"max_new_tokens": MAX_NEW_TOKENS, "do_sample": False}
    request_kwargs: Dict[str, Any] = {
        "max_context_length": MAX_CONTEXT_LENGTH,
        "truncate_from_middle": True,
    }
    benchmark.run_benchmark(
        adapter,
        str(result_dir),
        request_kwargs=request_kwargs,
        generation_kwargs=generation_kwargs,
    )
    metric_logger.flush()
    metrics_path: Path = result_dir / "metrics.json"
    if not metrics_path.exists():
        return {}
    with metrics_path.open("r", encoding="utf-8") as file:
        metrics: Dict[str, Any] = json.load(file)
    return metrics


def parse_args() -> argparse.Namespace:
    """Parse config and subset list for this process."""
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Run one Qwen3.5-27B LongBench sl16 sweep job."
    )
    parser.add_argument("--config", required=True, choices=ALL_CONFIGS)
    parser.add_argument(
        "--subsets",
        default=",".join(VALID_SUBSETS),
        help="Comma-separated LongBench v1 tasks (default: official 21).",
    )
    parser.add_argument(
        "--model-path",
        default=MODEL_PATH,
        help="Path to the text-only Qwen3.5-27B checkpoint.",
    )
    parser.add_argument(
        "--is-hybrid",
        dest="is_hybrid",
        action="store_true",
        default=True,
        help="Enable token-by-token question consume (required for GatedDeltaNet).",
    )
    parser.add_argument(
        "--no-hybrid",
        dest="is_hybrid",
        action="store_false",
        help="Disable hybrid token-by-token question consume (not recommended).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build the config and print the plan without loading the model.",
    )
    return parser.parse_args()


def main() -> None:
    """Load Qwen3.5-27B and evaluate the requested LongBench tasks."""
    args: argparse.Namespace = parse_args()
    subsets: List[str] = [
        part.strip() for part in args.subsets.split(",") if part.strip()
    ]
    invalid: List[str] = [name for name in subsets if name not in VALID_SUBSETS]
    if invalid:
        raise ValueError(f"invalid subsets {invalid}; expected one of {VALID_SUBSETS}")

    model_path: str = args.model_path
    sparse_attention_config: Optional[ResearchAttentionConfig] = build_config(
        args.config
    )
    job_root: Path = RESULT_ROOT / MODEL_KEY / args.config

    print(f"model: {MODEL_KEY} ({model_path})", flush=True)
    print(f"config: {args.config}", flush=True)
    print(f"sink/local: {SINK_SIZE}/{WINDOW_SIZE}", flush=True)
    print(f"subsets: {subsets}", flush=True)
    print(f"hybrid / is_hybrid: {args.is_hybrid}", flush=True)
    print(f"max_context_length: {MAX_CONTEXT_LENGTH} (official 32k middle truncate)", flush=True)
    print(f"max_new_tokens cap: {MAX_NEW_TOKENS} (official per-task dataset2maxlen)", flush=True)
    print("truncate_from_middle: True", flush=True)
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}", flush=True)
    if sparse_attention_config is not None:
        print(
            f"  apply_to_layer_types: {sparse_attention_config.apply_to_layer_types}",
            flush=True,
        )
        for masker_config in sparse_attention_config.masker_configs:
            print(f"  masker: {masker_config}", flush=True)

    if args.dry_run:
        print("dry run: config built OK", flush=True)
        return

    pending: List[str] = []
    for subset in subsets:
        if subset_is_complete(job_root / subset):
            print(f"skip {subset} (already complete)", flush=True)
        else:
            pending.append(subset)
    if not pending:
        print("all subsets complete; nothing to do", flush=True)
        return

    install_memory_efficient_kmeans()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    print("Loading model...", flush=True)
    adapter: ModelAdapterHF = ModelAdapterHF(
        model_name=model_path,
        sparse_attention_config=sparse_attention_config,
        model_kwargs={
            "torch_dtype": torch.bfloat16,
            "attn_implementation": "sdpa",
            "use_kernels": True,
        },
        tokenizer_kwargs={"padding_side": "left"},
        device="cuda:0",
        hybrid=args.is_hybrid,
    )
    print("Model loaded.", flush=True)
    print(f"adapter.hybrid={adapter.hybrid}", flush=True)
    torch.cuda.empty_cache()

    for subset in pending:
        subset_dir: Path = job_root / subset
        start_time: float = time.time()
        print(f"\n===== starting {MODEL_KEY}/{args.config}/{subset} =====", flush=True)
        overall: Dict[str, Any] = run_subset(adapter, subset, subset_dir)
        elapsed: float = time.time() - start_time
        print(
            f"===== finished {MODEL_KEY}/{args.config}/{subset} in {elapsed/60:.1f} min: "
            f"metrics={overall} =====",
            flush=True,
        )
        summary_path: Path = subset_dir / "run_summary.json"
        with summary_path.open("w", encoding="utf-8") as file:
            json.dump(
                {
                    "benchmark": "longbench",
                    "subset": subset,
                    "model": MODEL_KEY,
                    "model_path": model_path,
                    "config": args.config,
                    "sink_size": SINK_SIZE,
                    "window_size": WINDOW_SIZE,
                    "hybrid": args.is_hybrid,
                    "apply_to_layer_types": (
                        list(sparse_attention_config.apply_to_layer_types)
                        if sparse_attention_config is not None
                        and sparse_attention_config.apply_to_layer_types is not None
                        else None
                    ),
                    "max_context_length": MAX_CONTEXT_LENGTH,
                    "max_new_tokens": MAX_NEW_TOKENS,
                    "truncate_from_middle": True,
                    "official_longbench_v1": True,
                    "elapsed_seconds": round(elapsed, 1),
                    "metrics": overall,
                },
                file,
                indent=2,
            )
        print(f"Wrote {summary_path}", flush=True)


if __name__ == "__main__":
    main()
