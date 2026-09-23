#!/usr/bin/env python3
"""Ray Tune search for PQImportance (PQ + IS) on Qwen3.5-27B.

Uses the existing ConfigSearchManager / BenchmarkHelper plumbing with a
PQImportance builder. Does not modify OPTIMIZATION_EXPERIMENT.py.

Search is intentionally small: LongBench v1 hotpotqa, 4 requests, official
middle truncation at 31500, greedy 32-token generation. Grid is
heavy_frac in {0.50, 0.67, 0.80} x pq_group_factor in {2, 4} at 2% and 5%
density. Invalid heavy/sample combinations are rejected before the model loads.

Best configs are written to ``best_pq_is_{2,5}.json`` for the LongBench sweep.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT: Path = Path("/data/prithvi/skylight-research")
os.chdir(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT))
RAYTUNE_DIR: Path = REPO_ROOT / "benchmark" / "raytune"
sys.path.insert(0, str(RAYTUNE_DIR))
os.environ["PYTHONPATH"] = (
    os.environ.get("PYTHONPATH", "") + f":{REPO_ROOT}:{RAYTUNE_DIR}"
)

import ray
import torch

from config_builders.pq_importance import PQImportanceConfigBuilder
from config_builders.utility import serialize_sparse_config
from search_manager import ConfigSearchManager
from sparse_attention_hub.sparse_attention.research_attention import (
    ResearchAttentionConfig,
)

MODEL_PATH: str = "/data/prithvi/models/Qwen3.5-27B-text"
TUNE_DIR: Path = Path("/data/prithvi/pq_importance_runs/pq_is_tune_qwen35_27b")
TASK: str = "longbench/hotpotqa"
SPARSITY_OBJECTIVES: List[int] = [2, 5]
LAYER_TYPES: Tuple[str, ...] = ("full_attention",)


def write_sidecar(pct: int, optimal: Any) -> Path:
    """Write a stable sidecar the LongBench sweep can load.

    Args:
        pct: Density target in percent.
        optimal: OptimalConfig from ConfigSearchManager.

    Returns:
        Path to the sidecar JSON.
    """
    TUNE_DIR.mkdir(parents=True, exist_ok=True)
    sidecar: Path = TUNE_DIR / f"best_pq_is_{pct}.json"
    payload: Dict[str, Any] = {
        "sparsity_pct": pct,
        "model": MODEL_PATH,
        "task": TASK,
        "score": getattr(optimal, "score", None),
        "hyperparams": getattr(optimal, "hyperparams", {}),
        "search_time": getattr(optimal, "search_time", None),
        "num_trials": getattr(optimal, "num_trials", None),
        "sparse_config": serialize_sparse_config(optimal.sparse_config),
    }
    with sidecar.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)
    print(f"Wrote {sidecar}", flush=True)
    return sidecar


def parse_args() -> argparse.Namespace:
    """Parse Ray Tune flags."""
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Tune PQImportance (PQ+IS) on Qwen3.5-27B / LongBench v1 hotpotqa."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run search even if best_pq_is_{2,5}.json already exist.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build the search space and exit without starting Ray.",
    )
    return parser.parse_args()


def main() -> None:
    """Run PQ+IS Ray Tune for 2% and 5% and write sidecar configs."""
    args: argparse.Namespace = parse_args()
    TUNE_DIR.mkdir(parents=True, exist_ok=True)

    if (
        not args.force
        and (TUNE_DIR / "best_pq_is_2.json").exists()
        and (TUNE_DIR / "best_pq_is_5.json").exists()
    ):
        print(f"Tuned sidecars already exist in {TUNE_DIR}; skipping.", flush=True)
        return

    builder: PQImportanceConfigBuilder = PQImportanceConfigBuilder()
    _optimal: List
    to_optimize: List[Tuple[str, Optional[ResearchAttentionConfig], Optional[List]]]
    _optimal, to_optimize = builder.build_configs(
        model_config={},
        sparsity_objectives=SPARSITY_OBJECTIVES,
        memory_objectives=[],
    )
    for _name, config, _classes in to_optimize:
        if config is None:
            continue
        config.apply_to_layer_types = LAYER_TYPES
        print(f"search template: objective={config.objective}", flush=True)
        print(f"  apply_to_layer_types={config.apply_to_layer_types}", flush=True)
        for masker_config in config.masker_configs:
            print(f"  masker: {masker_config}", flush=True)
            search_space: Dict[str, Any] = getattr(masker_config, "search_space", {})
            if search_space:
                print(f"    search_space keys: {list(search_space.keys())}", flush=True)

    if args.dry_run:
        print("dry run: PQImportance search space built OK", flush=True)
        return

    n_gpus: int = torch.cuda.device_count()
    print(f"Initializing Ray with {n_gpus} GPUs", flush=True)
    ray.init(ignore_reinit_error=True)

    manager: ConfigSearchManager = ConfigSearchManager(
        optimal_configs_dir=str(TUNE_DIR),
        force_search=True,
        generation_kwargs={"max_new_tokens": 32, "do_sample": False},
        request_kwargs={
            "max_context_length": 31500,
            "max_requests": 4,
            "truncate_from_middle": True,
        },
        ray_results_dir=str(TUNE_DIR / "ray"),
        hybrid=True,
        extra_model_kwargs={
            "attn_implementation": "sdpa",
            "use_kernels": True,
        },
        use_memory_efficient_kmeans=True,
    )

    for name, config, classes in to_optimize:
        if config is None:
            continue
        pct: int = int(config.objective)
        print(f"\n===== tuning pq_is_{pct} ({name}) =====", flush=True)
        optimal = manager.search_optimal_config(
            model=MODEL_PATH,
            task=TASK,
            masker_name=f"pq_is_{pct}",
            masker_classes=classes,
            full_sparse_config=config,
            actors_per_gpu=1,
        )
        if optimal.sparse_config is not None:
            optimal.sparse_config.apply_to_layer_types = LAYER_TYPES
        write_sidecar(pct, optimal)
        print(
            f"===== done pq_is_{pct} score={optimal.score} "
            f"trials={optimal.num_trials} params={optimal.hyperparams} =====",
            flush=True,
        )

    ray.shutdown()


if __name__ == "__main__":
    main()
