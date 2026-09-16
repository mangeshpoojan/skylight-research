#!/usr/bin/env python3
"""LOFT RAG 32K sweep: {Llama-3.1-8B, Ministral-3-3B, Ministral-3-8B} x 17 configs.

Configs (density targets are nominal at seq_len 32768, incl. Sink(128)+Local(128)):

* dense
* oracle_topk_{20,10,5,2}   Sink+Local+OracleTopK
* pqcache_{20,10,5,2}       Sink+Local+PQCache
* vattn_pq_{20,10,5,2}      Sink+Local+PQCache+AdaptiveSampling
* pq_is_{20,10,5,2}         Sink+Local+PQImportance (2% point as specified;
                            other budgets keep the same heavy:sample ratio)

Run settings follow PR #97 (LOFT-faithful prompt: no primed answer prefix, no
row-level output cap): max_new_tokens=8192, max_context_length=32768.

One process owns one GPU (via CUDA_VISIBLE_DEVICES) and one (model, config)
job, running all requested subsets sequentially so the model loads once.
Includes the matmul k-means patch needed to fit 32K context at pq_bits=8.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

REPO_ROOT: Path = Path("/data/prithvi/skylight-research")
os.chdir(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT))

from benchmark.loft import LoftRag
from sparse_attention_hub.adapters import ModelAdapterHF
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
from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations.utils import (
    pq_utils,
)
from sparse_attention_hub.sparse_attention.research_attention.maskers.sampling.implementations import (
    AdaptiveSamplingMaskerConfig,
)

VALID_SUBSETS: List[str] = [
    "nq_32k",
    "hotpotqa_32k",
    "musique_32k",
    "qampari_32k",
    "quest_32k",
]

# Ministral-3 paths are text-only extractions of the multimodal BF16 repos
# (see extract_ministral3_text_model.py); AutoModelForCausalLM cannot load the
# Mistral3ForConditionalGeneration wrapper directly.
MODELS: Dict[str, str] = {
    "llama31-8b": "/data/jerry/for-lily/models/Llama-3.1-8B-Instruct",
    "ministral3-3b": "/data/prithvi/models/Ministral-3-3B-Instruct-2512-text",
    "ministral3-8b": "/data/prithvi/models/Ministral-3-8B-Instruct-2512-text",
}

MAX_CONTEXT_LENGTH: int = 32768
MAX_NEW_TOKENS: int = 8192
SINK_SIZE: int = 128
WINDOW_SIZE: int = 128
SEQ_LEN: int = 32768
FIXED_FRACTION: float = (SINK_SIZE + WINDOW_SIZE) / float(SEQ_LEN)

# User-specified PQCache heavy sizes per density target (fractions of seq_len).
PQCACHE_HEAVY: Dict[int, float] = {
    20: 0.1921875,
    10: 0.0921875,
    5: 0.0421875,
    2: 0.0121875,
}

# OracleTopK uses the same budget accounting: heavy = target - sink/local share.
ORACLE_HEAVY: Dict[int, float] = PQCACHE_HEAVY

# vAttention(PQCache): (pq_heavy, delta, epsilon, base_rate_sampling).
VATTN_PARAMS: Dict[int, Tuple[float, float, float, float]] = {
    20: (0.1, 0.1, 0.04, 0.05),
    10: (0.05, 0.075, 0.075, 0.025),
    5: (0.025, 0.25, 0.15, 0.01),
    2: (0.01, 0.4, 0.4, 0.005),
}

# PQImportance 2% operating point; other budgets keep the same heavy:sample ratio.
PQ_IS_HEAVY_2PCT: float = 0.00806
PQ_IS_SAMPLE_2PCT: float = 0.003996
PQ_IS_TOTAL_2PCT: float = PQ_IS_HEAVY_2PCT + PQ_IS_SAMPLE_2PCT

RESULT_ROOT: Path = Path("/data/prithvi/pq_importance_runs/loft32k_sweep")


def pq_is_heavy_sample(pct: int) -> Tuple[float, float]:
    """Return (heavy_size, sample_size) for a PQImportance density target.

    2% uses the requested fractions exactly; other budgets keep Sink+Local and
    split the remaining budget with the same heavy:sample ratio.

    Args:
        pct: Target density in percent (2, 5, 10, or 20).

    Returns:
        Tuple of (heavy_size, sample_size) as fractions of sequence length.
    """
    if pct == 2:
        return PQ_IS_HEAVY_2PCT, PQ_IS_SAMPLE_2PCT
    remaining: float = pct / 100.0 - FIXED_FRACTION
    heavy_size: float = remaining * PQ_IS_HEAVY_2PCT / PQ_IS_TOTAL_2PCT
    sample_size: float = remaining * PQ_IS_SAMPLE_2PCT / PQ_IS_TOTAL_2PCT
    return heavy_size, sample_size


def build_config(config_name: str) -> Optional[ResearchAttentionConfig]:
    """Build the sparse attention config for a named sweep point.

    Args:
        config_name: One of dense, oracle_topk_P, pqcache_P, vattn_pq_P,
            pq_is_P with P in {20, 10, 5, 2}.

    Returns:
        ResearchAttentionConfig, or None for dense.
    """
    if config_name == "dense":
        return None

    family: str
    pct_str: str
    family, pct_str = config_name.rsplit("_", 1)
    pct: int = int(pct_str)
    base: List[Any] = [
        SinkMaskerConfig(sink_size=SINK_SIZE),
        LocalMaskerConfig(window_size=WINDOW_SIZE),
    ]

    if family == "oracle_topk":
        return ResearchAttentionConfig(
            masker_configs=base + [OracleTopKConfig(heavy_size=ORACLE_HEAVY[pct])]
        )
    if family == "pqcache":
        return ResearchAttentionConfig(
            masker_configs=base
            + [
                PQCacheConfig(
                    heavy_size=PQCACHE_HEAVY[pct],
                    pq_group_factor=4,
                    pq_bits=8,
                    kmeans_iter=10,
                    init_offset=SINK_SIZE,
                    metric="euclidean",
                )
            ]
        )
    if family == "vattn_pq":
        pq_heavy: float
        delta: float
        epsilon: float
        base_rate: float
        pq_heavy, delta, epsilon, base_rate = VATTN_PARAMS[pct]
        return ResearchAttentionConfig(
            masker_configs=base
            + [
                PQCacheConfig(
                    heavy_size=pq_heavy,
                    pq_group_factor=4,
                    pq_bits=8,
                    kmeans_iter=10,
                    init_offset=SINK_SIZE,
                    metric="euclidean",
                ),
                AdaptiveSamplingMaskerConfig(
                    base_rate_sampling=base_rate,
                    epsilon=epsilon,
                    delta=delta,
                    init_offset=SINK_SIZE,
                    local_offset=WINDOW_SIZE,
                ),
            ]
        )
    if family == "pq_is":
        heavy_size: float
        sample_size: float
        heavy_size, sample_size = pq_is_heavy_sample(pct)
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
            ]
        )
    raise ValueError(f"unknown config {config_name}")


ALL_CONFIGS: List[str] = ["dense"] + [
    f"{family}_{pct}"
    for family in ("oracle_topk", "pqcache", "vattn_pq", "pq_is")
    for pct in (20, 10, 5, 2)
]


def pairwise_distance_matmul(
    data1: torch.Tensor,
    data2: torch.Tensor,
    device: torch.device = torch.device("cpu"),
    tqdm_flag: bool = False,
) -> torch.Tensor:
    """Euclidean pairwise distances without materializing (b, n, k, d).

    Uses ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a.b, which is algebraically the
    same as the broadcast implementation but uses ~1 GiB instead of ~32 GiB
    at 32K context / pq_bits=8.

    Args:
        data1: Samples of shape (b, n, d).
        data2: Cluster centers of shape (b, k, d).
        device: Device for computation.
        tqdm_flag: Unused; kept to match the original signature.

    Returns:
        Distances of shape (b, n, k).
    """
    del tqdm_flag
    data1 = data1.to(device)
    data2 = data2.to(device)
    a2: torch.Tensor = (data1 * data1).sum(dim=-1, keepdim=True)
    b2: torch.Tensor = (data2 * data2).sum(dim=-1, keepdim=True).transpose(1, 2)
    cross: torch.Tensor = torch.bmm(data1, data2.transpose(1, 2))
    dis: torch.Tensor = a2 + b2 - 2.0 * cross
    return dis


def kmeans_batched_efficient(
    X: torch.Tensor,
    num_clusters: int,
    distance: str = "euclidean",
    cluster_centers: Optional[torch.Tensor] = None,
    tol: float = 1e-4,
    tqdm_flag: bool = False,
    iter_limit: int = 0,
    device: torch.device = torch.device("cpu"),
    seed: Optional[int] = None,
) -> tuple:
    """Batched k-means using matmul distances and bmm centroid updates.

    Args:
        X: Input of shape (b, n, d).
        num_clusters: Number of clusters.
        distance: Must be ``euclidean``.
        cluster_centers: Optional initial centers of shape (b, k, d).
        tol: Convergence tolerance on summed center shift.
        tqdm_flag: Unused; kept to match the original signature.
        iter_limit: Max iterations (0 means no limit).
        device: Device for computation.
        seed: Optional RNG seed for initialization.

    Returns:
        Tuple of (cluster assignments (b, n), centers (b, k, d)).
    """
    if X.ndim != 3:
        raise ValueError(
            f"Expected 3D input (b, n, d), got {X.ndim}D tensor with shape {X.shape}"
        )
    if distance != "euclidean":
        raise NotImplementedError(
            f"Distance '{distance}' not yet implemented for batched k-means."
        )
    del tqdm_flag
    b, n, _d = X.shape
    original_dtype = X.dtype
    X = X.float().to(device)
    if cluster_centers is None:
        initial_state: torch.Tensor = pq_utils.initialize_batched(
            X, num_clusters, seed=seed
        )
    else:
        if cluster_centers.shape != (b, num_clusters, _d):
            raise ValueError(
                f"cluster_centers shape mismatch. Expected ({b}, {num_clusters}, {_d}), "
                f"got {cluster_centers.shape}"
            )
        initial_state = cluster_centers.float().to(device)

    iteration: int = 0
    choice_cluster: torch.Tensor
    while True:
        dis: torch.Tensor = pairwise_distance_matmul(X, initial_state, device=device)
        choice_cluster = torch.argmin(dis, dim=2)
        del dis
        initial_state_pre: torch.Tensor = initial_state.clone()
        mask: torch.Tensor = torch.nn.functional.one_hot(
            choice_cluster, num_clusters
        ).float()
        cluster_sums: torch.Tensor = torch.bmm(mask.transpose(1, 2), X)
        cluster_counts: torch.Tensor = mask.sum(dim=1)
        empty_clusters: torch.Tensor = cluster_counts == 0
        cluster_counts_safe: torch.Tensor = cluster_counts.clamp(min=1).unsqueeze(-1)
        new_centers: torch.Tensor = cluster_sums / cluster_counts_safe
        if empty_clusters.any():
            random_indices: torch.Tensor = torch.randint(
                0, n, (b, num_clusters), device=X.device
            )
            batch_idx: torch.Tensor = (
                torch.arange(b, device=X.device).unsqueeze(1).expand(-1, num_clusters)
            )
            random_samples: torch.Tensor = X[batch_idx, random_indices]
            empty_mask: torch.Tensor = empty_clusters.unsqueeze(-1)
            new_centers = torch.where(empty_mask, random_samples, new_centers)
        initial_state = new_centers
        center_shift: torch.Tensor = torch.sqrt(
            ((initial_state - initial_state_pre) ** 2).sum(dim=2)
        ).sum(dim=1)
        iteration += 1
        if (center_shift**2 < tol).all():
            break
        if iter_limit != 0 and iteration >= iter_limit:
            break
    return choice_cluster.to(original_dtype), initial_state.to(original_dtype)


def install_memory_efficient_kmeans() -> None:
    """Patch k-means to avoid the 32 GiB broadcast allocation at 32K context."""
    pq_utils.pairwise_distance_batched = pairwise_distance_matmul
    pq_utils.kmeans_batched = kmeans_batched_efficient
    print("Installed memory-efficient k-means patch (matmul distances).", flush=True)


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
    """Run one LOFT 32K task and return the overall metrics.

    Args:
        adapter: Loaded model adapter reused across subsets of this job.
        subset: LOFT RAG task name, e.g. ``nq_32k``.
        result_dir: Directory to write raw results and metrics.

    Returns:
        The ``overall`` metrics dict (empty if metrics were not written).
    """
    result_dir.mkdir(parents=True, exist_ok=True)
    benchmark: LoftRag = LoftRag([subset])
    generation_kwargs: Dict[str, Any] = {"max_new_tokens": MAX_NEW_TOKENS}
    request_kwargs: Dict[str, Any] = {"max_context_length": MAX_CONTEXT_LENGTH}
    benchmark.run_benchmark(
        adapter,
        str(result_dir),
        request_kwargs=request_kwargs,
        generation_kwargs=generation_kwargs,
    )
    metrics_path: Path = result_dir / "metrics.json"
    if not metrics_path.exists():
        return {}
    with metrics_path.open("r", encoding="utf-8") as file:
        metrics: Dict[str, Any] = json.load(file)
    return metrics.get("overall", {})


def parse_args() -> argparse.Namespace:
    """Parse model, config, and subset list for this process."""
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Run one (model, config) LOFT RAG 32K sweep job."
    )
    parser.add_argument("--model", required=True, choices=sorted(MODELS))
    parser.add_argument("--config", required=True, choices=ALL_CONFIGS)
    parser.add_argument(
        "--subsets",
        default=",".join(VALID_SUBSETS),
        help="Comma-separated LOFT 32K tasks (default: all five).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build the config and print the plan without loading the model.",
    )
    return parser.parse_args()


def main() -> None:
    """Load the requested model and evaluate the requested LOFT 32K tasks."""
    args: argparse.Namespace = parse_args()
    subsets: List[str] = [
        part.strip() for part in args.subsets.split(",") if part.strip()
    ]
    invalid: List[str] = [name for name in subsets if name not in VALID_SUBSETS]
    if invalid:
        raise ValueError(f"invalid subsets {invalid}; expected one of {VALID_SUBSETS}")

    model_path: str = MODELS[args.model]
    sparse_attention_config: Optional[ResearchAttentionConfig] = build_config(
        args.config
    )
    job_root: Path = RESULT_ROOT / args.model / args.config

    print(f"model: {args.model} ({model_path})", flush=True)
    print(f"config: {args.config}", flush=True)
    print(f"subsets: {subsets}", flush=True)
    print(f"max_context_length: {MAX_CONTEXT_LENGTH}", flush=True)
    print(f"max_new_tokens: {MAX_NEW_TOKENS}", flush=True)
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}", flush=True)
    if sparse_attention_config is not None:
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

    tokenizer_kwargs: Dict[str, Any] = {"padding_side": "left"}
    if args.model.startswith("ministral"):
        # Transformers 5.x flags the shipped Mistral pre-tokenizer regex as
        # buggy; this opts into the corrected pattern.
        tokenizer_kwargs["fix_mistral_regex"] = True

    print("Loading model...", flush=True)
    adapter: ModelAdapterHF = ModelAdapterHF(
        model_name=model_path,
        sparse_attention_config=sparse_attention_config,
        model_kwargs={
            "torch_dtype": torch.bfloat16,
            "attn_implementation": "sdpa",
        },
        tokenizer_kwargs=tokenizer_kwargs,
        device="cuda:0",
    )
    print("Model loaded.", flush=True)
    torch.cuda.empty_cache()

    for subset in pending:
        subset_dir: Path = job_root / subset
        start_time: float = time.time()
        print(f"\n===== starting {args.model}/{args.config}/{subset} =====", flush=True)
        overall: Dict[str, Any] = run_subset(adapter, subset, subset_dir)
        elapsed: float = time.time() - start_time
        print(
            f"===== finished {args.model}/{args.config}/{subset} in {elapsed/60:.1f} min: "
            f"overall={overall} =====",
            flush=True,
        )
        summary_path: Path = subset_dir / "run_summary.json"
        with summary_path.open("w", encoding="utf-8") as file:
            json.dump(
                {
                    "benchmark": "loft_rag",
                    "subset": subset,
                    "model": args.model,
                    "model_path": model_path,
                    "config": args.config,
                    "max_context_length": MAX_CONTEXT_LENGTH,
                    "max_new_tokens": MAX_NEW_TOKENS,
                    "elapsed_seconds": round(elapsed, 1),
                    "overall": overall,
                },
                file,
                indent=2,
            )
        print(f"Wrote {summary_path}", flush=True)


if __name__ == "__main__":
    main()
