"""Memory-efficient batched k-means used by PQCache / PQImportance at 32K.

The stock broadcast distance tensor is ~32 GiB at pq_bits=8 / 32K context.
This implementation is algebraically identical for Euclidean distance.
"""

from typing import Optional, Tuple

import torch

from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations.utils import (
    pq_utils,
)


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
) -> Tuple[torch.Tensor, torch.Tensor]:
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
