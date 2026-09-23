"""Configuration builder for PQImportance (PQ cache + importance sampling)."""

from functools import partial
from typing import Dict, List, Optional, Tuple

from ray import tune

from sparse_attention_hub.sparse_attention.research_attention import ResearchAttentionConfig
from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
    LocalMaskerConfig,
    PQImportanceConfig,
    SinkMaskerConfig,
)

from .base import BaseConfigBuilder
from .factory import register_builder
from .utility import get_masker_list_name

SINK_SIZE: int = 128
WINDOW_SIZE: int = 128
SEQ_LEN: int = 32768
FIXED_FRACTION: float = (SINK_SIZE + WINDOW_SIZE) / float(SEQ_LEN)
HEAVY_FRACS: List[float] = [0.50, 0.67, 0.80]


def _budget_matches(config: ResearchAttentionConfig, remaining: float) -> bool:
    """Return whether PQ heavy_size + sample_size matches the leftover budget.

    Args:
        config: Candidate research attention config.
        remaining: Target fraction after Sink+Local, e.g. 0.02 - 256/32768.

    Returns:
        True if the PQImportance budget matches ``remaining``.
    """
    pq_config: PQImportanceConfig = config.masker_configs[2]
    total: float = float(pq_config.heavy_size) + float(pq_config.sample_size)
    return abs(total - remaining) < 1e-5


@register_builder("pq_importance")
class PQImportanceConfigBuilder(BaseConfigBuilder):
    """Builder for Sink + Local + PQImportance sparse attention configurations."""

    sink_size: int
    window_size: int
    seq_len: int
    fixed_fraction: float

    def __init__(
        self,
        sink_size: int = SINK_SIZE,
        window_size: int = WINDOW_SIZE,
        seq_len: int = SEQ_LEN,
    ) -> None:
        """Store Sink/Local sizes used to carve the PQImportance budget.

        Args:
            sink_size: Absolute sink token count. Defaults to 128.
            window_size: Absolute local-window token count. Defaults to 128.
            seq_len: Sequence length used to convert Sink+Local to a fraction.
        """
        self.sink_size = sink_size
        self.window_size = window_size
        self.seq_len = seq_len
        self.fixed_fraction = (sink_size + window_size) / float(seq_len)

    def build_configs(
        self,
        model_config: Dict[str, str],
        sparsity_objectives: List[int],
        memory_objectives: List[int],
        **kwargs
    ) -> Tuple[List[Tuple[str, Optional[ResearchAttentionConfig], Optional[List]]],
               List[Tuple[str, Optional[ResearchAttentionConfig], Optional[List]]]]:
        """Build PQImportance configs that Ray Tune should search.

        Uses:
            sparsity_objectives: Target densities in percent (e.g. 2, 5).
        Ignores:
            memory_objectives: Unused.
            model_config: Unused.

        Returns:
            Tuple of (optimal_configs, to_optimize_configs). All PQImportance
            configs are placed in ``to_optimize_configs``.
        """
        del model_config, memory_objectives, kwargs
        optimal_configs: List[Tuple[str, Optional[ResearchAttentionConfig], Optional[List]]] = []
        to_optimize_configs: List[Tuple[str, Optional[ResearchAttentionConfig], Optional[List]]] = []

        for sparsity_objective in sparsity_objectives:
            remaining: float = float(sparsity_objective) / 100.0 - self.fixed_fraction
            if remaining <= 0:
                raise ValueError(
                    f"sparsity objective {sparsity_objective}% is smaller than "
                    f"Sink+Local ({self.fixed_fraction * 100:.3f}%)"
                )
            classes: List = [SinkMaskerConfig, LocalMaskerConfig, PQImportanceConfig]
            name: str = get_masker_list_name(
                classes,
                other_params={
                    "builder": "pq_importance",
                    "sparsity_obj": sparsity_objective,
                    "sink_size": self.sink_size,
                    "window_size": self.window_size,
                },
            )
            default_heavy: float = remaining * 0.67
            default_sample: float = remaining - default_heavy
            config: ResearchAttentionConfig = ResearchAttentionConfig(
                masker_configs=[
                    SinkMaskerConfig(sink_size=self.sink_size),
                    LocalMaskerConfig(window_size=self.window_size),
                    PQImportanceConfig(
                        heavy_size=default_heavy,
                        sample_size=default_sample,
                        pq_group_factor=4,
                        pq_bits=8,
                        kmeans_iter=10,
                        init_offset=self.sink_size,
                        metric="euclidean",
                    ),
                ]
            )
            heavies: List[float] = [remaining * frac for frac in HEAVY_FRACS]
            samples: List[float] = [remaining * (1.0 - frac) for frac in HEAVY_FRACS]
            config.masker_configs[2].search_space = {
                "heavy_size": tune.grid_search(heavies),
                "sample_size": tune.grid_search(samples),
                "pq_group_factor": tune.grid_search([2, 4]),
                "pq_bits": tune.grid_search([8]),
                "kmeans_iter": tune.grid_search([10]),
                "metric": tune.grid_search(["euclidean"]),
            }
            config.validity_constraint = partial(_budget_matches, remaining=remaining)
            config.objective = sparsity_objective
            to_optimize_configs.append((name, config, classes))

        return optimal_configs, to_optimize_configs
