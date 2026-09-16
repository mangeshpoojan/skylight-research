#!/usr/bin/env python3
"""Collect LOFT-32K sweep metrics into one table.

Scans /data/prithvi/pq_importance_runs/loft32k_sweep/<model>/<config>/<subset>/
for metrics.json files and prints a per-model table with one row per config and
one column per subset (subspan_em for the single-answer tasks, coverage for
qampari/quest, matching how LOFT reports them), plus a macro average.

Also writes the full long-format data to sweep_results.csv in the sweep root.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

SWEEP_ROOT: Path = Path("/data/prithvi/pq_importance_runs/loft32k_sweep")

SUBSETS: List[str] = ["nq_32k", "hotpotqa_32k", "musique_32k", "qampari_32k", "quest_32k"]
MULTI_VALUE: List[str] = ["qampari_32k", "quest_32k"]

CONFIG_ORDER: List[str] = ["dense"] + [
    f"{family}_{pct}"
    for family in ("oracle_topk", "pqcache", "vattn_pq", "pq_is")
    for pct in (20, 10, 5, 2)
]


def headline_metric(subset: str, task_metrics: Dict[str, Any]) -> Optional[float]:
    """Return the headline metric for one subset (coverage or subspan_em).

    Args:
        subset: LOFT task name.
        task_metrics: The per-task metrics dict from metrics.json.

    Returns:
        Coverage for multi-value tasks, subspan_em otherwise; None if absent.
    """
    key: str = "coverage" if subset in MULTI_VALUE else "subspan_em"
    value: Any = task_metrics.get(key)
    return float(value) if value is not None else None


def main() -> None:
    """Print per-model result tables and write the long-format CSV."""
    rows: List[Dict[str, Any]] = []
    for model_dir in sorted(SWEEP_ROOT.iterdir()):
        if not model_dir.is_dir() or model_dir.name == "logs":
            continue
        for config_dir in sorted(model_dir.iterdir()):
            for subset in SUBSETS:
                metrics_path: Path = config_dir / subset / "metrics.json"
                if not metrics_path.exists():
                    continue
                with metrics_path.open("r", encoding="utf-8") as file:
                    metrics: Dict[str, Any] = json.load(file)
                task_metrics: Dict[str, Any] = metrics.get("task_metrics", {}).get(
                    subset, {}
                )
                rows.append(
                    {
                        "model": model_dir.name,
                        "config": config_dir.name,
                        "subset": subset,
                        "headline": headline_metric(subset, task_metrics),
                        **{
                            k: task_metrics.get(k)
                            for k in ("em", "subspan_em", "f1", "coverage", "num_samples")
                        },
                    }
                )

    if not rows:
        print("No metrics found yet.")
        return

    df: pd.DataFrame = pd.DataFrame(rows)
    csv_path: Path = SWEEP_ROOT / "sweep_results.csv"
    df.to_csv(csv_path, index=False)

    for model in sorted(df["model"].unique()):
        model_df: pd.DataFrame = df[df["model"] == model]
        pivot: pd.DataFrame = model_df.pivot_table(
            index="config", columns="subset", values="headline"
        )
        pivot = pivot.reindex(
            [c for c in CONFIG_ORDER if c in pivot.index],
            columns=[s for s in SUBSETS if s in pivot.columns],
        )
        pivot["macro"] = pivot.mean(axis=1, skipna=False)
        done_subsets: int = int(model_df.shape[0])
        print(f"\n=== {model}  ({done_subsets}/85 subset runs complete) ===")
        print(
            "(headline = coverage for qampari/quest, subspan_em otherwise; "
            "macro over all 5)"
        )
        print(pivot.round(4).to_string())

    print(f"\nLong-format CSV: {csv_path}")


if __name__ == "__main__":
    main()
