"""LOFT RAG benchmark for long-context retrieval-augmented generation.

Fidelity to https://github.com/google-deepmind/loft (sha 219f68e), measured:

* METRICS are faithful -- scoring identical outputs through this module and through
  upstream's RagEvaluation / MultiValueRagEvaluation gives identical results.
* DATA is a third-party mirror (`f20180301/loft-rag-*`), not LOFT's download.sh +
  preprocess.py output.  Checked against the official rag/<ds>.zip files:
  - `*_128k` IS LOFT: `test` (100 rows) is LOFT's 128k test split and `dev` (10) its
    dev split, with identical gold answers, and context+question is byte-identical to
    upstream's own prompt (same corpus, same order) for every test row.
  - `*_32k` is NOT LOFT.  LOFT ships no 32k test split; the mirror pairs the first 100
    (60 for qampari/quest) of LOFT's 128k test queries with its own corpus drawn from
    the 128k one, ~1.5x LOFT's 32k length (29.8k-30.2k vs 18.3k-20.0k words), and puts
    every few-shot and test gold document BEFORE all distractors (qampari: 325 golds, 2
    distractors).  Treat 32k as an easier, position-biased variant, not a LOFT number.
    Its `dev` is LOFT's dev queries, but their golds are often missing from this corpus
    (nq 9, hotpotqa 6, musique 4, qampari 0, quest 0 of 10), so dev scores are floored
    by the data, not the model.
  `dev` and `test` share one context per subset.  `test` is the split to report --
  use `by_split["test"]`; `overall` pools both splits for backward compatibility.
* CONTEXT LENGTH: the mirror's 32k prompts are 42-46k Llama tokens (LOFT's own 32k
  prompts are 28-29k), and truncation keeps the head, so max_context_length=32768
  silently removes the 5-shot chain-of-thought block at the end of the context and
  some gold documents.  Set max_context_length from your tokenizer.
* GENERATION: greedy, matching upstream's temperature=0.  Upstream sets no
  max_output_tokens, so gemini-1.5-pro's 8192-token output limit applies; the mirror's
  max_new_tokens=256 is replaced with that (see `_load_datasets`).
* PROMPT: upstream sends Gemini a raw prompt; this harness applies the model's chat
  template (unavoidable for an Instruct model), which inserts a system turn before the
  corpus and an assistant header after the query.  Deliberate, documented deviation.
  For reasoning models (e.g. Qwen3.5) render the template with thinking disabled and
  stop on the chat end-of-turn token: extract_prediction takes the FIRST bracketed line,
  so an open reasoning block gets graded instead of the answer.
"""

from typing import Any, Dict, List

import pandas as pd

from ..base import Benchmark
from ..benchmark_registry import register_benchmark
from .calculate_metrics import calculate_metrics


@register_benchmark("loft_rag")
class LoftRag(Benchmark):
    """LOFT RAG benchmark for evaluating long-context retrieval-augmented generation.

    LOFT (Long-context Open Foundation Tasks) RAG evaluates the ability of models to
    answer questions given long retrieved contexts. This benchmark includes:

    - Single-value RAG datasets: nq, hotpotqa, musique
    - Multi-value RAG datasets: qampari, quest

    Each dataset is available in multiple context lengths: 32k, 128k, 1m.

    Metrics:
    - Single-value: EM (Exact Match), Subspan EM, F1
    - Multi-value: EM, Coverage, Subspan EM

    Reference:
        https://github.com/google-deepmind/loft

    Example:
        >>> loft_rag = LoftRag(subsets_to_run=["nq_32k", "hotpotqa_128k"])
        >>> results = loft_rag.run_benchmark(adapter, result_dir="/path/to/results")
        >>> print(f"EM score: {results['nq_32k']['em']}")
    """

    all_datasets: List[str] = [
        "nq_32k",
        "nq_128k",
        "nq_1m",
        "hotpotqa_32k",
        "hotpotqa_128k",
        "hotpotqa_1m",
        "musique_32k",
        "musique_128k",
        "musique_1m",
        "qampari_32k",
        "qampari_128k",
        "qampari_1m",
        "quest_32k",
        "quest_128k",
        "quest_1m",
    ]

    benchmark_name: str = "loft_rag"
    huggingface_dataset_id: str = "f20180301/rag"
    # LOFT's prompt ends at the query: the model emits the TITLE/ID reasoning step, THEN
    # "Final Answer: [...]".  Priming the prefix suppresses that CoT (+0.056 macro
    # subspan_em when removed).  Still used for scoring.
    prompt_includes_answer_prefix: bool = False
    # Upstream's output budget: gemini-1.5-pro's 8192-token output limit (upstream sets no
    # max_output_tokens of its own).
    max_new_tokens: int = 8192

    def _load_datasets(self) -> pd.DataFrame:
        """Load LOFT RAG datasets from HuggingFace Hub.

        Returns:
            Combined pandas DataFrame with all samples from subsets_to_run.
        """
        print(f"Loading LOFT RAG datasets: {self.subsets_to_run}")
        dfs: List[pd.DataFrame] = []

        for subset in self.subsets_to_run:
            parts: List[str] = subset.split("_")
            if len(parts) < 2:
                raise ValueError(
                    f"Invalid subset format: {subset} (expected: dataset_length)"
                )

            length: str = parts[-1]
            dataset: str = "_".join(parts[:-1])
            hf_dataset_id: str = f"f20180301/loft-rag-{dataset}-{length}"

            from datasets import load_dataset

            dataset_dict = load_dataset(hf_dataset_id)

            subset_dfs: List[pd.DataFrame] = []
            for split_name in ["dev", "test"]:
                if split_name in dataset_dict:
                    split_df: pd.DataFrame = dataset_dict[split_name].to_pandas()
                    split_df["split"] = split_name
                    subset_dfs.append(split_df)

            if not subset_dfs:
                raise ValueError(f"No splits found for {subset} ({hf_dataset_id})")

            subset_df: pd.DataFrame = pd.concat(subset_dfs, ignore_index=True)
            subset_df["task"] = subset
            dfs.append(subset_df)
            print(f"  ✓ Loaded {len(subset_df)} samples from {subset}")

        if not dfs:
            raise ValueError("No LOFT RAG subsets could be loaded")

        combined_df: pd.DataFrame = pd.concat(dfs, ignore_index=True)
        print(f"Combined {len(combined_df)} total samples from {len(dfs)} subsets")

        required_columns: List[str] = [
            "context",
            "question",
            "answers",
            "task",
            "answer_prefix",
            "max_new_tokens",
        ]
        missing_columns: List[str] = [
            col for col in required_columns if col not in combined_df.columns
        ]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        # base.py takes min(caller, row), so the mirror's max_new_tokens=256 was a ceiling
        # no caller could raise -- it truncated ~half the sparse-attention rows before
        # they emitted "Final Answer" once LOFT's chain-of-thought prompt was restored.
        # Upstream sets no max_output_tokens, so gemini-1.5-pro's own 8192 output limit
        # governs; use that as the row budget (callers may still pass a lower cap).
        # Never drop the column: the adapter's decode loop stops only on EOS.
        combined_df["max_new_tokens"] = self.max_new_tokens

        return combined_df

    def post_run_evaluate(self, results_df: pd.DataFrame) -> Dict[str, Any]:
        """Compute evaluation metrics for LOFT RAG results.

        Args:
            results_df: DataFrame containing benchmark results

        Returns:
            Dictionary containing computed metrics
        """
        if len(results_df) == 0:
            return {"error": "No results to evaluate"}

        task_groups = results_df.groupby("task")
        task_metrics: Dict[str, Dict[str, float]] = {}
        all_em_scores: List[float] = []
        all_subspan_em_scores: List[float] = []
        all_f1_scores: List[float] = []
        all_coverage_scores: List[float] = []

        for task_name, task_df in task_groups:
            metrics: Dict[str, Any] = calculate_metrics(task_df)

            if "error" in metrics:
                print(f"  ❌ Error evaluating {task_name}: {metrics['error']}")
                continue

            task_metrics[task_name] = metrics
            all_em_scores.append(metrics["em"])
            all_subspan_em_scores.append(metrics["subspan_em"])

            # Multi-value f1 exists only to mirror upstream's per-task key: upstream
            # appends f1 solely in the unparseable branch, so it is 0.0 by construction
            # and is NOT a measurement.  Upstream has no cross-task macro; folding this
            # placeholder into ours would deflate overall.f1 (~x0.6) for no model reason.
            if "f1" in metrics and "num_scored_for_coverage" not in metrics:
                all_f1_scores.append(metrics["f1"])
            if "coverage" in metrics:
                all_coverage_scores.append(metrics["coverage"])

            metric_str: str = (
                f"EM={metrics['em']:.4f}, Subspan_EM={metrics['subspan_em']:.4f}"
            )
            if "f1" in metrics:
                metric_str += f", F1={metrics['f1']:.4f}"
            if "coverage" in metrics:
                metric_str += f", Coverage={metrics['coverage']:.4f}"
            print(f"  ✓ {task_name}: {metric_str}")

        overall_metrics: Dict[str, Any] = {
            "overall": {
                "em": (
                    float(sum(all_em_scores) / len(all_em_scores))
                    if all_em_scores
                    else 0.0
                ),
                "subspan_em": (
                    float(sum(all_subspan_em_scores) / len(all_subspan_em_scores))
                    if all_subspan_em_scores
                    else 0.0
                ),
            },
            "task_metrics": {
                task: {k: round(v, 4) if isinstance(v, float) else v for k, v in m.items()}
                for task, m in task_metrics.items()
            },
            "summary": {"total_tasks": len(task_metrics), "total_samples": len(results_df)},
        }

        if all_f1_scores:
            overall_metrics["overall"]["f1"] = float(
                sum(all_f1_scores) / len(all_f1_scores)
            )
        if all_coverage_scores:
            overall_metrics["overall"]["coverage"] = float(
                sum(all_coverage_scores) / len(all_coverage_scores)
            )

        overall_metrics["overall"] = {
            k: round(v, 4) if isinstance(v, float) else v
            for k, v in overall_metrics["overall"].items()
        }

        # `overall` pools LOFT's dev and test splits against one shared corpus.  The
        # corpus is built around the TEST queries, so dev golds are largely absent from it
        # and dev scores are floored by the data -- expose the breakdown, and name `test`
        # (the larger, LOFT-comparable split) rather than reporting only the pooled figure.
        if "split" in results_df.columns:
            by_split: Dict[str, Dict[str, Any]] = {}
            for split_name, split_df in results_df.groupby("split"):
                per_task: Dict[str, Dict[str, float]] = {}
                for task_name, task_df in split_df.groupby("task"):
                    split_metrics = calculate_metrics(task_df)
                    if "error" in split_metrics:
                        continue
                    per_task[str(task_name)] = {
                        k: round(v, 4) if isinstance(v, float) else v
                        for k, v in split_metrics.items()
                    }
                if not per_task:
                    continue
                agg: Dict[str, Any] = {"n_samples": int(len(split_df))}
                for key in ("em", "subspan_em", "f1", "coverage"):
                    # Skip multi-value's structurally-zero f1, as above.
                    vals = [
                        m[key]
                        for m in per_task.values()
                        if key in m and not (key == "f1" and "num_scored_for_coverage" in m)
                    ]
                    if vals:
                        agg[key] = round(float(sum(vals) / len(vals)), 4)
                by_split[str(split_name)] = {"overall": agg, "task_metrics": per_task}
            if by_split:
                overall_metrics["by_split"] = by_split
                overall_metrics["summary"]["loft_comparable_split"] = "test"

        return overall_metrics

