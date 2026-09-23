"""LongBench benchmark implementation for long context evaluation."""

from typing import Dict, Any, List
import pandas as pd

from ..base import Benchmark
from ..benchmark_registry import register_benchmark
from .calculate_metrics import calculate_metrics, calculate_metrics_e
from .official_v1 import (
    DATASET_TO_MAX_NEW_TOKENS,
    OFFICIAL_V1_TASKS,
    format_official_sample,
    iter_official_rows,
)

# Official LongBench v1 lives on THUDM/LongBench. The class attribute below is
# kept for LongBench-E rows and for existing unit tests.
OFFICIAL_V1_DATASET_ID: str = "THUDM/LongBench"


@register_benchmark("longbench")
class LongBench(Benchmark):
    """LongBench benchmark for evaluating long context understanding.

    LongBench is a comprehensive benchmark for evaluating the ability of large language
    models to understand long contexts. It includes 35 tasks across various domains:
    22 standard tasks and 13 extended tasks with longer contexts.

    The benchmark evaluates different capabilities:
    - Question Answering: narrativeqa, qasper, multifieldqa_en/zh, hotpotqa, etc.
    - Summarization: gov_report, qmsum, multi_news, vcsum, samsum
    - Classification: trec, lsht
    - Retrieval: passage_retrieval_en, passage_retrieval_zh
    - Code Completion: lcc, repobench-p
    - Counting: passage_count

    Example:
        >>> longbench = LongBench(subsets_to_run=["narrativeqa", "qasper"])
        >>> results = longbench.run_benchmark(adapter, result_dir="/path/to/results")
        >>> print(f"Average score: {results}")
    """

    # All available LongBench datasets (22 standard + 13 extended)
    all_datasets: List[str] = [
        # Standard LongBench datasets
        "narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh", 
        "hotpotqa", "2wikimqa", "musique", "dureader", "gov_report", 
        "qmsum", "multi_news", "vcsum", "trec", "triviaqa", "samsum", 
        "lsht", "passage_count", "passage_retrieval_en", 
        "passage_retrieval_zh", "lcc", "repobench-p",
        # LongBench-E (extended) datasets with longer contexts
        "qasper_e", "multifieldqa_en_e", "hotpotqa_e", "2wikimqa_e", 
        "gov_report_e", "multi_news_e", "trec_e", "triviaqa_e", 
        "samsum_e", "passage_count_e", "passage_retrieval_en_e", 
        "lcc_e", "repobench-p_e"
    ]
    
    benchmark_name: str = "longbench"
    huggingface_dataset_id: str = "Xnhyacinth/LongBench"

    def _load_datasets(self) -> pd.DataFrame:
        """Load LongBench datasets by individual configs.

        Standard v1 tasks (the original 21, not LongBench v2) are loaded from
        the official THUDM/LongBench jsonl files and formatted with
        ``dataset2prompt`` / ``dataset2maxlen``. Extended ``*_e`` tasks use the
        same official zip when present.

        Returns:
            Combined pandas DataFrame with all samples from subsets_to_run.
        """
        print(f"Loading LongBench datasets: {self.subsets_to_run}")
        dfs: List[pd.DataFrame] = []

        for subset in self.subsets_to_run:
            try:
                from datasets import load_dataset

                if subset in OFFICIAL_V1_TASKS or (
                    subset.endswith("_e") and subset.removesuffix("_e") in OFFICIAL_V1_TASKS
                ):
                    prompt_task: str = subset.removesuffix("_e")
                    rows: List[Dict[str, Any]] = []
                    for row in iter_official_rows(subset):
                        context: str
                        question: str
                        prefix: str
                        context, question, prefix = format_official_sample(
                            prompt_task, row
                        )
                        rows.append(
                            {
                                "context": context,
                                "question": question,
                                "answer_prefix": prefix,
                                "answers": row["answers"],
                                "all_classes": row.get("all_classes"),
                                "length": row.get("length"),
                                "task": subset,
                                "max_new_tokens": DATASET_TO_MAX_NEW_TOKENS[prompt_task],
                            }
                        )
                    subset_df: pd.DataFrame = pd.DataFrame(rows)
                    print(
                        f"  ✓ Loaded {len(subset_df)} official v1 samples from "
                        f"THUDM/LongBench/{subset}.jsonl"
                    )
                else:
                    subset_dataset = load_dataset(
                        self.huggingface_dataset_id, subset, split="test"
                    )
                    subset_df = subset_dataset.to_pandas()
                    subset_df["task"] = subset
                    print(f"  ✓ Loaded {len(subset_df)} samples from {subset}")
                dfs.append(subset_df)
            except Exception as subset_error:
                print(f"  ❌ Failed to load {subset}: {str(subset_error)}")
                continue

        if not dfs:
            raise Exception("No LongBench subsets could be loaded successfully")

        combined_df: pd.DataFrame = pd.concat(dfs, ignore_index=True)
        print(f"Combined {len(combined_df)} total samples from {len(dfs)} subsets")
        return combined_df

    def post_run_evaluate(self, results_df: pd.DataFrame) -> Dict[str, Any]:
        """Compute evaluation metrics for LongBench results.

        Args:
            results_df: DataFrame containing benchmark results with columns:
                - task: Dataset/task name
                - predicted_answer: Model's predicted answer
                - answers: Ground truth answers (list)
                - all_classes: Available classes for classification tasks
                - length: Context length (for extended datasets)

        Returns:
            Dictionary containing computed metrics:
            - For standard datasets: {"overall_score": float, "task_scores": dict}
            - For extended datasets: length-based breakdown plus task scores
        """
        if len(results_df) == 0:
            return {"error": "No results to evaluate"}

        # Group results by task
        task_groups = results_df.groupby("task")
        task_scores: Dict[str, Any] = {}
        standard_scores: List[float] = []
        extended_scores: Dict[str, List[float]] = {"0-4k": [], "4-8k": [], "8k+": []}
        
        for task_name, task_df in task_groups:
            try:
                if task_name.endswith("_e"):
                    # Extended dataset - use calculate_metrics_e for length-based breakdown
                    length_scores: Dict[str, float] = calculate_metrics_e(task_df)
                    task_scores[task_name] = length_scores
                    
                    # Accumulate extended scores for overall calculation
                    for length_range in extended_scores:
                        if length_range in length_scores:
                            extended_scores[length_range].append(length_scores[length_range])
                else:
                    # Standard dataset - use calculate_metrics
                    score: float = calculate_metrics(task_df)
                    task_scores[task_name] = score
                    standard_scores.append(score)
                    
            except Exception as e:
                print(f"Error evaluating task {task_name}: {str(e)}")
                task_scores[task_name] = {"error": str(e)}

        # Compute overall metrics
        overall_metrics: Dict[str, Any] = {"task_scores": task_scores}
        
        # Overall score for standard datasets
        if standard_scores:
            overall_metrics["standard_overall_score"] = round(sum(standard_scores) / len(standard_scores), 2)
        
        # Overall scores for extended datasets by length range
        if any(extended_scores.values()):
            extended_overall: Dict[str, float] = {}
            for length_range, scores in extended_scores.items():
                if scores:
                    extended_overall[length_range] = round(sum(scores) / len(scores), 2)
            if extended_overall:
                overall_metrics["extended_overall_scores"] = extended_overall
        
        # Compute grand overall score if we have both standard and extended results
        all_scores: List[float] = standard_scores.copy()
        for scores in extended_scores.values():
            all_scores.extend(scores)
        
        if all_scores:
            overall_metrics["overall_score"] = round(sum(all_scores) / len(all_scores), 2)
        
        # Add summary statistics
        overall_metrics["summary"] = {
            "total_tasks": len(task_scores),
            "standard_tasks": len([t for t in task_scores.keys() if not t.endswith("_e")]),
            "extended_tasks": len([t for t in task_scores.keys() if t.endswith("_e")]),
            "total_samples": len(results_df)
        }
        
        return overall_metrics 