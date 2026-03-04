"""
Utility functions for the evaluation pipeline.

- Config loading and validation
- Results formatting and saving
- Dataset downloading
"""

import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List

import yaml
from tabulate import tabulate

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load YAML config file."""
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    logger.info(f"Loaded config from {config_path}")
    return cfg


def _custom_json_default(obj):
    """Custom JSON serializer for objects not serializable by default json dump."""
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    elif hasattr(obj, "model_dump"):  # Pydantic v2
        return obj.model_dump()
    elif hasattr(obj, "dict"):  # Pydantic v1
        return obj.dict()
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


def save_results(
    results: Dict[str, Any],
    model_name: str,
    dataset_name: str,
    output_dir: str,
) -> str:
    """Save evaluation results to a JSON file.

    Args:
        results: Evaluation results dict.
        model_name: Name of the model.
        dataset_name: Name of the dataset.
        output_dir: Directory to save results.

    Returns:
        Path to the saved JSON file.
    """
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{model_name}__{dataset_name}__{timestamp}.json"
    filepath = os.path.join(output_dir, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model": model_name,
                "dataset": dataset_name,
                "timestamp": timestamp,
                "results": results,
            },
            f,
            indent=2,
            ensure_ascii=False,
            default=_custom_json_default,
        )

    logger.info(f"Results saved to {filepath}")
    return filepath


def load_all_results(output_dir: str) -> List[Dict[str, Any]]:
    """Load all result JSON files from the output directory."""
    results = []
    if not os.path.exists(output_dir):
        return results

    for filename in sorted(os.listdir(output_dir)):
        if filename.endswith(".json"):
            filepath = os.path.join(output_dir, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                results.append(json.load(f))
    return results


def format_results_table(results: List[Dict[str, Any]]) -> str:
    """Format results as a readable table.

    Returns:
        Formatted string with a table of results.
    """
    if not results:
        return "No results found."

    # Flatten results into rows
    rows = []
    for r in results:
        model = r.get("model", "?")
        dataset = r.get("dataset", "?")
        metrics = r.get("results", {})

        if isinstance(metrics, dict):
            # MTEB returns nested dicts; extract main scores
            flat_metrics = _flatten_metrics(metrics)
            for metric_name, value in flat_metrics.items():
                # Normalize MTEB names to match pytrec_eval (e.g. ndcg_at_10 -> ndcg_cut_10)
                if "_at_" in metric_name and (
                    "ndcg" in metric_name or "map" in metric_name
                ):
                    normalized_name = metric_name.replace("_at_", "_cut_")
                elif (
                    metric_name.startswith("nauc_")
                    or "precision" in metric_name
                    or "recall" in metric_name
                ):
                    continue  # Skip verbose MTEB metrics for clarity in table
                else:
                    normalized_name = metric_name

                rows.append([model, dataset, normalized_name, f"{value:.4f}"])
        else:
            rows.append([model, dataset, "raw", str(metrics)])

    headers = ["Model", "Dataset", "Metric", "Score"]
    return tabulate(rows, headers=headers, tablefmt="github")


def _flatten_metrics(d: Any, prefix: str = "") -> Dict[str, float]:
    """Recursively flatten nested metric dicts, adapting for MTEB 2.10+ lists."""
    flat = {}
    if isinstance(d, dict):
        for key, value in d.items():
            # Skip non-metric metadata fields from MTEB 2.10+ outputs
            if key in [
                "hf_subset",
                "languages",
                "dataset_revision",
                "evaluation_time",
                "main_score",
            ]:
                continue

            full_key = f"{prefix}.{key}" if prefix else str(key)
            if isinstance(value, dict):
                flat.update(_flatten_metrics(value, full_key))
            elif isinstance(value, list):
                for item in value:
                    flat.update(_flatten_metrics(item, full_key))
            elif isinstance(value, (int, float)):
                # Keep the simplest name possible for the table (like ndcg_at_10)
                flat[str(key)] = float(value)
    elif isinstance(d, list):
        for item in d:
            flat.update(_flatten_metrics(item, prefix))
    return flat


def print_summary_table(output_dir: str) -> None:
    """Print a summary table of all results."""
    results = load_all_results(output_dir)
    table = format_results_table(results)
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS SUMMARY")
    print("=" * 80)
    print(table)
    print("=" * 80 + "\n")
