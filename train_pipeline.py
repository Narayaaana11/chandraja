"""CLI pipeline for loading data, preprocessing, training, evaluation, and versioning."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data.load_dataset import compute_dataset_hash
from smart_eval.config import MANIFEST_FILE, RUN_HISTORY_FILE
from smart_eval.ml.model_registry import get_last_dataset_hash, update_manifest_hash_and_best
from smart_eval.ml.run_history import load_run_history, compare_runs
from smart_eval.ml.train import compare_training_runs, train_and_save_model


def build_parser() -> argparse.ArgumentParser:
    """Create argument parser for train pipeline CLI."""
    parser = argparse.ArgumentParser(description="Train Smart Eval pipeline end-to-end")
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/sample_training_data.csv",
        help="Path to dataset source (CSV/JSON/folder)",
    )
    parser.add_argument(
        "--retrain",
        action="store_true",
        help="Explicitly trigger full retraining and best-model update",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force retraining even if dataset hash matches",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Print run history sorted by R2 descending after execution",
    )
    return parser


def main() -> None:
    """Execute training pipeline with hash check, run comparison, and optional skip."""
    parser = build_parser()
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"Error: Dataset file not found: {dataset_path}")
        sys.exit(1)

    current_hash = compute_dataset_hash(dataset_path)
    last_hash = get_last_dataset_hash(MANIFEST_FILE)

    should_train = args.force or (current_hash != last_hash)

    if not should_train:
        print("[SKIP] Dataset unchanged since last run. Use --force to retrain anyway.")
        runs = load_run_history(RUN_HISTORY_FILE)
        if runs:
            last_run = runs[-1]
            metrics = last_run.get("metrics", {})
            print(f"Last run metrics: MAE={metrics.get('mae', 'N/A'):.4f}, R2={metrics.get('r2', 'N/A'):.4f}")
        if args.compare:
            print("\nRun history (sorted by R2):")
            print(compare_training_runs())
        return

    if args.retrain or current_hash != last_hash:
        print("[TRAIN] Dataset changed (or --force). Starting retraining pipeline...")

    result = train_and_save_model(dataset_path)

    update_manifest_hash_and_best(
        MANIFEST_FILE,
        dataset_hash=current_hash,
        timestamp=result.timestamp,
        best_model_path=result.best_model_path,
        best_r2=result.metrics["r2"],
    )

    print("Pipeline complete")
    print(f"Run ID: {result.run_id}")
    print(f"Rows: {result.rows} (train={result.train_rows}, val={result.val_rows}, test={result.test_rows})")
    print(f"Metrics: MAE={result.metrics['mae']:.4f}, R2={result.metrics['r2']:.4f}, RMSE={result.metrics['rmse']:.4f}, MAPE={result.metrics['mape']:.4f}")
    print(f"Dataset summary: {result.dataset_summary}")
    print(f"Model saved: {result.model_path}")
    print(f"Best model path: {result.best_model_path}")
    print(f"Metadata saved: {result.metadata_path}")

    if args.compare:
        print("\nRun history (sorted by R2):")
        print(compare_training_runs())


if __name__ == "__main__":
    main()
