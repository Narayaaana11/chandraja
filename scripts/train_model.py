from pathlib import Path
import argparse
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from smart_eval.ml.train import train_and_save_model


def main() -> None:
    """Train model from dataset and print run summary."""
    parser = argparse.ArgumentParser(description="Train Smart Eval scoring model")
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/sample_training_data.csv",
        help="Path to CSV dataset",
    )
    args = parser.parse_args()

    result = train_and_save_model(Path(args.dataset))

    print("Training complete")
    print(f"Run ID: {result.run_id}")
    print(f"Rows: {result.rows} (train={result.train_rows}, val={result.val_rows}, test={result.test_rows})")
    print(f"MAE: {result.metrics['mae']:.4f}")
    print(f"R2: {result.metrics['r2']:.4f}")
    print(f"RMSE: {result.metrics['rmse']:.4f}")
    print(f"MAPE: {result.metrics['mape']:.4f}")
    print(f"Dataset summary: {result.dataset_summary}")
    print(f"Model saved: {result.model_path}")
    print(f"Best model: {result.best_model_path}")
    print(f"Metadata saved: {result.metadata_path}")


if __name__ == "__main__":
    main()
