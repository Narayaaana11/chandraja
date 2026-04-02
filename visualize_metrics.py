"""Plot MAE and R2 metric trends from historical training runs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from smart_eval.config import RUN_HISTORY_FILE
from smart_eval.ml.run_history import load_run_history


def main() -> None:
    """Render and optionally save MAE/R2 trend plots and model comparison across run history."""
    parser = argparse.ArgumentParser(description="Visualize MAE and R2 across training runs")
    parser.add_argument("--output", default="logs/metrics_trend.png", help="Path to save chart image")
    args = parser.parse_args()

    runs = load_run_history(RUN_HISTORY_FILE)
    if not runs:
        print("No run history found.")
        return

    ordered = sorted(runs, key=lambda row: row.get("timestamp", ""))
    x = list(range(1, len(ordered) + 1))
    mae = [float(r.get("metrics", {}).get("mae", 0.0)) for r in ordered]
    r2 = [float(r.get("metrics", {}).get("r2", 0.0)) for r in ordered]

    # Check if latest run has candidates (from tuning)
    latest_run = ordered[-1] if ordered else {}
    candidates = latest_run.get("candidates", [])
    has_candidates = len(candidates) > 1  # Only show comparison if multiple candidates

    # Create subplots: metric trend (always) + model comparison (if available)
    if has_candidates:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(10, 5))

    # Plot 1: Metric trends
    ax1.plot(x, mae, marker="o", label="MAE")
    ax1.plot(x, r2, marker="o", label="R2")
    ax1.set_xlabel("Run Number (chronological)")
    ax1.set_ylabel("Metric Value")
    ax1.set_title("Training Metrics Across Runs")
    ax1.grid(True, linestyle="--", alpha=0.3)
    ax1.legend()

    # Plot 2: Model comparison (if candidates available)
    if has_candidates:
        model_names = [c.get("model_name", "Unknown") for c in candidates]
        model_r2 = [float(c.get("metrics", {}).get("r2", 0.0)) for c in candidates]
        best_model = latest_run.get("model", "Unknown")

        # Color the best model differently
        colors = ["green" if name == best_model else "steelblue" for name in model_names]

        ax2.bar(model_names, model_r2, color=colors, alpha=0.7)
        ax2.set_ylabel("R² Score (Test Set)")
        ax2.set_title(f"Model Comparison - Latest Run ({latest_run.get('timestamp', '')})")
        ax2.set_ylim(0, max(model_r2) * 1.1 if model_r2 else 1.0)
        ax2.grid(True, linestyle="--", alpha=0.3, axis="y")

        # Add value labels on bars
        for i, (name, r2_val) in enumerate(zip(model_names, model_r2)):
            ax2.text(i, r2_val + 0.01, f"{r2_val:.4f}", ha="center", va="bottom", fontsize=9)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"Saved metric trend plot to: {out_path}")
    if has_candidates:
        print(f"  Showing {len(candidates)} candidate models from latest run")



if __name__ == "__main__":
    main()
