"""Model training pipeline with deterministic splits, metrics logging, and model versioning."""

from __future__ import annotations

import json
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from smart_eval.config import (
    BEST_MODEL_FILE,
    LEGACY_MODEL_FILE,
    LEGACY_MODEL_METADATA_FILE,
    MANIFEST_FILE,
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    RUN_HISTORY_FILE,
    SPLITS_FILE,
    TRAINING_CONFIG,
    TUNING_CONFIG,
)
from smart_eval.ml.features import extract_feature_metrics, parse_keywords
from smart_eval.ml.metrics import regression_metrics
from smart_eval.ml.model_registry import append_manifest_record, select_best_model
from smart_eval.ml.run_history import append_run_history, compare_runs, utc_now_iso
from smart_eval.ml.tuning import build_model, tune_model
from smart_eval.services.evaluation import SemanticScorer

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from data.load_dataset import load_dataset, preprocess, summarize_dataset

REQUIRED_COLUMNS = {
    "student_text",
    "reference_text",
    "keywords",
    "max_marks",
    "target_marks",
}


@dataclass
class TrainingResult:
    """Container for one training run outputs."""

    run_id: str
    timestamp: str
    rows: int
    train_rows: int
    val_rows: int
    test_rows: int
    metrics: Dict[str, float]
    model_path: str
    metadata_path: str
    best_model_path: str
    dataset_summary: Dict[str, Any]


def _validate_dataset(df: pd.DataFrame) -> None:
    """Validate dataset contains required supervised training columns."""
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"Dataset missing required columns: {missing_text}")


def _build_feature_dataframe(df: pd.DataFrame, scorer: SemanticScorer) -> pd.DataFrame:
    """Transform raw text rows into numeric model features and target ratio."""
    rows: List[Dict[str, float]] = []

    for row in df.to_dict(orient="records"):
        max_marks = float(row["max_marks"])
        target_marks = float(row["target_marks"])
        if max_marks <= 0:
            continue

        metrics = extract_feature_metrics(
            student_text=str(row["student_text"]),
            reference_text=str(row["reference_text"]),
            keywords=parse_keywords(str(row["keywords"])),
            semantic_fn=scorer.similarity,
        )

        rows.append(
            {
                "semantic_score": float(metrics["semantic_score"]),
                "keyword_score": float(metrics["keyword_score"]),
                "length_score": float(metrics["length_score"]),
                "target_ratio": float(max(0.0, min(target_marks / max_marks, 1.0))),
            }
        )

    feature_df = pd.DataFrame(rows)
    if feature_df.empty or len(feature_df) < 5:
        raise ValueError("Need at least 5 valid rows to train the model.")
    return feature_df


def _timestamp_slug() -> str:
    """Return timestamp slug used in model artifact names."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _build_model(model_cfg: Dict[str, Any]) -> RandomForestRegressor:
    """Build the configured RandomForest regressor."""
    return RandomForestRegressor(
        n_estimators=int(model_cfg.get("n_estimators", 300)),
        random_state=int(model_cfg.get("random_state", 42)),
        max_depth=model_cfg.get("max_depth"),
    )


def train_and_save_model(dataset_path: str | Path) -> TrainingResult:
    """Run the full training workflow and persist artifacts, metrics, and manifests."""
    dataset_df = load_dataset(dataset_path)
    _validate_dataset(dataset_df)

    semantic_scorer = SemanticScorer()
    feature_df = _build_feature_dataframe(dataset_df, semantic_scorer)

    split_cfg = TRAINING_CONFIG["split"]
    processed = preprocess(
        feature_df,
        target_column="target_ratio",
        splits_path=SPLITS_FILE,
        processed_dir=PROCESSED_DATA_DIR,
        train_size=float(split_cfg["train"]),
        val_size=float(split_cfg["validation"]),
        test_size=float(split_cfg["test"]),
        random_state=int(TRAINING_CONFIG["random_state"]),
        feature_columns=["semantic_score", "keyword_score", "length_score"],
    )

    X_train = processed["X_train"]
    X_test = processed["X_test"]
    y_train = np.asarray(processed["y_train"], dtype=float)
    y_test = np.asarray(processed["y_test"], dtype=float)

    run_id = str(uuid.uuid4())
    timestamp = utc_now_iso()
    dataset_summary = summarize_dataset(feature_df, "target_ratio")

    # Build candidate models list (either multi-model with tuning, or single default model)
    candidates_list = []
    model_cfg = TRAINING_CONFIG["model"]
    random_state = int(TRAINING_CONFIG["random_state"])

    if TUNING_CONFIG.get("enabled", False):
        # Multi-model tuning mode
        param_grid = TUNING_CONFIG.get("param_grid", {})
        cv_folds = TUNING_CONFIG.get("cv_folds", 5)
        scoring = TUNING_CONFIG.get("scoring", "r2")
        n_jobs = TUNING_CONFIG.get("n_jobs", -1)

        print("[INFO] Tuning enabled. Building and tuning candidate models...")
        for model_name in param_grid.keys():
            print(f"  [TUNE] Building {model_name}...")
            try:
                base_model = build_model(model_name, random_state=random_state)
                model_param_grid = param_grid[model_name]
                tuned_model, best_params = tune_model(
                    base_model,
                    X_train,
                    y_train,
                    model_param_grid,
                    cv=cv_folds,
                    scoring=scoring,
                    n_jobs=n_jobs,
                )
                y_pred = tuned_model.predict(X_test)
                metrics = regression_metrics(y_test, y_pred)

                candidate = {
                    "model_name": model_name,
                    "hyperparams": best_params,
                    "metrics": {k: round(v, 6) for k, v in metrics.items()},
                    "model": tuned_model,
                }
                candidates_list.append(candidate)
                print(f"    [TUNE] {model_name} R² = {metrics.get('r2', 0):.4f}")
            except Exception as e:
                print(f"    [ERROR] Failed to tune {model_name}: {e}")

        if not candidates_list:
            raise RuntimeError("No candidate models were successfully trained during tuning.")

        # Select best candidate by R² on test set
        best_candidate = max(candidates_list, key=lambda c: c["metrics"].get("r2", 0))
        best_model = best_candidate["model"]
        best_metrics = best_candidate["metrics"]
        best_model_name = best_candidate["model_name"]
        best_hyperparams = best_candidate["hyperparams"]

        print(f"[INFO] Best model: {best_model_name} with R² = {best_metrics.get('r2', 0):.4f}")

    else:
        # Single model mode (backward compatible)
        model = _build_model(model_cfg)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        best_metrics = regression_metrics(y_test, y_pred)
        best_model = model
        best_model_name = str(model_cfg.get("name", "RandomForestRegressor"))
        best_hyperparams = {
            "n_estimators": int(model_cfg.get("n_estimators", 300)),
            "random_state": int(model_cfg.get("random_state", 42)),
            "max_depth": model_cfg.get("max_depth"),
        }

        # Add single model to candidates list for logging
        candidates_list = [
            {
                "model_name": best_model_name,
                "hyperparams": best_hyperparams,
                "metrics": {k: round(v, 6) for k, v in best_metrics.items()},
                "model": best_model,
            }
        ]

    # Save best model artifacts
    model_path = MODELS_DIR / f"model_{_timestamp_slug()}.pkl"
    metadata_path = MODELS_DIR / f"model_{model_path.stem.split('_', 1)[1]}_meta.json"

    artifact = {
        "model": best_model,
        "preprocessor": processed["preprocessor"],
        "feature_names": ["semantic_score", "keyword_score", "length_score"],
    }
    joblib.dump(artifact, model_path)
    joblib.dump(artifact, LEGACY_MODEL_FILE)

    # Log run with candidates
    run_record = {
        "run_id": run_id,
        "timestamp": timestamp,
        "dataset_version": str(TRAINING_CONFIG.get("dataset_version", "v1")),
        "model": best_model_name,
        "hyperparams": best_hyperparams,
        "metrics": best_metrics,
        "test_set_size": int(len(y_test)),
        "candidates": [
            {
                "model_name": c["model_name"],
                "hyperparams": c["hyperparams"],
                "metrics": c["metrics"],
            }
            for c in candidates_list
        ],
    }
    append_run_history(RUN_HISTORY_FILE, run_record)

    manifest_record = {
        "run_id": run_id,
        "timestamp": timestamp,
        "model_path": str(model_path),
        "metrics": best_metrics,
        "dataset_version": run_record["dataset_version"],
    }
    append_manifest_record(MANIFEST_FILE, manifest_record)

    selected = select_best_model(MANIFEST_FILE, BEST_MODEL_FILE)
    if selected is None:
        raise RuntimeError("Could not select best model from manifest after saving run")

    metadata = {
        "run_id": run_id,
        "timestamp": timestamp,
        "dataset_path": str(dataset_path),
        "rows": int(len(feature_df)),
        "train_rows": int(len(processed["train_df"])),
        "validation_rows": int(len(processed["val_df"])),
        "test_rows": int(len(processed["test_df"])),
        "metrics": best_metrics,
        "model_type": best_model_name,
        "dataset_summary": dataset_summary,
        "best_model_path": str(BEST_MODEL_FILE),
        "tuning_enabled": TUNING_CONFIG.get("enabled", False),
        "candidates_count": len(candidates_list),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    LEGACY_MODEL_METADATA_FILE.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return TrainingResult(
        run_id=run_id,
        timestamp=timestamp,
        rows=int(len(feature_df)),
        train_rows=int(len(processed["train_df"])),
        val_rows=int(len(processed["val_df"])),
        test_rows=int(len(processed["test_df"])),
        metrics=best_metrics,
        model_path=str(model_path),
        metadata_path=str(metadata_path),
        best_model_path=str(BEST_MODEL_FILE),
        dataset_summary=dataset_summary,
    )


def compare_training_runs() -> str:
    """Return a formatted run comparison table sorted by R2 descending."""
    return compare_runs(RUN_HISTORY_FILE)
