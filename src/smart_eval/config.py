"""Application configuration loaded from config.yaml with safe defaults."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def _deep_update(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
	"""Merge nested dictionaries without dropping missing default keys."""
	merged = dict(base)
	for key, value in overrides.items():
		if isinstance(value, dict) and isinstance(merged.get(key), dict):
			merged[key] = _deep_update(merged[key], value)
		else:
			merged[key] = value
	return merged


def _default_config() -> Dict[str, Any]:
	"""Return default config values used when config.yaml is missing fields."""
	return {
		"paths": {
			"raw_data_dir": "data/raw",
			"processed_data_dir": "data/processed",
			"splits_file": "data/splits.json",
			"models_dir": "models",
			"best_model_path": "models/best_model.pkl",
			"manifest_path": "models/manifest.json",
			"run_history_path": "logs/run_history.json",
			"legacy_model_path": "model_artifacts/score_model.joblib",
			"legacy_metadata_path": "model_artifacts/score_model_meta.json",
		},
		"training": {
			"dataset_version": "v1",
			"random_state": 42,
			"split": {
				"train": 0.70,
				"validation": 0.15,
				"test": 0.15,
			},
			"model": {
				"name": "RandomForestRegressor",
				"n_estimators": 300,
				"random_state": 42,
				"max_depth": None,
			},
		},
		"tuning": {
			"enabled": False,
			"cv_folds": 5,
			"scoring": "r2",
			"n_jobs": -1,
			"param_grid": {
				"RandomForest": {
					"n_estimators": [50, 100, 200],
					"max_depth": [None, 5, 10, 20],
					"min_samples_split": [2, 5, 10],
				},
				"Ridge": {
					"alpha": [0.01, 0.1, 1.0, 10.0, 100.0],
				},
				"XGBoost": {
					"n_estimators": [50, 100, 200],
					"max_depth": [3, 5, 7],
					"learning_rate": [0.01, 0.1, 0.2],
					"subsample": [0.8, 1.0],
				},
			},
		},
	}


def load_config() -> Dict[str, Any]:
	"""Load YAML config and merge with defaults."""
	config_file = Path(__file__).resolve().parent.parent.parent / "config.yaml"
	if config_file.exists():
		with open(config_file, encoding="utf-8") as f:
			yaml_config = yaml.safe_load(f) or {}
		return _deep_update(_default_config(), yaml_config)
	return _default_config()


ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
UPLOAD_DIR = ROOT_DIR / "uploads"

CONFIG = load_config()

RAW_DATA_DIR = ROOT_DIR / CONFIG["paths"]["raw_data_dir"]
PROCESSED_DATA_DIR = ROOT_DIR / CONFIG["paths"]["processed_data_dir"]
SPLITS_FILE = ROOT_DIR / CONFIG["paths"]["splits_file"]

MODELS_DIR = ROOT_DIR / CONFIG["paths"]["models_dir"]
BEST_MODEL_FILE = ROOT_DIR / CONFIG["paths"]["best_model_path"]
MANIFEST_FILE = ROOT_DIR / CONFIG["paths"]["manifest_path"]
RUN_HISTORY_FILE = ROOT_DIR / CONFIG["paths"]["run_history_path"]

LEGACY_MODEL_FILE = ROOT_DIR / CONFIG["paths"]["legacy_model_path"]
LEGACY_MODEL_METADATA_FILE = ROOT_DIR / CONFIG["paths"]["legacy_metadata_path"]

TRAINING_CONFIG = CONFIG["training"]
TUNING_CONFIG = CONFIG.get("tuning", {})

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
RUN_HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
