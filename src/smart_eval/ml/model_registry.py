"""Model artifact manifest and best-model selection helpers."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict, Optional


def load_manifest(manifest_path: str | Path) -> Dict[str, Any]:
    """Load model manifest from disk; return empty dict if not found."""
    path = Path(manifest_path)
    if not path.exists():
        return {
            "last_dataset_hash": "",
            "last_updated": "",
            "best_model": "",
            "best_r2": 0.0,
            "models": [],
        }
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        return payload
    if isinstance(payload, list):
        return {
            "last_dataset_hash": "",
            "last_updated": "",
            "best_model": "",
            "best_r2": 0.0,
            "models": payload,
        }
    raise ValueError(f"Manifest file has invalid format: {path}")


def append_manifest_record(manifest_path: str | Path, record: Dict[str, Any]) -> None:
    """Append one model record to manifest models list."""
    path = Path(manifest_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(path)
    if "models" not in manifest:
        manifest["models"] = []
    manifest["models"].append(record)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def update_manifest_hash_and_best(
    manifest_path: str | Path,
    dataset_hash: str,
    timestamp: str,
    best_model_path: str,
    best_r2: float,
) -> None:
    """Update manifest with new dataset hash, timestamp, and best model info."""
    path = Path(manifest_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(path)
    manifest["last_dataset_hash"] = dataset_hash
    manifest["last_updated"] = timestamp
    manifest["best_model"] = best_model_path
    manifest["best_r2"] = round(best_r2, 6)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def get_last_dataset_hash(manifest_path: str | Path) -> str:
    """Get last recorded dataset hash from manifest."""
    manifest = load_manifest(manifest_path)
    return manifest.get("last_dataset_hash", "")


def select_best_model(manifest_path: str | Path, best_model_path: str | Path) -> Optional[Path]:
    """Select the model with highest R2 and mirror it to best_model.pkl."""
    manifest = load_manifest(manifest_path)
    models = manifest.get("models", [])
    if not models:
        return None

    best = max(models, key=lambda row: float(row.get("metrics", {}).get("r2", float("-inf"))))
    model_path = Path(best["model_path"])
    if not model_path.exists():
        return None

    best_target = Path(best_model_path)
    best_target.parent.mkdir(parents=True, exist_ok=True)

    try:
        if best_target.exists() or best_target.is_symlink():
            best_target.unlink()
        best_target.symlink_to(model_path)
    except Exception:
        shutil.copy2(model_path, best_target)

    return model_path
