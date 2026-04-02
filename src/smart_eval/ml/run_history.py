"""Run history persistence and comparison utilities."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


def append_run_history(history_path: str | Path, run_record: Dict[str, Any]) -> None:
    """Append one run record to logs/run_history.json."""
    path = Path(history_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    runs = load_run_history(path)
    runs.append(run_record)
    path.write_text(json.dumps(runs, indent=2), encoding="utf-8")


def load_run_history(history_path: str | Path) -> List[Dict[str, Any]]:
    """Load run history records; return empty list when file is absent."""
    path = Path(history_path)
    if not path.exists():
        return []

    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    raise ValueError(f"Run history file has invalid format: {path}")


def compare_runs(history_path: str | Path) -> str:
    """Return a table of runs sorted by R2 descending."""
    runs = load_run_history(history_path)
    if not runs:
        return "No runs found."

    sorted_runs = sorted(runs, key=lambda x: float(x.get("metrics", {}).get("r2", float("-inf"))), reverse=True)

    header = (
        f"{'run_id':<36}  {'timestamp':<25}  {'model':<18}  "
        f"{'dataset_version':<10}  {'MAE':>8}  {'R2':>8}  {'RMSE':>8}  {'MAPE':>8}"
    )
    sep = "-" * len(header)

    rows = [header, sep]
    for run in sorted_runs:
        metrics = run.get("metrics", {})
        rows.append(
            f"{str(run.get('run_id', '')):<36}  "
            f"{str(run.get('timestamp', '')):<25}  "
            f"{str(run.get('model', '')):<18}  "
            f"{str(run.get('dataset_version', '')):<10}  "
            f"{float(metrics.get('mae', 0.0)):>8.4f}  "
            f"{float(metrics.get('r2', 0.0)):>8.4f}  "
            f"{float(metrics.get('rmse', 0.0)):>8.4f}  "
            f"{float(metrics.get('mape', 0.0)):>8.4f}"
        )
    return "\n".join(rows)


def utc_now_iso() -> str:
    """Return UTC timestamp string in ISO format seconds precision."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
