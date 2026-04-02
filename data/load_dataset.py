"""Dataset loading and preprocessing pipeline for Smart Eval."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def compute_dataset_hash(filepath: str | Path) -> str:
    """Compute MD5 hash of dataset file for change detection.
    
    Args:
        filepath: Path to CSV, JSON, or JSONL file.
        
    Returns:
        Hex digest of MD5 hash of file contents.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    
    md5_hash = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(8192)
            if not chunk:
                break
            md5_hash.update(chunk)
    
    return md5_hash.hexdigest()


DEFAULT_TEXT_COLUMNS = {"student_text", "reference_text", "keywords", "text"}


def load_dataset(source: str | Path) -> pd.DataFrame:
    """Load dataset from CSV, JSON, or a folder containing labeled files."""
    source_path = Path(source)
    if not source_path.exists():
        raise FileNotFoundError(f"Dataset source does not exist: {source_path}")

    if source_path.is_dir():
        return _load_from_labeled_folder(source_path)

    suffix = source_path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(source_path)
    if suffix in {".json", ".jsonl"}:
        return _load_from_json(source_path)

    raise ValueError(f"Unsupported dataset source type: {source_path}")


def _load_from_json(path: Path) -> pd.DataFrame:
    """Load JSON array or JSONL dataset into a dataframe."""
    if path.suffix.lower() == ".jsonl":
        return pd.read_json(path, lines=True)

    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return pd.DataFrame(payload)
    if isinstance(payload, dict):
        if "records" in payload and isinstance(payload["records"], list):
            return pd.DataFrame(payload["records"])
        return pd.DataFrame([payload])
    raise ValueError(f"Unsupported JSON payload format in {path}")


def _iter_labeled_files(root_dir: Path) -> Iterable[tuple[str, Path]]:
    """Yield label and file path pairs from folder structure label_name/file.txt."""
    for label_dir in root_dir.iterdir():
        if not label_dir.is_dir():
            continue
        label = label_dir.name
        for file_path in label_dir.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in {".txt", ".md"}:
                yield label, file_path


def _load_from_labeled_folder(root_dir: Path) -> pd.DataFrame:
    """Load a folder dataset where each subfolder name is the class label."""
    records: List[Dict[str, Any]] = []
    for label, file_path in _iter_labeled_files(root_dir):
        records.append(
            {
                "text": file_path.read_text(encoding="utf-8", errors="ignore"),
                "label": label,
                "source_file": str(file_path),
            }
        )

    if not records:
        raise ValueError(f"No labeled .txt/.md files found under: {root_dir}")
    return pd.DataFrame(records)


def preprocess(
    df: pd.DataFrame,
    *,
    target_column: str,
    splits_path: str | Path,
    processed_dir: str | Path,
    train_size: float = 0.70,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
    feature_columns: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Clean data, encode/normalize features, split train/val/test, and persist split indices."""
    if abs((train_size + val_size + test_size) - 1.0) > 1e-9:
        raise ValueError("train_size + val_size + test_size must sum to 1.0")

    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset")

    clean_df = df.copy().reset_index(drop=True)
    clean_df = clean_df.drop_duplicates().reset_index(drop=True)

    numeric_cols = clean_df.select_dtypes(include=["number", "bool"]).columns.tolist()
    object_cols = clean_df.select_dtypes(include=["object", "string", "category"]).columns.tolist()

    for col in numeric_cols:
        if col == target_column:
            continue
        clean_df[col] = clean_df[col].fillna(clean_df[col].median())

    for col in object_cols:
        clean_df[col] = clean_df[col].fillna("").astype(str).str.strip()

    if feature_columns is None:
        feature_columns = [c for c in clean_df.columns if c != target_column]

    feature_columns = [c for c in feature_columns if c in clean_df.columns and c != target_column]
    if not feature_columns:
        raise ValueError("No feature columns available after preprocessing")

    train_idx, temp_idx = train_test_split(
        clean_df.index.to_list(),
        test_size=(1.0 - train_size),
        random_state=random_state,
        shuffle=True,
    )

    relative_test_size = test_size / (val_size + test_size)
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=relative_test_size,
        random_state=random_state,
        shuffle=True,
    )

    splits_payload = {
        "random_state": random_state,
        "ratios": {
            "train": train_size,
            "validation": val_size,
            "test": test_size,
        },
        "indices": {
            "train": sorted(int(i) for i in train_idx),
            "validation": sorted(int(i) for i in val_idx),
            "test": sorted(int(i) for i in test_idx),
        },
    }

    splits_path = Path(splits_path)
    splits_path.parent.mkdir(parents=True, exist_ok=True)
    splits_path.write_text(json.dumps(splits_payload, indent=2), encoding="utf-8")

    train_df = clean_df.loc[train_idx].copy()
    val_df = clean_df.loc[val_idx].copy()
    test_df = clean_df.loc[test_idx].copy()

    categorical_features = [
        col for col in feature_columns if col in object_cols and col not in DEFAULT_TEXT_COLUMNS
    ]
    numeric_features = [col for col in feature_columns if col not in categorical_features]

    if not numeric_features and not categorical_features:
        raise ValueError("Could not infer any preprocessable feature columns")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ],
        remainder="drop",
    )

    X_train = preprocessor.fit_transform(train_df[feature_columns])
    X_val = preprocessor.transform(val_df[feature_columns])
    X_test = preprocessor.transform(test_df[feature_columns])

    y_train = train_df[target_column].to_numpy()
    y_val = val_df[target_column].to_numpy()
    y_test = test_df[target_column].to_numpy()

    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(processed_dir / "train.csv", index=False)
    val_df.to_csv(processed_dir / "validation.csv", index=False)
    test_df.to_csv(processed_dir / "test.csv", index=False)

    return {
        "clean_df": clean_df,
        "feature_columns": feature_columns,
        "target_column": target_column,
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "preprocessor": preprocessor,
        "split_indices": splits_payload["indices"],
    }


def summarize_dataset(df: pd.DataFrame, target_column: str) -> Dict[str, Any]:
    """Build printable summary including size, feature count, and label distribution."""
    feature_count = max(len(df.columns) - 1, 0)
    if target_column not in df.columns:
        label_distribution: Dict[str, Any] = {}
    else:
        target_series = df[target_column]
        if pd.api.types.is_numeric_dtype(target_series) and target_series.nunique(dropna=True) > 20:
            bins = pd.cut(target_series, bins=10, include_lowest=True)
            label_distribution = bins.value_counts(dropna=False, sort=False).astype(int).to_dict()
            label_distribution = {str(k): int(v) for k, v in label_distribution.items()}
        else:
            label_distribution = target_series.value_counts(dropna=False).to_dict()

    return {
        "dataset_size": int(len(df)),
        "feature_count": int(feature_count),
        "label_distribution": label_distribution,
    }
