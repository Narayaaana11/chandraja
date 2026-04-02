"""Unit tests for dataset hash-based change detection."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data.load_dataset import compute_dataset_hash


class TestDatasetHash:
    """Test suite for compute_dataset_hash function."""

    def test_hash_consistency(self) -> None:
        """Hashing the same file twice should produce the same result."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("col1,col2\n1,2\n3,4\n")
            temp_path = f.name

        try:
            hash1 = compute_dataset_hash(temp_path)
            hash2 = compute_dataset_hash(temp_path)
            assert hash1 == hash2, "Same file should produce identical hashes"
        finally:
            Path(temp_path).unlink()

    def test_hash_differs_after_modification(self) -> None:
        """Modifying file content should produce a different hash."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("col1,col2\n1,2\n3,4\n")
            temp_path = f.name

        try:
            hash1 = compute_dataset_hash(temp_path)

            Path(temp_path).write_text("col1,col2\n1,2\n3,4\n5,6\n", encoding="utf-8")
            hash2 = compute_dataset_hash(temp_path)

            assert hash1 != hash2, "Modified file should produce different hash"
        finally:
            Path(temp_path).unlink()

    def test_hash_with_nonexistent_file(self) -> None:
        """Hashing a nonexistent file should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            compute_dataset_hash("/nonexistent/path/to/file.csv")

    def test_hash_large_file(self) -> None:
        """Hash function should handle large files with chunked reading."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            for i in range(1000):
                f.write(f"{i},{i*2}\n")
            temp_path = f.name

        try:
            hash1 = compute_dataset_hash(temp_path)
            hash2 = compute_dataset_hash(temp_path)
            assert hash1 == hash2, "Large file hashes should be consistent"
            assert len(hash1) == 32, "MD5 hex digest should be 32 characters"
        finally:
            Path(temp_path).unlink()

    def test_hash_json_file(self) -> None:
        """Hash function should work with JSON files."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write('{"key": "value"}')
            temp_path = f.name

        try:
            hash1 = compute_dataset_hash(temp_path)
            assert isinstance(hash1, str) and len(hash1) == 32
        finally:
            Path(temp_path).unlink()

    def test_hash_jsonl_file(self) -> None:
        """Hash function should work with JSONL files."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write('{"id": 1}\n{"id": 2}\n')
            temp_path = f.name

        try:
            hash1 = compute_dataset_hash(temp_path)
            assert isinstance(hash1, str) and len(hash1) == 32
        finally:
            Path(temp_path).unlink()
