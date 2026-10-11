"""
tests/test_paths.py — Lightweight path and configuration tests
==============================================================
Verifies that the data directory is configured correctly and required
files are accessible before any modelling or API test runs.
"""

import os
import pytest
from pathlib import Path
from kestrel.data_paths import get_data_dir, find_data_file


def test_data_dir_exists():
    """Data directory must be resolvable and must exist."""
    d = get_data_dir()
    assert d.is_dir(), f"Data directory not found: {d}"


def test_required_data_files_present():
    """All five required source files must be present in the data directory."""
    required = [
        "train.csv",
        "test_unlabelled.csv",
        "sample_submission.csv",
        "customers.csv",
        "products.csv",
    ]
    d = get_data_dir()
    missing = [f for f in required if not (d / f).exists()]
    assert not missing, (
        f"Missing data files in '{d}': {missing}\n"
        "Copy or symlink them from the original task data folder."
    )


def test_find_data_file_returns_path():
    """find_data_file should return a Path pointing to an existing file."""
    p = find_data_file("products.csv")
    assert isinstance(p, Path)
    assert p.exists()
    assert p.stat().st_size > 0


def test_find_data_file_raises_on_missing():
    """find_data_file must raise FileNotFoundError for a non-existent file."""
    with pytest.raises(FileNotFoundError):
        find_data_file("this_file_does_not_exist_xyz.csv")


def test_artifacts_present():
    """artifacts/model.joblib and artifacts/metadata.json must exist after training."""
    repo_root = Path(__file__).parent.parent
    assert (repo_root / "artifacts" / "model.joblib").exists(), (
        "artifacts/model.joblib not found. Run 'py train.py' first."
    )
    assert (repo_root / "artifacts" / "metadata.json").exists(), (
        "artifacts/metadata.json not found. Run 'py train.py' first."
    )
