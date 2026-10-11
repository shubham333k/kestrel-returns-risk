"""
kestrel/data_paths.py — Centralised data-location convention
=============================================================
Resolution order for source data files:
  1. KESTREL_DATA_DIR environment variable (absolute path to the folder
     containing train.csv, test_unlabelled.csv, products.csv, etc.)
  2. <repo_root>/data/  (the documented local data directory)

Source files must NOT be committed to Git (see .gitignore).
Run:
    Windows PowerShell:
        $env:KESTREL_DATA_DIR = "C:\\path\\to\\your\\data"
    macOS / Linux:
        export KESTREL_DATA_DIR="/path/to/your/data"
"""

import os
from pathlib import Path


def get_data_dir() -> Path:
    """Return the configured data directory, raising a clear error if missing."""
    env_val = os.environ.get("KESTREL_DATA_DIR", "").strip()
    if env_val:
        d = Path(env_val)
        if not d.is_dir():
            raise FileNotFoundError(
                f"KESTREL_DATA_DIR is set to '{d}' but that directory does not exist."
            )
        return d

    # Default: <repo_root>/data/
    repo_root = Path(__file__).parent.parent
    d = repo_root / "data"
    if not d.is_dir():
        raise FileNotFoundError(
            f"Data directory not found at '{d}'.\n"
            "Either:\n"
            "  1. Create the data/ folder and place the source CSV files inside it, OR\n"
            "  2. Set the KESTREL_DATA_DIR environment variable to your data folder path.\n"
            "\n"
            "Expected files:\n"
            "  train.csv, test_unlabelled.csv, sample_submission.csv,\n"
            "  customers.csv, products.csv\n"
            "\n"
            "See README.md for full setup instructions."
        )
    return d


def find_data_file(filename: str) -> Path:
    """Return path to a named data file, raising a clear error if not found."""
    d = get_data_dir()
    p = d / filename
    if not p.exists():
        raise FileNotFoundError(
            f"Required data file '{filename}' not found in '{d}'.\n"
            "Place the file there and retry."
        )
    return p
