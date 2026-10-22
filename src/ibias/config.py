"""Paths and settings shared by every step of the pipeline.

Everything the pipeline writes goes under DATA_DIR, which is gitignored:
    data/raw/<outlet>/YYYY/MM/DD.jsonl.gz   untouched API responses (source of truth)
    data/ibias.duckdb                       cleaned articles, queryable with SQL
    data/features/                          derived tables such as word counts (Parquet)
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# The project root is three levels up from this file: src/ibias/config.py
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# load_dotenv reads KEY=value lines from .env into os.environ,
# so secrets stay out of the code.
load_dotenv(PROJECT_ROOT / ".env")

# IBIAS_DATA_DIR lets tests (or you) point the pipeline at another folder.
DATA_DIR = Path(os.environ.get("IBIAS_DATA_DIR", PROJECT_ROOT / "data"))
RAW_DIR = DATA_DIR / "raw"
DB_PATH = DATA_DIR / "ibias.duckdb"
FEATURES_DIR = DATA_DIR / "features"


def guardian_api_key() -> str:
    key = os.environ.get("GUARDIAN_API_KEY", "").strip()
    if not key:
        raise SystemExit(
            "GUARDIAN_API_KEY is not set. Copy .env.example to .env and paste your key in."
        )
    return key
