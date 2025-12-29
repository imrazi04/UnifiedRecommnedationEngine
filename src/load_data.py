"""Data loading utilities for the recommendation prototype.

- Loads CSVs from `data/`
- Maps common alternate column names to canonical schema
- Validates and fills missing columns safely
- Returns pandas DataFrames ready for feature building
"""
from pathlib import Path
from typing import Dict
import logging

import pandas as pd

from .config import DATA_DIR, REQUIRED_COLUMNS

logger = logging.getLogger(__name__)

# Column alias mappings: map various source column names to the canonical names used in feature builder
_COLUMN_ALIASES = {
    "users": {
        "degree_program": "degree",
        "exams_subjects": "interests",
        "bio": "bio",
        "university": "university",
        "city": "city",
        "user_id": "user_id",
        # some datasets have 'interests' already
    },
    "events": {
        "title": "title",
        "description": "description",
        "category": "category",
        "tags": "tags",
        "city": "city",
        "event_id": "event_id",
    },
    "jobs": {
        "title": "title",
        "description": "description",
        "category": "category",
        "tags": "tags",
        "city": "city",
        "company": "company",
        "job_id": "job_id",
    },
    "posts": {
        "title": "title",
        "content": "content",
        "category": "category",
        "tags": "tags",
        "city": "city",
        "post_id": "post_id",
    },
}


def _map_columns(df: pd.DataFrame, aliases: Dict[str, str]) -> pd.DataFrame:
    """Rename dataframe columns using a set of aliases (left -> right)."""
    rename = {src: dst for src, dst in aliases.items() if src in df.columns and src != dst}
    if rename:
        df = df.rename(columns=rename)
    return df


def _ensure_columns(df: pd.DataFrame, required: list, id_col: str) -> pd.DataFrame:
    """Ensure all required columns exist; add defaults for missing ones."""
    for col in required:
        if col not in df.columns:
            # default empty string for text/categorical fields
            df[col] = ""
    # drop rows missing id
    df = df[df[id_col].notna()].copy()
    return df


def _safe_fill(df: pd.DataFrame, text_cols: list, categorical_cols: list) -> pd.DataFrame:
    # For text columns, replace NaN with empty string
    for c in text_cols:
        if c in df.columns:
            df[c] = df[c].fillna("")
    # For categorical columns, replace NaN with 'unknown'
    for c in categorical_cols:
        if c in df.columns:
            df[c] = df[c].fillna("unknown")
    return df


def load_csv(path: Path, name: str, required_cols: list) -> pd.DataFrame:
    logger.info(f"Loading {name} from {path}")

    # Try reading the CSV with several common encodings to handle files produced on different platforms
    df = None
    encodings_to_try = ["utf-8", "cp1252", "latin-1"]
    for enc in encodings_to_try:
        try:
            df = pd.read_csv(path, encoding=enc)
            if enc != "utf-8":
                logger.warning("Read %s with encoding %s (fallback)", path.name, enc)
            else:
                logger.debug("Read %s with encoding %s", path.name, enc)
            break
        except UnicodeDecodeError:
            logger.debug("Failed to read %s with encoding %s", path.name, enc)
            continue
    if df is None:
        # Last resort: try with latin-1 and replace errors
        try:
            df = pd.read_csv(path, encoding="latin-1")
            logger.warning("Read %s with latin-1 as last resort", path.name)
        except Exception as e:
            logger.exception("Unable to read %s: %s", path.name, e)
            raise

    # Map known aliases if present
    aliases = _COLUMN_ALIASES.get(name, {})
    df = _map_columns(df, aliases)

    # Use id_col defined by the required schema
    id_col = required_cols[0]
    df = _ensure_columns(df, required_cols, id_col)

    # Identify text vs categorical columns (simple heuristic)
    text_cols = [c for c in required_cols if c in ["title", "description", "content", "bio", "interests", "tags"]]
    categorical_cols = [c for c in required_cols if c not in text_cols and c != id_col]

    df = _safe_fill(df, text_cols, categorical_cols)
    df = df.reset_index(drop=True)
    logger.info(f"Loaded {len(df)} rows for {name} (columns: {list(df.columns)})")
    return df


def load_all(data_dir: Path = DATA_DIR) -> Dict[str, pd.DataFrame]:
    """Load all datasets and return as dict.

    Returns dict with keys: users, events, jobs, posts
    """
    datasets = {}
    files = {
        "users": data_dir / "users.csv",
        "events": data_dir / "events.csv",
        "jobs": data_dir / "jobs.csv",
        "posts": data_dir / "posts.csv",
    }

    for key, path in files.items():
        if not path.exists():
            raise FileNotFoundError(f"Expected data file not found: {path}")
        datasets[key] = load_csv(path, key, REQUIRED_COLUMNS.get(key, []))

    return datasets


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    d = load_all()
    for k, df in d.items():
        print(k, df.shape)
