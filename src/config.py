from pathlib import Path

# Project paths
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "output"
REPORTS_DIR = ROOT / "reports"

# TF-IDF configuration
TFIDF = {
    "max_features": 5000,
    "ngram_range": (1, 2),
    "min_df": 1
}

# Recommendation configuration
TOP_N = 10
SIMILARITY_BATCH = 1024

# Random seed for reproducibility
RANDOM_SEED = 42

# Required columns per dataset (used for validation)
REQUIRED_COLUMNS = {
    "users": ["user_id", "university", "degree", "interests", "bio", "city"],
    "events": ["event_id", "title", "description", "category", "tags", "city"],
    "jobs": ["job_id", "title", "description", "category", "tags", "city", "company"],
    "posts": ["post_id", "title", "content", "category", "tags", "city"]
}
