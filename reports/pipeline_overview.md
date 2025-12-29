# Pipeline Overview ✅

This document explains the end-to-end flow for the recommendation prototype.

## Data Flow
- CSV files are placed in `data/`.
- `src/load_data.py` loads each CSV, maps alternate column names to canonical fields, fills missing values, and returns pandas DataFrames.
- `src/feature_builder.py` constructs textual profiles for users and assets and produces TF-IDF embeddings using a shared vocabulary.
- `src/recommender.py` computes cosine similarity between user vectors and asset vectors to produce Top-N ranked recommendations.
- `src/active_learning.py` simulates light-weight feedback to adjust scores for demonstration.
- `src/main.py` orchestrates the above steps and writes JSON output into `output/`.

## Architecture
- Lightweight, modular Python code using pandas + scikit-learn.
- TF-IDF vectors stored in memory as sparse matrices; cosine similarity is computed via efficient matrix operations.
- Designed for clarity and easy extension to production (e.g., swap TF-IDF for neural embeddings or persist vectors in a vector DB).

## Why TF-IDF + Cosine?
- TF-IDF provides transparent, explainable text representations.
- Cosine similarity is a natural metric for comparing normalized TF-IDF vectors and is fast to compute for moderate datasets.

## How it scales later
- Vector storage can be migrated to a specialized store (FAISS, Milvus).
- Batch similarity computations can be offloaded to distributed workers.
- Real interaction logs can replace static popularity heuristics for better personalization.

---

**Note:** See `reports/recommendation_logic.md` for details on ranking and cold-start heuristics.