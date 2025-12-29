# Recommendation Logic 🔧

## Similarity Computation
- Users and assets are represented as TF-IDF vectors built from textual fields (title, description, degree, interests, etc.).
- Vectors are L2-normalized so cosine similarity = dot product.
- For a given user, cosine similarity is computed between the user vector and every asset vector; results are ranked descending.

## Ranking & Reasoning
- If the user profile contains signal (non-empty vector), ranking is dominated by cosine similarity ("text-based similarity").
- Small, explainable boosts are added for:
  - City match (+0.1)
  - Category keyword overlap (+0.05)
  - Tag overlap (+0.03)
- In case of a cold start (user vector is empty or trivial), we use a popularity fallback computed from attributes such as tag counts and description length.

## Cold Start Handling
- Popularity fallback provides reasonable defaults when no behavioral logs exist.
- Category and city boosts help surface locally relevant or topically relevant items.
- This is intentionally simple and explainable; in production, add recent trending score and collaborative signals.

## Active Learning Simulation
- `src/active_learning.py` simulates light feedback (like/dislike).
- Positive feedback adds a configurable additive boost to the score; negative feedback applies a small penalty.
- The simulation is deterministic in logic but uses random sampling to create a realistic distribution of feedback for demonstration.

---

This logic favors explainability and is intentionally modular to allow later upgrades (real logs, neural embeddings, per-user calibration).