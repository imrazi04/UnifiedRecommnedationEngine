"""Orchestration script to run the entire pipeline end-to-end.

Steps:
1. Load data
2. Build embeddings using TF-IDF
3. Generate recommendations (User -> Events/Jobs/Posts)
4. Simulate active learning feedback and adjust scores
5. Save outputs to JSON files in `output/`
"""
import logging
import json
from pathlib import Path

from .config import OUTPUT_DIR, TOP_N
from .load_data import load_all
from .feature_builder import build_embeddings
from .recommender import Recommender
from .active_learning import ActiveLearner

logger = logging.getLogger(__name__)


def save_recommendations(recs: list, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(recs, f, indent=2, ensure_ascii=False)
    logger.info("Wrote %d recommendations to %s", len(recs), path)


def run():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logger.info("Starting pipeline")

    # 1. Load
    datasets = load_all()

    # 2. Build embeddings
    embeddings, vectorizer = build_embeddings(datasets)

    # 3. Generate recommendations
    rec_engine = Recommender(embeddings, datasets)
    all_recs = rec_engine.generate_all_recommendations(top_n=TOP_N)

    # 4. Simulate active learning
    # Flatten recommendations for sampling
    flat_recs = []
    for k, v in all_recs.items():
        flat_recs.extend(v)

    learner = ActiveLearner()
    feedback = learner.simulate_feedback(flat_recs, positive_ratio=0.03, negative_ratio=0.01)

    # Apply feedback per asset type
    adjusted = {}
    for asset_type, recs in all_recs.items():
        adjusted_recs = learner.apply_feedback(recs, feedback)
        adjusted[asset_type] = adjusted_recs

    # 5. Save outputs
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save flat asset-type recommendations
    save_recommendations(adjusted.get("events", []), OUTPUT_DIR / "event_recommendations.json")
    save_recommendations(adjusted.get("jobs", []), OUTPUT_DIR / "job_recommendations.json")

    # Build a per-user aggregated recommendations file (events, jobs, posts)
    per_user = {uid: {"events": [], "jobs": [], "posts": []} for uid in embeddings["users"]["ids"]}
    for asset_type, recs in adjusted.items():
        for rec in recs:
            uid = rec["user_id"]
            if uid not in per_user:
                per_user[uid] = {"events": [], "jobs": [], "posts": []}
            per_user[uid].setdefault(asset_type, []).append(rec)

    # Optionally convert to list for JSON serializability and easier consumption
    per_user_list = [{"user_id": uid, "recommendations": per_user[uid]} for uid in per_user]
    save_recommendations(per_user_list, OUTPUT_DIR / "user_recommendations.json")

    logger.info("Pipeline completed")


if __name__ == "__main__":
    run()