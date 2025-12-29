"""Recommendation engine using TF-IDF + cosine similarity.

Each recommendation includes:
{ "user_id": "...", "asset_id": "...", "asset_type": "...", "score": float, "reason": "text-based similarity" }
"""
from typing import Dict, List, Any
import logging
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from .config import TOP_N

logger = logging.getLogger(__name__)


def _compute_popularity_score(df, id_col: str) -> Dict[str, float]:
    """Create a simple popularity score when no interaction logs exist.

    Popularity is derived from tag counts and description length.
    """
    scores = {}
    for _, row in df.iterrows():
        tags = str(row.get("tags", ""))
        base = max(1, len([t for t in tags.split(";") if t.strip()]))
        desc_len = len(str(row.get("description", "")))
        scores[row[id_col]] = base + (desc_len / 1000.0)
    # normalize 0..1
    vals = np.array(list(scores.values()))
    if vals.size == 0:
        return {}
    vmin, vmax = vals.min(), vals.max()
    if vmax > vmin:
        vals = (vals - vmin) / (vmax - vmin)
    for i, k in enumerate(scores.keys()):
        scores[k] = float(vals[i])
    return scores


class Recommender:
    def __init__(self, embeddings: Dict[str, Any], datasets: Dict[str, Any]):
        self.embeddings = embeddings
        self.datasets = datasets
        self.popularity = {
            "events": _compute_popularity_score(datasets["events"], "event_id"),
            "jobs": _compute_popularity_score(datasets["jobs"], "job_id"),
            "posts": _compute_popularity_score(datasets["posts"], "post_id"),
        }

    def recommend_for_user(self, user_id: str, asset_type: str, top_n: int = TOP_N) -> List[Dict[str, Any]]:
        if user_id not in self.embeddings["users"]["ids"]:
            raise KeyError(f"Unknown user {user_id}")
        uidx = self.embeddings["users"]["ids"].index(user_id)
        user_vec = self.embeddings["users"]["vectors"][uidx]

        entity_matrix = self.embeddings[asset_type]["vectors"]
        entity_ids = self.embeddings[asset_type]["ids"]

        # cold start if the user vector is all zeros
        if getattr(user_vec, "nnz", None) == 0:
            scores = np.array([self.popularity[asset_type].get(eid, 0.0) for eid in entity_ids])
            reason = "popularity fallback"
        else:
            sim = cosine_similarity(user_vec, entity_matrix).ravel()
            scores = sim
            reason = "text-based similarity"

        # apply simple boosts for city/category/tag overlap
        user_row = self.datasets["users"].query("user_id == @user_id").iloc[0]
        user_city = str(user_row.get("city", "")).lower()
        user_interests = str(user_row.get("interests", "")).lower()

        boosts = np.zeros_like(scores)
        for i, eid in enumerate(entity_ids):
            # find candidate row
            row = None
            if asset_type == "events":
                row = self.datasets["events"].query("event_id == @eid").iloc[0]
            elif asset_type == "jobs":
                row = self.datasets["jobs"].query("job_id == @eid").iloc[0]
            elif asset_type == "posts":
                row = self.datasets["posts"].query("post_id == @eid").iloc[0]

            cat = str(row.get("category", "")).lower()
            city = str(row.get("city", "")).lower()
            tags = str(row.get("tags", "")).lower()

            if user_city and user_city == city:
                boosts[i] += 0.1
            # category keyword overlap
            if any(k for k in cat.split() if k in user_interests):
                boosts[i] += 0.05
            # tag overlap
            if any(t.strip() in user_interests for t in tags.split(";")):
                boosts[i] += 0.03

        final_scores = scores + boosts

        order = np.argsort(-final_scores)
        recs = []
        for idx in order[:top_n]:
            recs.append({
                "user_id": user_id,
                "asset_id": entity_ids[idx],
                "asset_type": asset_type[:-1],
                "score": float(final_scores[idx]),
                "reason": reason,
            })
        return recs

    def generate_all_recommendations(self, top_n: int = TOP_N) -> Dict[str, List[Dict[str, Any]]]:
        all_recs = {"events": [], "jobs": [], "posts": []}
        for uid in self.embeddings["users"]["ids"]:
            for asset_type in all_recs.keys():
                recs = self.recommend_for_user(uid, asset_type, top_n=top_n)
                all_recs[asset_type].extend(recs)
        return all_recs


if __name__ == "__main__":
    import logging
    from load_data import load_all
    from feature_builder import build_embeddings

    logging.basicConfig(level=logging.INFO)
    data = load_all()
    embeddings, _ = build_embeddings(data)
    r = Recommender(embeddings, data)
    print(r.recommend_for_user(embeddings["users"]["ids"][0], "events", top_n=5))
