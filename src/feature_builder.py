"""Feature engineering: build TF-IDF embeddings for users and assets.

- Build textual profiles for users and assets
- Fit a shared TfidfVectorizer where possible
- Return normalized TF-IDF matrices per entity
"""
from typing import Dict, Tuple
import logging

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from .config import TFIDF

logger = logging.getLogger(__name__)


def build_user_profile_text(users: pd.DataFrame) -> pd.Series:
    """Combine user fields into a profile text.

    We are defensive: use available fields and join them into a single string.
    """
    parts = []
    ids = []
    for _, row in users.iterrows():
        university = str(row.get("university", ""))
        degree = str(row.get("degree", ""))
        interests = str(row.get("interests", ""))
        bio = str(row.get("bio", ""))
        city = str(row.get("city", ""))
        # create concise profile
        text = " ".join([university, degree, interests, bio, city])
        parts.append(text)
        ids.append(row["user_id"])
    return pd.Series(parts, index=ids)


def build_asset_text(df: pd.DataFrame, id_col: str, fields: list) -> pd.Series:
    """Build content text for assets (events/jobs/posts)."""
    texts = []
    ids = []
    for _, row in df.iterrows():
        values = [str(row.get(f, "")) for f in fields]
        texts.append(" ".join(values))
        ids.append(row[id_col])
    return pd.Series(texts, index=ids)


def fit_shared_vectorizer(corpus_texts: pd.Series, max_features: int = None) -> TfidfVectorizer:
    """Fit TF-IDF on all texts to create a shared vocabulary."""
    max_features = max_features if max_features is not None else TFIDF.get("max_features")
    logger.info("Fitting shared TF-IDF vectorizer on combined corpus")
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=TFIDF.get("ngram_range"), min_df=TFIDF.get("min_df"))
    vectorizer.fit(corpus_texts.values)
    logger.info("Vocabulary size: %d", len(vectorizer.vocabulary_))
    return vectorizer


def transform_texts(vectorizer: TfidfVectorizer, texts: pd.Series):
    """Transform texts to normalized TF-IDF vectors."""
    tfidf = vectorizer.transform(texts.values)
    # L2 normalize rows for cosine similarity
    tfidf = normalize(tfidf, norm="l2", axis=1)
    return tfidf


def build_embeddings(datasets: Dict[str, pd.DataFrame]):
    """Build text profiles and TF-IDF embeddings for all entities.

    Returns:
        embeddings: dict with keys users, events, jobs, posts and values dicts containing:
            - ids: list of ids
            - texts: pd.Series indexed by id
            - vectors: sparse matrix (n_items x n_features)
        vectorizer: fitted TfidfVectorizer
    """
    logger.info("Building texts for all entities")

    user_texts = build_user_profile_text(datasets["users"])  # index user_id
    event_texts = build_asset_text(datasets["events"], "event_id", ["title", "description", "city"])
    job_texts = build_asset_text(datasets["jobs"], "job_id", ["title", "description", "company", "city"])
    post_texts = build_asset_text(datasets["posts"], "post_id", ["title", "content", "city"])

    combined = pd.concat([user_texts, event_texts, job_texts, post_texts])
    vectorizer = fit_shared_vectorizer(combined)

    embeddings = {
        "users": {"ids": user_texts.index.tolist(), "texts": user_texts, "vectors": transform_texts(vectorizer, user_texts)},
        "events": {"ids": event_texts.index.tolist(), "texts": event_texts, "vectors": transform_texts(vectorizer, event_texts)},
        "jobs": {"ids": job_texts.index.tolist(), "texts": job_texts, "vectors": transform_texts(vectorizer, job_texts)},
        "posts": {"ids": post_texts.index.tolist(), "texts": post_texts, "vectors": transform_texts(vectorizer, post_texts)},
    }

    return embeddings, vectorizer


if __name__ == "__main__":
    import logging
    from load_data import load_all

    logging.basicConfig(level=logging.INFO)
    data = load_all()
    embeddings, vec = build_embeddings(data)
    print({k: v["vectors"].shape for k, v in embeddings.items()})
