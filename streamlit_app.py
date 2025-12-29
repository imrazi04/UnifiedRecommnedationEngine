"""Streamlit demo app for recommendation results (TF-IDF + Cosine similarity)

Run:
    streamlit run streamlit_app.py

Design goals:
- Read precomputed recommendation JSONs from `output/`
- Allow user selection via dropdown (no typing)
- Show Top-10 Events / Jobs / Posts with clear explanations
- Client-friendly language and visuals
"""
from pathlib import Path
import json
import logging
from typing import Dict, List

import streamlit as st
import pandas as pd

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("output")
USER_RECS_PATH = OUTPUT_DIR / "user_recommendations.json"
EVENTS_PATH = Path("data") / "events.csv"
JOBS_PATH = Path("data") / "jobs.csv"
POSTS_PATH = Path("data") / "posts.csv"


@st.cache_data
def load_user_recs(path: Path) -> Dict[str, Dict]:
    if not path.exists():
        st.error(f"Recommendations file not found: {path}. Run the pipeline first (python -m src.main)")
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # data expected as list of {"user_id": u, "recommendations": {"events": [...], "jobs": [...], "posts": [...]}}
    mapping = {entry["user_id"]: entry["recommendations"] for entry in data}
    return mapping


@st.cache_data
def load_asset_titles() -> Dict[str, Dict[str, str]]:
    """Load asset metadata to present human-friendly titles where available.

    Try several encodings (utf-8, cp1252, latin-1) to be robust to source CSV encodings.
    If a file cannot be read, the function logs a warning and returns partial results instead of crashing Streamlit.
    """
    result = {"events": {}, "jobs": {}, "posts": {}}
    encodings = ["utf-8", "cp1252", "latin-1"]

    def _try_read(path: Path):
        for enc in encodings:
            try:
                df = pd.read_csv(path, encoding=enc, low_memory=False)
                if enc != "utf-8":
                    logger.warning("Read %s using fallback encoding %s", path.name, enc)
                return df
            except UnicodeDecodeError:
                logger.debug("Failed reading %s with encoding %s", path.name, enc)
                continue
            except Exception as e:
                logger.debug("Error reading %s with encoding %s: %s", path.name, enc, e)
                continue
        # last resort: read with latin-1 and replace errors
        try:
            df = pd.read_csv(path, encoding="latin-1", low_memory=False)
            logger.warning("Read %s with latin-1 as last resort (some characters may be replaced)", path.name)
            return df
        except Exception as e:
            logger.exception("Unable to read %s with fallback encodings: %s", path.name, e)
            return None

    if EVENTS_PATH.exists():
        df = _try_read(EVENTS_PATH)
        if df is None:
            st.warning(f"Could not read events metadata from {EVENTS_PATH.name}; titles may be missing.")
        else:
            for _, r in df.iterrows():
                result["events"][str(r.get("event_id"))] = str(r.get("title", ""))

    if JOBS_PATH.exists():
        df = _try_read(JOBS_PATH)
        if df is None:
            st.warning(f"Could not read jobs metadata from {JOBS_PATH.name}; titles may be missing.")
        else:
            for _, r in df.iterrows():
                result["jobs"][str(r.get("job_id"))] = str(r.get("title", ""))

    if POSTS_PATH.exists():
        df = _try_read(POSTS_PATH)
        if df is None:
            st.warning(f"Could not read posts metadata from {POSTS_PATH.name}; titles may be missing.")
        else:
            for _, r in df.iterrows():
                result["posts"][str(r.get("post_id"))] = str(r.get("title", ""))

    return result


def friendly_reason(reason_raw: str) -> str:
    reason = reason_raw.lower() if reason_raw else ""
    parts = []
    if "text-based" in reason or "similarity" in reason:
        parts.append("Matches your profile based on textual similarity")
    if "popularity" in reason:
        parts.append("Popular item fallback (cold-start)")
    if "feedback adjusted (positive)" in reason:
        parts.append("Boosted by positive feedback")
    if "feedback adjusted (negative)" in reason:
        parts.append("Reduced after negative feedback")
    if not parts:
        return "Recommended based on relevance to your profile"
    return "; ".join(parts)


def score_to_percentage(score: float) -> int:
    """Convert [0..1] score to 0..100 integer for a simple progress visualization."""
    try:
        return int(max(0, min(1, float(score))) * 100)
    except Exception:
        # if score is outside 0..1 (popularity normalized) try to scale
        try:
            return int(float(score) * 100)
        except Exception:
            return 0


def render_recommendation(rec: dict, title: str, idx: int, assets_map: Dict[str, str]):
    """Render a single recommendation as a compact card/row."""
    score = float(rec.get("score", 0.0))
    score_3 = f"{score:.3f}"
    pct = score_to_percentage(score)

    asset_id = rec.get("asset_id")
    display_title = assets_map.get(asset_id, f"{title} {asset_id}")
    reason = friendly_reason(rec.get("reason", ""))

    # layout
    col_rank, col_main, col_score = st.columns([0.6, 8, 2])
    with col_rank:
        st.markdown(f"**#{idx}**")
    with col_main:
        st.markdown(f"**{display_title}**  \n*ID:* `{asset_id}`")
        st.caption(reason)
    with col_score:
        st.metric(label="Score", value=score_3)
        st.progress(pct)


def main():
    st.set_page_config(page_title="Recommendation Demo", layout="centered")

    st.title("🎯 Personalized Recommendations")
    st.write("A client-facing demo showing top recommendations per user. No model training here — only precomputed results are displayed.")

    # Load data
    user_recs = load_user_recs(USER_RECS_PATH)
    if not user_recs:
        st.stop()

    assets_map = load_asset_titles()

    user_ids = sorted(user_recs.keys())
    st.sidebar.header("Select user")
    user_select = st.sidebar.selectbox("User ID", options=user_ids)

    st.header(f"📌 Recommendations for user `{user_select}`")

    recs_for_user = user_recs.get(user_select, {})

    # Three columns: Events, Jobs, Posts (each a section)
    for asset_type, emoji, label in [("events", "📅", "Top Events"), ("jobs", "💼", "Top Jobs"), ("posts", "📝", "Top Posts")]:
        st.subheader(f"{emoji} {label}")
        items = recs_for_user.get(asset_type, [])
        if not items:
            st.info("No recommendations available for this category.")
            continue
        # Show top-10 (already top-n in saved file, but be safe)
        for i, rec in enumerate(sorted(items, key=lambda x: -float(x.get("score", 0.0)))[:10], start=1):
            render_recommendation(rec, label[:-1], i, assets_map.get(asset_type, {}))
        st.markdown("---")

    with st.expander("How these recommendations were generated (simple)"):
        st.write(
            "- Text fields from users and items were converted to TF-IDF vectors (a transparent text representation).\n"
            "- We compare user TF-IDF and item TF-IDF using cosine similarity to rank items — more similar text => higher rank.\n"
            "- For new/empty profiles we fall back to popularity heuristics; small explainable boosts are applied for city/category/tag overlap.\n"
            "- Active-learning feedback (if any) adjusts scores slightly to reflect likes/dislikes."
        )
        st.caption("This demo shows results already computed and saved by the pipeline. No training or re-computation happens in this UI.")

    st.sidebar.header("About this demo")
    st.sidebar.write(
        "This professional demo is designed for clients: it communicates *why* items are recommended and provides interpretable signals such as reason and score."
    )


if __name__ == "__main__":
    main()
