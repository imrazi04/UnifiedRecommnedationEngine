"""Lightweight active learning simulation.

Simulates feedback (like / dislike) and adjusts recommendation scores accordingly.
"""
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)


class ActiveLearner:
    def __init__(self, positive_weight: float = 0.2, negative_weight: float = -0.25):
        """Initialize with configurable weights used to adjust scores.

        positive_weight: additive boost for liked items
        negative_weight: multiplicative or additive penalty for disliked items
        """
        self.positive_weight = positive_weight
        self.negative_weight = negative_weight
        # store simple feedback history
        self.feedback = []

    def simulate_feedback(self, recommendations: List[Dict], positive_ratio: float = 0.05, negative_ratio: float = 0.02) -> List[Dict]:
        """Simulate random feedback over recommendations.

        This function will select a small subset as positive and negative feedback and return feedback entries.
        Feedback entries are tuples: (user_id, asset_type, asset_id, feedback)
        where feedback is +1 (like) or -1 (dislike).
        """
        import random

        feedback_entries = []
        for rec in recommendations:
            r = random.random()
            if r < negative_ratio:
                feedback_entries.append((rec["user_id"], rec["asset_type"], rec["asset_id"], -1))
            elif r < (negative_ratio + positive_ratio):
                feedback_entries.append((rec["user_id"], rec["asset_type"], rec["asset_id"], 1))
        self.feedback.extend(feedback_entries)
        logger.info("Simulated %d feedbacks", len(feedback_entries))
        return feedback_entries

    def apply_feedback(self, recs: List[Dict], feedback_entries: List[tuple]) -> List[Dict]:
        """Apply feedback to a list of recommendations (in-place semantic adjustment).

        This adjusts the 'score' field on matching recommendations.
        """
        # Build quick lookup
        fb_map = {}
        for u, atype, aid, f in feedback_entries:
            fb_map.setdefault((u, atype, aid), 0)
            fb_map[(u, atype, aid)] += f

        adjusted = []
        for rec in recs:
            key = (rec["user_id"], rec["asset_type"], rec["asset_id"])
            if key in fb_map:
                # positive feedback increases, negative decreases
                if fb_map[key] > 0:
                    rec = rec.copy()
                    rec["score"] = rec["score"] + self.positive_weight
                    rec["reason"] = rec.get("reason", "") + "; feedback adjusted (positive)"
                else:
                    rec = rec.copy()
                    rec["score"] = max(0.0, rec["score"] + self.negative_weight)
                    rec["reason"] = rec.get("reason", "") + "; feedback adjusted (negative)"
            adjusted.append(rec)
        return adjusted
