# Assumptions & Limits ⚠️

## Assumptions
- Dataset is synthetic (~50 rows per CSV) and clean but may use alternative column names; loader maps common aliases.
- There are no user-item interaction logs — system must be cold-start friendly.
- TF-IDF input is text-only; numeric features are not used in the prototype.

## Limits
- Popularity fallback is a simple heuristic and will not substitute for real collaborative signals.
- TF-IDF cannot capture deep semantic relationships (e.g., synonyms without vocabulary overlap); neural embeddings improve this.
- No online learning in this prototype — active learning simulation is offline and illustrative only.

## How real data will improve results
- Event attendance or click logs allow training collaborative or hybrid models and capturing real preferences.
- Richer user profiles and event/job metadata (structured features) allow better matching and personalization.
- Scale: real datasets will require vector stores and batched/approximate nearest neighbor search for latency.

---

Be explicit: this is a production-oriented prototype designed to be understandable and extensible; only real interaction logs, richer metadata, and specialized infrastructure are missing.