# UnifiedRecommnedationEngine
A production-oriented recommendation prototype using TF-IDF + cosine similarity to generate explainable recommendations for users (events, jobs, posts).

## How to run (local)

1. (Optional) Create a virtualenv and install dependencies:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

2. Run the pipeline:

```powershell
python src/main.py
```

3. Outputs are written into `output/`:
- `user_recommendations.json` — per-user aggregated recommendations (events, jobs, posts)
- `event_recommendations.json` — flattened list of event recommendations
- `job_recommendations.json` — flattened list of job recommendations

## Project structure
- `data/` — CSV inputs (users, events, jobs, posts)
- `src/` — pipeline code (loader, feature builder, recommender, active learning, orchestration)
- `output/` — generated recommendation JSON files
- `reports/` — architecture and logic documentation

## Future improvements
- Add real interaction logs: replace popularity fallback with collaborative signals
- Swap TF-IDF for neural embeddings (production step)
- Add vector store for large-scale retrieval

---

If you'd like, I can run a quick sanity check to generate an example output and show a sample of recommendations.