# UnifiedRecommnedationEngine
A production-oriented recommendation prototype using TF-IDF + cosine similarity to generate explainable recommendations for users (events, jobs, posts).

## How to run (local)

1. Create a virtualenv and install dependencies (recommended):

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

2. Run the data pipeline to generate recommendations (one-off):

```powershell
python -m src.main
```

3. Start the Streamlit demo to explore results (client-ready UI):

```powershell
# from project root
python -m streamlit run streamlit_app.py
# or, if streamlit is on PATH
streamlit run streamlit_app.py
```

4. Outputs are written into `output/`:
- `user_recommendations.json` — per-user aggregated recommendations (events, jobs, posts)
- `event_recommendations.json` — flattened list of event recommendations
- `job_recommendations.json` — flattened list of job recommendations
- `user_recommendations.json` (also available as per-user object list for UI consumption)


**Notes:**
- Prefer `python -m src.main` to ensure package imports are resolved correctly.
- If you see encoding issues when loading CSVs, ensure files are saved in UTF-8 or re-run the pipeline after fixing encodings.

## Project structure
- `data/` — CSV inputs (users, events, jobs, posts)
- `src/` — pipeline code (loader, feature builder, recommender, active learning, orchestration, Streamlit demo)
- `streamlit_app.py` — client-facing demo UI (reads precomputed JSON results)
- `output/` — generated recommendation JSON files
- `reports/` — architecture and logic documentation

## Future improvements
- Add real interaction logs: replace popularity fallback with collaborative signals
- Swap TF-IDF for neural embeddings (production step)
- Add vector store for large-scale retrieval
- Add CI checks and a small automated test that runs the pipeline and verifies output files

---

If you'd like, I can also add a short `run_demo.sh` / `run_demo.ps1` helper and a troubleshooting section for common issues.
---

If you'd like, I can run a quick sanity check to generate an example output and show a sample of recommendations.