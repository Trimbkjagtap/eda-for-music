# EDA for Music — Project Build Log

**Course:** INFO 7390 · Advances in Data Sciences & its Applications  
**Student:** Trimbkeshwar Jagtap · jagtap.t@northeastern.edu  
**Repo:** `/Users/trimbkjagtap/eda-for-music`  
**Sprint:** April 13–23, 2026 (10-day build)  
**Status as of April 19, 2026:** All 7 exercises complete, GNN trained, full-stack app deployed locally

---

## What This Project Is

A full-stack data science system for detecting **ghost artists** on Spotify — fake accounts (often AI-generated ambient/relaxation tracks) used to farm streaming royalties. Built using only public APIs (no scraping, no private data), a Kaggle audio features dataset, and a Neo4j graph database.

**Core thesis:** Ghost artists leave detectable signatures across 7 measurable signals even with Spotify's February 2026 API restrictions. The project proves this with real confirmed ghost artist data (from the Michael Smith fraud case) and builds a working automated detection pipeline.

---

## The Constraint That Shaped Everything

In **February 2026, Spotify restricted their public API** — removing:
- Audio features endpoint (`GET /audio-features`)
- Related artists endpoint (`GET /artists/{id}/related-artists`)
- Follower counts from artist objects
- Genre tags from artist objects

This happened mid-project. The workaround was to use the **Kaggle Spotify Audio Features dataset** (114K tracks) as a static fallback for Signal 1 (catalog coherence), and redesign Signals 4–7 to rely on metadata that was still accessible. This is why the project explicitly documents API state — it's a reproducibility concern for reviewers.

**Interview point:** "I had to adapt the signal architecture mid-sprint when Spotify changed their API. That decision — using Kaggle as a fallback rather than blocking on a broken endpoint — let me keep the project moving and actually produced a more interesting design."

---

## Ground Truth: The Seed Artists

Ghost artist labels come from real investigative reporting and court records:

| Artist | Spotify ID | Status | Source |
|---|---|---|---|
| Relaxing White Noise | `6bo3atMVp3qFECNALVwq9N` | Ghost | Michael Smith fraud case |
| Meditation Relax Club | `39t4EeLBfpT72UQJVkIeuj` | Ghost | Dagens Nyheter investigation |
| Calmo | `4Wx3ZL6d6p1gVMtwQ2YWsz` | Suspicious | Shared production companies with above |
| Nils Frahm | `5gqhueRUZEa7VDnQt4HODp` | Organic control | Established neo-classical artist |

**Why only 4 seeds:** Confirmed ghost artist identities are rare. Academic datasets for this specific problem don't exist. Journalist-derived ground truth was the best available option. The project explicitly documents this as Limitation #1.

---

## The 7-Signal Framework

The detection logic is modular — each signal returns a score in [0, 1]. The final verdict combines them with configurable weights.

### Signal 1 — Catalog Coherence (Audio Fingerprint Similarity)
- **Source:** Kaggle CSV joined on Spotify track ID
- **Method:** PCA on 12 audio features (danceability, energy, valence, acousticness, etc.), then measure variance within each artist's catalog
- **Finding:** Ghost artists have 12.5× lower intra-catalog variance than organic artists — they use a narrow template
- **File:** [src/signals/audio_similarity.py](src/signals/audio_similarity.py)
- **Decision:** Kaggle join works for the seed artists because the dataset covers ambient/meditation genre well

### Signal 2 — Release Cadence Synchrony
- **Source:** Neo4j (album release dates) + Exercise 5 walk metrics
- **Method:** Calculate "closure rate" — percentage of albums released within 3 days of a cluster
- **Finding:** RWN = 81%, MRC = 95% same-day releases. Nils Frahm = 4%
- **File:** [src/signals/cadence_sync.py](src/signals/cadence_sync.py)
- **Decision:** Used Isolation Forest as secondary check on release gap distribution

### Signal 3 — Playlist Co-occurrence Density
- **Source:** Neo4j ISRC-based proxy (real playlist endpoint returns 404 for non-partner apps)
- **Method:** Jaccard similarity on shared ISRC production company codes
- **Caveat:** This is the weakest signal. Editorial playlist data requires Spotify partnership. The ISRC proxy is a documented approximation.
- **File:** [src/signals/playlist_cooccurrence.py](src/signals/playlist_cooccurrence.py)

### Signal 4 — Catalog Density Anomaly (Upload Velocity)
- **Source:** Neo4j track count + earliest/latest release date span
- **Method:** `tracks_per_day = total_tracks / days_active`
- **Threshold:** > 0.15 tracks/day flagged as suspicious
- **File:** [src/signals/follower_ratio.py](src/signals/follower_ratio.py) *(named before API restrictions forced redesign)*

### Signal 5 — Metadata Similarity (Name Patterns)
- **Source:** Neo4j track/album names
- **Method:** TF-IDF + sentence-transformers embeddings on track names; detect template naming ("Relaxing Rain for Sleep", "Calming Ocean for Study")
- **File:** [src/signals/metadata_similarity.py](src/signals/metadata_similarity.py)

### Signal 6 — Graph Density / HHI
- **Source:** Neo4j ISRC → production company bipartite graph
- **Method:** Herfindahl-Hirschman Index on which production companies registered an artist's tracks
- **Finding:** RWN = 0.671 HHI, MRC = 0.515, Calmo = 0.452. Nils Frahm = 0.00 (all unique)
- **What this means:** Ghost artists funnel all tracks through 1–2 shell production companies. Organic artists have diverse distribution
- **File:** [src/signals/graph_density.py](src/signals/graph_density.py)

### Signal 7 — Cross-Platform Discrepancy
- **Source:** YouTube Data API v3 + Apple Music iTunes Search API (no key)
- **Method:** YouTube view count + presence on Apple Music
- **Surprising finding:** Ghost artists are NOT invisible on YouTube — RWN has 353,775,503 views, MRC has 157,581,316 views. The fraud is Spotify-specific stream farming, not fabricated identity. Calmo has only 319 YouTube views, which is suspicious in the other direction.
- **File:** [src/signals/cross_platform.py](src/signals/cross_platform.py)
- **Decision:** Made this signal opt-in (expensive API calls). Users can trigger it from the UI.

### Verdict Logic
- LIKELY_GHOST: overall_score > 0.70
- SUSPICIOUS: 0.40–0.70
- LIKELY_ORGANIC: < 0.40
- **File:** [src/signals/verdict.py](src/signals/verdict.py)

---

## Architecture Decisions

### Why Neo4j
Graph databases are a natural fit for the ISRC bipartite structure (artists ↔ production companies). Neo4j AuraDB Free tier was chosen because it's free, cloud-hosted, and the course exercises explicitly require graph-based analysis. Cypher queries for HHI and neighborhood analysis are cleaner than equivalent pandas code would be.

### Why CrewAI Multi-Agent
The course (INFO 7390) required demonstrating agent orchestration. CrewAI was chosen over LangGraph because it has a simpler API for 4-stage sequential pipelines. The project runs in two modes:
- **Pipeline mode** (default): Pure Python, no LLM, 2–5 seconds per artist
- **Crew mode**: CrewAI + GPT-4o reasoning, slower, requires OpenAI key

**Decision to minimize OpenAI usage:** After Day 6, the AI analysis was moved to on-demand only (user clicks "AI Deep-Dive" button). The baseline verdict is fully rule-based. This reduces API costs and makes the project reproducible without an OpenAI key.

### Why FastAPI + Streamlit (not just a notebook)
The professor asked for a "production-quality" system, not just notebooks. FastAPI handles the analysis backend; Streamlit provides the interactive frontend. This separation means the analysis logic can be called from the UI, from notebooks, or from the CLI. It also demonstrates software engineering discipline beyond pure data science.

### Why GNN (GAT + GCN)
Exercise 7 required deep learning on the graph. GAT (Graph Attention Network) was chosen as primary because attention heads weight neighbor signals differentially — useful when some ISRC connections are more informative than others. GCN was included as a simpler baseline for comparison. Both were trained with PyTorch Geometric.

**Honest limitation:** The training graph has 65 nodes and is partially synthetic because only 3 confirmed ghost artists were available. Test accuracy of 100% is a proof-of-concept, not a deployable model. This is documented in [notebooks/09_limitations.ipynb](notebooks/09_limitations.ipynb).

### API Caching Strategy
All external API calls are cached to `/data/raw/cache/` using MD5-hashed JSON files. The rate limiter (`src/utils/rate_limiter.py`) uses a token bucket with exponential backoff. Spotify calls are limited to ~30 per session. This was critical because:
1. Spotify's rate limits are undocumented and aggressive
2. Re-running notebooks shouldn't cost API quota
3. The Kaggle dataset is static — no need to re-fetch

---

## Day-by-Day Build Log

### Day 1 — April 13, 2026
**Commit:** `141c3ce Day 1: project scaffold + API clients + confirmed April 2026 API state`

Built from scratch:
- Project directory structure
- `.env` config system with `src/utils/config.py`
- Spotify API client with April 2026 workarounds (`src/api/spotify_client.py`)
- YouTube API client (`src/api/youtube_client.py`)
- Apple Music client (`src/api/apple_music_client.py`)
- Neo4j AuraDB client with schema setup (`src/graph/neo4j_client.py`)
- Token bucket rate limiter (`src/utils/rate_limiter.py`)
- Test suite confirming all API connections live

**Key discovery on Day 1:** `GET /audio-features` returns 403. `GET /artists/{id}` has no `followers` or `genres` field. Documented in `api_state.md`. Pivoted Signal 1 to use Kaggle.

---

### Day 2 — April 14, 2026
**Commit:** `ecdc47e Day 2: Signal 1 (catalog coherence), Signal 2 (release cadence), seed ingest pipeline`

- Built `src/ingest/seed_ingest.py` — ingests 4 seed artists into Neo4j
- Implemented Signal 1 (audio similarity via Kaggle)
- Implemented Signal 2 (release cadence)
- Established ground truth lists in `src/utils/ground_truth.py`

---

### Day 3 — April 15, 2026
**Commits:** `ba6082c` + `2ca9de4` + `b35b2af`

- Kaggle dataset exploration
- Neo4j population confirmed (490 tracks ingested)
- Exercises 1–3 notebooks completed with figures
- Streamlit dashboard skeleton
- Fixed Nils Frahm Spotify ID (wrong ID caused missing data)
- Added aggressive caching + call limits after hitting rate limits

---

### Day 4 — April 16, 2026
**Commit:** `3489329 Day 4: Exercise 4 bipartite neighborhood + Exercise 5 recommendation walk`

- Exercise 4: Bipartite neighborhood + HHI analysis → `fig4_bipartite_neighborhood.png`
- Exercise 5: Temporal walk on release cadence → `fig5_recommendation_walk.png`
- `data/processed/exercise4_ghost_neighborhood.json`
- `data/processed/exercise5_walk_graphs.json`

---

### Day 5 — April 17, 2026
**Commit:** `bc1e4bb Day 5: All 7 signal scoring functions + Signal 7 + verdict scorer`

- All 7 signal modules implemented
- Cross-platform Signal 7 (YouTube + Apple Music)
- Verdict scorer with configurable weights
- Signal testing notebook (Exercise 6) with radar chart + heatmap
- `data/processed/ex6_verdicts.json`

**Key finding on Day 5:** Ghost artists have massive YouTube presence (up to 353M views for RWN). The "fake" aspect is Spotify-specific, not cross-platform. This reframed the project thesis.

---

### Day 6 — April 18, 2026
**Commit:** `5ba9af7 Day 6: CrewAI 4-agent crew + FastAPI backend + Streamlit dashboard`

- `src/agents/` — 4-stage pipeline + CrewAI crew mode
- `backend/main.py` — FastAPI REST API (10 endpoints)
- `frontend/app.py` — Full Streamlit dashboard (7 pages)
- Exercise 7 notebook (agent orchestration)
- Cross-exercise integration notebook

---

### Day 7 — April 19, 2026
**Commits:** `b797c5c` + `35a9382` + `b292dd6` + `2cad971`

- `src/models/gnn_detector.py` — GAT + GCN model definitions
- `src/models/dataset_builder.py` — PyTorch Geometric dataset from Neo4j
- `notebooks/08_gnn_training.ipynb` — Training, ROC curves, confusion matrix
- `notebooks/09_limitations.ipynb` — Known limitations documented
- Removed Paper Generator page (not needed for submission)
- OpenAI usage minimized to on-demand only

---

## What's Left (Days 8–10)

Based on the original 10-day plan:

- [ ] **Day 8:** Final figure polish, export all publication-quality PNGs, write paper sections
- [ ] **Day 9:** End-to-end demo run, README completion, supplementary materials
- [ ] **Day 10:** Submission packaging, final review, arXiv preprint (if pursuing)

Outstanding items:
- `notebooks/10_cross_exercise_integration.ipynb` — exists but may need final narrative
- `paper/` directory has instructor guide and supplementary materials but no final paper draft
- The GNN could be retrained with more seed artists if more labeled data becomes available

---

## File Map for Reviewers

```
eda-for-music/
├── backend/main.py              # FastAPI: 10 endpoints, health + analysis + graph
├── frontend/app.py              # Streamlit: 7 pages, full interactive dashboard
├── src/
│   ├── agents/                  # CrewAI 4-stage detection pipeline
│   │   ├── crew.py              # Orchestrator (pipeline mode + crew mode)
│   │   ├── crawler_agent.py     # Stage 1: metadata gathering
│   │   ├── fingerprint_analyst.py # Stage 2: Signals 1, 4, 5
│   │   ├── graph_builder.py     # Stage 3: Signals 2, 3, 6
│   │   └── verdict_agent.py     # Stage 4: combine + threshold
│   ├── api/                     # External API clients (Spotify, YouTube, Apple Music)
│   ├── graph/neo4j_client.py    # Neo4j CRUD + Cypher queries
│   ├── signals/                 # 7 signal modules + verdict scorer
│   ├── models/                  # GNN (GAT + GCN) + dataset builder
│   ├── ingest/seed_ingest.py    # Seed artist ingestion → Neo4j
│   └── utils/                   # Config, ground truth, Kaggle loader, rate limiter
├── notebooks/                   # 11 Jupyter notebooks (exercises 1–7 + GNN + limits)
├── data/
│   ├── kaggle/dataset.csv       # 114K tracks, 12 audio features (static)
│   ├── raw/cache/               # MD5-hashed API response cache
│   └── processed/               # GNN weights, training summary, exercise outputs
├── paper/                       # Supplementary materials + figures
│   └── figures/                 # 8+ publication-quality PNGs
├── requirements.txt             # ~56 Python dependencies
├── .env.example                 # Credentials template
└── BUILD_LOG.md                 # This file
```

---

## How to Run

```bash
# 1. Setup
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # fill in SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET, NEO4J_URI, etc.

# 2. Backend
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload

# 3. Frontend
streamlit run frontend/app.py

# 4. Run detection on a single artist (CLI)
python -c "
from src.agents.crew import run_analysis
result = run_analysis('6bo3atMVp3qFECNALVwq9N')  # Relaxing White Noise
print(result)
"

# 5. Reproduce GNN training
jupyter notebook notebooks/08_gnn_training.ipynb
```

---

## Interview Talking Points

**"Why fake artists specifically?"**
It's an economic crime that's hard to detect programmatically — the fraud happens at the financial layer, not the content layer. The artists produce real music; the crime is in how streams are generated. That asymmetry is what makes it a compelling graph problem.

**"What was your hardest technical challenge?"**
Spotify restricted their API mid-sprint, removing audio features. I had to pivot Signal 1 to use a Kaggle fallback dataset and redesign 3 other signals to work with metadata-only. The constraint actually produced a better architecture — it forced me to think about what you can infer from graph structure alone.

**"How did you validate your results without ground truth labels?"**
I used journalist-reported confirmed ghost artists (Michael Smith fraud case, Dagens Nyheter investigation) as seed labels. That's 3 confirmed positives and 1 confirmed negative. The findings are directional, not generalizable — and I documented this as the primary limitation.

**"Why not just use supervised ML with a bigger dataset?"**
No labeled dataset exists for this problem at scale. The signals approach is interpretable — a compliance officer or A&R team can understand why an artist was flagged. A black-box classifier trained on unverified labels would be worse in both reliability and explainability.

**"What would you do with more time?"**
Ingest 50–100 more seed artists for the GNN training set, partner with a label or streaming service for ground truth labels, and extend Signal 3 with actual playlist co-occurrence data (requires Spotify partnership). The architecture would scale — the signals are all independently computable.

---

## Known Limitations (Summary)

1. **3 confirmed ghost artists only** — small ground truth, results are directional
2. **Kaggle dataset is static 2023 data** — doesn't cover all ambient/meditation tracks
3. **Playlist co-occurrence (S3) uses ISRC proxy** — real editorial playlist data requires Spotify partnership
4. **GNN trained on partially synthetic graph** — 100% test accuracy is proof-of-concept, not deployment-ready
5. **Results are genre-specific** — ghost artist patterns in ambient/relaxation may differ from pop or hip-hop
6. **Spotify API restrictions** — any endpoint returning 403 could be re-added or further restricted
7. **YouTube view counts** — surprise finding (ghost artists have massive YouTube presence) suggests the stream-farming hypothesis needs refinement for S7 scoring

---

*Last updated: April 19, 2026*
