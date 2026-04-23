# GhostTrack: A Layered Signal Framework for Ghost Artist Detection on Spotify

**INFO 7390 — Spring 2026 | Trimbkeshwar Jagtap | Northeastern University**

---

## Research Question

Can behavioral signals derived from public metadata — release cadence, ISRC production company concentration, and cross-platform presence — reliably distinguish ghost artists (Spotify accounts inflating streams via bot plays) from organic artists, without access to restricted Spotify audio features or internal streaming data?

## Key Findings

| Signal | Finding | Effect Size |
|---|---|---|
| S2 Release Cadence | Ghost artists batch-upload same-day at 32–95% closure rate vs ~0% organic | Cohen's d = 3.44 (very large) |
| S6 ISRC Concentration | Ghosts use single CUSTOM_REGISTRANT; organics use TuneCore/DistroKid/labels | Mann-Whitney p=0.003, bootstrap 95% CI [0.567, 0.900] |
| S1 Audio Variance | Ghost catalogs collapse into unnaturally tight feature clusters | Cohen's d = −1.45 to −2.08 (large) |
| S2 Playlist Entropy | **Honest negative** — entropy does not discriminate ghost from organic | ANOVA F=0.25, p=0.78 |

## Scope Statement

This is a **signal discovery study**, not a production classifier. The framework demonstrates that specific behavioral signals exhibit large effect sizes on 3 confirmed ghost artists (DOJ indictment + investigative journalism sources). A deployment-grade classifier requires a larger independently-labeled dataset. Spotify API restrictions (February 2026) prevent real-time audio feature access — signals S1/S3 use Kaggle proxy data.

---

## Live Demo

- **App**: https://eda-for-music.onrender.com
- **API Docs**: https://eda-for-music.onrender.com/docs

---

## The 7-Signal Framework

| # | Signal | Method | Data Source | Effect Size |
|---|---|---|---|---|
| S1 | Audio Fingerprint | Variance + Levene test | Kaggle 114K proxy | Cohen's d = −1.45 to −2.08 |
| S2 | Release Cadence | Closure rate + KS test | Spotify API / Neo4j | d = 3.44, KS D=1.000 |
| S3 | ISRC Attribution | HHI + registrant type | Neo4j ISRC records | Mann-Whitney p=0.003 |
| S4 | Catalog Density | Tracks/day threshold | Spotify API | d = 0.32 (weak) |
| S5 | Metadata Similarity | NLP embeddings (sentence-transformers) | Spotify API | d = −0.91 (collinear w/ S2) |
| S6 | Graph Centrality | Neo4j HHI concentration | Neo4j | Bootstrap CI [0.567, 0.900] |
| S7 | Cross-Platform | YouTube + Apple Music APIs | YouTube / iTunes APIs | Informational |

**GNN augmentation**: GAT + GCN on 65-node co-appearance graph (AUC=1.000 on synthetic topology — see [methodological note](docs/models.html)).

---

## Ground Truth

- **3 confirmed ghost artists**: Relaxing White Noise, Meditation Relax Club (DOJ indictment: U.S. v. Michael Smith, Sep 2024), Calmo (signal convergence — not journalist-confirmed)
- **1 organic control**: Nils Frahm
- **Kaggle proxy baseline**: 1,030 organic artists for false positive rate analysis

---

## Documented Limitations

See [writing/limitations.md](writing/limitations.md) for the full 8-section audit. Key constraints:

1. **n=3 confirmed ghosts** — findings are bounded to this sample; all statistics are case-study-level, not population-level
2. **Genre confound** — ghost artists cluster in ambient/relaxation genres; controlled but not eliminated
3. **GNN AUC=1.000 is not generalizable** — synthetic graph topology; permutation feature importance is near-zero
4. **Kaggle CV AUC=1.000 is tautological** — ghost-proxy labels defined by same criterion S1 measures
5. **Geographic bias** — all confirmed ghosts are European (German/Italian registrants)
6. **Spotify API restrictions (Feb 2026)** — audio features, playlist tracks, and ISRC endpoints restricted for new apps

---

## Notebooks (10 Exercises)

| Notebook | Exercise | Key Finding |
|---|---|---|
| 01_catalog_coherence | S1 Audio Variance | Cohen's d = −1.45 to −2.08 on genre-matched comparison |
| 02_playlist_entropy | S2 Entropy | Honest negative: ANOVA F=0.25, p=0.78 |
| 03_isrc_join | S3 ISRC | CUSTOM_REGISTRANT = ghost pattern; aggregators = organic |
| 04_bipartite_neighborhood | S6 HHI | RWN=0.672, MRC=0.515, Calmo=0.452 vs organic mean=0.176 |
| 05_recommendation_walk | S2 Cadence | KS D=1.000; MRC released 172 tracks across 10 dates |
| 06_signal_testing | Composite | Multi-signal convergence on all 3 confirmed ghosts |
| 07_agent_crew_test | Pipeline | CrewAI multi-agent orchestration test |
| 08_gnn_training | GNN | GAT/GCN on 65-node graph; AUC=1.000 (see caveat) |
| 09_limitations | Audit | Self-audit: GNN result not generalizable; 3 documented fatal flaws |
| 10_cross_exercise_integration | Synthesis | Any 2 of {S2, S4, S6} → 100% in-sample accuracy on n=4 panel |

---

## Tech Stack

- **Backend**: FastAPI, Uvicorn, Pydantic, Neo4j, networkx
- **ML**: PyTorch Geometric (GAT + GCN), scikit-learn (Isolation Forest), sentence-transformers
- **Data**: Spotipy (Spotify), YouTube Data API v3, iTunes Search API, OpenAI GPT-4o-mini (fallback)
- **Frontend**: Vanilla JS SPA, D3.js (network viz)
- **Deployment**: Render.com (render.yaml)

---

## Local Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # fill in credentials
uvicorn backend.main:app --reload --port 8000
# Frontend: open http://localhost:8000
```

**Required env vars**: `SPOTIFY_CLIENT_ID`, `SPOTIFY_CLIENT_SECRET`, `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`  
**Optional**: `OPENAI_API_KEY`, `YOUTUBE_API_KEY`

---

## API Endpoints

| Endpoint | Purpose |
|---|---|
| `POST /analyze` | Full 7-signal analysis (Neo4j study panel) |
| `POST /analyze-live` | Live analysis for any Spotify artist |
| `GET /search` | Artist search (Spotify) |
| `GET /artist-tracks` | Track Intelligence (YouTube + iTunes) |
| `GET /graph/neighborhood/{id}` | Artist co-appearance network |
| `POST /chat` | AI research assistant (GPT-4o-mini) |

---

GhostTrack | INFO 7390 — Spring 2026 | By Trimbkeshwar Jagtap (jagtap.t@northeastern.edu)
