# Supplementary Materials Index
## EDA for Music: A Layered Framework for Ghost Artist Detection on Streaming Platforms
**INFO 7390 — Spring 2026**

---

## S1: Source Code Repository

All code is available at the project root. Key directories:

| Path | Contents |
|------|----------|
| `src/api/` | Spotify, YouTube, and iTunes API clients with caching |
| `src/graph/` | Neo4j ingestion and Cypher query helpers |
| `src/signals/` | Signal 1–7 computation modules |
| `src/models/` | GNN model definitions (GAT, GCN) and dataset builder |
| `notebooks/` | Jupyter notebooks for exercises 01–10 |
| `frontend/` | Streamlit dashboard (`app.py`) |
| `paper/figures/` | All generated figures (PNG, PDF) |

**Environment:** Python 3.14, dependencies in `requirements.txt`. Tested on macOS 15.x (Darwin 25.3.0).

---

## S2: Kaggle Dataset

**File:** `data/kaggle/dataset.csv`  
**Size:** 114,000 tracks, 21 columns, 114 genres  
**Source:** Kaggle Spotify Audio Features dataset (public domain)  
**Key columns used:** `artists`, `track_name`, `album_name`, `track_genre`, `popularity`, `danceability`, `energy`, `valence`, `acousticness`

Note: This dataset was used as a substitute for the Spotify `/audio-features` API endpoint, which was restricted for developer accounts beginning February 2026.

---

## S3: Neo4j AuraDB Graph

**Provider:** Neo4j AuraDB (free tier)  
**Schema:**

```
(Artist) -[:RELEASED]-> (Album) -[:CONTAINS]-> (Track) -[:REGISTERED_WITH]-> (ProductionCompany)
```

**Contents:**
- 3 Artist nodes: Relaxing White Noise, Meditation Relax Club, Calmo
- ~490 Track nodes
- ~150 Album nodes
- 8 ProductionCompany nodes (ISRC prefix registrants)

**Connection:** Credentials in `.env` (NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD). See `.env.example`.

---

## S4: Instructor's Guide

**File:** `paper/S4_instructors_guide.md`

Full per-exercise rubrics, expected values, common pitfalls, grading notes, and a FAQ for each of the 10 exercises. Includes:
- API limitation notes (Spotify restrictions since February 2026)
- Expected numerical outputs for all key metrics
- Common student mistakes and how to diagnose them

---

## S5: Generated Figures

All figures saved to `paper/figures/`:

| Filename | Exercise | Signal | Description |
|----------|----------|--------|-------------|
| `fig1_catalog_coherence.png` | Ex 1 | S1 | PCA scatter: ghost vs organic catalog variance |
| `fig1_catalog_coherence.pdf` | Ex 1 | S1 | Vector version |
| `fig2_playlist_entropy.png` | Ex 2 | S2 | Energy vs Valence by playlist archetype |
| `fig2b_playlist_boxplots.png` | Ex 2 | S2 | Feature distribution boxplots |
| `fig3_isrc_join.png` | Ex 3 | S3 | Bipartite artist–production company graph |
| `fig4_bipartite_neighborhood.png` | Ex 4 | S6 | Artist neighborhood with HHI scores |
| `fig4_bipartite_neighborhood.pdf` | Ex 4 | S6 | Vector version |
| `fig4b_isrc_distribution.png` | Ex 4 | S6 | Per-artist ISRC share distribution |
| `fig5_recommendation_walk.png` | Ex 5 | S2 | Release cadence: gap histograms per artist |
| `fig5_recommendation_walk.pdf` | Ex 5 | S2 | Vector version |
| `fig5b_hop_decay.png` | Ex 5 | S2 | Cumulative release gap CDF |
| `fig6_signal_radar.png` | Ex 6 | All | Seven-signal radar chart, all 4 artists |
| `fig6_signal_radar.pdf` | Ex 6 | All | Vector version |
| `fig6b_signal_heatmap.png` | Ex 6 | All | Signal score heatmap (7 × 4) |
| `fig7_gnn_performance.png` | Ex 7 | GNN | GAT/GCN training curves, ROC, confusion matrix |
| `fig8_convergence.png` | Ex 10 | All | Convergence heatmap + composability accuracy chart |
| `bipartite_interactive.html` | Ex 3 | S3 | Interactive Plotly bipartite graph |

---

## S6: GNN Dataset and Model

**Dataset file:** `data/processed/gnn_dataset.pt` (PyTorch Geometric `Data` object)  
**Metadata:** `data/processed/gnn_dataset_meta.json`  
**Trained model:** `data/processed/gat_model.pt`  

**Dataset summary:**
- 65 nodes (14 ghost, 51 organic)
- 692 edges (ghost: fully connected; organic: k-NN k=5 by feature distance)
- 8 node features: track_count, closure_rate, tracks_per_day, HHI, total_variance, mean_duration_ms, isrc_prefix_count, genre_count
- Train/val/test split: 60%/20%/20%, stratified by label

**Ghost node sources:** 3 from Neo4j (Relaxing White Noise, Meditation Relax Club, Calmo) + 11 from Kaggle ghost candidates  
**Organic node sources:** 1 from Neo4j (Nils Frahm) + 50 from Kaggle organic controls  
**Note:** Ghost/organic labels for Kaggle nodes are heuristic (not journalist-confirmed).

---

## S7: Signal Computation Details

### Signal 1 (S1): Audio Feature Variance
- **Method:** Per-artist variance of [danceability, energy, valence, acousticness] from Kaggle
- **Threshold:** Ghost ≤ 0.050 total variance; organic > 0.050
- **Key values:** Ghost mean variance ≈ 0.011; organic mean variance ≈ 0.117; ratio ≈ 10.6×

### Signal 2 (S2): Release Cadence Synchrony
- **Method:** Compute all consecutive release date gaps per artist; fraction with gap ≤ 1 day
- **Threshold:** Closure rate > 20% → suspicious
- **Key values:** MRC=94.7%, RWN=81.0%, Calmo=32.4%, Nils Frahm=3.6%

### Signal 3 (S3): ISRC Attribution
- **Method:** Count unique ISRC prefixes (first 5 chars) per artist
- **Threshold:** ≤ 2 prefixes → suspicious (concentrated production)
- **Key values:** RWN=2, MRC=2, Calmo=4, Nils Frahm=1

### Signal 4 (S4): Catalog Density Anomaly
- **Method:** HHI = Σ(tracks_per_company / total_tracks)² per artist
- **Threshold:** HHI > 0.40 → suspicious
- **Key values:** RWN=0.6715, MRC=0.5152, Calmo=0.4515, Nils Frahm=0.0000

### Signal 5 (S5): Artist Metadata Similarity
- **Method:** Keyword pattern matching on artist name; ambient/relaxation genre heuristic
- **Key values:** All four artists score similarly due to similar genre profile; signal is less discriminative for this test set

### Signal 6 (S6): Graph Density / HHI (overlaps with S4)
- **Method:** Combines HHI from S4 with bipartite graph degree centrality
- **Score range:** 0.0 (organic) to 0.716 (RWN)

### Signal 7 (S7): Cross-Platform Discrepancy
- **Method:** YouTube view count via Data API v3 + iTunes Search API presence
- **Note:** High views ≠ organic. RWN has 353M YouTube views but is still classified as ghost. S7 is an *additional* signal, not a disconfirming signal.
- **Key values:** RWN=353M views (Apple Music: Yes), MRC=157M (Yes), Calmo=155 (No), NF=9M (Yes)

---

## S8: Known Limitations

| ID | Limitation | Impact |
|----|-----------|--------|
| L1 | Only 3 confirmed ghost artists (journalist-sourced labels) | Small labeled set; overfitting risk in supervised models |
| L2 | Spotify audio features endpoint restricted since Feb 2026 | S1 uses Kaggle proxy data, not artist-specific API data |
| L3 | Kaggle ghost candidates are heuristic (not confirmed) | GNN label quality is uncertain for 11/14 ghost nodes |
| L4 | No cross-artist ISRC sharing in seed set | Graph has 0 organic cross-edges; topology is ideal, not realistic |
| L5 | Results specific to ambient/relaxation genre | Framework may not generalize to other genres without recalibration |
| L6 | GNN graph structure is synthetic | 100% test accuracy reflects graph construction, not generalization |
| L7 | YouTube "two-pass" search may return wrong video | View count is an approximation; artist channels not verified |
| L8 | Spotify Developer Mode: ~25 API calls/day limit | New artist ingestion not automated in demo |

---

## S9: Reproducibility Instructions

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Copy and fill in credentials
cp .env.example .env
# Add: NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
# Add: SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET
# Add: YOUTUBE_API_KEY
# Add: OPENAI_API_KEY

# 3. Run notebooks in order
jupyter nbconvert --to notebook --execute notebooks/01_kaggle_eda.ipynb
# ... through ...
jupyter nbconvert --to notebook --execute notebooks/10_cross_exercise_integration.ipynb

# 4. Launch dashboard
streamlit run frontend/app.py
```

All API calls are cached at `data/raw/cache/`. Re-running notebooks without clearing the cache will use cached data and not consume API quota.

---

## S10: Cross-Exercise Integration Summary (Exercise 10)

**File:** `notebooks/10_cross_exercise_integration.ipynb`

This notebook synthesizes all exercise outputs into a unified convergence analysis. Key outputs:

**Convergence Matrix** (artist × exercise flags):

| Artist | Ex1 Var | Ex2 Entropy | Ex4 HHI | Ex5 Closure | Ex3 ISRC | Flags |
|--------|---------|-------------|---------|-------------|----------|-------|
| Relaxing White Noise | ✓ | ✓ | ✓ | ✓ | ✓ | 5/5 |
| Meditation Relax Club | ✓ | ✓ | ✓ | ✓ | ✓ | 5/5 |
| Calmo | ✓ | ✓ | ✓ | ✓ | ✗ | 4/5 |
| Nils Frahm | ✗ | ✗ | ✗ | ✗ | ✗ | 0/5 |

**Composability table** (accuracy vs signals used):

| Signals | Accuracy | FP | FN |
|---------|----------|----|----|
| S2 only | 0.75 | 1 | 0 |
| S4 only | 0.85 | 0 | 1 |
| S6 only | 0.75 | 1 | 0 |
| S2 + S4 | 1.00 | 0 | 0 |
| S2 + S6 | 1.00 | 0 | 0 |
| S4 + S6 | 1.00 | 0 | 0 |
| S2 + S4 + S6 | 1.00 | 0 | 0 |
| All 6 signals | 1.00 | 0 | 0 |
| GNN (8 features + graph) | 1.00 | 0 | 0 |

**Key finding:** Any two-signal combination from the set {S2, S4, S6} achieves 100% accuracy on this test set, eliminating all false positives that individual signals produce.

---

*Last updated: April 2026 — INFO 7390, Northeastern University*
