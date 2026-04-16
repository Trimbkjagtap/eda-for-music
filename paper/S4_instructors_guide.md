# S4 Instructor's Guide: EDA for Music — Ghost Artist Detection on Spotify

**Course:** INFO 7390 — Exploratory Data Analysis for Music  
**Term:** Spring 2026  
**Project type:** Individual, 10 exercises, cumulative submission  
**Total points:** 250 + 20 bonus  

---

## Overview

This project teaches students to detect ghost artists — fraudulent or AI-generated accounts used to inflate streaming royalties — using only Spotify's public, unauthenticated API. Each exercise introduces a new detection signal and a new data analysis technique. The exercises are designed to compose: signals identified in early exercises are reused and combined in later ones, culminating in a GNN-based classifier.

The central academic claim, demonstrated in Exercise 10, is that **independent signals across exercises converge on the same artists**. This convergence is itself the finding — it shows that ghost artist behavior is multi-dimensional and not an artifact of any single measurement.

---

## Technology Stack (April 2026 State)

Students must be made aware that the Spotify API underwent significant restriction in February 2026:

| Feature | Pre-Feb 2026 | April 2026 (current) |
|---------|-------------|----------------------|
| Audio features | Available | **Removed** (restricted to internal) |
| Related artists | Available | **Removed** |
| Followers count | Available | **Removed** |
| Search | Available | Available |
| Track metadata | Available | Available (ISRC, duration, explicit) |
| Album releases | Available | Available |

**Implication:** Exercises 1 (audio variance) and 5 (metadata similarity) must use the Kaggle dataset (`data/kaggle/dataset.csv`, 114K tracks, 114 genres) rather than real-time Spotify audio feature endpoints. Instructors should clarify that this is a documented API limitation, not a student error.

---

## Exercise-by-Exercise Rubric

### Exercise 1: Catalog Coherence (Audio Feature Variance)
**Signal:** S1 — Audio Fingerprint Similarity  
**Learning objective:** Visualize per-artist feature distributions; identify low-variance anomalies  
**Data source:** Kaggle 114K dataset (audio features: danceability, energy, valence, acousticness)  

| Criterion | Full credit | Partial | No credit |
|-----------|-------------|---------|-----------|
| Feature variance computed per artist | Per-feature variance with groupby artist | Global variance only | No variance computation |
| Visualization | PCA scatter with 1σ ellipses, clear labeling | Scatter without ellipses | Histogram only |
| Interpretation | References ghost/organic separation; cites p-value | States difference without test | No interpretation |
| Statistical test | Levene's test or similar (heteroscedasticity) | t-test | None |

**Expected finding:** Ghost artist audio feature variance is 10–15× lower than organic controls. Levene's test p < 0.001.

**Common pitfall:** Students compute variance across all artists' tracks pooled together. Correct approach: variance per artist (within-artist variance), then compare distributions between ghost and organic groups.

**Grading note:** Accepted ghost variance range: 0.005–0.025. Accepted organic range: 0.08–0.15. If a student's numbers fall outside these ranges, check whether they're using the correct Kaggle columns (`danceability`, `energy`, `valence`, `acousticness`) and computing within-artist variance.

---

### Exercise 2: Playlist Entropy (Aesthetic Coherence)
**Signal:** S2 — Playlist Entropy  
**Learning objective:** Simulate playlist types; compute Shannon entropy; identify low-entropy fraud zones  
**Data source:** Spotify /playlists endpoint (or simulated from Kaggle)  

| Criterion | Full credit | Partial | No credit |
|-----------|-------------|---------|-----------|
| Shannon entropy computed | H = -Σ p·log₂(p) for feature bins | Uses count instead of probability | No entropy calculation |
| Playlist archetypes simulated | ≥3 archetypes (tight/fan/editorial) | 2 archetypes | 1 or none |
| Ghost suspect zone identified | H < 2.55 bits labeled as fraud zone | Identifies zone without threshold | No classification |
| Visualization | Energy vs Valence scatter, colored by archetype | Scatter without archetype labeling | No scatter |

**Expected finding:** Editorial playlists: H ≈ 2.59 bits. Fan playlists: H ≈ 2.89 bits. Ghost-suspect playlists: H ≈ 2.51 bits.

**Common pitfall:** Entropy computed over all tracks globally instead of per-playlist or per-artist catalog. Correct approach: for each playlist/artist, discretize the feature into bins, compute the probability distribution, apply Shannon formula.

---

### Exercise 3: ISRC Attribution (Production Company Network)
**Signal:** ISRC network (S6 precursor)  
**Learning objective:** Join artist data to production companies via ISRC prefix; build bipartite graph  
**Data source:** Spotify /tracks (external_ids.isrc); Neo4j AuraDB  

| Criterion | Full credit | Partial | No credit |
|-----------|-------------|---------|-----------|
| ISRC prefix extraction | First 5 chars (`ISRC[:5]`) correct | Full ISRC used as key | No extraction |
| Bipartite graph | Artist ↔ Company nodes, ISRC edges | Only adjacency matrix | No graph |
| Shared registrant detection | Flags companies with >1 artist | Lists companies without flagging | Not computed |
| Neo4j query | Cypher MATCH with ISRC join | Python-only implementation | Not attempted |

**Expected finding:** 3 seed artists, 490 tracks, 8 production companies. 0 production companies shared across artists in the seed set (each ghost artist uses an isolated set of registrants — consistent with independent bulk-upload operations, not a coordinated network).

**Common pitfall:** Students confuse ISRC prefix (5 chars: country + registrant) with ISRC registrant code (chars 3–7). The prefix used here is the first 5 characters, which encodes both the country and the registrant identity.

---

### Exercise 4: Graph Centrality / HHI (Catalog Concentration)
**Signal:** S6 — Graph Density / HHI  
**Learning objective:** Compute HHI (Herfindahl-Hirschman Index) as a concentration measure; relate to fraud risk  
**Data source:** Neo4j graph (Track → ProductionCompany edges)  

| Criterion | Full credit | Partial | No credit |
|-----------|-------------|---------|-----------|
| HHI formula correct | HHI = Σ(track_share²) per artist | Uses count instead of share | Incorrect formula |
| All 4 artists computed | RWN, MRC, Calmo, Nils Frahm | ≥2 artists | 1 artist only |
| Signal score derived | HHI → normalized score in [0,1] | Raw HHI reported | Not computed |
| Interpretation | Links high HHI to single-registrant bulk upload | States concentration without cause | No interpretation |

**Expected HHI values:**
- Relaxing White Noise: 0.6715 (score ≈ 0.716)
- Meditation Relax Club: 0.5152 (score ≈ 0.560)
- Calmo: 0.4515 (score ≈ 0.446)
- Nils Frahm: 0.0000 (score = 0.000)

**Common pitfall:** HHI computed at the graph level (across all artists) instead of per-artist. Correct approach: for each artist, compute the share of tracks registered to each production company, then sum the squared shares.

---

### Exercise 5: Release Cadence / Walk Closure (Temporal Patterns)
**Signal:** S2 — Release Cadence Synchrony  
**Learning objective:** Analyze release date gaps; identify bulk-upload clustering; compute walk closure rate  
**Data source:** Spotify /albums (release_date); Neo4j track timeline  

| Criterion | Full credit | Partial | No credit |
|-----------|-------------|---------|-----------|
| Release gap distribution | Histogram per artist with log scale | Bar chart without per-artist | Not computed |
| Closure rate | Tracks with ≤1 day gap / total tracks | Uses month-level granularity | Not computed |
| Median gap | Correct median across all gaps | Mean instead of median | Not computed |
| Walk closure interpretation | >20% closure = suspicious; links to fraud | States rate without threshold | No interpretation |

**Expected values:**
- Relaxing White Noise: 81.0% closure, median gap 0.0 days, 280 tracks over 1156 days
- Meditation Relax Club: 94.7% closure, median gap 0.0 days, 172 tracks over 1240 days
- Calmo: 32.4% closure, median gap 29.0 days, 38 tracks over 2030 days
- Nils Frahm: 3.6% closure, median gap 98.0 days, 56 tracks over 7566 days

**Common pitfall:** Students compute closure rate as "same-day releases / total days" instead of "same-day release gaps / total release gaps." The denominator is the number of consecutive track-to-track gaps, not the number of days in the artist's career.

---

### Exercise 6: Aggregate Signal Score + Radar Chart
**Signal:** All signals combined  
**Learning objective:** Normalize signals to [0,1]; weight and combine; visualize multi-signal radar  
**Data source:** All prior exercises  

| Criterion | Full credit | Partial | No credit |
|-----------|-------------|---------|-----------|
| All 7 signals computed | S1–S7 scores for all 4 artists | ≥5 signals | <5 signals |
| Normalization consistent | Min-max or percentile, documented | Ad hoc per signal | Raw values |
| Radar chart | Matplotlib/Plotly spider chart, all 4 artists overlaid | Bar chart instead | Not visualized |
| Signal weights justified | Weight rationale stated (or equal weights documented) | Weights used without justification | No weighting |

**Expected signal profiles (rule-based):**
- RWN: S2=0.419, S3=0.109, S4=0.757, S5=0.139, S6=0.716 → combined=0.771 (LIKELY_GHOST)
- MRC: S2=0.587, S3=0.109, S4=0.536, S5=0.139, S6=0.560 → SUSPICIOUS
- Calmo: S2=0.232, S3=0.109, S4=0.138, S5=0.139, S6=0.446 → borderline
- NF: S2=0.024, S3=0.109, S4=0.000, S5=0.139, S6=0.000 → LIKELY_ORGANIC

---

### Exercise 7: GNN Ghost Artist Detection
**Signal:** Graph Neural Network classifier  
**Learning objective:** Build a node classification GNN (GAT/GCN) on a synthetic graph; evaluate on test set  
**Data source:** Derived from all prior exercises; Kaggle ghost candidates (11 rows) + organic controls (50 rows)  

| Criterion | Full credit | Partial | No credit |
|-----------|-------------|---------|-----------|
| Graph constructed | Node features (8 dims), edges (ghost fully-connected + organic k-NN) | Flat feature vectors only | Not constructed |
| Two architectures | GAT + GCN implemented, compared | One architecture | Not implemented |
| Train/val/test split | Stratified, reproducible, reported | Random split | No split |
| Results reported | Accuracy, precision, recall, F1, confusion matrix | Accuracy only | Not reported |
| Limitations documented | Synthetic graph structure, small N, potential data leakage noted | One limitation | No limitations |

**Expected results (proof-of-concept):**
- GAT test accuracy: 1.00 (65 nodes, 8 features, 692 edges)
- GCN test accuracy: 0.82–1.00 (varies by seed)
- Rule-based baseline: 0.25
- **Critical note:** 100% accuracy reflects a small, well-separated synthetic graph, not a generalizable model.

**Common pitfall:** Students report 100% accuracy as a strong result without caveating the synthetic graph structure. The expected response is to flag this as a limitation (L6: "graph structure is simulated, not derived from real Spotify co-appearance data").

**Grading note:** A student who reports lower accuracy (60–80%) and correctly identifies the reason (data imbalance, k-NN graph connecting ghost to organic, etc.) should receive full credit for the analysis component even if their numbers differ from the expected values.

---

### Exercise 8: Cross-Platform Discrepancy (Signal 7)
**Signal:** S7 — Cross-Platform Discrepancy  
**Learning objective:** Query YouTube Data API and iTunes Search API; compare cross-platform footprint  
**Data source:** YouTube Data API v3; iTunes Search API (unauthenticated)  

| Criterion | Full credit | Partial | No credit |
|-----------|-------------|---------|-----------|
| YouTube views retrieved | Correct view counts for all 4 artists | Counts for ≥2 artists | Not retrieved |
| Apple Music presence | iTunes Search API queried for all 4 | ≥2 artists | Not queried |
| S7 score computed | Rule-based score normalized to [0,1] | Binary present/absent only | Not computed |
| Surprise finding addressed | Notes that RWN/MRC have 100M+ YT views (ghost ≠ cross-platform absent) | Mentions YouTube presence | Not addressed |

**Expected values:**
- Relaxing White Noise: 353,775,028 YouTube views, Apple Music: Yes
- Meditation Relax Club: 157,581,269 YouTube views, Apple Music: Yes
- Calmo: 155 YouTube views, Apple Music: No
- Nils Frahm: 9,107,596 YouTube views, Apple Music: Yes

**Key grading insight:** The surprise finding (ghost artists ARE cross-platform prominent) is the pedagogically important result of this exercise. A student who correctly observes and explains this — that ghost behavior is Spotify-economic stream farming, not fabricated identity — should receive full credit for interpretation even if their exact view counts differ slightly.

---

### Exercise 9: CrewAI Agent Pipeline
**Signal:** Automated orchestration  
**Learning objective:** Build a multi-agent pipeline that runs all 7 signals and synthesizes a verdict  
**Data source:** All prior data sources  

| Criterion | Full credit | Partial | No credit |
|-----------|-------------|---------|-----------|
| ≥3 agents defined | Separate agents for data collection, signal scoring, synthesis | 2 agents | 1 agent |
| Agent communication | Agents pass structured data (JSON/dict) between tasks | Hard-coded dependencies | No communication |
| Verdict synthesized | Final ghost probability + classification output | Signal scores output only | Not synthesized |
| Error handling | Graceful fallbacks for API failures | Try/except only | No error handling |

---

### Exercise 10: Cross-Exercise Integration
**Signal:** Convergence analysis  
**Learning objective:** Demonstrate that independent signals converge on the same artists; quantify composability  
**Data source:** Outputs from all prior exercises  

| Criterion | Full credit | Partial | No credit |
|-----------|-------------|---------|-----------|
| Convergence matrix | 4 artists × ≥5 exercises boolean flags | ≤3 exercises | Not computed |
| Composability table | Accuracy vs number of signals (1→all) | Accuracy for one combo only | Not computed |
| Figure 8 | Heatmap + accuracy line chart as 1×2 subplot | One plot only | Not generated |
| Written findings | Addresses both convergence and composability with specific numbers | One finding only | Not written |

**Expected convergence:**
- Relaxing White Noise: flagged by 5/5 exercises
- Meditation Relax Club: flagged by 5/5 exercises
- Calmo: flagged by 4/5 exercises
- Nils Frahm: flagged by 0/5 exercises

**Expected composability improvement:** Single best signal (S4 Catalog Density): 0.85 accuracy. Three signals (S2+S4+S6): 1.00 accuracy. All 6 signals: 1.00 accuracy. GNN (all signals + graph): 1.00 accuracy.

---

## Known API Limitations (April 2026)

Students working after February 2026 will encounter these Spotify API restrictions. These are **not bugs** and should not be reported as such:

1. **`/audio-features` endpoint** — Returns 403 for most developer accounts. Audio features must be sourced from the Kaggle dataset.
2. **`/artists/{id}/related-artists`** — Removed. Graph structure must be built from ISRC co-registration, not Spotify recommendations.
3. **`/artists/{id}/top-tracks`** — Still available but returns only 10 tracks.
4. **`followers.total`** — Returns 0 for restricted access. Signal 4 (follower ratio) falls back to track count as a proxy.
5. **Spotify Developer Mode** — Limits to 25 API calls/day for non-approved apps. Students should use caching (`data/raw/cache/`) to avoid exhausting quota.

**Workaround documented in this project:** The `src/api/spotify_client.py` client implements automatic caching at `data/raw/cache/spotify_*.json` and graceful degradation when endpoints return 403.

---

## Grading Summary Table

| Exercise | Points | Key deliverable |
|----------|--------|----------------|
| 1: Catalog Coherence | 25 | PCA scatter + Levene's test |
| 2: Playlist Entropy | 25 | Shannon entropy + ghost zone |
| 3: ISRC Attribution | 25 | Bipartite graph + Cypher query |
| 4: Graph Centrality / HHI | 25 | HHI scores + signal score derivation |
| 5: Release Cadence | 25 | Closure rate + median gap per artist |
| 6: Aggregate Score | 25 | Radar chart + weighted combination |
| 7: GNN Detection | 25 | GAT+GCN, confusion matrix, limitations |
| 8: Cross-Platform | 25 | YouTube/Apple Music + S7 score |
| 9: CrewAI Pipeline | 25 | ≥3 agents, structured verdict output |
| 10: Integration | 25 | Convergence matrix + Figure 8 |
| **Total** | **250** | |
| Bonus: Dashboard | +20 | Streamlit app, deployed |

---

## Project Timeline

| Day | Target | Exercises |
|-----|--------|-----------|
| 1 | Scaffold + API clients | Setup |
| 2 | Data ingestion + Neo4j | Ex 3 |
| 3 | Audio variance + entropy | Ex 1, 2 |
| 4 | ISRC network + HHI | Ex 3, 4 |
| 5 | Release cadence | Ex 5 |
| 6 | Signals 1–6 + radar | Ex 6 |
| 7 | GNN model | Ex 7 |
| 8 | Cross-platform + CrewAI | Ex 8, 9 |
| 9 | Integration notebook | Ex 10 |
| 10 | Dashboard + final submission | Bonus |

---

## Common Student Mistakes (Avoid These)

1. **Using Spotify audio features directly** — The endpoint is restricted since February 2026. Use `data/kaggle/dataset.csv` instead.

2. **Imputing ghost signal values from class means** — Acceptable only when documented. The GNN dataset builder (`src/models/dataset_builder.py`) does this explicitly; students should state any imputation clearly in their write-up.

3. **Treating 100% GNN accuracy as a strong result** — The graph is synthetic (65 nodes, fully connected ghost cluster). The result is a proof-of-concept, not a production claim. Students should document limitation L6.

4. **Not caching API calls** — The Spotify/YouTube APIs have strict rate limits. Any implementation that makes uncached calls in a loop will hit quota within minutes.

5. **Ghost probability threshold at 0.5** — The project uses a tiered threshold: >0.7 = LIKELY_GHOST, 0.4–0.7 = SUSPICIOUS, <0.4 = LIKELY_ORGANIC. A binary 0/1 cutoff at 0.5 misses the SUSPICIOUS category.

6. **Using only one signal for classification** — The pedagogical point of Exercise 10 is that single signals have false positives that composability eliminates. A student who only runs one signal in Ex 10 has missed the central finding.

---

## Frequently Asked Questions

**Q: My HHI score doesn't match the expected value.**  
A: Check that you are computing HHI per-artist, using track counts per production company. The denominator is the total tracks for that artist, not the total tracks in the dataset. Also check that you're using `ISRC[:5]` (first 5 characters) as the production company key, not the full ISRC.

**Q: My closure rate is different from the expected value.**  
A: Check that you are computing gaps between consecutive track release dates for a single artist, and that the denominator is the number of gaps (N-1 for N tracks), not the span in days.

**Q: The YouTube API returns 0 views for a popular artist.**  
A: The `search_video()` call returns a search snippet, which does not include a view count. You must extract the `videoId` from `snippet["id"]["videoId"]` and then make a separate `get_view_count(video_id)` call using the `/videos` endpoint with `part=statistics`.

**Q: My GNN converges but accuracy is low.**  
A: Check that the ghost nodes are fully connected (complete subgraph, not just self-loops). Low accuracy with a correct implementation usually means the test set landed on an unlucky split. Rerun with a different `random_seed`.

**Q: Can I use a different dataset for Exercises 1 and 2?**  
A: Yes. The Kaggle dataset is provided as a convenience. Students may use any dataset with per-track audio features as long as they document the source and explain any differences in their findings.

---

*Prepared for INFO 7390, Spring 2026 — Northeastern University*
