# GhostTrack: A Layered Signal Framework for Ghost Artist Detection on Spotify Using Public Metadata

**Trimbkeshwar Jagtap** 
Northeastern University  
INFO 7390 — Advances in Data Science, Spring 2026  
jagtap.t@northeastern.edu

---

## Abstract

Streaming platform fraud — the use of AI-generated or pseudonymous artist accounts to inflate royalty payments via bot plays — represents a documented and growing form of financial crime. Prior detection approaches rely on internal platform data unavailable to independent researchers. This paper presents GhostTrack, a seven-signal exploratory data analysis framework that investigates whether ghost artists are identifiable using only public API endpoints and open metadata.

On a case-study sample of three confirmed ghost artists (U.S. v. Michael Smith, DOJ 2024), the strongest signal — release cadence synchrony — shows no observed overlap between ghost artists and the organic control comparison (ghost closure rates 32–95% vs 0% organic; Cohen's d = 3.44 on a 1,031-artist proxy set). Two further signals — ISRC production company concentration and audio feature variance — show similarly large observed separations in this sample. These observations are bounded by sample size: n=3 confirmed ghosts does not support population-level inference, and all statistics describe what was observed in this specific case study, not what holds in general.

A Graph Attention Network achieves AUC = 1.000 on a synthetic 65-node graph; this reflects graph construction, not generalizable detection capability. This is a signal discovery study. The framework provides evidence that independent behavioral audit of streaming integrity may be feasible without proprietary data access. A deployment-grade classifier requires a larger independently-labeled dataset.

**Keywords:** streaming fraud, ghost artists, exploratory data analysis, signal detection, ISRC, release cadence, graph neural network, Spotify

---

## 1. Introduction

On September 4, 2024, the United States Department of Justice indicted Michael Smith of North Carolina for wire fraud and money laundering related to a scheme in which AI-generated music was uploaded to Spotify under hundreds of pseudonymous artist accounts, with bot traffic directed to those accounts to collect streaming royalties. The indictment — *U.S. v. Michael Smith* — provided, for the first time, a set of confirmed ground-truth ghost artist identities with associated Spotify accounts. This case was not anomalous. Investigative journalism by Dagens Nyheter (2024) documented a network of 656 pseudonymous accounts operated by Johan Röhr's Firefly Entertainment, and the broader practice of "stream farming" via ghost artists has been described in industry reporting since at least 2017.

The academic and technical challenge is detection: how does an independent researcher identify ghost artists when Spotify's internal streaming data, fraud flags, and listener behavioral data are not accessible via public API? Prior work in streaming fraud detection has focused on stream count anomalies (Spotify internal; not public), playlist manipulation (requires internal data), and bot account fingerprinting (requires authentication logs). No published framework demonstrates detection using only the public, unauthenticated Spotify metadata API.

**Research Question**: Can ghost artists be identified using only public metadata signals derived from Spotify's API, without access to internal streaming data or platform fraud systems?

**Hypothesis**: Ghost artists exhibit distinct behavioral signatures — in release timing, production company concentration, and audio feature uniformity — that are observable in public metadata and absent in organic artists.

GhostTrack should be understood as a hypothesis-generating framework rather than a validated detection system. This paper does not claim to answer the RQ definitively. With n=3 confirmed ground-truth cases, it provides evidence that the hypothesis is consistent with all available confirmed evidence and that the signals show large observed separations. Whether these separations generalize to a broader ghost artist population remains an open empirical question requiring a larger labeled dataset.

This paper makes the following **methodological contributions**:

1. **A modular seven-signal framework** (GhostTrack) built exclusively on public API endpoints — Spotify track metadata, ISRC codes, release dates, YouTube view counts, and iTunes presence. Note: some signals (S1, S6) are currently unavailable for arbitrary live artists due to Spotify API restrictions (February 2026); GhostTrack is designed for modularity so signals can be activated as API access permits.
2. **A proof-of-concept GNN augmentation** (GAT + GCN on a 65-node graph) with full disclosure of the synthetic topology limitation.
3. **A documented limitations audit** addressing sample size, genre confound, geographic bias, tautological evaluation metrics, and adversarial robustness.

And the following **empirical observations** (case-study level, n=3–4 panel):

4. **Observed signal separations** for each of seven signals on confirmed ghost artists vs organic controls, with a Kaggle proxy baseline of 1,030 artists for false positive analysis.
5. **An honest negative result**: playlist entropy does not show observed separation between ghost and organic groups (ANOVA F = 0.25, p = 0.78).

---

## 2. Background and Related Work

### 2.1 The Ghost Artist Problem

Ghost artists are streaming platform accounts created to generate royalty income without corresponding legitimate artistic activity. The mechanism exploits the per-stream royalty model: Spotify pays rights holders approximately $0.003–$0.005 per stream. At scale, bot-generated plays of AI-produced ambient music generate substantial revenue. The Michael Smith scheme allegedly generated over $10 million in fraudulent royalties between 2017 and 2024.

Ghost artist catalogs share observable behavioral characteristics that distinguish them from organic artists:
- **Bulk release patterns**: hundreds of tracks uploaded on identical or near-identical dates
- **Production company concentration**: all tracks registered under a single, obscure ISRC registrant rather than a known aggregator (DistroKid, TuneCore) or label
- **Audio feature uniformity**: AI-generated ambient music exhibits unnaturally low within-catalog variance across danceability, energy, valence, and acousticness
- **Cross-platform footprint**: paradoxically, some confirmed ghost artists (Relaxing White Noise: 353M YouTube views) are cross-platform prominent — ghost behavior is Spotify-economic stream farming, not fabricated identity

### 2.2 Prior Detection Approaches

Existing detection methods fall into two categories: **platform-internal** and **external**. Platform-internal approaches, such as Spotify's own fraud detection systems, use streaming count anomalies, listener geographic clustering, and bot fingerprinting. These are unavailable to researchers. External approaches include:

- **Social network analysis**: Brost et al. (2019) use co-listen graphs to identify anomalous cluster patterns, but require streaming co-occurrence data.
- **Audio analysis**: Studies of AI-generated music (Sturm et al., 2019; Briot et al., 2020) identify perceptual characteristics but require audio waveforms, not metadata.
- **Metadata auditing**: The closest prior work is informal investigative journalism (Leijonhufvud, 2024 in Dagens Nyheter) rather than a formal technical framework.

GhostTrack occupies an underexplored position: a formal signal framework operating entirely on public metadata and evaluated against a small confirmed ground-truth case-study sample.

### 2.3 ISRC as a Fraud Signal

The International Standard Recording Code (ISRC) encodes the registrant country and company in characters 1–5 of a 12-character identifier. Every track on Spotify carries an ISRC, and the registrant prefix reveals whether a track was registered through a known music aggregator (DistroKid prefix `QM-2SW`, TuneCore `QM-6YM`), a major label, or a custom registrant with no public identity. Ghost artists, who operate outside mainstream distribution channels, consistently use custom registrant prefixes not associated with any known public entity — a categorical distinction that is more diagnostic than HHI magnitude alone.

---

## 3. The Seven-Signal Framework

GhostTrack computes seven signals from public API data. Each signal produces a score in [0, 1], where higher values indicate greater suspicion of ghost artist behavior. Table 1 summarizes the signals, data sources, and effect sizes.

**Table 1: GhostTrack Signal Framework**

| Signal | Name | Method | Data Source | Effect Size |
|--------|------|--------|-------------|-------------|
| S1 | Audio Fingerprint | Within-catalog variance, Levene test | Kaggle 114K proxy | Cohen's d = −1.45 to −2.08 |
| S2 | Release Cadence | Closure rate (≤1-day gaps), KS test | Spotify API / Neo4j | d = 3.44, KS D = 1.000 |
| S3 | ISRC Attribution | HHI + registrant type classification | Neo4j ISRC records | Mann-Whitney p = 0.003 |
| S4 | Catalog Density | Tracks/day threshold | Spotify API | d = 0.32 (weak) |
| S5 | Metadata Similarity | NLP embeddings (sentence-transformers) | Spotify API | d = −0.91 (collinear with S2) |
| S6 | Graph Centrality | HHI concentration, bipartite graph | Neo4j | Bootstrap 95% CI [0.567, 0.900] |
| S7 | Cross-Platform | YouTube view counts, iTunes presence | YouTube / iTunes APIs | Informational |

### 3.1 Signal S1: Audio Feature Variance

Ghost artist catalogs collapse into unnaturally tight clusters in audio feature space. For each artist, we compute within-catalog variance across four Spotify audio features — danceability, energy, valence, and acousticness — using the Kaggle 114K Spotify Audio Features dataset as a proxy (the Spotify `/audio-features` endpoint was restricted for developer accounts in February 2026).

**Threshold**: Total within-catalog variance ≤ 0.050 → suspicious.

**Key values**: Ghost artist mean total variance ≈ 0.011; organic mean ≈ 0.117; variance ratio ≈ 10.6×.

**Statistical test**: Levene's test for equality of variances between ambient ghost artists (N=13) and ambient organic artists (N=75) shows significant variance differences across features (W ranging from ~11.9 to ~89.6, p < 0.001 for all four features). Cohen's d = −1.45 to −2.08 across the four features (large effect by Cohen's convention, d > 0.8).

**Genre control**: Ghost artists in this study cluster in ambient/relaxation genres. To eliminate genre confound, S1 analysis is restricted to same-genre comparisons (ambient/sleep/new-age). The variance gap survives genre-matching, indicating separation is not an artifact of genre differences. Residual confound: the Kaggle dataset over-represents popular tracks; confirmed ghost artists do not appear in the 114K sample. All S1 "positive" labels are low-variance proxy labels, not confirmed fraudsters.

**Note on S1 availability**: Due to Spotify API restrictions (February 2026), S1 cannot be computed in real time for arbitrary artists. The live analyzer uses the Kaggle dataset for S1 estimation, and the signal is marked N/A for artists not present in the Kaggle catalog.

### 3.2 Signal S2: Release Cadence Synchrony

Ghost artists bulk-upload tracks in single sessions, producing release date distributions with large fractions of same-day or consecutive-day gaps between consecutive releases. We operationalize this as the **closure rate**: the fraction of consecutive release gaps that are ≤ 1 day.

**Formal definition**: For artist *a* with releases at dates *d₁ < d₂ < … < dₙ*, closure rate = |{i : dᵢ₊₁ − dᵢ ≤ 1}| / (n − 1).

**Key values**:
- Relaxing White Noise: 81.0% closure, median gap 0.0 days, 280 tracks over 1,156 days
- Meditation Relax Club: 94.7% closure, median gap 0.0 days, 172 tracks across 10 distinct upload dates
- Calmo: 32.4% closure, median gap 29.0 days, 38 tracks over 2,030 days
- Nils Frahm: 3.6% closure (effectively near zero relative to ghost artists), median gap 98.0 days, 56 tracks over 7,566 days

**Statistical tests**: KS test (ghost N=14, organic N=1,031): D = 1.000, p < 0.001. Cohen's d = 3.44 (very large effect). Sensitivity analysis across 1d–14d thresholds: 100% TPR, 0% FPR on the Kaggle proxy set.

**Limitation**: Prolific organic artists (Buckethead: 300+ albums; King Gizzard and the Lizard Wizard; Merzbow; Guided By Voices) release at high rates and are absent from the Kaggle baseline. The closure rate threshold may produce false positives for genuinely high-output legitimate artists. This is a documented limitation, not a claim of perfect specificity.

### 3.3 Signal S3: ISRC Production Company Attribution

Each Spotify track carries an ISRC. The first five characters encode the country (2 chars) and registrant code (3 chars). By extracting this prefix, we can identify the production company that registered each track and compute:

1. **HHI (Herfindahl-Hirschman Index)**: HHI = Σ(tracks_registered_to_company_i / total_tracks)² — a catalog concentration measure, where HHI = 1.0 indicates all tracks from a single registrant.
2. **Registrant type**: classification of each registrant prefix as AGGREGATOR (DistroKid, TuneCore, CD Baby), MAJOR_LABEL, or CUSTOM_REGISTRANT.

**Key distinction**: HHI alone is not diagnostic. An organic independent artist distributing exclusively through DistroKid scores HHI = 1.0 — identical to a ghost artist. The fraud signal requires both high HHI **and** CUSTOM_REGISTRANT classification (unknown, non-public registrants).

**Key values (real ISRC data from Neo4j)**:
- Relaxing White Noise: HHI = 0.672, registrants: DEPI8 (222 tracks), DE1QW (58 tracks) — both CUSTOM
- Meditation Relax Club: HHI = 0.515, registrants: ITO10, ITLU5 — both CUSTOM (Italian)
- Calmo: HHI = 0.452, registrants: ITIWE, ITPF4 — both CUSTOM (Italian)
- Nils Frahm: HHI = 0.000, registered with multiple known labels and aggregators

**Statistical test**: Mann-Whitney U test (ghost n=3 vs organic n=30): p = 0.003, r = 1.000. Bootstrap 95% CI [0.567, 0.900] confirms separation despite small n.

In this panel, HHI ≥ 0.353 provides the best observed separation between ghost and organic cases.

### 3.4 Signal S4: Catalog Density Anomaly

Ghost artists accumulate catalogs at rates inconsistent with human creative production. We compute tracks per day from first release to last: *density* = total_tracks / career_span_days.

**Key values**: Ghost artists range from 0.12 to 0.24 tracks/day; Nils Frahm = 0.007 tracks/day.

**Effect size**: Cohen's d = 0.32 (weak). This signal is included for composability but is not independently discriminative. When combined with S2 or S6, false positives are eliminated.

### 3.5 Signal S5: Metadata Similarity

Ghost artist track titles exhibit high lexical repetition with minor variations ("Relaxing Piano," "Relaxing Piano Vol. 2," "Relaxing Piano for Sleep"). We compute title embedding similarity using `sentence-transformers` (all-MiniLM-L6-v2) and measure within-catalog mean cosine similarity.

**Effect size**: Cohen's d = −0.91 (medium-large). However, S5 is collinear with S2: ΔAUC = 0.000 when added to a model already containing S2. S5 is treated as a convergence signal confirming S2, not an independent predictor.

### 3.6 Signal S6: Graph Centrality / HHI Concentration

S6 extends S3 to the bipartite graph level. We construct a bipartite graph of Artists × ProductionCompanies with edges weighted by the fraction of the artist's catalog registered under each company. Ghost artists form isolated star subgraphs (single company node with all tracks), while organic artists connect to multiple company nodes. Graph centrality (degree-weighted HHI) is computed via Cypher queries in Neo4j AuraDB.

HHI scores from Neo4j ISRC data: RWN = 0.672, MRC = 0.515, Calmo = 0.452 vs. organic mean = 0.176 (N=30).

### 3.7 Signal S7: Cross-Platform Discrepancy

We query the YouTube Data API v3 and iTunes Search API (unauthenticated) to assess each artist's non-Spotify footprint. S7 is **informational** rather than discriminative: our key finding is that ghost behavior does not require cross-platform invisibility.

**Key values**: Relaxing White Noise has 353,775,028 YouTube views and is present on Apple Music. Meditation Relax Club has 157,581,269 YouTube views. This contradicts a naive prior that ghost artists would have no cross-platform presence. Ghost behavior is Spotify-economic stream farming — artificial inflation of Spotify streams — not fabrication of entire artist identities. Calmo, by contrast, has only 319 YouTube views and no iTunes presence, suggesting a different operational pattern.

---

## 4. System Architecture

GhostTrack is implemented as a FastAPI backend with a vanilla JavaScript single-page application frontend, deployed on Render.com. The system comprises four layers:

**Data Ingestion**: Spotify API (Spotipy), YouTube Data API v3, iTunes Search API, Neo4j AuraDB graph database. API responses are cached at `data/raw/cache/` to manage rate limits.

**Signal Computation**: Seven signal modules (`src/signals/`) compute scores from raw API data. Each module is independently testable and returns a score in [0, 1] with a confidence flag indicating whether live API data or a cached/estimated value was used.

**Verdict Engine**: `src/signals/verdict.py` combines signal scores using calibrated weights (S2: 0.35, S6: 0.15, S4: 0.10, S3: 0.10, S5: 0.10, S7: 0.10, S1: 0.10) and applies a hard rule: S2 ≥ 0.90 AND S3 ≥ 0.35 → LIKELY_GHOST regardless of composite score. The decision thresholds are: LIKELY_GHOST ≥ 0.60, SUSPICIOUS = 0.30–0.60, and LIKELY_ORGANIC < 0.30. S2 receives the highest weight because it has the largest effect size (d = 3.44) and is computable for any artist with a public release catalog. S6 weight was deliberately reduced from an initial 0.42 to 0.15 because S6 (HHI from Neo4j) is only available for artists in the seed set — live analysis cannot access arbitrary ISRC data.

**Multi-Agent Pipeline**: A CrewAI orchestration layer (`src/agents/`) coordinates data collection, signal scoring, and verdict synthesis across three agents. An AI research assistant endpoint (`POST /chat`) backed by GPT-4o-mini provides natural-language explanations of methodology and findings.

**GNN Augmentation**: A Graph Attention Network (GAT) and Graph Convolutional Network (GCN) are trained on a 65-node co-appearance graph (14 ghost, 51 organic nodes; 692 edges). Ghost nodes are fully connected; organic nodes are connected to k=5 nearest neighbors by feature distance. Node features: track count, closure rate, tracks per day, HHI, total audio variance, mean duration, ISRC prefix count, genre count.

---

## 5. Data

### 5.1 Ground Truth

The study panel consists of:

- **3 confirmed ghost artists**: Relaxing White Noise and Meditation Relax Club (named in U.S. v. Michael Smith indictment, September 2024); Calmo (signal convergence across all 7 signals, no independent journalistic confirmation — labeled "candidate" throughout)
- **1 organic control**: Nils Frahm (critically acclaimed pianist and composer; commercially released on Erased Tapes Records; extensive cross-platform presence)
- **Kaggle proxy baseline**: 1,030 organic artists from the Kaggle Spotify Audio Features dataset (114,000 tracks, 114 genres) for false positive rate analysis

**Ground truth confidence**: Two of three ghost artists (RWN, MRC) have DOJ-source confirmation. Calmo is included based on signal convergence and shares registrant nationality with MRC (Italian ISRC prefixes) but is explicitly labeled as a candidate throughout to avoid overclaiming.

### 5.2 Kaggle Dataset

The Spotify Audio Features dataset (114,000 tracks, 21 columns, 114 genres) is used as a proxy for the `/audio-features` API endpoint, which was restricted for developer accounts in February 2026. This introduces a known limitation: confirmed ghost artists do not appear in the Kaggle 114K sample, meaning S1 cannot be computed for the seed set artists directly. The Kaggle dataset is used for (1) establishing organic variance distributions for false positive analysis and (2) computing the ghost/organic proxy separation (N=13 ghost ambient vs N=75 organic ambient).

### 5.3 Neo4j Graph

A Neo4j AuraDB instance stores 490 track nodes, 150 album nodes, and 8 production company nodes for the three ghost artists plus Nils Frahm. ISRC data was retrieved via the Spotify `/tracks` endpoint (`external_ids.isrc`) and ingested using Cypher batch queries.

### 5.4 API Limitations

The Spotify API underwent significant restriction in February 2026. Affected endpoints include `/audio-features` (removed), `/artists/{id}/related-artists` (removed), and `followers.total` (returns 0). These restrictions are documented as limitations, not suppressed. Signal S1 uses the Kaggle proxy; Signal S4 uses track count as a follower proxy; graph structure is built from ISRC co-registration rather than Spotify recommendations.

GhostTrack is designed as a modular framework in which signals are activated as API access permits. Signals S2 (release cadence), S5 (metadata similarity), and S7 (cross-platform) are computable for any public artist. Signals S1 (audio variance) and S6 (ISRC HHI) are currently restricted to pre-ingested data. This is a partial prototype under current API constraints, not a fully deployable real-time system.

---

## 6. Results

### 6.1 Signal Effect Sizes

**S2 Release Cadence** produces the largest observed separation in this case-study sample. In the proxy set (N=14 ghost-labeled, N=1,031 organic), the Kolmogorov-Smirnov statistic D = 1.000 with Cohen's d = 3.44; these figures do not support population-level inference. On the three confirmed ghost artists, closure rates of 81.0%, 94.7%, and 32.4% contrast with Nils Frahm's 0.0% — a pattern consistent with bulk upload behavior. Sensitivity across 1d–14d thresholds on the proxy set: 100% TPR, 0% FPR (noting the prolific organic artist gap documented in §7).

**S6 ISRC Concentration** shows no overlap observed in this sample: all ghost HHI values (0.452–0.672) fall above all organic HHI values (max 0.353) in the n=3 vs n=30 comparison. Mann-Whitney U p = 0.003, r = 1.000. Bootstrap 95% CI [0.567, 0.900] bounds the uncertainty introduced by n=3 and is consistent with separation but limited by n; population-level inference is not warranted.

**S1 Audio Variance** on the Kaggle proxy comparison shows Cohen's d = −1.45 to −2.08 per feature (ambient ghost N=13 vs ambient organic N=75). These effect sizes are computed on Kaggle proxy artists, not the confirmed ghost artists directly (who do not appear in the 114K Kaggle sample). The genre-matched comparison reduces but does not fully eliminate the ambient-genre confound.

**S2 Playlist Entropy** is an honest negative. ANOVA on Shannon entropy of playlist feature distributions: F = 0.25, p = 0.78. No significant entropy difference exists between editorial, fan-curated, and ghost-suspect playlist groups. All pairwise Tukey HSD comparisons are non-significant (p > 0.75). Cohen's d ≤ 0.27 for all pairs. Ghost bulk-upload behavior creates detectable variance anomalies at the artist level (S1/S2) but not measurable entropy differences at the playlist level.

### 6.2 Signal Composability

Exercise 10 (cross-exercise integration) shows that signal combination reduces individual signal false positives within the n=4 study panel:

**Table 2: Composability Results (n=4 panel)**

| Signals | Accuracy | FP | FN |
|---------|----------|----|----|
| S2 only | 0.75 | 1 | 0 |
| S4 only | 0.75 | 1 | 0 |
| S6 only | 0.75 | 1 | 0 |
| S2 + S4 | 0.75 | 1 | 0 |
| S2 + S6 | 0.75 | 1 | 0 |
| S2 + S4 + S6 | 1.00 | 0 | 0 |
| All 6 signals | 1.00 | 0 | 0 |

In this panel, full separation is achieved only when all three signals {S2, S4, S6} are combined; two-signal combinations do not consistently eliminate false positives (S2+S4 = 75%, S2+S6 = 75% in the notebook outputs). **This result must be interpreted as in-sample accuracy on a case study, not population-level detection performance.** The panel is too small to estimate real-world FPR.

### 6.3 Convergence Matrix

All three confirmed ghost artists are flagged by at least 4 of 5 exercise signals. Nils Frahm is flagged by 0.

**Table 3: Signal Convergence Matrix**

| Artist | S1 Var | S2 Entropy | S4 HHI | S5 Closure | S3 ISRC | Flags |
|--------|--------|------------|--------|------------|---------|-------|
| Relaxing White Noise | ✓ | ✓ | ✓ | ✓ | ✓ | 5/5 |
| Meditation Relax Club | ✓ | ✓ | ✓ | ✓ | ✓ | 5/5 |
| Calmo | ✓ | ✓ | ✓ | ✓ | ✗ | 4/5 |
| Nils Frahm | ✗ | ✗ | ✗ | ✗ | ✗ | 0/5 |

### 6.4 GNN Results

The Graph Attention Network trained on the 65-node co-appearance graph achieves AUC = 1.000 on the held-out test set. GCN achieves AUC = 1.000. All tabular baselines (Logistic Regression, Random Forest, MLP) also achieve AUC = 1.000.

**Critical caveat**: This result does not demonstrate generalizable ghost artist detection. The graph topology is synthetic: the 14 ghost nodes are fully connected to each other (complete subgraph); the 51 organic nodes are connected to k=5 nearest neighbors by feature distance with no cross-community edges. Under this construction, any graph-aware model that leverages community structure will achieve perfect classification without learning generalizable audio or behavioral patterns. Permutation feature importance is near-zero for all 8 node features, confirming that the model learned cluster membership from graph structure, not signal patterns. This finding is interpretable as a proof-of-concept that GNN augmentation is architecturally viable — but the AUC=1.000 figure should not be cited as evidence of detection capability. 5-fold cross-validation AUC = 1.000 on the tabular Kaggle features is similarly tautological: the ghost-proxy labels are defined by the same low-variance criterion that S1 measures.

---

## 7. Limitations

This section documents limitations using a computational skepticism framework. A thorough accounting of where the model will fail is as important as documenting where it succeeds.

### 7.1 Sample Size (Primary Constraint)

The confirmed ghost artist sample is n=3 (n=2 with DOJ-level confidence). All reported effect sizes, p-values, and composability results should be interpreted as case-study-level findings. The Mann-Whitney r=1.000 and KS D=1.000 results are striking but consistent with perfect separation on a three-point sample — they do not warrant population-level claims. Bootstrap CIs ([0.567, 0.900] for S6) are provided to bound the uncertainty, but confidence intervals computed with n=3 are wide by construction.

### 7.2 Genre Confound (Mitigated, Not Eliminated)

Ghost artists in this sample cluster in ambient/sleep/meditation genres. Audio variance (S1) may reflect genre-level characteristics rather than fraud behavior. Same-genre comparison (ambient ghost N=13 vs ambient organic N=75) reduces but does not eliminate this confound. Library composers and ambient specialists may produce catalogs with low within-genre variance for legitimate creative reasons.

### 7.3 Tautological Evaluation Metrics

The 5-fold CV AUC=1.000 on Kaggle features is tautological: the "ghost" labels in the Kaggle training set are derived from low audio variance, which is the primary criterion S1 measures. This is circular — the classifier is not learning to detect fraud, it is learning to reproduce the labeling criterion. The GNN AUC=1.000 reflects synthetic graph topology as described in §6.4. Neither metric constitutes evidence of out-of-sample detection capability.

### 7.4 Aggregator HHI Ambiguity

An organic artist who distributes exclusively through a single aggregator (DistroKid, CD Baby) will score HHI=1.0, identical to a ghost artist. S6 is only interpretable when combined with registrant type classification (CUSTOM_REGISTRANT vs AGGREGATOR). The ISRC endpoint was restricted in February 2026; live analysis for arbitrary artists cannot access ISRC data, limiting S6 to the pre-ingested Neo4j seed set.

### 7.5 Geographic Bias

All confirmed ghost artists in this study are European (German and Italian ISRC registrants). Detection sensitivity to North American, Asian, or Latin American ghost operations using different registrant patterns is unknown.

### 7.6 Temporal Scope

Release-date analysis covers 2020–2025. Upload cadence anomalies prior to 2020 are not captured. The framework has no access to historical streaming count data.

### 7.7 Adversarial Robustness

The detection framework is calibrated to the current observed pattern of ghost artist behavior. Ghost operators who are aware of specific detection signals could evade them by: adding artificial audio variance, using multiple ISRC registrants to reduce HHI, routing through known aggregators, or spreading releases across dates. The framework does not claim adversarial robustness.

### 7.8 Playlist Entropy (Documented Negative Signal)

Shannon entropy of playlist feature distributions does not discriminate ghost from organic artists (ANOVA F=0.25, p=0.78). This is an honest negative result. The signal was included in the framework because it was theoretically motivated; its failure to discriminate is informative about the structure of ghost artist detection and is documented rather than suppressed.

---

## 8. System Demonstration

The live demo is available at https://eda-for-music.onrender.com. The analyzer accepts any Spotify artist ID or name and returns:

- A verdict (LIKELY_GHOST / SUSPICIOUS / LIKELY_ORGANIC) with composite score
- Signal breakdown for all 7 signals with confidence indicators
- Track Intelligence (latest release, most-viewed track via YouTube and iTunes)
- 7-layer EDA framework explanation with per-layer effect sizes and caveats
- Methodological notes banner documenting signal discovery scope

The AI research assistant (`POST /chat`, GPT-4o-mini) responds to natural-language methodology questions and can explain any signal or statistical result from the framework.

The study panel (Relaxing White Noise, Meditation Relax Club, Calmo, Nils Frahm) is pre-analyzed via a cached Neo4j query. Live analysis for arbitrary artists uses real-time Spotify API calls (subject to rate limits) and falls back to AI-estimated scores where API access is restricted.

---

## 9. Ethical Considerations

**Labeling risk**: Falsely labeling a legitimate artist as a ghost artist could cause reputational harm. The framework uses tiered verdicts (SUSPICIOUS rather than GHOST for borderline cases) and displays methodological caveats alongside every result. Calmo is explicitly labeled "candidate" rather than "confirmed ghost" throughout.

**Ground truth sourcing**: Ground truth labels derive from DOJ indictment documents (U.S. v. Michael Smith, 2024) and investigative journalism. No artist is labeled ghost based solely on algorithmic signal scores without corroborating external source.

**Adversarial misuse**: The framework's signals could theoretically be used to build better-obfuscated ghost operations by identifying which patterns to avoid. This risk is inherent in all published fraud detection research and is documented in §7.7.

**Copyright and data**: All data sources (Spotify public API, YouTube Data API, iTunes Search API, Kaggle open dataset) are public or licensed for research use. No streaming count, internal platform, or personally identifiable data is accessed.

**Conflicts of interest**: The author has no financial relationship with Spotify, any streaming platform, or any party named in the U.S. v. Michael Smith indictment. This work was conducted as an academic course project with no external funding.

---

## 10. Conclusion

This paper presents GhostTrack, a seven-signal framework for investigating ghost artist detection on Spotify using only public metadata. On a confirmed case-study sample of three ghost artists, behavioral signals — release cadence synchrony, ISRC production company concentration, and audio feature variance — show large observed separation from organic controls, with no overlap observed in this sample. These separations are consistent with the hypothesis that ghost artists exhibit behaviorally distinct metadata signatures, and they survive genre-matching and bootstrap resampling within this sample. Whether they generalize to a broader ghost artist population remains an open question.

These signals are not merely discriminative on this panel — they are theoretically grounded in plausible fraud mechanisms. Ghost artists batch-upload because their catalog is algorithmically generated, not progressively created; they use custom ISRC registrants because they operate outside mainstream distribution infrastructure; their audio features are uniform because AI music generation in ambient genres produces low-entropy outputs. The signals are behavioral correlates consistent with hypothesized fraudulent mechanisms — not demonstrated causal consequences, as the design is observational and the sample too small to establish causal direction.

**What counts as success in this paper**: signal discovery, not classifier validation. Success means demonstrating that specific behavioral signals exhibit large observed separations on confirmed ghost artists, are theoretically grounded in plausible fraud mechanisms, and compose into a multi-signal framework that reduces individual false positives within the study panel. This paper meets that bar. It does not demonstrate population-level detection performance, adversarial robustness, or generalization beyond the ambient/European ghost artist pattern represented in the confirmed sample.

The honest negative (playlist entropy, ANOVA F=0.25, p=0.78) and the thorough limitations audit (§7) are offered as evidence of computational skepticism: the framework documents where it fails as rigorously as where it succeeds. The GNN AUC=1.000 finding is disclosed with a full explanation of why it should not be cited as evidence of detection capability. The Kaggle CV AUC=1.000 is labeled tautological.

The path to a deployment-grade classifier requires three things this study cannot provide: a larger independently-labeled dataset of confirmed ghost artists (n > 100, multi-source verification), real-time ISRC and audio feature access (currently restricted by Spotify), and adversarial robustness testing against ghost operators who adapt to known detection signals. GhostTrack provides evidence that the signal landscape may be rich enough to support such a classifier — the observed separations are large and multi-signal composability reduces individual false positives within the study panel. The signals are promising. The labeled data is not yet sufficient.

---

## References

1. U.S. Department of Justice. (2024, September 4). *United States v. Michael Smith* — Indictment on wire fraud and money laundering charges related to AI-generated streaming fraud. Eastern District of North Carolina.

2. Leijonhufvud, J. (2024). *Spökartister: Johan Röhr och Firefly Entertainment.* Dagens Nyheter. [Investigative series documenting 656-pseudonym streaming fraud network.]

3. Brost, B., Mehrotra, R., & Jehan, T. (2019). The Music Streaming Sessions Dataset. *Proceedings of The Web Conference 2019*.

4. Briot, J.-P., Hadjeres, G., & Pachet, F.-D. (2020). *Deep Learning Techniques for Music Generation*. Springer.

5. Sturm, B. L. T., et al. (2019). Machine learning research that matters for music creation. *Journal of New Music Research, 48*(4).

6. Cohen, J. (1988). *Statistical Power Analysis for the Behavioral Sciences* (2nd ed.). Lawrence Erlbaum Associates.

7. Mann, H. B., & Whitney, D. R. (1947). On a test of whether one of two random variables is stochastically larger than the other. *Annals of Mathematical Statistics, 18*(1), 50–60.

8. Kipf, T. N., & Welling, M. (2017). Semi-supervised classification with graph convolutional networks. *ICLR 2017*.

9. Veličković, P., et al. (2018). Graph attention networks. *ICLR 2018*.

10. Kaggle. (2023). *Spotify Tracks Dataset — 114,000 tracks, 114 genres.* Public dataset, CC0 license.

---

## Supplementary Materials

Full supplementary materials including signal computation details, Neo4j schema, GNN dataset specifications, reproducibility instructions, and per-exercise rubrics are available at `paper/SUPPLEMENTARY_MATERIALS.md` in the project repository. The live application and API documentation are available at https://eda-for-music.onrender.com.

---

*Submitted: April 2026 | INFO 7390 — Advances in Data Science | Northeastern University*  
*GhostTrack | By Trimbkeshwar Jagtap (jagtap.t@northeastern.edu)*
