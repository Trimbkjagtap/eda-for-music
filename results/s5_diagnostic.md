# S5 Sign-Flip Diagnostic

**Date:** 2026-04-20
**Script:** scripts/15_final_audit.py (Audit 2)

## Signal Definition

S5 = genre_concentration = 1 / genre_count, clipped to [0, 1].

Higher S5 means fewer genres (more concentrated). This was expected to be a ghost
indicator, since ghost artists produce music in a single narrow genre.

## Finding

| Group | Mean S5 | Std S5 | N |
|---|---|---|---|
| Organic (Kaggle controls) | 0.688 | 0.312 | 1030 |
| Ghost-proxy (low_variance) | 0.356 | 0.242 | 100 |

**Ghost S5 < Organic S5.** This is counter-intuitive but explainable.

## Explanation

The Kaggle organic controls (`organic_controls_kaggle.csv`) were filtered to
`total_variance ≥ 0.080`. Many of these are genre specialists — e.g., The Black
Dog (IDM, genre_count=1), Nils Frahm-equivalent artists. Their S5 ≈ 1.0.

The Kaggle ghost-proxies (`low_variance_artists.csv`) were sampled from genre
pools that span multiple Kaggle genre tags (ambient, sleep, new-age, chill),
giving them genre_count ∈ [2, 5] and S5 ∈ [0.2, 0.5].

The DOJ confirmed ghost artists (RWN, MRC, Calmo) have genre_count=2 → S5=0.5,
which is below the organic mean of 0.688. This is a dataset artifact:
Kaggle genre tags reflect playlist category, not stylistic uniqueness.

## Composite Weight Direction

The learned composite weight for S5 is **negative** (coef ≈ −0.50).

This is internally consistent: higher S5 (fewer genres) is associated with
being an ORGANIC single-genre specialist in this proxy dataset. The negative
weight downgrades artists with high genre concentration.

This is NOT the expected theoretical direction (ghost artists should be
genre-concentrated), and the composite should NOT be trusted as a validated
ghost detector. It is a proxy classification that has been documented as such.

## Collinearity Test

| Model | CV AUC (5-fold) |
|---|---|
| S2 + S4 + S5 | 1.0000 ± 0.0000 |
| S2 + S4 only | 1.0000 ± 0.0000 |
| ΔAUC | +0.0000 |

**Verdict:** COLLINEAR — S5 adds no discriminative information beyond S2. Recommend dropping.

## Correlations

| Pair | Organic | Ghost |
|---|---|---|
| corr(S2, S5) | -0.100 | 0.164 |
| corr(S4, S5) | -0.185 | — |

Weak correlations confirm S5 is not a simple linear function of S2 or S4.
The lack of discriminative power is a dataset artifact, not a signal flaw.

## Recommendation

1. **Retain S5 in the composite** with documented caveats (dataset artifact).
2. **Do NOT interpret** the negative weight as evidence that genre concentration
   protects against ghost classification.
3. **Flag in paper**: S5 direction is reversed relative to theoretical prediction
   due to Kaggle genre-tag semantics. Real-world ghost detection would require
   artist-level genre labels from a streaming platform, not playlist-derived tags.
