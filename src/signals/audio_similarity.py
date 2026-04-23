"""
Signal 1: Audio Fingerprint Similarity

Ghost artists produce tracks that are nearly identical in audio feature space
because they're optimized for narrow playlist targets.

Input: list of track IDs belonging to a cluster of artists
Output: similarity score 0.0 (all different) to 1.0 (all identical)

Methodology:
- Load audio features from Kaggle dataset (via kaggle_loader.py)
- Compute pairwise cosine similarity matrix over 6 features
- Score = mean of upper triangle (excluding diagonal) of similarity matrix
- Ghost artists score > 0.90 (near-identical tracks)
- Organic artists score ~ 0.50–0.75 (diverse catalogs)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

from src.utils.kaggle_loader import kaggle

# Features used for fingerprinting — 6D fingerprint
FINGERPRINT_FEATURES = [
    "danceability", "energy", "valence",
    "acousticness", "tempo", "speechiness",
]

# Suspicion thresholds (mean pairwise cosine similarity)
HIGH_THRESHOLD = 0.88   # > 0.88 → ghost-like
MEDIUM_THRESHOLD = 0.75  # > 0.75 → suspicious

_GRAPH_DF_CACHE = None


def _load_graph_df():
    """Load the processed artist-track graph once per process."""
    global _GRAPH_DF_CACHE
    if _GRAPH_DF_CACHE is not None:
        return _GRAPH_DF_CACHE

    from pathlib import Path

    cache_path = Path("data/processed/neo4j_full_graph.csv")
    if not cache_path.exists():
        _GRAPH_DF_CACHE = pd.DataFrame()
        return _GRAPH_DF_CACHE

    _GRAPH_DF_CACHE = pd.read_csv(cache_path)
    return _GRAPH_DF_CACHE


def score_tracks(track_ids: list[str], artist_name: str = "unknown") -> dict:
    """
    Compute audio fingerprint similarity score for a set of track IDs.

    Parameters
    ----------
    track_ids   : list of Spotify track IDs
    artist_name : display name for logging

    Returns
    -------
    dict with keys:
        track_count, kaggle_hit_count, kaggle_hit_rate,
        mean_cosine_similarity, std_cosine_similarity,
        min_similarity, max_similarity,
        suspicion_level ("HIGH"/"MEDIUM"/"LOW"),
        suspicion_score (0.0–1.0),
        features_used (list)
    """
    if not track_ids:
        return _empty_result()

    feat_df = kaggle.get_features(track_ids)
    available = [f for f in FINGERPRINT_FEATURES if f in feat_df.columns]
    if not available:
        logger.warning(f"{artist_name}: no audio features available in Kaggle dataset")
        return _empty_result()

    valid = feat_df[available].dropna()
    hit_count = len(valid)
    hit_rate = hit_count / len(track_ids)

    if hit_count < 3:
        logger.warning(
            f"{artist_name}: only {hit_count}/{len(track_ids)} tracks in Kaggle "
            "(need ≥3 for similarity). Returning partial result."
        )
        return {
            **_empty_result(),
            "track_count": len(track_ids),
            "kaggle_hit_count": hit_count,
            "kaggle_hit_rate": hit_rate,
        }

    # Normalise before cosine similarity
    scaler = StandardScaler()
    scaled = scaler.fit_transform(valid[available])

    sim_matrix = cosine_similarity(scaled)

    # Upper triangle excluding diagonal
    n = len(sim_matrix)
    upper_vals = sim_matrix[np.triu_indices(n, k=1)]

    mean_sim = float(np.mean(upper_vals))
    std_sim = float(np.std(upper_vals))
    min_sim = float(np.min(upper_vals))
    max_sim = float(np.max(upper_vals))

    suspicion_score = _sim_to_suspicion(mean_sim)
    suspicion_level = _level(suspicion_score)

    logger.info(
        f"{artist_name}: mean_cosine_similarity={mean_sim:.3f}, "
        f"suspicion={suspicion_level} ({suspicion_score:.2f}), "
        f"kaggle_hit_rate={hit_rate:.1%}"
    )

    return {
        "track_count": len(track_ids),
        "kaggle_hit_count": hit_count,
        "kaggle_hit_rate": round(hit_rate, 4),
        "mean_cosine_similarity": round(mean_sim, 4),
        "std_cosine_similarity": round(std_sim, 4),
        "min_similarity": round(min_sim, 4),
        "max_similarity": round(max_sim, 4),
        "suspicion_level": suspicion_level,
        "suspicion_score": round(suspicion_score, 4),
        "features_used": available,
    }


def score_artist(artist_id: str) -> dict:
    """
    Score a single artist using cached track data from processed CSVs.

    Reads data/processed/neo4j_full_graph.csv for track IDs,
    then joins with Kaggle dataset for audio features.

    Returns score dict (see score_tracks).
    """
    # Try to load track IDs from cached exercise 4 data (neo4j_full_graph.csv)
    df = _load_graph_df()
    if not df.empty:
        artist_tracks = df[df["artist_id"] == artist_id]["track_id"].dropna().tolist()
        artist_name = df[df["artist_id"] == artist_id]["artist_name"].iloc[0] if len(df[df["artist_id"] == artist_id]) else artist_id
        if artist_tracks:
            result = score_tracks(artist_tracks, artist_name)
            result["artist_id"] = artist_id
            result["artist_name"] = artist_name
            return result

    logger.warning(f"score_artist: no cached track data found for {artist_id} in neo4j_full_graph.csv")
    return {**_empty_result(), "artist_id": artist_id, "artist_name": artist_id}


def score_cluster(artist_ids: list[str]) -> dict:
    """
    Score across multiple artists — measures how similar the COMBINED
    track pool is (ghost networks all produce the same-sounding tracks).

    Returns score dict with additional key: artist_count.
    """
    df = _load_graph_df()
    if df.empty:
        logger.warning("score_cluster: neo4j_full_graph.csv not found")
        return _empty_result()

    cluster_df = df[df["artist_id"].isin(artist_ids)]
    track_ids = cluster_df["track_id"].dropna().unique().tolist()
    result = score_tracks(track_ids, f"cluster({len(artist_ids)} artists)")
    result["artist_count"] = len(artist_ids)
    result["artist_ids"] = artist_ids
    return result


# ──────────────────────────── helpers ────────────────────────────


def _sim_to_suspicion(mean_sim: float) -> float:
    """
    Map mean cosine similarity → suspicion score [0, 1].
    Anchored at empirical thresholds from ghost/organic seed calibration.
    """
    # sigmoid centred at HIGH_THRESHOLD (0.88)
    import math
    score = 1.0 / (1.0 + math.exp(-15.0 * (mean_sim - HIGH_THRESHOLD)))
    return float(np.clip(score, 0.0, 1.0))


def _level(score: float) -> str:
    if score >= 0.7:
        return "HIGH"
    elif score >= 0.4:
        return "MEDIUM"
    return "LOW"


def _empty_result() -> dict:
    return {
        "track_count": 0,
        "kaggle_hit_count": 0,
        "kaggle_hit_rate": 0.0,
        "mean_cosine_similarity": None,
        "std_cosine_similarity": None,
        "min_similarity": None,
        "max_similarity": None,
        "suspicion_level": "UNKNOWN",
        "suspicion_score": None,
        "features_used": [],
    }
