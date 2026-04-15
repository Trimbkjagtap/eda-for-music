"""
Signal 2: Release Cadence Synchrony

Ghost artist networks release albums/singles on coordinated schedules.
Isolation Forest detects anomalously regular release intervals.

Input: list of artist IDs
Output: synchrony score 0.0 (random timing) to 1.0 (perfectly synchronized)

Methodology:
- Get album release dates from Neo4j (no API calls needed — 91 albums cached)
- Compute coefficient of variation (CV) of inter-release intervals per artist
  Low CV = suspiciously regular cadence
- Compute closure rate: fraction of consecutive tracks released same-day
  (from exercise5_walk_metrics.csv — already computed)
- For a cluster: compute pairwise Jaccard overlap of release date sets
- Use Isolation Forest to flag anomalous regularity in gap sequences
- Score = weighted combination of signals

Thresholds calibrated on seed set:
  MRC: 94.7% closure, RWN: 81.0% closure → GHOST
  Calmo: 32.4% closure → SUSPICIOUS
  Nils Frahm: 3.6% closure → ORGANIC
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from datetime import date, datetime
from collections import Counter
from loguru import logger
from sklearn.ensemble import IsolationForest

from src.graph.neo4j_client import Neo4jClient

# Pre-computed closure rates from Exercise 5
_KNOWN_CLOSURE = {
    "6bo3atMVp3qFECNALVwq9N": 0.810,   # Relaxing White Noise
    "39t4EeLBfpT72UQJVkIeuj": 0.947,   # Meditation Relax Club
    "4Wx3ZL6d6p1gVMtwQ2YWsz": 0.324,   # Calmo
    "5gqhueRUZEa7VDnQt4HODp": 0.036,   # Nils Frahm
}

# Walk metrics CSV path
_WALK_METRICS_PATH = Path("data/processed/exercise5_walk_metrics.csv")


def score_artist(artist_id: str, neo4j: Neo4jClient | None = None) -> dict:
    """
    Compute release cadence suspicion score for one artist.

    Uses Neo4j album release dates + pre-computed closure from Exercise 5.
    Zero API calls.

    Returns
    -------
    dict with keys:
        artist_id, artist_name, total_releases, unique_release_dates,
        mean_gap_days, std_gap_days, cv_gap, burst_ratio, closure_rate,
        anomaly_rate, suspicion_level, suspicion_score
    """
    _neo4j = neo4j or Neo4jClient()

    # Get artist name
    rows = _neo4j.run(
        "MATCH (a:Artist {spotify_id: $id}) RETURN a.name AS name",
        id=artist_id,
    )
    artist_name = rows[0]["name"] if rows else artist_id

    # Get album release dates from Neo4j
    date_rows = _neo4j.run(
        """
        MATCH (a:Artist {spotify_id: $id})-[:RELEASED]->(al:Album)
        WHERE al.release_date IS NOT NULL AND al.release_date <> ''
        RETURN al.release_date AS dt
        ORDER BY al.release_date
        """,
        id=artist_id,
    )
    raw_dates = [r["dt"] for r in date_rows]

    # Also try cached Spotify albums from exercise5 walk metrics
    closure_rate = _KNOWN_CLOSURE.get(artist_id)
    if closure_rate is None and _WALK_METRICS_PATH.exists():
        wm = pd.read_csv(_WALK_METRICS_PATH)
        # match by partial name
        match = wm[wm["Artist"].str.lower().str.contains(artist_name.lower()[:8], na=False)]
        if not match.empty:
            closure_rate = float(match.iloc[0]["Closure (≤1d gap %)"] / 100.0)

    parsed = _parse_dates(raw_dates)

    if len(parsed) < 2:
        logger.warning(f"{artist_name}: only {len(parsed)} release dates in Neo4j")
        return {
            **_empty_result(artist_id, artist_name),
            "total_releases": len(parsed),
            "closure_rate": closure_rate or 0.0,
            "suspicion_score": _closure_to_suspicion(closure_rate or 0.0),
            "suspicion_level": _level(_closure_to_suspicion(closure_rate or 0.0)),
        }

    parsed.sort()
    date_counts = Counter(d.isoformat() for d in parsed)
    gaps = np.diff([d.toordinal() for d in parsed])

    mean_gap = float(np.mean(gaps))
    std_gap = float(np.std(gaps))
    cv_gap = std_gap / mean_gap if mean_gap > 0 else 0.0
    median_gap = float(np.median(gaps))

    unique_dates = len(date_counts)
    max_single_day = max(date_counts.values())
    single_day_fraction = max_single_day / len(parsed)

    # Burst ratio: fraction of releases in densest 7-day window
    burst_ratio = _compute_burst_ratio(parsed, 7)

    # Isolation Forest on gaps
    anomaly_rate = 0.0
    if len(gaps) >= 5:
        X = gaps.reshape(-1, 1).astype(float)
        iso = IsolationForest(contamination=0.1, random_state=42)
        preds = iso.fit_predict(X)
        anomaly_rate = float((preds == -1).sum() / len(preds))

    # Use pre-computed closure if available, else estimate from single_day_fraction
    if closure_rate is None:
        closure_rate = single_day_fraction

    suspicion_score = _compute_suspicion(
        closure_rate=closure_rate,
        burst_ratio=burst_ratio,
        single_day_fraction=single_day_fraction,
        cv_gap=cv_gap,
    )
    suspicion_level = _level(suspicion_score)

    logger.info(
        f"{artist_name}: closure={closure_rate:.1%}, burst={burst_ratio:.2f}, "
        f"cv_gap={cv_gap:.2f}, suspicion={suspicion_level} ({suspicion_score:.2f})"
    )

    return {
        "artist_id": artist_id,
        "artist_name": artist_name,
        "total_releases": len(parsed),
        "unique_release_dates": unique_dates,
        "mean_gap_days": round(mean_gap, 1),
        "std_gap_days": round(std_gap, 1),
        "median_gap_days": round(median_gap, 1),
        "cv_gap": round(cv_gap, 4),
        "burst_ratio": round(burst_ratio, 4),
        "single_day_fraction": round(single_day_fraction, 4),
        "closure_rate": round(closure_rate, 4),
        "anomaly_rate": round(anomaly_rate, 4),
        "suspicion_level": suspicion_level,
        "suspicion_score": round(suspicion_score, 4),
        "release_dates": [d.isoformat() for d in parsed],
        "date_counts": dict(date_counts),
    }


def score_cluster(artist_ids: list[str], neo4j: Neo4jClient | None = None) -> dict:
    """
    Score a cluster of artists — measures how synchronized their release calendars are.
    Returns individual scores + cross-artist Jaccard date overlap.
    """
    _neo4j = neo4j or Neo4jClient()
    results = [score_artist(aid, _neo4j) for aid in artist_ids]

    # Pairwise Jaccard overlap of release date sets
    date_sets = {
        r["artist_id"]: set(r.get("release_dates", []))
        for r in results
    }
    pairwise_overlaps = []
    ids = list(date_sets.keys())
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            a_set = date_sets[ids[i]]
            b_set = date_sets[ids[j]]
            union = len(a_set | b_set)
            inter = len(a_set & b_set)
            jaccard = inter / union if union > 0 else 0.0
            pairwise_overlaps.append(jaccard)

    synchrony_score = float(np.mean(pairwise_overlaps)) if pairwise_overlaps else 0.0
    mean_individual = float(np.mean([r["suspicion_score"] for r in results]))
    cluster_score = 0.6 * mean_individual + 0.4 * synchrony_score

    return {
        "artist_ids": artist_ids,
        "individual_scores": results,
        "cross_artist_jaccard": round(synchrony_score, 4),
        "mean_individual_suspicion": round(mean_individual, 4),
        "cluster_suspicion_score": round(cluster_score, 4),
        "cluster_suspicion_level": _level(cluster_score),
    }


# ──────────────────────────── helpers ────────────────────────────


def _parse_dates(raw: list[str]) -> list[date]:
    out = []
    for d in raw:
        try:
            if len(d) == 4:
                out.append(date(int(d), 1, 1))
            elif len(d) == 7:
                y, m = d.split("-")
                out.append(date(int(y), int(m), 1))
            elif len(d) == 10:
                out.append(datetime.strptime(d, "%Y-%m-%d").date())
        except Exception:
            pass
    return out


def _compute_burst_ratio(dates: list[date], window_days: int = 7) -> float:
    if len(dates) <= 1:
        return 0.0
    ordinals = sorted(d.toordinal() for d in dates)
    max_in_window = 0
    j = 0
    for i, o in enumerate(ordinals):
        while j < len(ordinals) and ordinals[j] <= o + window_days:
            j += 1
        max_in_window = max(max_in_window, j - i)
    return max_in_window / len(ordinals)


def _closure_to_suspicion(closure: float) -> float:
    """Map closure rate to suspicion score. 95% closure → ~1.0, 4% → ~0.05."""
    import math
    score = 1.0 / (1.0 + math.exp(-8.0 * (closure - 0.5)))
    return float(np.clip(score, 0.0, 1.0))


def _compute_suspicion(
    closure_rate: float,
    burst_ratio: float,
    single_day_fraction: float,
    cv_gap: float,
) -> float:
    """Weighted combination of cadence signals → suspicion score [0, 1]."""
    # closure_rate is the strongest signal (from Exercise 5)
    score = 0.5 * closure_rate
    score += 0.25 * min(burst_ratio, 1.0)
    score += 0.15 * min(single_day_fraction, 1.0)
    # Low CV (regular spacing) is also suspicious, but weaker signal
    cv_signal = max(0.0, 1.0 - cv_gap / 2.0) if cv_gap < 2.0 else 0.0
    score += 0.10 * cv_signal
    return float(np.clip(score, 0.0, 1.0))


def _level(score: float) -> str:
    if score >= 0.60:
        return "HIGH"
    elif score >= 0.35:
        return "MEDIUM"
    return "LOW"


def _empty_result(artist_id: str, artist_name: str) -> dict:
    return {
        "artist_id": artist_id,
        "artist_name": artist_name,
        "total_releases": 0,
        "unique_release_dates": 0,
        "mean_gap_days": 0.0,
        "std_gap_days": 0.0,
        "median_gap_days": 0.0,
        "cv_gap": 0.0,
        "burst_ratio": 0.0,
        "single_day_fraction": 0.0,
        "closure_rate": 0.0,
        "anomaly_rate": 0.0,
        "suspicion_level": "UNKNOWN",
        "suspicion_score": 0.0,
        "release_dates": [],
        "date_counts": {},
    }
