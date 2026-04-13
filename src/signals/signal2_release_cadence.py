"""
Signal 2 — Release Cadence Synchrony (Exercise 1 / Signal 2)
=============================================================
Ghost artist bot networks often release tracks in coordinated bursts:
50–100 tracks dropped on the same day, then silence for months.
Real artists release incrementally with natural temporal spacing.

Method:
- Fetch all release dates for an artist
- Compute inter-release gaps (days between consecutive releases)
- Run Isolation Forest on gap distribution to flag anomalous bursts
- Compute burst ratio: fraction of releases within 7-day windows
- Compute synchrony score: how many artists in the seed set share
  release dates (ghost networks release together)

Suspicion signals:
- Very low gap variance (everything released on same day or same week)
- High burst_ratio (> 0.6)
- Large single-day release count (> 10 tracks on one date)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from datetime import datetime, date
from collections import Counter
from loguru import logger
from sklearn.ensemble import IsolationForest

from src.api.spotify_client import SpotifyClient

# Thresholds
BURST_WINDOW_DAYS = 7          # window size for burst detection
BURST_RATIO_HIGH = 0.6         # fraction of releases in a single burst window → HIGH
SAME_DAY_HIGH = 10             # ≥ N tracks released on same date → suspicious
SINGLE_DAY_FRACTION_HIGH = 0.5 # if > 50% released on one day → HIGH suspicion


def score_artist(artist_id: str, spotify: SpotifyClient) -> dict:
    """
    Compute release cadence score for one artist.

    Returns
    -------
    dict with keys:
        artist_id, artist_name, total_releases, date_range_days,
        unique_release_dates, releases_per_day (avg), max_single_day_releases,
        burst_ratio, gap_stats (mean/std/min/max days),
        isolation_forest_anomaly_rate, suspicion_level, suspicion_score,
        release_dates (list), date_counts (dict)
    """
    artist_meta = spotify.get_artist(artist_id)
    artist_name = artist_meta.get("name", artist_id)

    raw_dates = spotify.get_release_dates(artist_id)

    if not raw_dates:
        logger.warning(f"No release dates for {artist_name}")
        return _empty_result(artist_id, artist_name)

    # --- 1. Parse dates (handle YYYY-MM-DD and YYYY formats) ---
    parsed = _parse_dates(raw_dates)
    if len(parsed) < 2:
        return {
            **_empty_result(artist_id, artist_name),
            "total_releases": len(raw_dates),
            "release_dates": raw_dates,
        }

    parsed.sort()
    date_counts = Counter(d.isoformat() for d in parsed)

    # --- 2. Gap analysis ---
    as_ordinals = [d.toordinal() for d in parsed]
    gaps = np.diff(as_ordinals)  # days between consecutive releases

    gap_stats = {
        "mean": float(np.mean(gaps)),
        "std": float(np.std(gaps)),
        "min": int(np.min(gaps)),
        "max": int(np.max(gaps)),
        "median": float(np.median(gaps)),
    }

    date_range_days = (parsed[-1] - parsed[0]).days
    max_single_day = max(date_counts.values())
    unique_dates = len(date_counts)
    avg_per_day = len(parsed) / max(date_range_days, 1)

    # --- 3. Burst ratio ---
    # What fraction of all releases fall within the single densest 7-day window?
    burst_ratio = _compute_burst_ratio(parsed, window_days=BURST_WINDOW_DAYS)

    # --- 4. Single-day fraction ---
    single_day_fraction = max_single_day / len(parsed)

    # --- 5. Isolation Forest on gap sequence ---
    anomaly_rate = 0.0
    if len(gaps) >= 5:
        X = gaps.reshape(-1, 1).astype(float)
        iso = IsolationForest(contamination=0.1, random_state=42)
        preds = iso.fit_predict(X)
        anomaly_rate = float((preds == -1).sum() / len(preds))

    # --- 6. Suspicion score ---
    suspicion_score = _compute_suspicion(
        burst_ratio=burst_ratio,
        single_day_fraction=single_day_fraction,
        max_single_day=max_single_day,
        gap_std=gap_stats["std"],
        anomaly_rate=anomaly_rate,
    )
    suspicion_level = _level(suspicion_score)

    logger.info(
        f"{artist_name}: {len(parsed)} releases, burst_ratio={burst_ratio:.2f}, "
        f"single_day_fraction={single_day_fraction:.2f}, "
        f"suspicion={suspicion_level} ({suspicion_score:.2f})"
    )

    return {
        "artist_id": artist_id,
        "artist_name": artist_name,
        "total_releases": len(parsed),
        "date_range_days": date_range_days,
        "unique_release_dates": unique_dates,
        "releases_per_day_avg": round(avg_per_day, 4),
        "max_single_day_releases": max_single_day,
        "single_day_fraction": round(single_day_fraction, 4),
        "burst_ratio": round(burst_ratio, 4),
        "gap_stats": gap_stats,
        "isolation_forest_anomaly_rate": round(anomaly_rate, 4),
        "suspicion_level": suspicion_level,
        "suspicion_score": round(suspicion_score, 4),
        "release_dates": [d.isoformat() for d in parsed],
        "date_counts": dict(date_counts),
    }


def score_seed_set(spotify: SpotifyClient, seeds: list[dict]) -> pd.DataFrame:
    """
    Score all seed artists and return a comparison DataFrame.
    Also computes cross-artist release date synchrony (ghost networks release together).
    """
    rows = []
    all_date_sets: dict[str, set] = {}

    for seed in seeds:
        sid = seed.get("spotify_id")
        if not sid:
            logger.warning(f"Skipping {seed['name']} — no spotify_id")
            continue
        result = score_artist(sid, spotify)
        result["is_ghost_label"] = seed.get("is_ghost", False)
        result["genre"] = seed.get("genre", "")
        rows.append(result)
        all_date_sets[sid] = set(result.get("release_dates", []))

    if not rows:
        return pd.DataFrame()

    # Cross-artist synchrony: how many artists share release dates?
    df = pd.DataFrame(rows)
    _add_synchrony_column(df, all_date_sets)

    df = df.sort_values("suspicion_score", ascending=False).reset_index(drop=True)
    return df


def cross_artist_synchrony(results: list[dict]) -> pd.DataFrame:
    """
    Given a list of score_artist() results, compute pairwise date overlap.
    Returns a DataFrame showing how many release dates artists share.
    """
    date_sets = {
        r["artist_name"]: set(r.get("release_dates", []))
        for r in results
        if r.get("release_dates")
    }
    names = sorted(date_sets.keys())
    matrix = []
    for a in names:
        row = []
        for b in names:
            overlap = len(date_sets[a] & date_sets[b])
            row.append(overlap)
        matrix.append(row)
    return pd.DataFrame(matrix, index=names, columns=names)


# ────────────────────────── helpers ──────────────────────────


def _parse_dates(raw_dates: list[str]) -> list[date]:
    parsed = []
    for d in raw_dates:
        try:
            if len(d) == 4:  # YYYY only
                parsed.append(date(int(d), 1, 1))
            elif len(d) == 7:  # YYYY-MM
                y, m = d.split("-")
                parsed.append(date(int(y), int(m), 1))
            else:
                parsed.append(datetime.strptime(d, "%Y-%m-%d").date())
        except Exception:
            pass
    return parsed


def _compute_burst_ratio(dates: list[date], window_days: int = 7) -> float:
    """Fraction of all releases that fall within the densest 7-day window."""
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


def _compute_suspicion(
    burst_ratio: float,
    single_day_fraction: float,
    max_single_day: int,
    gap_std: float,
    anomaly_rate: float,
) -> float:
    """Combine cadence signals into a single suspicion score [0, 1]."""
    score = 0.0

    # Burst ratio contributes up to 0.4
    score += min(burst_ratio, 1.0) * 0.4

    # Single-day fraction contributes up to 0.3
    score += min(single_day_fraction, 1.0) * 0.3

    # Large single-day release count contributes up to 0.15
    if max_single_day >= SAME_DAY_HIGH:
        score += 0.15
    elif max_single_day >= 5:
        score += 0.07

    # Low gap std (everything bunched together) contributes up to 0.15
    if gap_std < 7:
        score += 0.15
    elif gap_std < 30:
        score += 0.07

    return float(np.clip(score, 0.0, 1.0))


def _level(score: float) -> str:
    if score >= 0.6:
        return "HIGH"
    elif score >= 0.35:
        return "MEDIUM"
    return "LOW"


def _empty_result(artist_id: str, artist_name: str) -> dict:
    return {
        "artist_id": artist_id,
        "artist_name": artist_name,
        "total_releases": 0,
        "date_range_days": 0,
        "unique_release_dates": 0,
        "releases_per_day_avg": 0.0,
        "max_single_day_releases": 0,
        "single_day_fraction": 0.0,
        "burst_ratio": 0.0,
        "gap_stats": {},
        "isolation_forest_anomaly_rate": 0.0,
        "suspicion_level": "UNKNOWN",
        "suspicion_score": 0.0,
        "release_dates": [],
        "date_counts": {},
    }


def _add_synchrony_column(df: pd.DataFrame, date_sets: dict) -> None:
    """Add shared_dates_with_others column — ghost networks share release dates."""
    def count_shared(row):
        own = date_sets.get(row["artist_id"], set())
        total = 0
        for aid, dates in date_sets.items():
            if aid != row["artist_id"]:
                total += len(own & dates)
        return total

    if not df.empty and "artist_id" in df.columns:
        df["shared_release_dates_total"] = df.apply(count_shared, axis=1)
