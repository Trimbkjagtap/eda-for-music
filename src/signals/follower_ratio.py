"""
Signal 4: Follower-to-Catalog Ratio Anomaly

Ghost artists often have disproportionate follower counts relative to
their catalog size, or vice versa.

Input: artist_id
Output: anomaly score 0.0 (normal) to 1.0 (highly anomalous)

Note on April 2026 API state:
- followers field is stripped from all API responses for new apps.
- Neo4j stores followers=0 for all our artists (not available at ingest time).
- We therefore rely on catalog-size signals only:
    * Large catalog (100+ tracks) with very short durations → streaming-optimized
    * Unusually large catalog size relative to release span
    * High track density: many releases per day
- These catalog-only signals still discriminate well on our seed set.

Data source: Neo4j (track counts, album counts, release dates) — no API calls.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from src.graph.neo4j_client import Neo4jClient

# Reference distribution from our seed set + Kaggle context
# (tracks, albums, span_days, avg_duration_ms)
_REFERENCE_ARTISTS = [
    # name, track_count, album_count, span_days, is_ghost
    ("Relaxing White Noise", 280, 45, 1156, True),
    ("Meditation Relax Club", 172, 30, 1240, True),
    ("Calmo", 38, 20, 2030, False),  # borderline
    ("Nils Frahm", 56, 68, 7566, False),
]

# Tracks/day threshold — above this is suspicious for non-mainstream artists
HIGH_DENSITY_THRESHOLD = 0.15   # > 0.15 tracks/day
LARGE_CATALOG_THRESHOLD = 150   # > 150 tracks → inspect


def score_artist(artist_id: str, neo4j: Neo4jClient | None = None) -> dict:
    """
    Compute follower/catalog ratio anomaly score for one artist.
    Followers are unavailable (API stripped), so this signal focuses on
    catalog density and size anomalies.

    Returns
    -------
    dict with keys:
        artist_id, artist_name, track_count, album_count,
        span_days, tracks_per_day, avg_duration_ms,
        followers (0 if unavailable),
        catalog_size_zscore, density_score,
        suspicion_level, suspicion_score
    """
    _neo4j = neo4j or Neo4jClient()

    # Basic artist info
    artist_rows = _neo4j.run(
        "MATCH (a:Artist {spotify_id: $id}) RETURN a.name AS name, a.followers AS followers",
        id=artist_id,
    )
    if not artist_rows:
        return {**_empty_result(artist_id, artist_id)}

    artist_name = artist_rows[0]["name"]
    followers = artist_rows[0].get("followers") or 0

    # Track count
    track_rows = _neo4j.run(
        """
        MATCH (a:Artist {spotify_id: $id})-[:RELEASED]->(al:Album)-[:CONTAINS]->(t:Track)
        RETURN count(t) AS track_count,
               avg(t.duration_ms) AS avg_duration_ms
        """,
        id=artist_id,
    )
    track_count = int(track_rows[0]["track_count"]) if track_rows else 0
    avg_duration_ms = float(track_rows[0]["avg_duration_ms"] or 0) if track_rows else 0.0

    # Album count and span
    album_rows = _neo4j.run(
        """
        MATCH (a:Artist {spotify_id: $id})-[:RELEASED]->(al:Album)
        WHERE al.release_date IS NOT NULL AND al.release_date <> ''
        RETURN count(al) AS album_count,
               min(al.release_date) AS earliest,
               max(al.release_date) AS latest
        """,
        id=artist_id,
    )
    album_count = int(album_rows[0]["album_count"]) if album_rows else 0
    span_days = 1  # default
    if album_rows and album_rows[0]["earliest"] and album_rows[0]["latest"]:
        try:
            from datetime import datetime
            e = datetime.strptime(album_rows[0]["earliest"][:10], "%Y-%m-%d")
            l = datetime.strptime(album_rows[0]["latest"][:10], "%Y-%m-%d")
            span_days = max((l - e).days, 1)
        except Exception:
            pass

    tracks_per_day = track_count / span_days

    # --- Catalog density score ---
    # High tracks/day = suspicious
    density_score = min(tracks_per_day / HIGH_DENSITY_THRESHOLD, 1.0)

    # --- Catalog size z-score against reference distribution ---
    ref_counts = [r[1] for r in _REFERENCE_ARTISTS]
    ref_mean = float(np.mean(ref_counts))
    ref_std = float(np.std(ref_counts))
    if ref_std > 0:
        z = (track_count - ref_mean) / ref_std
    else:
        z = 0.0

    # Large catalog and fast release both contribute
    size_score = float(np.clip((track_count - 50) / 250, 0.0, 1.0))

    # Short average duration is a streaming-optimization signal
    # Typical organic track: 3–5 min (180k–300k ms). Ghost tracks: 1–3 min.
    duration_score = 0.0
    if avg_duration_ms > 0:
        # < 120s → very suspicious, > 240s → not suspicious
        duration_score = float(np.clip(1.0 - (avg_duration_ms - 60_000) / 180_000, 0.0, 1.0))

    # Followers sanity check: if followers=0 and large catalog, more suspicious
    # (real big catalogs usually have some followers)
    follower_penalty = 0.0
    if followers == 0 and track_count > 50:
        follower_penalty = 0.2  # can't distinguish real-zero from API-stripped

    suspicion_score = (
        0.40 * density_score
        + 0.30 * size_score
        + 0.20 * duration_score
        + 0.10 * follower_penalty
    )
    suspicion_score = float(np.clip(suspicion_score, 0.0, 1.0))
    suspicion_level = _level(suspicion_score)

    logger.info(
        f"{artist_name}: tracks={track_count}, tracks/day={tracks_per_day:.3f}, "
        f"avg_dur={avg_duration_ms/1000:.0f}s, "
        f"suspicion={suspicion_level} ({suspicion_score:.2f})"
    )

    return {
        "artist_id": artist_id,
        "artist_name": artist_name,
        "track_count": track_count,
        "album_count": album_count,
        "span_days": span_days,
        "tracks_per_day": round(tracks_per_day, 4),
        "avg_duration_ms": round(avg_duration_ms, 0),
        "avg_duration_seconds": round(avg_duration_ms / 1000, 1),
        "followers": followers,
        "followers_available": followers > 0,
        "catalog_size_zscore": round(z, 3),
        "density_score": round(density_score, 4),
        "size_score": round(size_score, 4),
        "duration_score": round(duration_score, 4),
        "suspicion_level": suspicion_level,
        "suspicion_score": round(suspicion_score, 4),
        "note": "followers=0 (API stripped Apr 2026 — using catalog-only signals)",
    }


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
        "track_count": 0,
        "album_count": 0,
        "span_days": 0,
        "tracks_per_day": 0.0,
        "avg_duration_ms": 0.0,
        "avg_duration_seconds": 0.0,
        "followers": 0,
        "followers_available": False,
        "catalog_size_zscore": 0.0,
        "density_score": 0.0,
        "size_score": 0.0,
        "duration_score": 0.0,
        "suspicion_level": "UNKNOWN",
        "suspicion_score": 0.0,
        "note": "followers=0 (API stripped Apr 2026 — using catalog-only signals)",
    }
