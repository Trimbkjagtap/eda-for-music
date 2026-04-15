"""
Signal 7: Cross-Platform Popularity Discrepancy

If an artist has significant Spotify presence but is invisible on
YouTube and Apple Music, that's a strong indicator of single-platform inflation.

Input: artist_name, track_name (optional), spotify_followers (optional)
Output: discrepancy score 0.0 (consistent) to 1.0 (highly suspicious)

Scoring logic:
  youtube_score:
    - no video found → 1.0
    - views < 100    → 0.9
    - views < 1,000  → 0.8
    - views < 10,000 → 0.6
    - views < 100,000 → 0.4
    - views >= 100,000 → 0.0

  apple_score:
    - not found → 1.0
    - 1-2 results → 0.5
    - 3+ results → 0.0

  combined = 0.6 * youtube_score + 0.4 * apple_score

APIs:
  - YouTube Data API v3 (10K/day quota) — requires YOUTUBE_API_KEY in .env
  - iTunes Search API (no key, no rate limit)

Results are cached to data/raw/cache/ to avoid redundant API calls.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from loguru import logger

from src.api.youtube_client import YouTubeClient
from src.api.apple_music_client import AppleMusicClient

_CACHE_DIR = Path("data/raw/cache")
_CACHE_DIR.mkdir(parents=True, exist_ok=True)

_YOUTUBE_CALL_COUNT = 0
_APPLE_CALL_COUNT = 0


def _cache_key(artist: str, track: str) -> Path:
    safe = f"cross_platform_{artist}_{track}".replace(" ", "_").replace("/", "_")[:80]
    return _CACHE_DIR / f"{safe}.json"


def score_artist(
    artist_name: str,
    track_name: str | None = None,
    artist_id: str | None = None,
    neo4j=None,
) -> dict:
    """
    Check cross-platform presence for an artist.

    If track_name is None, queries Neo4j (or uses a generic search)
    to find the artist's most prominent track.

    Returns
    -------
    dict with keys:
        artist_name, track_queried,
        youtube_views, youtube_found, youtube_score,
        apple_result_count, apple_found, apple_score,
        combined_score, suspicion_level, suspicion_score
    """
    global _YOUTUBE_CALL_COUNT, _APPLE_CALL_COUNT

    # Find a representative track if not provided
    if track_name is None:
        track_name = _get_sample_track(artist_name, artist_id, neo4j)

    # Check cache first
    cache_path = _cache_key(artist_name, track_name or "")
    if cache_path.exists():
        data = json.loads(cache_path.read_text())
        logger.info(f"Cross-platform cache hit: {artist_name}")
        return data

    yt_client = YouTubeClient()
    apple_client = AppleMusicClient()

    # ── YouTube ───────────────────────────────────────────────────────────
    yt_views = 0
    yt_found = False
    try:
        if track_name:
            yt_views = yt_client.get_track_views(artist_name, track_name)
        else:
            video = yt_client.search_video(artist_name, "official")
            if video:
                vid_id = video["id"].get("videoId")
                if vid_id:
                    yt_views = yt_client.get_view_count(vid_id)
        yt_found = yt_views > 0
        _YOUTUBE_CALL_COUNT += 2  # search + stats
        time.sleep(0.5)  # brief pause between API providers
    except Exception as e:
        logger.warning(f"YouTube lookup failed for {artist_name}: {e}")

    # ── Apple Music ────────────────────────────────────────────────────────
    apple_count = 0
    apple_found = False
    try:
        if track_name:
            apple_count = apple_client.get_track_count(artist_name, track_name)
        else:
            results = apple_client.search_track(artist_name, "")
            apple_count = len(results)
        apple_found = apple_count > 0
        _APPLE_CALL_COUNT += 1
    except Exception as e:
        logger.warning(f"Apple Music lookup failed for {artist_name}: {e}")

    # ── Score computation ─────────────────────────────────────────────────
    yt_score = _youtube_score(yt_views, yt_found)
    apple_score = _apple_score(apple_count)
    combined = 0.6 * yt_score + 0.4 * apple_score
    suspicion_score = float(np.clip(combined, 0.0, 1.0))
    suspicion_level = _level(suspicion_score)

    logger.info(
        f"{artist_name}: yt_views={yt_views:,}, yt_score={yt_score:.2f}, "
        f"apple_count={apple_count}, apple_score={apple_score:.2f}, "
        f"combined={suspicion_score:.2f} [{suspicion_level}]"
    )

    result = {
        "artist_name": artist_name,
        "artist_id": artist_id,
        "track_queried": track_name or "(artist search)",
        "youtube_views": yt_views,
        "youtube_found": yt_found,
        "youtube_score": round(yt_score, 4),
        "apple_result_count": apple_count,
        "apple_found": apple_found,
        "apple_score": round(apple_score, 4),
        "combined_score": round(combined, 4),
        "suspicion_level": suspicion_level,
        "suspicion_score": round(suspicion_score, 4),
    }

    # Cache result
    cache_path.write_text(json.dumps(result, indent=2))
    return result


def run_all_artists(artist_list: list[dict]) -> list[dict]:
    """
    Run cross-platform check for all artists in a list.

    artist_list items: {name, artist_id (optional), track_name (optional)}

    Returns list of score dicts with api_call_counts appended.
    Also prints a summary table.
    """
    results = []
    for a in artist_list:
        result = score_artist(
            artist_name=a["name"],
            track_name=a.get("track_name"),
            artist_id=a.get("artist_id"),
        )
        results.append(result)

    # Print summary table
    print("\n" + "=" * 90)
    print(f"{'Artist':<30} {'YT Views':>12} {'YT Score':>10} {'Apple':>8} {'A Score':>9} {'Combined':>10} {'Verdict':<15}")
    print("=" * 90)
    for r in results:
        print(
            f"{r['artist_name']:<30} "
            f"{r['youtube_views']:>12,} "
            f"{r['youtube_score']:>10.2f} "
            f"{r['apple_result_count']:>8} "
            f"{r['apple_score']:>9.2f} "
            f"{r['suspicion_score']:>10.2f} "
            f"{r['suspicion_level']:<15}"
        )
    print("=" * 90)
    print(f"YouTube API calls used: {_YOUTUBE_CALL_COUNT}")
    print(f"Apple Music API calls used: {_APPLE_CALL_COUNT}")

    return results


def get_call_counts() -> dict:
    """Return number of API calls used this session."""
    return {
        "youtube_calls": _YOUTUBE_CALL_COUNT,
        "apple_calls": _APPLE_CALL_COUNT,
    }


# ──────────────────────────── helpers ────────────────────────────


def _get_sample_track(artist_name: str, artist_id: str | None, neo4j) -> str | None:
    """Get a representative track name from Neo4j cache."""
    if artist_id and neo4j:
        try:
            rows = neo4j.run(
                """
                MATCH (a:Artist {spotify_id: $id})-[:RELEASED]->(al:Album)
                      -[:CONTAINS]->(t:Track)
                RETURN t.name AS name LIMIT 1
                """,
                id=artist_id,
            )
            if rows:
                return rows[0]["name"]
        except Exception:
            pass
    return None


def _youtube_score(views: int, found: bool) -> float:
    """Map YouTube view count → suspicion score."""
    if not found or views == 0:
        return 1.0
    if views < 100:
        return 0.9
    if views < 1_000:
        return 0.8
    if views < 10_000:
        return 0.6
    if views < 100_000:
        return 0.4
    if views < 1_000_000:
        return 0.2
    return 0.0


def _apple_score(result_count: int) -> float:
    """Map Apple Music result count → suspicion score."""
    if result_count == 0:
        return 1.0
    if result_count <= 2:
        return 0.5
    return 0.0


def _level(score: float) -> str:
    if score >= 0.70:
        return "HIGH"
    elif score >= 0.40:
        return "MEDIUM"
    return "LOW"
