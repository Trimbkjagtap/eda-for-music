"""
Live artist ingest — fetch data for any Spotify artist by name or ID.

Computes signals available without restricted endpoints:
  S2 — release cadence synchrony (from album release dates)
  S5 — intra-catalog title similarity (sentence-transformers)
  S7 — cross-platform (YouTube + iTunes)

S1 and S6 require audio-features / ISRC endpoints (restricted Apr 2026).
"""
from __future__ import annotations

import math
import time
from datetime import date, datetime
from pathlib import Path

import numpy as np
from loguru import logger

from src.api.spotify_client import SpotifyClient


def search_artist(name: str) -> list[dict]:
    """Return top-5 Spotify artist matches for a name query."""
    sp = SpotifyClient()
    results = sp.search_artists(name, limit=5)
    return [
        {
            "id": a["id"],
            "name": a["name"],
            "followers": a.get("followers", {}).get("total", 0),
            "popularity": a.get("popularity", 0),
            "image": (a.get("images") or [{}])[0].get("url"),
        }
        for a in results
    ]


def analyze_live(artist_id: str, artist_name: str | None = None, run_s7: bool = True) -> dict:
    """
    Run a live 3-signal analysis for any artist.

    Returns the same shape as run_analysis() so the frontend can reuse
    renderAnalysisResult without changes.
    """
    t0 = time.perf_counter()
    sp = SpotifyClient()

    # Resolve name
    if not artist_name:
        meta = sp.get_artist(artist_id)
        artist_name = meta.get("name", artist_id)

    logger.info(f"Live analysis: {artist_name} ({artist_id})")

    # --- fetch albums + tracks ---
    albums = sp.get_artist_albums(artist_id, include_groups="album,single")
    release_dates = [a["release_date"] for a in albums if a.get("release_date")]
    track_names: list[str] = []
    for album in albums[:20]:  # cap at 20 albums to stay within rate limits
        try:
            tracks = sp.get_album_tracks(album["id"])
            track_names.extend(t["name"] for t in tracks if t.get("name"))
        except Exception:
            pass

    # --- S2: cadence synchrony ---
    s2_result = _score_cadence(artist_id, artist_name, release_dates)

    # --- S5: intra-catalog title similarity ---
    s5_result = _score_title_similarity(artist_id, artist_name, track_names)

    # --- S7: cross-platform ---
    s7_result = {"suspicion_score": None, "youtube_views": None, "apple_music": None}
    if run_s7:
        try:
            from src.signals.cross_platform import score_artist as cp_score
            s7_result = cp_score(artist_name, None, artist_id, None)
        except Exception as e:
            logger.warning(f"S7 failed for {artist_name}: {e}")

    signal_scores = {
        "s1_audio_similarity": None,
        "s2_cadence_sync": s2_result.get("suspicion_score"),
        "s3_playlist_cooccurrence": None,
        "s4_follower_ratio": None,
        "s5_metadata_similarity": s5_result.get("suspicion_score"),
        "s6_graph_density": None,
        "s7_cross_platform": s7_result.get("suspicion_score"),
    }

    # Verdict from available signals only
    available = [v for v in signal_scores.values() if v is not None]
    overall_score = float(np.mean(available)) if available else 0.0
    verdict, confidence = _verdict(overall_score, len(available))

    explanation_parts = [
        f"{artist_name}: {verdict} (overall_score={overall_score:.3f})",
        f"{'HIGH' if s2_result.get('suspicion_score', 0) > 0.6 else 'LOW'} cadence suspicion"
        f" ({s2_result.get('suspicion_score', 0):.2f}): {s2_result.get('suspicion_level', '')}",
        f"Title similarity ({s5_result.get('suspicion_score', 0):.2f}):"
        f" mean cosine={s5_result.get('mean_cosine', 0):.3f}",
        "Data unavailable: s1_audio_similarity, s3_playlist_cooccurrence,"
        " s4_follower_ratio, s6_graph_density (Spotify API restrictions)",
    ]

    return {
        "artist_id": artist_id,
        "artist_name": artist_name,
        "signal_scores": signal_scores,
        "overall_score": overall_score,
        "verdict": verdict,
        "confidence": confidence,
        "explanation": " ".join(explanation_parts),
        "timing": {"total_seconds": round(time.perf_counter() - t0, 2)},
        "live_mode": True,
        "signals_available": len(available),
        "album_count": len(albums),
        "track_count": len(track_names),
        "s2_detail": s2_result,
        "s5_detail": s5_result,
        "s7_detail": s7_result,
    }


# ── signal helpers ────────────────────────────────────────────────────────────

def _score_cadence(artist_id: str, artist_name: str, raw_dates: list[str]) -> dict:
    parsed = _parse_dates(raw_dates)
    if len(parsed) < 2:
        return {"suspicion_score": 0.0, "suspicion_level": "LOW",
                "closure_rate": 0.0, "note": "too few releases"}
    parsed.sort()
    gaps = np.diff([d.toordinal() for d in parsed])
    same_day = int((gaps == 0).sum())
    closure_rate = same_day / len(gaps)
    median_gap = float(np.median(gaps))
    burst_ratio = _burst_ratio(parsed, 7)
    score = _closure_to_suspicion(closure_rate)
    score = float(np.clip(0.5 * score + 0.25 * min(burst_ratio, 1.0), 0, 1))
    level = "HIGH" if score >= 0.6 else "MEDIUM" if score >= 0.35 else "LOW"
    return {
        "suspicion_score": round(score, 4),
        "suspicion_level": level,
        "closure_rate": round(closure_rate, 4),
        "median_gap_days": round(median_gap, 1),
        "total_releases": len(parsed),
        "burst_ratio": round(float(burst_ratio), 4),
    }


def _score_title_similarity(artist_id: str, artist_name: str, titles: list[str]) -> dict:
    if len(titles) < 2:
        return {"suspicion_score": 0.0, "suspicion_level": "LOW",
                "mean_cosine": 0.0, "note": "too few tracks"}
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        embs = model.encode(titles, normalize_embeddings=True, show_progress_bar=False)
        # upper-triangle mean pairwise cosine
        n = len(embs)
        sims = embs @ embs.T
        idx = np.triu_indices(n, k=1)
        mean_cos = float(np.mean(sims[idx]))
        # organic baseline ~0.16, RWN ghost ~0.56; score linearly
        score = float(np.clip((mean_cos - 0.10) / (0.60 - 0.10), 0, 1))
        level = "HIGH" if score >= 0.6 else "MEDIUM" if score >= 0.35 else "LOW"
        return {
            "suspicion_score": round(score, 4),
            "suspicion_level": level,
            "mean_cosine": round(mean_cos, 4),
            "track_count": n,
        }
    except Exception as e:
        logger.warning(f"S5 sentence-transformer failed: {e}")
        return {"suspicion_score": 0.0, "suspicion_level": "LOW",
                "mean_cosine": 0.0, "note": str(e)}


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


def _burst_ratio(dates: list[date], window: int = 7) -> float:
    ordinals = sorted(d.toordinal() for d in dates)
    max_w, j = 0, 0
    for i, o in enumerate(ordinals):
        while j < len(ordinals) and ordinals[j] <= o + window:
            j += 1
        max_w = max(max_w, j - i)
    return max_w / len(ordinals)


def _closure_to_suspicion(closure: float) -> float:
    return float(np.clip(1.0 / (1.0 + math.exp(-8.0 * (closure - 0.5))), 0, 1))


def _verdict(score: float, n_signals: int) -> tuple[str, float]:
    confidence = min(0.4 + 0.2 * n_signals, 0.85)  # max 85% with partial signals
    if score >= 0.65:
        return "LIKELY_GHOST", confidence
    elif score >= 0.40:
        return "SUSPICIOUS", confidence
    return "LIKELY_ORGANIC", confidence
