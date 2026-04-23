"""
Live artist ingest — fetch data for any Spotify artist by name or ID.

Computes signals available without restricted endpoints:
  S2 — release cadence synchrony (from album release dates)
  S5 — intra-catalog title similarity (sentence-transformers)
  S7 — cross-platform (YouTube + iTunes)

S1 and S6 require audio-features / ISRC endpoints (restricted Apr 2026).
"""
from __future__ import annotations

import json
import math
import os
import re
import time
from datetime import date, datetime
from pathlib import Path

import numpy as np
from loguru import logger

from src.api.spotify_client import SpotifyClient


_SPOTIFY_ID_RE = re.compile(r"^[A-Za-z0-9]{22}$")
_S5_MODEL = None
_ENABLE_LIVE_EMBEDDINGS = os.getenv("ENABLE_LIVE_EMBEDDINGS", "false").lower() in {"1", "true", "yes", "on"}

_DATA_DIR = Path("data/processed")


def _kaggle_lookup(artist_name: str, artist_id: str | None = None) -> dict:
    """
    Search all local Kaggle/processed CSV datasets for this artist.
    Returns a dict of real signal values found, empty dict if not found.
    Provides: s1_audio_similarity, s2_cadence_sync, s3_playlist_cooccurrence (partial),
              release_dates, track_names, isrc_category.
    """
    import pandas as pd
    result: dict = {}
    name_lower = artist_name.lower()

    # ── 1. organic_controls + ghost_candidates: audio variance → S1 ──────
    for csv in ["organic_controls_kaggle.csv", "ghost_candidates_kaggle.csv"]:
        path = _DATA_DIR / csv
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path)
            # artist column may contain semicolon-joined names
            mask = df["artist"].str.lower().str.contains(name_lower[:12], na=False, regex=False)
            row = df[mask]
            if not row.empty:
                r = row.iloc[0]
                total_var = float(r.get("total_variance", 0))
                # low variance = ghost (S1: high score = suspicious)
                # organic baseline ~0.35, ghost ~0.0
                s1 = float(np.clip(1.0 - (total_var / 0.40), 0, 1))
                result["s1_audio_similarity"] = round(s1, 4)
                result["s1_track_count"] = int(r.get("track_count", 0))
                result["s1_flag"] = str(r.get("flag", ""))
                logger.info(f"Kaggle audio variance for {artist_name}: total_var={total_var:.3f} → S1={s1:.3f}")
                break
        except Exception as e:
            logger.warning(f"Kaggle CSV {csv} read error: {e}")

    # ── 2. exercise4_full_data: release dates + track names + ISRC → S2, S3 ──
    ex4_path = _DATA_DIR / "exercise4_full_data.csv"
    if ex4_path.exists():
        try:
            df4 = pd.read_csv(ex4_path)
            if artist_id:
                mask4 = df4["artist_id"] == artist_id
                if not mask4.any():
                    mask4 = df4["artist_name"].str.lower().str.contains(name_lower[:12], na=False, regex=False)
            else:
                mask4 = df4["artist_name"].str.lower().str.contains(name_lower[:12], na=False, regex=False)

            rows4 = df4[mask4]
            if not rows4.empty:
                dates = rows4["release_date"].dropna().tolist()
                tracks = rows4["track_name"].dropna().tolist()
                isrc_prefixes = rows4["prefix"].dropna().unique().tolist()
                result["kaggle_release_dates"] = dates
                result["kaggle_track_names"] = tracks
                result["kaggle_isrc_prefixes"] = isrc_prefixes
                logger.info(f"Kaggle ex4 for {artist_name}: {len(dates)} dates, {len(tracks)} tracks, prefixes={isrc_prefixes}")
        except Exception as e:
            logger.warning(f"Kaggle ex4 read error: {e}")

    # ── 3. isrc_classified: ISRC category → S3 (playlist / registrant signal) ──
    isrc_path = _DATA_DIR / "isrc_classified.csv"
    if isrc_path.exists():
        try:
            dfi = pd.read_csv(isrc_path)
            if artist_id:
                mask_i = dfi["artist_id"] == artist_id
            else:
                mask_i = dfi["artist_name"].str.lower().str.contains(name_lower[:12], na=False, regex=False)
            rows_i = dfi[mask_i]
            if not rows_i.empty:
                is_ghost = bool(rows_i.iloc[0].get("is_ghost_artist", False))
                category = str(rows_i.iloc[0].get("category", ""))
                result["isrc_category"] = category
                result["isrc_is_ghost"] = is_ghost
                # CUSTOM_REGISTRANT is a strong ghost signal
                if "CUSTOM" in category.upper():
                    result["s3_playlist_cooccurrence"] = 0.85
                elif is_ghost:
                    result["s3_playlist_cooccurrence"] = 0.70
                else:
                    result["s3_playlist_cooccurrence"] = 0.10
                logger.info(f"Kaggle ISRC for {artist_name}: category={category}, is_ghost={is_ghost}")
        except Exception as e:
            logger.warning(f"Kaggle ISRC read error: {e}")

    # ── 4. ex1_catalog_features: per-track variance → S1 (if not already set) ──
    if "s1_audio_similarity" not in result:
        ex1_path = _DATA_DIR / "ex1_catalog_features.csv"
        if ex1_path.exists():
            try:
                df1 = pd.read_csv(ex1_path)
                mask1 = df1["artist"].str.lower().str.contains(name_lower[:12], na=False, regex=False)
                rows1 = df1[mask1]
                if not rows1.empty:
                    variances = rows1[["danceability", "energy", "valence", "acousticness"]].var()
                    total_var = float(variances.sum())
                    s1 = float(np.clip(1.0 - (total_var / 0.40), 0, 1))
                    result["s1_audio_similarity"] = round(s1, 4)
                    logger.info(f"Kaggle ex1 audio for {artist_name}: total_var={total_var:.3f} → S1={s1:.3f}")
            except Exception as e:
                logger.warning(f"Kaggle ex1 read error: {e}")

    # ── 5. exercise5_walk_metrics: closure rate → S2 override ────────────
    ex5_path = _DATA_DIR / "exercise5_walk_metrics.csv"
    if ex5_path.exists():
        try:
            df5 = pd.read_csv(ex5_path)
            mask5 = df5["Artist"].str.lower().str.contains(name_lower[:8], na=False, regex=False)
            rows5 = df5[mask5]
            if not rows5.empty:
                closure_pct = float(rows5.iloc[0].get("Closure (≤1d gap %)", 0))
                closure_rate = closure_pct / 100.0
                import math as _math
                s2 = float(np.clip(1.0 / (1.0 + _math.exp(-8.0 * (closure_rate - 0.5))), 0, 1))
                result["s2_cadence_sync"] = round(s2, 4)
                result["s2_closure_rate"] = round(closure_rate, 4)
                logger.info(f"Kaggle ex5 cadence for {artist_name}: closure={closure_rate:.2%} → S2={s2:.3f}")
        except Exception as e:
            logger.warning(f"Kaggle ex5 read error: {e}")

    return result


def _get_s5_model():
    """Lazy-load and reuse the sentence model across requests."""
    global _S5_MODEL
    if _S5_MODEL is None:
        from sentence_transformers import SentenceTransformer
        _S5_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return _S5_MODEL


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
    Live analysis pipeline:
      1. Ask OpenAI to pre-assess the artist (instant baseline for all 7 signals)
      2. Try real data sources: Spotify cache → Neo4j → YouTube → iTunes
      3. Override OpenAI estimates wherever real data exists
      4. Fall back to OpenAI estimates for any signal all APIs failed to provide
    """
    t0 = time.perf_counter()
    sp = SpotifyClient()

    if not _SPOTIFY_ID_RE.fullmatch(artist_id):
        artist_name = artist_name or artist_id
        try:
            cached = sp._from_cache("search_artists", artist_name, 10)
            if cached:
                best_match = cached[0]
                artist_id = best_match["id"]
                artist_name = best_match["name"]
                logger.info(f"Resolved from cache: {artist_name} ({artist_id})")
        except Exception:
            pass

    if not artist_name:
        artist_name = artist_id

    logger.info(f"Live analysis: {artist_name} ({artist_id})")

    # ── STEP 1: OpenAI pre-assessment (instant baseline) ─────────────────
    ai_assessment = _openai_preassess(artist_name)
    logger.info(f"OpenAI pre-assessment for {artist_name}: {ai_assessment.get('verdict')} "
                f"(score={ai_assessment.get('overall_score', 0):.2f})")

    # ── STEP 2: Fetch real data ───────────────────────────────────────────
    release_dates: list[str] = []
    track_names: list[str] = []

    # 2a. Kaggle / local processed CSVs (instant, no API)
    kaggle = _kaggle_lookup(artist_name, artist_id if _SPOTIFY_ID_RE.fullmatch(artist_id) else None)
    if kaggle.get("kaggle_release_dates"):
        release_dates = kaggle["kaggle_release_dates"]
        logger.info(f"Kaggle release dates for {artist_name}: {len(release_dates)}")
    if kaggle.get("kaggle_track_names"):
        track_names = kaggle["kaggle_track_names"]
        logger.info(f"Kaggle track names for {artist_name}: {len(track_names)}")

    # 2b. Spotify local cache
    try:
        cached_albums = sp._from_cache("albums", artist_id, "album,single")
        if cached_albums:
            release_dates = [a["release_date"] for a in cached_albums if a.get("release_date")]
            for album in cached_albums[:6]:
                cached_tracks = sp._from_cache("album_tracks", album["id"])
                if cached_tracks:
                    track_names.extend(t["name"] for t in cached_tracks if t.get("name"))
                if len(track_names) >= 120:
                    break
            logger.info(f"Spotify cache: {len(release_dates)} dates, {len(track_names)} tracks")
    except Exception as e:
        logger.warning(f"Spotify cache read failed: {e}")

    # 2b. Neo4j fallback
    if not release_dates and not track_names:
        try:
            from src.graph.neo4j_client import Neo4jClient
            _neo4j = Neo4jClient()
            date_rows = _neo4j.run(
                "MATCH (a:Artist {spotify_id:$id})-[:RELEASED]->(al:Album) "
                "WHERE al.release_date IS NOT NULL AND al.release_date<>'' "
                "RETURN al.release_date AS dt ORDER BY al.release_date",
                id=artist_id,
            )
            release_dates = [r["dt"] for r in date_rows]
            track_rows = _neo4j.run(
                "MATCH (a:Artist {spotify_id:$id})-[:RELEASED]->(al:Album)-[:CONTAINS]->(t:Track) "
                "RETURN t.name AS name LIMIT 160",
                id=artist_id,
            )
            track_names = [r["name"] for r in track_rows if r.get("name")]
            if release_dates or track_names:
                logger.info(f"Neo4j: {len(release_dates)} dates, {len(track_names)} tracks")
        except Exception as e:
            logger.warning(f"Neo4j fallback: {e}")

    # ── STEP 3: Compute real signals where data exists ────────────────────
    has_cadence_data = len(release_dates) >= 2
    has_title_data = len(track_names) >= 2

    # S2: use real cadence data if available, else OpenAI estimate
    if has_cadence_data:
        s2_result = _score_cadence(artist_id, artist_name, release_dates)
        s2_source = "real"
    else:
        s2_score = ai_assessment.get("s2_cadence_sync")
        s2_result = {
            "suspicion_score": s2_score,
            "suspicion_level": "HIGH" if s2_score and s2_score >= 0.6 else "LOW",
            "note": "estimated by OpenAI",
        }
        s2_source = "openai"

    # S5: use real title data if available, else OpenAI estimate
    if has_title_data:
        logger.info(f"S5 scoring for {artist_name}")
        s5_result = _score_title_similarity(artist_id, artist_name, track_names)
        s5_source = "real"
    else:
        s5_score = ai_assessment.get("s5_metadata_similarity")
        s5_result = {
            "suspicion_score": s5_score,
            "suspicion_level": "HIGH" if s5_score and s5_score >= 0.6 else "LOW",
            "note": "estimated by OpenAI",
        }
        s5_source = "openai"

    # S7: try cross-platform APIs, fall back to OpenAI estimate
    s7_result = {"suspicion_score": None}
    s7_source = "none"
    if run_s7:
        try:
            from src.signals.cross_platform import score_artist as cp_score
            s7_result = cp_score(artist_name, None, artist_id, None)
            s7_source = s7_result.get("source", "apis")
            logger.info(f"S7 from {s7_source}: {s7_result.get('suspicion_score')}")
        except Exception as e:
            logger.warning(f"S7 APIs failed for {artist_name}: {e}")

    # If S7 APIs all failed, use OpenAI estimate
    if s7_result.get("suspicion_score") is None or s7_source == "none":
        s7_score = ai_assessment.get("s7_cross_platform")
        s7_result = {
            "suspicion_score": s7_score,
            "note": "estimated by OpenAI",
        }
        s7_source = "openai"

    # ── STEP 4: Build final signal scores ─────────────────────────────────
    # Priority: real-time API > Kaggle CSV > OpenAI estimate
    ai_sigs = ai_assessment.get("signal_scores", {})

    def _pick(real_val, kaggle_key, ai_key):
        if real_val is not None:
            return real_val, "real"
        if kaggle.get(kaggle_key) is not None:
            return kaggle[kaggle_key], "kaggle"
        if ai_sigs.get(ai_key) is not None:
            return ai_sigs[ai_key], "openai"
        return None, "unavailable"

    s1_val, s1_src = _pick(None, "s1_audio_similarity", "s1_audio_similarity")
    s2_val, s2_src = _pick(
        s2_result.get("suspicion_score") if s2_source == "real" else None,
        "s2_cadence_sync", "s2_cadence_sync"
    )
    if s2_src == "unavailable" and s2_result.get("suspicion_score") is not None:
        s2_val, s2_src = s2_result.get("suspicion_score"), s2_source
    s3_val, s3_src = _pick(None, "s3_playlist_cooccurrence", "s3_playlist_cooccurrence")
    s4_val, s4_src = _pick(None, None, "s4_follower_ratio")
    s5_val, s5_src = _pick(
        s5_result.get("suspicion_score") if s5_source == "real" else None,
        None, "s5_metadata_similarity"
    )
    if s5_src == "unavailable" and s5_result.get("suspicion_score") is not None:
        s5_val, s5_src = s5_result.get("suspicion_score"), s5_source
    s6_val, s6_src = _pick(None, None, "s6_graph_density")
    s7_val, s7_src = _pick(
        s7_result.get("suspicion_score") if s7_source not in ("none", "openai") else None,
        None, "s7_cross_platform"
    )
    if s7_src == "unavailable" and s7_result.get("suspicion_score") is not None:
        s7_val, s7_src = s7_result.get("suspicion_score"), s7_source

    signal_scores = {
        "s1_audio_similarity":      s1_val,
        "s2_cadence_sync":          s2_val,
        "s3_playlist_cooccurrence": s3_val,
        "s4_follower_ratio":        s4_val,
        "s5_metadata_similarity":   s5_val,
        "s6_graph_density":         s6_val,
        "s7_cross_platform":        s7_val,
    }

    signal_sources = {
        "s1_audio_similarity":      s1_src,
        "s2_cadence_sync":          s2_src,
        "s3_playlist_cooccurrence": s3_src,
        "s4_follower_ratio":        s4_src,
        "s5_metadata_similarity":   s5_src,
        "s6_graph_density":         s6_src,
        "s7_cross_platform":        s7_src,
    }

    # Weighted scoring — real signals get full weight, OpenAI estimates get 70% weight
    _WEIGHTS = {
        "s1_audio_similarity":      0.10,
        "s2_cadence_sync":          0.20,
        "s3_playlist_cooccurrence": 0.10,
        "s4_follower_ratio":        0.10,
        "s5_metadata_similarity":   0.15,
        "s6_graph_density":         0.20,
        "s7_cross_platform":        0.15,
    }
    weighted_sum = 0.0
    weight_total = 0.0
    for key, w in _WEIGHTS.items():
        val = signal_scores.get(key)
        if val is None:
            continue
        src = signal_sources.get(key, "unavailable")
        effective_w = w if src == "real" else w * 0.7
        weighted_sum += val * effective_w
        weight_total += effective_w

    overall_score = float(weighted_sum / weight_total) if weight_total > 0 else ai_assessment.get("overall_score", 0.0)
    overall_score = float(np.clip(overall_score, 0.0, 1.0))

    available = [v for v in signal_scores.values() if v is not None]
    verdict, confidence = _verdict(overall_score, len(available), signal_scores)

    # Build explanation
    sources_used = [k for k, v in signal_sources.items() if v != "unavailable"]
    real_sources = [k for k, v in signal_sources.items() if v == "real"]
    ai_sources = [k for k, v in signal_sources.items() if v == "openai"]

    explanation_parts = [
        f"{artist_name}: {verdict} (overall_score={overall_score:.3f})",
        f"Real data: {', '.join(real_sources) if real_sources else 'none'}.",
        f"AI-estimated: {', '.join(ai_sources) if ai_sources else 'none'}.",
        f"AI assessment: {ai_assessment.get('reasoning', '')}",
    ]

    return {
        "artist_id": artist_id,
        "artist_name": artist_name,
        "signal_scores": signal_scores,
        "signal_sources": signal_sources,
        "overall_score": overall_score,
        "verdict": verdict,
        "confidence": confidence,
        "explanation": " ".join(explanation_parts),
        "timing": {"total_seconds": round(time.perf_counter() - t0, 2)},
        "live_mode": True,
        "signals_available": len(available),
        "album_count": len(release_dates),
        "track_count": len(track_names),
        "s2_detail": s2_result,
        "s5_detail": s5_result,
        "s7_detail": s7_result,
        "ai_assessment": ai_assessment,
    }


# ── OpenAI pre-assessment ─────────────────────────────────────────────────────

def _openai_preassess(artist_name: str) -> dict:
    """
    Ask GPT-4o-mini to estimate all 7 signal scores for an artist.
    Returns a dict with overall_score, verdict, reasoning, and per-signal estimates.
    Used as baseline; real API data overrides these estimates where available.
    """
    _cache_dir = Path("data/raw/cache")
    _cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = _cache_dir / f"openai_assess_{artist_name.replace(' ', '_')[:60]}.json"
    if cache_path.exists():
        try:
            return json.loads(cache_path.read_text())
        except Exception:
            pass

    try:
        import json as _json
        from openai import OpenAI
        oai = OpenAI()
        prompt = f"""You are an expert music fraud analyst detecting ghost/fake Spotify artists.

Artist name: "{artist_name}"

Ghost artists are fake identities used to collect Spotify royalties. They typically:
- Have generic descriptive names ("Relaxing Piano Music", "Baby Sleep Sounds", "Nature Sounds")
- Produce bulk lo-fi/ambient/sleep/meditation/nature content
- Have zero YouTube presence and no Apple Music catalogue
- Release dozens of albums in bursts on the same day
- Have no social media, interviews, or public persona
- Titles are repetitive variations ("Calming Piano Vol.1", "Calming Piano Vol.2")

Real organic artists have unique names, diverse catalogs, YouTube channels, press coverage, and Apple Music presence.

Based on your knowledge of this artist, estimate all 7 ghost-detection signals (0.0 = organic, 1.0 = ghost):
- s1_audio_similarity: audio feature variance (high = unnaturally similar tracks)
- s2_cadence_sync: release cadence regularity (high = batch same-day uploads)
- s3_playlist_cooccurrence: playlist isolation (high = only appears with other ghosts)
- s4_follower_ratio: follower-to-stream ratio anomaly
- s5_metadata_similarity: title repetition / name recycling
- s6_graph_density: network isolation (high = no connections to organic artists)
- s7_cross_platform: absent on YouTube/Apple Music (high = ghost)

Reply in JSON only:
{{
  "verdict": "LIKELY_GHOST" | "SUSPICIOUS" | "LIKELY_ORGANIC",
  "overall_score": 0.0-1.0,
  "reasoning": "one concise sentence explaining why",
  "signal_scores": {{
    "s1_audio_similarity": 0.0-1.0,
    "s2_cadence_sync": 0.0-1.0,
    "s3_playlist_cooccurrence": 0.0-1.0,
    "s4_follower_ratio": 0.0-1.0,
    "s5_metadata_similarity": 0.0-1.0,
    "s6_graph_density": 0.0-1.0,
    "s7_cross_platform": 0.0-1.0
  }}
}}"""

        resp = oai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=400,
            response_format={"type": "json_object"},
        )
        import json as _json2
        data = _json2.loads(resp.choices[0].message.content)

        # Normalise and clip all scores
        sigs = data.get("signal_scores", {})
        for k in sigs:
            try:
                sigs[k] = round(float(np.clip(sigs[k], 0.0, 1.0)), 4)
            except Exception:
                sigs[k] = None

        result = {
            "verdict": data.get("verdict", "LIKELY_ORGANIC"),
            "overall_score": round(float(np.clip(data.get("overall_score", 0.5), 0.0, 1.0)), 4),
            "reasoning": data.get("reasoning", ""),
            "signal_scores": sigs,
            # convenience flat access
            "s1_audio_similarity":      sigs.get("s1_audio_similarity"),
            "s2_cadence_sync":          sigs.get("s2_cadence_sync"),
            "s3_playlist_cooccurrence": sigs.get("s3_playlist_cooccurrence"),
            "s4_follower_ratio":        sigs.get("s4_follower_ratio"),
            "s5_metadata_similarity":   sigs.get("s5_metadata_similarity"),
            "s6_graph_density":         sigs.get("s6_graph_density"),
            "s7_cross_platform":        sigs.get("s7_cross_platform"),
        }
        cache_path.write_text(_json.dumps(result, indent=2))
        return result

    except Exception as e:
        logger.warning(f"OpenAI pre-assessment failed for {artist_name}: {e}")
        return {
            "verdict": "LIKELY_ORGANIC",
            "overall_score": 0.3,
            "reasoning": "OpenAI assessment unavailable",
            "signal_scores": {},
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
    # Deduplicate and cap to avoid quadratic pairwise-cost on very large catalogs.
    uniq_titles = list(dict.fromkeys(titles))
    if len(uniq_titles) > 120:
        uniq_titles = uniq_titles[:120]

    # Fast lexical fallback (default): avoids first-request model download stalls on Render.
    if not _ENABLE_LIVE_EMBEDDINGS:
        mean_cos = _mean_lexical_cosine(uniq_titles)
        score = float(np.clip((mean_cos - 0.10) / (0.60 - 0.10), 0, 1))
        level = "HIGH" if score >= 0.6 else "MEDIUM" if score >= 0.35 else "LOW"
        return {
            "suspicion_score": round(score, 4),
            "suspicion_level": level,
            "mean_cosine": round(mean_cos, 4),
            "track_count": len(titles),
            "sampled_track_count": len(uniq_titles),
            "method": "lexical_tfidf",
        }

    try:
        model = _get_s5_model()
        embs = model.encode(uniq_titles, normalize_embeddings=True, show_progress_bar=False)
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
            "track_count": len(titles),
            "sampled_track_count": n,
            "method": "sentence_transformer",
        }
    except Exception as e:
        logger.warning(f"S5 sentence-transformer failed, falling back to lexical TF-IDF: {e}")
        mean_cos = _mean_lexical_cosine(uniq_titles)
        score = float(np.clip((mean_cos - 0.10) / (0.60 - 0.10), 0, 1))
        level = "HIGH" if score >= 0.6 else "MEDIUM" if score >= 0.35 else "LOW"
        return {
            "suspicion_score": round(score, 4),
            "suspicion_level": level,
            "mean_cosine": round(mean_cos, 4),
            "track_count": len(titles),
            "sampled_track_count": len(uniq_titles),
            "method": "lexical_tfidf_fallback",
            "note": str(e),
        }


def _mean_lexical_cosine(titles: list[str]) -> float:
    """Compute mean pairwise similarity from title text without heavy model downloads."""
    if len(titles) < 2:
        return 0.0

    from sklearn.feature_extraction.text import TfidfVectorizer

    vec = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), token_pattern=r"(?u)\\b\\w+\\b")
    try:
        X = vec.fit_transform(titles)
    except ValueError:
        return 0.0
    sims = (X * X.T).toarray()
    n = sims.shape[0]
    idx = np.triu_indices(n, k=1)
    if len(idx[0]) == 0:
        return 0.0
    return float(np.mean(sims[idx]))


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


def _verdict(score: float, n_signals: int, signal_scores: dict | None = None) -> tuple[str, float]:
    confidence = min(0.4 + 0.2 * n_signals, 0.85)  # max 85% with partial signals
    # Hard rule: extreme cadence + any ISRC suspicion = ghost regardless of score
    if signal_scores:
        s2 = signal_scores.get("s2_cadence_sync", 0) or 0
        s3 = signal_scores.get("s3_playlist_cooccurrence", 0) or 0
        if s2 >= 0.90 and s3 >= 0.35:
            return "LIKELY_GHOST", min(confidence + 0.05, 0.95)
    if score >= 0.60:
        return "LIKELY_GHOST", confidence
    elif score >= 0.40:
        return "SUSPICIOUS", confidence
    return "LIKELY_ORGANIC", confidence
