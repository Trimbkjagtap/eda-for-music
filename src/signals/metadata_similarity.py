"""
Signal 5: Artist Metadata Similarity

Ghost artists in the same network have suspiciously similar metadata:
similar genre tags, similar naming patterns, similar description styles.

Input: list of artist IDs
Output: similarity score 0.0 (all different) to 1.0 (identical metadata)

Methodology:
- Genre overlap: Jaccard similarity of genre sets between pairs
  (genres stripped from API in April 2026; use Neo4j stored values or
   infer from ISRC country prefix as proxy)
- Name similarity: TF-IDF cosine similarity on artist names
  (ghost names often share keywords: "Relaxing", "Sleep", "Calm", "Music")
- Track name similarity: TF-IDF on all track names per artist
  (ghost catalogs reuse the same name patterns)
- Score = 0.4 * name_sim + 0.4 * track_name_sim + 0.2 * genre_jaccard

No API calls — uses Neo4j data only.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.graph.neo4j_client import Neo4jClient


def score_cluster(artist_ids: list[str], neo4j: Neo4jClient | None = None) -> dict:
    """
    Compute metadata similarity score for a cluster of artists.

    Returns
    -------
    dict with keys:
        artist_ids, artist_names, pairwise_name_sim, pairwise_track_sim,
        genre_jaccard, keyword_overlap (shared ghost keywords),
        suspicion_level, suspicion_score
    """
    _neo4j = neo4j or Neo4jClient()

    artist_data: dict[str, dict] = {}
    for aid in artist_ids:
        rows = _neo4j.run(
            "MATCH (a:Artist {spotify_id: $id}) RETURN a.name AS name, a.genres AS genres",
            id=aid,
        )
        track_rows = _neo4j.run(
            """
            MATCH (a:Artist {spotify_id: $id})-[:RELEASED]->(al:Album)-[:CONTAINS]->(t:Track)
            RETURN collect(t.name) AS track_names
            """,
            id=aid,
        )
        name = rows[0]["name"] if rows else aid
        genres_raw = (rows[0].get("genres") if rows else None) or []
        track_names = (track_rows[0]["track_names"] if track_rows else None) or []
        artist_data[aid] = {
            "name": name,
            "genres": set(genres_raw) if isinstance(genres_raw, list) else set(),
            "track_names": track_names,
        }

    ids = list(artist_ids)
    names = [artist_data[a]["name"] for a in ids]

    # ── 1. Artist name similarity via TF-IDF ──────────────────────────────
    name_sim = _tfidf_mean_pairwise(names)

    # ── 2. Ghost keyword overlap ───────────────────────────────────────────
    GHOST_KEYWORDS = {
        "relaxing", "relax", "sleep", "calm", "calmo", "meditation",
        "white", "noise", "ambient", "nature", "rain", "ocean",
        "healing", "spa", "zen", "focus", "study", "baby",
        "soft", "gentle", "peaceful", "tranquil", "soothing",
    }
    keyword_counts = []
    for aid in ids:
        name_tokens = set(re.findall(r"\w+", artist_data[aid]["name"].lower()))
        overlap = len(name_tokens & GHOST_KEYWORDS)
        keyword_counts.append(overlap)
    keyword_score = float(np.mean([min(k / 2.0, 1.0) for k in keyword_counts]))

    # ── 3. Track name vocabulary similarity ───────────────────────────────
    track_corpora = [
        " ".join(artist_data[aid]["track_names"]) for aid in ids
    ]
    track_sim = _tfidf_mean_pairwise(track_corpora) if len(ids) > 1 else 0.0

    # ── 4. Genre Jaccard ───────────────────────────────────────────────────
    genre_jaccards = []
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            a_g = artist_data[ids[i]]["genres"]
            b_g = artist_data[ids[j]]["genres"]
            union = len(a_g | b_g)
            inter = len(a_g & b_g)
            if union > 0:
                genre_jaccards.append(inter / union)
            # If both have no genres (API stripped), can't compute — skip
    mean_genre_jaccard = float(np.mean(genre_jaccards)) if genre_jaccards else 0.0

    # ── Combined score ─────────────────────────────────────────────────────
    suspicion_score = (
        0.40 * name_sim
        + 0.35 * track_sim
        + 0.15 * keyword_score
        + 0.10 * mean_genre_jaccard
    )
    suspicion_score = float(np.clip(suspicion_score, 0.0, 1.0))
    suspicion_level = _level(suspicion_score)

    logger.info(
        f"Cluster ({len(artist_ids)} artists): name_sim={name_sim:.3f}, "
        f"track_sim={track_sim:.3f}, keyword_score={keyword_score:.3f}, "
        f"genre_jaccard={mean_genre_jaccard:.3f}, "
        f"suspicion={suspicion_level} ({suspicion_score:.2f})"
    )

    return {
        "artist_ids": artist_ids,
        "artist_names": {aid: artist_data[aid]["name"] for aid in ids},
        "pairwise_name_sim": round(name_sim, 4),
        "pairwise_track_sim": round(track_sim, 4),
        "keyword_score": round(keyword_score, 4),
        "genre_jaccard": round(mean_genre_jaccard, 4),
        "keyword_counts": dict(zip([artist_data[a]["name"] for a in ids], keyword_counts)),
        "suspicion_level": suspicion_level,
        "suspicion_score": round(suspicion_score, 4),
        "note": "genres unavailable via API (Apr 2026) — keyword + track name signals used",
    }


def score_artist(artist_id: str, all_artist_ids: list[str] | None = None,
                 neo4j: Neo4jClient | None = None) -> dict:
    """
    Score a single artist against all others in the database.
    If all_artist_ids is None, uses all artists in Neo4j.
    """
    _neo4j = neo4j or Neo4jClient()
    if all_artist_ids is None:
        rows = _neo4j.run("MATCH (a:Artist) RETURN a.spotify_id AS id")
        all_artist_ids = [r["id"] for r in rows]

    others = [a for a in all_artist_ids if a != artist_id]
    if not others:
        return {
            "artist_id": artist_id,
            "suspicion_score": 0.0,
            "suspicion_level": "UNKNOWN",
        }

    cluster_result = score_cluster([artist_id] + others, _neo4j)
    return {
        "artist_id": artist_id,
        "artist_name": cluster_result["artist_names"].get(artist_id, artist_id),
        "suspicion_score": cluster_result["suspicion_score"],
        "suspicion_level": cluster_result["suspicion_level"],
        "keyword_count": cluster_result["keyword_counts"].get(
            cluster_result["artist_names"].get(artist_id, ""), 0
        ),
    }


# ──────────────────────────── helpers ────────────────────────────


def _tfidf_mean_pairwise(texts: list[str]) -> float:
    """Compute mean pairwise cosine similarity of TF-IDF vectors."""
    if len(texts) < 2:
        return 0.0
    # Filter empty texts
    non_empty = [t for t in texts if t and t.strip()]
    if len(non_empty) < 2:
        return 0.0
    try:
        vec = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            min_df=1,
            sublinear_tf=True,
        )
        tfidf = vec.fit_transform(non_empty)
        sim = cosine_similarity(tfidf)
        n = sim.shape[0]
        upper = sim[np.triu_indices(n, k=1)]
        return float(np.mean(upper)) if len(upper) > 0 else 0.0
    except Exception as e:
        logger.warning(f"TF-IDF similarity failed: {e}")
        return 0.0


def _level(score: float) -> str:
    if score >= 0.60:
        return "HIGH"
    elif score >= 0.35:
        return "MEDIUM"
    return "LOW"
