"""
Fingerprint Analyst Agent — Scores audio and metadata signals.

Analyzes catalog coherence, audio similarity, metadata patterns.
Runs Signals 1 (audio similarity), 4 (follower ratio), 5 (metadata similarity).
"""
from __future__ import annotations

import json

from crewai import Agent
from loguru import logger
from pydantic import BaseModel, Field

try:
    from crewai.tools import BaseTool
except Exception:
    try:
        from crewai.tools.base_tool import BaseTool
    except Exception:
        try:
            from crewai import BaseTool
        except Exception:
            class BaseTool(BaseModel):
                name: str = "tool"
                description: str = "compat tool"
                args_schema: type[BaseModel] | None = None

                def _run(self, *args, **kwargs) -> str:
                    raise NotImplementedError("BaseTool unavailable in current CrewAI install")


# ── Tool input schemas ────────────────────────────────────────────────────────

class _SingleArtistInput(BaseModel):
    artist_id: str = Field(description="Spotify artist ID")


class _MultiArtistInput(BaseModel):
    artist_ids: list[str] = Field(description="List of Spotify artist IDs")


# ── Tools ─────────────────────────────────────────────────────────────────────

class ScoreAudioSimilarityTool(BaseTool):
    name: str = "score_audio_similarity"
    description: str = (
        "Compute audio fingerprint similarity for a list of artist IDs. "
        "Returns mean cosine similarity of audio features (0=diverse, 1=identical). "
        "Ghost artists score > 0.88."
    )
    args_schema: type[BaseModel] = _MultiArtistInput

    def _run(self, artist_ids: list[str]) -> str:
        from src.signals.audio_similarity import score_cluster
        try:
            result = score_cluster(artist_ids)
            return json.dumps({
                "signal": "audio_similarity",
                "score": result.get("suspicion_score"),
                "mean_cosine_similarity": result.get("mean_cosine_similarity"),
                "kaggle_hit_rate": result.get("kaggle_hit_rate"),
                "suspicion_level": result.get("suspicion_level"),
            })
        except Exception as e:
            return json.dumps({"error": str(e), "signal": "audio_similarity", "score": None})


class ScoreFollowerRatioTool(BaseTool):
    name: str = "score_follower_ratio"
    description: str = (
        "Compute catalog density anomaly score for an artist. "
        "High track-per-day rate → suspicious. Returns score 0-1."
    )
    args_schema: type[BaseModel] = _SingleArtistInput

    def _run(self, artist_id: str) -> str:
        from src.signals.follower_ratio import score_artist
        try:
            result = score_artist(artist_id)
            return json.dumps({
                "signal": "follower_ratio",
                "score": result.get("suspicion_score"),
                "track_count": result.get("track_count"),
                "tracks_per_day": result.get("tracks_per_day"),
                "avg_duration_seconds": result.get("avg_duration_seconds"),
                "suspicion_level": result.get("suspicion_level"),
            })
        except Exception as e:
            return json.dumps({"error": str(e), "signal": "follower_ratio", "score": None})


class ScoreMetadataSimilarityTool(BaseTool):
    name: str = "score_metadata_similarity"
    description: str = (
        "Compute metadata similarity for a list of artist IDs. "
        "Measures name patterns, keyword overlap, and track name vocabulary. "
        "Ghost networks share keywords like 'relaxing', 'sleep', 'ambient'."
    )
    args_schema: type[BaseModel] = _MultiArtistInput

    def _run(self, artist_ids: list[str]) -> str:
        from src.signals.metadata_similarity import score_cluster
        try:
            result = score_cluster(artist_ids)
            return json.dumps({
                "signal": "metadata_similarity",
                "score": result.get("suspicion_score"),
                "name_similarity": result.get("pairwise_name_sim"),
                "track_name_similarity": result.get("pairwise_track_sim"),
                "keyword_score": result.get("keyword_score"),
                "suspicion_level": result.get("suspicion_level"),
            })
        except Exception as e:
            return json.dumps({"error": str(e), "signal": "metadata_similarity", "score": None})


class GetCatalogVarianceTool(BaseTool):
    name: str = "get_catalog_variance"
    description: str = (
        "Get audio feature variance for an artist's catalog using the Kaggle dataset. "
        "Low variance = ghost-like. Returns per-feature variance dict."
    )
    args_schema: type[BaseModel] = _SingleArtistInput

    def _run(self, artist_id: str) -> str:
        from pathlib import Path
        import pandas as pd
        from src.utils.kaggle_loader import kaggle
        try:
            df = pd.read_csv("data/processed/neo4j_full_graph.csv")
            track_ids = df[df["artist_id"] == artist_id]["track_id"].dropna().tolist()
            if not track_ids:
                return json.dumps({"found": False, "artist_id": artist_id})
            variance = kaggle.compute_catalog_variance(track_ids)
            return json.dumps({
                "artist_id": artist_id,
                "track_count_in_kaggle": variance.get("track_count", 0),
                "overall_variance": variance.get("overall_variance", 0),
                "variances": {k: v for k, v in variance.items()
                              if k not in ("track_count", "overall_variance")},
            })
        except Exception as e:
            return json.dumps({"error": str(e), "artist_id": artist_id})


# ── Pure Python analysis function (no LLM needed) ────────────────────────────

def analyze_fingerprints(artist_id: str, all_artist_ids: list[str] | None = None) -> dict:
    """
    Run signals 1, 4, 5 for an artist. Returns dict of scores.
    No LLM needed — pure signal computation.
    """
    from src.signals import audio_similarity, follower_ratio, metadata_similarity
    from src.graph.neo4j_client import Neo4jClient

    neo4j = Neo4jClient()

    if all_artist_ids is None:
        rows = neo4j.run("MATCH (a:Artist) RETURN a.spotify_id AS id")
        all_artist_ids = [r["id"] for r in rows]

    results: dict = {"artist_id": artist_id}

    # S1: Audio similarity (often None for niche artists)
    try:
        s1 = audio_similarity.score_artist(artist_id)
        results["s1_audio_similarity"] = s1.get("suspicion_score")
        results["s1_detail"] = {
            "mean_cosine_similarity": s1.get("mean_cosine_similarity"),
            "kaggle_hit_rate": s1.get("kaggle_hit_rate"),
            "suspicion_level": s1.get("suspicion_level"),
        }
    except Exception as e:
        logger.warning(f"S1 failed: {e}")
        results["s1_audio_similarity"] = None

    # S4: Follower/catalog ratio
    try:
        s4 = follower_ratio.score_artist(artist_id, neo4j)
        results["s4_follower_ratio"] = s4.get("suspicion_score")
        results["s4_detail"] = {
            "track_count": s4.get("track_count"),
            "tracks_per_day": s4.get("tracks_per_day"),
            "avg_duration_seconds": s4.get("avg_duration_seconds"),
            "suspicion_level": s4.get("suspicion_level"),
        }
    except Exception as e:
        logger.warning(f"S4 failed: {e}")
        results["s4_follower_ratio"] = None

    # S5: Metadata similarity (cluster signal)
    try:
        s5 = metadata_similarity.score_artist(artist_id, all_artist_ids, neo4j)
        results["s5_metadata_similarity"] = s5.get("suspicion_score")
        results["s5_detail"] = {
            "keyword_count": s5.get("keyword_count"),
            "suspicion_level": s5.get("suspicion_level"),
        }
    except Exception as e:
        logger.warning(f"S5 failed: {e}")
        results["s5_metadata_similarity"] = None

    logger.info(
        f"Fingerprint analysis for {artist_id}: "
        f"S1={results.get('s1_audio_similarity')}, "
        f"S4={results.get('s4_follower_ratio')}, "
        f"S5={results.get('s5_metadata_similarity')}"
    )
    return results


# ── CrewAI Agent definition ───────────────────────────────────────────────────

def build_fingerprint_agent() -> Agent:
    return Agent(
        role="Audio Fingerprint Analyst",
        goal="Detect anomalous audio patterns and metadata signatures in artist catalogs",
        backstory=(
            "Specialist in music information retrieval with expertise in audio feature "
            "analysis and catalog forensics. Identifies ghost artists by detecting "
            "unnaturally uniform audio fingerprints and suspicious metadata patterns."
        ),
        tools=[
            ScoreAudioSimilarityTool(),
            ScoreFollowerRatioTool(),
            ScoreMetadataSimilarityTool(),
            GetCatalogVarianceTool(),
        ],
        verbose=True,
        allow_delegation=False,
    )
