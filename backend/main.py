"""
EDA-Music FastAPI Backend

Serves the ghost-artist detection pipeline via REST API.
Connects to Neo4j for graph data, runs signal scoring,
and returns structured results for the Streamlit frontend.

Start:
    uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload

Docs:
    http://localhost:8000/docs
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure project root is on path when run via uvicorn
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from loguru import logger

app = FastAPI(
    title="EDA for Music — Ghost Artist Detection API",
    description=(
        "REST API for detecting ghost/AI-generated artists on Spotify. "
        "Uses a 7-signal detection framework with Neo4j graph backend. "
        "INFO 7390 · Spring 2026"
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS — allow Streamlit frontend on any local port
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Request/Response models ───────────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    artist_id: str
    artist_name: str | None = None
    track_name: str | None = None
    run_cross_platform: bool = False


class SignalScore(BaseModel):
    score: float | None
    level: str | None


class AnalyzeResponse(BaseModel):
    artist_id: str
    artist_name: str
    signals: dict[str, float | None]
    verdict_score: float
    verdict_label: str
    confidence: float
    explanation: str
    timing_seconds: float
    rule_based_score: float | None = None
    gnn_score: float | None = None
    gnn_available: bool = False


# ── Lazy singletons ───────────────────────────────────────────────────────────

_neo4j_client = None

def _get_neo4j():
    global _neo4j_client
    if _neo4j_client is None:
        from src.graph.neo4j_client import Neo4jClient
        _neo4j_client = Neo4jClient()
    return _neo4j_client


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/", summary="Health check")
async def root():
    """Health check endpoint."""
    return {
        "status": "ok",
        "project": "EDA for Music",
        "version": "1.0.0",
        "description": "Ghost Artist Detection API",
    }


@app.get("/health", summary="Detailed health check")
async def health():
    """Check connectivity to Neo4j and verify signal modules load."""
    checks = {"api": "ok", "neo4j": "unknown", "signals": "unknown"}
    try:
        neo4j = _get_neo4j()
        neo4j.run("RETURN 1 AS ok")
        checks["neo4j"] = "ok"
    except Exception as e:
        checks["neo4j"] = f"error: {e}"

    try:
        from src.agents.crew import run_analysis  # noqa: F401
        checks["signals"] = "ok"
    except Exception as e:
        checks["signals"] = f"error: {e}"

    all_ok = all(v == "ok" for v in checks.values())
    return JSONResponse(
        content={"status": "ok" if all_ok else "degraded", "checks": checks},
        status_code=200 if all_ok else 503,
    )


@app.post("/analyze", response_model=AnalyzeResponse, summary="Full artist analysis")
async def analyze_artist(req: AnalyzeRequest):
    """
    Run the full 7-signal ghost detection pipeline for an artist.

    Uses cached data and Neo4j — no Spotify API calls unless live_api=True
    (not yet exposed via this endpoint).
    """
    from src.agents.crew import run_analysis

    logger.info(f"POST /analyze — artist_id={req.artist_id}")

    try:
        result = run_analysis(
            artist_id=req.artist_id,
            artist_name=req.artist_name,
            track_name=req.track_name,
            run_cross_platform=req.run_cross_platform,
        )
    except Exception as e:
        logger.error(f"/analyze failed for {req.artist_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    # Augment with GNN score (non-fatal — falls back gracefully)
    rule_score = result.get("overall_score", 0.0)
    gnn_score = None
    gnn_available = False
    combined_score = rule_score
    try:
        from src.signals.verdict import compute_verdict_gnn
        gnn_result = compute_verdict_gnn(
            artist_id=req.artist_id,
            artist_name=result.get("artist_name"),
            run_s7=False,
        )
        gnn_score = gnn_result.get("gnn_score")
        gnn_available = gnn_result.get("gnn_available", False)
        combined_score = gnn_result.get("combined_score", rule_score)
        # Use the GNN-combined verdict label and score
        result["verdict"] = gnn_result.get("verdict", result.get("verdict"))
        result["overall_score"] = combined_score
    except Exception as e:
        logger.warning(f"GNN augmentation failed for {req.artist_id}: {e}")

    return AnalyzeResponse(
        artist_id=result["artist_id"],
        artist_name=result["artist_name"],
        signals=result.get("signal_scores", {}),
        verdict_score=combined_score,
        verdict_label=result.get("verdict", "UNKNOWN"),
        confidence=result.get("confidence", 0.0),
        explanation=result.get("explanation", ""),
        timing_seconds=result.get("timing", {}).get("total_seconds", 0.0),
        rule_based_score=rule_score,
        gnn_score=gnn_score,
        gnn_available=gnn_available,
    )


@app.get("/artist/{artist_id}/signals", summary="Per-signal scores for an artist")
async def get_artist_signals(artist_id: str):
    """
    Get all 7 signal scores for a specific artist.
    Returns individual score + detail dict per signal.
    """
    from src.graph.neo4j_client import Neo4jClient
    from src.signals import (
        cadence_sync, follower_ratio, graph_density,
        metadata_similarity, playlist_cooccurrence, audio_similarity,
    )

    neo4j = _get_neo4j()

    # Resolve artist name
    rows = neo4j.run(
        "MATCH (a:Artist {spotify_id: $id}) RETURN a.name AS name", id=artist_id
    )
    if not rows:
        raise HTTPException(status_code=404, detail=f"Artist {artist_id} not found in Neo4j")
    artist_name = rows[0]["name"]

    all_ids = [r["id"] for r in neo4j.run("MATCH (a:Artist) RETURN a.spotify_id AS id")]

    signals = {}
    try:
        s1 = audio_similarity.score_artist(artist_id)
        signals["signal_1"] = {
            "name": "Audio Fingerprint Similarity",
            "score": s1.get("suspicion_score"),
            "detail": {"kaggle_hit_rate": s1.get("kaggle_hit_rate"),
                        "mean_cosine_similarity": s1.get("mean_cosine_similarity")},
        }
    except Exception as e:
        signals["signal_1"] = {"name": "Audio Fingerprint Similarity", "score": None, "error": str(e)}

    try:
        s2 = cadence_sync.score_artist(artist_id, neo4j)
        signals["signal_2"] = {
            "name": "Release Cadence Synchrony",
            "score": s2.get("suspicion_score"),
            "detail": {"closure_rate": s2.get("closure_rate"),
                        "burst_ratio": s2.get("burst_ratio"),
                        "median_gap_days": s2.get("median_gap_days")},
        }
    except Exception as e:
        signals["signal_2"] = {"name": "Release Cadence Synchrony", "score": None, "error": str(e)}

    try:
        s3 = playlist_cooccurrence.score_artist(artist_id, all_ids, neo4j)
        signals["signal_3"] = {
            "name": "Playlist Co-occurrence",
            "score": s3.get("suspicion_score"),
        }
    except Exception as e:
        signals["signal_3"] = {"name": "Playlist Co-occurrence", "score": None, "error": str(e)}

    try:
        s4 = follower_ratio.score_artist(artist_id, neo4j)
        signals["signal_4"] = {
            "name": "Catalog Density Anomaly",
            "score": s4.get("suspicion_score"),
            "detail": {"track_count": s4.get("track_count"),
                        "tracks_per_day": s4.get("tracks_per_day"),
                        "avg_duration_seconds": s4.get("avg_duration_seconds")},
        }
    except Exception as e:
        signals["signal_4"] = {"name": "Catalog Density Anomaly", "score": None, "error": str(e)}

    try:
        s5 = metadata_similarity.score_artist(artist_id, all_ids, neo4j)
        signals["signal_5"] = {
            "name": "Metadata Similarity",
            "score": s5.get("suspicion_score"),
            "detail": {"keyword_count": s5.get("keyword_count")},
        }
    except Exception as e:
        signals["signal_5"] = {"name": "Metadata Similarity", "score": None, "error": str(e)}

    try:
        s6 = graph_density.score_artist(artist_id, neo4j)
        signals["signal_6"] = {
            "name": "Graph Density / HHI",
            "score": s6.get("suspicion_score"),
            "detail": {"hhi": s6.get("hhi"),
                        "dominant_share": s6.get("dominant_share"),
                        "num_companies": s6.get("num_companies")},
        }
    except Exception as e:
        signals["signal_6"] = {"name": "Graph Density / HHI", "score": None, "error": str(e)}

    signals["signal_7"] = {
        "name": "Cross-Platform Discrepancy",
        "score": None,
        "note": "Run /analyze with run_cross_platform=true to include Signal 7",
    }

    return {"artist_id": artist_id, "artist_name": artist_name, "signals": signals}


@app.get("/graph/stats", summary="Neo4j database statistics")
async def get_graph_stats():
    """Return count of each node type in the Neo4j database."""
    try:
        neo4j = _get_neo4j()
        counts = neo4j.count_nodes()

        # Also count relationships
        rel_rows = neo4j.run(
            "MATCH ()-[r]->() RETURN type(r) AS rel_type, count(r) AS cnt ORDER BY cnt DESC"
        )
        relationships = {r["rel_type"]: r["cnt"] for r in rel_rows}

        return {
            "nodes": counts,
            "relationships": relationships,
            "total_nodes": sum(counts.values()),
            "total_relationships": sum(relationships.values()),
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Neo4j error: {e}")


@app.get("/graph/neighborhood/{artist_id}", summary="Artist neighborhood graph")
async def get_neighborhood(artist_id: str, hops: int = 1):
    """
    Get the ISRC bipartite neighborhood graph for an artist.
    Returns nodes (artist + production companies) and edges.
    """
    try:
        neo4j = _get_neo4j()
        rows = neo4j.run(
            """
            MATCH (a:Artist {spotify_id: $id})-[:RELEASED]->(al:Album)
                  -[:CONTAINS]->(t:Track)-[:REGISTERED_WITH]->(c:ProductionCompany)
            RETURN c.isrc_prefix AS prefix, c.name AS company,
                   count(t) AS track_count
            ORDER BY track_count DESC
            """,
            id=artist_id,
        )
        name_row = neo4j.run(
            "MATCH (a:Artist {spotify_id: $id}) RETURN a.name AS name, a.is_ghost AS ghost",
            id=artist_id,
        )
        if not name_row:
            raise HTTPException(status_code=404, detail=f"Artist {artist_id} not found")

        artist_name = name_row[0]["name"]
        is_ghost = name_row[0].get("ghost", False)

        nodes = [{
            "id": artist_id,
            "label": artist_name,
            "type": "artist",
            "is_ghost": is_ghost,
            "color": "#e74c3c" if is_ghost else "#2ecc71",
        }]
        edges = []

        for r in rows:
            node_id = f"company_{r['prefix']}"
            nodes.append({
                "id": node_id,
                "label": r["company"] or r["prefix"],
                "type": "production_company",
                "color": "#a78bfa",
            })
            edges.append({
                "source": artist_id,
                "target": node_id,
                "weight": r["track_count"],
                "label": f"{r['track_count']} tracks",
            })

        return {
            "artist_id": artist_id,
            "artist_name": artist_name,
            "nodes": nodes,
            "edges": edges,
            "company_count": len(rows),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.get("/graph/isrc-clusters", summary="ISRC-based production company clusters")
async def get_isrc_clusters():
    """
    Find production companies that register tracks for multiple artists.
    High-value signal for coordinated ghost networks.
    """
    try:
        neo4j = _get_neo4j()

        # All artists per company
        rows = neo4j.run(
            """
            MATCH (a:Artist)-[:RELEASED]->(al:Album)-[:CONTAINS]->(t:Track)
                  -[:REGISTERED_WITH]->(c:ProductionCompany)
            WITH c, collect(DISTINCT a.name) AS artists,
                 count(DISTINCT t) AS track_count
            RETURN c.isrc_prefix AS prefix,
                   c.name AS company_name,
                   artists,
                   size(artists) AS artist_count,
                   track_count
            ORDER BY track_count DESC
            """
        )

        clusters = []
        for r in rows:
            clusters.append({
                "prefix": r["prefix"],
                "company_name": r["company_name"],
                "artists": r["artists"],
                "artist_count": r["artist_count"],
                "track_count": r["track_count"],
                "shared": r["artist_count"] > 1,
            })

        return {
            "clusters": clusters,
            "total_companies": len(clusters),
            "shared_companies": sum(1 for c in clusters if c["shared"]),
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.get("/exercises/summary", summary="Summary of all exercise findings")
async def get_exercises_summary():
    """Return key findings from all 5 exercises + signal testing."""
    return {
        "exercise_1": {
            "name": "Catalog Coherence",
            "finding": "12.5× variance ratio — ghost catalogs have 12.5× lower audio feature variance than organic",
            "method": "PCA + Levene's test on Kaggle 114K-track dataset",
            "figure": "fig1_catalog_coherence.png",
            "p_value": "< 0.001 all 4 features",
        },
        "exercise_2": {
            "name": "Playlist Entropy",
            "finding": "Editorial=2.59 bits, Fan=2.89 bits, Ghost-suspect=2.51 bits Shannon entropy",
            "method": "Shannon entropy of audio feature distributions, Kaggle simulation",
            "figure": "fig2_playlist_entropy.png",
            "note": "Editorial playlist API returns 404 — used Kaggle genre simulation",
        },
        "exercise_3": {
            "name": "ISRC Attribution",
            "finding": "8 production companies, 490 tracks, no cross-artist ISRC sharing in seed set",
            "method": "ISRC prefix → ProductionCompany mapping via Neo4j",
            "figure": "fig3_isrc_join.png",
        },
        "exercise_4": {
            "name": "Bipartite Neighborhood",
            "finding": "HHI: RWN=0.88, MRC=0.66, Calmo=0.54 — ghost artists show extreme ISRC concentration",
            "method": "Herfindahl-Hirschman Index on bipartite Artist×Company graph",
            "figure": "fig4_bipartite_neighborhood.png",
        },
        "exercise_5": {
            "name": "Recommendation Walk",
            "finding": "Walk closure: RWN=81%, MRC=95% vs Nils Frahm=0% — bulk-upload fingerprint",
            "method": "Temporal release-date walk, closure = % consecutive same-day releases",
            "figure": "fig5_recommendation_walk.png",
        },
        "exercise_6": {
            "name": "Signal Testing",
            "finding": (
                "S2 cadence + S4 catalog density + S6 HHI most discriminative. "
                "SURPRISE: RWN has 353M YouTube views — cross-platform presence ≠ organic."
            ),
            "method": "7-signal framework on 4 artists (3 ghost, 1 organic)",
            "figure": "fig6_signal_radar.png",
            "youtube_views": {
                "Relaxing White Noise": 353775028,
                "Meditation Relax Club": 157581269,
                "Calmo": 155,
                "Nils Frahm": 9107596,
            },
        },
    }


@app.get("/model/info", summary="GNN model architecture and training stats")
async def get_model_info():
    """Return GAT model architecture, training summary, and feature list."""
    import json as _json
    from pathlib import Path

    ROOT = Path(__file__).resolve().parents[1]
    summary_path = ROOT / "data" / "processed" / "gnn_training_summary.json"
    meta_path = ROOT / "data" / "processed" / "gnn_dataset_meta.json"
    model_path = ROOT / "data" / "processed" / "gat_model.pt"

    summary = _json.loads(summary_path.read_text()) if summary_path.exists() else {}
    meta = _json.loads(meta_path.read_text()) if meta_path.exists() else {}

    return {
        "architecture": {
            "model": "GhostDetectorGAT",
            "type": "Graph Attention Network (GAT)",
            "in_channels": meta.get("num_features", 8),
            "hidden_channels": 32,
            "heads": 4,
            "dropout": 0.3,
            "epochs_trained": 200,
        },
        "features": meta.get("feature_names", []),
        "dataset": meta.get("num_nodes", 0),
        "training_summary": summary,
        "model_available": model_path.exists(),
    }


@app.get("/artists", summary="List all artists in the database")
async def list_artists():
    """List all artists in Neo4j with their ghost labels."""
    try:
        neo4j = _get_neo4j()
        rows = neo4j.run(
            """
            MATCH (a:Artist)
            OPTIONAL MATCH (a)-[:RELEASED]->(al:Album)
            WITH a, count(al) AS album_count
            RETURN a.spotify_id AS id, a.name AS name,
                   a.is_ghost AS is_ghost, a.label AS label,
                   album_count
            ORDER BY a.name
            """
        )
        return {"artists": rows, "count": len(rows)}
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))
