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

import asyncio
import os
import sys
from pathlib import Path

# Ensure project root is on path when run via uvicorn
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from loguru import logger
from starlette.concurrency import run_in_threadpool

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

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (CSS, JS) and paper figures
STATIC_DIR = ROOT / "static"
FIGURES_DIR = ROOT / "paper" / "figures"
STATIC_DIR.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
app.mount("/css", StaticFiles(directory=str(STATIC_DIR / "css")), name="css")
app.mount("/js", StaticFiles(directory=str(STATIC_DIR / "js")), name="js")
app.mount("/figures", StaticFiles(directory=str(FIGURES_DIR)), name="figures")


def _serve_index_html() -> FileResponse:
    """Serve SPA index with a clear error if the build artifact is missing."""
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(
            status_code=503,
            detail="Frontend is not built yet: static/index.html is missing.",
        )
    return FileResponse(str(index_file))


@app.get("/app", include_in_schema=False)
@app.get("/app/{rest_of_path:path}", include_in_schema=False)
async def serve_spa(rest_of_path: str = ""):
    """Serve the SPA index for all frontend routes."""
    return _serve_index_html()

# ── Request/Response models ───────────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    artist_id: str
    artist_name: str | None = None
    track_name: str | None = None
    run_cross_platform: bool = False


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

@app.get("/", include_in_schema=False)
async def root():
    """Serve the SPA."""
    return _serve_index_html()


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

    all_ok = checks["api"] == "ok" and checks["neo4j"] == "ok" and checks["signals"] == "ok"
    return JSONResponse(
        content={"status": "ok" if all_ok else "degraded", "checks": checks},
        status_code=200,
    )


@app.post("/analyze", response_model=AnalyzeResponse, summary="Full artist analysis")
async def analyze_artist(req: AnalyzeRequest):
    """
    Run the full 7-signal ghost detection pipeline for an artist.

    Uses cached data and Neo4j — no Spotify API calls unless live_api=True
    (not yet exposed via this endpoint).
    """
    from src.agents.crew import run_analysis
    from src.graph.neo4j_client import Neo4jClient

    artist_id = req.artist_id.strip()
    if not artist_id:
        raise HTTPException(status_code=400, detail="artist_id is required")

    logger.info(f"POST /analyze — artist_id={artist_id}")

    # Reject artists not in our dataset — avoids misleading "LIKELY ORGANIC" for unknown IDs
    KNOWN_IDS = {
        "6bo3atMVp3qFECNALVwq9N",  # Relaxing White Noise
        "39t4EeLBfpT72UQJVkIeuj",  # Meditation Relax Club (legacy ID)
        "3BqBPFLxBkzKQTkuBPGMNF",  # Meditation Relax Club (frontend ID)
        "4Wx3ZL6d6p1gVMtwQ2YWsz",  # Calmo
        "5gqhueRUZEa7VDnQt4HODp",  # Nils Frahm (legacy ID)
        "5hVghJ3sCFHFJoLnSHySjL",  # Nils Frahm (frontend ID)
    }
    if artist_id not in KNOWN_IDS:
        # Also accept artists present in Neo4j
        try:
            neo4j = Neo4jClient()
            rows = neo4j.run(
                "MATCH (a:Artist {spotify_id: $id}) RETURN a.name AS name",
                id=artist_id,
            )
            if not rows:
                raise HTTPException(
                    status_code=404,
                    detail=(
                        f"Artist '{artist_id}' is not in the GhostTrack dataset. "
                        "This tool only analyzes the four artists in our study panel. "
                        "Use one of the quick-pick buttons below the search box."
                    ),
                )
        except HTTPException:
            raise
        except Exception:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"Artist '{artist_id}' not found. Use one of the four study-panel artists."
                ),
            )

    try:
        result = run_analysis(
            artist_id=artist_id,
            artist_name=req.artist_name,
            track_name=req.track_name,
            run_cross_platform=req.run_cross_platform,
        )
    except Exception as e:
        logger.error(f"/analyze failed for {req.artist_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    # Augment with GNN score (optional). Disabled by default because it
    # re-runs heavy verdict computation and can significantly increase latency.
    rule_score = result.get("overall_score", 0.0)
    gnn_score = None
    gnn_available = False
    combined_score = rule_score
    enable_gnn = os.getenv("ENABLE_GNN_AUGMENT", "0").strip().lower() in {"1", "true", "yes", "on"}
    if enable_gnn:
        try:
            from src.signals.verdict import compute_verdict_gnn
            gnn_result = compute_verdict_gnn(
                artist_id=artist_id,
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


@app.get("/search", summary="Search artists by name")
async def search_artists(q: str, limit: int = 5):
    """
    Search Spotify for artists by name. Returns top matches with id, name, image.
    """
    if not q or len(q.strip()) < 2:
        raise HTTPException(status_code=400, detail="Query too short")
    limit = max(1, min(limit, 25))
    try:
        from src.ingest.live_ingest import search_artist
        results = search_artist(q.strip())
        return {"results": results[:limit]}
    except Exception as e:
        logger.error(f"/search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/artist-tracks", summary="Fetch latest & top tracks with YouTube view counts")
async def artist_tracks(artist: str):
    """
    Given an artist name, returns:
      - latest_track: most recently released track found
      - top_track: highest-viewed track on YouTube
      - source: which API provided the data (youtube / itunes / openai)
    Fallback chain: YouTube API → iTunes Search → OpenAI GPT-4o.
    """
    artist = artist.strip()
    if not artist:
        raise HTTPException(status_code=400, detail="artist name required")
    if len(artist) > 200:
        raise HTTPException(status_code=400, detail="artist name too long")

    # ── 1. YouTube Data API v3 ─────────────────────────────────────────────
    yt_key = os.getenv("YOUTUBE_API_KEY", "")
    if yt_key:
        try:
            import httpx
            # Search for up to 10 videos from this artist (official channel / topic)
            search_params = {
                "part": "snippet",
                "q": f"{artist} official music",
                "type": "video",
                "maxResults": 10,
                "order": "relevance",
                "key": yt_key,
            }
            with httpx.Client(timeout=10) as client:
                r = client.get("https://www.googleapis.com/youtube/v3/search", params=search_params)
                r.raise_for_status()
                items = r.json().get("items", [])

            if items:
                video_ids = [it["id"]["videoId"] for it in items if it.get("id", {}).get("videoId")]
                # Get stats for all found videos
                stats_params = {
                    "part": "statistics,snippet",
                    "id": ",".join(video_ids),
                    "key": yt_key,
                }
                with httpx.Client(timeout=10) as client:
                    r2 = client.get("https://www.googleapis.com/youtube/v3/videos", params=stats_params)
                    r2.raise_for_status()
                    vids = r2.json().get("items", [])

                if vids:
                    # Prefer videos from a channel whose name contains the artist name
                    artist_lower = artist.lower()
                    artist_vids = [v for v in vids if artist_lower in v["snippet"].get("channelTitle","").lower()]
                    pool = artist_vids if artist_vids else vids
                    # Latest = most recently published
                    latest = max(pool, key=lambda v: v["snippet"].get("publishedAt",""))
                    # Top = highest view count
                    top = max(pool, key=lambda v: int(v.get("statistics", {}).get("viewCount", 0)))

                    def yt_track(v):
                        return {
                            "title": v["snippet"]["title"],
                            "views": int(v.get("statistics", {}).get("viewCount", 0)),
                            "published": v["snippet"].get("publishedAt", "")[:10],
                            "url": f"https://www.youtube.com/watch?v={v['id']}",
                            "thumbnail": v["snippet"].get("thumbnails", {}).get("medium", {}).get("url", ""),
                        }

                    return {
                        "source": "youtube",
                        "artist": artist,
                        "latest_track": yt_track(latest),
                        "top_track": yt_track(top),
                    }
        except Exception as e:
            logger.warning(f"/artist-tracks YouTube failed for '{artist}': {e}")

    # ── 2. iTunes Search API (free, no key) ───────────────────────────────
    try:
        import httpx
        itunes_params = {
            "term": artist,
            "media": "music",
            "entity": "song",
            "limit": 10,
            "sort": "recent",
        }
        with httpx.Client(timeout=10) as client:
            r = client.get("https://itunes.apple.com/search", params=itunes_params)
            r.raise_for_status()
            results = r.json().get("results", [])

        if results:
            # iTunes doesn't give views — use trackCount as proxy
            latest = results[0]
            top = max(results, key=lambda t: t.get("trackTimeMillis", 0))

            def itunes_track(t):
                return {
                    "title": t.get("trackName", "Unknown"),
                    "views": None,
                    "published": t.get("releaseDate", "")[:10],
                    "url": t.get("trackViewUrl", ""),
                    "thumbnail": t.get("artworkUrl100", ""),
                    "album": t.get("collectionName", ""),
                    "duration_ms": t.get("trackTimeMillis", 0),
                }

            return {
                "source": "itunes",
                "artist": artist,
                "latest_track": itunes_track(latest),
                "top_track": itunes_track(top),
            }
    except Exception as e:
        logger.warning(f"/artist-tracks iTunes failed for '{artist}': {e}")

    # ── 3. OpenAI fallback ────────────────────────────────────────────────
    try:
        from openai import OpenAI
        oai = OpenAI()
        prompt = (
            f"For the music artist '{artist}', give me:\n"
            f"1. Their most recent / latest known track title and approximate release year\n"
            f"2. Their most popular / highest-viewed track title and approximate YouTube view count\n"
            f"Reply in JSON only: {{\"latest\": {{\"title\": \"...\", \"year\": \"...\"}}, "
            f"\"top\": {{\"title\": \"...\", \"views\": 123456789}}}}"
        )
        resp = oai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=200,
            response_format={"type": "json_object"},
        )
        import json
        data = json.loads(resp.choices[0].message.content)
        return {
            "source": "openai",
            "artist": artist,
            "latest_track": {
                "title": data.get("latest", {}).get("title", "Unknown"),
                "views": None,
                "published": data.get("latest", {}).get("year", ""),
                "url": "",
                "thumbnail": "",
            },
            "top_track": {
                "title": data.get("top", {}).get("title", "Unknown"),
                "views": data.get("top", {}).get("views"),
                "published": "",
                "url": "",
                "thumbnail": "",
            },
        }
    except Exception as e:
        logger.warning(f"/artist-tracks OpenAI fallback failed for '{artist}': {e}")

    return {
        "source": "unavailable",
        "artist": artist,
        "latest_track": None,
        "top_track": None,
    }


@app.post("/analyze-live", summary="Live analysis for any Spotify artist")
async def analyze_live(req: AnalyzeRequest):
    """
    Run live 3-signal analysis (S2, S5, S7) for any artist.
    Fetches albums/tracks from Spotify API in real time.
    S1, S3, S4, S6 shown as unavailable (restricted endpoints).
    """
    artist_id = req.artist_id.strip()
    if not artist_id:
        raise HTTPException(status_code=400, detail="artist_id is required")

    analyze_live_timeout_seconds = int(os.getenv("ANALYZE_LIVE_TIMEOUT_SECONDS", "40"))

    logger.info(f"POST /analyze-live — artist_id={artist_id}")
    try:
        from src.ingest.live_ingest import analyze_live as _analyze_live
        result = await asyncio.wait_for(
            run_in_threadpool(
                _analyze_live,
                artist_id=artist_id,
                artist_name=req.artist_name,
                run_s7=req.run_cross_platform,
            ),
            timeout=analyze_live_timeout_seconds,
        )
    except asyncio.TimeoutError:
        logger.error(
            f"/analyze-live timeout for {artist_id} "
            f"after {analyze_live_timeout_seconds}s"
        )
        raise HTTPException(
            status_code=504,
            detail=(
                "Live analysis timed out while waiting on upstream APIs. "
                "Please retry in a moment."
            ),
        )
    except Exception as e:
        logger.error(f"/analyze-live failed for {req.artist_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    return AnalyzeResponse(
        artist_id=result["artist_id"],
        artist_name=result["artist_name"],
        signals=result.get("signal_scores", {}),
        verdict_score=result.get("overall_score", 0.0),
        verdict_label=result.get("verdict", "UNKNOWN"),
        confidence=result.get("confidence", 0.0),
        explanation=result.get("explanation", ""),
        timing_seconds=result.get("timing", {}).get("total_seconds", 0.0),
        rule_based_score=result.get("overall_score"),
        gnn_score=None,
        gnn_available=False,
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
        if hops < 1 or hops > 2:
            raise HTTPException(status_code=400, detail="hops must be 1 or 2")
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
            "hops": hops,
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


class ChatRequest(BaseModel):
    question: str
    history: list[dict] = []


@app.post("/chat", summary="AI research assistant")
async def chat(req: ChatRequest):
    """Answer research questions about the GhostTrack project using GPT-4o."""
    try:
        from openai import OpenAI
        client = OpenAI()
        system_prompt = """You are a research assistant for the GhostTrack project.
PROJECT: GhostTrack — Ghost Artist Detection on Spotify | INFO 7390, Spring 2026

POST-AUDIT EXERCISE RESULTS (use these values, NOT older proxy numbers):

- Ex1 (Catalog Variance): Same-genre comparison N=13 ghost ambient vs N=75 organic ambient. Levene W=15.7 p=0.0002; Cohen's d=-1.45 to -2.08 per feature (large effect). Genre confound eliminated.

- Ex2 (Playlist Entropy): 7-feature marginal entropy across 30 Kaggle-proxy playlists. One-way ANOVA F=0.25 p=0.78 — HONEST NEGATIVE result. Playlist-level aggregation does not distinguish ghost-suspect from editorial or fan-curated groups.

- Ex3 (ISRC Attribution): Expanded bipartite graph of 3 ghost + 17 organic artists across 27 registrant codes. ALL ghost artists use CUSTOM_REGISTRANT (small, non-public registrants). Organic artists use known aggregators (TuneCore, DistroKid) or labels. The categorical distinction beats HHI magnitude as a fraud signal.

- Ex4 (HHI Concentration): REAL ISRC-derived HHI values:
    RWN = 0.672
    MRC = 0.515
    Calmo = 0.452
  Mann-Whitney U p=0.003, rank-biserial r=1.000 vs 30 organic artists. Youden-optimal threshold HHI>=0.353. NOTE: Previous proxy values (0.88/0.66/0.54) were variance-derived and tautological; the real values are smaller but defensible.

- Ex5 (Release Cadence): Ghost N=14 vs organic baseline N=1031 across 5 genres. KS D=1.000, p<0.001. Cohen's d=3.44 (very large effect). Robust sensitivity across 1d-14d thresholds: 100% TPR, 0% FPR. Caveat: prolific organic artists (Buckethead, King Gizzard, Merzbow, Guided By Voices) absent from Kaggle dataset — documented limitation.

- Ex6 (Signal Framework): Post-audit, 2 of 7 signals are discriminative at scale: S2 Release Cadence (d=3.44) and S5 Metadata Similarity (d=-0.91, direction documented as collinear with S2). S4 Catalog Density (d=0.32) is below threshold. S1 Audio, S3 Playlist Co-occur, S6 Graph/HHI, S7 Cross-Platform marked N/A at scale due to API access constraints.

- Ex7 (GNN Model): All 5 baselines (LogisticRegression, RandomForest, MLP, GraphSAGE, HGT) achieve AUC=1.000 on n=76 artists. Graph structure adds NO discriminative signal beyond tabular features at this scale. SHAP analysis shows track_count is the dominant feature. This is a NEGATIVE result for the GNN hypothesis — reported honestly. Perfect AUC reflects task triviality at small scale, not model sophistication.

KEY BEHAVIORAL INSIGHT: Ghost artist fraud is Spotify-economic stream farming with pre-staged infrastructure, NOT cross-platform absence. Relaxing White Noise has 353M YouTube views yet exhibits ghost behavioral signatures.

METHODOLOGICAL CONTRIBUTION: Proxy-labeled evaluation produces tautological AUC=1.000 results — a pattern documented throughout this work. External ground truth from independent sources (DOJ Smith indictment, Dagens Nyheter investigation) is essential for credible validation in streaming fraud research.

Respond clearly and concisely. Use markdown. Cite specific exercise results when answering. If a user asks about the older proxy HHI values, explain that those were variance-derived and have been superseded by real ISRC calculations."""
        messages = [
            {"role": "system", "content": system_prompt},
        ]
        for turn in req.history[-6:]:
            messages.append({"role": "user", "content": turn.get("user", "")})
            messages.append({"role": "assistant", "content": turn.get("assistant", "")})
        messages.append({"role": "user", "content": req.question})
        resp = client.chat.completions.create(model="gpt-4o", messages=messages, temperature=0.4, max_tokens=1500)
        return {"answer": resp.choices[0].message.content}
    except Exception as e:
        return {"answer": f"AI unavailable: {e}"}


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
