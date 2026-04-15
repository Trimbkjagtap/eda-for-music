"""
EDA-Music Agent Crew — Orchestrates the full analysis pipeline.

Architecture
------------
Two execution modes:

  PIPELINE MODE (default, run_analysis):
    Runs the 4 agents as pure Python functions — no LLM API calls.
    Fast (~2-5s per artist), works entirely from cached data and Neo4j.
    This is what the FastAPI backend and notebooks use.

  CREW MODE (run_analysis_with_crew):
    Wraps the pipeline in proper CrewAI Agent/Task/Crew objects.
    Requires OPENAI_API_KEY for the LLM-backed reasoning layer.
    Adds natural language synthesis on top of the numerical signals.

Usage
-----
    from src.agents.crew import run_analysis

    # Pipeline mode (fast, no LLM)
    result = run_analysis("6bo3atMVp3qFECNALVwq9N")

    # Crew mode (LLM-backed)
    result = run_analysis_with_crew("6bo3atMVp3qFECNALVwq9N")

    # Batch
    results = run_batch_analysis(["id1", "id2", "id3"])
"""
from __future__ import annotations

import time
from typing import Any

from loguru import logger

from src.agents.crawler_agent import crawl_artist_from_cache
from src.agents.fingerprint_analyst import analyze_fingerprints
from src.agents.graph_builder import analyze_graph
from src.agents.verdict_agent import synthesize_verdict
from src.graph.neo4j_client import Neo4jClient


# ── Pipeline Mode (no LLM) ────────────────────────────────────────────────────

def run_analysis(
    artist_id: str,
    artist_name: str | None = None,
    track_name: str | None = None,
    live_api: bool = False,
    run_cross_platform: bool = False,
) -> dict[str, Any]:
    """
    Run the full 4-stage analysis pipeline for one artist.

    Stages:
      1. Crawler   — gather cached metadata, count albums/tracks
      2. Fingerprint Analyst — signals 1, 4, 5 (audio + metadata)
      3. Graph Builder       — signals 2, 3, 6 (cadence + graph)
      4. Verdict Agent       — combine into final verdict

    Parameters
    ----------
    artist_id          : Spotify artist ID
    artist_name        : Optional display name (resolved from Neo4j if None)
    track_name         : Optional track for cross-platform check
    live_api           : If True, allow Spotify API calls (uses cache first)
    run_cross_platform : If True, run Signal 7 (YouTube + Apple Music API calls)

    Returns
    -------
    Full result dict with signal_scores, verdict, explanation, timing.
    """
    t0 = time.perf_counter()
    logger.info(f"=== Starting analysis pipeline for {artist_id} ===")

    neo4j = Neo4jClient()
    all_rows = neo4j.run("MATCH (a:Artist) RETURN a.spotify_id AS id")
    all_artist_ids = [r["id"] for r in all_rows]
    if artist_id not in all_artist_ids:
        all_artist_ids.append(artist_id)

    # Resolve artist name
    if artist_name is None:
        rows = neo4j.run(
            "MATCH (a:Artist {spotify_id: $id}) RETURN a.name AS name", id=artist_id
        )
        artist_name = rows[0]["name"] if rows else artist_id

    # ── Stage 1: Crawl ───────────────────────────────────────────────────────
    t1 = time.perf_counter()
    crawl_result = _safe_run("crawler", crawl_artist_from_cache, artist_id)
    t1_end = time.perf_counter()
    logger.info(f"Stage 1 (crawl) done in {t1_end - t1:.2f}s")

    # ── Stage 2: Fingerprint signals (1, 4, 5) ───────────────────────────────
    t2 = time.perf_counter()
    fingerprint_result = _safe_run(
        "fingerprint", analyze_fingerprints, artist_id, all_artist_ids
    )
    t2_end = time.perf_counter()
    logger.info(f"Stage 2 (fingerprint) done in {t2_end - t2:.2f}s")

    # ── Stage 3: Graph signals (2, 3, 6) ─────────────────────────────────────
    t3 = time.perf_counter()
    graph_result = _safe_run(
        "graph", analyze_graph, artist_id, all_artist_ids
    )
    t3_end = time.perf_counter()
    logger.info(f"Stage 3 (graph) done in {t3_end - t3:.2f}s")

    # ── Stage 4: Signal 7 (optional cross-platform) ──────────────────────────
    s7_score = None
    s7_detail = {}
    if run_cross_platform:
        from src.signals.cross_platform import score_artist as cross_score
        t4 = time.perf_counter()
        try:
            r7 = cross_score(artist_name, track_name, artist_id, neo4j)
            s7_score = r7.get("suspicion_score")
            s7_detail = r7
        except Exception as e:
            logger.warning(f"S7 cross-platform failed: {e}")
        t4_end = time.perf_counter()
        logger.info(f"Stage 3b (cross-platform) done in {t4_end - t4:.2f}s")

    # ── Stage 4: Verdict ─────────────────────────────────────────────────────
    all_signal_scores = {
        "s1_audio_similarity": fingerprint_result.get("s1_audio_similarity"),
        "s2_cadence_sync": graph_result.get("s2_cadence_sync"),
        "s3_playlist_cooccurrence": graph_result.get("s3_playlist_cooccurrence"),
        "s4_follower_ratio": fingerprint_result.get("s4_follower_ratio"),
        "s5_metadata_similarity": fingerprint_result.get("s5_metadata_similarity"),
        "s6_graph_density": graph_result.get("s6_graph_density"),
        "s7_cross_platform": s7_score,
    }

    signal_details = {
        "s1": fingerprint_result.get("s1_detail", {}),
        "s2": graph_result.get("s2_detail", {}),
        "s3": graph_result.get("s3_detail", {}),
        "s4": fingerprint_result.get("s4_detail", {}),
        "s5": fingerprint_result.get("s5_detail", {}),
        "s6": graph_result.get("s6_detail", {}),
        "s7": s7_detail,
    }

    verdict = synthesize_verdict(
        artist_id=artist_id,
        all_signal_scores=all_signal_scores,
        signal_details=signal_details,
        artist_name=artist_name,
    )

    total_time = time.perf_counter() - t0

    result = {
        **verdict,
        "signal_details": signal_details,
        "crawl_summary": crawl_result,
        "timing": {
            "total_seconds": round(total_time, 2),
            "crawl_seconds": round(t1_end - t1, 2),
            "fingerprint_seconds": round(t2_end - t2, 2),
            "graph_seconds": round(t3_end - t3, 2),
        },
        "pipeline_mode": "python_pipeline",
        "live_api": live_api,
    }

    logger.info(
        f"=== Pipeline complete for {artist_name}: "
        f"{verdict['verdict']} ({verdict['overall_score']:.3f}) "
        f"in {total_time:.2f}s ==="
    )
    return result


def run_batch_analysis(
    artist_ids: list[str],
    run_cross_platform: bool = False,
) -> list[dict[str, Any]]:
    """
    Run the full pipeline for multiple artists.
    Returns list of result dicts in the same order as artist_ids.
    """
    results = []
    for i, aid in enumerate(artist_ids, 1):
        logger.info(f"Batch analysis {i}/{len(artist_ids)}: {aid}")
        result = run_analysis(aid, run_cross_platform=run_cross_platform)
        results.append(result)
    return results


# ── CrewAI Crew Mode (LLM-backed) ─────────────────────────────────────────────

def run_analysis_with_crew(
    artist_id: str,
    artist_name: str | None = None,
    track_name: str | None = None,
) -> dict[str, Any]:
    """
    Run analysis using the full CrewAI crew with LLM-backed reasoning.
    Requires OPENAI_API_KEY in environment.

    Falls back to pipeline mode if LLM is unavailable.
    """
    try:
        from crewai import Agent, Task, Crew, Process
        from src.agents.crawler_agent import build_crawler_agent
        from src.agents.fingerprint_analyst import build_fingerprint_agent
        from src.agents.graph_builder import build_graph_agent
        from src.agents.verdict_agent import build_verdict_agent

        # First run the pipeline to get numerical scores
        pipeline_result = run_analysis(
            artist_id, artist_name=artist_name, track_name=track_name
        )
        signal_scores = pipeline_result.get("signal_scores", {})
        display_name = pipeline_result.get("artist_name", artist_id)

        # Build agents
        crawler = build_crawler_agent()
        fingerprint = build_fingerprint_agent()
        graph = build_graph_agent()
        verdict = build_verdict_agent()

        # Build tasks
        task_crawl = Task(
            description=(
                f"Retrieve cached metadata for artist ID: {artist_id}. "
                f"Check artist name, album count, track count from cache. "
                f"Report what data is available."
            ),
            expected_output="Summary of cached data: artist name, album count, track count",
            agent=crawler,
        )

        task_fingerprint = Task(
            description=(
                f"Analyze audio fingerprint and metadata for artist {display_name} ({artist_id}). "
                f"Pre-computed signal scores: {signal_scores}. "
                f"Interpret what the S1 (audio), S4 (catalog density), and S5 (metadata) scores mean."
            ),
            expected_output="Written analysis of audio fingerprint and metadata patterns",
            agent=fingerprint,
            context=[task_crawl],
        )

        task_graph = Task(
            description=(
                f"Analyze network graph signals for artist {display_name} ({artist_id}). "
                f"Pre-computed signal scores: {signal_scores}. "
                f"Interpret S2 (cadence), S3 (co-occurrence), S6 (ISRC HHI) scores."
            ),
            expected_output="Written analysis of network graph patterns and cadence anomalies",
            agent=graph,
            context=[task_crawl],
        )

        task_verdict = Task(
            description=(
                f"Synthesize all evidence for {display_name} ({artist_id}) into a final verdict. "
                f"Signal scores: {signal_scores}. "
                f"Overall score: {pipeline_result.get('overall_score', 0):.3f}. "
                f"Provide a clear, justified assessment of whether this is a ghost artist."
            ),
            expected_output=(
                "Final verdict: LIKELY_GHOST, SUSPICIOUS, or LIKELY_ORGANIC with justification, "
                "confidence level, and actionable recommendation"
            ),
            agent=verdict,
            context=[task_fingerprint, task_graph],
        )

        crew = Crew(
            agents=[crawler, fingerprint, graph, verdict],
            tasks=[task_crawl, task_fingerprint, task_graph, task_verdict],
            process=Process.sequential,
            verbose=True,
        )

        crew_output = crew.kickoff()
        crew_narrative = str(crew_output)

        # Merge pipeline numerical results with LLM narrative
        pipeline_result["crew_narrative"] = crew_narrative
        pipeline_result["pipeline_mode"] = "crewai_llm"
        return pipeline_result

    except Exception as e:
        logger.error(f"CrewAI mode failed ({e}), falling back to pipeline mode")
        result = run_analysis(artist_id, artist_name=artist_name, track_name=track_name)
        result["crew_error"] = str(e)
        result["pipeline_mode"] = "python_pipeline_fallback"
        return result


# ── helpers ───────────────────────────────────────────────────────────────────

def _safe_run(stage: str, fn, *args, **kwargs) -> dict:
    """Run a pipeline stage with error isolation."""
    try:
        return fn(*args, **kwargs) or {}
    except Exception as e:
        logger.error(f"Stage '{stage}' failed: {e}")
        return {"_error": str(e)}
