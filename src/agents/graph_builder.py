"""
Graph Builder Agent — Constructs and analyzes the relationship graph.

Builds the artist network in Neo4j, runs community detection,
computes graph metrics. Runs Signals 2 (cadence), 3 (co-occurrence), 6 (graph density).
"""
from __future__ import annotations

import json

import networkx as nx
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

class BuildNeighborhoodGraphTool(BaseTool):
    name: str = "build_neighborhood_graph"
    description: str = (
        "Build the ISRC-based bipartite neighborhood graph for an artist from Neo4j. "
        "Returns nodes (artist + production companies) and edges (track registrations)."
    )
    args_schema: type[BaseModel] = _SingleArtistInput

    def _run(self, artist_id: str) -> str:
        from src.graph.neo4j_client import Neo4jClient
        try:
            neo4j = Neo4jClient()
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
                "MATCH (a:Artist {spotify_id: $id}) RETURN a.name AS name", id=artist_id
            )
            artist_name = name_row[0]["name"] if name_row else artist_id

            nodes = [{"id": artist_id, "type": "artist", "name": artist_name}]
            edges = []
            for r in rows:
                nodes.append({"id": r["prefix"], "type": "company", "name": r["company"]})
                edges.append({"source": artist_id, "target": r["prefix"],
                               "weight": r["track_count"]})
            return json.dumps({"nodes": nodes, "edges": edges, "company_count": len(rows)})
        except Exception as e:
            return json.dumps({"error": str(e)})


class RunCommunityDetectionTool(BaseTool):
    name: str = "run_community_detection"
    description: str = (
        "Run Louvain community detection on the artist network stored in Neo4j. "
        "Returns community assignments for each artist."
    )
    args_schema: type[BaseModel] = _MultiArtistInput

    def _run(self, artist_ids: list[str]) -> str:
        from src.graph.neo4j_client import Neo4jClient
        try:
            neo4j = Neo4jClient()
            # Build a graph based on shared production companies (bipartite projection)
            all_company_data = {}
            for aid in artist_ids:
                rows = neo4j.run(
                    """
                    MATCH (a:Artist {spotify_id: $id})-[:RELEASED]->(al:Album)
                          -[:CONTAINS]->(t:Track)-[:REGISTERED_WITH]->(c:ProductionCompany)
                    RETURN collect(DISTINCT c.isrc_prefix) AS prefixes
                    """,
                    id=aid,
                )
                all_company_data[aid] = set(rows[0]["prefixes"]) if rows else set()

            # Build artist-artist graph via shared companies
            G = nx.Graph()
            for aid in artist_ids:
                G.add_node(aid)
            for i, a in enumerate(artist_ids):
                for b in artist_ids[i+1:]:
                    shared = all_company_data[a] & all_company_data[b]
                    if shared:
                        G.add_edge(a, b, weight=len(shared))

            # Try Louvain community detection
            try:
                import community as community_louvain
                partition = community_louvain.best_partition(G)
            except ImportError:
                # Fallback: connected components as communities
                partition = {}
                for i, component in enumerate(nx.connected_components(G)):
                    for node in component:
                        partition[node] = i

            return json.dumps({
                "communities": partition,
                "num_communities": len(set(partition.values())),
                "graph_density": round(nx.density(G), 4),
            })
        except Exception as e:
            return json.dumps({"error": str(e)})


class ScorePlaylistCooccurrenceTool(BaseTool):
    name: str = "score_playlist_cooccurrence"
    description: str = (
        "Score playlist co-occurrence density for a list of artist IDs. "
        "Uses ISRC prefix overlap as proxy (playlist API restricted). "
        "Returns suspicion score 0-1."
    )
    args_schema: type[BaseModel] = _MultiArtistInput

    def _run(self, artist_ids: list[str]) -> str:
        from src.signals.playlist_cooccurrence import score_cluster
        try:
            result = score_cluster(artist_ids)
            return json.dumps({
                "signal": "playlist_cooccurrence",
                "score": result.get("suspicion_score"),
                "company_jaccard": result.get("pairwise_company_jaccard"),
                "isrc_sharing": result.get("isrc_sharing_score"),
                "shared_companies": result.get("shared_companies"),
                "suspicion_level": result.get("suspicion_level"),
            })
        except Exception as e:
            return json.dumps({"error": str(e), "signal": "playlist_cooccurrence", "score": None})


class ScoreGraphDensityTool(BaseTool):
    name: str = "score_graph_density"
    description: str = (
        "Score ISRC-based graph density (HHI) for an artist. "
        "HHI > 0.65 = single-company monopoly = ghost-like."
    )
    args_schema: type[BaseModel] = _SingleArtistInput

    def _run(self, artist_id: str) -> str:
        from src.signals.graph_density import score_artist
        try:
            result = score_artist(artist_id)
            return json.dumps({
                "signal": "graph_density",
                "score": result.get("suspicion_score"),
                "hhi": result.get("hhi"),
                "dominant_share": result.get("dominant_share"),
                "num_companies": result.get("num_companies"),
                "suspicion_level": result.get("suspicion_level"),
            })
        except Exception as e:
            return json.dumps({"error": str(e), "signal": "graph_density", "score": None})


class ScoreCadenceSyncTool(BaseTool):
    name: str = "score_cadence_sync"
    description: str = (
        "Score release cadence synchrony for an artist. "
        "High closure rate (% same-day releases) = bulk upload = ghost-like."
    )
    args_schema: type[BaseModel] = _SingleArtistInput

    def _run(self, artist_id: str) -> str:
        from src.signals.cadence_sync import score_artist
        try:
            result = score_artist(artist_id)
            return json.dumps({
                "signal": "cadence_sync",
                "score": result.get("suspicion_score"),
                "closure_rate": result.get("closure_rate"),
                "burst_ratio": result.get("burst_ratio"),
                "median_gap_days": result.get("median_gap_days"),
                "suspicion_level": result.get("suspicion_level"),
            })
        except Exception as e:
            return json.dumps({"error": str(e), "signal": "cadence_sync", "score": None})


# ── Pure Python graph analysis (no LLM needed) ───────────────────────────────

def analyze_graph(artist_id: str, all_artist_ids: list[str] | None = None) -> dict:
    """
    Run signals 2, 3, 6 for an artist. Returns dict of scores.
    """
    from src.signals import cadence_sync, playlist_cooccurrence, graph_density
    from src.graph.neo4j_client import Neo4jClient

    neo4j = Neo4jClient()

    if all_artist_ids is None:
        rows = neo4j.run("MATCH (a:Artist) RETURN a.spotify_id AS id")
        all_artist_ids = [r["id"] for r in rows]

    results: dict = {"artist_id": artist_id}

    # S2: Release cadence
    try:
        s2 = cadence_sync.score_artist(artist_id, neo4j)
        results["s2_cadence_sync"] = s2.get("suspicion_score")
        results["s2_detail"] = {
            "closure_rate": s2.get("closure_rate"),
            "burst_ratio": s2.get("burst_ratio"),
            "median_gap_days": s2.get("median_gap_days"),
            "suspicion_level": s2.get("suspicion_level"),
        }
    except Exception as e:
        logger.warning(f"S2 failed: {e}")
        results["s2_cadence_sync"] = None

    # S3: Playlist co-occurrence (cluster signal)
    try:
        s3 = playlist_cooccurrence.score_artist(artist_id, all_artist_ids, neo4j)
        results["s3_playlist_cooccurrence"] = s3.get("suspicion_score")
        results["s3_detail"] = {"suspicion_level": s3.get("suspicion_level")}
    except Exception as e:
        logger.warning(f"S3 failed: {e}")
        results["s3_playlist_cooccurrence"] = None

    # S6: Graph density (HHI)
    try:
        s6 = graph_density.score_artist(artist_id, neo4j)
        results["s6_graph_density"] = s6.get("suspicion_score")
        results["s6_detail"] = {
            "hhi": s6.get("hhi"),
            "dominant_share": s6.get("dominant_share"),
            "num_companies": s6.get("num_companies"),
            "suspicion_level": s6.get("suspicion_level"),
        }
    except Exception as e:
        logger.warning(f"S6 failed: {e}")
        results["s6_graph_density"] = None

    logger.info(
        f"Graph analysis for {artist_id}: "
        f"S2={results.get('s2_cadence_sync')}, "
        f"S3={results.get('s3_playlist_cooccurrence')}, "
        f"S6={results.get('s6_graph_density')}"
    )
    return results


# ── CrewAI Agent definition ───────────────────────────────────────────────────

def build_graph_agent() -> Agent:
    return Agent(
        role="Network Graph Analyst",
        goal="Build and analyze the artist relationship graph to identify coordinated clusters",
        backstory=(
            "Expert in network science and graph theory, specializing in detecting "
            "coordinated inauthentic behavior in social networks. Uses ISRC bipartite "
            "graphs as a proxy for co-occurrence when playlist APIs are restricted."
        ),
        tools=[
            BuildNeighborhoodGraphTool(),
            RunCommunityDetectionTool(),
            ScorePlaylistCooccurrenceTool(),
            ScoreGraphDensityTool(),
            ScoreCadenceSyncTool(),
        ],
        verbose=True,
        allow_delegation=False,
    )
