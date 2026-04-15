"""
Signal 6: Related Artist Graph Density

Ghost artist networks form dense subgraphs in the related-artist graph.
Organic artists have sparser, more diffuse neighborhoods.

Input: artist_id or list of artist_ids
Output: density score 0.0 (sparse/diffuse) to 1.0 (dense clique)

April 2026 constraint:
- related-artists endpoint returns 403 Forbidden.
- Neo4j RELATED_TO edges: 0 Artist→Artist edges (the SIMILAR_TO edges in Neo4j
  connect Domain nodes from a different project — not music data).
- We therefore use the ISRC bipartite graph as a proxy:
    Artists sharing production companies are treated as "related".
    This is actually a STRONGER signal than Spotify's opaque related-artists algorithm.
- For single-artist scoring: density = HHI (from Exercise 4).
- For cluster scoring: density = how many artists in the cluster share the same
  production company subgraph.

Data source: Neo4j + exercise4_metrics.csv — no API calls.
"""
from __future__ import annotations

from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from loguru import logger

from src.graph.neo4j_client import Neo4jClient

_METRICS4_PATH = Path("data/processed/exercise4_metrics.csv")
_FULL4_PATH = Path("data/processed/exercise4_full_data.csv")


def score_artist(artist_id: str, neo4j: Neo4jClient | None = None) -> dict:
    """
    Compute graph density score for one artist using their ISRC bipartite neighborhood.

    High HHI (single dominant company) = dense, concentrated neighborhood = suspicious.

    Returns
    -------
    dict with keys:
        artist_id, artist_name, track_count, num_companies,
        hhi, dominant_share, isrc_entropy,
        bipartite_density, suspicion_level, suspicion_score
    """
    _neo4j = neo4j or Neo4jClient()

    # Get artist name
    name_rows = _neo4j.run(
        "MATCH (a:Artist {spotify_id: $id}) RETURN a.name AS name", id=artist_id
    )
    artist_name = name_rows[0]["name"] if name_rows else artist_id

    # Get ISRC bipartite data from Neo4j
    rows = _neo4j.run(
        """
        MATCH (a:Artist {spotify_id: $id})-[:RELEASED]->(al:Album)
              -[:CONTAINS]->(t:Track)-[:REGISTERED_WITH]->(c:ProductionCompany)
        RETURN c.isrc_prefix AS prefix, count(t) AS track_count
        ORDER BY track_count DESC
        """,
        id=artist_id,
    )

    if not rows:
        # Try pre-computed metrics
        if _METRICS4_PATH.exists():
            m4 = pd.read_csv(_METRICS4_PATH)
            row = m4[m4["Artist"] == artist_name]
            if not row.empty:
                hhi = float(row.iloc[0]["HHI"])
                return _from_hhi(artist_id, artist_name, hhi)
        return _empty_result(artist_id, artist_name)

    track_counts = {r["prefix"]: r["track_count"] for r in rows}
    total = sum(track_counts.values())
    shares = [v / total for v in track_counts.values()]

    # HHI = sum of squared shares
    hhi = float(sum(s ** 2 for s in shares))
    dominant_share = max(shares)
    num_companies = len(shares)

    # ISRC entropy (bits) — low entropy = concentrated = suspicious
    import math
    isrc_entropy = -sum(s * math.log2(s) for s in shares if s > 0)

    # Build bipartite graph and compute density
    G = nx.Graph()
    G.add_node(f"artist_{artist_id}", bipartite=0)
    for prefix, count in track_counts.items():
        G.add_node(f"company_{prefix}", bipartite=1)
        G.add_edge(f"artist_{artist_id}", f"company_{prefix}", weight=count)

    # For a star graph (one artist, N companies), density = 1.0 when N=1 (one edge)
    bipartite_density = 1.0 / num_companies if num_companies > 0 else 0.0

    suspicion_score = _compute_suspicion(hhi, dominant_share, num_companies)
    suspicion_level = _level(suspicion_score)

    logger.info(
        f"{artist_name}: HHI={hhi:.3f}, dominant_share={dominant_share:.1%}, "
        f"num_companies={num_companies}, suspicion={suspicion_level} ({suspicion_score:.2f})"
    )

    return {
        "artist_id": artist_id,
        "artist_name": artist_name,
        "track_count": total,
        "num_companies": num_companies,
        "hhi": round(hhi, 4),
        "dominant_share": round(dominant_share, 4),
        "isrc_entropy_bits": round(isrc_entropy, 4),
        "bipartite_density": round(bipartite_density, 4),
        "company_shares": {k: round(v, 4) for k, v in zip(track_counts.keys(), shares)},
        "suspicion_level": suspicion_level,
        "suspicion_score": round(suspicion_score, 4),
    }


def score_cluster(artist_ids: list[str], neo4j: Neo4jClient | None = None) -> dict:
    """
    Score a cluster of artists.

    In addition to individual HHI scores, computes a subgraph metric:
    how interconnected are the artists through shared production companies?
    A dense ghost network will share the same 1-2 companies across all artists.
    """
    _neo4j = neo4j or Neo4jClient()

    # Get full bipartite graph for all artists
    rows = _neo4j.run(
        """
        MATCH (a:Artist)-[:RELEASED]->(al:Album)-[:CONTAINS]->(t:Track)
              -[:REGISTERED_WITH]->(c:ProductionCompany)
        WHERE a.spotify_id IN $ids
        RETURN a.spotify_id AS artist_id, a.name AS artist_name,
               c.isrc_prefix AS prefix, count(t) AS track_count
        """,
        ids=artist_ids,
    )

    if not rows:
        individual = [score_artist(aid, _neo4j) for aid in artist_ids]
        mean_score = float(np.mean([r["suspicion_score"] for r in individual]))
        return {
            "artist_ids": artist_ids,
            "individual_scores": individual,
            "cluster_suspicion_score": round(mean_score, 4),
            "cluster_suspicion_level": _level(mean_score),
            "shared_companies": [],
        }

    df = pd.DataFrame(rows)

    # Build multi-artist bipartite graph
    G = nx.Graph()
    for _, row in df.iterrows():
        a_node = f"artist_{row['artist_id']}"
        c_node = f"company_{row['prefix']}"
        G.add_node(a_node, bipartite=0, name=row["artist_name"])
        G.add_node(c_node, bipartite=1)
        G.add_edge(a_node, c_node, weight=row["track_count"])

    # Find shared production companies (connected to 2+ artists)
    company_nodes = [n for n in G.nodes() if n.startswith("company_")]
    shared_companies = []
    for cn in company_nodes:
        neighbors = list(G.neighbors(cn))
        artist_neighbors = [n for n in neighbors if n.startswith("artist_")]
        if len(artist_neighbors) > 1:
            shared_companies.append({
                "company": cn.replace("company_", ""),
                "artists": [G.nodes[a].get("name", a) for a in artist_neighbors],
                "artist_count": len(artist_neighbors),
            })

    # Cluster density = shared companies / total companies
    total_companies = len(company_nodes)
    shared_count = len(shared_companies)
    sharing_ratio = shared_count / total_companies if total_companies > 0 else 0.0

    individual = [score_artist(aid, _neo4j) for aid in artist_ids]
    mean_hhi = float(np.mean([r["hhi"] for r in individual]))
    cluster_score = float(np.clip(0.6 * mean_hhi + 0.4 * sharing_ratio, 0.0, 1.0))

    return {
        "artist_ids": artist_ids,
        "individual_scores": individual,
        "total_companies": total_companies,
        "shared_companies": shared_companies,
        "sharing_ratio": round(sharing_ratio, 4),
        "mean_hhi": round(mean_hhi, 4),
        "cluster_suspicion_score": round(cluster_score, 4),
        "cluster_suspicion_level": _level(cluster_score),
    }


# ──────────────────────────── helpers ────────────────────────────


def _from_hhi(artist_id: str, artist_name: str, hhi: float) -> dict:
    suspicion_score = _compute_suspicion(hhi, hhi ** 0.5, max(1, int(1 / hhi)))
    return {
        "artist_id": artist_id,
        "artist_name": artist_name,
        "hhi": round(hhi, 4),
        "dominant_share": round(hhi ** 0.5, 4),
        "suspicion_level": _level(suspicion_score),
        "suspicion_score": round(suspicion_score, 4),
        "source": "exercise4_metrics.csv",
    }


def _compute_suspicion(hhi: float, dominant_share: float, num_companies: int) -> float:
    """
    Map ISRC concentration metrics → suspicion score [0, 1].
    HHI=1.0 (one company controls everything) = most suspicious.
    """
    # HHI contributes most (0-1 already)
    score = 0.6 * hhi
    # High dominant share is also suspicious
    score += 0.3 * dominant_share
    # Low num_companies (monopoly) is suspicious
    monopoly_signal = max(0.0, 1.0 - (num_companies - 1) * 0.25)
    score += 0.1 * monopoly_signal
    return float(np.clip(score, 0.0, 1.0))


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
        "num_companies": 0,
        "hhi": 0.0,
        "dominant_share": 0.0,
        "isrc_entropy_bits": 0.0,
        "bipartite_density": 0.0,
        "company_shares": {},
        "suspicion_level": "UNKNOWN",
        "suspicion_score": 0.0,
    }
