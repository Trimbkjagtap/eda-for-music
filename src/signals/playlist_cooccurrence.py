"""
Signal 3: Playlist Co-occurrence Density

Ghost artists appear together on the same playlists far more often
than chance would predict.

Input: list of artist IDs
Output: co-occurrence score 0.0 (never together) to 1.0 (always together)

Methodology:
- Since playlist items API is restricted (editorial=404, user playlists inaccessible),
  we use ISRC prefix overlap as a proxy for co-occurrence:
  Artists sharing ISRC prefixes were registered by the same production company,
  meaning they're almost certainly coordinated (co-placed in the same playlists by
  their operator).
- Also compute Jaccard similarity of production company sets between artist pairs.
- Score = 0.7 * isrc_sharing_score + 0.3 * company_jaccard
- NO API calls — uses Neo4j data only.

Data available:
  - 3 Artist nodes in Neo4j
  - 8 ProductionCompany nodes
  - 490 Track nodes with REGISTERED_WITH edges
  - ISRC prefix data from exercise4_metrics.csv
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from src.graph.neo4j_client import Neo4jClient

# Pre-computed ISRC data from Exercise 4
_EXERCISE4_PATH = Path("data/processed/exercise4_full_data.csv")
_METRICS4_PATH = Path("data/processed/exercise4_metrics.csv")


def score_cluster(artist_ids: list[str], neo4j: Neo4jClient | None = None) -> dict:
    """
    Compute playlist co-occurrence score for a cluster of artists.

    Parameters
    ----------
    artist_ids  : list of Spotify artist IDs
    neo4j       : Neo4jClient instance (created if None)

    Returns
    -------
    dict with keys:
        artist_ids, pairwise_company_jaccard, isrc_sharing_score,
        shared_companies (list of {company, artists}),
        suspicion_level, suspicion_score
    """
    _neo4j = neo4j or Neo4jClient()

    # Get production company sets per artist from Neo4j
    company_sets: dict[str, set[str]] = {}
    artist_names: dict[str, str] = {}

    for aid in artist_ids:
        rows = _neo4j.run(
            """
            MATCH (a:Artist {spotify_id: $id})-[:RELEASED]->(al:Album)
                  -[:CONTAINS]->(t:Track)-[:REGISTERED_WITH]->(c:ProductionCompany)
            RETURN a.name AS name, collect(DISTINCT c.isrc_prefix) AS prefixes
            """,
            id=aid,
        )
        if rows:
            company_sets[aid] = set(rows[0]["prefixes"])
            artist_names[aid] = rows[0]["name"]
        else:
            company_sets[aid] = set()
            artist_names[aid] = aid

    # Pairwise Jaccard on production company sets
    ids = list(artist_ids)
    pairwise_jaccards = []
    shared_companies = []

    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            a_set = company_sets[ids[i]]
            b_set = company_sets[ids[j]]
            union = len(a_set | b_set)
            inter = len(a_set & b_set)
            jaccard = inter / union if union > 0 else 0.0
            pairwise_jaccards.append(jaccard)
            if inter > 0:
                shared_companies.append({
                    "artist_a": artist_names[ids[i]],
                    "artist_b": artist_names[ids[j]],
                    "shared_prefixes": list(a_set & b_set),
                    "jaccard": round(jaccard, 4),
                })

    mean_jaccard = float(np.mean(pairwise_jaccards)) if pairwise_jaccards else 0.0

    # ISRC sharing score: what fraction of artist pairs share at least one prefix?
    sharing_pairs = sum(1 for j in pairwise_jaccards if j > 0)
    total_pairs = len(pairwise_jaccards)
    isrc_sharing_score = sharing_pairs / total_pairs if total_pairs > 0 else 0.0

    # Cluster HHI as additional signal (from Exercise 4 metrics)
    hhi_bonus = 0.0
    if _METRICS4_PATH.exists():
        m4 = pd.read_csv(_METRICS4_PATH)
        hhis = []
        for aid in artist_ids:
            name_rows = _neo4j.run(
                "MATCH (a:Artist {spotify_id: $id}) RETURN a.name AS name", id=aid
            )
            if name_rows:
                name = name_rows[0]["name"]
                row = m4[m4["Artist"] == name]
                if not row.empty:
                    hhis.append(float(row.iloc[0]["HHI"]))
        if hhis:
            hhi_bonus = float(np.mean(hhis))  # already 0-1

    # Combined score
    suspicion_score = 0.5 * isrc_sharing_score + 0.3 * mean_jaccard + 0.2 * hhi_bonus
    suspicion_score = float(np.clip(suspicion_score, 0.0, 1.0))
    suspicion_level = _level(suspicion_score)

    logger.info(
        f"Cluster ({len(artist_ids)} artists): company_jaccard={mean_jaccard:.3f}, "
        f"isrc_sharing={isrc_sharing_score:.2f}, hhi_bonus={hhi_bonus:.3f}, "
        f"suspicion={suspicion_level} ({suspicion_score:.2f})"
    )

    return {
        "artist_ids": artist_ids,
        "artist_names": artist_names,
        "pairwise_company_jaccard": round(mean_jaccard, 4),
        "isrc_sharing_score": round(isrc_sharing_score, 4),
        "mean_hhi": round(hhi_bonus, 4),
        "shared_companies": shared_companies,
        "suspicion_level": suspicion_level,
        "suspicion_score": round(suspicion_score, 4),
    }


def score_artist(artist_id: str, all_artist_ids: list[str] | None = None,
                 neo4j: Neo4jClient | None = None) -> dict:
    """
    Score a single artist for production-company concentration (ISRC proxy for playlist co-occurrence).

    Ghost artists each use private/unknown registrants (ISRC sharing between artists is 0),
    so the dominant signal is per-artist HHI (monopolistic production = suspicious).
    Cross-artist ISRC sharing is kept as a secondary bonus when it fires.
    """
    _neo4j = neo4j or Neo4jClient()

    # Resolve name
    name_rows = _neo4j.run(
        "MATCH (a:Artist {spotify_id: $id}) RETURN a.name AS name", id=artist_id
    )
    artist_name = name_rows[0]["name"] if name_rows else artist_id

    # Per-artist HHI from pre-computed metrics (most reliable signal)
    hhi = 0.0
    if _METRICS4_PATH.exists():
        m4 = pd.read_csv(_METRICS4_PATH)
        row = m4[m4["Artist"] == artist_name]
        if not row.empty:
            hhi = float(row.iloc[0]["HHI"])

    # If no pre-computed HHI, compute from Neo4j ISRC data
    if hhi == 0.0:
        rows = _neo4j.run(
            """
            MATCH (a:Artist {spotify_id: $id})-[:RELEASED]->(al:Album)
                  -[:CONTAINS]->(t:Track)-[:REGISTERED_WITH]->(c:ProductionCompany)
            RETURN c.isrc_prefix AS prefix, count(t) AS cnt
            ORDER BY cnt DESC
            """,
            id=artist_id,
        )
        if rows:
            total = sum(r["cnt"] for r in rows)
            if total > 0:
                hhi = sum((r["cnt"] / total) ** 2 for r in rows)

    # Cross-artist ISRC sharing (bonus — fires rarely but is strong when it does)
    if all_artist_ids is None:
        id_rows = _neo4j.run("MATCH (a:Artist) RETURN a.spotify_id AS id")
        all_artist_ids = [r["id"] for r in id_rows]

    others = [a for a in all_artist_ids if a != artist_id]
    sharing_score = 0.0
    if others:
        my_prefixes_rows = _neo4j.run(
            """
            MATCH (a:Artist {spotify_id: $id})-[:RELEASED]->(al:Album)
                  -[:CONTAINS]->(t:Track)-[:REGISTERED_WITH]->(c:ProductionCompany)
            RETURN collect(DISTINCT c.isrc_prefix) AS prefixes
            """,
            id=artist_id,
        )
        my_prefixes = set(my_prefixes_rows[0]["prefixes"]) if my_prefixes_rows else set()
        shared_pairs = 0
        for oid in others[:20]:  # cap to avoid slow queries on large DBs
            o_rows = _neo4j.run(
                """
                MATCH (a:Artist {spotify_id: $id})-[:RELEASED]->(al:Album)
                      -[:CONTAINS]->(t:Track)-[:REGISTERED_WITH]->(c:ProductionCompany)
                RETURN collect(DISTINCT c.isrc_prefix) AS prefixes
                """,
                id=oid,
            )
            if o_rows:
                other_prefixes = set(o_rows[0]["prefixes"])
                if my_prefixes & other_prefixes:
                    shared_pairs += 1
        sharing_score = shared_pairs / min(len(others), 20)

    # HHI is the dominant per-artist signal; cross-artist sharing is a bonus
    suspicion_score = float(np.clip(0.75 * hhi + 0.25 * sharing_score, 0.0, 1.0))
    suspicion_level = _level(suspicion_score)

    logger.info(
        f"{artist_name}: hhi={hhi:.3f}, sharing={sharing_score:.3f}, "
        f"suspicion={suspicion_level} ({suspicion_score:.2f})"
    )

    return {
        "artist_id": artist_id,
        "artist_name": artist_name,
        "hhi": round(hhi, 4),
        "sharing_score": round(sharing_score, 4),
        "suspicion_score": round(suspicion_score, 4),
        "suspicion_level": suspicion_level,
    }


def _level(score: float) -> str:
    if score >= 0.6:
        return "HIGH"
    elif score >= 0.3:
        return "MEDIUM"
    return "LOW"
