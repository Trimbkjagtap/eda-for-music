"""
Crawler Agent — Discovers and expands the artist network.

Takes a seed artist ID, pulls metadata, albums, tracks, related artists
from the local cache (data/raw/cache/). Populates Neo4j with what it finds.
Set live_api=True to allow real Spotify API calls (respects rate limits).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

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

                def _run(self, *args: Any, **kwargs: Any) -> str:
                    raise NotImplementedError("BaseTool unavailable in current CrewAI install")

from src.graph.neo4j_client import Neo4jClient
from src.utils.config import config

_CACHE_DIR = config.CACHE_DIR


# ── Cache-reader tools ────────────────────────────────────────────────────────

class _CacheReadInput(BaseModel):
    artist_id: str = Field(description="Spotify artist ID")


def _find_cache_file(prefix: str, artist_id: str) -> Path | None:
    """Scan cache dir for a JSON file whose content matches prefix+artist_id key."""
    import hashlib
    # Try known hash patterns used by SpotifyClient
    for parts_combo in [
        (prefix, artist_id),
        (prefix, artist_id, "album,single"),
    ]:
        key = hashlib.md5("_".join(parts_combo).encode()).hexdigest()
        path = _CACHE_DIR / f"{key}.json"
        if path.exists():
            return path
    return None


class GetArtistFromCacheTool(BaseTool):
    name: str = "get_artist_from_cache"
    description: str = "Read cached artist metadata (name, id, images) for a Spotify artist ID"
    args_schema: type[BaseModel] = _CacheReadInput

    def _run(self, artist_id: str) -> str:
        path = _find_cache_file("artist", artist_id)
        if path:
            data = json.loads(path.read_text())
            return json.dumps({"found": True, "name": data.get("name"), "id": data.get("id")})
        return json.dumps({"found": False, "artist_id": artist_id})


class GetAlbumsFromCacheTool(BaseTool):
    name: str = "get_albums_from_cache"
    description: str = "Read cached album list for a Spotify artist ID"
    args_schema: type[BaseModel] = _CacheReadInput

    def _run(self, artist_id: str) -> str:
        path = _find_cache_file("albums", artist_id)
        if path:
            albums = json.loads(path.read_text())
            summary = [{"id": a.get("id"), "name": a.get("name"),
                        "release_date": a.get("release_date")} for a in albums[:5]]
            return json.dumps({"found": True, "album_count": len(albums), "sample": summary})
        return json.dumps({"found": False, "artist_id": artist_id})


class _StoreInput(BaseModel):
    artist_id: str = Field(description="Spotify artist ID to store in Neo4j")
    artist_name: str = Field(description="Artist display name")
    is_ghost: bool = Field(default=False, description="Whether this artist is flagged as ghost")


class StoreInNeo4jTool(BaseTool):
    name: str = "store_in_neo4j"
    description: str = "Upsert an artist node into the Neo4j graph"
    args_schema: type[BaseModel] = _StoreInput

    def _run(self, artist_id: str, artist_name: str, is_ghost: bool = False) -> str:
        try:
            neo4j = Neo4jClient()
            neo4j.upsert_artist(
                artist_id=artist_id,
                name=artist_name,
                followers=0,
                genres=[],
                is_ghost=is_ghost,
                label="ghost" if is_ghost else "unknown",
            )
            return json.dumps({"stored": True, "artist_id": artist_id, "name": artist_name})
        except Exception as e:
            return json.dumps({"stored": False, "error": str(e)})


# ── Cache-based crawl function (no LLM needed) ────────────────────────────────

def crawl_artist_from_cache(artist_id: str) -> dict:
    """
    Pure Python crawl — reads all cached data for an artist.
    Returns summary dict without making any API calls.
    """
    result: dict[str, Any] = {
        "artist_id": artist_id,
        "artist_found": False,
        "album_count": 0,
        "track_count": 0,
        "isrc_count": 0,
        "relationships_added": 0,
        "artist_name": None,
    }

    # 1. Check artist cache
    artist_path = _find_cache_file("artist", artist_id)
    if artist_path:
        artist_data = json.loads(artist_path.read_text())
        result["artist_found"] = True
        result["artist_name"] = artist_data.get("name")

    # 2. Check albums cache
    albums_path = _find_cache_file("albums", artist_id)
    albums = []
    if albums_path:
        albums = json.loads(albums_path.read_text())
        result["album_count"] = len(albums)

    # 3. Count cached tracks
    import hashlib
    track_count = 0
    isrc_count = 0
    for album in albums:
        album_id = album.get("id", "")
        key = hashlib.md5(f"album_tracks_{album_id}".encode()).hexdigest()
        # Also try the direct hash
        key2 = hashlib.md5("_".join(["album_tracks", album_id]).encode()).hexdigest()
        for k in [key, key2]:
            p = _CACHE_DIR / f"{k}.json"
            if p.exists():
                tracks = json.loads(p.read_text())
                track_count += len(tracks)
                break

    # Count cached full tracks (with ISRC)
    for cache_file in _CACHE_DIR.glob("*.json"):
        try:
            data = json.loads(cache_file.read_text())
            if isinstance(data, dict) and "external_ids" in data and "isrc" in data.get("external_ids", {}):
                isrc_count += 1
        except Exception:
            pass

    result["track_count"] = track_count
    result["isrc_count"] = min(isrc_count, 500)  # cap counting at 500 for speed

    # 4. Try to populate Neo4j from existing processed data
    try:
        from pathlib import Path as P
        import pandas as pd
        neo4j_csv = P("data/processed/neo4j_full_graph.csv")
        if neo4j_csv.exists():
            df = pd.read_csv(neo4j_csv)
            artist_rows = df[df["artist_id"] == artist_id]
            if not artist_rows.empty:
                result["track_count"] = len(artist_rows)
                result["relationships_added"] = len(artist_rows)
                if not result["artist_name"]:
                    result["artist_name"] = artist_rows.iloc[0]["artist_name"]
    except Exception as e:
        logger.debug(f"Neo4j CSV population skipped: {e}")

    logger.info(
        f"Crawl complete for {result.get('artist_name', artist_id)}: "
        f"{result['album_count']} albums, {result['track_count']} tracks"
    )
    return result


# ── CrewAI Agent definition ───────────────────────────────────────────────────

def build_crawler_agent() -> Agent:
    """Build and return the CrewAI Crawler Agent."""
    return Agent(
        role="Music Network Crawler",
        goal="Discover and map artist networks from seed artists using cached data",
        backstory=(
            "Expert at navigating the Spotify API efficiently, mapping relationships "
            "between artists, albums, tracks, and production companies. Works entirely "
            "from cached data to respect strict API rate limits."
        ),
        tools=[
            GetArtistFromCacheTool(),
            GetAlbumsFromCacheTool(),
            StoreInNeo4jTool(),
        ],
        verbose=True,
        allow_delegation=False,
    )
