"""
Seed Ingest — Day 2
====================
Populates Neo4j AuraDB with all seed artists (ghost + organic),
their album catalogs, and ISRC-linked production companies.

Run:
    python -m src.ingest.seed_ingest                  # full ingest
    python -m src.ingest.seed_ingest --dry-run        # no DB writes
    python -m src.ingest.seed_ingest --max-albums 5   # cap albums per artist

Rate-limit safety:
- Rate limiter is set to 2 calls/second (config.py)
- ISRC is fetched for the FIRST track of each album only (1 call per album)
  This is enough to identify the production company without burning quota.
- Albums are capped at --max-albums (default 20) per artist
- A 0.5s pause is added between albums

What it does:
1. Resolves spotify_id for any seed missing one (ghost artists have None)
2. Upserts Artist nodes with is_ghost label
3. Fetches albums (capped) → upserts Album nodes + Artist-RELEASED->Album edges
4. Fetches tracks per album → upserts Track nodes + Album-CONTAINS->Track edges
5. Fetches ISRC for first track per album only → links ProductionCompany
6. Writes progress to data/processed/seed_catalog.csv
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

from loguru import logger

from src.api.spotify_client import SpotifyClient
from src.graph.neo4j_client import Neo4jClient
from src.utils.config import config
from src.utils.ground_truth import get_all_seeds, KNOWN_ISRC_PREFIXES


def run(dry_run: bool = False, max_albums: int = 20) -> dict:
    """
    Full seed ingest pipeline.

    Parameters
    ----------
    dry_run     If True, fetch data but skip Neo4j writes.
    max_albums  Max albums to process per artist (default 20).
                Set to a small number (3-5) for a quick test run.

    Returns
    -------
    dict with summary stats:
        artists_processed, albums_ingested, tracks_ingested,
        isrc_found, production_companies_linked, errors
    """
    spotify = SpotifyClient()
    neo4j = Neo4jClient() if not dry_run else None

    if not spotify.test_connection():
        logger.error("Spotify connection failed — aborting ingest.")
        sys.exit(1)

    if neo4j and not neo4j.test_connection():
        logger.error("Neo4j connection failed — aborting ingest.")
        sys.exit(1)

    seeds = get_all_seeds()
    logger.info(
        f"Starting seed ingest: {len(seeds)} artists "
        f"({sum(1 for s in seeds if s['is_ghost'])} ghost, "
        f"{sum(1 for s in seeds if not s['is_ghost'])} organic), "
        f"max_albums={max_albums}"
    )

    stats = {
        "artists_processed": 0,
        "albums_ingested": 0,
        "tracks_ingested": 0,
        "isrc_found": 0,
        "production_companies_linked": 0,
        "errors": [],
    }

    all_rows: list[dict] = []

    for seed in seeds:
        name = seed["name"]
        is_ghost = seed["is_ghost"]
        sid = seed.get("spotify_id")

        # --- Step 1: Resolve spotify_id if missing ---
        if not sid:
            logger.info(f"Searching Spotify for: {name}")
            sid = spotify.artist_id_from_name(name)
            if not sid:
                logger.warning(f"Could not find '{name}' on Spotify — skipping")
                stats["errors"].append(f"Not found on Spotify: {name}")
                continue
            logger.info(f"  Found: {name} → {sid}")

        label = "ghost" if is_ghost else "organic"

        try:
            # --- Step 2: Upsert Artist node ---
            artist_meta = spotify.get_artist(sid)
            artist_name = artist_meta.get("name", name)

            if neo4j:
                neo4j.upsert_artist(
                    artist_id=sid,
                    name=artist_name,
                    followers=0,      # not available April 2026
                    genres=[],        # not available April 2026
                    is_ghost=is_ghost,
                    label=label,
                )

            logger.info(f"Processing [{label}] {artist_name} ({sid})")

            # --- Step 3: Fetch albums (capped) ---
            all_albums = spotify.get_artist_albums(sid, include_groups="album,single")
            albums = all_albums[:max_albums]
            logger.info(f"  {len(all_albums)} albums/singles total, processing {len(albums)}")

            for album in albums:
                album_id = album["id"]
                album_name = album["name"]
                release_date = album.get("release_date", "")

                if neo4j:
                    neo4j.upsert_album(
                        album_id=album_id,
                        name=album_name,
                        release_date=release_date,
                        isrc_prefix="",
                    )
                    neo4j.link_artist_album(sid, album_id)

                stats["albums_ingested"] += 1

                # --- Step 4: Fetch tracks per album ---
                tracks = spotify.get_album_tracks(album_id)

                # --- Step 5: ISRC — fetch full track for FIRST track only ---
                # One ISRC per album is enough to identify the production company.
                # Fetching every track would burn API quota for large catalogs.
                isrc_for_album = None
                if tracks:
                    try:
                        full_track = spotify.get_track(tracks[0]["id"])
                        isrc_for_album = full_track.get("external_ids", {}).get("isrc")
                        if isrc_for_album:
                            stats["isrc_found"] += 1
                            prefix = _extract_isrc_prefix(isrc_for_album)
                            if prefix:
                                company_name = KNOWN_ISRC_PREFIXES.get(
                                    prefix, f"Unknown ({prefix})"
                                )
                                if neo4j:
                                    neo4j.upsert_production_company(prefix, company_name)
                                stats["production_companies_linked"] += 1
                    except Exception as e:
                        logger.debug(f"    ISRC fetch failed for album {album_id}: {e}")

                for track in tracks:
                    track_id = track["id"]
                    track_name = track["name"]
                    # Use album-level ISRC for all tracks (same production company)
                    isrc = isrc_for_album if track["id"] == tracks[0]["id"] else None

                    if neo4j:
                        neo4j.upsert_track(
                            track_id=track_id,
                            name=track_name,
                            isrc=isrc or "",
                            duration_ms=track.get("duration_ms", 0),
                            release_date=release_date,
                        )
                        neo4j.link_album_track(album_id, track_id)
                        if isrc:
                            prefix = _extract_isrc_prefix(isrc)
                            if prefix:
                                neo4j.link_track_company(track_id, prefix)

                    stats["tracks_ingested"] += 1
                    all_rows.append({
                        "artist_id": sid,
                        "artist_name": artist_name,
                        "is_ghost": is_ghost,
                        "album_id": album_id,
                        "album_name": album_name,
                        "release_date": release_date,
                        "track_id": track_id,
                        "track_name": track_name,
                        "isrc": isrc or "",
                        "isrc_prefix": _extract_isrc_prefix(isrc) if isrc else "",
                    })

                time.sleep(0.5)  # 0.5s pause between albums — keeps well under rate limit

            stats["artists_processed"] += 1
            logger.info(
                f"  Done: {len(albums)} albums, "
                f"{stats['tracks_ingested']} total tracks so far"
            )

        except Exception as e:
            logger.error(f"Error processing {name} ({sid}): {e}")
            stats["errors"].append(f"{name}: {e}")

    # --- Step 6: Save catalog CSV ---
    _save_catalog_csv(all_rows)

    # --- Summary ---
    logger.info("=" * 50)
    logger.info("Seed ingest complete:")
    logger.info(f"  Artists processed:         {stats['artists_processed']}")
    logger.info(f"  Albums ingested:            {stats['albums_ingested']}")
    logger.info(f"  Tracks ingested:            {stats['tracks_ingested']}")
    logger.info(f"  ISRC codes found:           {stats['isrc_found']}")
    logger.info(f"  Production company links:   {stats['production_companies_linked']}")
    if stats["errors"]:
        logger.warning(f"  Errors ({len(stats['errors'])}):")
        for e in stats["errors"]:
            logger.warning(f"    - {e}")

    if neo4j:
        counts = neo4j.count_nodes()
        logger.info(f"  Neo4j node counts: {counts}")
        neo4j.close()

    return stats


def _extract_isrc_prefix(isrc: str | None) -> str:
    """
    Extract the registrant prefix from an ISRC code.
    ISRC format: CC-XXX-YY-NNNNN
      CC  = country code (2 chars)
      XXX = registrant code (3 chars)  ← this is what we want + CC
    We use the first 5 chars (country + registrant) as the prefix key.
    """
    if not isrc:
        return ""
    # Strip dashes for normalization
    clean = isrc.replace("-", "").upper()
    if len(clean) >= 5:
        return clean[:5]
    return clean


def _save_catalog_csv(rows: list[dict]) -> None:
    """Write all ingested tracks to data/processed/seed_catalog.csv."""
    if not rows:
        return
    output_path = config.PROCESSED_DIR / "seed_catalog.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "artist_id", "artist_name", "is_ghost",
        "album_id", "album_name", "release_date",
        "track_id", "track_name", "isrc", "isrc_prefix",
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    logger.info(f"Catalog saved to {output_path} ({len(rows)} rows)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Ingest seed artists into Neo4j")
    parser.add_argument("--dry-run", action="store_true",
                        help="Fetch data without writing to Neo4j")
    parser.add_argument("--max-albums", type=int, default=20,
                        help="Max albums per artist (default 20). Use 3-5 for a quick test.")
    args = parser.parse_args()
    run(dry_run=args.dry_run, max_albums=args.max_albums)
