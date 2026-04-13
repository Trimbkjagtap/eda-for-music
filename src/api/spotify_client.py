"""
Spotify API client — wraps Spotipy with caching, rate limiting,
and accurate workarounds for April 2026 API state.

WHAT WORKS (confirmed April 2026):
- GET /artists/{id}           => name, id, images, external_urls (NO followers, NO genres)
- GET /artists/{id}/albums    => album list with release_date, name, id
- GET /albums/{id}/tracks     => track list (no external_ids in album track objects)
- GET /tracks/{id}            => full track with external_ids.isrc  <-- ISRC confirmed working
- GET /search                 => minimal artist/track/playlist objects (no followers/genres)

WHAT IS BLOCKED (403/404 for new apps):
- GET /artists/{id}/related-artists  => 403 Forbidden
- GET /playlists/{id}                => 404 for editorial playlists
- Audio features endpoint            => 403
- Popularity field                   => stripped from all responses
- Followers / genres                 => stripped from all responses

STRATEGY:
- ISRC: use GET /tracks/{id} (external_ids.isrc) — confirmed working
- Release dates: use album release_date — confirmed working
- Graph expansion: use search-based artist discovery (no related-artists)
- Audio features: Kaggle dataset join on track ID
- Followers/genres: use Kaggle dataset or skip; not available via API
"""
import json
import time
import hashlib
from pathlib import Path
from typing import Optional

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from loguru import logger

from src.utils.config import config
from src.utils.rate_limiter import default_limiter, with_retry


class SpotifyClient:
    def __init__(self):
        self._sp: Optional[spotipy.Spotify] = None
        self.cache_dir = config.CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @property
    def sp(self) -> spotipy.Spotify:
        if self._sp is None:
            auth = SpotifyClientCredentials(
                client_id=config.SPOTIFY_CLIENT_ID,
                client_secret=config.SPOTIFY_CLIENT_SECRET,
            )
            self._sp = spotipy.Spotify(auth_manager=auth, requests_timeout=15)
            logger.info("Spotify client initialized")
        return self._sp

    # ------------------------------------------------------------------ cache

    def _cache_key(self, *parts: str) -> Path:
        key = hashlib.md5("_".join(str(p) for p in parts).encode()).hexdigest()
        return self.cache_dir / f"{key}.json"

    def _from_cache(self, *parts) -> Optional[dict | list]:
        path = self._cache_key(*parts)
        if path.exists():
            return json.loads(path.read_text())
        return None

    def _to_cache(self, data, *parts):
        path = self._cache_key(*parts)
        path.write_text(json.dumps(data, indent=2))

    # --------------------------------------------------------------- artists

    @with_retry()
    def get_artist(self, artist_id: str) -> dict:
        """
        Get artist metadata.
        April 2026: returns only name, id, images, external_urls.
        followers and genres are stripped — not available via Client Credentials.
        """
        cached = self._from_cache("artist", artist_id)
        if cached:
            return cached
        default_limiter.wait()
        result = self.sp.artist(artist_id)
        self._to_cache(result, "artist", artist_id)
        return result

    @with_retry()
    def get_artist_albums(self, artist_id: str, include_groups: str = "album,single") -> list[dict]:
        """
        Get all albums for an artist (paginated).
        Returns: list of {id, name, release_date, total_tracks, album_type, artists}
        """
        cached = self._from_cache("albums", artist_id, include_groups)
        if cached:
            return cached

        albums = []
        default_limiter.wait()
        # April 2026: max limit is 10 for artist albums (was 50)
        page = self.sp.artist_albums(artist_id, include_groups=include_groups, limit=10)
        while page:
            albums.extend(page["items"])
            if page["next"]:
                default_limiter.wait()
                page = self.sp.next(page)
            else:
                break

        self._to_cache(albums, "albums", artist_id, include_groups)
        return albums

    def get_artist_full_catalog(self, artist_id: str) -> list[dict]:
        """
        Get all tracks for an artist with ISRC codes.
        Used in Exercise 1 (catalog coherence) and Exercise 3 (ISRC join).

        ISRC is fetched via GET /tracks/{id} (confirmed working April 2026).
        Album track objects do NOT have external_ids — must fetch full track.

        Returns list of dicts: {id, name, isrc, album_id, album_name, release_date, duration_ms}
        """
        albums = self.get_artist_albums(artist_id)
        tracks = []
        for album in albums:
            album_tracks = self.get_album_tracks(album["id"])
            for t in album_tracks:
                # Fetch full track to get ISRC (external_ids not in album track objects)
                isrc = None
                try:
                    full = self.get_track(t["id"])
                    isrc = full.get("external_ids", {}).get("isrc")
                except Exception:
                    pass
                tracks.append({
                    "id": t["id"],
                    "name": t["name"],
                    "isrc": isrc,
                    "album_id": album["id"],
                    "album_name": album["name"],
                    "release_date": album.get("release_date", ""),
                    "duration_ms": t.get("duration_ms", 0),
                })
            time.sleep(0.2)  # small pause between albums
        return tracks

    # --------------------------------------------------------------- albums

    @with_retry()
    def get_album_tracks(self, album_id: str) -> list[dict]:
        """
        Get all tracks in an album.
        NOTE: album track objects don't include external_ids.
        Use get_track() on each to get ISRC.
        """
        cached = self._from_cache("album_tracks", album_id)
        if cached:
            return cached

        tracks = []
        default_limiter.wait()
        page = self.sp.album_tracks(album_id, limit=50)
        while page:
            tracks.extend(page["items"])
            if page["next"]:
                default_limiter.wait()
                page = self.sp.next(page)
            else:
                break

        self._to_cache(tracks, "album_tracks", album_id)
        return tracks

    # --------------------------------------------------------------- tracks

    @with_retry()
    def get_track(self, track_id: str) -> dict:
        """
        Get full track metadata including external_ids (ISRC).
        This is the ONLY endpoint that reliably returns ISRC (April 2026).
        """
        cached = self._from_cache("track", track_id)
        if cached:
            return cached
        default_limiter.wait()
        result = self.sp.track(track_id)
        self._to_cache(result, "track", track_id)
        return result

    # --------------------------------------------------------------- search

    @with_retry()
    def search_tracks(self, query: str, limit: int = 10) -> list[dict]:
        """Search tracks. Max 10 per page."""
        default_limiter.wait()
        result = self.sp.search(q=query, type="track", limit=min(limit, 10))
        return result.get("tracks", {}).get("items", [])

    @with_retry()
    def search_artists(self, query: str, limit: int = 10) -> list[dict]:
        """Search artists by name. Returns minimal objects (name, id, images)."""
        default_limiter.wait()
        result = self.sp.search(q=query, type="artist", limit=min(limit, 10))
        return result.get("artists", {}).get("items", [])

    @with_retry()
    def search_playlists(self, query: str, limit: int = 10) -> list[dict]:
        """Search playlists by keyword."""
        default_limiter.wait()
        result = self.sp.search(q=query, type="playlist", limit=min(limit, 10))
        return result.get("playlists", {}).get("items", [])

    # --------------------------------------------------------------- playlists

    @with_retry()
    def get_playlist(self, playlist_id: str) -> Optional[dict]:
        """
        Get playlist metadata.
        NOTE: Editorial playlists (37i9...) return 404.
        Works for user-created public playlists.
        """
        cached = self._from_cache("playlist", playlist_id)
        if cached:
            return cached
        try:
            default_limiter.wait()
            result = self.sp.playlist(playlist_id, fields="id,name,owner,description,tracks.total")
            self._to_cache(result, "playlist", playlist_id)
            return result
        except Exception as e:
            logger.warning(f"Could not get playlist {playlist_id}: {e}")
            return None

    @with_retry()
    def get_playlist_tracks(self, playlist_id: str) -> list[dict]:
        """
        Get tracks from a playlist.
        Works for user-created public playlists.
        Editorial playlists (37i9...) are 404.
        """
        cached = self._from_cache("playlist_tracks", playlist_id)
        if cached:
            return cached

        tracks = []
        try:
            default_limiter.wait()
            page = self.sp.playlist_tracks(playlist_id, limit=100)
            while page:
                for item in page.get("items", []):
                    if item and item.get("track"):
                        tracks.append(item["track"])
                if page["next"]:
                    default_limiter.wait()
                    page = self.sp.next(page)
                else:
                    break
            self._to_cache(tracks, "playlist_tracks", playlist_id)
        except Exception as e:
            logger.warning(f"Could not get playlist tracks for {playlist_id}: {e}")

        return tracks

    # --------------------------------------------------------------- graph walk
    # related-artists endpoint is 403, so we use search-based expansion

    def find_similar_artists_by_isrc(self, isrc_prefix: str) -> list[dict]:
        """
        Find artists sharing an ISRC prefix (same production company).
        Used for Exercise 3 and graph expansion when related-artists is blocked.
        """
        # Search for tracks with this ISRC prefix
        results = self.search_tracks(f"isrc:{isrc_prefix}*", limit=10)
        artist_ids = set()
        for t in results:
            for artist in t.get("artists", []):
                artist_ids.add(artist["id"])
        return [self.get_artist(aid) for aid in artist_ids]

    def build_artist_graph_from_albums(
        self,
        seed_id: str,
        max_artists: int = 50,
    ) -> dict:
        """
        Build artist graph by finding co-artists on albums.
        Works without the blocked related-artists endpoint.
        Looks at album contributors and featured artists.

        Returns:
            {
                "nodes": [{id, name, release_count}, ...],
                "edges": [{source, target, album_name}, ...]
            }
        """
        visited = {}
        edges = []
        queue = [seed_id]
        seen_edges = set()

        while queue and len(visited) < max_artists:
            current_id = queue.pop(0)
            if current_id in visited:
                continue

            try:
                artist = self.get_artist(current_id)
                albums = self.get_artist_albums(current_id, include_groups="album,single")
                visited[current_id] = {
                    "id": current_id,
                    "name": artist["name"],
                    "release_count": len(albums),
                }
                logger.debug(f"  processing: {artist['name']} ({len(albums)} albums)")
            except Exception as e:
                logger.warning(f"Could not fetch {current_id}: {e}")
                continue

            # Find co-artists on albums
            for album in albums[:10]:  # limit to recent 10 albums
                for co_artist in album.get("artists", []):
                    cid = co_artist["id"]
                    if cid != current_id:
                        edge_key = tuple(sorted([current_id, cid]))
                        if edge_key not in seen_edges:
                            edges.append({
                                "source": current_id,
                                "target": cid,
                                "album_name": album["name"],
                            })
                            seen_edges.add(edge_key)
                        if cid not in visited:
                            queue.append(cid)

        return {"nodes": list(visited.values()), "edges": edges}

    # --------------------------------------------------------------- convenience

    def artist_id_from_name(self, name: str) -> Optional[str]:
        """Look up artist ID by name search."""
        results = self.search_artists(name, limit=1)
        if results:
            return results[0]["id"]
        return None

    def get_release_dates(self, artist_id: str) -> list[str]:
        """
        Get all release dates for an artist.
        Used in Signal 2 (cadence synchrony analysis).
        """
        albums = self.get_artist_albums(artist_id, include_groups="album,single")
        return [a["release_date"] for a in albums if a.get("release_date")]

    def test_connection(self) -> bool:
        """Quick connectivity test using search."""
        try:
            result = self.search_artists("Radiohead", limit=1)
            if result:
                logger.info(f"Spotify connection OK — found: {result[0]['name']}")
                return True
        except Exception as e:
            logger.error(f"Spotify connection failed: {e}")
        return False

    def test_isrc(self) -> bool:
        """Test that ISRC retrieval works."""
        try:
            # Use a known Radiohead track
            track = self.get_track("1HnY1Lu9tH5YK8DiIxho9i")
            isrc = track.get("external_ids", {}).get("isrc")
            if isrc:
                logger.info(f"ISRC test OK — got: {isrc}")
                return True
            logger.warning("ISRC field empty in response")
        except Exception as e:
            logger.error(f"ISRC test failed: {e}")
        return False
