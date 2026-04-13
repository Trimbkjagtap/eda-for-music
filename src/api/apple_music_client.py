"""
Apple Music / iTunes Search API client — no API key required.
Used for Signal 7 (cross-platform discrepancy).
Checks if a track appears in Apple Music's catalog.
"""
import httpx
from loguru import logger

from src.utils.rate_limiter import with_retry

ITUNES_SEARCH_URL = "https://itunes.apple.com/search"


class AppleMusicClient:

    @with_retry()
    def search_track(self, artist: str, track: str) -> list[dict]:
        """
        Search Apple Music (iTunes) catalog for a track.
        Returns list of matching results (empty = not found).
        No API key needed.
        """
        params = {
            "term": f"{artist} {track}",
            "media": "music",
            "entity": "song",
            "limit": 5,
        }
        with httpx.Client(timeout=10) as client:
            resp = client.get(ITUNES_SEARCH_URL, params=params)
            resp.raise_for_status()
            results = resp.json().get("results", [])
            logger.info(f"Apple Music: '{artist} - {track}' => {len(results)} results")
            return results

    def is_on_apple_music(self, artist: str, track: str) -> bool:
        """Return True if track appears in Apple Music catalog."""
        results = self.search_track(artist, track)
        return len(results) > 0

    def get_track_count(self, artist: str, track: str) -> int:
        """Return number of Apple Music results (0 = invisible on platform)."""
        results = self.search_track(artist, track)
        return len(results)
