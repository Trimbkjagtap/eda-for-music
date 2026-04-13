"""
YouTube Data API v3 client — used for Signal 7 (cross-platform discrepancy).
Searches for a track by name + artist and returns view count.
"""
import httpx
from loguru import logger

from src.utils.config import config
from src.utils.rate_limiter import with_retry

YOUTUBE_SEARCH_URL = "https://www.googleapis.com/youtube/v3/search"
YOUTUBE_VIDEOS_URL = "https://www.googleapis.com/youtube/v3/videos"


class YouTubeClient:
    def __init__(self):
        self.api_key = config.YOUTUBE_API_KEY

    @with_retry()
    def search_video(self, artist: str, track: str) -> dict | None:
        """
        Search YouTube for a track. Returns the top result or None.
        """
        if not self.api_key:
            logger.warning("YOUTUBE_API_KEY not set — skipping YouTube lookup")
            return None

        query = f"{artist} {track} official"
        params = {
            "part": "snippet",
            "q": query,
            "type": "video",
            "maxResults": 1,
            "key": self.api_key,
        }
        with httpx.Client(timeout=10) as client:
            resp = client.get(YOUTUBE_SEARCH_URL, params=params)
            resp.raise_for_status()
            items = resp.json().get("items", [])
            if not items:
                return None
            return items[0]

    @with_retry()
    def get_view_count(self, video_id: str) -> int:
        """Get view count for a YouTube video ID."""
        if not self.api_key:
            return 0

        params = {
            "part": "statistics",
            "id": video_id,
            "key": self.api_key,
        }
        with httpx.Client(timeout=10) as client:
            resp = client.get(YOUTUBE_VIDEOS_URL, params=params)
            resp.raise_for_status()
            items = resp.json().get("items", [])
            if not items:
                return 0
            stats = items[0].get("statistics", {})
            return int(stats.get("viewCount", 0))

    def get_track_views(self, artist: str, track: str) -> int:
        """
        Combined: search for a track, return its view count.
        Returns 0 if not found or API unavailable.
        """
        video = self.search_video(artist, track)
        if not video:
            logger.info(f"YouTube: no results for '{artist} - {track}'")
            return 0
        video_id = video["id"]["videoId"]
        views = self.get_view_count(video_id)
        logger.info(f"YouTube: '{artist} - {track}' => {views:,} views")
        return views
