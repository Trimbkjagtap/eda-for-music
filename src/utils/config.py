"""
Configuration — loads all settings from .env
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root
ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")


class Config:
    # Spotify
    SPOTIFY_CLIENT_ID: str = os.getenv("SPOTIFY_CLIENT_ID", "")
    SPOTIFY_CLIENT_SECRET: str = os.getenv("SPOTIFY_CLIENT_SECRET", "")
    SPOTIFY_REDIRECT_URI: str = os.getenv("SPOTIFY_REDIRECT_URI", "http://127.0.0.1:8888/callback")

    # Neo4j
    NEO4J_URI: str = os.getenv("NEO4J_URI", "")
    NEO4J_USER: str = os.getenv("NEO4J_USER", "neo4j")
    NEO4J_PASSWORD: str = os.getenv("NEO4J_PASSWORD", "")

    # YouTube
    YOUTUBE_API_KEY: str = os.getenv("YOUTUBE_API_KEY", "")

    # Paths
    DATA_DIR: Path = ROOT / "data"
    RAW_DIR: Path = ROOT / "data" / "raw"
    CACHE_DIR: Path = ROOT / "data" / "raw" / "cache"
    PROCESSED_DIR: Path = ROOT / "data" / "processed"
    KAGGLE_DIR: Path = ROOT / "data" / "kaggle"

    # Rate limiting
    RATE_LIMIT_CALLS_PER_SECOND: int = int(os.getenv("RATE_LIMIT_CALLS_PER_SECOND", "10"))

    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    def validate(self) -> list[str]:
        """Return list of missing required credentials."""
        missing = []
        if not self.SPOTIFY_CLIENT_ID:
            missing.append("SPOTIFY_CLIENT_ID")
        if not self.SPOTIFY_CLIENT_SECRET:
            missing.append("SPOTIFY_CLIENT_SECRET")
        return missing


config = Config()
