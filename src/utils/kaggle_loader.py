"""
Kaggle audio features loader.
The Spotify audio-features endpoint is blocked for new apps (Nov 2024+).
We join a Kaggle dataset on Spotify track ID to get:
  danceability, energy, loudness, speechiness, acousticness,
  instrumentalness, liveness, valence, tempo, time_signature

Dataset: https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset
Place CSV at: data/kaggle/dataset.csv
"""
import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger

from src.utils.config import config

AUDIO_FEATURES = [
    "danceability", "energy", "loudness", "speechiness",
    "acousticness", "instrumentalness", "liveness", "valence", "tempo",
]

FEATURE_4D = ["danceability", "energy", "valence", "acousticness"]


class KaggleLoader:
    def __init__(self):
        self._df: pd.DataFrame | None = None
        self.path = config.KAGGLE_DIR / "dataset.csv"

    def load(self) -> pd.DataFrame:
        if self._df is not None:
            return self._df

        if not self.path.exists():
            logger.warning(
                f"Kaggle dataset not found at {self.path}. "
                "Download from https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset"
            )
            return pd.DataFrame()

        logger.info(f"Loading Kaggle dataset from {self.path}")
        df = pd.read_csv(self.path, low_memory=False)

        # Standardize column name for track ID
        if "track_id" in df.columns:
            df = df.rename(columns={"track_id": "spotify_id"})
        elif "id" in df.columns:
            df = df.rename(columns={"id": "spotify_id"})

        # Keep only relevant columns
        keep = ["spotify_id"] + [f for f in AUDIO_FEATURES if f in df.columns]
        df = df[keep].drop_duplicates(subset=["spotify_id"])

        self._df = df
        logger.info(f"Kaggle dataset loaded: {len(df):,} tracks")
        return df

    def get_features(self, track_ids: list[str]) -> pd.DataFrame:
        """
        Get audio features for a list of Spotify track IDs.
        Returns DataFrame with one row per track. Missing tracks are NaN rows.
        """
        df = self.load()
        if df.empty:
            return pd.DataFrame({"spotify_id": track_ids})

        result = (
            pd.DataFrame({"spotify_id": track_ids})
            .merge(df, on="spotify_id", how="left")
        )
        missing = result[AUDIO_FEATURES[0]].isna().sum() if AUDIO_FEATURES[0] in result.columns else 0
        logger.info(f"Audio features: {len(track_ids)} requested, {missing} missing from Kaggle dataset")
        return result

    def compute_catalog_variance(self, track_ids: list[str]) -> dict:
        """
        Compute variance of a catalog in 4D audio feature space.
        Used in Exercise 1 (catalog coherence).
        Returns: {feature: variance, ...} plus 'overall_variance' scalar.
        """
        feats = self.get_features(track_ids)
        available = [f for f in FEATURE_4D if f in feats.columns]
        if not available:
            return {}

        variances = {}
        for f in available:
            col = feats[f].dropna()
            variances[f] = float(col.var()) if len(col) > 1 else 0.0

        # Overall variance = mean eigenvalue of covariance matrix
        valid = feats[available].dropna()
        if len(valid) > 1:
            cov = np.cov(valid.T)
            if cov.ndim == 0:
                overall = float(cov)
            else:
                overall = float(np.mean(np.linalg.eigvalsh(cov)))
        else:
            overall = 0.0

        variances["overall_variance"] = overall
        variances["track_count"] = len(feats)
        return variances


# Shared instance
kaggle = KaggleLoader()
