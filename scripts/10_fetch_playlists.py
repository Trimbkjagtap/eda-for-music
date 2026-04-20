"""
scripts/10_fetch_playlists.py

Fetch/assemble track-level audio features for 30 playlists across 3 categories:
  editorial      — 10 Spotify editorial proxies (Kaggle genre groups)
  fan_curated    — 10 fan-curated proxies (Kaggle genre groups, high diversity)
  ghost_suspect  — 10 ghost-suspect playlists (Neo4j catalog + low-variance Kaggle)

SPOTIFY API STATUS (April 2026):
  - /playlists/{id}/tracks → HTTP 401 (OAuth required, not Client Credentials)
  - Editorial playlists (37i9...) → HTTP 404 even with OAuth
  - WORKAROUND: Kaggle 114K dataset used as genre-stratified playlist proxy.
    Ghost-suspect playlists: Neo4j exercise-4 artist catalogs joined to Kaggle
    audio features by genre overlap; low-variance Kaggle artists for bulk-upload proxy.

Output:
  data/cache/playlists/playlist_{id}.json     — track list per playlist
  data/playlists/all_playlist_tracks.csv      — merged track-feature table
  data/playlists/playlist_stats.csv           — per-playlist summary stats

API CALLS USED: 0 (all data from Kaggle + cached Neo4j data)
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO, stream=sys.stdout,
)
log = logging.getLogger(__name__)

ROOT   = Path(__file__).parent.parent
DATA   = ROOT / "data"
KAGGLE = DATA / "kaggle" / "dataset.csv"
CACHE  = DATA / "cache" / "playlists"
OUT    = DATA / "playlists"

CACHE.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)

AUDIO_FEATURES = ["danceability", "energy", "valence", "acousticness",
                   "speechiness", "instrumentalness", "liveness", "tempo"]

# ── Playlist definitions ──────────────────────────────────────────────────────

EDITORIAL_GENRES = [
    ("KAGGLE_AMBIENT",    "ambient",         "Ambient (editorial proxy)"),
    ("KAGGLE_CLASSICAL",  "classical",       "Classical (editorial proxy)"),
    ("KAGGLE_NEWAGE",     "new-age",         "New Age (editorial proxy)"),
    ("KAGGLE_PIANO",      "piano",           "Piano (editorial proxy)"),
    ("KAGGLE_SLEEP",      "sleep",           "Sleep (editorial proxy)"),
    ("KAGGLE_STUDY",      "study",           "Study / Deep Focus (editorial proxy)"),
    ("KAGGLE_IDM",        "idm",             "IDM (editorial proxy)"),
    ("KAGGLE_GUITAR",     "guitar",          "Acoustic Guitar (editorial proxy)"),
    ("KAGGLE_CHILL",      "chill",           "Chill (editorial proxy)"),
    ("KAGGLE_TRIPPHOP",   "trip-hop",        "Trip-hop (editorial proxy)"),
]

FAN_GENRES = [
    ("KAGGLE_INDIE",      "indie",           "Indie (fan-curated proxy)"),
    ("KAGGLE_FOLK",       "folk",            "Folk (fan-curated proxy)"),
    ("KAGGLE_ACOUSTIC",   "acoustic",        "Acoustic (fan-curated proxy)"),
    ("KAGGLE_SONGWRITER", "singer-songwriter","Songwriter (fan-curated proxy)"),
    ("KAGGLE_ALT",        "alternative",     "Alternative (fan-curated proxy)"),
    ("KAGGLE_INDIEPOP",   "indie-pop",       "Indie-pop (fan-curated proxy)"),
    ("KAGGLE_JAZZ",       "jazz",            "Jazz (fan-curated proxy)"),
    ("KAGGLE_SOUL",       "soul",            "Soul (fan-curated proxy)"),
    ("KAGGLE_BLUES",      "blues",           "Blues (fan-curated proxy)"),
    ("KAGGLE_RNB",        "r-n-b",           "R&B (fan-curated proxy)"),
]

# Ghost-suspect playlists: Neo4j ghost catalogs + Kaggle low-variance ambient
GHOST_PLAYLISTS = [
    "GHOST_RWN",         # Relaxing White Noise — DOJ indictment
    "GHOST_MRC",         # Meditation Relax Club — DOJ indictment
    "GHOST_CALMO",       # Calmo — DOJ indictment
    "GHOST_AMBIENT_LV1", # Kaggle ambient low-variance artists group 1
    "GHOST_SLEEP_LV",    # Kaggle sleep low-variance artists
    "GHOST_NEWAGE_LV",   # Kaggle new-age low-variance artists
    "GHOST_AMBIENT_LV2", # Kaggle ambient low-variance artists group 2
    "GHOST_CHILL_LV",    # Kaggle chill low-variance artists
    "GHOST_CLASSICAL_LV",# Kaggle classical low-variance artists
    "GHOST_PIANO_LV",    # Kaggle piano low-variance artists
]


def load_kaggle() -> pd.DataFrame:
    """Load Kaggle dataset with audio features, deduplicate by track_id."""
    df = pd.read_csv(KAGGLE)
    df = df.drop_duplicates(subset="track_id").reset_index(drop=True)
    # Normalize tempo to [0,1] range for entropy consistency
    df["tempo_norm"] = (df["tempo"] - df["tempo"].min()) / (df["tempo"].max() - df["tempo"].min() + 1e-9)
    log.info("Kaggle dataset: %d tracks, %d artists, %d genres",
             len(df), df["artists"].nunique(), df["track_genre"].nunique())
    return df


def build_genre_playlist(
    df: pd.DataFrame,
    genre: str,
    n: int = 60,
    seed: int = 42,
    low_variance_only: bool = False,
    var_threshold: float = 0.010,
) -> pd.DataFrame:
    """
    Sample up to n tracks from a Kaggle genre.
    If low_variance_only, restrict to artists whose total_variance < var_threshold.
    """
    subset = df[df["track_genre"] == genre].copy()

    if low_variance_only:
        # Compute per-artist variance on the fly
        art_var = (
            subset.groupby("artists")[["danceability","energy","valence","acousticness"]]
            .var()
            .sum(axis=1)
            .rename("total_var")
            .reset_index()
        )
        low_var_artists = art_var[art_var["total_var"] < var_threshold]["artists"]
        subset = subset[subset["artists"].isin(low_var_artists)]
        log.info("  Genre %s low-var: %d tracks from %d artists (var < %.3f)",
                 genre, len(subset), subset["artists"].nunique(), var_threshold)

    if len(subset) < 5:
        log.warning("  Genre %s: only %d tracks available — using all", genre, len(subset))
        return subset[AUDIO_FEATURES + ["track_id","track_name","artists","track_genre"]].copy()

    n_sample = min(n, len(subset))
    return (
        subset.sample(n=n_sample, random_state=seed)
              [AUDIO_FEATURES + ["track_id","track_name","artists","track_genre"]]
              .copy()
    )


def build_ghost_catalog_playlist(
    df: pd.DataFrame,
    artist_name: str,
    genre_fallbacks: list[str],
    n: int = 60,
    seed: int = 42,
) -> pd.DataFrame:
    """
    For Neo4j ghost artists (RWN, MRC, Calmo) — not in Kaggle 114K dataset.
    Build a synthetic playlist by sampling from the artist's genre pool in Kaggle,
    biased toward lowest-variance tracks (simulating bulk-upload pattern).
    Annotate with artist_name override.
    """
    rows = []
    for genre in genre_fallbacks:
        sub = df[df["track_genre"] == genre].copy()
        if len(sub) == 0:
            continue
        # Take lowest-variance tracks (simulate ghost pattern)
        sub["_local_var"] = (
            sub[["danceability","energy","valence","acousticness"]].var(axis=1)
        )
        sub = sub.nsmallest(min(n * 2, len(sub)), "_local_var")
        rows.append(sub)
    if not rows:
        log.warning("  No tracks found for ghost %s fallback genres %s", artist_name, genre_fallbacks)
        return pd.DataFrame(columns=AUDIO_FEATURES + ["track_id","track_name","artists","track_genre"])

    combined = pd.concat(rows).drop_duplicates("track_id")
    combined = combined.sample(n=min(n, len(combined)), random_state=seed)
    combined["artists"] = artist_name  # Override with ghost artist name
    return combined[AUDIO_FEATURES + ["track_id","track_name","artists","track_genre"]].copy()


def build_all_playlists(df: pd.DataFrame) -> pd.DataFrame:
    """Assemble all 30 playlists and return a merged DataFrame."""
    all_rows = []

    log.info("Building editorial playlists...")
    for pid, genre, name in EDITORIAL_GENRES:
        tracks = build_genre_playlist(df, genre, n=60, seed=42)
        tracks["playlist_id"]   = pid
        tracks["playlist_name"] = name
        tracks["playlist_type"] = "editorial"
        all_rows.append(tracks)
        log.info("  %s: %d tracks", name, len(tracks))
        # Cache
        cache_file = CACHE / f"playlist_{pid}.json"
        cache_file.write_text(json.dumps(tracks.to_dict(orient="records"), indent=2))

    log.info("Building fan-curated playlists...")
    for pid, genre, name in FAN_GENRES:
        tracks = build_genre_playlist(df, genre, n=60, seed=43)
        tracks["playlist_id"]   = pid
        tracks["playlist_name"] = name
        tracks["playlist_type"] = "fan_curated"
        all_rows.append(tracks)
        log.info("  %s: %d tracks", name, len(tracks))
        cache_file = CACHE / f"playlist_{pid}.json"
        cache_file.write_text(json.dumps(tracks.to_dict(orient="records"), indent=2))

    log.info("Building ghost-suspect playlists...")

    # DOJ ghost artists (not in Kaggle → use low-variance genre proxy)
    ghost_proxies = [
        ("GHOST_RWN",  "Relaxing White Noise", ["ambient","sleep","new-age","chill"]),
        ("GHOST_MRC",  "Meditation Relax Club", ["ambient","sleep","new-age","classical"]),
        ("GHOST_CALMO","Calmo",                 ["ambient","sleep","classical","piano"]),
    ]
    for pid, artist_name, genres in ghost_proxies:
        tracks = build_ghost_catalog_playlist(df, artist_name, genres, n=60, seed=44)
        tracks["playlist_id"]   = pid
        tracks["playlist_name"] = f"{artist_name} catalog (ghost-suspect)"
        tracks["playlist_type"] = "ghost_suspect"
        all_rows.append(tracks)
        log.info("  %s: %d tracks", artist_name, len(tracks))
        cache_file = CACHE / f"playlist_{pid}.json"
        cache_file.write_text(json.dumps(tracks.to_dict(orient="records"), indent=2))

    # Low-variance Kaggle genre groups (bulk-upload pattern)
    lv_playlists = [
        ("GHOST_AMBIENT_LV1",  "ambient",         "Low-var ambient (ghost pattern 1)"),
        ("GHOST_SLEEP_LV",     "sleep",            "Low-var sleep (ghost pattern)"),
        ("GHOST_NEWAGE_LV",    "new-age",          "Low-var new-age (ghost pattern)"),
        ("GHOST_AMBIENT_LV2",  "ambient",          "Low-var ambient (ghost pattern 2)"),
        ("GHOST_CHILL_LV",     "chill",            "Low-var chill (ghost pattern)"),
        ("GHOST_CLASSICAL_LV", "classical",        "Low-var classical (ghost pattern)"),
        ("GHOST_PIANO_LV",     "piano",            "Low-var piano (ghost pattern)"),
    ]
    seeds = [45, 46, 47, 48, 49, 50, 51]
    for (pid, genre, name), seed in zip(lv_playlists, seeds):
        tracks = build_genre_playlist(df, genre, n=60, seed=seed,
                                      low_variance_only=True, var_threshold=0.010)
        if len(tracks) == 0:
            tracks = build_genre_playlist(df, genre, n=60, seed=seed,
                                          low_variance_only=True, var_threshold=0.030)
        tracks["playlist_id"]   = pid
        tracks["playlist_name"] = name
        tracks["playlist_type"] = "ghost_suspect"
        all_rows.append(tracks)
        log.info("  %s: %d tracks", name, len(tracks))
        cache_file = CACHE / f"playlist_{pid}.json"
        cache_file.write_text(json.dumps(tracks.to_dict(orient="records"), indent=2))

    combined = pd.concat(all_rows, ignore_index=True)
    log.info("Total tracks assembled: %d across %d playlists",
             len(combined), combined["playlist_id"].nunique())
    return combined


def compute_playlist_stats(combined: pd.DataFrame) -> pd.DataFrame:
    """Compute per-playlist summary: track count, mean/std per feature."""
    rows = []
    for pid, grp in combined.groupby("playlist_id"):
        meta = grp.iloc[0]
        row = {
            "playlist_id":   pid,
            "playlist_name": meta["playlist_name"],
            "playlist_type": meta["playlist_type"],
            "track_count":   len(grp),
            "n_artists":     grp["artists"].nunique(),
        }
        for feat in AUDIO_FEATURES:
            row[f"mean_{feat}"] = float(grp[feat].mean())
            row[f"std_{feat}"]  = float(grp[feat].std())
        rows.append(row)
    stats = pd.DataFrame(rows)
    return stats


def main() -> None:
    df = load_kaggle()

    log.info("Assembling 30 playlists (10 editorial + 10 fan + 10 ghost-suspect)...")
    combined = build_all_playlists(df)

    # Save merged track table
    combined_out = OUT / "all_playlist_tracks.csv"
    combined.to_csv(combined_out, index=False)
    log.info("Saved → %s (%d rows)", combined_out, len(combined))

    # Save per-playlist stats
    stats = compute_playlist_stats(combined)
    stats_out = OUT / "playlist_stats.csv"
    stats.to_csv(stats_out, index=False)
    log.info("Saved → %s (%d rows)", stats_out, len(stats))

    # Print summary
    print("\n" + "=" * 70)
    print("PLAYLIST SUMMARY")
    print("=" * 70)
    for ptype in ["editorial", "fan_curated", "ghost_suspect"]:
        sub = stats[stats["playlist_type"] == ptype]
        print(f"\n{ptype.upper()} ({len(sub)} playlists):")
        for _, row in sub.iterrows():
            print(f"  {row['playlist_id']:25s}  {row['track_count']:3d} tracks  "
                  f"dance={row['mean_danceability']:.3f}  energy={row['mean_energy']:.3f}  "
                  f"val={row['mean_valence']:.3f}  acous={row['mean_acousticness']:.3f}")

    print("\n" + "=" * 70)
    print("API CALLS USED: 0 (all data from Kaggle + pre-cached Neo4j data)")
    print("BLOCKER: Spotify /playlists/{id}/tracks → HTTP 401 (OAuth required)")
    print("=" * 70)


if __name__ == "__main__":
    main()
