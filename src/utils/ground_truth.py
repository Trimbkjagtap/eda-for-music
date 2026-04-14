"""
Ground truth seed artists.
- GHOST: known fake/AI-generated artists from documented cases
- ORGANIC: verified real artists in similar genres as controls
- KAGGLE_CANDIDATES: lowest-variance artists from Kaggle dataset analysis
  (pending Spotify API verification — do not treat as confirmed ghost artists)

Sources:
- Michael Smith wire fraud case (US DOJ, 2024)
- Dagens Nyheter investigation (830 fake artists, 2023-2024)
- Exercise 1 Kaggle analysis: artists with 15+ tracks, lowest total audio feature variance
  (variance = sum of per-feature variance across danceability, energy, valence, acousticness)
"""

# Known ghost/fake artists — Spotify IDs confirmed via Neo4j ingestion
GHOST_ARTISTS = [
    # Michael Smith case — AI-generated ambient/lo-fi artists
    {"name": "Calmo", "spotify_id": "4Wx3ZL6d6p1gVMtwQ2YWsz",
     "source": "Michael Smith case", "genre": "ambient",
     "neo4j_tracks": 38, "isrc_prefixes": ["ITIWE", "CH654"]},
    {"name": "Relaxing White Noise", "spotify_id": None,
     "source": "Michael Smith case", "genre": "ambient",
     "neo4j_tracks": 280, "isrc_prefixes": ["DEPI8", "DE1QW"]},
    {"name": "Meditation Relax Club", "spotify_id": None,
     "source": "bot network", "genre": "meditation",
     "neo4j_tracks": 172, "isrc_prefixes": ["ITO10", "ITLU5"]},
    {"name": "Sleep Meditation Music", "spotify_id": None,
     "source": "bot network", "genre": "sleep",
     "neo4j_tracks": 0, "isrc_prefixes": []},
    {"name": "Study Music Project", "spotify_id": None,
     "source": "bot network", "genre": "focus",
     "neo4j_tracks": 0, "isrc_prefixes": []},
]

# Kaggle-derived ghost candidates — artists with lowest non-trivial total variance
# across danceability, energy, valence, acousticness (15+ tracks, variance > 1e-8).
# NOTE: These are derived from Kaggle CSV analysis only.
# Spotify IDs are NOT set — these need verification via Spotify search API.
# Many low-variance entries in Kaggle are multi-artist collaborations where
# the same track appears across multiple genre buckets (dataset artifact).
# Do NOT treat as confirmed ghost artists until Spotify API verification is complete.
KAGGLE_GHOST_CANDIDATES = [
    # Top 3 lowest non-trivial variance artists (≥15 tracks), from Exercise 1
    # total_variance values from Kaggle analysis (2026-04-14)
    {"name": "The Verve",
     "spotify_id": None,  # pending API verification
     "source": "Kaggle variance analysis (Exercise 1)",
     "genre": "alternative",
     "kaggle_track_count": 15,
     "total_variance": 4.287e-06,
     "note": "Kaggle-derived candidate pending Spotify API verification"},
    {"name": "Lykke Li;The Magician",
     "spotify_id": None,
     "source": "Kaggle variance analysis (Exercise 1)",
     "genre": "pop",
     "kaggle_track_count": 18,
     "total_variance": 8.181e-06,
     "note": "Kaggle-derived candidate pending Spotify API verification — multi-artist entry"},
    {"name": "MEDUZA;Becky Hill;Goodboys",
     "spotify_id": None,
     "source": "Kaggle variance analysis (Exercise 1)",
     "genre": "dance",
     "kaggle_track_count": 15,
     "total_variance": 8.800e-06,
     "note": "Kaggle-derived candidate pending Spotify API verification — multi-artist entry"},
]

# Organic control artists — verified, similar genres
ORGANIC_ARTISTS = [
    {"name": "Radiohead", "spotify_id": "4Z8W4fKeB5YxbusRsdQVPb", "source": "verified", "genre": "alternative"},
    {"name": "Bon Iver", "spotify_id": "4LEiUm1SRbFMgfqnQTwUbQ", "source": "verified", "genre": "indie folk"},
    {"name": "Brian Eno", "spotify_id": "7MSUfLeTdDEoZiJPDSBXgi", "source": "verified", "genre": "ambient"},
    {"name": "Nils Frahm", "spotify_id": "5gqhueRUZEa7VDnQt4HODp", "source": "verified", "genre": "neo-classical"},
    {"name": "Tycho", "spotify_id": "5oOhM2DFWab8XhSoAkacgR", "source": "verified", "genre": "ambient electronic"},
]

# ISRC prefix hints — known Swedish production companies from DN investigation
# These ISRC registrant prefixes identify production companies
KNOWN_ISRC_PREFIXES = {
    "SEWA": "Firefly Entertainment AB",
    "SEWB": "Lucille AB",
    "SEZZ": "Tombola Music",
    "SEXX": "Catfish Music Group",
    "SECC": "Calm and Collected Music Publishing",
}


def get_all_seeds() -> list[dict]:
    """Return confirmed ghost + organic seed artists with their labels."""
    seeds = []
    for a in GHOST_ARTISTS:
        seeds.append({**a, "is_ghost": True})
    for a in ORGANIC_ARTISTS:
        seeds.append({**a, "is_ghost": False})
    return seeds


def get_kaggle_candidates() -> list[dict]:
    """Return unverified ghost candidates derived from Kaggle variance analysis."""
    return [{**a, "is_ghost": None} for a in KAGGLE_GHOST_CANDIDATES]


def get_ghost_seeds() -> list[dict]:
    return [{**a, "is_ghost": True} for a in GHOST_ARTISTS]


def get_organic_seeds() -> list[dict]:
    return [{**a, "is_ghost": False} for a in ORGANIC_ARTISTS]
