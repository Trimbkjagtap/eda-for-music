"""
Ground truth seed artists.
- GHOST: known fake/AI-generated artists from documented cases
- ORGANIC: verified real artists in similar genres as controls

Sources:
- Michael Smith wire fraud case (US DOJ, 2024)
- Dagens Nyheter investigation (830 fake artists, 2023-2024)
"""

# Known ghost/fake artists — Spotify IDs to be confirmed via search
# These come from the Michael Smith case and DN investigation
GHOST_ARTISTS = [
    # Michael Smith case — AI-generated ambient/lo-fi artists
    # IDs need to be verified via search (search by name first)
    {"name": "Calmo", "spotify_id": None, "source": "Michael Smith case", "genre": "ambient"},
    {"name": "Relaxing White Noise", "spotify_id": None, "source": "Michael Smith case", "genre": "ambient"},
    {"name": "Meditation Relax Club", "spotify_id": None, "source": "bot network", "genre": "meditation"},
    {"name": "Sleep Meditation Music", "spotify_id": None, "source": "bot network", "genre": "sleep"},
    {"name": "Study Music Project", "spotify_id": None, "source": "bot network", "genre": "focus"},
]

# Organic control artists — verified, similar genres
ORGANIC_ARTISTS = [
    {"name": "Radiohead", "spotify_id": "4Z8W4fKeB5YxbusRsdQVPb", "source": "verified", "genre": "alternative"},
    {"name": "Bon Iver", "spotify_id": "4LEiUm1SRbFMgfqnQTwUbQ", "source": "verified", "genre": "indie folk"},
    {"name": "Brian Eno", "spotify_id": "7MSUfLeTdDEoZiJPDSBXgi", "source": "verified", "genre": "ambient"},
    {"name": "Nils Frahm", "spotify_id": "5hVghJ4KaYES3BFUATCYn0", "source": "verified", "genre": "neo-classical"},
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
    """Return all seed artists with their labels."""
    seeds = []
    for a in GHOST_ARTISTS:
        seeds.append({**a, "is_ghost": True})
    for a in ORGANIC_ARTISTS:
        seeds.append({**a, "is_ghost": False})
    return seeds


def get_ghost_seeds() -> list[dict]:
    return [{**a, "is_ghost": True} for a in GHOST_ARTISTS]


def get_organic_seeds() -> list[dict]:
    return [{**a, "is_ghost": False} for a in ORGANIC_ARTISTS]
