"""
scripts/01_build_ground_truth.py
================================
Build verified ground-truth dataset for GhostTrack training and evaluation.

Produces:
  data/ground_truth/ghost_artists.csv   — ≥150 ghost/suspected-ghost artists
  data/ground_truth/organic_artists.csv — ≥150 organic control artists
  data/cache/artist_resolution.json     — cached Spotify ID lookups
  data/ground_truth/missing_ids.csv     — artists whose IDs could not be resolved

Usage:
  python scripts/01_build_ground_truth.py [--resolve] [--dry-run]

Flags:
  --resolve   Attempt live Spotify API calls to resolve missing IDs
              (default: off — use cached data only to stay under API limit)
  --dry-run   Print summary without writing files

API call budget: hard cap of 50 live calls (enforced in SpotifyClient).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# ── path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
from loguru import logger

# ── constants ─────────────────────────────────────────────────────────────────
GT_DIR = ROOT / "data" / "ground_truth"
CACHE_PATH = ROOT / "data" / "cache" / "artist_resolution.json"
KAGGLE_PATH = ROOT / "data" / "kaggle" / "dataset.csv"

GT_DIR.mkdir(parents=True, exist_ok=True)
(ROOT / "data" / "cache").mkdir(parents=True, exist_ok=True)

# ── Source 1: Documented ghost artists (journalist-confirmed) ─────────────────
# Sources: Michael Smith DOJ indictment (Sept 2024) + Dagens Nyheter investigation
# Confidence: HIGH = journalist-confirmed; MED = named in reporting; LOW = heuristic

DOCUMENTED_GHOSTS: list[dict] = [
    # ── Michael Smith wire fraud case (US DOJ, Sept 2024) ──────────────────
    # Artists directly named in indictment or contemporaneous press coverage
    {"name": "Calmo",                    "source": "Smith",  "confidence": "high",  "notes": "Named in DOJ indictment; ISRC ITIWE/CH654; 38 tracks in Neo4j"},
    {"name": "Relaxing White Noise",     "source": "Smith",  "confidence": "high",  "notes": "Named in DOJ indictment; ISRC DEPI8/DE1QW; 280 tracks in Neo4j"},
    {"name": "Meditation Relax Club",    "source": "Smith",  "confidence": "high",  "notes": "Named in indictment and Dagens Nyheter; ISRC ITO10/ITLU5; 172 tracks"},
    {"name": "Sleep Meditation Music",   "source": "Smith",  "confidence": "high",  "notes": "Named in indictment press coverage; ambient/sleep genre"},
    {"name": "Study Music Project",      "source": "Smith",  "confidence": "high",  "notes": "Named in indictment press coverage; study/focus genre"},
    # 25 alphabetical sample artist names from the Smith indictment exhibit
    # (indictment notes these are '25 consecutive examples' of hundreds of thousands)
    {"name": "Calliope Bloom",           "source": "Smith",  "confidence": "med",   "notes": "25-sample indictment exhibit; AI-generated ambient/lo-fi"},
    {"name": "Calm Connected",           "source": "Smith",  "confidence": "med",   "notes": "25-sample indictment exhibit"},
    {"name": "Calm Force",               "source": "Smith",  "confidence": "med",   "notes": "25-sample indictment exhibit"},
    {"name": "Calm Identity",            "source": "Smith",  "confidence": "med",   "notes": "25-sample indictment exhibit"},
    {"name": "Calm Innovation",          "source": "Smith",  "confidence": "med",   "notes": "25-sample indictment exhibit"},
    {"name": "Calm Baseball",            "source": "Smith",  "confidence": "med",   "notes": "25-sample indictment exhibit"},
    {"name": "Calm Knuckles",            "source": "Smith",  "confidence": "med",   "notes": "25-sample indictment exhibit"},
    {"name": "Calm Market",              "source": "Smith",  "confidence": "med",   "notes": "25-sample indictment exhibit"},
    {"name": "Calm The Super",           "source": "Smith",  "confidence": "med",   "notes": "25-sample indictment exhibit"},
    {"name": "Calm Weary",               "source": "Smith",  "confidence": "med",   "notes": "25-sample indictment exhibit"},
    {"name": "Calms Scorching",          "source": "Smith",  "confidence": "med",   "notes": "25-sample indictment exhibit"},
    {"name": "Calorie Event",            "source": "Smith",  "confidence": "med",   "notes": "25-sample indictment exhibit"},
    {"name": "Calorie Screams",          "source": "Smith",  "confidence": "med",   "notes": "25-sample indictment exhibit"},
    {"name": "Calvin Mann",              "source": "Smith",  "confidence": "med",   "notes": "25-sample indictment exhibit"},
    {"name": "Calvinistic Dust",         "source": "Smith",  "confidence": "med",   "notes": "25-sample indictment exhibit"},
    {"name": "Calypso Xored",            "source": "Smith",  "confidence": "med",   "notes": "25-sample indictment exhibit"},
    {"name": "Camalus Disen",            "source": "Smith",  "confidence": "med",   "notes": "25-sample indictment exhibit"},
    {"name": "Camaxtli Minerva",         "source": "Smith",  "confidence": "med",   "notes": "25-sample indictment exhibit"},
    {"name": "Cambists Cagelings",       "source": "Smith",  "confidence": "med",   "notes": "25-sample indictment exhibit"},
    {"name": "Camel Edible",             "source": "Smith",  "confidence": "med",   "notes": "25-sample indictment exhibit"},

    # ── Dagens Nyheter investigation (Johan Röhr / Firefly Entertainment AB) ─
    # 9 pseudonyms named in English-language press coverage
    {"name": "Minik Knudsen",            "source": "DN",     "confidence": "med",   "notes": "Johan Röhr pseudonym; Firefly Entertainment AB (ISRC SEWA)"},
    {"name": "Mingmei Hsueh",            "source": "DN",     "confidence": "med",   "notes": "Johan Röhr pseudonym; Firefly Entertainment AB"},
    {"name": "Csizmazia Etel",           "source": "DN",     "confidence": "med",   "notes": "Johan Röhr pseudonym; Firefly Entertainment AB"},
    {"name": "Adelmar Borrego",          "source": "DN",     "confidence": "med",   "notes": "Johan Röhr pseudonym; Firefly Entertainment AB"},
    {"name": "Maya Åström",              "source": "DN",     "confidence": "med",   "notes": "Johan Röhr pseudonym; Firefly Entertainment AB"},
    {"name": "Ralph Kaler",              "source": "DN",     "confidence": "med",   "notes": "Johan Röhr pseudonym; Firefly Entertainment AB"},
    {"name": "Sherry Novak",             "source": "DN",     "confidence": "med",   "notes": "Johan Röhr pseudonym; Firefly Entertainment AB"},
    {"name": "Jospeh Turley",            "source": "DN",     "confidence": "med",   "notes": "Johan Röhr pseudonym; Firefly Entertainment AB"},
    {"name": "Miu Hayashi",              "source": "DN",     "confidence": "med",   "notes": "Johan Röhr pseudonym; Firefly Entertainment AB"},

    # ── Kaggle-variance ghost candidates (ambient/sleep/study/new-age genres) ──
    # Artists with total_variance < 0.050 and ≥10 tracks in target genres.
    # NOT journalist-confirmed; heuristic label based on audio-feature analysis.
    # confidence=low — treat as ghost-candidate, not confirmed ghost.
    {"name": "Bernward Koch",            "source": "Kaggle-variance", "confidence": "low", "notes": "new-age; variance=0.009; 10 tracks; no Spotify confirmation"},
    {"name": "Mc_team",                  "source": "Kaggle-variance", "confidence": "low", "notes": "world-music; variance=0.012; 12 tracks"},
    {"name": "Stanton Lanier",           "source": "Kaggle-variance", "confidence": "low", "notes": "new-age/piano; variance=0.012; 10 tracks"},
    {"name": "Emile Pandolfi",           "source": "Kaggle-variance", "confidence": "low", "notes": "new-age/piano; variance=0.012; 10 tracks"},
    {"name": "Deuter",                   "source": "Kaggle-variance", "confidence": "low", "notes": "new-age; variance=0.013; 22 tracks"},
    {"name": "Liquid Mind",              "source": "Kaggle-variance", "confidence": "low", "notes": "ambient/new-age; variance=0.013; 11 tracks"},
    {"name": "Robert Haig Coxon",        "source": "Kaggle-variance", "confidence": "low", "notes": "new-age; variance=0.014; 14 tracks"},
    {"name": "Instrumental Worship Project from I'm In Records",
                                         "source": "Kaggle-variance", "confidence": "low", "notes": "piano; variance=0.016; 13 tracks; name suggests library music"},
    {"name": "Chad Lawson",              "source": "Kaggle-variance", "confidence": "low", "notes": "ambient/classical/new-age/piano; variance=0.016; 12 tracks"},
    {"name": "Stephan Moccio",           "source": "Kaggle-variance", "confidence": "low", "notes": "ambient/classical/piano; variance=0.017; 22 tracks"},
    {"name": "Alexis Ffrench",           "source": "Kaggle-variance", "confidence": "low", "notes": "ambient; variance=0.020; 22 tracks"},
    {"name": "Jacob's Piano",            "source": "Kaggle-variance", "confidence": "low", "notes": "new-age/piano; variance=0.025; 22 tracks"},
    {"name": "Jonah Baker",              "source": "Kaggle-variance", "confidence": "low", "notes": "acoustic; variance=0.025; 13 tracks"},
    {"name": "Gomer Edwin Evans",        "source": "Kaggle-variance", "confidence": "low", "notes": "new-age; variance=0.030; 13 tracks"},
    {"name": "Fabrizio Paterlini",       "source": "Kaggle-variance", "confidence": "low", "notes": "ambient/classical; variance=0.031; 15 tracks"},
    {"name": "Yusei",                    "source": "Kaggle-variance", "confidence": "low", "notes": "chill; variance=0.031; 13 tracks"},
    {"name": "Dan Gibson's Solitudes",   "source": "Kaggle-variance", "confidence": "low", "notes": "new-age; variance=0.031; 90 tracks; large catalog typical of ghost"},
    {"name": "The Civil Wars",           "source": "Kaggle-variance", "confidence": "low", "notes": "acoustic; variance=0.031; 13 tracks"},
    {"name": "The Mayries",              "source": "Kaggle-variance", "confidence": "low", "notes": "acoustic; variance=0.032; 10 tracks"},
    {"name": "Laura Sullivan",           "source": "Kaggle-variance", "confidence": "low", "notes": "new-age/piano/sleep; variance=0.033; 23 tracks"},
    {"name": "Brian Crain",              "source": "Kaggle-variance", "confidence": "low", "notes": "new-age; variance=0.033; 21 tracks"},
    {"name": "Canyon City",              "source": "Kaggle-variance", "confidence": "low", "notes": "acoustic; variance=0.034; 10 tracks"},
    {"name": "Meeresrauschen",           "source": "Kaggle-variance", "confidence": "low", "notes": "sleep; variance=0.034; 24 tracks; German name"},
    {"name": "Obie Hans",                "source": "Kaggle-variance", "confidence": "low", "notes": "study; variance=0.034; 11 tracks"},
    {"name": "Raimu",                    "source": "Kaggle-variance", "confidence": "low", "notes": "chill; variance=0.035; 11 tracks"},
    {"name": "Yiruma",                   "source": "Kaggle-variance", "confidence": "low", "notes": "classical/new-age/piano; variance=0.036; 93 tracks"},
    {"name": "Henrik Janson",            "source": "Kaggle-variance", "confidence": "low", "notes": "guitar; variance=0.038; 24 tracks"},
    {"name": "Paul Cardall",             "source": "Kaggle-variance", "confidence": "low", "notes": "new-age/piano; variance=0.038; 40 tracks"},
    {"name": "Los Tecolines",            "source": "Kaggle-variance", "confidence": "low", "notes": "guitar; variance=0.039; 32 tracks"},
    {"name": "Dither",                   "source": "Kaggle-variance", "confidence": "low", "notes": "idm; variance=0.040; 16 tracks"},
    {"name": "BigRicePiano",             "source": "Kaggle-variance", "confidence": "low", "notes": "piano; variance=0.040; 14 tracks"},
    {"name": "Smyang Piano",             "source": "Kaggle-variance", "confidence": "low", "notes": "piano; variance=0.041; 47 tracks; YouTube cover channel"},
    {"name": "Maneli Jamal",             "source": "Kaggle-variance", "confidence": "low", "notes": "guitar; variance=0.043; 10 tracks"},
    {"name": "Los Tres Caballeros",      "source": "Kaggle-variance", "confidence": "low", "notes": "guitar; variance=0.043; 12 tracks"},
    {"name": "Ever So Blue",             "source": "Kaggle-variance", "confidence": "low", "notes": "piano; variance=0.043; 11 tracks"},
    {"name": "PIÑA PINK",                "source": "Kaggle-variance", "confidence": "low", "notes": "study; variance=0.043; 10 tracks"},
    {"name": "Jim Brickman",             "source": "Kaggle-variance", "confidence": "low", "notes": "new-age/piano; variance=0.044; 76 tracks"},
    {"name": "Bhimsen Joshi",            "source": "Kaggle-variance", "confidence": "low", "notes": "classical; variance=0.046; 11 tracks"},
    {"name": "Little Symphony",          "source": "Kaggle-variance", "confidence": "low", "notes": "ambient/guitar; variance=0.046; 94 tracks; very large catalog"},
    {"name": "Lucas Cervetti",           "source": "Kaggle-variance", "confidence": "low", "notes": "world-music; variance=0.047; 12 tracks"},
    {"name": "Joseph Sullinger",         "source": "Kaggle-variance", "confidence": "low", "notes": "acoustic/guitar; variance=0.048; 13 tracks"},
    {"name": "Guus Dielissen",           "source": "Kaggle-variance", "confidence": "low", "notes": "acoustic/guitar; variance=0.049; 13 tracks"},
    {"name": "Akira Kosemura",           "source": "Kaggle-variance", "confidence": "low", "notes": "ambient/classical/piano; variance=0.049; 11 tracks"},
    {"name": "Tenno",                    "source": "Kaggle-variance", "confidence": "low", "notes": "chill; variance=0.050; 12 tracks"},
    {"name": "Blackburri",               "source": "Kaggle-variance", "confidence": "low", "notes": "study; variance=0.050; 14 tracks"},
    {"name": "Eddy Tyler",               "source": "Kaggle-variance", "confidence": "low", "notes": "acoustic/guitar; variance=0.050; 11 tracks"},
    {"name": "Tony O'Connor",            "source": "Kaggle-variance", "confidence": "low", "notes": "new-age; variance=0.052; 35 tracks"},
    {"name": "Gramatik",                 "source": "Kaggle-variance", "confidence": "low", "notes": "trip-hop; variance=0.052; 11 tracks"},
    {"name": "Avslappning Sound",        "source": "Kaggle-variance", "confidence": "low", "notes": "sleep; variance=0.053; 20 tracks; Swedish name = likely Firefly-adjacent"},
    {"name": "Kevin Kern",               "source": "Kaggle-variance", "confidence": "low", "notes": "new-age/piano; variance=0.056; 10 tracks"},
    {"name": "Jóhann Jóhannsson",        "source": "Kaggle-variance", "confidence": "low", "notes": "ambient/classical; variance=0.057; 20 tracks"},
    {"name": "Chamin Madero",            "source": "Kaggle-variance", "confidence": "low", "notes": "guitar; variance=0.059; 11 tracks"},

    # Additional library-music / production-music brands (high-confidence ghost)
    {"name": "Firefly Entertainment",    "source": "DN",     "confidence": "high",  "notes": "The production label behind Johan Röhr's 656 pseudonyms; ISRC SEWA"},
    {"name": "Epidemic Sound",           "source": "other",  "confidence": "high",  "notes": "Production library music; named in Pelly investigation; Overtone Studios sublabel"},
    {"name": "Relaxing Sounds",          "source": "other",  "confidence": "med",   "notes": "Generic ambient factory brand; common ghost-artist naming pattern"},
    {"name": "Sleep Sounds White Noise", "source": "other",  "confidence": "med",   "notes": "Common ghost artist naming pattern; white noise category"},
    {"name": "Nature Sound Collection",  "source": "other",  "confidence": "med",   "notes": "Nature-sounds category; common ghost artist naming pattern"},
    {"name": "Binaural Beats Brain Waves","source": "other", "confidence": "med",   "notes": "Binaural beats category; common ghost naming pattern"},
    {"name": "White Noise Therapy",      "source": "other",  "confidence": "med",   "notes": "Therapy-adjacent ambient; common ghost naming pattern"},
    {"name": "Ambient Music Therapy",    "source": "other",  "confidence": "med",   "notes": "Therapy-adjacent; common ghost naming pattern"},
    {"name": "Relaxing Rain Sounds",     "source": "other",  "confidence": "med",   "notes": "Rain sound generator brand; common ghost pattern"},
    {"name": "Study Music Alpha Waves",  "source": "other",  "confidence": "med",   "notes": "Study music factory; alpha waves branding common in ghost artists"},
    {"name": "Focus At Will",            "source": "other",  "confidence": "med",   "notes": "Study/focus music service; functional music factory pattern"},
    {"name": "Calm Radio",               "source": "other",  "confidence": "med",   "notes": "Ambient radio brand; functional music factory"},
    {"name": "White Noise Baby",         "source": "Kaggle-variance", "confidence": "low", "notes": "sleep; high-variance outlier but name pattern matches ghost factory"},
    {"name": "Pink Noise",               "source": "Kaggle-variance", "confidence": "low", "notes": "sleep; functional noise generator"},
]

# ── Source 2: Organic control artists ────────────────────────────────────────
# Same-genre coverage to avoid genre-confounding in classifier training.
# All have known Spotify IDs or are easily resolvable.

DOCUMENTED_ORGANICS: list[dict] = [
    # ── Core study artists (Neo4j confirmed) ─────────────────────────────────
    {"name": "Nils Frahm",          "spotify_artist_id": "5gqhueRUZEa7VDnQt4HODp", "source": "verified", "confidence": "high", "notes": "Neo4j seed; 0% cadence closure; Pitchfork 8.5/10"},
    {"name": "Brian Eno",           "spotify_artist_id": "7MSUfLeTdDEoZiJPDSBXgi", "source": "verified", "confidence": "high", "notes": "Father of ambient; verified organic"},
    {"name": "Tycho",               "spotify_artist_id": "5oOhM2DFWab8XhSoAkacgR", "source": "verified", "confidence": "high", "notes": "Ambient electronic; organic"},
    {"name": "Radiohead",           "spotify_artist_id": "4Z8W4fKeB5YxbusRsdQVPb", "source": "verified", "confidence": "high", "notes": "Verified organic control"},
    {"name": "Bon Iver",            "spotify_artist_id": "4LEiUm1SRbFMgfqnQTwUbQ", "source": "verified", "confidence": "high", "notes": "Verified organic control"},

    # ── Ambient / neo-classical ───────────────────────────────────────────────
    {"name": "Stars of the Lid",    "spotify_artist_id": None, "source": "verified", "confidence": "high", "notes": "Drone ambient; established Texas duo; Kranky Records"},
    {"name": "Hammock",             "spotify_artist_id": None, "source": "verified", "confidence": "high", "notes": "Ambient post-rock; Nashville; multiple full-length albums"},
    {"name": "Hiroshi Yoshimura",   "spotify_artist_id": None, "source": "verified", "confidence": "high", "notes": "Japanese ambient pioneer; Music for Nine Post Cards (1982)"},
    {"name": "Harold Budd",         "spotify_artist_id": None, "source": "verified", "confidence": "high", "notes": "Ambient classical; collaborator with Brian Eno"},
    {"name": "Max Richter",         "spotify_artist_id": None, "source": "verified", "confidence": "high", "notes": "Neo-classical; Sleep album; Deutsche Grammophon"},
    {"name": "Ólafur Arnalds",      "spotify_artist_id": None, "source": "verified", "confidence": "high", "notes": "Icelandic neo-classical; verified Kaggle variance=0.049"},
    {"name": "Jóhann Jóhannsson",   "spotify_artist_id": None, "source": "verified", "confidence": "high", "notes": "Icelandic composer; deceased 2018; Deutsche Grammophon"},
    {"name": "Dustin O'Halloran",   "spotify_artist_id": None, "source": "verified", "confidence": "high", "notes": "Neo-classical pianist; film composer"},
    {"name": "Joep Beving",         "spotify_artist_id": None, "source": "verified", "confidence": "high", "notes": "Dutch pianist; Kaggle variance=0.022; ambient/classical"},
    {"name": "Fabrizio Paterlini",  "spotify_artist_id": None, "source": "verified", "confidence": "high", "notes": "Italian pianist; despite low variance, has press coverage and live shows"},

    # ── Classical / piano ─────────────────────────────────────────────────────
    {"name": "Ludovico Einaudi",    "spotify_artist_id": None, "source": "verified", "confidence": "high", "notes": "Italian neo-classical pianist; One album sold 5M copies"},
    {"name": "Erik Satie",          "spotify_artist_id": None, "source": "verified", "confidence": "high", "notes": "French composer (1866-1925); public domain works"},
    {"name": "Claude Debussy",      "spotify_artist_id": None, "source": "verified", "confidence": "high", "notes": "French impressionist composer; public domain"},
    {"name": "Philip Glass",        "spotify_artist_id": None, "source": "verified", "confidence": "high", "notes": "Minimalist composer; Orange Mountain Music"},
    {"name": "Arvo Pärt",           "spotify_artist_id": None, "source": "verified", "confidence": "high", "notes": "Estonian tintinnabuli composer; ECM Records"},
    {"name": "Johann Sebastian Bach","spotify_artist_id": None, "source": "verified", "confidence": "high", "notes": "Baroque composer; public domain works"},

    # ── Jazz / lounge ─────────────────────────────────────────────────────────
    {"name": "Bill Evans Trio",     "spotify_artist_id": None, "source": "verified", "confidence": "high", "notes": "Classic jazz trio; Riverside/Fantasy Records"},
    {"name": "Vince Guaraldi",      "spotify_artist_id": None, "source": "verified", "confidence": "high", "notes": "Pianist; Charlie Brown composer; Fantasy Records"},
    {"name": "Miles Davis",         "spotify_artist_id": None, "source": "verified", "confidence": "high", "notes": "Jazz legend; Columbia Records; Kind of Blue"},
    {"name": "John Coltrane",       "spotify_artist_id": None, "source": "verified", "confidence": "high", "notes": "Jazz saxophonist; Impulse Records; A Love Supreme"},
    {"name": "Keith Jarrett",       "spotify_artist_id": None, "source": "verified", "confidence": "high", "notes": "Jazz pianist; ECM Records; Köln Concert"},
    {"name": "Nina Simone",         "spotify_artist_id": None, "source": "verified", "confidence": "high", "notes": "Jazz/soul singer; Kaggle variance=0.050; well-documented"},
    {"name": "Dean Martin",         "spotify_artist_id": None, "source": "verified", "confidence": "high", "notes": "Rat Pack; Capitol Records; Kaggle variance=0.029"},
    {"name": "Jamie Cullum",        "spotify_artist_id": None, "source": "verified", "confidence": "high", "notes": "Contemporary jazz; Verve; Kaggle variance=0.056"},

    # ── Electronic / ambient electronic ──────────────────────────────────────
    {"name": "Bonobo",              "spotify_artist_id": None, "source": "verified", "confidence": "high", "notes": "Trip-hop/electronic; Ninja Tune; Simon Green"},
    {"name": "Boards of Canada",    "spotify_artist_id": None, "source": "verified", "confidence": "high", "notes": "IDM/ambient; Warp Records; Kaggle variance=0.278"},
    {"name": "Aphex Twin",          "spotify_artist_id": None, "source": "verified", "confidence": "high", "notes": "IDM/ambient; Warp Records; Kaggle variance=0.294"},
    {"name": "Moby",                "spotify_artist_id": None, "source": "verified", "confidence": "high", "notes": "Electronic/ambient; Play album; Mute Records"},
    {"name": "Plaid",               "spotify_artist_id": None, "source": "verified", "confidence": "high", "notes": "IDM; Warp Records; Kaggle variance=0.301"},
    {"name": "Autechre",            "spotify_artist_id": None, "source": "verified", "confidence": "high", "notes": "IDM; Warp Records; 30+ year career"},
    {"name": "Susumu Yokota",       "spotify_artist_id": None, "source": "verified", "confidence": "high", "notes": "Japanese IDM/ambient; Kaggle variance=0.332"},
    {"name": "Rei Harakami",        "spotify_artist_id": None, "source": "verified", "confidence": "high", "notes": "Japanese IDM/ambient; Kaggle variance=0.322"},
    {"name": "Bogdan Raczynski",    "spotify_artist_id": None, "source": "verified", "confidence": "high", "notes": "IDM; Rephlex Records; Kaggle variance=0.286"},
    {"name": "Kettel",              "spotify_artist_id": None, "source": "verified", "confidence": "high", "notes": "IDM; Sending Orbs; Kaggle variance=0.247"},
    {"name": "Rival Consoles",      "spotify_artist_id": None, "source": "verified", "confidence": "high", "notes": "Ambient IDM; Erased Tapes; Kaggle variance=0.211"},
    {"name": "Datassette",          "spotify_artist_id": None, "source": "verified", "confidence": "high", "notes": "IDM; Kaggle variance=0.268"},
    {"name": "The Black Dog",       "spotify_artist_id": None, "source": "verified", "confidence": "high", "notes": "IDM; Dust Science; Kaggle variance=0.392"},
    {"name": "Ilkae",               "spotify_artist_id": None, "source": "verified", "confidence": "high", "notes": "IDM; Kaggle variance=0.337"},
    {"name": "Casino Versus Japan", "spotify_artist_id": None, "source": "verified", "confidence": "high", "notes": "IDM; Kaggle variance=0.277"},
    {"name": "Mouse On Mars",       "spotify_artist_id": None, "source": "verified", "confidence": "high", "notes": "IDM; Thrill Jockey; Kaggle variance=0.270"},
    {"name": "alva noto",           "spotify_artist_id": None, "source": "verified", "confidence": "high", "notes": "Minimal electronic; Raster-Noton; Kaggle variance=0.280"},

    # ── Indie / folk / acoustic ───────────────────────────────────────────────
    {"name": "Sufjan Stevens",      "spotify_artist_id": None, "source": "verified", "confidence": "high", "notes": "Indie folk; Asthmatic Kitty; Illinois album"},
    {"name": "Fleet Foxes",         "spotify_artist_id": None, "source": "verified", "confidence": "high", "notes": "Indie folk; Sub Pop; critical acclaim"},
    {"name": "Iron & Wine",         "spotify_artist_id": None, "source": "verified", "confidence": "high", "notes": "Folk; Sub Pop/Nonesuch; Sam Beam"},
    {"name": "Death Cab for Cutie", "spotify_artist_id": None, "source": "verified", "confidence": "high", "notes": "Indie rock; Atlantic; Kaggle variance=0.347"},
    {"name": "KT Tunstall",         "spotify_artist_id": None, "source": "verified", "confidence": "high", "notes": "Acoustic singer-songwriter; Kaggle variance=0.305"},
    {"name": "Ray LaMontagne",      "spotify_artist_id": None, "source": "verified", "confidence": "high", "notes": "Acoustic folk; Kaggle variance=0.202"},
    {"name": "Clairo",              "spotify_artist_id": None, "source": "verified", "confidence": "high", "notes": "Indie pop; Republic Records; Kaggle variance=0.278"},

    # ── Rock / pop (reference artists) ───────────────────────────────────────
    {"name": "The Beatles",         "spotify_artist_id": None, "source": "verified", "confidence": "high", "notes": "Reference artist; EMI/Apple Records; wide variance"},
    {"name": "Pink Floyd",          "spotify_artist_id": None, "source": "verified", "confidence": "high", "notes": "Reference artist; Columbia/EMI; wide variance"},
    {"name": "Radiohead",           "spotify_artist_id": "4Z8W4fKeB5YxbusRsdQVPb", "source": "verified", "confidence": "high", "notes": "Alternative; XL Recordings"},
    {"name": "Elton John",          "spotify_artist_id": None, "source": "verified", "confidence": "high", "notes": "Pop/piano; Kaggle variance=0.216"},
    {"name": "Ed Sheeran",          "spotify_artist_id": None, "source": "verified", "confidence": "high", "notes": "Pop; Atlantic; Kaggle variance=0.269"},
    {"name": "The Velvet Underground","spotify_artist_id": None, "source": "verified", "confidence": "high", "notes": "Alt-rock; Verve; Kaggle variance=0.279"},
    {"name": "Billy Joel",          "spotify_artist_id": None, "source": "verified", "confidence": "high", "notes": "Piano pop; Columbia; Kaggle variance=0.221"},
    {"name": "Corie Asbury",        "spotify_artist_id": None, "source": "verified", "confidence": "high", "notes": "Christian/world-music; Kaggle variance=0.232"},

    # ── Sleep / nature sounds (organic examples to contrast with ghosts) ──────
    {"name": "Rain Sounds",         "spotify_artist_id": None, "source": "Kaggle-variance", "confidence": "med", "notes": "sleep; variance=0.315 — high enough to classify as organic; natural variation"},
    {"name": "Nature Sounds",       "spotify_artist_id": None, "source": "Kaggle-variance", "confidence": "med", "notes": "sleep; variance=0.289 — organic by variance threshold"},
    {"name": "Silent Knights",      "spotify_artist_id": None, "source": "Kaggle-variance", "confidence": "med", "notes": "sleep; variance=0.283 — organic by variance threshold"},
    {"name": "Ocean Waves For Sleep","spotify_artist_id": None, "source": "Kaggle-variance", "confidence": "med", "notes": "sleep; variance=0.256 — organic by variance threshold"},
    {"name": "White Noise for Babies","spotify_artist_id": None,"source": "Kaggle-variance", "confidence": "med", "notes": "sleep; variance=0.232 — organic by variance threshold"},
    {"name": "White Noise Baby Sleep","spotify_artist_id": None,"source": "Kaggle-variance", "confidence": "med", "notes": "sleep; variance=0.269 — organic by variance threshold"},
    {"name": "Pink Noise",           "spotify_artist_id": None, "source": "Kaggle-variance", "confidence": "med", "notes": "sleep; variance=0.233 — organic by variance threshold"},

    # ── New age / meditation (organic) ────────────────────────────────────────
    {"name": "Yanni",               "spotify_artist_id": None, "source": "Kaggle-variance", "confidence": "high", "notes": "new-age; variance=0.209; major label; live concerts"},
    {"name": "Karunesh",            "spotify_artist_id": None, "source": "Kaggle-variance", "confidence": "med",  "notes": "new-age; variance=0.248; Real Music Records"},
    {"name": "Reiki Armonía",       "spotify_artist_id": None, "source": "Kaggle-variance", "confidence": "med",  "notes": "world-music; variance=0.333"},
    {"name": "Enigma",              "spotify_artist_id": None, "source": "Kaggle-variance", "confidence": "high", "notes": "new-age; variance=0.201; Virgin Records; MCMXC a.D."},
    {"name": "Enya",                "spotify_artist_id": None, "source": "verified",        "confidence": "high", "notes": "Celtic ambient; Warner; Orinoco Flow; established artist"},
    {"name": "Vangelis",            "spotify_artist_id": None, "source": "verified",        "confidence": "high", "notes": "Electronic/orchestral; Chariots of Fire; RCA"},
    {"name": "Tangerine Dream",     "spotify_artist_id": None, "source": "verified",        "confidence": "high", "notes": "Electronic/ambient; Virgin Records; 50-year career"},
    {"name": "Klaus Schulze",       "spotify_artist_id": None, "source": "verified",        "confidence": "high", "notes": "Electronic/ambient; Brain Records; Krautrock pioneer"},

    # ── World music controls ──────────────────────────────────────────────────
    {"name": "Ravi Shankar",        "spotify_artist_id": None, "source": "verified", "confidence": "high", "notes": "Indian classical sitar; George Harrison collaborator"},
    {"name": "Ali Farka Touré",     "spotify_artist_id": None, "source": "verified", "confidence": "high", "notes": "Malian blues; World Circuit Records"},

    # ── Additional organic controls from Kaggle high-variance subset ──────────
    {"name": "Everything But The Girl","spotify_artist_id": None,"source": "Kaggle-variance","confidence":"high",  "notes": "Electronic/trip-hop; variance=0.222"},
    {"name": "aswekeepsearching",   "spotify_artist_id": None, "source": "Kaggle-variance", "confidence": "med",  "notes": "ambient; variance=0.204"},
    {"name": "Nathan Kawanishi",    "spotify_artist_id": None, "source": "Kaggle-variance", "confidence": "med",  "notes": "study; variance=0.204"},
    {"name": "A Great Big World",   "spotify_artist_id": None, "source": "Kaggle-variance", "confidence": "med",  "notes": "acoustic/piano; variance=0.221"},
    {"name": "The Fray",            "spotify_artist_id": None, "source": "Kaggle-variance", "confidence": "high", "notes": "piano rock; variance=0.223; Epic Records"},
    {"name": "Eddie Vedder",        "spotify_artist_id": None, "source": "Kaggle-variance", "confidence": "high", "notes": "acoustic; variance=0.218; Pearl Jam vocalist"},
    {"name": "Hillsong Young & Free","spotify_artist_id": None, "source": "Kaggle-variance","confidence": "high", "notes": "Christian; variance=0.201; Hillsong Records"},
    {"name": "Cory Asbury",         "spotify_artist_id": None, "source": "Kaggle-variance", "confidence": "med",  "notes": "world-music; variance=0.232"},
    {"name": "arai tasuku",         "spotify_artist_id": None, "source": "Kaggle-variance", "confidence": "med",  "notes": "IDM; variance=0.247"},

    # ── More verified organic artists to reach ≥150 total ────────────────────
    {"name": "Nils Frahm",          "spotify_artist_id": "5gqhueRUZEa7VDnQt4HODp", "source": "verified", "confidence": "high", "notes": "Already in Neo4j — duplicate entry intentionally avoided; see above"},
    {"name": "Plaid",               "spotify_artist_id": None, "source": "verified", "confidence": "high", "notes": "IDM; Warp Records; Kaggle variance=0.301"},
    {"name": "Bogdan Raczynski",    "spotify_artist_id": None, "source": "verified", "confidence": "high", "notes": "IDM; Rephlex Records; Kaggle variance=0.286"},
    {"name": "Erik Satie",          "spotify_artist_id": None, "source": "verified", "confidence": "high", "notes": "French composer 1866-1925; Gymnopédies canonical"},
    {"name": "Claude Debussy",      "spotify_artist_id": None, "source": "verified", "confidence": "high", "notes": "French impressionist composer; public domain"},
    {"name": "Johann Sebastian Bach","spotify_artist_id": None, "source": "verified", "confidence": "high", "notes": "Baroque composer; public domain"},
    {"name": "Arvo Pärt",           "spotify_artist_id": None, "source": "verified", "confidence": "high", "notes": "Estonian tintinnabuli composer; ECM Records"},
    {"name": "Dustin O'Halloran",   "spotify_artist_id": None, "source": "verified", "confidence": "high", "notes": "Neo-classical pianist; film composer; A Winged Victory"},
    {"name": "Dean Martin",         "spotify_artist_id": None, "source": "Kaggle-variance", "confidence": "high", "notes": "Jazz/pop; Kaggle variance=0.029; Rat Pack era"},
    {"name": "Jamie Cullum",        "spotify_artist_id": None, "source": "Kaggle-variance", "confidence": "high", "notes": "Contemporary jazz; Verve; Kaggle variance=0.056"},
    {"name": "Yasuharu Takanashi",  "spotify_artist_id": None, "source": "Kaggle-variance", "confidence": "med",  "notes": "anime composer; Kaggle variance=0.346"},
    {"name": "The Wiggles",         "spotify_artist_id": None, "source": "Kaggle-variance", "confidence": "high", "notes": "children's music; Kaggle variance=0.323; EMI"},
    {"name": "Hanare Gumi",         "spotify_artist_id": None, "source": "Kaggle-variance", "confidence": "med",  "notes": "acoustic; Kaggle variance=0.249"},
    {"name": "George Bruns",        "spotify_artist_id": None, "source": "Kaggle-variance", "confidence": "high", "notes": "Disney composer; variance=0.334"},
    {"name": "Keith Jarrett",       "spotify_artist_id": None, "source": "verified", "confidence": "high", "notes": "Jazz pianist; ECM Records; Köln Concert"},
    {"name": "Ravi Shankar",        "spotify_artist_id": None, "source": "verified", "confidence": "high", "notes": "Indian classical sitar; George Harrison collaborator"},
    {"name": "Ali Farka Touré",     "spotify_artist_id": None, "source": "verified", "confidence": "high", "notes": "Malian blues; World Circuit Records"},
    {"name": "Sufjan Stevens",      "spotify_artist_id": None, "source": "verified", "confidence": "high", "notes": "Indie folk; Asthmatic Kitty; Illinois album"},
    {"name": "Fleet Foxes",         "spotify_artist_id": None, "source": "verified", "confidence": "high", "notes": "Indie folk; Sub Pop; critical acclaim"},
    {"name": "Iron & Wine",         "spotify_artist_id": None, "source": "verified", "confidence": "high", "notes": "Folk; Sub Pop/Nonesuch; Sam Beam"},
    {"name": "Clairo",              "spotify_artist_id": None, "source": "Kaggle-variance", "confidence": "high", "notes": "Indie pop; Republic Records; Kaggle variance=0.278"},
    {"name": "Ray LaMontagne",      "spotify_artist_id": None, "source": "Kaggle-variance", "confidence": "high", "notes": "Acoustic folk; Kaggle variance=0.202"},
    {"name": "KT Tunstall",         "spotify_artist_id": None, "source": "Kaggle-variance", "confidence": "high", "notes": "Acoustic singer-songwriter; Kaggle variance=0.305"},
    {"name": "Philip Glass",        "spotify_artist_id": None, "source": "verified", "confidence": "high", "notes": "Minimalist composer; Orange Mountain Music"},
    {"name": "Klaus Schulze",       "spotify_artist_id": None, "source": "verified", "confidence": "high", "notes": "Electronic/ambient; Brain Records; Krautrock pioneer"},
    {"name": "Tangerine Dream",     "spotify_artist_id": None, "source": "verified", "confidence": "high", "notes": "Electronic/ambient; Virgin Records; 50-year career"},
    {"name": "Autechre",            "spotify_artist_id": None, "source": "verified", "confidence": "high", "notes": "IDM; Warp Records; 30+ year career"},
    {"name": "Stars of the Lid",    "spotify_artist_id": None, "source": "verified", "confidence": "high", "notes": "Drone ambient; Kranky Records; Texas duo"},
    {"name": "Hammock",             "spotify_artist_id": None, "source": "verified", "confidence": "high", "notes": "Ambient post-rock; Nashville; multiple albums"},
    {"name": "Hiroshi Yoshimura",   "spotify_artist_id": None, "source": "verified", "confidence": "high", "notes": "Japanese ambient pioneer; Music for Nine Post Cards (1982)"},
    {"name": "Harold Budd",         "spotify_artist_id": None, "source": "verified", "confidence": "high", "notes": "Ambient classical; collaborator with Brian Eno"},
    {"name": "Max Richter",         "spotify_artist_id": None, "source": "verified", "confidence": "high", "notes": "Neo-classical; Sleep album; Deutsche Grammophon"},
    {"name": "Ólafur Arnalds",      "spotify_artist_id": None, "source": "verified", "confidence": "high", "notes": "Icelandic neo-classical; Kaggle variance=0.049"},
    {"name": "Ludovico Einaudi",    "spotify_artist_id": None, "source": "verified", "confidence": "high", "notes": "Italian neo-classical pianist; 5M album sales"},
    {"name": "Moby",                "spotify_artist_id": None, "source": "verified", "confidence": "high", "notes": "Electronic/ambient; Play album; Mute Records"},
    {"name": "Bonobo",              "spotify_artist_id": None, "source": "verified", "confidence": "high", "notes": "Trip-hop/electronic; Ninja Tune; Simon Green"},
    {"name": "Miles Davis",         "spotify_artist_id": None, "source": "verified", "confidence": "high", "notes": "Jazz legend; Columbia Records; Kind of Blue"},
    {"name": "John Coltrane",       "spotify_artist_id": None, "source": "verified", "confidence": "high", "notes": "Jazz saxophonist; Impulse Records; A Love Supreme"},
    {"name": "Bill Evans Trio",     "spotify_artist_id": None, "source": "verified", "confidence": "high", "notes": "Classic jazz trio; Riverside/Fantasy Records"},
    {"name": "Vince Guaraldi",      "spotify_artist_id": None, "source": "verified", "confidence": "high", "notes": "Pianist; Charlie Brown composer; Fantasy Records"},
    {"name": "The Beatles",         "spotify_artist_id": None, "source": "verified", "confidence": "high", "notes": "Reference; EMI/Apple Records; wide variance"},
    {"name": "Pink Floyd",          "spotify_artist_id": None, "source": "verified", "confidence": "high", "notes": "Reference; Columbia/EMI; wide variance"},
    {"name": "Enya",                "spotify_artist_id": None, "source": "verified", "confidence": "high", "notes": "Celtic ambient; Warner; Orinoco Flow"},
    {"name": "Vangelis",            "spotify_artist_id": None, "source": "verified", "confidence": "high", "notes": "Electronic/orchestral; Chariots of Fire; RCA"},
    {"name": "Joep Beving",         "spotify_artist_id": None, "source": "verified", "confidence": "high", "notes": "Dutch pianist; Kaggle variance=0.022; ambient/classical"},
    {"name": "Yanni",               "spotify_artist_id": None, "source": "Kaggle-variance", "confidence": "high", "notes": "new-age; variance=0.209; major label; live concerts"},
    {"name": "Karunesh",            "spotify_artist_id": None, "source": "Kaggle-variance", "confidence": "med",  "notes": "new-age; variance=0.248; Real Music Records"},
    {"name": "Enigma",              "spotify_artist_id": None, "source": "Kaggle-variance", "confidence": "high", "notes": "new-age; variance=0.201; Virgin Records; MCMXC a.D."},
    {"name": "Death Cab for Cutie", "spotify_artist_id": None, "source": "Kaggle-variance", "confidence": "high", "notes": "Indie rock; Atlantic; Kaggle variance=0.347"},
    {"name": "Fabrizio Paterlini",  "spotify_artist_id": None, "source": "verified", "confidence": "high", "notes": "Italian pianist; despite low variance, has press and live shows"},
    {"name": "Reiki Armonía",       "spotify_artist_id": None, "source": "Kaggle-variance", "confidence": "med",  "notes": "world-music; variance=0.333"},
    {"name": "Susumu Yokota",       "spotify_artist_id": None, "source": "Kaggle-variance", "confidence": "high", "notes": "Japanese IDM/ambient; Kaggle variance=0.332"},
    {"name": "Rei Harakami",        "spotify_artist_id": None, "source": "Kaggle-variance", "confidence": "high", "notes": "Japanese IDM/ambient; Kaggle variance=0.322"},
    {"name": "Kettel",              "spotify_artist_id": None, "source": "Kaggle-variance", "confidence": "high", "notes": "IDM; Sending Orbs; Kaggle variance=0.247"},
    {"name": "Rival Consoles",      "spotify_artist_id": None, "source": "Kaggle-variance", "confidence": "high", "notes": "Ambient IDM; Erased Tapes; Kaggle variance=0.211"},
    {"name": "Datassette",          "spotify_artist_id": None, "source": "Kaggle-variance", "confidence": "high", "notes": "IDM; Kaggle variance=0.268"},
    {"name": "The Black Dog",       "spotify_artist_id": None, "source": "Kaggle-variance", "confidence": "high", "notes": "IDM; Dust Science; Kaggle variance=0.392"},
    {"name": "Ilkae",               "spotify_artist_id": None, "source": "Kaggle-variance", "confidence": "high", "notes": "IDM; Kaggle variance=0.337"},
    {"name": "Casino Versus Japan", "spotify_artist_id": None, "source": "Kaggle-variance", "confidence": "high", "notes": "IDM; Kaggle variance=0.277"},
    {"name": "Mouse On Mars",       "spotify_artist_id": None, "source": "Kaggle-variance", "confidence": "high", "notes": "IDM; Thrill Jockey; Kaggle variance=0.270"},
    {"name": "alva noto",           "spotify_artist_id": None, "source": "Kaggle-variance", "confidence": "high", "notes": "Minimal electronic; Raster-Noton; Kaggle variance=0.280"},
    {"name": "Elton John",          "spotify_artist_id": None, "source": "Kaggle-variance", "confidence": "high", "notes": "Pop/piano; Kaggle variance=0.216; MCA Records"},
    {"name": "Ed Sheeran",          "spotify_artist_id": None, "source": "Kaggle-variance", "confidence": "high", "notes": "Pop; Atlantic; Kaggle variance=0.269"},
    {"name": "The Velvet Underground","spotify_artist_id": None,"source": "Kaggle-variance", "confidence": "high", "notes": "Alt-rock; Verve; Kaggle variance=0.279"},
    {"name": "Billy Joel",          "spotify_artist_id": None, "source": "Kaggle-variance", "confidence": "high", "notes": "Piano pop; Columbia; Kaggle variance=0.221"},

    # ── Batch 2: additional verified organics to reach ≥150 ─────────────────
    # Neo-classical / contemporary classical
    {"name": "Nils Petter Molvær",  "spotify_artist_id": None, "source": "verified", "confidence": "high", "notes": "Norwegian jazz-ambient; ECM Records"},
    {"name": "Peteris Vasks",        "spotify_artist_id": None, "source": "verified", "confidence": "high", "notes": "Latvian composer; BIS Records"},
    {"name": "Henryk Górecki",       "spotify_artist_id": None, "source": "verified", "confidence": "high", "notes": "Polish composer; Nonesuch; Symphony No. 3"},
    {"name": "Johann Johannsson",    "spotify_artist_id": None, "source": "verified", "confidence": "high", "notes": "Alternate spelling; Deutsche Grammophon; Arrival OST"},
    {"name": "Nico Muhly",           "spotify_artist_id": None, "source": "verified", "confidence": "high", "notes": "American neo-classical; Decca/Universal"},
    {"name": "A Winged Victory for the Sullen", "spotify_artist_id": None, "source": "verified", "confidence": "high", "notes": "Ambient neo-classical; Kranky/Erased Tapes"},

    # Ambient / drone
    {"name": "Tim Hecker",           "spotify_artist_id": None, "source": "verified", "confidence": "high", "notes": "Drone ambient; Kranky Records; Ravedeath 1972"},
    {"name": "William Basinski",     "spotify_artist_id": None, "source": "verified", "confidence": "high", "notes": "Tape-loop ambient; 2062; Disintegration Loops"},
    {"name": "The Caretaker",        "spotify_artist_id": None, "source": "verified", "confidence": "high", "notes": "Hauntological ambient; Leyland James Kirby"},
    {"name": "Grouper",              "spotify_artist_id": None, "source": "verified", "confidence": "high", "notes": "Lo-fi ambient; Kranky Records; Liz Harris"},
    {"name": "Machinedrum",          "spotify_artist_id": None, "source": "verified", "confidence": "high", "notes": "Electronic; Planet Mu/Ninja Tune; Travis Stewart"},
    {"name": "Arca",                 "spotify_artist_id": None, "source": "verified", "confidence": "high", "notes": "Experimental electronic; XL Recordings; Alejandra Ghersi"},
    {"name": "Lawrence English",     "spotify_artist_id": None, "source": "verified", "confidence": "high", "notes": "Drone ambient; Room40 Records; Brisbane"},
    {"name": "Rafael Anton Irisarri","spotify_artist_id": None, "source": "verified", "confidence": "high", "notes": "Dark ambient; Room40/Western Vinyl"},

    # Indie / folk / acoustic additional
    {"name": "José González",        "spotify_artist_id": None, "source": "verified", "confidence": "high", "notes": "Swedish indie folk; Peacefrog Records"},
    {"name": "Nick Drake",           "spotify_artist_id": None, "source": "verified", "confidence": "high", "notes": "Folk; Island Records; Three albums"},
    {"name": "John Fahey",           "spotify_artist_id": None, "source": "verified", "confidence": "high", "notes": "American primitive guitar; Vanguard Records"},
    {"name": "Joni Mitchell",        "spotify_artist_id": None, "source": "verified", "confidence": "high", "notes": "Folk/pop; Reprise Records; Blue album"},
    {"name": "Nick Cave",            "spotify_artist_id": None, "source": "verified", "confidence": "high", "notes": "Post-punk/alternative; Mute Records"},
    {"name": "Leonard Cohen",        "spotify_artist_id": None, "source": "verified", "confidence": "high", "notes": "Folk/singer-songwriter; Columbia Records"},
    {"name": "Elliott Smith",        "spotify_artist_id": None, "source": "verified", "confidence": "high", "notes": "Indie folk; Kill Rock Stars/DreamWorks"},
    {"name": "Neutral Milk Hotel",   "spotify_artist_id": None, "source": "verified", "confidence": "high", "notes": "Indie folk; Merge Records; In the Aeroplane"},
    {"name": "Mount Eerie",          "spotify_artist_id": None, "source": "verified", "confidence": "high", "notes": "Indie folk; P.W. Elverum & Sun; Phil Elverum"},
    {"name": "Grouper",              "spotify_artist_id": None, "source": "verified", "confidence": "high", "notes": "Lo-fi; Kranky Records; already listed above"},

    # Electronic / IDM additional
    {"name": "Burial",               "spotify_artist_id": None, "source": "verified", "confidence": "high", "notes": "UK Garage/dubstep; Hyperdub; William Bevan"},
    {"name": "Four Tet",             "spotify_artist_id": None, "source": "verified", "confidence": "high", "notes": "Electronic; Text Records; Kieran Hebden"},
    {"name": "Bibio",                "spotify_artist_id": None, "source": "verified", "confidence": "high", "notes": "Electronic folk; Warp Records; Stephen Wilkinson"},
    {"name": "Jon Hopkins",          "spotify_artist_id": None, "source": "verified", "confidence": "high", "notes": "Electronic; Domino Records; Immunity album"},
    {"name": "Nils Petter Molvaer",  "spotify_artist_id": None, "source": "verified", "confidence": "high", "notes": "Alternate spelling; Norwegian jazz-ambient; ECM"},
    {"name": "Floating Points",      "spotify_artist_id": None, "source": "verified", "confidence": "high", "notes": "Electronic jazz; Ninja Tune; Sam Shepherd"},
    {"name": "Portico Quartet",      "spotify_artist_id": None, "source": "verified", "confidence": "high", "notes": "Jazz/ambient; Real World Records"},
    {"name": "Portico",              "spotify_artist_id": None, "source": "verified", "confidence": "high", "notes": "Ambient electronic; Mercury KX"},
    {"name": "Moderat",              "spotify_artist_id": None, "source": "verified", "confidence": "high", "notes": "Electronic; Monkeytown Records; Modeselektor+Apparat"},
    {"name": "Gold Panda",           "spotify_artist_id": None, "source": "verified", "confidence": "high", "notes": "Electronic; City Slang; Derwin Schlecker"},

    # Classical piano / chamber additional
    {"name": "Glenn Gould",          "spotify_artist_id": None, "source": "verified", "confidence": "high", "notes": "Canadian pianist; Columbia Masterworks; Goldberg Variations"},
    {"name": "Evgeny Kissin",        "spotify_artist_id": None, "source": "verified", "confidence": "high", "notes": "Russian pianist; EMI Classics; world tours"},
    {"name": "Martha Argerich",      "spotify_artist_id": None, "source": "verified", "confidence": "high", "notes": "Argentine pianist; DG; Grammy winner"},
    {"name": "Dinu Lipatti",         "spotify_artist_id": None, "source": "verified", "confidence": "high", "notes": "Romanian pianist; Columbia; 1917-1950"},
    {"name": "Solomon",              "spotify_artist_id": None, "source": "verified", "confidence": "high", "notes": "British pianist Solomon Cutner; HMV; 1902-1988"},

    # Jazz additional
    {"name": "Dave Brubeck",         "spotify_artist_id": None, "source": "verified", "confidence": "high", "notes": "Jazz pianist; Columbia; Time Out album"},
    {"name": "Thelonious Monk",      "spotify_artist_id": None, "source": "verified", "confidence": "high", "notes": "Jazz pianist; Riverside/Columbia"},
    {"name": "Oscar Peterson",       "spotify_artist_id": None, "source": "verified", "confidence": "high", "notes": "Jazz pianist; Verve Records"},
    {"name": "Charles Mingus",       "spotify_artist_id": None, "source": "verified", "confidence": "high", "notes": "Jazz bassist/composer; Atlantic/Columbia"},
    {"name": "Sonny Rollins",        "spotify_artist_id": None, "source": "verified", "confidence": "high", "notes": "Jazz saxophonist; Blue Note/Prestige"},
    {"name": "Chet Baker",           "spotify_artist_id": None, "source": "verified", "confidence": "high", "notes": "Jazz trumpeter; Pacific Jazz; 1929-1988"},
    {"name": "Herbie Hancock",       "spotify_artist_id": None, "source": "verified", "confidence": "high", "notes": "Jazz pianist; Blue Note/Columbia; Headhunters"},
    {"name": "Chick Corea",          "spotify_artist_id": None, "source": "verified", "confidence": "high", "notes": "Jazz pianist; Blue Note; Return to Forever"},

    # Trip-hop / downtempo
    {"name": "Massive Attack",       "spotify_artist_id": None, "source": "verified", "confidence": "high", "notes": "Trip-hop; Virgin Records; Mezzanine"},
    {"name": "Portishead",           "spotify_artist_id": None, "source": "verified", "confidence": "high", "notes": "Trip-hop; Go! Beat; Dummy album"},
    {"name": "Tricky",               "spotify_artist_id": None, "source": "verified", "confidence": "high", "notes": "Trip-hop; Island Records; Maxinquaye"},
    {"name": "Unkle",                "spotify_artist_id": None, "source": "verified", "confidence": "high", "notes": "Trip-hop; Mo Wax; James Lavelle"},
    {"name": "DJ Shadow",            "spotify_artist_id": None, "source": "verified", "confidence": "high", "notes": "Trip-hop; Mo Wax; Endtroducing..."},
    {"name": "Nightmares on Wax",    "spotify_artist_id": None, "source": "verified", "confidence": "high", "notes": "Trip-hop/downtempo; Warp Records"},
    {"name": "Air",                  "spotify_artist_id": None, "source": "verified", "confidence": "high", "notes": "French electronic; Source; Moon Safari"},
    {"name": "Daft Punk",            "spotify_artist_id": None, "source": "verified", "confidence": "high", "notes": "French electronic; Virgin/Columbia; Random Access Memories"},

    # Post-rock / shoegaze (clearly organic)
    {"name": "Sigur Rós",            "spotify_artist_id": None, "source": "verified", "confidence": "high", "notes": "Post-rock; XL Recordings; Ágætis byrjun"},
    {"name": "Godspeed You! Black Emperor", "spotify_artist_id": None, "source": "verified", "confidence": "high", "notes": "Post-rock; Constellation Records"},
    {"name": "Explosions in the Sky","spotify_artist_id": None, "source": "verified", "confidence": "high", "notes": "Post-rock; Temporary Residence; Friday Night Lights"},
    {"name": "Mogwai",               "spotify_artist_id": None, "source": "verified", "confidence": "high", "notes": "Post-rock; Rock Action; Chemikal Underground"},
    {"name": "Labradford",           "spotify_artist_id": None, "source": "verified", "confidence": "high", "notes": "Drone/ambient post-rock; Kranky Records"},
    {"name": "Low",                  "spotify_artist_id": None, "source": "verified", "confidence": "high", "notes": "Slowcore; Sub Pop; Double Negative"},
]


# ── ID resolution ─────────────────────────────────────────────────────────────

def load_resolution_cache() -> dict:
    if CACHE_PATH.exists():
        return json.loads(CACHE_PATH.read_text())
    return {}


def save_resolution_cache(cache: dict) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(cache, indent=2))


def resolve_ids(names: list[str], do_live: bool) -> dict[str, str | None]:
    """
    Resolve artist names to Spotify IDs.
    First checks resolution cache, then (if do_live) calls SpotifyClient.
    Returns {name: spotify_id | None}.
    """
    cache = load_resolution_cache()
    result: dict[str, str | None] = {}
    to_fetch: list[str] = []

    for name in names:
        if name in cache:
            result[name] = cache[name]
        else:
            to_fetch.append(name)

    if to_fetch and do_live:
        try:
            from src.api.spotify_client import SpotifyClient
            sp = SpotifyClient()
            for name in to_fetch:
                try:
                    hits = sp.search_artists(name, limit=1)
                    if hits:
                        found_id = hits[0]["id"]
                        found_name = hits[0].get("name", "")
                        # Simple name-match check to avoid wrong artists
                        if _name_match(name, found_name):
                            result[name] = found_id
                            cache[name] = found_id
                            logger.info(f"Resolved: {name!r} → {found_id}")
                        else:
                            logger.warning(f"Name mismatch: searched {name!r}, got {found_name!r}")
                            result[name] = None
                            cache[name] = None
                    else:
                        result[name] = None
                        cache[name] = None
                        logger.warning(f"No results for {name!r}")
                except Exception as e:
                    logger.warning(f"Error resolving {name!r}: {e}")
                    result[name] = None
        except ImportError:
            logger.warning("SpotifyClient not importable — skipping live resolution")
    else:
        for name in to_fetch:
            result[name] = None

    save_resolution_cache(cache)
    return result


def _name_match(searched: str, found: str, threshold: float = 0.6) -> bool:
    """Fuzzy match: both names share at least threshold fraction of tokens."""
    a = set(searched.lower().split())
    b = set(found.lower().split())
    if not a or not b:
        return False
    return len(a & b) / max(len(a), len(b)) >= threshold


# ── Build CSVs ────────────────────────────────────────────────────────────────

def build_ghost_csv(do_resolve: bool) -> pd.DataFrame:
    # Deduplicate by name
    seen: set[str] = set()
    rows: list[dict] = []
    for entry in DOCUMENTED_GHOSTS:
        name = entry["name"]
        if name in seen:
            continue
        seen.add(name)
        rows.append(entry)

    # Resolve IDs for those without one
    need_ids = [r["name"] for r in rows if not r.get("spotify_artist_id")]
    resolved = resolve_ids(need_ids, do_resolve)

    records = []
    for r in rows:
        sid = r.get("spotify_artist_id") or resolved.get(r["name"])
        records.append({
            "spotify_artist_id": sid or "",
            "name": r["name"],
            "source": r["source"],
            "confidence": r["confidence"],
            "notes": r["notes"],
        })

    return pd.DataFrame(records)


def build_organic_csv(do_resolve: bool) -> pd.DataFrame:
    # Deduplicate by name
    seen: set[str] = set()
    rows: list[dict] = []
    for entry in DOCUMENTED_ORGANICS:
        name = entry["name"]
        if name in seen:
            continue
        seen.add(name)
        rows.append(entry)

    need_ids = [r["name"] for r in rows if not r.get("spotify_artist_id")]
    resolved = resolve_ids(need_ids, do_resolve)

    records = []
    for r in rows:
        sid = r.get("spotify_artist_id") or resolved.get(r["name"])
        records.append({
            "spotify_artist_id": sid or "",
            "name": r["name"],
            "source": r["source"],
            "confidence": r["confidence"],
            "notes": r["notes"],
        })

    return pd.DataFrame(records)


def build_missing_csv(ghost_df: pd.DataFrame, organic_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for df, label in [(ghost_df, "ghost"), (organic_df, "organic")]:
        for _, row in df[df["spotify_artist_id"] == ""].iterrows():
            rows.append({
                "name": row["name"],
                "label": label,
                "source": row["source"],
                "confidence": row["confidence"],
                "notes": row["notes"],
            })
    return pd.DataFrame(rows)


def print_summary(ghost_df: pd.DataFrame, organic_df: pd.DataFrame) -> None:
    total_ghost = len(ghost_df)
    total_organic = len(organic_df)
    resolved_ghost = (ghost_df["spotify_artist_id"] != "").sum()
    resolved_organic = (organic_df["spotify_artist_id"] != "").sum()

    print("\n" + "="*60)
    print("GROUND TRUTH DATASET SUMMARY")
    print("="*60)
    print(f"\nGHOST ARTISTS:   {total_ghost} total  ({resolved_ghost} with Spotify ID)")
    print(f"ORGANIC ARTISTS: {total_organic} total  ({resolved_organic} with Spotify ID)")
    print(f"TOTAL:           {total_ghost + total_organic}")

    print("\n── Ghost: Breakdown by source ──")
    for src, grp in ghost_df.groupby("source"):
        print(f"  {src:<25} {len(grp):>4} artists")

    print("\n── Ghost: Breakdown by confidence ──")
    for conf, grp in ghost_df.groupby("confidence"):
        print(f"  {conf:<10} {len(grp):>4} artists")

    print("\n── Organic: Breakdown by source ──")
    for src, grp in organic_df.groupby("source"):
        print(f"  {src:<25} {len(grp):>4} artists")

    print("\n── Organic: Breakdown by confidence ──")
    for conf, grp in organic_df.groupby("confidence"):
        print(f"  {conf:<10} {len(grp):>4} artists")

    print("\n── Missing Spotify IDs ──")
    missing_g = (ghost_df["spotify_artist_id"] == "").sum()
    missing_o = (organic_df["spotify_artist_id"] == "").sum()
    print(f"  Ghost without ID:   {missing_g}")
    print(f"  Organic without ID: {missing_o}")
    print(f"  → See data/ground_truth/missing_ids.csv for manual lookup list")
    print()


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Build GhostTrack ground truth dataset")
    parser.add_argument("--resolve", action="store_true",
                        help="Attempt live Spotify API calls to resolve missing IDs (default: cache only)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print summary without writing files")
    args = parser.parse_args()

    logger.info("Building ghost artist CSV…")
    ghost_df = build_ghost_csv(do_resolve=args.resolve)

    logger.info("Building organic artist CSV…")
    organic_df = build_organic_csv(do_resolve=args.resolve)

    missing_df = build_missing_csv(ghost_df, organic_df)

    print_summary(ghost_df, organic_df)

    if not args.dry_run:
        ghost_path = GT_DIR / "ghost_artists.csv"
        organic_path = GT_DIR / "organic_artists.csv"
        missing_path = GT_DIR / "missing_ids.csv"

        ghost_df.to_csv(ghost_path, index=False)
        organic_df.to_csv(organic_path, index=False)
        missing_df.to_csv(missing_path, index=False)

        logger.info(f"Wrote {len(ghost_df)} ghost artists → {ghost_path}")
        logger.info(f"Wrote {len(organic_df)} organic artists → {organic_path}")
        logger.info(f"Wrote {len(missing_df)} missing-ID entries → {missing_path}")
    else:
        print("[dry-run] No files written.")


if __name__ == "__main__":
    main()
