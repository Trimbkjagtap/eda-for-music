"""
EDA for Music — PhD-Level Streamlit Dashboard
AI-Powered Ghost Artist Detection via Public API Analysis
"""
import streamlit as st
import os
import sys
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EDA for Music",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    [data-testid="stSidebar"] {
        background-color: #0f0f1a;
        border-right: 1px solid #2a2a4a;
    }
    [data-testid="stSidebar"] .stRadio label { color: #ccc; font-size: 15px; }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 { color: #a78bfa; }
    .main .block-container { background-color: #12121f; padding-top: 1.5rem; }
    h1 { color: #a78bfa; font-weight: 800; }
    h2 { color: #818cf8; }
    h3 { color: #94a3b8; }
    .metric-card {
        background: linear-gradient(135deg, #1e1e3f 0%, #16213e 100%);
        border: 1px solid #2a2a5a; border-radius: 12px;
        padding: 20px; text-align: center;
    }
    .metric-card .number { font-size: 2.4rem; font-weight: 800; color: #a78bfa; }
    .metric-card .label { font-size: 0.85rem; color: #94a3b8; margin-top: 4px; }
    .signal-card {
        background: #1a1a2e; border-left: 4px solid #a78bfa;
        border-radius: 8px; padding: 14px 18px; margin: 6px 0;
    }
    .signal-card .score { font-size: 1.5rem; font-weight: 700; color: #f59e0b; float: right; }
    .signal-card .name { color: #e2e8f0; font-weight: 600; }
    .signal-card .desc { color: #94a3b8; font-size: 0.82rem; margin-top: 3px; }
    .layer-card {
        background: #1e1e3f; border: 1px solid #2a2a5a;
        border-radius: 8px; padding: 12px 16px; margin: 5px 0;
    }
    .fig-frame {
        background: #1a1a2e; border: 1px solid #2a2a5a;
        border-radius: 10px; padding: 16px; margin-bottom: 16px;
    }
    .tag {
        display: inline-block; background: #312e81; color: #a5b4fc;
        border-radius: 20px; padding: 2px 12px; font-size: 0.75rem; margin: 2px;
    }
    .verdict-ghost {
        background: #7f1d1d; color: #fca5a5; border-radius: 8px;
        padding: 10px 20px; font-size: 1.3rem; font-weight: 800; text-align: center; display: block;
    }
    .verdict-suspicious {
        background: #78350f; color: #fcd34d; border-radius: 8px;
        padding: 10px 20px; font-size: 1.3rem; font-weight: 800; text-align: center; display: block;
    }
    .verdict-organic {
        background: #14532d; color: #86efac; border-radius: 8px;
        padding: 10px 20px; font-size: 1.3rem; font-weight: 800; text-align: center; display: block;
    }
    .ai-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #0f2027 100%);
        border: 1px solid #4c1d95; border-radius: 12px; padding: 20px; margin: 12px 0;
    }
    .chat-user {
        background: #1e1e3f; border-radius: 12px 12px 4px 12px;
        padding: 12px 16px; margin: 8px 0; color: #e2e8f0;
    }
    .chat-assistant {
        background: #0f2027; border: 1px solid #4c1d95; border-radius: 12px 12px 12px 4px;
        padding: 12px 16px; margin: 8px 0; color: #e2e8f0;
    }
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
FIGURES_DIR = ROOT / "paper" / "figures"
PAPER_DIR = ROOT / "paper"
PAPER_DIR.mkdir(exist_ok=True)

FIGURES_META = {
    "fig1_catalog_coherence.png": {
        "title": "Figure 1: Catalog Coherence in Audio Feature Space",
        "caption": "PCA projection of per-track audio features for ghost-like artists (left) vs organic controls (right). Ghost-like artists cluster into tight, compact ellipses. Ellipses = 1σ contour. Data: Kaggle 114K-track dataset.",
        "exercise": "Exercise 1", "signal": "Catalog Variance",
    },
    "fig2_playlist_entropy.png": {
        "title": "Figure 2: Playlist Aesthetic Coherence",
        "caption": "Energy vs Valence scatter for three simulated playlist archetypes. TIGHT playlists (single genre) show low Shannon entropy — the hallmark of a fraud target zone.",
        "exercise": "Exercise 2", "signal": "Playlist Entropy",
    },
    "fig3_isrc_join.png": {
        "title": "Figure 3: Artist to Production Company Attribution via ISRC",
        "caption": "Bipartite graph connecting artists to production companies via ISRC prefix. 3 seed artists, 490 tracks, 8 production companies from Neo4j.",
        "exercise": "Exercise 3", "signal": "ISRC Attribution",
    },
    "fig4_bipartite_neighborhood.png": {
        "title": "Figure 4: Artist × Production Company Bipartite Neighborhood",
        "caption": "HHI scores: RWN=0.88, MRC=0.66, Calmo=0.54 — ghost artists show extreme ISRC concentration consistent with bulk-upload operations.",
        "exercise": "Exercise 4", "signal": "Graph Centrality",
    },
    "fig5_recommendation_walk.png": {
        "title": "Figure 5: Recommendation Walk — Release Cadence as Walk Closure Signal",
        "caption": "Ghost artists: RWN=81%, MRC=95% closure — nearly all tracks bulk-uploaded in daily batches. Organic control (Nils Frahm): 0% closure, median gap 105 days.",
        "exercise": "Exercise 5", "signal": "Release Cadence",
    },
    "fig6_signal_radar.png": {
        "title": "Figure 6: Seven-Signal Ghost Artist Detection Radar",
        "caption": "S2 Release Cadence, S4 Catalog Density, and S6 Graph/HHI are the most discriminative signals. RWN=353M YouTube views — ghost behavior is Spotify-stream-farming, not platform absence.",
        "exercise": "Exercise 6", "signal": "Aggregate Score",
    },
    "fig6b_signal_heatmap.png": {
        "title": "Figure 6b: Signal Report Card Heatmap",
        "caption": "Heatmap of all 7 signal scores across 4 artists. S2/S4/S6 cleanly separate ghost from organic with 0.3–0.7+ score gaps.",
        "exercise": "Exercise 6", "signal": "Aggregate Score",
    },
    "fig7_gnn_performance.png": {
        "title": "Figure 7: GNN Ghost Artist Detection Performance",
        "caption": "GAT vs GCN training curves, ROC, confusion matrix, feature importance. Dataset: 65 nodes (14 ghost, 51 organic), 692 edges, 8 node features.",
        "exercise": "Exercise 7", "signal": "GNN Model",
    },
}

FRAMEWORK_LAYERS = [
    {"num": "1", "name": "Catalog Coherence", "desc": "Audio feature variance per artist. Ghost artists show unnaturally low variance.", "data": "Kaggle + Spotify /tracks"},
    {"num": "2", "name": "Playlist Entropy", "desc": "Shannon entropy of playlist feature distributions. Fraud playlists cluster tight.", "data": "Spotify /playlists"},
    {"num": "3", "name": "ISRC Attribution", "desc": "Production company identification via ISRC prefix. Shared registrants reveal networks.", "data": "Spotify /tracks (external_ids)"},
    {"num": "4", "name": "Release Cadence", "desc": "Statistical analysis of release date spacing. Machine outputs have inhuman regularity.", "data": "Spotify /albums"},
    {"num": "5", "name": "Metadata Similarity", "desc": "NLP embedding of track/artist names. Ghost catalogs reuse names with minor variations.", "data": "Spotify metadata"},
    {"num": "6", "name": "Graph Centrality", "desc": "Co-appearance network analysis. Fraud artists cluster in isolated low-degree subgraphs.", "data": "Neo4j graph"},
    {"num": "7", "name": "Aggregate Score", "desc": "Weighted combination of signals 1–6 into a final ghost probability score.", "data": "All layers"},
]

KNOWN_ARTISTS = {
    "Relaxing White Noise (ghost)": "6bo3atMVp3qFECNALVwq9N",
    "Meditation Relax Club (ghost)": "3BqBPFLxBkzKQTkuBPGMNF",
    "Calmo (ghost candidate)": "4Wx3ZL6d6p1gVMtwQ2YWsz",
    "Nils Frahm (organic)": "5hVghJ3sCFHFJoLnSHySjL",
}

CROSS_PLATFORM_DATA = {
    "Relaxing White Noise": {"artist_id": "6bo3atMVp3qFECNALVwq9N", "youtube_views": 353_775_028, "youtube_channel": "Relaxing White Noise", "apple_music": True, "s7_score": 0.00, "verdict": "LIKELY_GHOST", "note": "353M YouTube views — ghost behavior is Spotify-specific, not cross-platform absence"},
    "Meditation Relax Club": {"artist_id": "3BqBPFLxBkzKQTkuBPGMNF", "youtube_views": 157_581_269, "youtube_channel": "Meditation Relax Club", "apple_music": True, "s7_score": 0.00, "verdict": "LIKELY_GHOST", "note": "157M YouTube views — major cross-platform presence alongside Spotify"},
    "Calmo": {"artist_id": "4Wx3ZL6d6p1gVMtwQ2YWsz", "youtube_views": 155, "youtube_channel": None, "apple_music": False, "s7_score": 0.48, "verdict": "SUSPICIOUS", "note": "155 YouTube views — truly invisible cross-platform, unlike RWN/MRC"},
    "Nils Frahm": {"artist_id": "5hVghJ3sCFHFJoLnSHySjL", "youtube_views": 9_107_596, "youtube_channel": "Nils Frahm", "apple_music": True, "s7_score": 0.00, "verdict": "LIKELY_ORGANIC", "note": "9M YouTube views with verified Apple Music — consistent organic cross-platform presence"},
}

PROJECT_CONTEXT = """
PROJECT: EDA for Music — Ghost Artist Detection on Spotify
COURSE: INFO 7390, Spring 2026

EXERCISE RESULTS:
- Ex1: Ghost catalog variance is 12.5x lower than organic (Levene's test p<0.001)
- Ex2: Shannon entropy: editorial=2.59 bits, fan=2.89 bits, ghost-suspect=2.51 bits
- Ex3: 8 production companies, 490 tracks, 0 cross-artist ISRC sharing in seed set
- Ex4: HHI: RWN=0.671 (score=0.716), MRC=0.515 (0.560), Calmo=0.452 (0.446), NF=0.000
- Ex5: Walk closure: RWN=81%, MRC=95%, Calmo=32%, Nils Frahm=0%
- Ex6: Signal scores computed; GNN trained (100% proof-of-concept accuracy)

SIGNAL SCORES (rule-based):
- Relaxing White Noise:   S2=0.419, S3=0.109, S4=0.757, S5=0.139, S6=0.716  → combined=0.771 (LIKELY_GHOST with GNN)
- Meditation Relax Club:  S2=0.587, S3=0.109, S4=0.536, S5=0.139, S6=0.560  → SUSPICIOUS
- Calmo:                  S2=0.232, S3=0.109, S4=0.138, S5=0.139, S6=0.446  → LIKELY_ORGANIC (rule-based)
- Nils Frahm:             S2=0.024, S3=0.109, S4=0.000, S5=0.139, S6=0.000  → LIKELY_ORGANIC

CROSS-PLATFORM (S7):
- Relaxing White Noise: 353,775,028 YouTube views — ghost behavior is SPOTIFY-SPECIFIC
- Meditation Relax Club: 157,581,269 YouTube views — major cross-platform presence
- Calmo: 155 YouTube views — truly invisible cross-platform
- Nils Frahm: 9,107,596 YouTube views — organic baseline

KEY INSIGHT: Ghost artists are NOT cross-platform invisible. RWN/MRC have 100M+ YouTube views.
Ghost behavior = Spotify economic exploitation, not content fabrication.

LIMITATIONS:
- Only 3 confirmed ghost artists labeled (journalist-derived)
- Spotify API restricted Feb 2026 (no audio features, no related artists, no followers)
- Kaggle ghost artists are 0% match for niche ambient genre
- GNN uses synthetic graph structure (real graph has 0 cross-artist edges)
- Results specific to ambient/relaxation genre
"""

# ── OpenAI Client ─────────────────────────────────────────────────────────────
@st.cache_resource
def get_openai_client():
    from openai import OpenAI
    return OpenAI()


# ── AI Helper Functions ───────────────────────────────────────────────────────

def ai_analyze_artist(artist_name: str, collected_data: dict) -> dict:
    """Call GPT-4o to analyze an artist for ghost behavior."""
    client = get_openai_client()
    system_prompt = """You are a PhD-level music information retrieval researcher specializing in streaming platform fraud detection. You analyze artists for signs of ghost artist behavior using a 7-signal detection framework.

Known baselines:
GHOST - Relaxing White Noise: S2=0.42, S4=0.76, S6=0.72, cadence_closure=81%, HHI=0.67, youtube_views=353M, apple_music=True
GHOST - Meditation Relax Club: S2=0.59, S4=0.54, S6=0.56, cadence_closure=95%, HHI=0.52, youtube_views=157M, apple_music=True
GHOST - Calmo: S2=0.23, S4=0.14, S6=0.45, cadence_closure=32%, HHI=0.45, youtube_views=155, apple_music=False
ORGANIC - Nils Frahm: S2=0.02, S4=0.00, S6=0.00, cadence_closure=0%, HHI=0.00, youtube_views=9M, apple_music=True

Signal definitions:
S1: Audio Fingerprint Similarity — catalog variance in feature space (low variance = suspicious)
S2: Release Cadence Synchrony — bulk-upload detection via same-day release clustering
S3: Playlist Co-occurrence Density — coordinated playlist placement patterns
S4: Catalog Density Anomaly — tracks_per_day upload velocity (>0.15/day = suspicious)
S5: Artist Metadata Similarity — ghost keyword patterns in names/genres
S6: Graph Density / HHI — ISRC production company concentration (HHI > 0.6 = suspicious)
S7: Cross-Platform Discrepancy — YouTube + Apple Music presence vs Spotify

CRITICAL INSIGHT: Cross-platform presence does NOT rule out ghost behavior. RWN has 353M YouTube views and is still a ghost artist. Ghost behavior in ambient music is SPOTIFY-ECONOMIC, not about fake identity.

Respond with valid JSON only, no markdown fences. Estimate all signal scores based on available data, reasoning from the artist's genre, release patterns, and cross-platform presence. Be bold with your estimates."""

    user_prompt = f"""Analyze this artist for ghost behavior:

Artist: {artist_name}
Available data: {json.dumps(collected_data, indent=2)}

Respond in this exact JSON format:
{{
    "ghost_probability": <integer 0-100>,
    "confidence": <integer 0-100>,
    "estimated_signals": {{
        "s1_audio": <float 0.0-1.0 or null>,
        "s2_cadence": <float 0.0-1.0 or null>,
        "s3_playlist": <float 0.0-1.0 or null>,
        "s4_catalog": <float 0.0-1.0 or null>,
        "s5_metadata": <float 0.0-1.0 or null>,
        "s6_graph": <float 0.0-1.0 or null>,
        "s7_crossplatform": <float 0.0-1.0 or null>
    }},
    "verdict": "LIKELY_GHOST or SUSPICIOUS or LIKELY_ORGANIC or INSUFFICIENT_DATA",
    "analysis": "<3-paragraph expert analysis: (1) what the data shows, (2) which signals are concerning vs clean, (3) final verdict with confidence level>",
    "comparison": "<how this artist compares to known ghost/organic baselines — cite specific numbers>",
    "missing_data": ["<what additional data would help>"],
    "key_signals": ["<top 2-3 most informative signals for this artist>"]
}}"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.3,
        max_tokens=2000,
    )
    raw = response.choices[0].message.content.strip()
    # Strip markdown fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw)


@st.cache_data(show_spinner=False)
def _load_kaggle_df():
    """Load kaggle/dataset.csv once and cache it for the session."""
    import pandas as pd
    return pd.read_csv(ROOT / "data" / "kaggle" / "dataset.csv", low_memory=False)


def search_kaggle_for_artist(artist_name: str) -> dict:
    """Search Kaggle dataset for an artist's tracks."""
    try:
        df = _load_kaggle_df()
        name_lower = artist_name.lower()
        mask = df["artists"].str.lower().str.contains(name_lower, na=False)
        matches = df[mask]
        if matches.empty:
            return {"found": False, "track_count": 0}
        genres = matches["track_genre"].value_counts().head(3).to_dict()
        return {
            "found": True,
            "track_count": len(matches),
            "genres": genres,
            "avg_popularity": round(float(matches["popularity"].mean()), 1),
            "avg_danceability": round(float(matches["danceability"].mean()), 3),
            "avg_energy": round(float(matches["energy"].mean()), 3),
            "avg_valence": round(float(matches["valence"].mean()), 3),
            "total_variance": round(float(matches[["danceability","energy","valence","acousticness"]].var().sum()), 4),
            "sample_tracks": matches["track_name"].head(5).tolist(),
        }
    except Exception as e:
        return {"found": False, "error": str(e)}


def smart_search(raw_query: str) -> dict:
    """
    Resolve any input — artist name, Spotify ID, track name, album name —
    to a canonical artist name plus match metadata.

    Returns:
      {
        "resolved_artist": str,          # name to use for downstream lookup
        "match_type": str,               # "spotify_id" | "neo4j_name" | "kaggle_artist"
                                         # | "kaggle_track" | "kaggle_album" | "raw"
        "match_label": str,              # human-readable match description
        "track_row": dict | None,        # set when matched via track name (audio features)
        "found": bool,
      }
    """
    q = raw_query.strip()
    q_lower = q.lower()

    # ── Step 1: Spotify ID (22-char alphanumeric) ─────────────────────────────
    import re
    if re.fullmatch(r"[A-Za-z0-9]{22}", q):
        try:
            from src.graph.neo4j_client import Neo4jClient
            rows = Neo4jClient().run(
                "MATCH (a:Artist {spotify_id: $id}) RETURN a.name AS name LIMIT 1", id=q
            )
            if rows:
                return {"resolved_artist": rows[0]["name"], "match_type": "spotify_id",
                        "match_label": f"Spotify ID → Artist: {rows[0]['name']}", "track_row": None, "found": True}
        except Exception:
            pass

    # ── Step 2: Neo4j artist name match ──────────────────────────────────────
    try:
        from src.graph.neo4j_client import Neo4jClient
        rows = Neo4jClient().run(
            "MATCH (a:Artist) WHERE toLower(a.name) CONTAINS toLower($q) RETURN a.name AS name LIMIT 1",
            q=q,
        )
        if rows:
            return {"resolved_artist": rows[0]["name"], "match_type": "neo4j_name",
                    "match_label": f"Artist name (Neo4j): {rows[0]['name']}", "track_row": None, "found": True}
    except Exception:
        pass

    # ── Steps 3a-3c: Kaggle dataset search ───────────────────────────────────
    try:
        df = _load_kaggle_df()

        # 3a: artist column — word-boundary match; pick the matching artist from
        #     the semicolon-separated list rather than always taking the first one
        mask_artist = df["artists"].str.lower().str.contains(
            r"(?<![a-z])" + re.escape(q_lower) + r"(?![a-z])", na=False, regex=True
        )
        if mask_artist.any():
            raw_artists = df.loc[mask_artist, "artists"].iloc[0]
            # Find which part of the ';'-separated list actually matches the query
            parts = [p.strip() for p in raw_artists.split(";")]
            artist_name = next(
                (p for p in parts if q_lower in p.lower()),
                parts[0],
            )
            return {"resolved_artist": artist_name, "match_type": "kaggle_artist",
                    "match_label": f"Artist name (Kaggle): {artist_name}", "track_row": None, "found": True}

        # 3b: track name column
        mask_track = df["track_name"].str.lower().str.contains(q_lower, na=False)
        if mask_track.any():
            row = df[mask_track].iloc[0]
            artist_name = str(row["artists"]).split(";")[0].strip()
            track_row = {
                "track_name": str(row["track_name"]),
                "album_name": str(row.get("album_name", "")),
                "danceability": float(row.get("danceability", 0)),
                "energy": float(row.get("energy", 0)),
                "valence": float(row.get("valence", 0)),
                "acousticness": float(row.get("acousticness", 0)),
                "popularity": int(row.get("popularity", 0)),
                "track_genre": str(row.get("track_genre", "")),
            }
            return {"resolved_artist": artist_name, "match_type": "kaggle_track",
                    "match_label": f"Track \"{row['track_name']}\" → Artist: {artist_name}",
                    "track_row": track_row, "found": True}

        # 3c: album name column
        if "album_name" in df.columns:
            mask_album = df["album_name"].str.lower().str.contains(q_lower, na=False)
            if mask_album.any():
                row = df[mask_album].iloc[0]
                artist_name = str(row["artists"]).split(";")[0].strip()
                return {"resolved_artist": artist_name, "match_type": "kaggle_album",
                        "match_label": f"Album \"{row['album_name']}\" → Artist: {artist_name}",
                        "track_row": None, "found": True}
    except Exception:
        pass

    # ── Step 4 & 5: fall back — treat raw input as artist name for YT/iTunes ──
    return {"resolved_artist": q, "match_type": "raw",
            "match_label": f"Searching as artist name: {q}", "track_row": None, "found": True}


def search_youtube_for_artist(artist_name: str) -> dict:
    """Search YouTube for a top video view count. Two-pass: 'official' then bare name."""
    try:
        cache_path = ROOT / "data" / "raw" / "cache" / f"yt_artist_{artist_name.replace(' ','_')[:40]}.json"
        if cache_path.exists():
            return json.loads(cache_path.read_text())

        from src.api.youtube_client import YouTubeClient
        yt = YouTubeClient()

        def _fetch(query_suffix: str) -> dict | None:
            """Search with given suffix, resolve video_id → view count."""
            snippet = yt.search_video(artist_name, query_suffix)
            if not snippet:
                return None
            # search_video returns the raw search item; video_id is nested
            vid_id = None
            if isinstance(snippet.get("id"), dict):
                vid_id = snippet["id"].get("videoId")
            elif isinstance(snippet.get("id"), str):
                vid_id = snippet["id"]
            if not vid_id:
                return None
            title = snippet.get("snippet", {}).get("title", "")
            views = yt.get_view_count(vid_id)
            return {"video_id": vid_id, "title": title, "views": views}

        # Pass 1: search "<artist> official"
        result = _fetch("official")

        # Pass 2 fallback: if no result or 0 views, try bare name
        if not result or result["views"] == 0:
            result2 = _fetch("")
            if result2 and result2["views"] > (result["views"] if result else 0):
                result = result2

        if result and result["views"] > 0:
            data = {
                "found": True,
                "video_title": result["title"],
                "video_id": result["video_id"],
                "views": result["views"],
            }
        else:
            data = {
                "found": bool(result),
                "video_title": result["title"] if result else "",
                "views": 0,
            }
        cache_path.write_text(json.dumps(data))
        return data
    except Exception as e:
        return {"found": False, "error": str(e), "views": 0}


def search_itunes_for_artist(artist_name: str) -> dict:
    """Search iTunes Search API for artist presence."""
    try:
        cache_path = ROOT / "data" / "raw" / "cache" / f"itunes_artist_{artist_name.replace(' ','_')[:40]}.json"
        if cache_path.exists():
            return json.loads(cache_path.read_text())
        import httpx
        resp = httpx.get(
            "https://itunes.apple.com/search",
            params={"term": artist_name, "entity": "musicArtist", "limit": 5},
            timeout=8,
        )
        results = resp.json().get("results", [])
        data = {
            "found": len(results) > 0,
            "result_count": len(results),
            "primary_genre": results[0].get("primaryGenreName", "") if results else "",
            "artist_name_match": results[0].get("artistName", "") if results else "",
        }
        cache_path.write_text(json.dumps(data))
        return data
    except Exception as e:
        return {"found": False, "error": str(e)}


def check_neo4j_for_artist(artist_name: str) -> dict:
    """Check if artist has cached Neo4j data."""
    try:
        from src.graph.neo4j_client import Neo4jClient
        neo4j = Neo4jClient()
        rows = neo4j.run(
            "MATCH (a:Artist) WHERE toLower(a.name) CONTAINS toLower($name) "
            "RETURN a.spotify_id AS id, a.name AS name, a.is_ghost AS ghost LIMIT 1",
            name=artist_name,
        )
        if not rows:
            return {"found": False}
        row = rows[0]
        artist_id = row["id"]
        track_rows = neo4j.run(
            "MATCH (a:Artist {spotify_id:$id})-[:RELEASED]->(:Album)-[:CONTAINS]->(t:Track) RETURN count(t) AS n",
            id=artist_id,
        )
        track_count = track_rows[0]["n"] if track_rows else 0
        return {
            "found": True,
            "spotify_id": artist_id,
            "name": row["name"],
            "is_ghost_labeled": row.get("ghost"),
            "cached_track_count": track_count,
        }
    except Exception:
        return {"found": False}


def research_assistant_query(question: str, history: list) -> str:
    """Answer a research question using GPT-4o with full project context."""
    client = get_openai_client()
    messages = [
        {"role": "system", "content": f"""You are a PhD-level research assistant for the 'Exploratory Data Analysis for Music' project at Northeastern University (INFO 7390, Spring 2026).

{PROJECT_CONTEXT}

Respond at PhD level with specific numbers and citations to the data. Be rigorous but accessible. Use markdown formatting for structure."""},
    ]
    # Add conversation history (last 6 turns)
    for turn in history[-6:]:
        messages.append({"role": "user", "content": turn["user"]})
        messages.append({"role": "assistant", "content": turn["assistant"]})
    messages.append({"role": "user", "content": question})

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=0.4,
        max_tokens=3000,
    )
    return response.choices[0].message.content


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎵 EDA for Music")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        [
            "🏠 Home",
            "📊 Exercise Gallery",
            "🔍 Artist Analyzer",
            "🌐 Network Explorer",
            "📡 Cross-Platform",
            "🤖 AI Research Assistant",
            "ℹ️ About",
        ],
        label_visibility="collapsed",
    )
    st.markdown("---")
    # Project health
    st.markdown("<div style='color:#a78bfa;font-size:0.8rem;font-weight:700;'>PROJECT HEALTH</div>", unsafe_allow_html=True)
    try:
        from src.graph.neo4j_client import Neo4jClient
        _n = Neo4jClient(); _n.run("RETURN 1")
        st.markdown("<div style='color:#22c55e;font-size:0.75rem;'>✅ Neo4j connected</div>", unsafe_allow_html=True)
    except:
        st.markdown("<div style='color:#e74c3c;font-size:0.75rem;'>❌ Neo4j offline</div>", unsafe_allow_html=True)

    gnn_path = ROOT / "data" / "processed" / "gat_model.pt"
    if gnn_path.exists():
        st.markdown("<div style='color:#22c55e;font-size:0.75rem;'>✅ GNN model loaded</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div style='color:#f59e0b;font-size:0.75rem;'>⚠️ GNN model missing</div>", unsafe_allow_html=True)

    yt_key = os.getenv("YOUTUBE_API_KEY", "")
    openai_key = os.getenv("OPENAI_API_KEY", "")
    st.markdown(f"<div style='color:{'#22c55e' if yt_key else '#f59e0b'};font-size:0.75rem;'>{'✅' if yt_key else '⚠️'} YouTube API</div>", unsafe_allow_html=True)
    st.markdown(f"<div style='color:{'#22c55e' if openai_key else '#e74c3c'};font-size:0.75rem;'>{'✅' if openai_key else '❌'} OpenAI API</div>", unsafe_allow_html=True)
    st.markdown("<div style='color:#94a3b8;font-size:0.75rem;'>✅ Kaggle 114K tracks</div>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(
        "<div style='color:#64748b;font-size:0.78rem;'>INFO 7390 · Spring 2026<br>Fake Artist Detection<br>via Public API Analysis</div>",
        unsafe_allow_html=True,
    )


# ═════════════════════════════════════════════════════════════════════════════
# HOME PAGE
# ═════════════════════════════════════════════════════════════════════════════
if page == "🏠 Home":
    st.markdown("# 🎵 Exploratory Data Analysis for Music")
    st.markdown("### A Layered Framework for Public-API Discovery of Ghost Artists on Streaming Platforms")

    st.markdown("""<div style='display:flex;flex-wrap:wrap;gap:16px;margin-bottom:8px;'>
        <div class='metric-card' style='flex:1;min-width:140px;'><div class='number'>3</div><div class='label'>Ghost Artists Analyzed</div></div>
        <div class='metric-card' style='flex:1;min-width:140px;'><div class='number'>490</div><div class='label'>Tracks in Neo4j</div></div>
        <div class='metric-card' style='flex:1;min-width:140px;'><div class='number'>7</div><div class='label'>Detection Signals</div></div>
        <div class='metric-card' style='flex:1;min-width:140px;'><div class='number'>114K</div><div class='label'>Kaggle Training Tracks</div></div>
    </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Key Findings
    st.markdown("## Key Findings")
    findings_list = [
        ("📊", "12.5× catalog variance ratio", "Ghost artists show 12.5× lower audio feature variance than organic controls (Levene's test p&lt;0.001)."),
        ("⏱️", "81–95% cadence closure", "Ghost artists bulk-upload tracks with 81–95% same-day release clustering vs 0% for Nils Frahm."),
        ("📺", "Ghost artists ARE on YouTube", "Relaxing White Noise has 353M YouTube views — fraud is Spotify-economic stream farming, not fabricated identity."),
        ("🏭", "HHI 0.88 concentration", "Single production company controls 88% of RWN's catalog, revealing bulk-upload operations via ISRC attribution."),
        ("🧠", "100% GNN test accuracy", "GAT model achieves 100% test accuracy on the proof-of-concept 65-node graph (note: synthetic graph structure)."),
    ]
    for icon, title, detail in findings_list:
        st.markdown(f"""<div class='ai-card' style='padding:14px 18px;margin:8px 0;'>
            <div style='display:flex;align-items:flex-start;gap:12px;'>
                <span style='font-size:1.4rem;'>{icon}</span>
                <div>
                    <div style='color:#a78bfa;font-weight:700;font-size:0.95rem;'>{title}</div>
                    <div style='color:#e2e8f0;font-size:0.88rem;margin-top:3px;line-height:1.5;'>{detail}</div>
                </div>
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("## The 7-Layer Detection Framework")
    st.markdown("Ghost artists leave detectable traces across Spotify's public API. This framework requires **only public, unauthenticated access**, making it replicable without special permissions.")

    for layer in FRAMEWORK_LAYERS:
        st.markdown(f"""<div class='layer-card'>
            <span style='color:#a78bfa;font-weight:700;'>Layer {layer['num']}: {layer['name']}</span>
            <span style='float:right;'><span class='tag'>{layer['data']}</span></span>
            <div style='color:#94a3b8;font-size:0.85rem;margin-top:4px;'>{layer['desc']}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Research Impact
    st.markdown("## Research Impact")
    col_i1, col_i2 = st.columns(2)
    with col_i1:
        st.markdown("""<div style='background:#1a1a2e;border:1px solid #4c1d95;border-radius:10px;padding:16px;'>
            <div style='color:#a78bfa;font-weight:700;margin-bottom:8px;'>💡 Key Contribution</div>
            <div style='color:#e2e8f0;font-size:0.9rem;'>This framework demonstrates that <strong>independent platform audit is possible</strong> using only public API endpoints — without access to Spotify's internal fraud systems.</div>
        </div>""", unsafe_allow_html=True)
    with col_i2:
        st.markdown("""<div style='background:#1a1a2e;border:1px solid #4c1d95;border-radius:10px;padding:16px;'>
            <div style='color:#a78bfa;font-weight:700;margin-bottom:8px;'>🔍 Surprise Finding</div>
            <div style='color:#e2e8f0;font-size:0.9rem;'>Ghost artists are <strong>NOT cross-platform invisible</strong>. Relaxing White Noise has 353M YouTube views. Ghost behavior is Spotify-economic stream farming, not fabricated identity.</div>
        </div>""", unsafe_allow_html=True)

    st.info("**Status:** Exercises 1–7 complete · All 7 signals implemented · GAT/GCN models trained (100% test accuracy) · OpenAI GPT-4o integrated")


# ═════════════════════════════════════════════════════════════════════════════
# EXERCISE GALLERY
# ═════════════════════════════════════════════════════════════════════════════
elif page == "📊 Exercise Gallery":
    st.markdown("# 📊 Exercise Gallery")
    st.markdown("Figures generated from Kaggle dataset (114K tracks) and Neo4j graph (490 tracks).")
    st.markdown("---")

    any_found = False
    for filename, meta in FIGURES_META.items():
        fig_path = FIGURES_DIR / filename
        if fig_path.exists():
            any_found = True
            st.markdown(f"""<div class='fig-frame'><span class='tag'>{meta['exercise']}</span> <span class='tag'>{meta['signal']}</span>
            <h3 style='margin-top:10px;'>{meta['title']}</h3></div>""", unsafe_allow_html=True)
            st.image(str(fig_path), use_container_width=True)
            st.markdown(f"<div style='color:#94a3b8;font-size:0.85rem;margin-bottom:24px;'>📌 {meta['caption']}</div>", unsafe_allow_html=True)
            st.markdown("---")

    if not any_found:
        st.warning("No figures found. Run notebooks 01–07 to generate figures.")


# ═════════════════════════════════════════════════════════════════════════════
# ARTIST ANALYZER
# ═════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Artist Analyzer":
    st.markdown("# 🔍 Artist Analyzer")
    st.markdown("Search by artist name, Spotify ID, album, or track. We search across YouTube, Apple Music, Kaggle (114K tracks), and our Neo4j graph to assess ghost artist probability.")

    artist_input = st.text_input(
        "Search anything",
        placeholder="Artist name, Spotify ID, album name, track name...",
        label_visibility="collapsed",
    )
    analyze_btn = st.button("🔍 Search", type="primary", use_container_width=True)

    # Quick-pick — 2×2 grid so labels don't get squeezed into vertical text
    st.markdown("<div style='color:#64748b;font-size:0.8rem;margin-bottom:4px;'>Quick-pick:</div>", unsafe_allow_html=True)
    qp_row1 = st.columns(2)
    qp_row2 = st.columns(2)
    quick_picks = [
        (qp_row1[0], "Relaxing White Noise", "Relaxing White Noise"),
        (qp_row1[1], "Meditation Relax Club", "Meditation Relax Club"),
        (qp_row2[0], "Calmo", "Calmo"),
        (qp_row2[1], "Nils Frahm", "Nils Frahm"),
    ]
    for col, label, full_name in quick_picks:
        if col.button(label, key=f"qp_{label}", use_container_width=True):
            artist_input = full_name
            analyze_btn = True

    if analyze_btn and artist_input.strip():
        raw_query = artist_input.strip()

        with st.spinner(f"Searching for **{raw_query}**…"):

            # ── Step 1-3: Smart search — resolve to artist name ───────────────
            st.caption("🔎 Resolving input (Spotify ID / Neo4j / Kaggle track / album)…")
            search_meta = smart_search(raw_query)
            artist_query = search_meta["resolved_artist"]

            # ── Steps 4-5 + pipeline: collect all sources ─────────────────────
            collected_data = {
                "artist_name": artist_query,
                "original_query": raw_query,
                "search_meta": search_meta,
                "sources": {},
            }

            # If matched via track, store the specific track's audio features
            if search_meta.get("track_row"):
                collected_data["matched_track"] = search_meta["track_row"]

            st.caption("🔍 Checking Neo4j graph…")
            neo4j_data = check_neo4j_for_artist(artist_query)
            collected_data["neo4j"] = neo4j_data
            collected_data["sources"]["neo4j"] = neo4j_data["found"]

            st.caption("🔍 Searching Kaggle dataset (114K tracks)…")
            kaggle_data = search_kaggle_for_artist(artist_query)
            collected_data["kaggle"] = kaggle_data
            collected_data["sources"]["kaggle"] = kaggle_data["found"]

            st.caption("▶️ Searching YouTube…")
            yt_data = search_youtube_for_artist(artist_query)
            collected_data["youtube"] = yt_data
            collected_data["sources"]["youtube"] = yt_data.get("found", False)

            st.caption("🎵 Checking Apple Music…")
            itunes_data = search_itunes_for_artist(artist_query)
            collected_data["apple_music"] = itunes_data
            collected_data["sources"]["apple_music"] = itunes_data.get("found", False)

            # Signal pipeline (Neo4j artists only)
            signal_scores = None
            gnn_score = None
            rule_score = None
            if neo4j_data["found"]:
                st.caption("⚙️ Running 7-signal pipeline…")
                try:
                    from src.signals.verdict import compute_verdict_gnn
                    pipeline_result = compute_verdict_gnn(
                        artist_id=neo4j_data["spotify_id"],
                        artist_name=neo4j_data["name"],
                        run_s7=False,
                    )
                    signal_scores = pipeline_result.get("signal_scores", {})
                    gnn_score = pipeline_result.get("gnn_score")
                    rule_score = pipeline_result.get("rule_based_score")
                    collected_data["signal_scores"] = {k: v for k, v in signal_scores.items() if v is not None}
                    collected_data["rule_based_score"] = rule_score
                    collected_data["gnn_score"] = gnn_score
                except Exception as e:
                    collected_data["pipeline_error"] = str(e)

            # Check if nothing at all was found
            nothing_found = (
                not neo4j_data["found"]
                and not kaggle_data.get("found")
                and not yt_data.get("found")
                and not itunes_data.get("found")
            )
            collected_data["nothing_found"] = nothing_found

        # Store in session state
        st.session_state["analyzer_data"] = collected_data
        st.session_state["analyzer_query"] = artist_query
        st.session_state["analyzer_raw_query"] = raw_query
        st.session_state["analyzer_signal_scores"] = signal_scores
        st.session_state["analyzer_gnn_score"] = gnn_score
        st.session_state["analyzer_rule_score"] = rule_score

    # ── Display results whenever we have collected data ───────────────────────
    if st.session_state.get("analyzer_data") and st.session_state.get("analyzer_query"):
        artist_query  = st.session_state["analyzer_query"]
        raw_query     = st.session_state.get("analyzer_raw_query", artist_query)
        collected_data = st.session_state["analyzer_data"]
        signal_scores = st.session_state.get("analyzer_signal_scores")
        gnn_score     = st.session_state.get("analyzer_gnn_score")
        rule_score    = st.session_state.get("analyzer_rule_score")
        search_meta   = collected_data.get("search_meta", {})

        st.markdown("---")

        # Nothing found anywhere
        if collected_data.get("nothing_found"):
            st.error(f"No results found for **\"{raw_query}\"**. Try searching by artist name.")
            st.stop()

        # "Matched as" pill + results heading
        match_label = search_meta.get("match_label", f"Artist name: {artist_query}")
        st.markdown(
            f"<div style='display:inline-block;background:#1e1b4b;border:1px solid #4c1d95;"
            f"border-radius:20px;padding:4px 14px;font-size:0.8rem;color:#a5b4fc;"
            f"margin-bottom:10px;'>🔎 Matched as: {match_label}</div>",
            unsafe_allow_html=True,
        )
        st.markdown(f"## Results: {artist_query}")

        # If matched via a specific track, show that track's audio features
        matched_track = collected_data.get("matched_track")
        if matched_track:
            tn = matched_track.get("track_name", "")
            alb = matched_track.get("album_name", "")
            tg = matched_track.get("track_genre", "")
            pop = matched_track.get("popularity", 0)
            st.markdown(
                f"<div style='background:#1a1a2e;border:1px solid #4c1d95;border-radius:8px;"
                f"padding:12px 16px;margin-bottom:12px;'>"
                f"<div style='color:#a78bfa;font-size:0.8rem;font-weight:700;margin-bottom:6px;'>🎵 MATCHED TRACK</div>"
                f"<div style='color:#e2e8f0;font-weight:600;'>{tn}</div>"
                f"<div style='color:#94a3b8;font-size:0.85rem;margin-top:2px;'>"
                f"{('Album: ' + alb + ' · ') if alb else ''}"
                f"{('Genre: ' + tg + ' · ') if tg else ''}"
                f"Popularity: {pop}/100</div>"
                f"<div style='display:flex;gap:16px;margin-top:8px;flex-wrap:wrap;'>"
                + "".join(
                    f"<span style='color:#94a3b8;font-size:0.8rem;'>"
                    f"<span style='color:#a78bfa;font-weight:600;'>{k.title()}</span> {v:.3f}</span>"
                    for k, v in {
                        "dance": matched_track.get("danceability", 0),
                        "energy": matched_track.get("energy", 0),
                        "valence": matched_track.get("valence", 0),
                        "acoustic": matched_track.get("acousticness", 0),
                    }.items()
                )
                + "</div></div>",
                unsafe_allow_html=True,
            )

        # ── Shared data extraction ────────────────────────────────────────────
        yt_views  = collected_data.get("youtube", {}).get("views", 0)
        yt_found  = collected_data.get("youtube", {}).get("found", False)
        yt_title  = collected_data.get("youtube", {}).get("video_title", "")
        apple_found = collected_data.get("apple_music", {}).get("found", False)
        kaggle_data = collected_data.get("kaggle", {})
        neo4j_found = collected_data.get("neo4j", {}).get("found", False)

        # ── Verdict logic ─────────────────────────────────────────────────────
        if neo4j_found and (gnn_score is not None or rule_score is not None):
            # Pipeline path: use numeric score
            _score = gnn_score if gnn_score is not None else rule_score
            _score_src = "GNN" if gnn_score is not None else "rule-based"
            if _score > 0.7:
                _verdict, _icon, _bg, _border, _conf = "LIKELY GHOST",     "🚨", "#4a1a1a", "#e74c3c", int(60 + _score * 35)
                _reason = f"{_score_src} score {_score:.3f} exceeds ghost threshold (>0.7)"
            elif _score > 0.4:
                _verdict, _icon, _bg, _border, _conf = "SUSPICIOUS",       "⚠️", "#3d2a0a", "#f59e0b", int(40 + _score * 30)
                _reason = f"{_score_src} score {_score:.3f} in ambiguous range (0.4–0.7)"
            else:
                _verdict, _icon, _bg, _border, _conf = "LIKELY ORGANIC",   "✅", "#0a2e14", "#22c55e", int(70 + (0.4 - _score) * 60)
                _reason = f"{_score_src} score {_score:.3f} below ghost threshold (<0.4)"
        else:
            # Cross-platform-only path for artists not in Neo4j
            _views_fmt = f"{yt_views/1e6:.1f}M" if yt_views >= 1_000_000 else (f"{yt_views:,}" if yt_views > 0 else "0")
            if yt_views < 1_000 and not apple_found:
                _verdict, _icon, _bg, _border, _conf = "LIKELY GHOST",     "🚨", "#4a1a1a", "#e74c3c", 82
                _reason = f"{_views_fmt} YouTube views + not on Apple Music — invisible outside Spotify"
            elif yt_views < 10_000 and not apple_found:
                _verdict, _icon, _bg, _border, _conf = "SUSPICIOUS",       "⚠️", "#3d2a0a", "#f59e0b", 61
                _reason = f"{_views_fmt} YouTube views + not on Apple Music — minimal cross-platform presence"
            elif yt_views >= 10_000_000 and apple_found:
                _verdict, _icon, _bg, _border, _conf = "LIKELY ORGANIC",   "✅", "#0a2e14", "#22c55e", 92
                _reason = f"{_views_fmt} YouTube views + Apple Music presence — consistent cross-platform footprint"
            else:
                _verdict, _icon, _bg, _border, _conf = "INSUFFICIENT DATA","❓", "#1e1e2e", "#475569", 30
                _reason = "Moderate presence — full Spotify signal analysis needed for classification"

        st.markdown(
            f"<div style='background:{_bg};border:2px solid {_border};border-radius:10px;"
            f"padding:18px 24px;margin:12px 0 20px 0;'>"
            f"<div style='font-size:1.5rem;font-weight:800;color:#fff;letter-spacing:0.02em;'>"
            f"{_icon} {_verdict} &nbsp;<span style='font-size:1rem;font-weight:500;color:#cbd5e1;'>"
            f"— {_conf}% confidence</span></div>"
            f"<div style='color:#e2e8f0;font-size:0.92rem;margin-top:6px;'>{_reason}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

        def _src_row(icon: str, source: str, detail: str, ok: bool) -> str:
            border = "#22c55e" if ok else "#475569"
            bg     = "#0f1f14" if ok else "#1a1a1a"
            return (
                f"<div style='display:flex;align-items:flex-start;gap:10px;background:{bg};"
                f"border-left:3px solid {border};border-radius:6px;padding:10px 14px;margin-bottom:6px;'>"
                f"<span style='font-size:1.1rem;line-height:1.4;'>{icon}</span>"
                f"<div><span style='color:#e2e8f0;font-weight:600;'>{source}</span>"
                f"<span style='color:#94a3b8;font-size:0.88rem;margin-left:8px;'>{detail}</span></div></div>"
            )

        # Neo4j row
        if neo4j_found:
            tc = collected_data.get("neo4j", {}).get("cached_track_count", "?")
            neo4j_detail = f"Found — {tc} tracks in graph, full 7-signal pipeline available"
            neo4j_ok = True
        else:
            neo4j_detail = "Not cached — would need Spotify API ingestion (rate limited)"
            neo4j_ok = False

        # Kaggle row
        if kaggle_data.get("found"):
            ktc = kaggle_data.get("track_count", 0)
            kvar = kaggle_data.get("total_variance", None)
            var_str = f", audio variance={kvar:.4f}" if kvar is not None else ""
            kaggle_detail = f"Found {ktc} tracks with audio features{var_str}"
            kaggle_ok = True
        else:
            kaggle_detail = "Artist not in 114K Kaggle dataset (likely non-mainstream or ambient genre)"
            kaggle_ok = False

        # YouTube row
        if not yt_found:
            yt_detail = "Not found on YouTube"
            yt_ok = False
        elif yt_views >= 10_000_000:
            yt_detail = f"{yt_views/1e6:.1f}M views — Strong cross-platform presence"
            if yt_title:
                yt_detail += f" · top video: \"{yt_title[:50]}\""
            yt_ok = True
        elif yt_views >= 1_000:
            yt_detail = f"{yt_views:,} views — Moderate presence"
            yt_ok = True
        elif yt_views > 0:
            yt_detail = f"{yt_views} views — Minimal presence (suspicious)"
            yt_ok = False
        else:
            yt_detail = "Found but 0 views reported"
            yt_ok = False

        # Apple Music row
        if apple_found:
            genre_str = collected_data.get("apple_music", {}).get("primary_genre", "")
            ap_detail = f"Found on Apple Music" + (f" · genre: {genre_str}" if genre_str else "")
            ap_ok = True
        else:
            ap_detail = "Not found on Apple Music"
            ap_ok = False

        # Pipeline row
        if signal_scores:
            if gnn_score is not None:
                pipe_detail = f"Full 7-signal run complete · GNN score: {gnn_score:.3f}"
            else:
                pipe_detail = f"Full 7-signal run complete · rule score: {rule_score:.3f}"
            pipe_ok = True
        else:
            pipe_detail = "Full 7-signal scoring requires Neo4j data — not available for this artist"
            pipe_ok = False

        st.markdown(
            _src_row("🗄️", "Neo4j",       neo4j_detail,  neo4j_ok)
            + _src_row("📊", "Kaggle",     kaggle_detail, kaggle_ok)
            + _src_row("▶️", "YouTube",    yt_detail,     yt_ok)
            + _src_row("🎵", "Apple Music", ap_detail,    ap_ok)
            + _src_row("⚙️", "Pipeline",   pipe_detail,   pipe_ok),
            unsafe_allow_html=True,
        )

        # ── "What would full analysis show?" block (new artists only) ────────
        if not neo4j_found:
            with st.expander("ℹ️ What would full analysis show?"):
                st.markdown("""
**To run all 7 detection signals, this artist needs to be ingested into Neo4j via the Spotify API.**

Full analysis would compute:
- **S1** Audio fingerprint similarity — requires audio features per track
- **S2** Release cadence synchrony — requires album/track release dates
- **S3** Playlist co-occurrence — requires playlist membership data
- **S4** Catalog density anomaly — tracks per day upload velocity
- **S5** Metadata similarity — track name embedding clusters
- **S6** Graph density / HHI — ISRC production company concentration
- **S7** Cross-platform discrepancy — YouTube + Apple Music (available above)

**Required Spotify API calls:** artist albums → album tracks → track ISRC → related artists graph

**Current constraint:** Spotify Developer Mode limits to ~30 API calls/day per app. Ingesting a new artist requires ~50–200 calls depending on catalog size. Ingestion is possible but not automated in this demo.
""")


        # Pipeline signal scores if available
        if signal_scores:
            st.markdown("### Signal Scores (pipeline)")
            SIGNAL_META_DISPLAY = [
                ("s1_audio_similarity",      "S1", "Audio Fingerprint"),
                ("s2_cadence_sync",          "S2", "Release Cadence"),
                ("s3_playlist_cooccurrence", "S3", "Playlist Co-occurrence"),
                ("s4_follower_ratio",        "S4", "Catalog Density"),
                ("s5_metadata_similarity",   "S5", "Metadata Similarity"),
                ("s6_graph_density",         "S6", "Graph Density / HHI"),
                ("s7_cross_platform",        "S7", "Cross-Platform"),
            ]
            for pipeline_key, num, name in SIGNAL_META_DISPLAY:
                val = signal_scores.get(pipeline_key)
                if val is None:
                    score_str, score_color, bar_pct = "N/A", "#64748b", 0
                else:
                    score_str = f"{val:.3f}"
                    bar_pct = int(val * 100)
                    score_color = "#e74c3c" if val > 0.7 else ("#f59e0b" if val > 0.4 else "#22c55e")
                st.markdown(f"""<div class='signal-card'>
                    <span class='score' style='color:{score_color};'>{score_str}</span>
                    <div class='name'>{num}: {name}</div>
                    <div style='margin-top:6px;background:#2a2a4a;border-radius:4px;height:6px;'>
                        <div style='width:{bar_pct}%;background:{score_color};border-radius:4px;height:6px;'></div>
                    </div>
                </div>""", unsafe_allow_html=True)

        # ── Optional AI deep-dive ─────────────────────────────────────────────
        st.markdown("---")
        st.markdown("### 🤖 AI Deep-Dive Analysis *(optional)*")
        st.caption("Calls GPT-4o once to generate a PhD-level ghost detection analysis from the data above.")
        ai_btn = st.button("Get AI Analysis", type="secondary")

        if ai_btn:
            with st.spinner("🤖 GPT-4o analyzing for ghost behavior…"):
                try:
                    ai_result = ai_analyze_artist(artist_query, collected_data)
                    st.session_state["analyzer_ai_result"] = ai_result
                except Exception as e:
                    st.error(f"AI analysis failed: {e}")
                    st.session_state.pop("analyzer_ai_result", None)

        ai_result = st.session_state.get("analyzer_ai_result")
        if ai_result and st.session_state.get("analyzer_query") == artist_query:
            ghost_prob = ai_result.get("ghost_probability", 50)
            verdict = ai_result.get("verdict", "INSUFFICIENT_DATA")
            confidence = ai_result.get("confidence", 0)

            if verdict == "LIKELY_GHOST":
                badge_class, badge_icon, gauge_color = "verdict-ghost", "🚨", "#e74c3c"
            elif verdict == "SUSPICIOUS":
                badge_class, badge_icon, gauge_color = "verdict-suspicious", "⚠️", "#f59e0b"
            elif verdict == "LIKELY_ORGANIC":
                badge_class, badge_icon, gauge_color = "verdict-organic", "✅", "#22c55e"
            else:
                badge_class, badge_icon, gauge_color = "verdict-suspicious", "❓", "#64748b"

            hdr_c1, hdr_c2, hdr_c3 = st.columns(3)
            with hdr_c1:
                st.markdown(f"<span class='{badge_class}'>{badge_icon} {verdict}</span>", unsafe_allow_html=True)
            with hdr_c2:
                st.metric("Ghost Probability", f"{ghost_prob}%")
            with hdr_c3:
                st.metric("AI Confidence", f"{confidence}%")

            try:
                import plotly.graph_objects as go
                gauge_fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=ghost_prob,
                    domain={"x": [0, 1], "y": [0, 1]},
                    title={"text": "Ghost Probability %", "font": {"color": "#e2e8f0", "size": 14}},
                    gauge={
                        "axis": {"range": [0, 100], "tickcolor": "#64748b"},
                        "bar": {"color": gauge_color},
                        "bgcolor": "#1a1a2e",
                        "bordercolor": "#2a2a4a",
                        "steps": [
                            {"range": [0, 40], "color": "#14532d"},
                            {"range": [40, 70], "color": "#78350f"},
                            {"range": [70, 100], "color": "#7f1d1d"},
                        ],
                        "threshold": {"line": {"color": gauge_color, "width": 4}, "value": ghost_prob},
                    },
                    number={"font": {"color": gauge_color, "size": 36}, "suffix": "%"},
                ))
                gauge_fig.update_layout(paper_bgcolor="#12121f", height=220, margin=dict(t=40, b=0, l=30, r=30))
                st.plotly_chart(gauge_fig, use_container_width=True)
            except ImportError:
                st.progress(ghost_prob / 100)

            analysis_text = ai_result.get("analysis", "")
            st.markdown(f"""<div class='ai-card'>
                <div style='color:#a78bfa;font-size:0.8rem;font-weight:700;margin-bottom:12px;'>GPT-4o ANALYSIS — {artist_query.upper()}</div>
                <div style='color:#e2e8f0;line-height:1.8;font-size:0.92rem;'>{analysis_text.replace(chr(10), "<br>")}</div>
            </div>""", unsafe_allow_html=True)

            comparison = ai_result.get("comparison", "")
            if comparison:
                with st.expander("📊 Comparison to Known Baselines"):
                    st.markdown(comparison)
                    import pandas as pd
                    baseline_df = pd.DataFrame([
                        {"Artist": "Relaxing White Noise", "Type": "GHOST", "Cadence": "81%", "HHI": "0.67", "YT Views": "353M"},
                        {"Artist": "Meditation Relax Club", "Type": "GHOST", "Cadence": "95%", "HHI": "0.52", "YT Views": "157M"},
                        {"Artist": "Calmo", "Type": "GHOST?", "Cadence": "32%", "HHI": "0.45", "YT Views": "155"},
                        {"Artist": "Nils Frahm", "Type": "ORGANIC", "Cadence": "0%", "HHI": "0.00", "YT Views": "9M"},
                        {"Artist": artist_query, "Type": verdict, "Cadence": "?", "HHI": "?",
                         "YT Views": f"{collected_data.get('youtube', {}).get('views', '?'):,}" if isinstance(collected_data.get('youtube', {}).get('views'), int) else "?"},
                    ])
                    st.dataframe(baseline_df, use_container_width=True, hide_index=True)

            ks_col, md_col = st.columns(2)
            with ks_col:
                key_signals = ai_result.get("key_signals", [])
                if key_signals:
                    st.markdown("**🔑 Key Signals**")
                    for s in key_signals:
                        st.markdown(f"- {s}")
            with md_col:
                missing = ai_result.get("missing_data", [])
                if missing:
                    st.markdown("**📋 Additional Data Needed**")
                    for m in missing:
                        st.markdown(f"- {m}")

    elif analyze_btn and not artist_input.strip():
        st.error("Please enter a search term.")

    elif not st.session_state.get("analyzer_data"):
        st.markdown("<div style='color:#64748b;text-align:center;padding:40px;'>Enter an artist name, Spotify ID, track, or album above and click Search.</div>", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# NETWORK EXPLORER
# ═════════════════════════════════════════════════════════════════════════════
elif page == "🌐 Network Explorer":
    st.markdown("# 🌐 Network Explorer")
    st.markdown("ISRC-based production company graph. High HHI concentration (> 0.6) signals bulk-upload operations.")
    st.markdown("---")

    # ── Search Artist Network ─────────────────────────────────────────────────
    st.markdown("### Search Artist Network")
    st.caption("Compare any artist's cross-platform footprint against the 4 seed artists. Full graph analysis requires Neo4j ingestion.")

    net_search_col, net_btn_col = st.columns([4, 1])
    with net_search_col:
        net_artist_input = st.text_input(
            "Artist name",
            placeholder='e.g. "Karan Aujla", "Taylor Swift", "Brian Eno"',
            key="net_search_input",
            label_visibility="collapsed",
        )
    with net_btn_col:
        net_search_btn = st.button("🔍 Compare", type="primary", use_container_width=True, key="net_search_btn")

    if net_search_btn and net_artist_input.strip():
        with st.spinner(f"Fetching cross-platform data for **{net_artist_input.strip()}**…"):
            _ns_yt    = search_youtube_for_artist(net_artist_input.strip())
            _ns_itunes = search_itunes_for_artist(net_artist_input.strip())
        st.session_state["net_search_result"] = {
            "name": net_artist_input.strip(),
            "yt": _ns_yt,
            "itunes": _ns_itunes,
        }

    if st.session_state.get("net_search_result"):
        _ns = st.session_state["net_search_result"]
        _ns_name = _ns["name"]
        _ns_views = _ns["yt"].get("views", 0)
        _ns_apple = _ns["itunes"].get("found", False)

        # Reference data for seed artists
        _seed = [
            ("Relaxing White Noise", 353_775_028, True,  "LIKELY_GHOST"),
            ("Meditation Relax Club", 157_581_269, True,  "LIKELY_GHOST"),
            ("Calmo",                 155,         False, "SUSPICIOUS"),
            ("Nils Frahm",            9_107_596,   True,  "LIKELY_ORGANIC"),
            (_ns_name,                _ns_views,   _ns_apple, "SEARCHED"),
        ]

        import plotly.graph_objects as go
        import pandas as pd

        # Bar chart — YouTube views comparison
        _colors = {
            "LIKELY_GHOST":    "#e74c3c",
            "SUSPICIOUS":      "#f59e0b",
            "LIKELY_ORGANIC":  "#22c55e",
            "SEARCHED":        "#a78bfa",
        }
        bar_fig = go.Figure(go.Bar(
            x=[r[0] for r in _seed],
            y=[r[1] for r in _seed],
            marker_color=[_colors[r[3]] for r in _seed],
            text=[f"{r[1]/1e6:.1f}M" if r[1] >= 1_000_000 else str(r[1]) for r in _seed],
            textposition="outside",
        ))
        bar_fig.update_layout(
            title=f"YouTube Views: {_ns_name} vs Seed Artists",
            yaxis_title="Views",
            plot_bgcolor="#12121f", paper_bgcolor="#12121f", font_color="#e2e8f0",
            title_font_color="#a78bfa", yaxis=dict(gridcolor="#2a2a4a"),
            showlegend=False, margin=dict(t=50, b=40),
        )
        st.plotly_chart(bar_fig, use_container_width=True)

        # Apple Music comparison table
        _rows = [{"Artist": r[0], "YouTube Views": f"{r[1]/1e6:.1f}M" if r[1] >= 1_000_000 else str(r[1]),
                  "Apple Music": "✅" if r[2] else "❌", "Status": r[3]} for r in _seed]
        st.dataframe(pd.DataFrame(_rows), use_container_width=True, hide_index=True)

        st.info("Full graph analysis (ISRC, HHI, release cadence) requires Spotify API ingestion into Neo4j. Currently showing cross-platform comparison only.")

    st.markdown("---")

    try:
        from src.graph.neo4j_client import Neo4jClient
        neo4j = Neo4jClient()
        counts = neo4j.count_nodes()
        rel_rows = neo4j.run("MATCH ()-[r]->() RETURN type(r) AS rel_type, count(r) AS cnt ORDER BY cnt DESC")
        total_rels = sum(r["cnt"] for r in rel_rows)

        st.markdown("### Graph Stats (Neo4j AuraDB)")
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Total Nodes", f"{sum(counts.values()):,}")
        col2.metric("Artists", counts.get("Artist", 0))
        col3.metric("Tracks", counts.get("Track", 0))
        col4.metric("Albums", counts.get("Album", 0))
        col5.metric("Relationships", f"{total_rels:,}")

        st.markdown("---")
        st.markdown("### ISRC Production Company Clusters")

        cluster_rows = neo4j.run("""
            MATCH (a:Artist)-[:RELEASED]->(al:Album)-[:CONTAINS]->(t:Track)
                  -[:REGISTERED_WITH]->(c:ProductionCompany)
            WITH c, collect(DISTINCT a.name) AS artists, count(DISTINCT t) AS track_count
            RETURN c.isrc_prefix AS prefix, c.name AS company_name, artists,
                   size(artists) AS artist_count, track_count
            ORDER BY track_count DESC""")

        if cluster_rows:
            import pandas as pd
            df = pd.DataFrame(cluster_rows)
            df["artists"] = df["artists"].apply(lambda x: ", ".join(x) if x else "—")
            df["shared"] = df["artist_count"].apply(lambda x: "⚠️ SHARED" if x > 1 else "—")
            df = df.rename(columns={"prefix": "ISRC Prefix", "company_name": "Company", "artists": "Artists", "artist_count": "# Artists", "track_count": "# Tracks", "shared": "Flag"})
            st.dataframe(df, use_container_width=True, hide_index=True)

        st.markdown("---")
        st.markdown("### Per-Artist Neighborhood")
        neo4j_artists = [k for k in KNOWN_ARTISTS.keys() if "ghost" in k.lower() or "candidate" in k.lower()]
        selected_artist_graph = st.selectbox("Select artist", neo4j_artists[:3], key="net_artist_select")
        if selected_artist_graph:
            aid = KNOWN_ARTISTS[selected_artist_graph]
            nhood = neo4j.run("""
                MATCH (a:Artist {spotify_id: $id})-[:RELEASED]->(al:Album)
                      -[:CONTAINS]->(t:Track)-[:REGISTERED_WITH]->(c:ProductionCompany)
                RETURN c.isrc_prefix AS prefix, c.name AS company, count(t) AS track_count
                ORDER BY track_count DESC""", id=aid)
            if nhood:
                import pandas as pd
                ndf = pd.DataFrame(nhood)
                ndf = ndf.rename(columns={"prefix": "ISRC Prefix", "company": "Production Company", "track_count": "Tracks"})
                total_t = ndf["Tracks"].sum()
                ndf["Share %"] = (ndf["Tracks"] / total_t * 100).round(1)
                st.dataframe(ndf, use_container_width=True, hide_index=True)
                shares = ndf["Tracks"] / total_t
                hhi = (shares ** 2).sum()
                hhi_color = "#e74c3c" if hhi > 0.6 else ("#f59e0b" if hhi > 0.35 else "#22c55e")
                st.markdown(f"**HHI:** <span style='color:{hhi_color};font-weight:700;'>{hhi:.3f}</span> ({'Highly concentrated' if hhi > 0.6 else 'Moderately concentrated'})", unsafe_allow_html=True)

    except Exception as e:
        st.warning(f"Neo4j unavailable: {e}")

    st.markdown("---")
    fig4_path = FIGURES_DIR / "fig4_bipartite_neighborhood.png"
    if fig4_path.exists():
        st.markdown("**Exercise 4: Bipartite Neighborhood**")
        st.image(str(fig4_path), use_container_width=True)
    fig3_path = FIGURES_DIR / "fig3_isrc_join.png"
    if fig3_path.exists():
        st.markdown("**Exercise 3: ISRC Attribution**")
        st.image(str(fig3_path), use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# CROSS-PLATFORM
# ═════════════════════════════════════════════════════════════════════════════
elif page == "📡 Cross-Platform":
    st.markdown("# 📡 Cross-Platform Discrepancy (Signal 7)")
    st.warning("**Key finding:** Relaxing White Noise has **353M YouTube views** — ghost behavior is Spotify-stream-farming, not cross-platform absence.")

    st.markdown("---")
    st.markdown("### Analyze Any Artist Cross-Platform")
    cp_input = st.text_input("Artist name", placeholder='e.g. "Drake", "Calmo", "Brian Eno"', key="cp_input")
    cp_btn = st.button("🔍 Search", key="cp_btn")

    if cp_btn and cp_input.strip():
        with st.spinner(f"Searching YouTube + Apple Music for {cp_input}…"):
            yt = search_youtube_for_artist(cp_input)
            itunes = search_itunes_for_artist(cp_input)

        views = yt.get("views", 0)
        apple_found = itunes.get("found", False)
        yt_found = yt.get("found", False)

        # Determine likely verdict based on cross-platform
        if views > 10_000_000 and apple_found:
            cp_verdict = "Cross-platform presence is STRONG"
        elif views < 10_000 and not apple_found:
            cp_verdict = "Cross-platform presence is ABSENT (suspicious)"
        else:
            cp_verdict = "Cross-platform presence is MODERATE"

        col_yt, col_ap = st.columns(2)
        with col_yt:
            views_str = f"{views/1e6:.1f}M" if views >= 1_000_000 else (f"{views/1e3:.0f}K" if views >= 1_000 else str(views))
            color = "#22c55e" if views > 1_000_000 else ("#f59e0b" if views > 1_000 else "#e74c3c")
            st.markdown(f"""<div style='background:#1a1a2e;border:1px solid #2a2a4a;border-radius:8px;padding:16px;text-align:center;'>
                <div style='color:#94a3b8;font-size:0.8rem;'>YouTube Views</div>
                <div style='color:{color};font-size:2rem;font-weight:800;'>{views_str if yt_found else "Not found"}</div>
                <div style='color:#64748b;font-size:0.75rem;'>{yt.get("video_title","")[:50]}</div>
            </div>""", unsafe_allow_html=True)
        with col_ap:
            ap_color = "#22c55e" if apple_found else "#e74c3c"
            st.markdown(f"""<div style='background:#1a1a2e;border:1px solid #2a2a4a;border-radius:8px;padding:16px;text-align:center;'>
                <div style='color:#94a3b8;font-size:0.8rem;'>Apple Music</div>
                <div style='color:{ap_color};font-size:2rem;font-weight:800;'>{"✅ Found" if apple_found else "❌ Not found"}</div>
                <div style='color:#64748b;font-size:0.75rem;'>{itunes.get("primary_genre","")}</div>
            </div>""", unsafe_allow_html=True)

        # Rule-based assessment (no OpenAI)
        if views >= 10_000_000:
            presence_label, presence_color = "Strong presence", "#22c55e"
        elif views >= 1_000:
            presence_label, presence_color = "Moderate presence", "#f59e0b"
        elif views > 0:
            presence_label, presence_color = "Minimal presence", "#f59e0b"
        else:
            presence_label, presence_color = "Not found", "#64748b"
        st.markdown(f"""<div style='background:#1a1a2e;border-left:4px solid {presence_color};border-radius:8px;padding:14px 18px;margin-top:12px;'>
            <span style='color:{presence_color};font-weight:700;'>{presence_label}</span>
            <span style='color:#94a3b8;font-size:0.85rem;margin-left:12px;'>{cp_verdict}</span>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Verified Data: Seed Artists")

    for artist_name, data in CROSS_PLATFORM_DATA.items():
        verdict = data["verdict"]
        bc = "#e74c3c" if verdict == "LIKELY_GHOST" else ("#f59e0b" if verdict == "SUSPICIOUS" else "#22c55e")
        vi = "🚨" if verdict == "LIKELY_GHOST" else ("⚠️" if verdict == "SUSPICIOUS" else "✅")
        s7 = data["s7_score"]
        s7c = "#e74c3c" if s7 > 0.7 else ("#f59e0b" if s7 > 0.4 else "#22c55e")
        yt_views = data["youtube_views"]
        yt_str = f"{yt_views/1e6:.0f}M" if yt_views >= 1_000_000 else str(yt_views)
        apple_str = "✅ Apple Music" if data["apple_music"] else "❌ No Apple Music"
        ch_str = f"📺 {data['youtube_channel']}" if data["youtube_channel"] else "No channel"
        st.markdown(f"""<div style='background:#1a1a2e;border-left:4px solid {bc};border-radius:8px;padding:16px 20px;margin:10px 0;'>
            <div style='display:flex;justify-content:space-between;align-items:center;'>
                <span style='color:#e2e8f0;font-size:1.1rem;font-weight:700;'>{vi} {artist_name}</span>
                <span style='color:{s7c};font-weight:700;'>S7: {s7:.2f}</span>
            </div>
            <div style='margin-top:8px;display:flex;gap:24px;flex-wrap:wrap;'>
                <span style='color:#a78bfa;'>▶ {yt_str} views</span>
                <span style='color:#a78bfa;'>{ch_str}</span>
                <span style='color:#a78bfa;'>{apple_str}</span>
            </div>
            <div style='margin-top:8px;color:#94a3b8;font-size:0.85rem;'>{data["note"]}</div>
        </div>""", unsafe_allow_html=True)

    try:
        import plotly.graph_objects as go
        artists_list = list(CROSS_PLATFORM_DATA.keys())
        views_list = [d["youtube_views"] for d in CROSS_PLATFORM_DATA.values()]
        verdicts_list = [d["verdict"] for d in CROSS_PLATFORM_DATA.values()]
        colors_list = ["#e74c3c" if v == "LIKELY_GHOST" else ("#f59e0b" if v == "SUSPICIOUS" else "#22c55e") for v in verdicts_list]
        bar_fig = go.Figure(go.Bar(x=artists_list, y=views_list, marker_color=colors_list,
            text=[f"{v/1e6:.0f}M" if v > 1e6 else str(v) for v in views_list], textposition="outside"))
        bar_fig.update_layout(title="YouTube Views by Artist", yaxis_title="Views",
            plot_bgcolor="#12121f", paper_bgcolor="#12121f", font_color="#e2e8f0",
            title_font_color="#a78bfa", yaxis=dict(gridcolor="#2a2a4a"), showlegend=False)
        st.plotly_chart(bar_fig, use_container_width=True)
    except ImportError:
        pass

    radar_path = FIGURES_DIR / "fig6_signal_radar.png"
    if radar_path.exists():
        with st.expander("View Exercise 6 Signal Radar"):
            st.image(str(radar_path), use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# AI RESEARCH ASSISTANT
# ═════════════════════════════════════════════════════════════════════════════
elif page == "🤖 AI Research Assistant":
    st.markdown("# 🤖 AI Research Assistant")
    st.markdown("Ask any research question about the project. Powered by GPT-4o with full project context.")
    st.info("Each question uses one OpenAI API call.")

    # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Example questions — full-width vertical stack (avoids text compression in narrow layouts)
    st.markdown("**Quick questions:**")
    examples = [
        "What makes Relaxing White Noise a ghost artist?",
        "Why is cross-platform presence unreliable for the relaxation genre?",
        "Compare the signal profiles of all 3 ghost artists",
        "What would it take to definitively classify an unknown artist?",
        "Summarize the key findings from all 5 exercises",
        "What are the main limitations of this analysis?",
        "Draft an abstract for the paper",
        "Explain the 7-layer framework to a non-technical audience",
    ]
    selected_example = None
    for i, ex in enumerate(examples):
        if st.button(ex, key=f"ex_{i}", use_container_width=True):
            selected_example = ex

    # Chat input
    st.markdown("---")
    chat_input = st.chat_input("Ask a research question…")
    question = selected_example or chat_input

    if question:
        with st.spinner("🤖 GPT-4o thinking…"):
            try:
                answer = research_assistant_query(question, st.session_state.chat_history)
                st.session_state.chat_history.append({"user": question, "assistant": answer})
            except Exception as e:
                st.error(f"AI error: {e}")
                answer = None

    # Display conversation
    if st.session_state.chat_history:
        for turn in st.session_state.chat_history:
            st.markdown(f"<div class='chat-user'><strong>You:</strong> {turn['user']}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='chat-assistant'>{turn['assistant']}</div>", unsafe_allow_html=True)

        if st.button("🗑️ Clear conversation"):
            st.session_state.chat_history = []
            st.rerun()
    else:
        st.markdown("<div style='color:#64748b;text-align:center;padding:40px;'>Click a quick question above or type in the chat box to start.</div>", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# ABOUT
# ═════════════════════════════════════════════════════════════════════════════
elif page == "ℹ️ About":
    st.markdown("# ℹ️ About This Project")

    col1, col2 = st.columns([3, 2])
    with col1:
        st.markdown("## Abstract")
        st.markdown("""
        This project presents a **7-layer exploratory data analysis framework** for detecting
        ghost artists — AI-generated or fraudulent accounts used to inflate streaming revenue —
        using only Spotify's public, unauthenticated API endpoints.

        Prior detection work relies on internal data unavailable to researchers.
        We demonstrate that catalog coherence, playlist entropy, ISRC attribution,
        release cadence, metadata similarity, and graph topology each provide
        independent discriminative signal, and that their combination yields robust
        classification without any proprietary access.

        The framework is validated on three seed artists and scaled using the Kaggle
        Spotify Audio Features dataset (114,000 tracks, 114 genres).
        """)
        st.markdown("## Architecture")
        st.markdown("""```
DATA SOURCES        PROCESSING           OUTPUTS
──────────          ──────────           ───────
Spotify API  ──►    SpotifyClient        Neo4j Graph
Kaggle CSV   ──►    7 Signal Modules ──► Signal Scores
YouTube API  ──►    Neo4j Ingestion      Research Paper
iTunes API   ──►    CrewAI Pipeline      paper/figures/
OpenAI GPT-4o ──►  FastAPI Backend      paper/draft.md
                    (this dashboard)
```""")

    with col2:
        st.markdown("## Tech Stack")
        stack = {
            "Data": ["Python 3.14", "pandas 3.0", "numpy 2.4"],
            "APIs": ["spotipy 2.26", "httpx 0.28", "openai 1.x"],
            "Graph": ["Neo4j 6.1", "networkx 3.6"],
            "ML": ["scikit-learn 1.8", "torch 2.11", "torch_geometric 2.7"],
            "Viz": ["matplotlib 3.10", "plotly 6.7", "seaborn 0.13"],
            "Backend": ["FastAPI 0.135", "uvicorn 0.44"],
            "Frontend": ["Streamlit 1.56"],
            "Agents": ["crewai 1.14", "GPT-4o"],
        }
        for category, packages in stack.items():
            st.markdown(f"**{category}:** {' · '.join(packages)}")
        st.markdown("---")
        st.markdown("## Course")
        st.markdown("**INFO 7390** — Advances in Data Science  \nSpring 2026  \nSubmitted by: Trimbkeshwar Jagtap")
