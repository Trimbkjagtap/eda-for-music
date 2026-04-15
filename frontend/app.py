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


def search_kaggle_for_artist(artist_name: str) -> dict:
    """Search Kaggle dataset for an artist's tracks."""
    try:
        import pandas as pd
        df = pd.read_csv(ROOT / "data" / "kaggle" / "dataset.csv")
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


def search_youtube_for_artist(artist_name: str) -> dict:
    """Search YouTube for total channel views."""
    try:
        cache_path = ROOT / "data" / "raw" / "cache" / f"yt_artist_{artist_name.replace(' ','_')[:40]}.json"
        if cache_path.exists():
            return json.loads(cache_path.read_text())

        from src.api.youtube_client import YouTubeClient
        yt = YouTubeClient()
        result = yt.search_video(artist_name, "")
        if result:
            data = {"found": True, "video_title": result.get("title", ""), "views": result.get("view_count", 0)}
        else:
            data = {"found": False, "views": 0}
        cache_path.write_text(json.dumps(data))
        return data
    except Exception as e:
        return {"found": False, "error": str(e)}


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


@st.cache_data(ttl=3600)
def get_ai_key_findings() -> str:
    """Generate AI key findings for the Home page (cached 1 hour)."""
    client = get_openai_client()
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": f"You are a research assistant summarizing findings from the EDA for Music project.\n{PROJECT_CONTEXT}"},
            {"role": "user", "content": "List the 5 most important and surprising findings from this project as a numbered list. Each finding should be 1-2 sentences, specific, and cite actual numbers. Format as plain text with numbers 1-5."},
        ],
        temperature=0.3,
        max_tokens=500,
    )
    return response.choices[0].message.content


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


def ai_interpret_cross_platform(artist_name: str, yt_views: int, apple_music: bool, verdict: str) -> str:
    """GPT-4o interpretation of cross-platform data."""
    client = get_openai_client()
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": f"You are a ghost artist detection researcher. {PROJECT_CONTEXT}"},
            {"role": "user", "content": f"""Interpret the cross-platform data for {artist_name}:
- YouTube views: {yt_views:,}
- On Apple Music: {apple_music}
- Current ghost verdict: {verdict}

Compare to baselines (RWN=353M views, MRC=157M views, Calmo=155 views, NF=9M views).
What does this cross-platform presence tell us about this artist's behavior?
Is cross-platform presence consistent or inconsistent with the ghost verdict?
Write 2-3 sentences, cite specific numbers, be precise."""},
        ],
        temperature=0.3,
        max_tokens=300,
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

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("<div class='metric-card'><div class='number'>3</div><div class='label'>Ghost Artists Analyzed</div></div>", unsafe_allow_html=True)
    with col2:
        st.markdown("<div class='metric-card'><div class='number'>490</div><div class='label'>Tracks in Neo4j</div></div>", unsafe_allow_html=True)
    with col3:
        st.markdown("<div class='metric-card'><div class='number'>7</div><div class='label'>Detection Signals</div></div>", unsafe_allow_html=True)
    with col4:
        st.markdown("<div class='metric-card'><div class='number'>114K</div><div class='label'>Kaggle Training Tracks</div></div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # AI Key Findings
    st.markdown("## 🤖 AI-Generated Key Findings")
    with st.spinner("Generating research insights…"):
        try:
            findings = get_ai_key_findings()
            st.markdown(f"""<div class='ai-card'>
                <div style='color:#a78bfa;font-size:0.8rem;font-weight:700;margin-bottom:8px;'>GPT-4o ANALYSIS</div>
                <div style='color:#e2e8f0;line-height:1.7;'>{findings.replace(chr(10), '<br>')}</div>
            </div>""", unsafe_allow_html=True)
        except Exception as e:
            st.info(f"AI findings unavailable: {e}")

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
# ARTIST ANALYZER (AI-Powered)
# ═════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Artist Analyzer":
    st.markdown("# 🔍 Artist Analyzer")
    st.markdown("Search **any artist** by name. The AI will collect all available data and provide a PhD-level ghost detection analysis.")

    search_col, btn_col = st.columns([4, 1])
    with search_col:
        artist_input = st.text_input(
            "Artist name or Spotify ID",
            placeholder='e.g. "Drake", "Calmo", "Relaxing White Noise", or paste a Spotify ID',
            label_visibility="collapsed",
        )
    with btn_col:
        analyze_btn = st.button("🔍 Analyze", type="primary", use_container_width=True)

    # Quick-pick
    st.markdown("<div style='color:#64748b;font-size:0.8rem;margin-bottom:4px;'>Quick-pick:</div>", unsafe_allow_html=True)
    qp_cols = st.columns(4)
    quick_picks = ["Relaxing White Noise", "Meditation Relax Club", "Calmo", "Nils Frahm"]
    for i, qp in enumerate(quick_picks):
        if qp_cols[i].button(qp, key=f"qp_{i}"):
            artist_input = qp
            analyze_btn = True

    if analyze_btn and artist_input.strip():
        artist_query = artist_input.strip()

        with st.spinner(f"Collecting data for **{artist_query}**…"):

            # ── Phase 1: Data collection ──────────────────────────────────────
            collected_data = {"artist_name": artist_query, "sources": {}}

            # Check if it looks like a Spotify ID (22 chars, alphanumeric)
            is_spotify_id = len(artist_query) == 22 and artist_query.replace("_","").replace("-","").isalnum()

            # Neo4j check
            st.caption("🔍 Checking Neo4j cache…")
            neo4j_data = check_neo4j_for_artist(artist_query)
            collected_data["neo4j"] = neo4j_data
            collected_data["sources"]["neo4j"] = neo4j_data["found"]

            # Kaggle check
            st.caption("🔍 Searching Kaggle dataset (114K tracks)…")
            kaggle_data = search_kaggle_for_artist(artist_query)
            collected_data["kaggle"] = kaggle_data
            collected_data["sources"]["kaggle"] = kaggle_data["found"]

            # YouTube
            st.caption("🔍 Searching YouTube…")
            yt_data = search_youtube_for_artist(artist_query)
            collected_data["youtube"] = yt_data
            collected_data["sources"]["youtube"] = yt_data.get("found", False)

            # iTunes
            st.caption("🔍 Checking Apple Music (iTunes)…")
            itunes_data = search_itunes_for_artist(artist_query)
            collected_data["apple_music"] = itunes_data
            collected_data["sources"]["apple_music"] = itunes_data.get("found", False)

            # If in Neo4j, run full pipeline
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

        # ── Phase 2: AI Analysis ──────────────────────────────────────────────
        with st.spinner("🤖 GPT-4o analyzing for ghost behavior…"):
            try:
                ai_result = ai_analyze_artist(artist_query, collected_data)
                analysis_ok = True
            except Exception as e:
                ai_result = {"verdict": "INSUFFICIENT_DATA", "ghost_probability": 50, "confidence": 0, "analysis": f"AI analysis failed: {e}", "estimated_signals": {}, "comparison": "", "missing_data": [], "key_signals": []}
                analysis_ok = False

        # ── Display Results ───────────────────────────────────────────────────
        st.markdown("---")
        st.markdown(f"## Analysis: {artist_query}")

        # Data sources checklist
        sources = collected_data.get("sources", {})
        src_cols = st.columns(5)
        src_cols[0].markdown(f"{'✅' if sources.get('neo4j') else '❌'} Neo4j cache", unsafe_allow_html=True)
        src_cols[1].markdown(f"{'✅' if sources.get('kaggle') else '❌'} Kaggle dataset", unsafe_allow_html=True)
        src_cols[2].markdown(f"{'✅' if sources.get('youtube') else '❌'} YouTube", unsafe_allow_html=True)
        src_cols[3].markdown(f"{'✅' if sources.get('apple_music') else '❌'} Apple Music", unsafe_allow_html=True)
        src_cols[4].markdown(f"{'✅' if signal_scores else '❌'} Signal pipeline", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Verdict header
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

        hdr_c1, hdr_c2, hdr_c3, hdr_c4 = st.columns(4)
        with hdr_c1:
            st.markdown(f"<span class='{badge_class}'>{badge_icon} {verdict}</span>", unsafe_allow_html=True)
        with hdr_c2:
            st.metric("Ghost Probability", f"{ghost_prob}%")
        with hdr_c3:
            st.metric("AI Confidence", f"{confidence}%")
        with hdr_c4:
            if gnn_score is not None:
                st.metric("GNN Score (GAT)", f"{gnn_score:.3f}")
            elif rule_score is not None:
                st.metric("Rule-based Score", f"{rule_score:.3f}")

        # Ghost probability gauge
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

        # AI Analysis card
        st.markdown("### 🤖 Expert Analysis")
        analysis_text = ai_result.get("analysis", "")
        st.markdown(f"""<div class='ai-card'>
            <div style='color:#a78bfa;font-size:0.8rem;font-weight:700;margin-bottom:12px;'>GPT-4o ANALYSIS — {artist_query.upper()}</div>
            <div style='color:#e2e8f0;line-height:1.8;font-size:0.92rem;'>{analysis_text.replace(chr(10), "<br>")}</div>
        </div>""", unsafe_allow_html=True)

        # Comparison to baselines
        comparison = ai_result.get("comparison", "")
        if comparison:
            with st.expander("📊 Comparison to Known Baselines"):
                st.markdown(comparison)
                # Mini comparison table
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

        # Signal scores (AI estimated + pipeline if available)
        st.markdown("### Signal Scores")
        estimated = ai_result.get("estimated_signals", {})
        SIGNAL_META_DISPLAY = [
            ("s1_audio_similarity",     "s1_audio",    "S1", "Audio Fingerprint Similarity"),
            ("s2_cadence_sync",         "s2_cadence",  "S2", "Release Cadence Synchrony"),
            ("s3_playlist_cooccurrence","s3_playlist", "S3", "Playlist Co-occurrence"),
            ("s4_follower_ratio",       "s4_catalog",  "S4", "Catalog Density Anomaly"),
            ("s5_metadata_similarity",  "s5_metadata", "S5", "Metadata Similarity"),
            ("s6_graph_density",        "s6_graph",    "S6", "Graph Density / HHI"),
            ("s7_cross_platform",       "s7_crossplatform", "S7", "Cross-Platform Discrepancy"),
        ]
        for pipeline_key, ai_key, num, name in SIGNAL_META_DISPLAY:
            pipeline_val = (signal_scores or {}).get(pipeline_key)
            ai_val = estimated.get(ai_key)
            # Prefer pipeline value, fall back to AI estimate
            val = pipeline_val if pipeline_val is not None else ai_val
            source_tag = "pipeline" if pipeline_val is not None else ("AI estimate" if ai_val is not None else "N/A")

            if val is None:
                score_str, score_color, bar_pct = "N/A", "#64748b", 0
            else:
                score_str = f"{val:.3f}"
                bar_pct = int(val * 100)
                score_color = "#e74c3c" if val > 0.7 else ("#f59e0b" if val > 0.4 else "#22c55e")

            st.markdown(f"""<div class='signal-card'>
                <span class='score' style='color:{score_color};'>{score_str}</span>
                <div class='name'>{num}: {name} <span style='color:#64748b;font-size:0.75rem;font-weight:400;'>({source_tag})</span></div>
                <div style='margin-top:6px;background:#2a2a4a;border-radius:4px;height:6px;'>
                    <div style='width:{bar_pct}%;background:{score_color};border-radius:4px;height:6px;'></div>
                </div>
            </div>""", unsafe_allow_html=True)

        # Key signals + missing data
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
        st.error("Please enter an artist name.")

    elif not analyze_btn:
        st.markdown("<div style='color:#64748b;text-align:center;padding:40px;'>Enter an artist name above and click Analyze to run AI-powered ghost detection.</div>", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# NETWORK EXPLORER
# ═════════════════════════════════════════════════════════════════════════════
elif page == "🌐 Network Explorer":
    st.markdown("# 🌐 Network Explorer")
    st.markdown("ISRC-based production company graph. High HHI concentration (> 0.6) signals bulk-upload operations.")
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

        with st.spinner("🤖 GPT-4o interpreting cross-platform data…"):
            try:
                interp = ai_interpret_cross_platform(cp_input, views, apple_found, cp_verdict)
                st.markdown(f"""<div class='ai-card'>
                    <div style='color:#a78bfa;font-size:0.8rem;font-weight:700;margin-bottom:8px;'>GPT-4o INTERPRETATION</div>
                    <div style='color:#e2e8f0;'>{interp}</div>
                </div>""", unsafe_allow_html=True)
            except Exception as e:
                st.info(f"AI interpretation unavailable: {e}")

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

    # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Example questions
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
    ex_cols = st.columns(4)
    selected_example = None
    for i, ex in enumerate(examples):
        if ex_cols[i % 4].button(ex[:35] + ("…" if len(ex) > 35 else ""), key=f"ex_{i}"):
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
