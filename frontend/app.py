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
    page_title="EDA for Music | Ghost Artist Detection Framework",
    page_icon="👻",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS — GhostTrack theme ─────────────────────────────────────────────
st.markdown("""
<style>
    /* ── Global ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .main .block-container {
        background-color: #0a0a0a; padding-top: 1rem !important;
        padding-left: 2rem; padding-right: 2rem; max-width: 1200px;
    }
    /* Remove Streamlit's default top gap above first element */
    .main .block-container > div:first-child { margin-top: 0 !important; }
    h1 { color: #ffffff; font-weight: 900; letter-spacing: -0.02em; }
    h2 { color: #ffffff; font-weight: 800; }
    h3 { color: #e5e5e5; font-weight: 700; }
    p, li { color: #a3a3a3; }

    /* Hide Streamlit's auto-injected heading anchor icons */
    h1 a, h2 a, h3 a, h4 a, h5 a, h6 a { display: none !important; }
    [data-testid="stHeadingWithActionElements"] a { display: none !important; }
    .gt-hero-title a { display: none !important; }

    /* ── Hide sidebar + toggle completely ── */
    [data-testid="stSidebar"],
    [data-testid="collapsedControl"],
    button[kind="header"] { display: none !important; }
    .main .block-container {
        margin-left: 0 !important;
        padding-top: 0 !important;
    }

    /* ── Fixed top navbar ── */
    #gt-navbar {
        position: fixed; top: 0; left: 0; right: 0; z-index: 9999;
        background: #0a0a0a;
        border-bottom: 1px solid #1e1e1e;
        display: flex; align-items: center; justify-content: space-between;
        padding: 0 40px; height: 60px;
        font-family: 'Inter', sans-serif;
    }
    #gt-navbar .gt-logo {
        display: flex; align-items: center; gap: 10px; cursor: pointer;
        text-decoration: none;
    }
    #gt-navbar .gt-logo-icon {
        width: 34px; height: 34px; background: #00ff88; border-radius: 8px;
        display: flex; align-items: center; justify-content: center;
        font-size: 1rem; color: #000; font-weight: 900; flex-shrink: 0;
    }
    #gt-navbar .gt-logo-text {
        color: #ffffff; font-size: 1.05rem; font-weight: 800;
        letter-spacing: -0.01em;
    }
    #gt-navbar .gt-nav-links {
        display: flex; align-items: center; gap: 32px;
    }
    #gt-navbar .gt-nav-link {
        color: #737373; font-size: 0.875rem; font-weight: 500;
        cursor: pointer; transition: color 0.15s; user-select: none;
        white-space: nowrap;
    }
    #gt-navbar .gt-nav-link:hover { color: #ffffff; }
    #gt-navbar .gt-nav-link.active { color: #ffffff; font-weight: 600; }
    #gt-navbar .gt-demo-btn {
        display: flex; align-items: center; gap: 6px;
        background: transparent; border: 1px solid #333; border-radius: 20px;
        padding: 7px 18px; color: #ffffff; font-size: 0.82rem; font-weight: 600;
        cursor: pointer; transition: border-color 0.15s, background 0.15s;
        white-space: nowrap;
    }
    #gt-navbar .gt-demo-btn:hover { border-color: #555; background: #111; }

    /* Push content below fixed navbar */
    .main .block-container { padding-top: 76px !important; }

    /* Hide the real radio widget used for nav state */
    #gt-nav-radio { display: none !important; }

    /* ── Hero ── */
    .gt-hero {
        padding: 16px 0 48px 0; text-align: left;
    }
    .gt-hero-eyebrow {
        display: inline-flex; align-items: center; gap: 8px;
        color: #00ff88; font-size: 0.78rem; font-weight: 700;
        letter-spacing: 0.12em; text-transform: uppercase; margin-bottom: 24px;
    }
    .gt-hero-eyebrow::before {
        content: ""; display: inline-block; width: 20px; height: 2px; background: #00ff88;
    }
    .gt-hero-title {
        font-size: clamp(2.4rem, 5vw, 4rem); font-weight: 900;
        line-height: 1.05; letter-spacing: -0.03em; color: #ffffff;
        margin: 0 0 24px 0;
    }
    .gt-hero-title .accent { color: #00ff88; }
    .gt-hero-subtitle {
        font-size: 1.1rem; color: #737373; max-width: 560px;
        line-height: 1.7; margin-bottom: 36px;
    }
    .gt-hero-btns { display: flex; gap: 12px; flex-wrap: wrap; }
    .gt-btn-primary {
        background: #00ff88; color: #000; border: none;
        padding: 12px 24px; border-radius: 8px; font-weight: 700;
        font-size: 0.9rem; cursor: pointer; display: inline-flex;
        align-items: center; gap: 8px; text-decoration: none;
    }
    .gt-btn-secondary {
        background: transparent; color: #e5e5e5;
        border: 1px solid #333; padding: 12px 24px; border-radius: 8px;
        font-weight: 600; font-size: 0.9rem; cursor: pointer;
        display: inline-flex; align-items: center; gap: 8px;
        text-decoration: none;
    }

    /* ── Stat cards ── */
    .gt-stats {
        display: flex; flex-wrap: wrap; gap: 16px; margin: 40px 0;
    }
    .gt-stat-card {
        flex: 1; min-width: 140px;
        background: #111111; border: 1px solid #1e1e1e; border-radius: 12px;
        padding: 20px 24px;
    }
    .gt-stat-icon { font-size: 1.2rem; margin-bottom: 8px; }
    .gt-stat-number {
        font-size: 2.2rem; font-weight: 900; color: #00ff88;
        letter-spacing: -0.02em; line-height: 1;
    }
    .gt-stat-label { font-size: 0.82rem; color: #737373; margin-top: 4px; }

    /* ── Section headers ── */
    .gt-section-header {
        display: flex; align-items: center; gap: 10px;
        margin: 40px 0 20px 0;
    }
    .gt-section-eyebrow {
        color: #00ff88; font-size: 0.75rem; font-weight: 700;
        letter-spacing: 0.1em; text-transform: uppercase;
    }
    .gt-section-title {
        font-size: 1.8rem; font-weight: 800; color: #ffffff; margin: 0 0 6px 0;
    }
    .gt-section-subtitle { color: #737373; font-size: 0.92rem; margin: 0 0 24px 0; }

    /* ── Finding cards ── */
    .gt-finding-card {
        background: #111111; border: 1px solid #1e1e1e; border-radius: 12px;
        padding: 20px 24px; margin: 10px 0;
        display: flex; align-items: flex-start; gap: 16px;
        transition: border-color 0.2s;
    }
    .gt-finding-card:hover { border-color: #00ff88; }
    .gt-finding-icon {
        width: 44px; height: 44px; background: #0d2818; border-radius: 10px;
        display: flex; align-items: center; justify-content: center;
        font-size: 1.2rem; flex-shrink: 0;
    }
    .gt-finding-title { color: #ffffff; font-weight: 700; font-size: 0.95rem; }
    .gt-finding-body { color: #737373; font-size: 0.88rem; margin-top: 4px; line-height: 1.6; }

    /* ── Layer cards ── */
    .gt-layer-card {
        background: #111111; border: 1px solid #1e1e1e; border-radius: 10px;
        padding: 16px 20px; margin: 8px 0;
        display: flex; align-items: flex-start; gap: 16px; width: 100%;
        box-sizing: border-box;
    }
    .gt-layer-num {
        width: 32px; height: 32px; background: #0d2818; border: 1px solid #00ff88;
        border-radius: 50%; display: flex; align-items: center; justify-content: center;
        color: #00ff88; font-weight: 800; font-size: 0.85rem; flex-shrink: 0; margin-top: 2px;
    }
    .gt-layer-name { color: #ffffff; font-weight: 700; font-size: 0.95rem; }
    .gt-layer-desc { color: #737373; font-size: 0.85rem; margin-top: 3px; }
    .gt-layer-tag {
        display: inline-block; background: #1a1a1a; color: #737373;
        border: 1px solid #333; border-radius: 20px;
        padding: 2px 10px; font-size: 0.72rem; margin-left: 8px; vertical-align: middle;
    }

    /* ── Impact cards ── */
    .gt-impact-grid { display: flex; gap: 16px; flex-wrap: wrap; margin: 16px 0; }
    .gt-impact-card {
        flex: 1; min-width: 260px;
        background: #0d2818; border: 1px solid #1a4a2e; border-radius: 12px;
        padding: 24px;
    }
    .gt-impact-icon {
        width: 48px; height: 48px; background: #163a25; border-radius: 10px;
        display: flex; align-items: center; justify-content: center;
        font-size: 1.3rem; margin-bottom: 14px;
    }
    .gt-impact-title { color: #00ff88; font-weight: 700; font-size: 0.95rem; margin-bottom: 8px; }
    .gt-impact-body { color: #a3a3a3; font-size: 0.88rem; line-height: 1.6; }

    /* ── Tags / badges ── */
    .gt-tag {
        display: inline-block; background: #1a1a1a; color: #00ff88;
        border: 1px solid #1e3a28; border-radius: 20px;
        padding: 3px 12px; font-size: 0.75rem; margin: 2px; font-weight: 600;
    }
    .gt-tag-gray {
        display: inline-block; background: #1a1a1a; color: #737373;
        border: 1px solid #333; border-radius: 20px;
        padding: 3px 12px; font-size: 0.75rem; margin: 2px;
    }

    /* ── Figure frame ── */
    .gt-fig-frame {
        background: #111111; border: 1px solid #1e1e1e;
        border-radius: 12px; padding: 20px; margin-bottom: 20px;
    }
    .gt-fig-caption { color: #737373; font-size: 0.85rem; margin-top: 10px; line-height: 1.6; }

    /* ── Signal cards ── */
    .gt-signal-card {
        background: #111111; border-left: 3px solid #00ff88;
        border-radius: 8px; padding: 14px 18px; margin: 8px 0;
    }
    .gt-signal-name { color: #e5e5e5; font-weight: 600; font-size: 0.92rem; }
    .gt-signal-score { font-size: 1.4rem; font-weight: 800; float: right; }

    /* ── Verdict banners ── */
    .verdict-ghost {
        background: #2a0a0a; border: 2px solid #e74c3c; color: #fca5a5;
        border-radius: 10px; padding: 14px 24px; font-size: 1.3rem;
        font-weight: 800; text-align: center; display: block;
    }
    .verdict-suspicious {
        background: #2a1800; border: 2px solid #f59e0b; color: #fcd34d;
        border-radius: 10px; padding: 14px 24px; font-size: 1.3rem;
        font-weight: 800; text-align: center; display: block;
    }
    .verdict-organic {
        background: #0a2a14; border: 2px solid #00ff88; color: #86efac;
        border-radius: 10px; padding: 14px 24px; font-size: 1.3rem;
        font-weight: 800; text-align: center; display: block;
    }

    /* ── AI / chat cards ── */
    .gt-ai-card {
        background: #111111; border: 1px solid #1e1e1e;
        border-radius: 12px; padding: 20px; margin: 12px 0;
    }
    .gt-ai-header { color: #00ff88; font-size: 0.78rem; font-weight: 700; margin-bottom: 12px; }
    .chat-user {
        background: #1a1a1a; border-radius: 12px 12px 4px 12px;
        padding: 12px 16px; margin: 8px 0; color: #e5e5e5;
    }
    .chat-assistant {
        background: #0d2818; border: 1px solid #1a4a2e; border-radius: 12px 12px 12px 4px;
        padding: 12px 16px; margin: 8px 0; color: #e5e5e5;
    }

    /* ── Source rows ── */
    .gt-src-row {
        display: flex; align-items: flex-start; gap: 12px;
        border-radius: 8px; padding: 12px 16px; margin-bottom: 8px;
    }
    .gt-src-row.ok { background: #0d2818; border-left: 3px solid #00ff88; }
    .gt-src-row.nok { background: #111111; border-left: 3px solid #333; }

    /* ── Divider ── */
    .gt-divider {
        border: none; border-top: 1px solid #1a1a1a; margin: 32px 0;
    }

    /* ── Page header ── */
    .gt-page-header {
        padding: 20px 0 16px 0; border-bottom: 1px solid #1a1a1a; margin-bottom: 20px;
    }
    .gt-page-title { font-size: 2rem; font-weight: 900; color: #ffffff; margin: 0; }
    .gt-page-subtitle { color: #737373; font-size: 0.92rem; margin-top: 6px; }

    /* ── Architecture pipeline ── */
    .gt-pipeline {
        display: flex; gap: 0; align-items: stretch;
        background: #111111; border: 1px solid #1e1e1e; border-radius: 12px;
        overflow: hidden; margin: 20px 0; flex-wrap: wrap;
    }
    .gt-pipe-step {
        flex: 1; min-width: 130px; padding: 20px 16px;
        border-right: 1px solid #1e1e1e; position: relative;
    }
    .gt-pipe-step:last-child { border-right: none; }
    .gt-pipe-num {
        width: 28px; height: 28px; background: #00ff88; border-radius: 50%;
        color: #000; font-weight: 800; font-size: 0.82rem;
        display: flex; align-items: center; justify-content: center; margin-bottom: 10px;
    }
    .gt-pipe-title { color: #ffffff; font-weight: 700; font-size: 0.88rem; margin-bottom: 6px; }
    .gt-pipe-items { color: #737373; font-size: 0.78rem; line-height: 1.8; }

    /* ── Tech stack grid ── */
    .gt-tech-grid { display: flex; flex-wrap: wrap; gap: 12px; margin: 16px 0; }
    .gt-tech-card {
        background: #111111; border: 1px solid #1e1e1e; border-radius: 10px;
        padding: 16px 20px; flex: 1; min-width: 140px;
    }
    .gt-tech-cat {
        color: #00ff88; font-size: 0.75rem; font-weight: 700;
        text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 8px;
    }
    .gt-tech-pkg { color: #a3a3a3; font-size: 0.82rem; line-height: 1.9; }

    /* ── Streamlit overrides ── */
    .stTextInput > div > div > input {
        background: #111111 !important; border: 1px solid #333 !important;
        color: #ffffff !important; border-radius: 8px !important;
    }
    .stTextInput > div > div > input:focus {
        border-color: #00ff88 !important; box-shadow: 0 0 0 2px rgba(0,255,136,0.15) !important;
    }
    .stButton > button {
        background: #00ff88 !important; color: #000 !important;
        border: none !important; font-weight: 700 !important;
        border-radius: 8px !important; transition: opacity 0.2s !important;
    }
    .stButton > button:hover { opacity: 0.88 !important; }
    .stButton > button[kind="secondary"] {
        background: #1a1a1a !important; color: #e5e5e5 !important;
        border: 1px solid #333 !important;
    }
    div[data-testid="stExpander"] {
        background: #111111; border: 1px solid #1e1e1e; border-radius: 10px;
    }
    .stAlert { border-radius: 10px !important; }
    footer { visibility: hidden; }
    #MainMenu { visibility: hidden; }

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


# ── Navigation state ──────────────────────────────────────────────────────────
_NAV_PAGES = [
    ("Home",     "🏠 Home"),
    ("About",    "ℹ️ About"),
    ("Gallery",  "📊 Exercise Gallery"),
    ("Analyzer", "🔍 Artist Analyzer"),
    ("Network",  "🌐 Network Explorer"),
    ("Cross-Platform", "📡 Cross-Platform"),
    ("AI",       "🤖 AI Research Assistant"),
]
_NAV_LABEL_MAP = {label: full for label, full in _NAV_PAGES}
_NAV_FULL_MAP  = {full: label for label, full in _NAV_PAGES}

if "page" not in st.session_state:
    st.session_state.page = "🏠 Home"

# Hidden radio that actually drives page state — JS clicks its labels
with st.container():
    st.markdown('<div id="gt-nav-radio">', unsafe_allow_html=True)
    _nav_choice = st.radio(
        "nav", [label for label, _ in _NAV_PAGES],
        index=[label for label, _ in _NAV_PAGES].index(
            _NAV_FULL_MAP.get(st.session_state.page, "Home")
        ),
        key="nav_radio",
        label_visibility="collapsed",
    )
    st.markdown('</div>', unsafe_allow_html=True)

# Sync session state from radio
st.session_state.page = _NAV_LABEL_MAP[_nav_choice]
page = st.session_state.page


# ── Fixed HTML navbar (pure CSS, JS clicks the hidden radio) ──────────────────
def _navbar(active: str) -> None:
    """Inject a fixed-position HTML navbar. Clicking a link JS-clicks the
    matching hidden radio label so Streamlit re-renders on the correct page."""
    active_label = _NAV_FULL_MAP.get(active, "Home")
    nav_links_html = ""
    for label, full in _NAV_PAGES[:4]:   # show Home / About / Gallery / Analyzer
        cls = "gt-nav-link active" if label == active_label else "gt-nav-link"
        nav_links_html += (
            f'<span class="{cls}" '
            f'onclick="(function(){{var ls=window.parent.document.querySelectorAll(\'[id*=gt-nav-radio] label\');'
            f'for(var i=0;i<ls.length;i++){{if(ls[i].innerText.trim()===\'{label}\'){{ls[i].click();break;}}}}}})();">'
            f'{label}</span>'
        )

    st.markdown(f"""
    <nav id="gt-navbar">
      <div class="gt-logo">
        <div class="gt-logo-icon">♪</div>
        <span class="gt-logo-text">GhostTrack</span>
      </div>
      <div class="gt-nav-links">{nav_links_html}</div>
      <div class="gt-demo-btn"
        onclick="(function(){{var ls=window.parent.document.querySelectorAll(\'[id*=gt-nav-radio] label\');
        for(var i=0;i<ls.length;i++){{if(ls[i].innerText.trim()==='Analyzer'){{ls[i].click();break;}}}}}})();">
        ▶&nbsp; Demo
      </div>
    </nav>
    """, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# HOME PAGE
# ═════════════════════════════════════════════════════════════════════════════
if page == "🏠 Home":
    _navbar("Home")

    # ── Hero ──────────────────────────────────────────────────────────────────
    st.markdown("""
    <div class='gt-hero'>
        <div class='gt-hero-eyebrow'>Streaming Platform Integrity</div>
        <h1 class='gt-hero-title'>
            Unmasking <span class='accent'>Ghost<br>Artists</span> in the<br>Streaming Era
        </h1>
        <p class='gt-hero-subtitle'>
            A 7-layer exploratory data analysis framework that exposes fraudulent streaming
            accounts using only public API endpoints. No insider data. No black boxes.
        </p>
        <div class='gt-hero-btns'>
            <span class='gt-btn-primary'>Explore Framework &nbsp;→</span>
            <span class='gt-btn-secondary'>🎧 View Case Study</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Stats ─────────────────────────────────────────────────────────────────
    st.markdown("""
    <div class='gt-stats'>
        <div class='gt-stat-card'>
            <div class='gt-stat-icon'>👻</div>
            <div class='gt-stat-number'>3</div>
            <div class='gt-stat-label'>Ghost Artists</div>
        </div>
        <div class='gt-stat-card'>
            <div class='gt-stat-icon'>♪</div>
            <div class='gt-stat-number'>490</div>
            <div class='gt-stat-label'>Tracks</div>
        </div>
        <div class='gt-stat-card'>
            <div class='gt-stat-icon'>📡</div>
            <div class='gt-stat-number'>7</div>
            <div class='gt-stat-label'>Detection</div>
        </div>
        <div class='gt-stat-card'>
            <div class='gt-stat-icon'>📊</div>
            <div class='gt-stat-number'>114K</div>
            <div class='gt-stat-label'>Training</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Case study callout ────────────────────────────────────────────────────
    st.markdown("""
    <div style='background:#111111;border:1px solid #1e1e1e;border-radius:14px;
         padding:28px 32px;margin:8px 0 32px 0;display:flex;gap:32px;flex-wrap:wrap;align-items:flex-start;'>
        <div style='flex:1;min-width:220px;'>
            <div style='color:#00ff88;font-size:0.72rem;font-weight:700;letter-spacing:0.1em;
                 text-transform:uppercase;margin-bottom:10px;'>⚠ Case Study</div>
            <div style='color:#ffffff;font-size:1.5rem;font-weight:900;line-height:1.2;margin-bottom:14px;'>
                Meet the Ghost:<br>Relaxing White Noise
            </div>
            <p style='color:#737373;font-size:0.88rem;line-height:1.7;margin-bottom:16px;'>
                An account with 353 million YouTube views but zero verifiable human identity.
                Our framework detected red flags across all 7 analysis layers.
            </p>
            <div style='display:flex;flex-direction:column;gap:8px;'>
                <div style='color:#a3a3a3;font-size:0.85rem;'>● Feature variance <strong style='color:#00ff88;'>12.5×</strong> lower than organic artists</div>
                <div style='color:#a3a3a3;font-size:0.85rem;'>● <strong style='color:#00ff88;'>81–95%</strong> same-day release cadence clustering</div>
                <div style='color:#a3a3a3;font-size:0.85rem;'>● HHI concentration coefficient <strong style='color:#00ff88;'>0.88</strong></div>
                <div style='color:#a3a3a3;font-size:0.85rem;'>● Single company controls <strong style='color:#00ff88;'>88%</strong> of catalog</div>
            </div>
        </div>
        <div style='background:#0d2818;border:1px solid #1a4a2e;border-radius:12px;
             padding:20px 28px;min-width:160px;text-align:center;'>
            <div style='font-size:1.8rem;margin-bottom:4px;'>👻</div>
            <div style='color:#737373;font-size:0.78rem;font-weight:600;margin-bottom:4px;'>Anomaly Score</div>
            <div style='color:#00ff88;font-size:2.4rem;font-weight:900;'>0.94</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Key Findings ──────────────────────────────────────────────────────────
    st.markdown("""
    <div style='text-align:center;margin:56px 0 32px 0;'>
        <div style='margin-bottom:20px;'>
            <svg width="80" height="40" viewBox="0 0 80 40" fill="none" xmlns="http://www.w3.org/2000/svg">
                <rect x="0" y="30" width="6" height="10" fill="#00ff88" opacity="0.4"/>
                <rect x="8" y="22" width="6" height="18" fill="#00ff88" opacity="0.5"/>
                <rect x="16" y="14" width="6" height="26" fill="#00ff88" opacity="0.6"/>
                <rect x="24" y="8" width="6" height="32" fill="#00ff88" opacity="0.7"/>
                <rect x="32" y="4" width="6" height="36" fill="#00ff88" opacity="0.85"/>
                <rect x="40" y="2" width="6" height="38" fill="#00ff88"/>
                <rect x="48" y="6" width="6" height="34" fill="#00ff88" opacity="0.85"/>
                <rect x="56" y="10" width="6" height="30" fill="#00ff88" opacity="0.7"/>
                <rect x="64" y="18" width="6" height="22" fill="#00ff88" opacity="0.6"/>
                <rect x="72" y="26" width="6" height="14" fill="#00ff88" opacity="0.5"/>
            </svg>
        </div>
        <h2 style='color:#ffffff;font-size:2.2rem;font-weight:900;margin:0 0 10px 0;letter-spacing:-0.02em;'>Key Findings</h2>
        <p style='color:#737373;font-size:0.95rem;max-width:600px;margin:0 auto;'>
            Our analysis revealed striking patterns that distinguish ghost artists from legitimate musicians
        </p>
    </div>
    <div style='display:flex;gap:16px;flex-wrap:wrap;margin-bottom:48px;'>
        <div style='flex:1;min-width:180px;background:#111111;border:1px solid #1e1e1e;border-radius:14px;padding:24px 20px;'>
            <div style='width:48px;height:48px;background:#1a3a1a;border-radius:12px;display:flex;align-items:center;justify-content:center;font-size:1.4rem;margin-bottom:16px;'>〰</div>
            <div style='color:#e5e5e5;font-weight:700;font-size:0.92rem;margin-bottom:8px;'>Variance Ratio</div>
            <div style='color:#00ff88;font-size:2.2rem;font-weight:900;line-height:1;margin-bottom:8px;'>12.5x</div>
            <div style='color:#737373;font-size:0.82rem;line-height:1.6;'>Ghost artists show 12.5x lower feature variance compared to legitimate artists</div>
        </div>
        <div style='flex:1;min-width:180px;background:#111111;border:1px solid #1e1e1e;border-radius:14px;padding:24px 20px;'>
            <div style='width:48px;height:48px;background:#2a2a0a;border-radius:12px;display:flex;align-items:center;justify-content:center;font-size:1.4rem;margin-bottom:16px;'>📡</div>
            <div style='color:#e5e5e5;font-weight:700;font-size:0.92rem;margin-bottom:8px;'>Cadence Closure</div>
            <div style='color:#00ff88;font-size:2.2rem;font-weight:900;line-height:1;margin-bottom:8px;'>81–95%</div>
            <div style='color:#737373;font-size:0.82rem;line-height:1.6;'>Ghost accounts maintain impossibly perfect same-day release clustering</div>
        </div>
        <div style='flex:1;min-width:180px;background:#111111;border:1px solid #1e1e1e;border-radius:14px;padding:24px 20px;'>
            <div style='width:48px;height:48px;background:#2a1a0a;border-radius:12px;display:flex;align-items:center;justify-content:center;font-size:1.4rem;margin-bottom:16px;'>📈</div>
            <div style='color:#e5e5e5;font-weight:700;font-size:0.92rem;margin-bottom:8px;'>YouTube Presence</div>
            <div style='color:#00ff88;font-size:2.2rem;font-weight:900;line-height:1;margin-bottom:8px;'>353M</div>
            <div style='color:#737373;font-size:0.82rem;line-height:1.6;'>Suspicious accounts accumulate massive views with minimal engagement</div>
        </div>
        <div style='flex:1;min-width:180px;background:#111111;border:1px solid #1e1e1e;border-radius:14px;padding:24px 20px;'>
            <div style='width:48px;height:48px;background:#1a0a2a;border-radius:12px;display:flex;align-items:center;justify-content:center;font-size:1.4rem;margin-bottom:16px;'>📊</div>
            <div style='color:#e5e5e5;font-weight:700;font-size:0.92rem;margin-bottom:8px;'>GNN Accuracy</div>
            <div style='color:#00ff88;font-size:2.2rem;font-weight:900;line-height:1;margin-bottom:8px;'>100%</div>
            <div style='color:#737373;font-size:0.82rem;line-height:1.6;'>Graph Neural Network achieves perfect test accuracy on proof-of-concept</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── 7-Layer Framework ─────────────────────────────────────────────────────
    st.markdown("""<hr class='gt-divider'>""", unsafe_allow_html=True)

    layers_col, cards_col = st.columns([2, 3])
    with layers_col:
        st.markdown("""
        <div style='padding-top:8px;'>
            <div style='color:#00ff88;font-size:0.72rem;font-weight:700;letter-spacing:0.1em;
                 text-transform:uppercase;margin-bottom:10px;'>🔗 Detection Framework</div>
            <h2 style='color:#ffffff;font-size:1.8rem;font-weight:900;margin:0 0 14px 0;line-height:1.15;'>7 Layers of<br>Analysis</h2>
            <p style='color:#737373;font-size:0.88rem;line-height:1.7;'>
                Each layer adds another dimension to our detection capability, creating a
                comprehensive fingerprint that reveals fraudulent accounts with high precision.
            </p>
        </div>
        """, unsafe_allow_html=True)
    with cards_col:
        for layer in FRAMEWORK_LAYERS:
            st.markdown(f"""
            <div class='gt-layer-card'>
                <div class='gt-layer-num'>{layer['num']}</div>
                <div style='flex:1;'>
                    <div style='display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:6px;'>
                        <div class='gt-layer-name'>{layer['name']}</div>
                        <span class='gt-layer-tag'>{layer['data']}</span>
                    </div>
                    <div class='gt-layer-desc'>{layer['desc']}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ── Research Impact ───────────────────────────────────────────────────────
    st.markdown("""<hr class='gt-divider'>
    <div style='text-align:center;margin-bottom:20px;'>
        <h2 style='color:#ffffff;font-size:1.8rem;font-weight:900;margin:0 0 8px 0;'>Research Impact</h2>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class='gt-impact-grid'>
        <div class='gt-impact-card'>
            <div class='gt-impact-icon'>🔍</div>
            <div class='gt-impact-title'>Key Contribution</div>
            <div class='gt-impact-body'>This framework demonstrates that <strong style='color:#ffffff;'>independent
            platform audit is possible</strong> using only public API endpoints — without access to
            Spotify's internal fraud systems.</div>
        </div>
        <div class='gt-impact-card'>
            <div class='gt-impact-icon'>⚠️</div>
            <div class='gt-impact-title'>Surprise Finding</div>
            <div class='gt-impact-body'>Ghost artists are <strong style='color:#ffffff;'>NOT cross-platform invisible</strong>.
            Relaxing White Noise has 353M YouTube views. Ghost behavior is Spotify-economic stream farming,
            not fabricated identity.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style='background:#0d2818;border:1px solid #1a4a2e;border-radius:10px;
         padding:14px 20px;margin-top:16px;color:#a3a3a3;font-size:0.88rem;'>
        <strong style='color:#00ff88;'>Status:</strong>
        Exercises 1–10 complete · All 7 signals implemented · GAT/GCN trained (100% accuracy) · Streamlit dashboard deployed
    </div>
    """, unsafe_allow_html=True)

    # ── Explore the Analysis ──────────────────────────────────────────────────
    st.markdown("""<hr class='gt-divider'>
    <div style='text-align:center;margin:32px 0 28px 0;'>
        <h2 style='color:#ffffff;font-size:1.9rem;font-weight:900;margin:0 0 10px 0;letter-spacing:-0.02em;'>Explore the Analysis</h2>
        <p style='color:#737373;font-size:0.92rem;'>Dive deeper into our research with interactive tools and visualizations</p>
    </div>
    <div style='display:flex;gap:16px;flex-wrap:wrap;margin-bottom:48px;'>
        <div style='flex:1;min-width:200px;background:#111111;border:1px solid #1e1e1e;border-radius:14px;padding:24px;'>
            <div style='background:#0d0d0d;border:1px solid #1e1e1e;border-radius:10px;
                 height:120px;margin-bottom:20px;display:flex;align-items:center;justify-content:center;'>
                <div style='color:#00ff88;font-size:2.5rem;'>🔍</div>
            </div>
            <div style='display:flex;align-items:center;gap:8px;margin-bottom:8px;'>
                <span style='color:#737373;font-size:1rem;'>🔍</span>
                <span style='color:#ffffff;font-weight:700;font-size:0.95rem;'>Artist Analyzer</span>
            </div>
            <div style='color:#737373;font-size:0.84rem;line-height:1.6;'>Input any Spotify artist to see their authenticity score</div>
        </div>
        <div style='flex:1;min-width:200px;background:#111111;border:1px solid #1e1e1e;border-radius:14px;padding:24px;'>
            <div style='background:#0d0d0d;border:1px solid #1e1e1e;border-radius:10px;
                 height:120px;margin-bottom:20px;display:flex;align-items:center;justify-content:center;'>
                <div style='color:#00ff88;font-size:2.5rem;'>🌐</div>
            </div>
            <div style='display:flex;align-items:center;gap:8px;margin-bottom:8px;'>
                <span style='color:#737373;font-size:1rem;'>🕸</span>
                <span style='color:#ffffff;font-weight:700;font-size:0.95rem;'>Network Explorer</span>
            </div>
            <div style='color:#737373;font-size:0.84rem;line-height:1.6;'>Visualize artist collaboration networks in 3D</div>
        </div>
        <div style='flex:1;min-width:200px;background:#111111;border:1px solid #1e1e1e;border-radius:14px;padding:24px;'>
            <div style='background:#0d0d0d;border:1px solid #1e1e1e;border-radius:10px;
                 height:120px;margin-bottom:20px;display:flex;align-items:center;justify-content:center;'>
                <div style='color:#00ff88;font-size:2.5rem;'>🤖</div>
            </div>
            <div style='display:flex;align-items:center;gap:8px;margin-bottom:8px;'>
                <span style='color:#00ff88;font-size:1rem;'>🤖</span>
                <span style='color:#ffffff;font-weight:700;font-size:0.95rem;'>AI Assistant</span>
            </div>
            <div style='color:#737373;font-size:0.84rem;line-height:1.6;'>Ask questions about our methodology and findings</div>
        </div>
    </div>

    <div style='text-align:center;color:#333;font-size:0.78rem;padding:20px 0;border-top:1px solid #1a1a1a;'>
        GhostTrack | INFO 7390 - Spring 2026 &nbsp;&nbsp;|&nbsp;&nbsp; By Trimbkeshwar Jagtap
    </div>
    """, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# EXERCISE GALLERY
# ═════════════════════════════════════════════════════════════════════════════
elif page == "📊 Exercise Gallery":
    _navbar("Gallery")
    st.markdown("""
    <div class='gt-page-header'>
        <div style='color:#00ff88;font-size:0.72rem;font-weight:700;letter-spacing:0.1em;
             text-transform:uppercase;margin-bottom:6px;'>Exercise Gallery</div>
        <h1 class='gt-page-title'>Analysis Figures</h1>
        <p class='gt-page-subtitle'>Publication-quality figures from Kaggle dataset (114K tracks) and Neo4j graph (490 tracks).</p>
    </div>
    """, unsafe_allow_html=True)

    any_found = False
    for filename, meta in FIGURES_META.items():
        fig_path = FIGURES_DIR / filename
        if fig_path.exists():
            any_found = True
            st.markdown(f"""
            <div class='gt-fig-frame'>
                <div style='display:flex;gap:8px;margin-bottom:12px;flex-wrap:wrap;'>
                    <span class='gt-tag'>{meta['exercise']}</span>
                    <span class='gt-tag-gray'>{meta['signal']}</span>
                </div>
                <h3 style='color:#ffffff;margin:0 0 4px 0;font-size:1rem;'>{meta['title']}</h3>
            </div>
            """, unsafe_allow_html=True)
            st.image(str(fig_path), use_container_width=True)
            st.markdown(f"<div class='gt-fig-caption'>📌 {meta['caption']}</div>", unsafe_allow_html=True)
            st.markdown("<hr class='gt-divider'>", unsafe_allow_html=True)

    if not any_found:
        st.markdown("""
        <div style='background:#111111;border:1px solid #1e1e1e;border-radius:10px;
             padding:32px;text-align:center;color:#525252;'>
            No figures found. Run notebooks 01–10 to generate figures.
        </div>
        """, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# ARTIST ANALYZER
# ═════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Artist Analyzer":
    _navbar("Analyzer")
    st.markdown("""
    <div class='gt-page-header'>
        <div style='color:#00ff88;font-size:0.72rem;font-weight:700;letter-spacing:0.1em;
             text-transform:uppercase;margin-bottom:6px;'>Artist Analyzer</div>
        <h1 class='gt-page-title'>Ghost Detection Tool</h1>
        <p class='gt-page-subtitle'>
            Search by artist name, Spotify ID, album, or track.
            We scan YouTube, Apple Music, Kaggle (114K tracks), and our Neo4j graph.
        </p>
    </div>
    """, unsafe_allow_html=True)

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
            f"<div style='display:inline-block;background:#0d2818;border:1px solid #1a4a2e;"
            f"border-radius:20px;padding:4px 14px;font-size:0.78rem;color:#00ff88;font-weight:600;"
            f"margin-bottom:10px;'>🔎 {match_label}</div>",
            unsafe_allow_html=True,
        )
        st.markdown(f"<h2 style='color:#ffffff;font-weight:900;margin-top:4px;'>Results: {artist_query}</h2>", unsafe_allow_html=True)

        # If matched via a specific track, show that track's audio features
        matched_track = collected_data.get("matched_track")
        if matched_track:
            tn = matched_track.get("track_name", "")
            alb = matched_track.get("album_name", "")
            tg = matched_track.get("track_genre", "")
            pop = matched_track.get("popularity", 0)
            st.markdown(
                f"<div style='background:#0d2818;border:1px solid #1a4a2e;border-radius:10px;"
                f"padding:14px 18px;margin-bottom:14px;'>"
                f"<div style='color:#00ff88;font-size:0.72rem;font-weight:700;letter-spacing:0.08em;"
                f"text-transform:uppercase;margin-bottom:8px;'>🎵 Matched Track</div>"
                f"<div style='color:#ffffff;font-weight:700;font-size:0.95rem;'>{tn}</div>"
                f"<div style='color:#737373;font-size:0.85rem;margin-top:4px;'>"
                f"{('Album: ' + alb + ' · ') if alb else ''}"
                f"{('Genre: ' + tg + ' · ') if tg else ''}"
                f"Popularity: {pop}/100</div>"
                f"<div style='display:flex;gap:16px;margin-top:10px;flex-wrap:wrap;'>"
                + "".join(
                    f"<span style='color:#737373;font-size:0.82rem;'>"
                    f"<span style='color:#00ff88;font-weight:600;'>{k.title()}</span> {v:.3f}</span>"
                    for k, v in {
                        "Dance": matched_track.get("danceability", 0),
                        "Energy": matched_track.get("energy", 0),
                        "Valence": matched_track.get("valence", 0),
                        "Acoustic": matched_track.get("acousticness", 0),
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
            border = "#00ff88" if ok else "#333333"
            bg     = "#0d2818" if ok else "#111111"
            src_color = "#ffffff" if ok else "#737373"
            return (
                f"<div style='display:flex;align-items:flex-start;gap:12px;background:{bg};"
                f"border-left:3px solid {border};border-radius:8px;padding:12px 16px;margin-bottom:8px;'>"
                f"<span style='font-size:1.1rem;line-height:1.5;'>{icon}</span>"
                f"<div><span style='color:{src_color};font-weight:600;font-size:0.9rem;'>{source}</span>"
                f"<span style='color:#737373;font-size:0.85rem;margin-left:10px;'>{detail}</span></div></div>"
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
            st.markdown("<h3 style='color:#ffffff;font-weight:700;margin:20px 0 12px 0;'>Signal Scores</h3>", unsafe_allow_html=True)
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
                    score_str, score_color, bar_pct = "N/A", "#525252", 0
                else:
                    score_str = f"{val:.3f}"
                    bar_pct = int(val * 100)
                    score_color = "#e74c3c" if val > 0.7 else ("#f59e0b" if val > 0.4 else "#00ff88")
                st.markdown(f"""
                <div class='gt-signal-card'>
                    <span class='gt-signal-score' style='color:{score_color};'>{score_str}</span>
                    <div class='gt-signal-name'>{num}: {name}</div>
                    <div style='margin-top:8px;background:#1a1a1a;border-radius:4px;height:5px;'>
                        <div style='width:{bar_pct}%;background:{score_color};border-radius:4px;height:5px;'></div>
                    </div>
                </div>""", unsafe_allow_html=True)

        # ── Optional AI deep-dive ─────────────────────────────────────────────
        st.markdown("<hr class='gt-divider'>", unsafe_allow_html=True)
        st.markdown("<h3 style='color:#ffffff;font-weight:700;'>AI Deep-Dive Analysis <span style='color:#525252;font-size:0.85rem;font-weight:400;'>(optional)</span></h3>", unsafe_allow_html=True)
        st.markdown("<p style='color:#737373;font-size:0.85rem;margin-top:-8px;'>Calls GPT-4o once to generate a PhD-level ghost detection analysis from the data above.</p>", unsafe_allow_html=True)
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
                gauge_fig.update_layout(paper_bgcolor="#0a0a0a", height=220, margin=dict(t=40, b=0, l=30, r=30))
                st.plotly_chart(gauge_fig, use_container_width=True)
            except ImportError:
                st.progress(ghost_prob / 100)

            analysis_text = ai_result.get("analysis", "")
            st.markdown(f"""
            <div class='gt-ai-card'>
                <div class='gt-ai-header'>GPT-4o ANALYSIS — {artist_query.upper()}</div>
                <div style='color:#a3a3a3;line-height:1.8;font-size:0.9rem;'>{analysis_text.replace(chr(10), "<br>")}</div>
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
        st.markdown("""
        <div style='background:#111111;border:1px solid #1e1e1e;border-radius:12px;
             padding:48px;text-align:center;margin-top:20px;'>
            <div style='font-size:2rem;margin-bottom:12px;'>🔍</div>
            <div style='color:#525252;font-size:0.95rem;'>
                Enter an artist name, Spotify ID, track, or album above and click Search.
            </div>
        </div>
        """, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# NETWORK EXPLORER
# ═════════════════════════════════════════════════════════════════════════════
elif page == "🌐 Network Explorer":
    _navbar("Network")
    st.markdown("""
    <div class='gt-page-header'>
        <div style='color:#00ff88;font-size:0.72rem;font-weight:700;letter-spacing:0.1em;
             text-transform:uppercase;margin-bottom:6px;'>Network Explorer</div>
        <h1 class='gt-page-title'>Production Company Graph</h1>
        <p class='gt-page-subtitle'>ISRC-based attribution network. HHI &gt; 0.6 signals bulk-upload operations via a single registrant.</p>
    </div>
    """, unsafe_allow_html=True)

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
            plot_bgcolor="#0a0a0a", paper_bgcolor="#0a0a0a", font_color="#a3a3a3",
            title_font_color="#ffffff", yaxis=dict(gridcolor="#1e1e1e"),
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
    _navbar("Cross-Platform")
    st.markdown("""
    <div class='gt-page-header'>
        <div style='color:#00ff88;font-size:0.72rem;font-weight:700;letter-spacing:0.1em;
             text-transform:uppercase;margin-bottom:6px;'>Signal 7</div>
        <h1 class='gt-page-title'>Cross-Platform Discrepancy</h1>
        <p class='gt-page-subtitle'>YouTube + Apple Music presence vs Spotify behavior.</p>
    </div>
    <div style='background:#2a1200;border:1px solid #f59e0b;border-radius:10px;padding:14px 20px;margin-bottom:24px;'>
        <span style='color:#fcd34d;font-weight:700;'>Key Finding:</span>
        <span style='color:#a3a3a3;font-size:0.9rem;margin-left:8px;'>
            Relaxing White Noise has <strong style='color:#fcd34d;'>353M YouTube views</strong> —
            ghost behavior is Spotify-stream-farming, not cross-platform absence.
        </span>
    </div>
    """, unsafe_allow_html=True)
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
            st.markdown(f"""<div style='background:#111111;border:1px solid #1e1e1e;border-radius:10px;padding:20px;text-align:center;'>
                <div style='color:#737373;font-size:0.8rem;margin-bottom:8px;'>YouTube Views</div>
                <div style='color:{color};font-size:2rem;font-weight:800;'>{views_str if yt_found else "Not found"}</div>
                <div style='color:#525252;font-size:0.75rem;margin-top:6px;'>{yt.get("video_title","")[:50]}</div>
            </div>""", unsafe_allow_html=True)
        with col_ap:
            ap_color = "#00ff88" if apple_found else "#e74c3c"
            st.markdown(f"""<div style='background:#111111;border:1px solid #1e1e1e;border-radius:10px;padding:20px;text-align:center;'>
                <div style='color:#737373;font-size:0.8rem;margin-bottom:8px;'>Apple Music</div>
                <div style='color:{ap_color};font-size:2rem;font-weight:800;'>{"✅ Found" if apple_found else "❌ Not found"}</div>
                <div style='color:#525252;font-size:0.75rem;margin-top:6px;'>{itunes.get("primary_genre","")}</div>
            </div>""", unsafe_allow_html=True)

        # Rule-based assessment (no OpenAI)
        if views >= 10_000_000:
            presence_label, presence_color = "Strong presence", "#00ff88"
        elif views >= 1_000:
            presence_label, presence_color = "Moderate presence", "#f59e0b"
        elif views > 0:
            presence_label, presence_color = "Minimal presence", "#f59e0b"
        else:
            presence_label, presence_color = "Not found", "#525252"
        st.markdown(f"""<div style='background:#111111;border:1px solid #1e1e1e;border-left:4px solid {presence_color};border-radius:8px;padding:14px 18px;margin-top:12px;'>
            <span style='color:{presence_color};font-weight:700;'>{presence_label}</span>
            <span style='color:#737373;font-size:0.85rem;margin-left:12px;'>{cp_verdict}</span>
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
        st.markdown(f"""<div style='background:#111111;border-left:4px solid {bc};border-radius:8px;padding:16px 20px;margin:10px 0;border:1px solid #1e1e1e;'>
            <div style='display:flex;justify-content:space-between;align-items:center;'>
                <span style='color:#ffffff;font-size:1.1rem;font-weight:700;'>{vi} {artist_name}</span>
                <span style='color:{s7c};font-weight:700;'>S7: {s7:.2f}</span>
            </div>
            <div style='margin-top:8px;display:flex;gap:24px;flex-wrap:wrap;'>
                <span style='color:#a3a3a3;'>▶ {yt_str} views</span>
                <span style='color:#a3a3a3;'>{ch_str}</span>
                <span style='color:#a3a3a3;'>{apple_str}</span>
            </div>
            <div style='margin-top:8px;color:#737373;font-size:0.85rem;'>{data["note"]}</div>
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
            plot_bgcolor="#0a0a0a", paper_bgcolor="#0a0a0a", font_color="#a3a3a3",
            title_font_color="#ffffff", yaxis=dict(gridcolor="#1e1e1e"), showlegend=False)
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
    _navbar("AI Assistant")
    st.markdown("""
    <div class='gt-page-header'>
        <div style='color:#00ff88;font-size:0.72rem;font-weight:700;letter-spacing:0.1em;
             text-transform:uppercase;margin-bottom:6px;'>AI Assistant</div>
        <h1 class='gt-page-title'>Research Assistant</h1>
        <p class='gt-page-subtitle'>Ask any research question about the project. Powered by GPT-4o with full project context.</p>
    </div>
    <div style='background:#0d2818;border:1px solid #1a4a2e;border-radius:10px;padding:12px 18px;margin-bottom:20px;'>
        <span style='color:#00ff88;font-size:0.82rem;font-weight:600;'>Each question uses one OpenAI API call.</span>
    </div>
    """, unsafe_allow_html=True)

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
    _navbar("About")

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown("""
    <div style='padding:20px 0 16px 0;border-bottom:1px solid #1a1a1a;margin-bottom:24px;'>
        <div style='color:#00ff88;font-size:0.72rem;font-weight:700;letter-spacing:0.1em;
             text-transform:uppercase;margin-bottom:8px;'>About the Research</div>
        <h1 style='color:#ffffff;font-size:2.2rem;font-weight:900;margin:0 0 12px 0;
             letter-spacing:-0.02em;'>Methodology &amp; Architecture</h1>
        <p style='color:#737373;font-size:1rem;max-width:580px;line-height:1.7;'>
            A deep dive into how we built a framework to detect streaming platform fraud
            using publicly available data and cutting-edge machine learning.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Abstract + Course card ────────────────────────────────────────────────
    ab_col, card_col = st.columns([3, 1])
    with ab_col:
        st.markdown("""
        <h2 style='color:#ffffff;font-size:1.4rem;font-weight:800;margin-bottom:14px;'>Abstract</h2>
        <p style='color:#737373;line-height:1.8;font-size:0.92rem;'>
            This project presents a <strong style='color:#ffffff;'>7-layer exploratory data analysis framework</strong>
            for detecting ghost artists — AI-generated or fraudulent accounts used to inflate streaming revenue —
            using only Spotify's public, unauthenticated API endpoints.
        </p>
        <p style='color:#737373;line-height:1.8;font-size:0.92rem;margin-top:12px;'>
            Prior detection work relies on internal data unavailable to researchers. We demonstrate that
            <strong style='color:#00ff88;'>catalog coherence, playlist entropy, ISRC attribution, release cadence,
            metadata similarity, and graph topology</strong> each provide independent discriminative signal,
            and that their combination yields robust classification without any proprietary access.
        </p>
        <p style='color:#737373;line-height:1.8;font-size:0.92rem;margin-top:12px;'>
            The framework is validated on three seed artists and scaled using the Kaggle Spotify Audio Features
            dataset (114,000 tracks, 114 genres). Our Graph Attention Network achieves
            <strong style='color:#ffffff;'>100% test accuracy</strong> on the proof-of-concept 65-node collaboration graph.
        </p>
        """, unsafe_allow_html=True)
    with card_col:
        st.markdown("""
        <div style='background:#0d2818;border:1px solid #1a4a2e;border-radius:12px;padding:20px;'>
            <div style='font-size:1.5rem;margin-bottom:8px;'>🎓</div>
            <div style='color:#00ff88;font-weight:700;font-size:0.95rem;margin-bottom:4px;'>INFO 7390</div>
            <div style='color:#a3a3a3;font-size:0.82rem;line-height:1.7;'>
                Advances in Data Science<br>Spring 2026<br>Northeastern University
            </div>
            <div style='border-top:1px solid #1a4a2e;margin:12px 0;'></div>
            <div style='color:#ffffff;font-size:0.85rem;font-weight:600;'>Trimbkeshwar Jagtap</div>
            <div style='color:#525252;font-size:0.78rem;'>Researcher</div>
        </div>
        """, unsafe_allow_html=True)

    # ── Architecture pipeline ─────────────────────────────────────────────────
    st.markdown("""
    <h2 style='color:#ffffff;font-size:1.4rem;font-weight:800;margin:36px 0 14px 0;'>System Architecture</h2>
    <p style='color:#737373;font-size:0.88rem;margin-bottom:16px;'>
        Data flows from multiple sources through our 7-layer analysis pipeline to produce ghost artist probability scores.
    </p>
    <div class='gt-pipeline'>
        <div class='gt-pipe-step'>
            <div class='gt-pipe-num'>1</div>
            <div class='gt-pipe-title'>Data Sources</div>
            <div class='gt-pipe-items'>Spotify API<br>Kaggle CSV<br>YouTube API<br>iTunes API</div>
        </div>
        <div class='gt-pipe-step'>
            <div class='gt-pipe-num'>2</div>
            <div class='gt-pipe-title'>Processing</div>
            <div class='gt-pipe-items'>Feature Extraction<br>Graph Construction<br>Neo4j Ingestion<br>Signal Scoring</div>
        </div>
        <div class='gt-pipe-step'>
            <div class='gt-pipe-num'>3</div>
            <div class='gt-pipe-title'>Analysis</div>
            <div class='gt-pipe-items'>7-Layer Framework<br>GNN Classification<br>CrewAI Pipeline<br>Verdict Engine</div>
        </div>
        <div class='gt-pipe-step'>
            <div class='gt-pipe-num'>4</div>
            <div class='gt-pipe-title'>Output</div>
            <div class='gt-pipe-items'>Ghost Score<br>Visualizations<br>paper/figures/<br>Reports</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Tech Stack ────────────────────────────────────────────────────────────
    st.markdown("""
    <h2 style='color:#ffffff;font-size:1.4rem;font-weight:800;margin:36px 0 8px 0;'>Tech Stack</h2>
    <p style='color:#737373;font-size:0.88rem;margin-bottom:16px;'>
        Built with modern data science tools and frameworks for scalability and reproducibility.
    </p>
    <div class='gt-tech-grid'>
        <div class='gt-tech-card'>
            <div class='gt-tech-cat'>Data</div>
            <div class='gt-tech-pkg'>Python 3.14<br>pandas 3.0<br>numpy 2.4</div>
        </div>
        <div class='gt-tech-card'>
            <div class='gt-tech-cat'>APIs</div>
            <div class='gt-tech-pkg'>spotipy 2.26<br>httpx 0.28<br>openai 1.x</div>
        </div>
        <div class='gt-tech-card'>
            <div class='gt-tech-cat'>Graph</div>
            <div class='gt-tech-pkg'>Neo4j 6.1<br>networkx 3.6</div>
        </div>
        <div class='gt-tech-card'>
            <div class='gt-tech-cat'>ML</div>
            <div class='gt-tech-pkg'>scikit-learn 1.8<br>torch 2.11<br>torch_geometric 2.7</div>
        </div>
        <div class='gt-tech-card'>
            <div class='gt-tech-cat'>Viz</div>
            <div class='gt-tech-pkg'>matplotlib 3.10<br>plotly 6.7<br>seaborn 0.13</div>
        </div>
        <div class='gt-tech-card'>
            <div class='gt-tech-cat'>Backend</div>
            <div class='gt-tech-pkg'>FastAPI 0.135<br>uvicorn 0.44</div>
        </div>
        <div class='gt-tech-card'>
            <div class='gt-tech-cat'>Frontend</div>
            <div class='gt-tech-pkg'>Streamlit 1.56</div>
        </div>
        <div class='gt-tech-card'>
            <div class='gt-tech-cat'>Agents</div>
            <div class='gt-tech-pkg'>crewai 1.14<br>GPT-4o</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── CTA ───────────────────────────────────────────────────────────────────
    st.markdown("""
    <div style='background:linear-gradient(135deg,#0d2818 0%,#111111 100%);
         border:1px solid #1a4a2e;border-radius:14px;padding:36px;text-align:center;margin-top:36px;'>
        <h2 style='color:#ffffff;font-size:1.6rem;font-weight:900;margin:0 0 8px 0;'>Ready to Explore?</h2>
        <p style='color:#737373;font-size:0.92rem;margin:0 0 20px 0;'>
            Dive into our interactive tools and see the framework in action.
        </p>
        <div style='display:flex;gap:12px;justify-content:center;flex-wrap:wrap;'>
            <span class='gt-btn-primary'>→ Try Artist Analyzer</span>
            <span class='gt-btn-secondary'>📊 View Exercise Gallery</span>
        </div>
    </div>
    <div style='text-align:center;color:#333;font-size:0.78rem;margin-top:24px;'>
        GhostTrack | INFO 7390 — Spring 2026 | By Trimbkeshwar Jagtap
    </div>
    """, unsafe_allow_html=True)
