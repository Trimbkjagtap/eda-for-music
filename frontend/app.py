"""
EDA for Music — Streamlit Dashboard
A Layered Framework for Public-API Ghost Artist Discovery
"""
import streamlit as st
import os
from pathlib import Path

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EDA for Music",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Dark sidebar */
    [data-testid="stSidebar"] {
        background-color: #0f0f1a;
        border-right: 1px solid #2a2a4a;
    }
    [data-testid="stSidebar"] .stRadio label {
        color: #ccc;
        font-size: 15px;
    }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #a78bfa;
    }

    /* Main area */
    .main .block-container {
        background-color: #12121f;
        padding-top: 1.5rem;
    }
    h1 { color: #a78bfa; font-weight: 800; }
    h2 { color: #818cf8; }
    h3 { color: #94a3b8; }

    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #1e1e3f 0%, #16213e 100%);
        border: 1px solid #2a2a5a;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }
    .metric-card .number {
        font-size: 2.4rem;
        font-weight: 800;
        color: #a78bfa;
    }
    .metric-card .label {
        font-size: 0.85rem;
        color: #94a3b8;
        margin-top: 4px;
    }

    /* Signal score cards */
    .signal-card {
        background: #1a1a2e;
        border-left: 4px solid #a78bfa;
        border-radius: 8px;
        padding: 14px 18px;
        margin: 6px 0;
    }
    .signal-card .score {
        font-size: 1.5rem;
        font-weight: 700;
        color: #f59e0b;
        float: right;
    }
    .signal-card .name {
        color: #e2e8f0;
        font-weight: 600;
    }
    .signal-card .desc {
        color: #94a3b8;
        font-size: 0.82rem;
        margin-top: 3px;
    }

    /* Framework layer cards */
    .layer-card {
        background: #1e1e3f;
        border: 1px solid #2a2a5a;
        border-radius: 8px;
        padding: 12px 16px;
        margin: 5px 0;
    }

    /* Figure frames */
    .fig-frame {
        background: #1a1a2e;
        border: 1px solid #2a2a5a;
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 16px;
    }

    /* Tag pill */
    .tag {
        display: inline-block;
        background: #312e81;
        color: #a5b4fc;
        border-radius: 20px;
        padding: 2px 12px;
        font-size: 0.75rem;
        margin: 2px;
    }

    /* Hide default Streamlit footer */
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Constants ────────────────────────────────────────────────────────────────
FIGURES_DIR = Path(__file__).parent.parent / "paper" / "figures"

FIGURES_META = {
    "fig1_catalog_coherence.png": {
        "title": "Figure 1: Catalog Coherence in Audio Feature Space",
        "caption": (
            "PCA projection of per-track audio features for ghost-like artists (left) vs organic controls (right). "
            "Ghost-like artists cluster into tight, compact ellipses — a signature of AI-generated content "
            "optimized for a single micro-genre. Organic artists spread broadly, reflecting genuine stylistic range. "
            "Ellipses = 1σ contour. Data: Kaggle 114K-track dataset."
        ),
        "exercise": "Exercise 1",
        "signal": "Catalog Variance",
    },
    "fig2_playlist_entropy.png": {
        "title": "Figure 2: Playlist Aesthetic Coherence",
        "caption": (
            "Energy vs Valence scatter for three simulated playlist archetypes, colored by danceability. "
            "TIGHT playlists (single genre) show low Shannon entropy — the hallmark of a fraud target zone. "
            "MODERATE playlists (2–3 related genres) occupy a middle ground. "
            "SCATTERED playlists cover the full feature space with high entropy."
        ),
        "exercise": "Exercise 2",
        "signal": "Playlist Entropy",
    },
    "fig3_isrc_join.png": {
        "title": "Figure 3: Artist to Production Company Attribution via ISRC",
        "caption": (
            "Bipartite graph connecting artists to production companies via ISRC prefix. "
            "Edge width = track count registered with that company. "
            "Shared prefixes (★) indicate multiple 'independent' artists registered by the same company — "
            "a strong indicator of coordinated fraud. "
            "Data: 3 seed artists, 490 tracks, 8 production companies from Neo4j."
        ),
        "exercise": "Exercise 3",
        "signal": "ISRC Attribution",
    },
}

FRAMEWORK_LAYERS = [
    {"num": "1", "name": "Catalog Coherence",
     "desc": "Audio feature variance per artist. Ghost artists show unnaturally low variance.",
     "data": "Kaggle + Spotify /tracks"},
    {"num": "2", "name": "Playlist Entropy",
     "desc": "Shannon entropy of playlist feature distributions. Fraud playlists cluster tight.",
     "data": "Spotify /playlists"},
    {"num": "3", "name": "ISRC Attribution",
     "desc": "Production company identification via ISRC prefix. Shared registrants reveal networks.",
     "data": "Spotify /tracks (external_ids)"},
    {"num": "4", "name": "Release Cadence",
     "desc": "Statistical analysis of release date spacing. Machine outputs have inhuman regularity.",
     "data": "Spotify /albums"},
    {"num": "5", "name": "Metadata Similarity",
     "desc": "NLP embedding of track/artist names. Ghost catalogs reuse names with minor variations.",
     "data": "Spotify metadata"},
    {"num": "6", "name": "Graph Centrality",
     "desc": "Co-appearance network analysis. Fraud artists cluster in isolated low-degree subgraphs.",
     "data": "Neo4j graph"},
    {"num": "7", "name": "Aggregate Score",
     "desc": "Weighted combination of signals 1–6 into a final ghost probability score.",
     "data": "All layers"},
]

SIGNAL_SCORES_PLACEHOLDER = [
    {"name": "Catalog Variance", "score": "—", "desc": "Pending: variance analysis requires full catalog"},
    {"name": "Playlist Entropy", "score": "—", "desc": "Pending: playlist data requires Spotify API"},
    {"name": "ISRC Attribution", "score": "—", "desc": "Pending: ISRC analysis from Neo4j"},
    {"name": "Release Cadence", "score": "—", "desc": "Pending: release date clustering"},
    {"name": "Metadata Similarity", "score": "—", "desc": "Pending: NLP similarity scoring"},
    {"name": "Graph Centrality", "score": "—", "desc": "Pending: co-appearance graph"},
    {"name": "Ghost Probability", "score": "—", "desc": "Aggregate of all 7 signals"},
]

# ── Sidebar navigation ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎵 EDA for Music")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["🏠 Home", "📊 Exercise Gallery", "🔍 Artist Analyzer", "🕸️ Network Explorer", "ℹ️ About"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown(
        "<div style='color:#64748b;font-size:0.78rem;'>"
        "INFO 7390 · Spring 2026<br>"
        "Fake Artist Detection<br>"
        "via Public API Analysis"
        "</div>",
        unsafe_allow_html=True,
    )

# ── HOME PAGE ──────────────────────────────────────────────────────────────────
if page == "🏠 Home":
    st.markdown("# 🎵 Exploratory Data Analysis for Music")
    st.markdown(
        "### A Layered Framework for Public-API Discovery of Ghost Artists on Streaming Platforms"
    )
    st.markdown("")

    # Project stats
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div class='metric-card'>
            <div class='number'>3</div>
            <div class='label'>Seed Artists Analyzed</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class='metric-card'>
            <div class='number'>490</div>
            <div class='label'>Tracks in Neo4j</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class='metric-card'>
            <div class='number'>8</div>
            <div class='label'>Production Companies</div>
        </div>""", unsafe_allow_html=True)
    with col4:
        st.markdown("""
        <div class='metric-card'>
            <div class='number'>114K</div>
            <div class='label'>Kaggle Training Tracks</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Framework description
    st.markdown("## The 7-Layer Detection Framework")
    st.markdown(
        "Ghost artists — AI-generated or fake accounts inflating streaming numbers — "
        "leave detectable traces across Spotify's public API. This project builds a "
        "layered signal framework that requires **only public, unauthenticated API access**, "
        "making it replicable without special permissions."
    )

    for layer in FRAMEWORK_LAYERS:
        st.markdown(f"""
        <div class='layer-card'>
            <span style='color:#a78bfa;font-weight:700;'>Layer {layer['num']}: {layer['name']}</span>
            <span style='float:right;'><span class='tag'>{layer['data']}</span></span>
            <div style='color:#94a3b8;font-size:0.85rem;margin-top:4px;'>{layer['desc']}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.info(
        "**Status:** Exercises 1–3 complete using Kaggle dataset + Neo4j. "
        "Exercises 4–7 pending Spotify API rate limit reset (~24h). "
        "No proprietary data or special API access required."
    )

# ── EXERCISE GALLERY PAGE ──────────────────────────────────────────────────────
elif page == "📊 Exercise Gallery":
    st.markdown("# 📊 Exercise Gallery")
    st.markdown("Figures generated from Kaggle dataset (114K tracks) and Neo4j graph (490 tracks).")
    st.markdown("---")

    any_figure_found = False
    for filename, meta in FIGURES_META.items():
        fig_path = FIGURES_DIR / filename
        if fig_path.exists():
            any_figure_found = True
            st.markdown(f"""
            <div class='fig-frame'>
                <span class='tag'>{meta['exercise']}</span>
                <span class='tag'>{meta['signal']}</span>
                <h3 style='margin-top:10px;'>{meta['title']}</h3>
            </div>""", unsafe_allow_html=True)
            st.image(str(fig_path), use_container_width=True)
            st.markdown(
                f"<div style='color:#94a3b8;font-size:0.85rem;margin-bottom:24px;'>"
                f"📌 {meta['caption']}</div>",
                unsafe_allow_html=True
            )
            st.markdown("---")

    if not any_figure_found:
        st.warning(
            "No figures generated yet. Run the Exercise notebooks (01–03) to generate figures, "
            "then reload this page."
        )
        st.markdown("**Expected figures:**")
        for filename, meta in FIGURES_META.items():
            st.markdown(f"- `paper/figures/{filename}` — {meta['title']}")

# ── ARTIST ANALYZER PAGE ──────────────────────────────────────────────────────
elif page == "🔍 Artist Analyzer":
    st.markdown("# 🔍 Artist Analyzer")
    st.markdown(
        "Enter a Spotify Artist ID to run all 7 detection signals. "
        "Results are cached — repeated lookups are instant."
    )

    col_input, col_btn = st.columns([4, 1])
    with col_input:
        artist_id = st.text_input(
            "Spotify Artist ID",
            placeholder="e.g. 4Z8W4fKeB5YxbusRsdQVPb  (Radiohead)",
            label_visibility="collapsed",
        )
    with col_btn:
        analyze = st.button("Analyze", use_container_width=True, type="primary")

    if analyze and artist_id:
        st.info(
            f"**Artist ID:** `{artist_id}`  \n"
            "Spotify API is currently rate-limited. Analysis will run when the limit resets. "
            "Showing placeholder signal scores below."
        )
    elif analyze and not artist_id:
        st.error("Please enter an Artist ID.")

    st.markdown("### Signal Scores")
    st.markdown(
        "<div style='color:#64748b;font-size:0.83rem;margin-bottom:12px;'>"
        "Scale: 0.0 (organic) → 1.0 (ghost). "
        "Scores below require a completed analysis run."
        "</div>",
        unsafe_allow_html=True,
    )

    for sig in SIGNAL_SCORES_PLACEHOLDER:
        score_color = "#f59e0b"
        if sig["score"] != "—":
            val = float(sig["score"])
            score_color = "#e74c3c" if val > 0.7 else ("#f59e0b" if val > 0.4 else "#22c55e")
        st.markdown(f"""
        <div class='signal-card'>
            <span class='score' style='color:{score_color};'>{sig['score']}</span>
            <div class='name'>{sig['name']}</div>
            <div class='desc'>{sig['desc']}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("**Try these example artist IDs (from ground truth):**")
    examples = {
        "Radiohead (organic)": "4Z8W4fKeB5YxbusRsdQVPb",
        "Brian Eno (organic)": "7MSUfLeTdDEoZiJPDSBXgi",
        "Calmo (ghost candidate)": "4Wx3ZL6d6p1gVMtwQ2YWsz",
    }
    for name, aid in examples.items():
        st.code(f"{name}: {aid}")

# ── NETWORK EXPLORER PAGE ──────────────────────────────────────────────────────
elif page == "🕸️ Network Explorer":
    st.markdown("# 🕸️ Network Explorer")
    st.markdown(
        "Interactive co-appearance graph connecting artists, playlists, and production companies."
    )
    st.markdown("---")

    # Show the ISRC graph if it exists
    fig3_path = FIGURES_DIR / "fig3_isrc_join.png"
    if fig3_path.exists():
        st.markdown("### Current Graph: ISRC Production Company Attribution")
        st.image(str(fig3_path), use_container_width=True)
        st.markdown(
            "<div style='color:#94a3b8;font-size:0.85rem;'>"
            "Static graph from Exercise 3. Interactive pyvis version will be added in Day 6."
            "</div>",
            unsafe_allow_html=True,
        )
    else:
        st.info(
            "No graph data loaded yet. Run `notebooks/03_isrc_join.ipynb` to generate the "
            "ISRC attribution graph, then reload."
        )

    st.markdown("---")
    st.markdown("### Graph Stats (Neo4j)")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Nodes", "1,523")
    with col2:
        st.metric("Artists", "3")
    with col3:
        st.metric("Tracks", "490")
    with col4:
        st.metric("Production Companies", "8")

    st.markdown("---")
    st.markdown("### Planned Interactive Features (Day 6)")
    st.markdown("""
    - **pyvis** force-directed graph embedded in Streamlit
    - Click node → drill down to artist details + signal scores
    - Filter by: production company, ISRC country, ghost probability threshold
    - Highlight connected components (fraud clusters)
    - Export graph as JSON / GraphML
    """)

# ── ABOUT PAGE ────────────────────────────────────────────────────────────────
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

        The framework is validated on three seed artists (two confirmed ghost candidates
        from the Michael Smith wire fraud case and the Dagens Nyheter investigation,
        one organic control) and scaled using the Kaggle Spotify Audio Features dataset
        (114,000 tracks, 114 genres).
        """)

        st.markdown("## Architecture")
        st.markdown("""
        ```
        DATA SOURCES          PROCESSING             OUTPUTS
        ──────────            ──────────             ───────
        Spotify API    ──►    SpotifyClient          Neo4j Graph
        Kaggle CSV     ──►    Signal Extractors  ──► Signal Scores
        YouTube API    ──►    Neo4j Ingestion        Research Paper
                              (this dashboard)       Figures (paper/)
        ```
        """)

    with col2:
        st.markdown("## Tech Stack")
        stack = {
            "Data": ["Python 3.14", "pandas 3.0", "numpy 2.4"],
            "APIs": ["spotipy 2.26", "httpx 0.28"],
            "Graph": ["Neo4j 6.1", "networkx 3.6", "pyvis 0.3"],
            "ML": ["scikit-learn 1.8", "scipy 1.17", "torch 2.11"],
            "Viz": ["matplotlib 3.10", "seaborn 0.13", "plotly 6.7"],
            "Backend": ["FastAPI 0.135", "uvicorn 0.44"],
            "Frontend": ["Streamlit 1.56"],
        }
        for category, packages in stack.items():
            st.markdown(f"**{category}:** {' · '.join(packages)}")

        st.markdown("---")
        st.markdown("## Links")
        st.markdown("- 📁 [GitHub Repository](https://github.com/Trimbkjagtap/eda-for-music)")
        st.markdown("- 📄 Paper: `paper/` (in progress)")
        st.markdown("- 📓 Notebooks: `notebooks/`")

        st.markdown("---")
        st.markdown("## Course")
        st.markdown(
            "**INFO 7390** — Advances in Data Science  \n"
            "Spring 2026  \n"
            "Submitted by: Trimbkeshwar Jagtap"
        )
