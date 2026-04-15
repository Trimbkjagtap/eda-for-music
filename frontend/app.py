"""
EDA for Music — Streamlit Dashboard
A Layered Framework for Public-API Ghost Artist Discovery
"""
import streamlit as st
import os
import sys
from pathlib import Path

# Ensure project root is on sys.path so src.* imports work
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

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

    /* Verdict badge */
    .verdict-ghost {
        background: #7f1d1d;
        color: #fca5a5;
        border-radius: 8px;
        padding: 10px 20px;
        font-size: 1.3rem;
        font-weight: 800;
        text-align: center;
        display: block;
    }
    .verdict-suspicious {
        background: #78350f;
        color: #fcd34d;
        border-radius: 8px;
        padding: 10px 20px;
        font-size: 1.3rem;
        font-weight: 800;
        text-align: center;
        display: block;
    }
    .verdict-organic {
        background: #14532d;
        color: #86efac;
        border-radius: 8px;
        padding: 10px 20px;
        font-size: 1.3rem;
        font-weight: 800;
        text-align: center;
        display: block;
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
    "fig4_bipartite_neighborhood.png": {
        "title": "Figure 4: Artist × Production Company Bipartite Neighborhood",
        "caption": (
            "Bipartite neighborhood graph showing ISRC-based production company concentration per artist. "
            "Node size = track count. Edge weight = tracks registered with that company. "
            "HHI scores: RWN=0.88, MRC=0.66, Calmo=0.54 — ghost artists show extreme ISRC concentration "
            "(single production company controls 80–95% of catalog), consistent with bulk-upload operations. "
            "Data: 490 tracks, 8 production companies from Neo4j AuraDB."
        ),
        "exercise": "Exercise 4",
        "signal": "Graph Centrality",
    },
    "fig5_recommendation_walk.png": {
        "title": "Figure 5: Recommendation Walk — Release Cadence as Walk Closure Signal",
        "caption": (
            "Temporal walk through each artist's catalog in release-date order. "
            "Walk closure = % of consecutive tracks released on the same day (bulk-upload fingerprint). "
            "Ghost artists: RWN=81%, MRC=95% closure — nearly all tracks bulk-uploaded in daily batches. "
            "Organic control (Nils Frahm): 0% closure, median gap 105 days between releases. "
            "The cadence gap is 81–95× larger for ghost catalogs, providing a near-perfect discriminator."
        ),
        "exercise": "Exercise 5",
        "signal": "Release Cadence",
    },
    "fig6_signal_radar.png": {
        "title": "Figure 6: Seven-Signal Ghost Artist Detection Radar",
        "caption": (
            "Radar chart showing all 7 signal scores per artist (left) and averaged ghost vs organic (right). "
            "S2 Release Cadence, S4 Catalog Density, and S6 Graph/HHI are the most discriminative signals. "
            "S7 Cross-Platform is inconclusive for the relaxation sub-genre: "
            "Relaxing White Noise has 353M YouTube views, Meditation Relax Club 157M — "
            "these artists are NOT invisible cross-platform. "
            "Ghost behavior is Spotify-stream-farming, not platform absence."
        ),
        "exercise": "Exercise 6",
        "signal": "Aggregate Score",
    },
    "fig6b_signal_heatmap.png": {
        "title": "Figure 6b: Signal Report Card Heatmap",
        "caption": (
            "Heatmap of all 7 signal scores across 4 artists. "
            "Red = ghost-like behavior (score ≥ 0.6), Green = organic behavior (≤ 0.4), "
            "Grey = N/A (data unavailable). "
            "S1 Audio Similarity returns N/A for all artists (Kaggle dataset miss — niche genre). "
            "S2/S4/S6 cleanly separate ghost from organic with 0.3–0.7+ score gaps."
        ),
        "exercise": "Exercise 6",
        "signal": "Aggregate Score",
    },
    "fig7_gnn_performance.png": {
        "title": "Figure 7: GNN Ghost Artist Detection Performance",
        "caption": (
            "Four-panel summary of the trained Graph Attention Network (GAT). "
            "Top-left: training loss curves (GAT converges faster than GCN). "
            "Top-right: ROC curves — both GNN models achieve AUC=1.0 on test set vs "
            "rule-based baseline (0.25 accuracy). "
            "Bottom-left: confusion matrix — zero false positives/negatives. "
            "Bottom-right: feature importance (permutation method). "
            "Dataset: 65 nodes (14 ghost, 51 organic), 692 edges, 8 node features."
        ),
        "exercise": "Exercise 7",
        "signal": "GNN Model",
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

# Known artists in Neo4j (for the analyzer quick-pick)
KNOWN_ARTISTS = {
    "Relaxing White Noise (ghost)": "6bo3atMVp3qFECNALVwq9N",
    "Meditation Relax Club (ghost)": "3BqBPFLxBkzKQTkuBPGMNF",
    "Calmo (ghost candidate)": "4Wx3ZL6d6p1gVMtwQ2YWsz",
    "Nils Frahm (organic)": "5hVghJ3sCFHFJoLnSHySjL",
}

CROSS_PLATFORM_DATA = {
    "Relaxing White Noise": {
        "artist_id": "6bo3atMVp3qFECNALVwq9N",
        "youtube_views": 353_775_028,
        "youtube_channel": "Relaxing White Noise",
        "apple_music": True,
        "s7_score": 0.00,
        "verdict": "LIKELY_GHOST",
        "note": "353M YouTube views — ghost behavior is Spotify-specific, not cross-platform absence",
    },
    "Meditation Relax Club": {
        "artist_id": "3BqBPFLxBkzKQTkuBPGMNF",
        "youtube_views": 157_581_269,
        "youtube_channel": "Meditation Relax Club",
        "apple_music": True,
        "s7_score": 0.00,
        "verdict": "LIKELY_GHOST",
        "note": "157M YouTube views — major cross-platform presence alongside Spotify",
    },
    "Calmo": {
        "artist_id": "4Wx3ZL6d6p1gVMtwQ2YWsz",
        "youtube_views": 155,
        "youtube_channel": None,
        "apple_music": False,
        "s7_score": 0.48,
        "verdict": "SUSPICIOUS",
        "note": "155 YouTube views — truly invisible cross-platform, unlike RWN/MRC",
    },
    "Nils Frahm": {
        "artist_id": "5hVghJ3sCFHFJoLnSHySjL",
        "youtube_views": 9_107_596,
        "youtube_channel": "Nils Frahm",
        "apple_music": True,
        "s7_score": 0.00,
        "verdict": "LIKELY_ORGANIC",
        "note": "9M YouTube views with verified Apple Music — consistent organic cross-platform presence",
    },
}

# ── Sidebar navigation ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎵 EDA for Music")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["🏠 Home", "📊 Exercise Gallery", "🔍 Artist Analyzer", "🕸️ Network Explorer", "📡 Cross-Platform", "ℹ️ About"],
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
        "**Status:** Exercises 1–7 complete. All 7 signals implemented and tested. "
        "GAT + GCN models trained (100% test accuracy). "
        "Key finding: ghost artists ARE visible on YouTube (353M views for RWN) — "
        "stream farming is Spotify-specific, not cross-platform absence."
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
        "Run all 7 ghost-detection signals on any artist in Neo4j. "
        "Analysis uses **only cached data** — no live Spotify API calls."
    )

    # Quick-pick from known artists
    col_pick, col_or, col_input = st.columns([2, 0.3, 3])
    with col_pick:
        quick_pick = st.selectbox(
            "Quick-pick a known artist",
            ["— select —"] + list(KNOWN_ARTISTS.keys()),
        )
    with col_or:
        st.markdown("<div style='text-align:center;padding-top:30px;color:#64748b;'>or</div>", unsafe_allow_html=True)
    with col_input:
        manual_id = st.text_input(
            "Enter Spotify Artist ID manually",
            placeholder="e.g. 6bo3atMVp3qFECNALVwq9N",
            label_visibility="visible",
        )

    # Resolve the artist ID to use
    if manual_id.strip():
        selected_id = manual_id.strip()
        selected_name = None
    elif quick_pick != "— select —":
        selected_id = KNOWN_ARTISTS[quick_pick]
        selected_name = quick_pick.split(" (")[0]
    else:
        selected_id = None
        selected_name = None

    analyze = st.button("🔍 Run Analysis", type="primary", use_container_width=False)

    if analyze and not selected_id:
        st.error("Please select an artist or enter an Artist ID.")

    if analyze and selected_id:
        with st.spinner(f"Running 7-signal analysis for `{selected_id}`…"):
            try:
                from src.signals.verdict import compute_verdict_gnn
                result = compute_verdict_gnn(
                    artist_id=selected_id,
                    artist_name=selected_name,
                    run_s7=False,
                )

                artist_name = result.get("artist_name", selected_id)
                verdict_label = result.get("verdict", "UNKNOWN")
                overall_score = result.get("overall_score", 0.0)
                rule_score = result.get("rule_based_score", overall_score)
                gnn_score = result.get("gnn_score")
                gnn_available = result.get("gnn_available", False)
                confidence = result.get("confidence", 0.0)
                explanation = result.get("explanation", "")
                signal_scores = result.get("signal_scores", {})

                st.markdown(f"## Results: {artist_name}")
                st.caption(f"Artist ID: `{selected_id}`")

                # Verdict badge
                if verdict_label == "LIKELY_GHOST":
                    badge_class = "verdict-ghost"
                    badge_icon = "🚨"
                elif verdict_label == "SUSPICIOUS":
                    badge_class = "verdict-suspicious"
                    badge_icon = "⚠️"
                else:
                    badge_class = "verdict-organic"
                    badge_icon = "✅"

                col_v1, col_v2, col_v3, col_v4 = st.columns(4)
                with col_v1:
                    st.markdown(
                        f"<span class='{badge_class}'>{badge_icon} {verdict_label}</span>",
                        unsafe_allow_html=True,
                    )
                with col_v2:
                    st.metric("Combined Score", f"{overall_score:.3f}", help="0=organic, 1=ghost")
                with col_v3:
                    st.metric(
                        "Rule-based Score",
                        f"{rule_score:.3f}",
                        help="Weighted average of 7 signals",
                    )
                with col_v4:
                    if gnn_available and gnn_score is not None:
                        st.metric("GNN Score (GAT)", f"{gnn_score:.3f}", help="Graph Attention Network probability")
                    else:
                        st.metric("GNN Score (GAT)", "N/A", help="Model not available or isolated inference")

                if explanation:
                    st.markdown(f"> {explanation}")

                st.markdown("---")
                st.markdown("### Signal Scores")

                SIGNAL_META = {
                    "s1_audio_similarity":     ("S1", "Audio Fingerprint Similarity",   "Cosine sim vs Kaggle dataset"),
                    "s2_cadence_sync":          ("S2", "Release Cadence Synchrony",       "Same-day release closure rate"),
                    "s3_playlist_cooccurrence": ("S3", "Playlist Co-occurrence",           "ISRC Jaccard proxy"),
                    "s4_follower_ratio":        ("S4", "Catalog Density Anomaly",          "Tracks/day upload rate"),
                    "s5_metadata_similarity":   ("S5", "Artist Metadata Similarity",       "TF-IDF ghost keyword score"),
                    "s6_graph_density":         ("S6", "Graph Density / HHI",              "ISRC production company HHI"),
                    "s7_cross_platform":        ("S7", "Cross-Platform Discrepancy",        "YouTube + Apple Music presence"),
                }

                for key, (num, name, desc) in SIGNAL_META.items():
                    raw = signal_scores.get(key)
                    if raw is None:
                        score_str = "N/A"
                        score_color = "#64748b"
                        bar_pct = 0
                    else:
                        score_str = f"{raw:.3f}"
                        bar_pct = int(raw * 100)
                        score_color = "#e74c3c" if raw > 0.7 else ("#f59e0b" if raw > 0.4 else "#22c55e")

                    st.markdown(f"""
                    <div class='signal-card'>
                        <span class='score' style='color:{score_color};'>{score_str}</span>
                        <div class='name'>{num}: {name}</div>
                        <div class='desc'>{desc}</div>
                        <div style='margin-top:6px;background:#2a2a4a;border-radius:4px;height:6px;'>
                            <div style='width:{bar_pct}%;background:{score_color};border-radius:4px;height:6px;'></div>
                        </div>
                    </div>""", unsafe_allow_html=True)

                # Show radar figure if it exists
                radar_path = FIGURES_DIR / "fig6_signal_radar.png"
                if radar_path.exists():
                    st.markdown("---")
                    st.markdown("### Signal Radar (Exercise 6 — all artists)")
                    st.image(str(radar_path), use_container_width=True)

            except Exception as e:
                st.error(f"Analysis failed: {e}")
                st.markdown(
                    "Make sure Neo4j is running and the artist is in the database. "
                    "Check `backend/main.py` for connection details."
                )

    elif not analyze:
        # Show placeholder when nothing has been run yet
        st.markdown("### Signal Scores")
        st.markdown(
            "<div style='color:#64748b;font-size:0.83rem;margin-bottom:12px;'>"
            "Select an artist above and click **Run Analysis** to see live scores."
            "</div>",
            unsafe_allow_html=True,
        )

        placeholder_sigs = [
            ("S1", "Audio Fingerprint Similarity", "—", "#64748b"),
            ("S2", "Release Cadence Synchrony",    "—", "#64748b"),
            ("S3", "Playlist Co-occurrence",        "—", "#64748b"),
            ("S4", "Catalog Density Anomaly",       "—", "#64748b"),
            ("S5", "Artist Metadata Similarity",    "—", "#64748b"),
            ("S6", "Graph Density / HHI",           "—", "#64748b"),
            ("S7", "Cross-Platform Discrepancy",    "—", "#64748b"),
        ]
        for num, name, score, color in placeholder_sigs:
            st.markdown(f"""
            <div class='signal-card'>
                <span class='score' style='color:{color};'>{score}</span>
                <div class='name'>{num}: {name}</div>
            </div>""", unsafe_allow_html=True)

# ── NETWORK EXPLORER PAGE ──────────────────────────────────────────────────────
elif page == "🕸️ Network Explorer":
    st.markdown("# 🕸️ Network Explorer")
    st.markdown(
        "ISRC-based production company graph — connects artists to the companies "
        "that registered their tracks. High concentration (HHI > 0.6) signals bulk-upload operations."
    )
    st.markdown("---")

    # Live stats from Neo4j
    st.markdown("### Graph Stats (Neo4j AuraDB)")
    try:
        from src.graph.neo4j_client import Neo4jClient
        neo4j = Neo4jClient()
        counts = neo4j.count_nodes()
        rel_rows = neo4j.run(
            "MATCH ()-[r]->() RETURN type(r) AS rel_type, count(r) AS cnt ORDER BY cnt DESC"
        )
        total_rels = sum(r["cnt"] for r in rel_rows)

        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Total Nodes", f"{sum(counts.values()):,}")
        col2.metric("Artists", counts.get("Artist", 0))
        col3.metric("Tracks", counts.get("Track", 0))
        col4.metric("Albums", counts.get("Album", 0))
        col5.metric("Relationships", f"{total_rels:,}")

        # ISRC cluster table
        st.markdown("---")
        st.markdown("### ISRC Production Company Clusters")
        st.markdown(
            "Companies that register tracks for multiple artists are the strongest fraud signal."
        )

        cluster_rows = neo4j.run(
            """
            MATCH (a:Artist)-[:RELEASED]->(al:Album)-[:CONTAINS]->(t:Track)
                  -[:REGISTERED_WITH]->(c:ProductionCompany)
            WITH c, collect(DISTINCT a.name) AS artists,
                 count(DISTINCT t) AS track_count
            RETURN c.isrc_prefix AS prefix,
                   c.name AS company_name,
                   artists,
                   size(artists) AS artist_count,
                   track_count
            ORDER BY track_count DESC
            """
        )

        if cluster_rows:
            import pandas as pd
            df = pd.DataFrame(cluster_rows)
            df["artists"] = df["artists"].apply(lambda x: ", ".join(x) if x else "—")
            df["shared"] = df["artist_count"].apply(lambda x: "⚠️ SHARED" if x > 1 else "—")
            df = df.rename(columns={
                "prefix": "ISRC Prefix",
                "company_name": "Company",
                "artists": "Artists",
                "artist_count": "# Artists",
                "track_count": "# Tracks",
                "shared": "Flag",
            })
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("No ISRC cluster data found in Neo4j.")

        # Per-artist neighborhood
        st.markdown("---")
        st.markdown("### Artist Neighborhood")
        selected_artist_graph = st.selectbox(
            "Select artist to view neighborhood",
            list(KNOWN_ARTISTS.keys())[:3],  # Only the 3 in Neo4j
            key="net_artist_select",
        )
        if selected_artist_graph:
            artist_id_graph = KNOWN_ARTISTS[selected_artist_graph]
            neighborhood = neo4j.run(
                """
                MATCH (a:Artist {spotify_id: $id})-[:RELEASED]->(al:Album)
                      -[:CONTAINS]->(t:Track)-[:REGISTERED_WITH]->(c:ProductionCompany)
                RETURN c.isrc_prefix AS prefix, c.name AS company,
                       count(t) AS track_count
                ORDER BY track_count DESC
                """,
                id=artist_id_graph,
            )
            if neighborhood:
                import pandas as pd
                ndf = pd.DataFrame(neighborhood)
                ndf = ndf.rename(columns={
                    "prefix": "ISRC Prefix",
                    "company": "Production Company",
                    "track_count": "Tracks",
                })
                total_tracks = ndf["Tracks"].sum()
                ndf["Share %"] = (ndf["Tracks"] / total_tracks * 100).round(1)
                st.dataframe(ndf, use_container_width=True, hide_index=True)

                # HHI
                shares = ndf["Tracks"] / total_tracks
                hhi = (shares ** 2).sum()
                hhi_color = "#e74c3c" if hhi > 0.6 else ("#f59e0b" if hhi > 0.35 else "#22c55e")
                st.markdown(
                    f"**HHI:** <span style='color:{hhi_color};font-weight:700;'>{hhi:.3f}</span> "
                    f"({'Highly concentrated' if hhi > 0.6 else 'Moderately concentrated' if hhi > 0.35 else 'Distributed'})",
                    unsafe_allow_html=True,
                )
            else:
                st.info(f"No neighborhood data found for {selected_artist_graph}.")

    except Exception as e:
        st.warning(f"Could not connect to Neo4j: {e}")
        st.markdown("Showing static figures from Exercise notebooks instead.")

    # Static figures as fallback / complement
    st.markdown("---")
    st.markdown("### Exercise Figures")
    fig4_path = FIGURES_DIR / "fig4_bipartite_neighborhood.png"
    if fig4_path.exists():
        st.markdown("**Exercise 4: Artist × Production Company Bipartite Neighborhood**")
        st.image(str(fig4_path), use_container_width=True)

    fig3_path = FIGURES_DIR / "fig3_isrc_join.png"
    if fig3_path.exists():
        st.markdown("**Exercise 3: ISRC Production Company Attribution**")
        st.image(str(fig3_path), use_container_width=True)

# ── CROSS-PLATFORM PAGE ────────────────────────────────────────────────────────
elif page == "📡 Cross-Platform":
    st.markdown("# 📡 Cross-Platform Discrepancy (Signal 7)")
    st.markdown(
        "Ghost artists are Spotify-specific stream farmers. "
        "**Key finding:** Major 'ghost' artists have massive YouTube presence — "
        "invalidating the assumption that ghost = invisible cross-platform."
    )

    st.warning(
        "**Surprise finding (Exercise 6):** "
        "Relaxing White Noise has **353 million YouTube views**. "
        "Meditation Relax Club has **157 million**. "
        "These artists are not hiding — they are exploiting Spotify's stream-farming economics specifically."
    )

    st.markdown("---")
    st.markdown("### Signal 7 Results by Artist")

    for artist_name, data in CROSS_PLATFORM_DATA.items():
        verdict = data["verdict"]
        if verdict == "LIKELY_GHOST":
            border_color = "#e74c3c"
            verdict_icon = "🚨"
        elif verdict == "SUSPICIOUS":
            border_color = "#f59e0b"
            verdict_icon = "⚠️"
        else:
            border_color = "#22c55e"
            verdict_icon = "✅"

        s7 = data["s7_score"]
        s7_color = "#e74c3c" if s7 > 0.7 else ("#f59e0b" if s7 > 0.4 else "#22c55e")

        yt_views = data["youtube_views"]
        if yt_views >= 1_000_000:
            yt_str = f"{yt_views / 1_000_000:.0f}M"
        elif yt_views >= 1_000:
            yt_str = f"{yt_views / 1_000:.0f}K"
        else:
            yt_str = str(yt_views)

        apple_str = "✅ On Apple Music" if data["apple_music"] else "❌ Not on Apple Music"
        channel_str = f"YouTube: {data['youtube_channel']}" if data["youtube_channel"] else "No YouTube channel found"

        st.markdown(f"""
        <div style='background:#1a1a2e;border-left:4px solid {border_color};border-radius:8px;padding:16px 20px;margin:10px 0;'>
            <div style='display:flex;justify-content:space-between;align-items:center;'>
                <span style='color:#e2e8f0;font-size:1.1rem;font-weight:700;'>{verdict_icon} {artist_name}</span>
                <span style='color:{s7_color};font-weight:700;font-size:1.1rem;'>S7: {s7:.2f}</span>
            </div>
            <div style='margin-top:8px;display:flex;gap:24px;flex-wrap:wrap;'>
                <span style='color:#a78bfa;'>▶ YouTube Views: <b style='color:#e2e8f0;'>{yt_str}</b></span>
                <span style='color:#a78bfa;'>{channel_str}</span>
                <span style='color:#a78bfa;'>{apple_str}</span>
            </div>
            <div style='margin-top:8px;color:#94a3b8;font-size:0.85rem;'>{data["note"]}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### YouTube Views Comparison")

    # Bar chart
    try:
        import plotly.graph_objects as go

        artists = list(CROSS_PLATFORM_DATA.keys())
        views = [d["youtube_views"] for d in CROSS_PLATFORM_DATA.values()]
        verdicts = [d["verdict"] for d in CROSS_PLATFORM_DATA.values()]
        colors = ["#e74c3c" if v == "LIKELY_GHOST" else ("#f59e0b" if v == "SUSPICIOUS" else "#22c55e")
                  for v in verdicts]

        fig = go.Figure(go.Bar(
            x=artists,
            y=views,
            marker_color=colors,
            text=[f"{v/1e6:.0f}M" if v > 1e6 else str(v) for v in views],
            textposition="outside",
        ))
        fig.update_layout(
            title="YouTube View Counts by Artist",
            yaxis_title="YouTube Views",
            plot_bgcolor="#12121f",
            paper_bgcolor="#12121f",
            font_color="#e2e8f0",
            title_font_color="#a78bfa",
            yaxis=dict(gridcolor="#2a2a4a"),
            xaxis=dict(gridcolor="#2a2a4a"),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)
    except ImportError:
        # Fallback to st.bar_chart
        import pandas as pd
        chart_df = pd.DataFrame({
            "Artist": list(CROSS_PLATFORM_DATA.keys()),
            "YouTube Views": [d["youtube_views"] for d in CROSS_PLATFORM_DATA.values()],
        }).set_index("Artist")
        st.bar_chart(chart_df)

    st.markdown("---")
    st.markdown("### What This Means")
    st.markdown("""
    **Traditional assumption:** Ghost artists are invisible cross-platform — no YouTube channel,
    no Apple Music, no social media footprint.

    **What we found:**
    - **Relaxing White Noise** and **Meditation Relax Club** have massive YouTube presence (100M–353M views).
    - These are legitimate ambient/relaxation content creators — but they also farm Spotify streams.
    - The ghost behavior is **Spotify-economic**, not content-quality based.
    - Only **Calmo** (155 YouTube views) matches the "invisible" stereotype.

    **Implication for Signal 7:**
    - S7 is inconclusive for the relaxation sub-genre.
    - For other genres (e.g., fake pop artists), cross-platform absence remains a valid signal.
    - Weight S7 lower (0.10) and rely on S2/S4/S6 as primary discriminators.
    """)

    # Show radar chart for reference
    radar_path = FIGURES_DIR / "fig6_signal_radar.png"
    if radar_path.exists():
        st.markdown("---")
        st.markdown("### Exercise 6: Full Signal Radar")
        st.image(str(radar_path), use_container_width=True)
        st.caption(
            "S7 (outermost ring) scores near-zero for RWN and MRC despite ghost verdict — "
            "confirming cross-platform presence does not rule out stream farming."
        )

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
                              CrewAI Pipeline        Figures (paper/)
                              FastAPI Backend
                              (this dashboard)
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
            "Agents": ["crewai 1.14", "openai"],
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
