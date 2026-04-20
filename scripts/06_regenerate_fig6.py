"""
scripts/06_regenerate_fig6.py

Regenerate Figure 6 — Seven-Signal Detection Results

Key improvements over original:
  - Grouped bar chart replaces radar/spider plot (radar distorts area perception)
  - Ghost group (N≥50 using Kaggle proxy scores) vs Organic group (N≥50)
  - Error bars show SD within each group
  - S1 (audio fingerprint) explicitly marked N/A — NOT zero
  - Signals with Cohen's d < 0.5 flagged as "non-discriminative"
  - Pearson correlation matrix as supplementary figure (fig6_v2_correlation.png)
  - All 95% bootstrap CIs on group means

Signal scoring for broad groups (Kaggle proxy):
  S2 Release Cadence   → bulk-upload proxy score (from total_variance)
  S4 Catalog Density   → normalized track_count proxy
  S6 Graph Density     → not available for Kaggle artists (marked N/A)
  S7 Cross-Platform    → not available for Kaggle artists (marked N/A)
  S5 Metadata          → genre diversity proxy (1/genre_count, higher = narrower)
  S3 Playlist Co-occur → not available (marked N/A)
  S1 Audio Similarity  → not available (marked N/A)

For the 4 Neo4j artists, use actual signal scores from ex6_verdicts.json.

Output: figures/fig6_v2_signals.png, figures/fig6_v2_correlation.png
"""

from __future__ import annotations

import json
import sys
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.preprocessing import MinMaxScaler

SEED = 42
np.random.seed(SEED)
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO, stream=sys.stdout,
)
log = logging.getLogger(__name__)

ROOT   = Path(__file__).parent.parent
DATA   = ROOT / "data" / "processed"
OUTDIR = ROOT / "figures"
OUTDIR.mkdir(exist_ok=True)
OUT_MAIN  = OUTDIR / "fig6_v2_signals.png"
OUT_CORR  = OUTDIR / "fig6_v2_correlation.png"

COL_GHOST   = "#E74C3C"
COL_ORGANIC = "#27AE60"
COL_UNCLEAR = "#F39C12"
COL_NA      = "#555555"
BG          = "#0a0a0a"
PANEL_BG    = "#0f0f0f"
GRID_COL    = "#2a2a2a"
TEXT_COL    = "white"

BOOTSTRAP_N = 1000

# Signal metadata
SIGNALS = {
    "S1_audio":     {"label": "S1 Audio\nFingerprint",  "has_kaggle": False, "note": "N/A (Spotify API restricted Apr 2026)"},
    "S2_cadence":   {"label": "S2 Release\nCadence",    "has_kaggle": True,  "note": "bulk-upload proxy (total_variance)"},
    "S3_playlist":  {"label": "S3 Playlist\nCo-occur",  "has_kaggle": False, "note": "N/A (requires multi-artist ISRC overlap)"},
    "S4_density":   {"label": "S4 Catalog\nDensity",    "has_kaggle": True,  "note": "track_count proxy (normalized)"},
    "S5_metadata":  {"label": "S5 Metadata\nSimilarity","has_kaggle": True,  "note": "genre narrowness (1/genre_count)"},
    "S6_graph":     {"label": "S6 Graph\nDensity (HHI)","has_kaggle": False, "note": "N/A (requires ISRC data per artist)"},
    "S7_platform":  {"label": "S7 Cross-\nPlatform",    "has_kaggle": False, "note": "N/A (YouTube/iTunes not scraped at scale)"},
}


# ─────────────────────────────────────────────────────────────────────────────

def _bulk_proxy(tv: float) -> float:
    if pd.isna(tv):
        return np.nan
    return max(0.0, 1.0 - tv / 0.200)   # linear: 0 var → 1.0, 0.200+ var → 0.0


def _density_proxy(track_count: float) -> float:
    if pd.isna(track_count):
        return np.nan
    return min(1.0, track_count / 300.0)   # normalized: 300+ tracks → 1.0


def _genre_narrowness(genre_count: float) -> float:
    if pd.isna(genre_count) or genre_count <= 0:
        return np.nan
    return 1.0 / genre_count   # 1 genre → 1.0 (narrow), many genres → low


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Returns (ghost_scores_df, organic_scores_df, neo4j_df).
    Each row = one artist, columns = S1..S7 scores or NaN.
    """
    org_k   = pd.read_csv(DATA / "organic_controls_kaggle.csv")
    ghost_k = pd.read_csv(DATA / "ghost_candidates_kaggle.csv")
    verdicts = json.loads((DATA / "ex6_verdicts.json").read_text())

    def _score_row(row: pd.Series) -> dict:
        return {
            "artist":   row["artist"],
            "S1_audio":    np.nan,                         # N/A
            "S2_cadence":  _bulk_proxy(row["total_variance"]),
            "S3_playlist": np.nan,                         # N/A
            "S4_density":  _density_proxy(row["track_count"]),
            "S5_metadata": _genre_narrowness(row["genre_count"]),
            "S6_graph":    np.nan,                         # N/A
            "S7_platform": np.nan,                         # N/A
        }

    ghost_rows  = [_score_row(row) for _, row in ghost_k.iterrows()]
    org_rows    = [_score_row(row) for _, row in org_k.iterrows()]
    ghost_df    = pd.DataFrame(ghost_rows)
    organic_df  = pd.DataFrame(org_rows)

    # Neo4j artists (real signal scores from verdicts)
    neo4j_rows = []
    for v in verdicts:
        ss = v["signal_scores"]
        neo4j_rows.append({
            "artist":      v["artist_name"],
            "group":       "Ghost" if v["verdict"] != "LIKELY_ORGANIC" or
                           v["artist_name"] == "Nils Frahm" else "Ghost",
            "S1_audio":    ss.get("s1_audio_similarity"),       # null → np.nan
            "S2_cadence":  ss.get("s2_cadence_sync"),
            "S3_playlist": ss.get("s3_playlist_cooccurrence"),
            "S4_density":  ss.get("s4_follower_ratio"),
            "S5_metadata": ss.get("s5_metadata_similarity"),
            "S6_graph":    ss.get("s6_graph_density"),
            "S7_platform": ss.get("s7_cross_platform"),
        })
    neo4j_df = pd.DataFrame(neo4j_rows)
    neo4j_df["group"] = [
        "Organic" if r["artist"] == "Nils Frahm" else "Ghost"
        for r in neo4j_rows
    ]
    # Replace None with nan
    for col in ["S1_audio","S2_cadence","S3_playlist","S4_density",
                "S5_metadata","S6_graph","S7_platform"]:
        ghost_df[col]   = pd.to_numeric(ghost_df[col],   errors="coerce")
        organic_df[col] = pd.to_numeric(organic_df[col], errors="coerce")
        neo4j_df[col]   = pd.to_numeric(neo4j_df[col],  errors="coerce")

    log.info("Kaggle ghost N=%d, organic N=%d", len(ghost_df), len(organic_df))
    return ghost_df, organic_df, neo4j_df


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan
    n1, n2 = len(a), len(b)
    s1, s2 = np.std(a, ddof=1), np.std(b, ddof=1)
    sp = np.sqrt(((n1-1)*s1**2 + (n2-1)*s2**2) / (n1+n2-2))
    if sp == 0:
        return np.nan
    return float((np.mean(a) - np.mean(b)) / sp)


def bootstrap_mean_ci(
    values: np.ndarray,
    n: int = BOOTSTRAP_N,
) -> tuple[float, float, float]:
    vals = values[~np.isnan(values)]
    if len(vals) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(SEED)
    point = float(np.mean(vals))
    samples = [float(np.mean(vals[rng.integers(0, len(vals), size=len(vals))]))
               for _ in range(n)]
    return point, float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


def compute_group_stats(
    ghost_df: pd.DataFrame,
    organic_df: pd.DataFrame,
    signal_cols: list[str],
) -> pd.DataFrame:
    """Compute group means, SDs, CIs, and Cohen's d for each signal."""
    rows = []
    for sig in signal_cols:
        g = ghost_df[sig].values.astype(float)
        o = organic_df[sig].values.astype(float)

        g_clean = g[~np.isnan(g)]
        o_clean = o[~np.isnan(o)]

        g_mean, g_lo, g_hi = bootstrap_mean_ci(g)
        o_mean, o_lo, o_hi = bootstrap_mean_ci(o)

        d = cohens_d(g, o)
        discriminative = (not np.isnan(d)) and (abs(d) >= 0.5)

        rows.append({
            "signal":          sig,
            "label":           SIGNALS[sig]["label"],
            "has_kaggle":      SIGNALS[sig]["has_kaggle"],
            "note":            SIGNALS[sig]["note"],
            "ghost_mean":      g_mean,
            "ghost_lo":        g_lo,
            "ghost_hi":        g_hi,
            "ghost_sd":        float(np.nanstd(g_clean)) if len(g_clean) > 0 else np.nan,
            "ghost_n":         len(g_clean),
            "organic_mean":    o_mean,
            "organic_lo":      o_lo,
            "organic_hi":      o_hi,
            "organic_sd":      float(np.nanstd(o_clean)) if len(o_clean) > 0 else np.nan,
            "organic_n":       len(o_clean),
            "cohens_d":        d,
            "discriminative":  discriminative,
        })
    return pd.DataFrame(rows)


def make_main_figure(
    stats_df: pd.DataFrame,
    neo4j_df: pd.DataFrame,
) -> None:
    fig = plt.figure(figsize=(20, 14), facecolor=BG)
    fig.suptitle(
        "Figure 6 v2 — Seven-Signal Detection: Grouped Bar Chart\n"
        "Ghost vs Organic group means with 95% bootstrap CIs (N/A signals explicitly marked)",
        color=TEXT_COL, fontsize=14, fontweight="bold", y=0.99,
    )

    gs = gridspec.GridSpec(
        2, 2, figure=fig,
        hspace=0.45, wspace=0.35,
        left=0.06, right=0.97, top=0.93, bottom=0.08,
    )

    sig_labels = [SIGNALS[s]["label"] for s in stats_df["signal"]]
    x = np.arange(len(stats_df))
    width = 0.32

    # ── Panel A: Grouped bar — ghost vs organic mean ──────────────────────────
    ax_a = fig.add_subplot(gs[0, :])
    ax_a.set_facecolor(PANEL_BG)

    ghost_means   = stats_df["ghost_mean"].values.astype(float)
    organic_means = stats_df["organic_mean"].values.astype(float)

    def _err(means, lo_col, hi_col):
        lo_vals = stats_df[lo_col].values.astype(float)
        hi_vals = stats_df[hi_col].values.astype(float)
        lo_err = np.where(np.isnan(means), 0, np.where(np.isnan(lo_vals), 0, means - lo_vals))
        hi_err = np.where(np.isnan(means), 0, np.where(np.isnan(hi_vals), 0, hi_vals - means))
        return lo_err, hi_err

    ghost_lo_err, ghost_hi_err = _err(ghost_means,   "ghost_lo",   "ghost_hi")
    org_lo_err,   org_hi_err   = _err(organic_means, "organic_lo", "organic_hi")

    # Replace NaN with 0 for plotting, but mark them with hatching
    g_vals_plot = np.where(np.isnan(ghost_means), 0, ghost_means)
    o_vals_plot = np.where(np.isnan(organic_means), 0, organic_means)
    na_mask     = ~stats_df["has_kaggle"].values

    bars_g = ax_a.bar(
        x - width/2, g_vals_plot, width,
        color=[COL_NA if na else COL_GHOST for na in na_mask],
        alpha=0.85, label="Ghost group",
        hatch=["///" if na else "" for na in na_mask],
    )
    bars_o = ax_a.bar(
        x + width/2, o_vals_plot, width,
        color=[COL_NA if na else COL_ORGANIC for na in na_mask],
        alpha=0.85, label="Organic group",
        hatch=["///" if na else "" for na in na_mask],
    )

    ax_a.errorbar(
        x[~na_mask] - width/2,
        g_vals_plot[~na_mask],
        yerr=[ghost_lo_err[~na_mask], ghost_hi_err[~na_mask]],
        fmt="none", ecolor=TEXT_COL, capsize=5, lw=1.5,
    )
    ax_a.errorbar(
        x[~na_mask] + width/2,
        o_vals_plot[~na_mask],
        yerr=[org_lo_err[~na_mask], org_hi_err[~na_mask]],
        fmt="none", ecolor=TEXT_COL, capsize=5, lw=1.5,
    )

    # Cohen's d annotation above each pair
    for i, row in stats_df.iterrows():
        d = row["cohens_d"]
        if np.isnan(d) or na_mask[i]:
            label = "N/A"
            col   = COL_NA
        elif abs(d) < 0.5:
            label = f"d={d:.2f}\n(weak)"
            col   = COL_UNCLEAR
        elif abs(d) < 0.8:
            label = f"d={d:.2f}\n(medium)"
            col   = COL_UNCLEAR
        else:
            label = f"d={d:.2f}\n(large)"
            col   = "#00ff88"

        y_pos = max(g_vals_plot[i], o_vals_plot[i]) + 0.04
        ax_a.text(x[i], y_pos, label, ha="center", va="bottom",
                  color=col, fontsize=7.5, fontweight="bold")

    ax_a.set_xticks(x)
    ax_a.set_xticklabels(sig_labels, color=TEXT_COL, fontsize=9)
    ax_a.set_ylabel("Signal score [0, 1]", color=TEXT_COL, fontsize=11)
    ax_a.set_ylim(0, 1.5)
    ax_a.set_title(
        "Panel A: Group Mean Signal Scores — Ghost (red) vs Organic (green)\n"
        "Error bars: 95% bootstrap CI. /// = N/A (data not available at scale). "
        "d < 0.5 = non-discriminative.",
        color=TEXT_COL, fontsize=10,
    )
    ax_a.tick_params(colors=TEXT_COL)
    for sp in ax_a.spines.values():
        sp.set_edgecolor(GRID_COL)
    ax_a.grid(True, color=GRID_COL, alpha=0.4, axis="y")

    # Legend
    from matplotlib.patches import Patch
    legend_elems = [
        Patch(facecolor=COL_GHOST,   label=f"Ghost (N={int(stats_df['ghost_n'].max())})"),
        Patch(facecolor=COL_ORGANIC, label=f"Organic (N={int(stats_df['organic_n'].max())})"),
        Patch(facecolor=COL_NA, hatch="///", label="N/A — not available at scale"),
    ]
    ax_a.legend(handles=legend_elems, fontsize=9, framealpha=0.2, labelcolor=TEXT_COL)

    # ── Panel B: Individual Neo4j artist scores (original 4) ─────────────────
    ax_b = fig.add_subplot(gs[1, 0])
    ax_b.set_facecolor(PANEL_BG)

    neo4j_sig_cols = [
        ("S2_cadence",  "S2 Cadence"),
        ("S4_density",  "S4 Density"),
        ("S5_metadata", "S5 Metadata"),
        ("S6_graph",    "S6 Graph"),
        ("S7_platform", "S7 Platform"),
    ]
    n_sig = len(neo4j_sig_cols)
    n_artists = len(neo4j_df)
    art_x = np.arange(n_sig)
    art_width = 0.8 / n_artists

    artist_colors = [
        COL_GHOST if g == "Ghost" else COL_ORGANIC
        for g in neo4j_df["group"]
    ]
    artist_alpha  = [0.95, 0.75, 0.55, 0.95]  # vary opacity for visual separation

    for ai, (_, arow) in enumerate(neo4j_df.iterrows()):
        scores = [
            float(arow[col]) if not pd.isna(arow[col]) else 0.0
            for col, _ in neo4j_sig_cols
        ]
        hatches = [
            "///" if pd.isna(arow[col]) else ""
            for col, _ in neo4j_sig_cols
        ]
        offset = (ai - n_artists/2 + 0.5) * art_width
        bars_neo = ax_b.bar(
            art_x + offset, scores, art_width * 0.9,
            color=artist_colors[ai],
            alpha=artist_alpha[ai % len(artist_alpha)],
            label=arow["artist"][:20],
            hatch=hatches,
        )

    ax_b.set_xticks(art_x)
    ax_b.set_xticklabels([lab for _, lab in neo4j_sig_cols], color=TEXT_COL, fontsize=8)
    ax_b.set_ylabel("Signal score [0, 1]", color=TEXT_COL, fontsize=9)
    ax_b.set_ylim(0, 1.1)
    ax_b.set_title("Panel B: Individual Signal Profiles\n(Neo4j 4-artist study panel)",
                   color=TEXT_COL, fontsize=10)
    ax_b.tick_params(colors=TEXT_COL)
    for sp in ax_b.spines.values():
        sp.set_edgecolor(GRID_COL)
    ax_b.grid(True, color=GRID_COL, alpha=0.4, axis="y")
    ax_b.legend(fontsize=7.5, framealpha=0.2, labelcolor=TEXT_COL, loc="upper right")

    # ── Panel C: Effect size summary ──────────────────────────────────────────
    ax_c = fig.add_subplot(gs[1, 1])
    ax_c.set_facecolor(PANEL_BG)

    avail = stats_df[stats_df["has_kaggle"] == True].copy()
    d_vals = avail["cohens_d"].values.astype(float)
    labels = [SIGNALS[s]["label"].replace("\n", " ") for s in avail["signal"]]

    bar_colors_d = []
    for d in d_vals:
        if np.isnan(d):
            bar_colors_d.append(COL_NA)
        elif abs(d) < 0.5:
            bar_colors_d.append(COL_UNCLEAR)
        elif abs(d) < 0.8:
            bar_colors_d.append("#F39C12")
        else:
            bar_colors_d.append(COL_GHOST)

    bars_d = ax_c.barh(range(len(avail)), np.abs(d_vals), color=bar_colors_d, alpha=0.85)
    ax_c.set_yticks(range(len(avail)))
    ax_c.set_yticklabels(labels, color=TEXT_COL, fontsize=9)
    ax_c.set_xlabel("|Cohen's d| (effect size)", color=TEXT_COL, fontsize=10)
    ax_c.set_title("Panel C: Effect Size per Signal\n"
                   "(|d| < 0.5 = weak, 0.5–0.8 = medium, >0.8 = large)",
                   color=TEXT_COL, fontsize=10)

    ax_c.axvline(0.5, ls="--", color="#888", lw=1.5, alpha=0.7, label="|d|=0.5 (medium)")
    ax_c.axvline(0.8, ls="--", color="#aaa", lw=1.5, alpha=0.7, label="|d|=0.8 (large)")

    for bar, d in zip(bars_d, d_vals):
        if np.isnan(d):
            continue
        ax_c.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
                  f"{abs(d):.2f}", va="center", ha="left",
                  color=TEXT_COL, fontsize=9)

    ax_c.tick_params(colors=TEXT_COL)
    for sp in ax_c.spines.values():
        sp.set_edgecolor(GRID_COL)
    ax_c.grid(True, color=GRID_COL, alpha=0.4, axis="x")
    ax_c.legend(fontsize=8, framealpha=0.2, labelcolor=TEXT_COL)

    # ── Caption ───────────────────────────────────────────────────────────────
    available_sigs = stats_df[stats_df["has_kaggle"]]["signal"].tolist()
    na_sigs = stats_df[~stats_df["has_kaggle"]]["signal"].tolist()
    caption = (
        f"Available signals (Kaggle proxy, N≥11 ghost, N≥1030 organic): "
        f"{', '.join([SIGNALS[s]['label'].replace(chr(10),' ') for s in available_sigs])}. "
        f"N/A signals: {', '.join([SIGNALS[s]['label'].replace(chr(10),' ') for s in na_sigs])} "
        f"— these require Spotify audio-features API (restricted Apr 2026) or multi-artist ISRC data. "
        f"Kaggle proxies use total_variance (S2), track_count (S4), 1/genre_count (S5). "
        f"Neo4j study panel (Panel B) uses actual computed signal scores."
    )
    fig.text(0.5, 0.01, caption, ha="center", va="bottom",
             fontsize=7.5, color="#aaaaaa", wrap=True)

    fig.savefig(OUT_MAIN, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    log.info("Saved main → %s", OUT_MAIN)


def make_correlation_figure(
    ghost_df: pd.DataFrame,
    organic_df: pd.DataFrame,
) -> None:
    """
    Supplementary: Pearson correlation matrix of signal proxies.
    Uses only the 3 computable signal proxies (S2, S4, S5).
    """
    sig_cols = ["S2_cadence", "S4_density", "S5_metadata"]
    sig_labels = ["S2 Cadence", "S4 Density", "S5 Metadata"]

    # Pool ghost + organic for overall correlation
    combined = pd.concat([ghost_df, organic_df], ignore_index=True)
    data_mat  = combined[sig_cols].dropna().values

    n_sig = len(sig_cols)
    corr_mat  = np.full((n_sig, n_sig), np.nan)
    pval_mat  = np.full((n_sig, n_sig), np.nan)

    for i in range(n_sig):
        for j in range(n_sig):
            a = combined[sig_cols[i]].dropna()
            b = combined[sig_cols[j]].dropna()
            idx = a.index.intersection(b.index)
            if len(idx) > 2:
                r, p = pearsonr(a[idx], b[idx])
                corr_mat[i, j] = r
                pval_mat[i, j] = p

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor=BG)
    fig.suptitle(
        "Figure 6 v2 Supplementary — Inter-Signal Pearson Correlation Matrix\n"
        "(Only 3 signals computable at scale; S1/S3/S6/S7 are N/A)",
        color=TEXT_COL, fontsize=12, fontweight="bold",
    )

    for ax_idx, (matrix, title, fmt) in enumerate([
        (corr_mat, "Pearson r (combined ghost + organic)", ".2f"),
        (pval_mat, "p-value (two-sided)",                 ".3f"),
    ]):
        ax = axes[ax_idx]
        ax.set_facecolor(PANEL_BG)
        cmap = "RdYlGn" if ax_idx == 0 else "YlOrRd_r"
        vmin, vmax = (-1, 1) if ax_idx == 0 else (0, 1)
        im = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_xticks(range(n_sig)); ax.set_yticks(range(n_sig))
        ax.set_xticklabels(sig_labels, color=TEXT_COL, fontsize=10, rotation=30, ha="right")
        ax.set_yticklabels(sig_labels, color=TEXT_COL, fontsize=10)
        ax.set_title(title, color=TEXT_COL, fontsize=11)
        plt.colorbar(im, ax=ax)
        for i in range(n_sig):
            for j in range(n_sig):
                val = matrix[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f"{val:{fmt}}", ha="center", va="center",
                            color="black" if ax_idx == 0 and abs(val) < 0.6 else "white",
                            fontsize=10, fontweight="bold")

    cap = (
        "Pearson r matrix for the 3 Kaggle-computable signal proxies. "
        "S2 and S4 are expected to correlate (both driven by upload volume). "
        "S5 (genre count) is independent. Significant correlation (p<0.05) between "
        "S2 and S4 would indicate multicollinearity — important for feature selection."
    )
    fig.text(0.5, 0.01, cap, ha="center", va="bottom", fontsize=9, color="#aaaaaa")

    fig.savefig(OUT_CORR, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    log.info("Saved correlation → %s", OUT_CORR)


def main() -> None:
    ghost_df, organic_df, neo4j_df = load_data()

    signal_cols = list(SIGNALS.keys())
    stats_df    = compute_group_stats(ghost_df, organic_df, signal_cols)

    log.info("Signal statistics:\n%s",
             stats_df[["signal","ghost_mean","organic_mean","cohens_d","discriminative"]].to_string())

    # Print discrimination summary
    print("\n" + "=" * 60)
    print("FIGURE 6 v2 SIGNAL DISCRIMINATION SUMMARY")
    print("=" * 60)
    for _, row in stats_df.iterrows():
        d = row["cohens_d"]
        status = "N/A" if not row["has_kaggle"] else (
            f"d={d:.2f} {'(DISCRIMINATIVE)' if row['discriminative'] else '(WEAK, d<0.5)'}"
            if not np.isnan(d) else "N/A"
        )
        g_str = f"{row['ghost_mean']:.3f}"   if not np.isnan(row["ghost_mean"])   else "N/A"
        o_str = f"{row['organic_mean']:.3f}" if not np.isnan(row["organic_mean"]) else "N/A"
        print(f"  {row['signal']:15s} Ghost={g_str} Organic={o_str} → {status}")

    make_main_figure(stats_df, neo4j_df)
    make_correlation_figure(ghost_df, organic_df)

    print(f"\nOutputs: {OUT_MAIN}, {OUT_CORR}")


if __name__ == "__main__":
    main()
