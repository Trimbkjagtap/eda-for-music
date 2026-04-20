"""
scripts/04_regenerate_fig1.py

Regenerate Figure 1 — Catalog Coherence PCA

Key improvements over original:
  - Same-genre controls: ambient/sleep organics vs ambient/sleep ghosts
    (original compared ambient Deuter vs rock Beatles — genre confound)
  - Levene's test for variance equality between ghost vs organic groups
  - F-test on σ² ratio (H0: ghost variance == organic variance)
  - Bootstrap CI on cluster σ² (1000 resamples)
  - Cohen's d effect size reported per feature
  - Report variance explained for each PC

Data sources:
  data/processed/organic_controls_kaggle.csv   — 1030 organic artists (genre-tagged)
  data/processed/ghost_candidates_kaggle.csv   — 11 Kaggle ghost candidates
  data/processed/ex1_catalog_features.csv      — original 6-artist dataset
  data/processed/ex1_variance_table.csv        — original variance summary

Output: figures/fig1_v2_catalog_coherence.png
"""

from __future__ import annotations

import json
import sys
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Ellipse
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import levene, f as f_dist
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

SEED = 42
np.random.seed(SEED)
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO, stream=sys.stdout,
)
log = logging.getLogger(__name__)

ROOT   = Path(__file__).parent.parent
DATA   = ROOT / "data" / "processed"
GT     = ROOT / "data" / "ground_truth"
OUTDIR = ROOT / "figures"
OUTDIR.mkdir(exist_ok=True)
OUT    = OUTDIR / "fig1_v2_catalog_coherence.png"

# ── colour palette (consistent across all figures) ────────────────────────────
COL_GHOST   = "#E74C3C"  # red
COL_ORGANIC = "#27AE60"  # green
COL_UNCLEAR = "#F39C12"  # amber
BG          = "#0a0a0a"
PANEL_BG    = "#0f0f0f"
GRID_COL    = "#2a2a2a"
TEXT_COL    = "white"

BOOTSTRAP_N = 1000

AUDIO_FEATS = ["danceability", "energy", "valence", "acousticness"]
# Kaggle dataset provides 4 features; we use all 4 (S1 signal's 4D fingerprint)

# Ambient / sleep / new-age genre keywords (for same-genre matching)
AMBIENT_KEYWORDS = [
    "ambient", "sleep", "new-age", "chill", "piano",
    "acoustic", "classical", "meditation",
]

# These Kaggle "organic" sleep artists are borderline — they may be legitimate
# ghost-adjacent (white noise generators). We exclude purely noise artists.
EXCLUDE_ARTISTS = {
    "Rain Sounds", "Nature Sounds", "Pink Noise", "White Noise Baby Sleep",
    "White Noise for Babies", "Silent Knights", "Ocean Waves For Sleep",
    "Granular", "Weißes Rauschen HD",
}


# ─────────────────────────────────────────────────────────────────────────────

def _has_ambient(genres: str) -> bool:
    if pd.isna(genres):
        return False
    gs = {x.strip().lower() for x in str(genres).split(",")}
    return bool(gs & set(AMBIENT_KEYWORDS))


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Returns (all_df, ghost_df, organic_df) where each row is one artist
    with columns: artist, group, track_count, total_variance, + 4 var columns.

    Ghost group:
      - Deuter, Bernward Koch (from ex1 — confirmed ghost-like, ambient genre)
      - Low-variance (<0.020) artists from ghost_candidates_kaggle
      - Any ghost_artists.csv entry that has Kaggle data

    Organic ambient group:
      - Ambient/sleep/new-age artists from organic_controls_kaggle
      - With min 10 tracks, not in EXCLUDE_ARTISTS
    """
    org_k  = pd.read_csv(DATA / "organic_controls_kaggle.csv")
    ghost_k = pd.read_csv(DATA / "ghost_candidates_kaggle.csv")
    ex1    = pd.read_csv(DATA / "ex1_catalog_features.csv")

    # ── Ghost ambient artists ─────────────────────────────────────────────────
    # Original ghost-like artists from ex1 (Deuter, Bernward Koch)
    ghost_ex1 = (
        ex1[ex1["group"] == "Ghost-like"]
        .groupby("artist")[AUDIO_FEATS]
        .agg(["var"])
        .reset_index()
    )
    ghost_ex1.columns = ["artist"] + [f"var_{c}" for c in AUDIO_FEATS]
    ghost_ex1["total_variance"] = ghost_ex1[[f"var_{c}" for c in AUDIO_FEATS]].sum(axis=1)
    ghost_ex1["track_count"]    = ex1[ex1["group"] == "Ghost-like"].groupby("artist").size().values
    ghost_ex1["source"]         = "ex1"

    # Kaggle ghost candidates (these are featured collab artists with 0 variance —
    # likely all tracks are the same recording, clearest ghost signal)
    ghost_kaggle = ghost_k[ghost_k["total_variance"] < 0.020].copy()
    ghost_kaggle["source"] = "kaggle"
    for c in AUDIO_FEATS:
        ghost_kaggle[f"var_{c}"] = ghost_kaggle[f"var_{c}"]

    # Combine ghosts
    ghost_cols = ["artist", "track_count", "total_variance", "source"] + [f"var_{c}" for c in AUDIO_FEATS]
    ghost_df = pd.concat([
        ghost_ex1[ghost_cols],
        ghost_kaggle[ghost_cols],
    ], ignore_index=True)
    ghost_df["group"] = "Ghost"
    ghost_df = ghost_df.drop_duplicates(subset="artist")

    # ── Organic ambient artists ───────────────────────────────────────────────
    org_ambient = org_k[
        org_k["genres"].apply(_has_ambient) &
        (org_k["track_count"] >= 10) &
        (~org_k["artist"].isin(EXCLUDE_ARTISTS))
    ].copy()
    org_ambient["source"] = "kaggle_organic"
    for c in AUDIO_FEATS:
        if f"var_{c}" not in org_ambient.columns:
            org_ambient[f"var_{c}"] = np.nan
    org_df = org_ambient[ghost_cols].copy()
    org_df["group"] = "Organic"

    log.info("Ghost ambient artists: %d", len(ghost_df))
    log.info("Organic ambient artists: %d", len(org_df))

    all_df = pd.concat([ghost_df, org_df], ignore_index=True)
    return all_df, ghost_df, org_df


# ─────────────────────────────────────────────────────────────────────────────

def bootstrap_variance_ci(
    values: np.ndarray,
    n: int = BOOTSTRAP_N,
) -> tuple[float, float, float]:
    """Bootstrap CI on group total variance. Returns (point, lo95, hi95)."""
    rng = np.random.default_rng(SEED)
    point = float(np.var(values, ddof=1))
    samples = []
    for _ in range(n):
        idx  = rng.integers(0, len(values), size=len(values))
        samp = values[idx]
        if len(samp) > 1:
            samples.append(float(np.var(samp, ddof=1)))
    lo = float(np.percentile(samples, 2.5))
    hi = float(np.percentile(samples, 97.5))
    return point, lo, hi


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d effect size (pooled SD)."""
    n1, n2 = len(a), len(b)
    s1, s2 = np.std(a, ddof=1), np.std(b, ddof=1)
    sp = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
    if sp == 0:
        return float("nan")
    return float((np.mean(a) - np.mean(b)) / sp)


def run_statistics(
    ghost_df: pd.DataFrame,
    org_df: pd.DataFrame,
) -> dict:
    """
    Run Levene's test, F-test on σ², Cohen's d per feature, bootstrap CIs.
    """
    results = {}

    # Total variance
    g_var = ghost_df["total_variance"].dropna().values
    o_var = org_df["total_variance"].dropna().values

    lev_stat, lev_p = levene(g_var, o_var)
    results["levene_stat"] = float(lev_stat)
    results["levene_p"]    = float(lev_p)

    # F-test on σ² ratio (ghost_var / organic_var)
    f_stat = np.var(g_var, ddof=1) / np.var(o_var, ddof=1)
    df1    = len(g_var) - 1
    df2    = len(o_var) - 1
    f_p    = 2 * min(
        f_dist.cdf(f_stat, df1, df2),
        1 - f_dist.cdf(f_stat, df1, df2),
    )
    results["f_stat"]  = float(f_stat)
    results["f_df"]    = (df1, df2)
    results["f_p"]     = float(f_p)

    # Bootstrap CI on cluster σ²
    g_pt, g_lo, g_hi = bootstrap_variance_ci(g_var)
    o_pt, o_lo, o_hi = bootstrap_variance_ci(o_var)
    results["ghost_var_point"] = g_pt
    results["ghost_var_ci"]    = (g_lo, g_hi)
    results["organic_var_point"] = o_pt
    results["organic_var_ci"]  = (o_lo, o_hi)

    # Per-feature Cohen's d
    d_vals = {}
    for feat in AUDIO_FEATS:
        col = f"var_{feat}"
        g = ghost_df[col].dropna().values
        o = org_df[col].dropna().values
        if len(g) > 1 and len(o) > 1:
            d_vals[feat] = cohens_d(g, o)
    results["cohens_d"] = d_vals

    # t-test on total_variance
    t_stat, t_p = stats.ttest_ind(g_var, o_var, equal_var=False)
    results["ttest_stat"] = float(t_stat)
    results["ttest_p"]    = float(t_p)

    return results


def run_pca(all_df: pd.DataFrame) -> tuple[np.ndarray, PCA, np.ndarray]:
    """
    Fit PCA on per-artist aggregate variance features.
    Returns (transformed coords, pca object, feature matrix).
    """
    feat_cols = [f"var_{c}" for c in AUDIO_FEATS]
    X = all_df[feat_cols].fillna(0).values.astype(float)

    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    pca = PCA(n_components=2, random_state=SEED)
    coords = pca.fit_transform(X_s)
    return coords, pca, X_s


def _confidence_ellipse(
    x: np.ndarray, y: np.ndarray, ax, n_std: float = 1.5, **kwargs
) -> None:
    """Draw a 1.5-sigma confidence ellipse for a 2D point cloud."""
    if len(x) < 3:
        return
    cov  = np.cov(x, y)
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    theta  = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    width  = 2 * n_std * np.sqrt(vals[0])
    height = 2 * n_std * np.sqrt(vals[1])
    ell = Ellipse(
        xy=(np.mean(x), np.mean(y)),
        width=width, height=height, angle=theta,
        **kwargs
    )
    ax.add_patch(ell)


def make_figure(
    all_df: pd.DataFrame,
    ghost_df: pd.DataFrame,
    org_df: pd.DataFrame,
    coords: np.ndarray,
    pca: PCA,
    stats_results: dict,
) -> None:
    fig = plt.figure(figsize=(20, 10), facecolor=BG)
    fig.suptitle(
        "Figure 1 v2 — Catalog Coherence: Audio Feature PCA\n"
        "Same-genre comparison: Ambient/Sleep Ghost Artists vs Organic Ambient Artists",
        color=TEXT_COL, fontsize=14, fontweight="bold", y=0.98,
    )

    # ── Left panel: PCA scatter ───────────────────────────────────────────────
    ax1 = fig.add_axes([0.05, 0.12, 0.42, 0.75])
    ax1.set_facecolor(PANEL_BG)

    g_mask = all_df["group"].values == "Ghost"
    o_mask = ~g_mask

    g_coords = coords[g_mask]
    o_coords = coords[o_mask]

    ax1.scatter(
        o_coords[:, 0], o_coords[:, 1],
        c=COL_ORGANIC, s=80, alpha=0.75, zorder=3,
        label=f"Organic ambient (N={o_mask.sum()})", edgecolors="none",
    )
    ax1.scatter(
        g_coords[:, 0], g_coords[:, 1],
        c=COL_GHOST, s=100, alpha=0.85, marker="^", zorder=4,
        label=f"Ghost ambient (N={g_mask.sum()})", edgecolors="white", linewidths=0.4,
    )

    # Ellipses
    if o_mask.sum() >= 3:
        _confidence_ellipse(
            o_coords[:, 0], o_coords[:, 1], ax1,
            n_std=1.5, edgecolor=COL_ORGANIC, facecolor=COL_ORGANIC,
            alpha=0.12, linewidth=2, linestyle="--",
        )
    if g_mask.sum() >= 3:
        _confidence_ellipse(
            g_coords[:, 0], g_coords[:, 1], ax1,
            n_std=1.5, edgecolor=COL_GHOST, facecolor=COL_GHOST,
            alpha=0.12, linewidth=2, linestyle="--",
        )

    pc1_var = pca.explained_variance_ratio_[0] * 100
    pc2_var = pca.explained_variance_ratio_[1] * 100
    ax1.set_xlabel(f"PC1 ({pc1_var:.1f}% variance explained)", color=TEXT_COL, fontsize=11)
    ax1.set_ylabel(f"PC2 ({pc2_var:.1f}% variance explained)", color=TEXT_COL, fontsize=11)
    ax1.set_title("PCA on Audio Feature Variances\n(per-artist: danceability, energy, valence, acousticness)",
                  color=TEXT_COL, fontsize=10)
    ax1.tick_params(colors=TEXT_COL)
    for sp in ax1.spines.values():
        sp.set_edgecolor(GRID_COL)
    ax1.grid(True, color=GRID_COL, alpha=0.5, linewidth=0.5)
    ax1.legend(fontsize=9, framealpha=0.2, labelcolor=TEXT_COL)

    # ── Right panel: Total variance distribution ──────────────────────────────
    ax2 = fig.add_axes([0.55, 0.12, 0.42, 0.75])
    ax2.set_facecolor(PANEL_BG)

    g_vals = ghost_df["total_variance"].dropna().values
    o_vals = org_df["total_variance"].dropna().values

    bins = np.linspace(0, max(o_vals.max(), g_vals.max()) * 1.05, 30)
    ax2.hist(o_vals, bins=bins, color=COL_ORGANIC, alpha=0.6, label=f"Organic (N={len(o_vals)})", density=True)
    ax2.hist(g_vals, bins=bins, color=COL_GHOST,   alpha=0.7, label=f"Ghost (N={len(g_vals)})",   density=True)

    # Vertical mean lines
    ax2.axvline(o_vals.mean(), color=COL_ORGANIC, lw=2, ls="--", alpha=0.9)
    ax2.axvline(g_vals.mean(), color=COL_GHOST,   lw=2, ls="--", alpha=0.9)

    # Bootstrap CI bands
    g_pt, g_lo, g_hi = stats_results["ghost_var_point"], *stats_results["ghost_var_ci"]
    o_pt, o_lo, o_hi = stats_results["organic_var_point"], *stats_results["organic_var_ci"]
    # show CI as shaded horizontal bar at y=0
    y_ci = ax2.get_ylim()[1] * 0.03 if ax2.get_ylim()[1] > 0 else 0.5
    ax2.barh(
        -0.15, g_hi - g_lo, left=g_lo, height=0.08,
        color=COL_GHOST, alpha=0.5, zorder=5,
    )
    ax2.barh(
        -0.23, o_hi - o_lo, left=o_lo, height=0.08,
        color=COL_ORGANIC, alpha=0.5, zorder=5,
    )

    # Per-feature Cohen's d annotation
    d_text = "Cohen's d per feature:\n"
    for feat, d in stats_results["cohens_d"].items():
        interp = "large" if abs(d) >= 0.8 else ("medium" if abs(d) >= 0.5 else "small")
        d_text += f"  {feat[:4]}: d={d:.2f} ({interp})\n"

    ax2.text(
        0.98, 0.98, d_text, transform=ax2.transAxes,
        color=TEXT_COL, fontsize=8.5, va="top", ha="right",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#1a1a2a", alpha=0.8),
    )

    ax2.set_xlabel("Total audio feature variance\n(sum of 4 per-feature variances, per artist)", color=TEXT_COL, fontsize=11)
    ax2.set_ylabel("Density", color=TEXT_COL, fontsize=11)
    ax2.set_title("Distribution of Within-Catalog Audio Variance\n(same genre: ambient/sleep/new-age artists only)",
                  color=TEXT_COL, fontsize=10)
    ax2.tick_params(colors=TEXT_COL)
    for sp in ax2.spines.values():
        sp.set_edgecolor(GRID_COL)
    ax2.grid(True, color=GRID_COL, alpha=0.5, linewidth=0.5)
    ax2.legend(fontsize=9, framealpha=0.2, labelcolor=TEXT_COL)

    # ── Statistical test caption ──────────────────────────────────────────────
    lev_p = stats_results["levene_p"]
    f_p   = stats_results["f_p"]
    t_p   = stats_results["ttest_p"]
    g_var_pt = stats_results["ghost_var_point"]
    o_var_pt = stats_results["organic_var_point"]
    g_lo_, g_hi_ = stats_results["ghost_var_ci"]
    o_lo_, o_hi_ = stats_results["organic_var_ci"]
    n_ghost   = len(g_vals)
    n_organic = len(o_vals)

    lev_sig   = "✓ significant" if lev_p < 0.05 else "✗ not significant"
    t_sig     = "✓" if t_p < 0.05 else "✗"

    caption = (
        f"Ghost group: N={n_ghost}, σ²={g_var_pt:.4f} [95% CI {g_lo_:.4f}–{g_hi_:.4f}]\n"
        f"Organic group: N={n_organic}, σ²={o_var_pt:.4f} [95% CI {o_lo_:.4f}–{o_hi_:.4f}]\n"
        f"Levene's test (H₀: equal variances): W={stats_results['levene_stat']:.3f}, "
        f"p={'<0.001' if lev_p < 0.001 else f'{lev_p:.3f}'} — {lev_sig}\n"
        f"F-test σ² ratio: F({stats_results['f_df'][0]},{stats_results['f_df'][1]})="
        f"{stats_results['f_stat']:.3f}, p={'<0.001' if f_p < 0.001 else f'{f_p:.3f}'}\n"
        f"Welch t-test on total variance: t={stats_results['ttest_stat']:.3f}, "
        f"p={'<0.001' if t_p < 0.001 else f'{t_p:.4f}'} {t_sig}\n"
        f"PC1={pc1_var:.1f}%, PC2={pc2_var:.1f}% variance explained\n"
        f"Ellipses: 1.5σ confidence regions. Genre filter: ambient/sleep/new-age/chill/piano/acoustic.\n"
        f"Caution: ghost group includes Kaggle featured collaborations with near-zero variance\n"
        f"(likely duplicate recordings, not necessarily fraud). Same-genre confound partially addressed."
    )
    fig.text(
        0.5, 0.01, caption,
        ha="center", va="bottom", fontsize=8, color="#aaaaaa",
        wrap=True,
    )

    fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    log.info("Saved → %s", OUT)


def main() -> None:
    all_df, ghost_df, org_df = load_data()

    log.info("Ghost ambient artists: %d | Organic ambient artists: %d",
             len(ghost_df), len(org_df))
    log.info("Ghost total_var: mean=%.4f std=%.4f",
             ghost_df["total_variance"].mean(), ghost_df["total_variance"].std())
    log.info("Organic total_var: mean=%.4f std=%.4f",
             org_df["total_variance"].mean(), org_df["total_variance"].std())

    stat_results = run_statistics(ghost_df, org_df)
    log.info("Levene p=%.4f | F-test p=%.4f | t-test p=%.4f",
             stat_results["levene_p"], stat_results["f_p"], stat_results["ttest_p"])
    log.info("Cohen's d: %s", {k: f"{v:.2f}" for k, v in stat_results["cohens_d"].items()})

    coords, pca, _ = run_pca(all_df)
    log.info("PCA variance explained: PC1=%.1f%% PC2=%.1f%%",
             pca.explained_variance_ratio_[0]*100,
             pca.explained_variance_ratio_[1]*100)

    make_figure(all_df, ghost_df, org_df, coords, pca, stat_results)

    # Print summary
    print("\n" + "=" * 60)
    print("FIGURE 1 v2 STATISTICAL SUMMARY")
    print("=" * 60)
    print(f"Ghost ambient  : N={len(ghost_df)}, mean σ²={ghost_df['total_variance'].mean():.4f}")
    print(f"Organic ambient: N={len(org_df)}, mean σ²={org_df['total_variance'].mean():.4f}")
    print(f"Levene W={stat_results['levene_stat']:.3f}, p={stat_results['levene_p']:.4f}")
    print(f"F-test ratio={stat_results['f_stat']:.3f}, p={stat_results['f_p']:.4f}")
    print(f"t-test: t={stat_results['ttest_stat']:.3f}, p={stat_results['ttest_p']:.4f}")
    print("Cohen's d per feature:")
    for feat, d in stat_results["cohens_d"].items():
        print(f"  {feat}: d={d:.2f}")
    print(f"\nPC1={pca.explained_variance_ratio_[0]*100:.1f}%, PC2={pca.explained_variance_ratio_[1]*100:.1f}%")
    print(f"Output: {OUT}")


if __name__ == "__main__":
    main()
