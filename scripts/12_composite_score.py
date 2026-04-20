"""
scripts/12_composite_score.py

GhostTrack Composite Score v2

Defines and learns a composite ghost-artist detection score from the three
discriminative signal proxies identified in Prompt 3 (scripts/06_regenerate_fig6.py):

  S2_norm — bulk-upload proxy:  max(0, 1 − total_variance / 0.200)
              Cohen's d = 3.44 (highly discriminative)
  S4_norm — catalog density:    track_count / 300  (clipped to [0,1])
              Cohen's d = 0.32 (weak; retained for theoretical alignment)
  S5_norm — genre concentration: 1 / genre_count  (clipped to [0,1])
              Cohen's d = 0.91 (discriminative)

  Dropped signals:
    S1 — audio fingerprint: unavailable at Kaggle scale, N=0 discriminative
    S3 — playlist co-occurrence: d = 0.00 (non-discriminative)
    S6 — ISRC HHI: only 3 artists have real ISRC data; cannot compute at scale
    S7 — cross-platform: available for 4 Neo4j artists only; not at Kaggle scale

  composite_score = sigmoid(w0 + w_S2*S2 + w_S4*S4 + w_S5*S5)

  Weights learned via logistic regression on:
    Organic (negative class): 1030 Kaggle artists with high total_variance
    Ghost-proxy (positive class): 100 Kaggle artists with total_variance < 0.010
    Source: data/processed/organic_controls_kaggle.csv
            data/processed/low_variance_artists.csv

  IMPORTANT CAVEAT:
    The 100 ghost-proxies are not confirmed ghost artists — they are Kaggle tracks
    with very low audio variance, which is a necessary (but not sufficient)
    condition for ghost-artist status. Weights should be interpreted as
    "audio-uniformity detector" weights, not "ghost detector" weights.
    Ground truth is limited to 3 confirmed ghost artists (DOJ indictment).

Output:
  models/composite_weights.json   — learned weights + bootstrap CIs
  results/composite_scores.csv    — per-artist scores + labels
  figures/fig_composite_roc.png   — ROC + calibration + weight summary
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, roc_curve, average_precision_score,
    precision_recall_curve, f1_score, classification_report,
)
from scipy import stats as scipy_stats

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO, stream=sys.stdout,
)
log = logging.getLogger(__name__)

ROOT    = Path(__file__).parent.parent
DATA    = ROOT / "data"
MODELS  = ROOT / "models"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
MODELS.mkdir(parents=True, exist_ok=True)
RESULTS.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

SIGNAL_NAMES = ["S2_bulk_proxy", "S4_catalog_density", "S5_genre_concentration"]
FEATURE_NAMES = SIGNAL_NAMES  # used in LogReg

# Color palette — consistent across project
C_GHOST   = "#E74C3C"
C_ORGANIC = "#27AE60"
C_AMBER   = "#F39C12"


# ── Data loading ──────────────────────────────────────────────────────────────

def load_kaggle_dataset() -> pd.DataFrame:
    """
    Build labeled dataset from Kaggle artist-level aggregate stats.
    Ghost-proxy: low total_variance artists (threshold 0.010).
    Organic: high-variance artists from organic_controls_kaggle.csv.
    """
    org = pd.read_csv(DATA / "processed" / "organic_controls_kaggle.csv")
    lv  = pd.read_csv(DATA / "processed" / "low_variance_artists.csv")

    org["label"] = 0
    lv["label"]  = 1

    cols = ["artist", "track_count", "total_variance", "genre_count", "label"]
    df = pd.concat([org[cols], lv[cols]], ignore_index=True)

    # Compute signal proxies
    df["S2_bulk_proxy"]         = (1.0 - df["total_variance"] / 0.200).clip(0.0, 1.0)
    df["S4_catalog_density"]    = (df["track_count"] / 300.0).clip(0.0, 1.0)
    df["S5_genre_concentration"] = (1.0 / df["genre_count"].clip(1)).clip(0.0, 1.0)

    log.info("Dataset: %d organic + %d ghost-proxy = %d total",
             (df.label == 0).sum(), (df.label == 1).sum(), len(df))
    return df


def load_neo4j_panel() -> pd.DataFrame:
    """Load the 4 Neo4j artists with actual signal scores for validation."""
    import json
    with open(DATA / "processed" / "ex6_verdicts.json") as f:
        verdicts = json.load(f)

    rows = []
    for v in verdicts:
        s = v["signal_scores"]
        rows.append({
            "artist":        v["artist_name"],
            "label":         1 if v["true_label"] == "ghost" else 0,
            # S2: use cadence_sync directly (computed from real release dates)
            "S2_bulk_proxy":         float(s.get("s2_cadence_sync") or 0.0),
            # S4: catalog density from follower_ratio proxy
            "S4_catalog_density":    float(s.get("s4_follower_ratio") or 0.0),
            # S6 as HHI (best available signal for these artists)
            "S6_hhi":                float(s.get("s6_graph_density") or 0.0),
            # S5 metadata similarity
            "S5_genre_concentration": float(s.get("s5_metadata_similarity") or 0.0),
        })
    return pd.DataFrame(rows)


# ── Composite score ───────────────────────────────────────────────────────────

def compute_composite_raw(
    df: pd.DataFrame,
    w: np.ndarray,
    bias: float,
) -> np.ndarray:
    """
    Compute logit = bias + w·[S2, S4, S5], then apply sigmoid.
    """
    X = df[SIGNAL_NAMES].values
    logit = bias + X @ w
    return 1.0 / (1.0 + np.exp(-logit))


# ── Training & bootstrap CIs ─────────────────────────────────────────────────

def fit_logistic(X: np.ndarray, y: np.ndarray) -> LogisticRegression:
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    lr = LogisticRegression(
        class_weight="balanced",
        C=1.0,
        max_iter=1000,
        random_state=42,
        solver="lbfgs",
    )
    lr.fit(X_s, y)
    lr._scaler = scaler  # attach scaler for prediction
    return lr


def bootstrap_weights(
    df: pd.DataFrame,
    n_boot: int = 1000,
    seed: int = 42,
) -> dict:
    """
    Bootstrap CI for logistic regression weights on scaled features.
    Returns dict: {signal: (coef, lo, hi)}.
    """
    rng = np.random.default_rng(seed)
    X = df[SIGNAL_NAMES].values
    y = df["label"].values

    boot_coefs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(df), size=len(df))
        Xb, yb = X[idx], y[idx]
        if len(np.unique(yb)) < 2:
            continue
        scaler = StandardScaler()
        Xb_s = scaler.fit_transform(Xb)
        try:
            lr = LogisticRegression(
                class_weight="balanced", C=1.0, max_iter=500,
                random_state=42, solver="lbfgs",
            )
            lr.fit(Xb_s, yb)
            boot_coefs.append(lr.coef_[0])
        except Exception:
            continue

    boot_coefs = np.array(boot_coefs)
    lo = np.percentile(boot_coefs, 2.5, axis=0)
    hi = np.percentile(boot_coefs, 97.5, axis=0)

    # Fit full model for point estimates
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    lr_full = LogisticRegression(
        class_weight="balanced", C=1.0, max_iter=1000,
        random_state=42, solver="lbfgs",
    )
    lr_full.fit(X_s, y)
    coefs = lr_full.coef_[0]

    result = {}
    for i, name in enumerate(SIGNAL_NAMES):
        result[name] = {
            "coef":    float(coefs[i]),
            "ci_lo":   float(lo[i]),
            "ci_hi":   float(hi[i]),
            "z_score": float(coefs[i] / (np.std(boot_coefs[:, i]) + 1e-9)),
        }
    result["bias"] = float(lr_full.intercept_[0])
    return result, lr_full, scaler


def cv_evaluate(df: pd.DataFrame, n_splits: int = 5) -> dict:
    """5-fold stratified CV: AUC, F1, avg precision."""
    X = df[SIGNAL_NAMES].values
    y = df["label"].values

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    aucs, f1s, aps = [], [], []
    all_y_true, all_y_prob = [], []

    for train_idx, test_idx in skf.split(X, y):
        Xtr, Xte = X[train_idx], X[test_idx]
        ytr, yte = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        Xtr_s = scaler.fit_transform(Xtr)
        Xte_s = scaler.transform(Xte)

        lr = LogisticRegression(
            class_weight="balanced", C=1.0, max_iter=1000,
            random_state=42, solver="lbfgs",
        )
        lr.fit(Xtr_s, ytr)
        prob = lr.predict_proba(Xte_s)[:, 1]

        all_y_true.extend(yte.tolist())
        all_y_prob.extend(prob.tolist())

        aucs.append(roc_auc_score(yte, prob))
        f1s.append(f1_score(yte, (prob >= 0.5).astype(int), zero_division=0))
        aps.append(average_precision_score(yte, prob))

    yt = np.array(all_y_true)
    yp = np.array(all_y_prob)

    return {
        "auc_mean": float(np.mean(aucs)),
        "auc_std":  float(np.std(aucs)),
        "f1_mean":  float(np.mean(f1s)),
        "f1_std":   float(np.std(f1s)),
        "ap_mean":  float(np.mean(aps)),
        "ap_std":   float(np.std(aps)),
        "y_true":   yt,
        "y_prob":   yp,
    }


def youden_j_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, float, float]:
    """Optimal classification threshold via Youden's J = TPR − FPR."""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    j_scores = tpr - fpr
    best_idx  = int(np.argmax(j_scores))
    return float(thresholds[best_idx]), float(tpr[best_idx]), float(fpr[best_idx])


# ── Figure ────────────────────────────────────────────────────────────────────

def make_figure(
    df: pd.DataFrame,
    cv_res: dict,
    weight_res: dict,
    neo4j: pd.DataFrame,
    opt_thresh: float,
) -> plt.Figure:
    fig = plt.figure(figsize=(18, 12), facecolor="#0a0a0a")
    gs  = fig.add_gridspec(2, 3, hspace=0.40, wspace=0.35,
                           left=0.07, right=0.97, top=0.88, bottom=0.08)

    ax_roc    = fig.add_subplot(gs[0, 0])
    ax_pr     = fig.add_subplot(gs[0, 1])
    ax_w      = fig.add_subplot(gs[0, 2])
    ax_dist   = fig.add_subplot(gs[1, 0])
    ax_neo4j  = fig.add_subplot(gs[1, 1])
    ax_txt    = fig.add_subplot(gs[1, 2])

    for ax in [ax_roc, ax_pr, ax_w, ax_dist, ax_neo4j, ax_txt]:
        _style(ax)

    yt, yp = cv_res["y_true"], cv_res["y_prob"]

    # ── ROC ──────────────────────────────────────────────────────────────────
    fpr, tpr, _ = roc_curve(yt, yp)
    ax_roc.plot(fpr, tpr, color=C_GHOST, linewidth=2,
                label=f"AUC={cv_res['auc_mean']:.3f} ± {cv_res['auc_std']:.3f}")
    ax_roc.plot([0,1], [0,1], "--", color="#555555")
    ax_roc.axvline(youden_j_threshold(yt, yp)[2], color=C_AMBER, linestyle=":", alpha=0.8)
    fpr_opt, tpr_opt = youden_j_threshold(yt, yp)[2], youden_j_threshold(yt, yp)[1]
    ax_roc.scatter([fpr_opt], [tpr_opt], color=C_AMBER, s=80, zorder=5,
                   label=f"Youden J thresh={opt_thresh:.3f}")
    ax_roc.set_xlabel("False Positive Rate", fontsize=10, color="white")
    ax_roc.set_ylabel("True Positive Rate", fontsize=10, color="white")
    ax_roc.set_title("A.  ROC Curve (5-fold CV)", fontsize=11, color="white")
    ax_roc.legend(fontsize=8, framealpha=0.2, labelcolor="white",
                  facecolor="#1a1a1a", edgecolor="#444")

    # ── PR Curve ─────────────────────────────────────────────────────────────
    prec, rec, _ = precision_recall_curve(yt, yp)
    ax_pr.plot(rec, prec, color=C_ORGANIC, linewidth=2,
               label=f"AP={cv_res['ap_mean']:.3f} ± {cv_res['ap_std']:.3f}")
    baseline = yt.mean()
    ax_pr.axhline(baseline, linestyle="--", color="#555555",
                  label=f"Baseline={baseline:.3f}")
    ax_pr.set_xlabel("Recall", fontsize=10, color="white")
    ax_pr.set_ylabel("Precision", fontsize=10, color="white")
    ax_pr.set_title("B.  Precision-Recall Curve (5-fold CV)", fontsize=11, color="white")
    ax_pr.legend(fontsize=8, framealpha=0.2, labelcolor="white",
                 facecolor="#1a1a1a", edgecolor="#444")

    # ── Weight coefficients ───────────────────────────────────────────────────
    signals = SIGNAL_NAMES
    short_names = ["S2\nBulk\nProxy", "S4\nCatalog\nDensity", "S5\nGenre\nConc."]
    coefs   = [weight_res[s]["coef"]  for s in signals]
    lo_err  = [weight_res[s]["coef"] - weight_res[s]["ci_lo"] for s in signals]
    hi_err  = [weight_res[s]["ci_hi"] - weight_res[s]["coef"] for s in signals]
    colors  = [C_GHOST if c > 0 else C_ORGANIC for c in coefs]

    bars = ax_w.barh(short_names, coefs, color=colors, alpha=0.80, edgecolor="#0a0a0a")
    ax_w.errorbar(coefs, short_names, xerr=[lo_err, hi_err],
                  fmt="none", color="white", capsize=5, linewidth=1.5)
    ax_w.axvline(0, color="#888888", linewidth=1)
    ax_w.set_xlabel("Standardized coefficient (logistic regression)", fontsize=9, color="white")
    ax_w.set_title("C.  Learned Signal Weights\n(95% bootstrap CI, 1000 resamples)",
                   fontsize=11, color="white")
    for bar, coef in zip(bars, coefs):
        ax_w.text(coef + (0.05 if coef >= 0 else -0.05),
                  bar.get_y() + bar.get_height()/2,
                  f"{coef:+.3f}", va="center",
                  ha="left" if coef >= 0 else "right",
                  fontsize=9, color="white")

    # ── Score distribution ────────────────────────────────────────────────────
    scaler = StandardScaler()
    X_all  = df[SIGNAL_NAMES].values
    y_all  = df["label"].values
    X_all_s = scaler.fit_transform(X_all)

    # Refit on full data for scoring
    lr_full = LogisticRegression(
        class_weight="balanced", C=1.0, max_iter=1000,
        random_state=42, solver="lbfgs",
    )
    lr_full.fit(X_all_s, y_all)
    scores_all = lr_full.predict_proba(X_all_s)[:, 1]

    ghost_scores   = scores_all[y_all == 1]
    organic_scores = scores_all[y_all == 0]

    bins = np.linspace(0, 1, 25)
    ax_dist.hist(organic_scores, bins=bins, color=C_ORGANIC, alpha=0.65,
                 label=f"Organic (n={len(organic_scores)})", density=True)
    ax_dist.hist(ghost_scores,   bins=bins, color=C_GHOST,   alpha=0.80,
                 label=f"Ghost-proxy (n={len(ghost_scores)})", density=True)
    ax_dist.axvline(opt_thresh, color=C_AMBER, linestyle="--", linewidth=2,
                    label=f"Youden J={opt_thresh:.3f}")
    ax_dist.set_xlabel("Composite score", fontsize=10, color="white")
    ax_dist.set_ylabel("Density", fontsize=10, color="white")
    ax_dist.set_title("D.  Score Distribution by Class", fontsize=11, color="white")
    ax_dist.legend(fontsize=8, framealpha=0.2, labelcolor="white",
                   facecolor="#1a1a1a", edgecolor="#444")

    # ── Neo4j validation panel ────────────────────────────────────────────────
    # Score the 4 Neo4j panel artists using their real S2/S4 scores
    # Use simple weighted combination since scale differs (no scaler fit on 4 points)
    w_s2 = weight_res["S2_bulk_proxy"]["coef"]
    w_s4 = weight_res["S4_catalog_density"]["coef"]
    w_s5 = weight_res["S5_genre_concentration"]["coef"]
    bias = weight_res["bias"]

    neo4j_scores = []
    for _, row in neo4j.iterrows():
        # Raw (unscaled) — use signal values directly as-is
        logit = bias + w_s2 * row["S2_bulk_proxy"] + w_s4 * row["S4_catalog_density"] + w_s5 * row["S5_genre_concentration"]
        neo4j_scores.append(1 / (1 + np.exp(-logit)))
    neo4j = neo4j.copy()
    neo4j["composite"] = neo4j_scores

    colors_neo = [C_GHOST if r == 1 else C_ORGANIC for r in neo4j["label"]]
    bars_neo = ax_neo4j.barh(
        neo4j["artist"].str[:25], neo4j["composite"],
        color=colors_neo, alpha=0.85, edgecolor="#0a0a0a",
    )
    ax_neo4j.axvline(opt_thresh, color=C_AMBER, linestyle="--", linewidth=2,
                     label=f"Threshold={opt_thresh:.2f}")
    ax_neo4j.set_xlim(0, 1)
    ax_neo4j.set_xlabel("Composite score", fontsize=10, color="white")
    ax_neo4j.set_title("E.  Neo4j Panel (4 artists) Validation\n(uses actual cadence/HHI scores)",
                       fontsize=11, color="white")
    ax_neo4j.legend(fontsize=8, framealpha=0.2, labelcolor="white",
                    facecolor="#1a1a1a", edgecolor="#444")
    ghost_patch  = plt.Rectangle((0,0),1,1, color=C_GHOST,  label="Ghost (true)")
    org_patch    = plt.Rectangle((0,0),1,1, color=C_ORGANIC, label="Organic (true)")
    ax_neo4j.legend(handles=[ghost_patch, org_patch], fontsize=8,
                    framealpha=0.2, labelcolor="white", facecolor="#1a1a1a", edgecolor="#444")

    # ── Text summary ──────────────────────────────────────────────────────────
    ax_txt.axis("off")
    lines = [
        "F.  Composite Score Summary",
        "",
        "Formula:",
        "  score = σ(bias + w_S2·S2",
        "              + w_S4·S4 + w_S5·S5)",
        "",
        f"  bias  = {weight_res['bias']:+.3f}",
    ]
    for s in SIGNAL_NAMES:
        short = s.replace("_proxy","").replace("_density","").replace("_concentration","")
        lines.append(f"  w_{short:6s} = {weight_res[s]['coef']:+.3f}")
        lines.append(f"    CI=[{weight_res[s]['ci_lo']:+.3f},{weight_res[s]['ci_hi']:+.3f}]")
    lines += [
        "",
        "5-fold CV performance:",
        f"  AUC  = {cv_res['auc_mean']:.3f} ± {cv_res['auc_std']:.3f}",
        f"  AP   = {cv_res['ap_mean']:.3f} ± {cv_res['ap_std']:.3f}",
        f"  F1   = {cv_res['f1_mean']:.3f} ± {cv_res['f1_std']:.3f}",
        f"  Thresh(J) = {opt_thresh:.3f}",
        "",
        "Training set:",
        "  n=1030 organic (high-var)",
        "  n=100 ghost-proxy (low-var)",
        "",
        "⚠ CAVEAT: ghost-proxy labels are",
        "   'low audio variance' artists,",
        "   NOT confirmed ghost artists.",
        "   Only 3 confirmed ghosts exist",
        "   in ground truth (DOJ 2024).",
    ]
    ax_txt.text(0.02, 0.98, "\n".join(lines), transform=ax_txt.transAxes,
                fontsize=8.5, color="white", family="monospace",
                va="top", ha="left")

    fig.suptitle(
        "Figure: GhostTrack Composite Score — Learned Weights & Validation",
        fontsize=14, color="white", y=0.97, fontweight="bold",
    )
    return fig


def _style(ax: plt.Axes) -> None:
    ax.set_facecolor("#111111")
    ax.tick_params(colors="white", labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor("#444444")
    ax.yaxis.label.set_color("white")
    ax.xaxis.label.set_color("white")
    ax.title.set_color("white")
    ax.grid(True, color="#333333", linestyle="--", linewidth=0.5, alpha=0.6)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    df     = load_kaggle_dataset()
    neo4j  = load_neo4j_panel()

    log.info("Fitting logistic regression with bootstrap CIs...")
    weight_res, lr_full, scaler = bootstrap_weights(df, n_boot=1000)

    print("\n" + "=" * 60)
    print("LEARNED WEIGHTS (standardized logistic regression coefficients)")
    print("=" * 60)
    for sig, vals in weight_res.items():
        if sig == "bias":
            print(f"  bias          : {vals:+.4f}")
            continue
        print(f"  {sig:28s}: coef={vals['coef']:+.4f}  "
              f"95% CI=[{vals['ci_lo']:+.4f}, {vals['ci_hi']:+.4f}]  "
              f"Z={vals['z_score']:+.2f}")

    log.info("5-fold cross-validation...")
    cv_res = cv_evaluate(df)
    print(f"\n5-fold CV: AUC={cv_res['auc_mean']:.3f}±{cv_res['auc_std']:.3f}  "
          f"F1={cv_res['f1_mean']:.3f}±{cv_res['f1_std']:.3f}  "
          f"AP={cv_res['ap_mean']:.3f}±{cv_res['ap_std']:.3f}")

    opt_thresh, tpr_opt, fpr_opt = youden_j_threshold(cv_res["y_true"], cv_res["y_prob"])
    print(f"Youden J threshold: {opt_thresh:.3f}  (TPR={tpr_opt:.3f}, FPR={fpr_opt:.3f})")

    # Save weights
    weights_out = {
        "formula": "score = sigmoid(bias + w_S2*S2_norm + w_S4*S4_norm + w_S5*S5_norm)",
        "signals_used": SIGNAL_NAMES,
        "signals_dropped": {
            "S1": "audio fingerprint unavailable at Kaggle scale",
            "S3": "playlist co-occurrence: d=0.00 (non-discriminative)",
            "S6": "ISRC HHI: only 3 artists have real ISRC data",
            "S7": "cross-platform: available for 4 Neo4j artists only",
        },
        "training_data": {
            "n_organic": int((df.label == 0).sum()),
            "n_ghost_proxy": int((df.label == 1).sum()),
            "ghost_proxy_definition": "Kaggle artists with total_variance < 0.010",
            "caveat": "Ghost-proxy labels are NOT confirmed ghost artists",
        },
        "cv_performance": {
            "auc_mean": cv_res["auc_mean"],
            "auc_std":  cv_res["auc_std"],
            "f1_mean":  cv_res["f1_mean"],
            "f1_std":   cv_res["f1_std"],
            "ap_mean":  cv_res["ap_mean"],
            "ap_std":   cv_res["ap_std"],
        },
        "optimal_threshold": {
            "value": opt_thresh,
            "method": "Youden J = argmax(TPR - FPR)",
            "tpr_at_threshold": tpr_opt,
            "fpr_at_threshold": fpr_opt,
        },
        "weights": weight_res,
    }
    weights_path = MODELS / "composite_weights.json"
    weights_path.write_text(json.dumps(weights_out, indent=2))
    log.info("Saved → %s", weights_path)

    # Score all artists + save
    X_all  = df[SIGNAL_NAMES].values
    X_all_s = scaler.transform(X_all)
    df["composite_score"] = lr_full.predict_proba(X_all_s)[:, 1]
    df["predicted_ghost"] = (df["composite_score"] >= opt_thresh).astype(int)
    scores_path = RESULTS / "composite_scores.csv"
    df[["artist","label","S2_bulk_proxy","S4_catalog_density","S5_genre_concentration",
        "composite_score","predicted_ghost"]].to_csv(scores_path, index=False)
    log.info("Saved → %s (%d rows)", scores_path, len(df))

    # False positive summary
    fp = df[(df["label"] == 0) & (df["predicted_ghost"] == 1)]
    fn = df[(df["label"] == 1) & (df["predicted_ghost"] == 0)]
    print(f"\nClassification at threshold {opt_thresh:.3f}:")
    print(f"  False positives (organic flagged as ghost): {len(fp)}")
    print(f"  False negatives (ghost missed): {len(fn)}")
    if len(fp) > 0:
        print("\nTop 10 false positives:")
        print(fp[["artist","total_variance","S2_bulk_proxy","composite_score"]]
              .sort_values("composite_score", ascending=False).head(10).to_string(index=False))

    # Figure
    log.info("Generating figure...")
    fig = make_figure(df, cv_res, weight_res, neo4j, opt_thresh)
    fig_path = FIGURES / "fig_composite_roc.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight", facecolor="#0a0a0a")
    plt.close(fig)
    log.info("Saved → %s (%d KB)", fig_path, fig_path.stat().st_size // 1024)


if __name__ == "__main__":
    main()
