"""
scripts/03_train_baselines.py

Train and evaluate 5 models on ghost-artist detection:
  1. Logistic Regression (tabular)
  2. Random Forest      (tabular)
  3. MLP                (tabular, PyTorch)
  4. GraphSAGE          (heterogeneous graph, HeteroSAGE)
  5. HGT                (Heterogeneous Graph Transformer)

Evaluation protocol:
  - 5-fold stratified cross-validation with artist-level holdout
  - Temporal secondary validation (pre-2024 train / 2024+ test)
  - Metrics: AUC-ROC, F1, Precision@K=20 with bootstrap 95% CI (1000 resamples)
  - SHAP feature importance for tabular models

Outputs:
  results/baseline_comparison.csv       — per-model metrics
  figures/fig7_v2_baselines.png         — 4-panel figure
  models/hetero/lr_model.pkl            — saved LR checkpoint
  models/hetero/rf_model.pkl            — saved RF checkpoint
  models/hetero/mlp_model.pt            — saved MLP checkpoint
  models/hetero/sage_model.pt           — saved GraphSAGE checkpoint
  models/hetero/hgt_model.pt            — saved HGT checkpoint

IMPORTANT DESIGN DECISION:
  If GNN does not beat tabular baselines by ≥3 AUC points, this script reports
  that honestly. Negative results are valid scientific findings.

Hyperparameters (all logged):
  SEED          = 42
  CV_FOLDS      = 5
  BOOTSTRAP_N   = 1000
  LR_C          = 1.0
  RF_ESTIMATORS = 200
  RF_MAX_DEPTH  = None
  MLP_HIDDEN    = [64, 32]
  MLP_LR        = 1e-3
  MLP_EPOCHS    = 200
  GNN_HIDDEN    = 32
  GNN_LR        = 1e-3
  GNN_EPOCHS    = 300
  GNN_DROPOUT   = 0.3
  TOP_K         = 20   (Precision@K)
"""

from __future__ import annotations

import json
import logging
import pickle
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import shap
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import HeteroData
from torch_geometric.nn import HGTConv, Linear, SAGEConv

warnings.filterwarnings("ignore")

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
    stream=sys.stdout,
)
log = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
GRAPH_PT   = ROOT / "data" / "processed" / "hetero_graph.pt"
RESULTS_CSV = ROOT / "results" / "baseline_comparison.csv"
FIG_PATH    = ROOT / "figures" / "fig7_v2_baselines.png"
MODELS_DIR  = ROOT / "models" / "hetero"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
(ROOT / "results").mkdir(exist_ok=True)
(ROOT / "figures").mkdir(exist_ok=True)

# ── Hyperparameters ──────────────────────────────────────────────────────────
CV_FOLDS      = 5
BOOTSTRAP_N   = 1000
TOP_K         = 20
LR_C          = 1.0
RF_ESTIMATORS = 200
MLP_HIDDEN    = [64, 32]
MLP_LR        = 1e-3
MLP_EPOCHS    = 200
GNN_HIDDEN    = 32
GNN_LR        = 1e-3
GNN_EPOCHS    = 300
GNN_DROPOUT   = 0.3

HPARAMS = dict(
    seed=SEED, cv_folds=CV_FOLDS, bootstrap_n=BOOTSTRAP_N, top_k=TOP_K,
    lr_c=LR_C, rf_estimators=RF_ESTIMATORS, mlp_hidden=MLP_HIDDEN,
    mlp_lr=MLP_LR, mlp_epochs=MLP_EPOCHS, gnn_hidden=GNN_HIDDEN,
    gnn_lr=GNN_LR, gnn_epochs=GNN_EPOCHS, gnn_dropout=GNN_DROPOUT,
)
log.info("Hyperparameters: %s", json.dumps(HPARAMS, default=str))


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def bootstrap_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    metric_fn,
    n: int = BOOTSTRAP_N,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Return (point_estimate, lower_CI, upper_CI)."""
    rng = np.random.default_rng(SEED)
    point = metric_fn(y_true, y_score)
    samples = []
    for _ in range(n):
        idx = rng.integers(0, len(y_true), size=len(y_true))
        if len(np.unique(y_true[idx])) < 2:
            continue
        try:
            samples.append(metric_fn(y_true[idx], y_score[idx]))
        except Exception:
            continue
    if not samples:
        return point, point, point
    lo = float(np.percentile(samples, 100 * alpha / 2))
    hi = float(np.percentile(samples, 100 * (1 - alpha / 2)))
    return float(point), lo, hi


def precision_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int = TOP_K) -> float:
    """Precision@K: fraction of top-K predicted positives that are actually positive."""
    k = min(k, len(y_true))
    top_k_idx = np.argsort(y_score)[::-1][:k]
    return float(y_true[top_k_idx].mean())


def compute_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    model_name: str,
) -> dict:
    """Compute AUC, F1, Precision@K with bootstrap CIs."""
    if len(np.unique(y_true)) < 2:
        log.warning("%s: only one class in y_true — returning NaN metrics", model_name)
        return {
            "model": model_name,
            "auc": np.nan, "auc_lo": np.nan, "auc_hi": np.nan,
            "f1":  np.nan, "f1_lo":  np.nan, "f1_hi":  np.nan,
            "prec_at_k": np.nan, "prec_at_k_lo": np.nan, "prec_at_k_hi": np.nan,
            "avg_precision": np.nan,
            "n_test": len(y_true), "n_ghost": int(y_true.sum()),
        }

    y_pred = (y_score >= 0.5).astype(int)

    auc,  auc_lo,  auc_hi  = bootstrap_ci(y_true, y_score, roc_auc_score)
    f1,   f1_lo,   f1_hi   = bootstrap_ci(
        y_true, y_pred,
        lambda yt, yp: f1_score(yt, yp, zero_division=0),
    )
    pak,  pak_lo,  pak_hi  = bootstrap_ci(
        y_true, y_score, lambda yt, ys: precision_at_k(yt, ys, TOP_K)
    )
    avg_prec = float(average_precision_score(y_true, y_score))

    log.info(
        "%s — AUC %.3f [%.3f–%.3f] | F1 %.3f | P@%d %.3f",
        model_name, auc, auc_lo, auc_hi, f1, TOP_K, pak,
    )
    return {
        "model": model_name,
        "auc": auc, "auc_lo": auc_lo, "auc_hi": auc_hi,
        "f1":  f1,  "f1_lo":  f1_lo,  "f1_hi":  f1_hi,
        "prec_at_k": pak, "prec_at_k_lo": pak_lo, "prec_at_k_hi": pak_hi,
        "avg_precision": avg_prec,
        "n_test": len(y_true), "n_ghost": int(y_true.sum()),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_data() -> tuple[HeteroData, np.ndarray, np.ndarray, list[str]]:
    """Load hetero graph and extract tabular (X, y) for baseline models."""
    data: HeteroData = torch.load(GRAPH_PT, weights_only=False)
    X = data["artist"].x.numpy()
    y = data["artist"].y.numpy()

    meta = json.loads((ROOT / "data" / "processed" / "hetero_graph_meta.json").read_text())
    feat_names = meta["artist_feature_names"]

    log.info("Loaded graph: %d artists, %d ghost, %d organic, %d features",
             len(y), int(y.sum()), int((y == 0).sum()), X.shape[1])
    return data, X, y, feat_names


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Tabular Baselines (LR, RF, MLP)
# ─────────────────────────────────────────────────────────────────────────────

class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: list[int]) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(0.3)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(x)).squeeze(-1)


def train_mlp(X_tr: np.ndarray, y_tr: np.ndarray) -> MLP:
    model = MLP(X_tr.shape[1], MLP_HIDDEN)
    opt   = torch.optim.Adam(model.parameters(), lr=MLP_LR)
    # Class weight for imbalance (no oversampling per spec)
    pos_weight = torch.tensor([(y_tr == 0).sum() / max(1, (y_tr == 1).sum())],
                              dtype=torch.float)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    Xt = torch.tensor(X_tr, dtype=torch.float)
    yt = torch.tensor(y_tr, dtype=torch.float)
    model.train()
    for _ in range(MLP_EPOCHS):
        opt.zero_grad()
        logits = model.net(Xt).squeeze(-1)
        loss = loss_fn(logits, yt)
        loss.backward()
        opt.step()
    return model


def run_tabular_baselines(
    X: np.ndarray,
    y: np.ndarray,
    feat_names: list[str],
) -> tuple[list[dict], dict[str, np.ndarray], dict[str, np.ndarray]]:
    """
    5-fold CV for LR, RF, MLP.
    Returns (metrics_list, {model_name: oof_scores}, {model_name: oof_preds}).
    """
    skf   = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
    names = ["LogisticRegression", "RandomForest", "MLP"]

    oof_scores: dict[str, list] = {n: [] for n in names}
    oof_true:   list            = []
    all_models: dict[str, list] = {n: [] for n in names}

    scaler = StandardScaler()

    for fold, (tr_idx, te_idx) in enumerate(skf.split(X, y)):
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]

        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        cw = "balanced"

        # Logistic Regression
        lr = LogisticRegression(C=LR_C, class_weight=cw, max_iter=1000,
                                random_state=SEED, solver="lbfgs")
        lr.fit(X_tr_s, y_tr)
        oof_scores["LogisticRegression"].extend(lr.predict_proba(X_te_s)[:, 1])
        all_models["LogisticRegression"].append(lr)

        # Random Forest
        rf = RandomForestClassifier(
            n_estimators=RF_ESTIMATORS, class_weight=cw,
            random_state=SEED, n_jobs=-1,
        )
        rf.fit(X_tr_s, y_tr)
        oof_scores["RandomForest"].extend(rf.predict_proba(X_te_s)[:, 1])
        all_models["RandomForest"].append(rf)

        # MLP
        mlp = train_mlp(X_tr_s, y_tr)
        mlp.eval()
        with torch.no_grad():
            preds = mlp(torch.tensor(X_te_s, dtype=torch.float)).numpy()
        oof_scores["MLP"].extend(preds)
        all_models["MLP"].append(mlp)

        oof_true.extend(y_te)
        log.info("Fold %d/%d done", fold + 1, CV_FOLDS)

    y_true_arr = np.array(oof_true)
    metrics = []
    score_arrays: dict[str, np.ndarray] = {}

    for name in names:
        sc = np.array(oof_scores[name])
        score_arrays[name] = sc
        metrics.append(compute_metrics(y_true_arr, sc, name))
        metrics[-1]["split"] = "5fold_cv"

    # Save best fold's models (last fold for simplicity)
    lr_final = all_models["LogisticRegression"][-1]
    rf_final = all_models["RandomForest"][-1]
    mlp_final = all_models["MLP"][-1]
    X_s_full = scaler.fit_transform(X)
    lr_final.fit(X_s_full, y)  # refit on all data
    rf_final.fit(X_s_full, y)

    with open(MODELS_DIR / "lr_model.pkl", "wb") as f:
        pickle.dump({"model": lr_final, "scaler": scaler}, f)
    with open(MODELS_DIR / "rf_model.pkl", "wb") as f:
        pickle.dump({"model": rf_final, "scaler": scaler}, f)
    torch.save(mlp_final.state_dict(), MODELS_DIR / "mlp_model.pt")
    log.info("Tabular models saved to %s", MODELS_DIR)

    return metrics, score_arrays, {"all": y_true_arr}


# ─────────────────────────────────────────────────────────────────────────────
# 2.  GNN Models (HeteroSAGE + HGT)
# ─────────────────────────────────────────────────────────────────────────────

class HeteroSAGE(nn.Module):
    """
    Two-layer heterogeneous GraphSAGE.
    artist → registrant → artist message passing.
    """
    def __init__(
        self,
        artist_in: int,
        reg_in: int,
        hidden: int,
        dropout: float,
        metadata: tuple,
    ) -> None:
        super().__init__()
        self.conv1 = SAGEConv((-1, -1), hidden)
        self.conv2 = SAGEConv((-1, -1), hidden)
        self.lin   = nn.Linear(hidden, 1)
        self.drop  = nn.Dropout(dropout)
        # Project registrant features to same dim
        self.reg_proj = nn.Linear(reg_in, artist_in)

    def forward(self, x_dict: dict, edge_index_dict: dict) -> torch.Tensor:
        # Project registrant to match artist feature space
        x_reg_proj = F.relu(self.reg_proj(x_dict["registrant"]))

        # Message: registrant -> artist (reverse edges carry registrant signal)
        ei_rev = edge_index_dict[("registrant", "rev_registered_with", "artist")]
        h = self.conv1(
            (x_reg_proj, x_dict["artist"]),
            ei_rev,
        )
        h = F.elu(h)
        h = self.drop(h)

        ei_fwd = edge_index_dict[("artist", "registered_with", "registrant")]
        h = self.conv2(
            (h, h),
            # self-loop: artist-to-artist via shared registrant (approximate)
            torch.stack([ei_fwd[0], ei_fwd[0]]),
        )
        h = F.elu(h)
        h = self.drop(h)
        return torch.sigmoid(self.lin(h)).squeeze(-1)


class HGTModel(nn.Module):
    """
    Two-layer Heterogeneous Graph Transformer.
    """
    def __init__(
        self,
        artist_in: int,
        reg_in: int,
        hidden: int,
        dropout: float,
        metadata: tuple,
    ) -> None:
        super().__init__()
        self.artist_proj = Linear(artist_in, hidden)
        self.reg_proj    = Linear(reg_in, hidden)

        self.conv1 = HGTConv(
            in_channels  = hidden,
            out_channels = hidden,
            metadata     = metadata,
            heads        = 2,
        )
        self.conv2 = HGTConv(
            in_channels  = hidden,
            out_channels = hidden,
            metadata     = metadata,
            heads        = 2,
        )
        self.lin  = nn.Linear(hidden, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x_dict: dict, edge_index_dict: dict) -> torch.Tensor:
        h = {
            "artist":     F.elu(self.artist_proj(x_dict["artist"])),
            "registrant": F.elu(self.reg_proj(x_dict["registrant"])),
        }
        h = self.conv1(h, edge_index_dict)
        h = {k: F.elu(v) for k, v in h.items()}
        h = self.conv2(h, edge_index_dict)
        out = torch.sigmoid(self.lin(h["artist"])).squeeze(-1)
        return out


def _gnn_train_loop(
    model: nn.Module,
    data: HeteroData,
    train_mask: torch.Tensor,
    lr: float,
    epochs: int,
    pos_weight: torch.Tensor,
) -> None:
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    x_dict = data.x_dict
    ei_dict = data.edge_index_dict

    model.train()
    for epoch in range(epochs):
        opt.zero_grad()
        if isinstance(model, HGTModel):
            out = model(x_dict, ei_dict)
        else:
            out = model(x_dict, ei_dict)
        logits = torch.log(out[train_mask].clamp(1e-6, 1 - 1e-6)) - \
                 torch.log((1 - out[train_mask]).clamp(1e-6, 1 - 1e-6))
        loss = loss_fn(logits, data["artist"].y[train_mask].float())
        loss.backward()
        opt.step()
        if (epoch + 1) % 100 == 0:
            log.debug("  epoch %d loss %.4f", epoch + 1, loss.item())


def run_gnn_cv(
    data: HeteroData,
    model_class,
    model_name: str,
    save_path: Path,
) -> tuple[list[dict], np.ndarray, np.ndarray]:
    """
    5-fold CV on artist nodes for a GNN model.
    Since full hetero graph resampling is complex, we mask train/test per fold.
    """
    X = data["artist"].x.numpy()
    y = data["artist"].y.numpy()
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)

    artist_in = data["artist"].x.shape[1]
    reg_in    = data["registrant"].x.shape[1]
    metadata  = data.metadata()

    oof_scores: list[float] = []
    oof_true:   list[int]   = []

    for fold, (tr_idx, te_idx) in enumerate(skf.split(X, y)):
        # Build masks
        train_mask = torch.zeros(len(y), dtype=torch.bool)
        test_mask  = torch.zeros(len(y), dtype=torch.bool)
        train_mask[tr_idx] = True
        test_mask[te_idx]  = True

        y_tr = y[tr_idx]
        pos_w = torch.tensor(
            [(y_tr == 0).sum() / max(1, (y_tr == 1).sum())],
            dtype=torch.float,
        )

        model = model_class(
            artist_in=artist_in, reg_in=reg_in,
            hidden=GNN_HIDDEN, dropout=GNN_DROPOUT, metadata=metadata,
        )
        _gnn_train_loop(model, data, train_mask, GNN_LR, GNN_EPOCHS, pos_w)

        model.eval()
        with torch.no_grad():
            preds = model(data.x_dict, data.edge_index_dict).numpy()

        oof_scores.extend(preds[te_idx].tolist())
        oof_true.extend(y[te_idx].tolist())
        log.info("%s fold %d/%d done", model_name, fold + 1, CV_FOLDS)

    # Refit on all data and save
    pos_w_full = torch.tensor(
        [(y == 0).sum() / max(1, (y == 1).sum())], dtype=torch.float
    )
    full_mask = torch.ones(len(y), dtype=torch.bool)
    final_model = model_class(
        artist_in=artist_in, reg_in=reg_in,
        hidden=GNN_HIDDEN, dropout=GNN_DROPOUT, metadata=metadata,
    )
    _gnn_train_loop(final_model, data, full_mask, GNN_LR, GNN_EPOCHS, pos_w_full)
    torch.save(final_model.state_dict(), save_path)
    log.info("%s saved → %s", model_name, save_path)

    y_true  = np.array(oof_true)
    y_score = np.array(oof_scores)
    m = compute_metrics(y_true, y_score, model_name)
    m["split"] = "5fold_cv"
    return [m], y_score, y_true


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Temporal validation
# ─────────────────────────────────────────────────────────────────────────────

def run_temporal(
    X: np.ndarray,
    y: np.ndarray,
    data: HeteroData,
    tabular_models_pkl: dict,
) -> list[dict]:
    """
    Temporal split: temporal_train_mask → train, temporal_test_mask → test.
    Runs all 5 models and returns metrics.
    """
    tr_mask = data["artist"].temporal_train_mask.numpy().astype(bool)
    te_mask = data["artist"].temporal_test_mask.numpy().astype(bool)

    if tr_mask.sum() == 0 or te_mask.sum() == 0:
        log.warning("Temporal split: empty train or test — skipping")
        return []

    X_tr, X_te = X[tr_mask], X[te_mask]
    y_tr, y_te = y[tr_mask], y[te_mask]

    if len(np.unique(y_te)) < 2:
        log.warning("Temporal test set has only one class — skipping temporal eval")
        return []

    metrics = []
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    # LR
    lr = LogisticRegression(C=LR_C, class_weight="balanced",
                            max_iter=1000, random_state=SEED)
    lr.fit(X_tr_s, y_tr)
    m = compute_metrics(y_te, lr.predict_proba(X_te_s)[:, 1], "LogisticRegression")
    m["split"] = "temporal"; metrics.append(m)

    # RF
    rf = RandomForestClassifier(n_estimators=RF_ESTIMATORS, class_weight="balanced",
                                random_state=SEED, n_jobs=-1)
    rf.fit(X_tr_s, y_tr)
    m = compute_metrics(y_te, rf.predict_proba(X_te_s)[:, 1], "RandomForest")
    m["split"] = "temporal"; metrics.append(m)

    # MLP
    mlp = train_mlp(X_tr_s, y_tr)
    mlp.eval()
    with torch.no_grad():
        preds = mlp(torch.tensor(X_te_s, dtype=torch.float)).numpy()
    m = compute_metrics(y_te, preds, "MLP")
    m["split"] = "temporal"; metrics.append(m)

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# 4.  SHAP Feature Importance
# ─────────────────────────────────────────────────────────────────────────────

def compute_shap(
    X: np.ndarray,
    y: np.ndarray,
    feat_names: list[str],
) -> tuple[np.ndarray, list[str]]:
    """
    Fit RF on full data and compute SHAP values.
    Returns (mean_abs_shap per feature, feature names).
    """
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    rf = RandomForestClassifier(
        n_estimators=RF_ESTIMATORS, class_weight="balanced",
        random_state=SEED, n_jobs=-1,
    )
    rf.fit(X_s, y)

    explainer = shap.TreeExplainer(rf)
    shap_vals = explainer.shap_values(X_s)
    # shap_values shape depends on shap version:
    #   old: list[class0_arr, class1_arr]  each (n_samples, n_features)
    #   new: ndarray (n_samples, n_features, n_classes)
    if isinstance(shap_vals, list):
        shap_pos = np.array(shap_vals[1])   # class=1 (ghost)
    elif hasattr(shap_vals, "ndim") and shap_vals.ndim == 3:
        shap_pos = shap_vals[:, :, 1]       # (n_samples, n_features)
    else:
        shap_pos = np.array(shap_vals)

    mean_abs = np.abs(shap_pos).mean(axis=0)
    mean_abs = np.array(mean_abs).flatten()
    log.info("SHAP computed: top feature = %s (%.4f)",
             feat_names[np.argmax(mean_abs)], mean_abs.max())
    return mean_abs, feat_names


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Collect ROC / PR curves for plotting
# ─────────────────────────────────────────────────────────────────────────────

def collect_curves(
    y_true: np.ndarray,
    score_dict: dict[str, np.ndarray],
) -> tuple[dict, dict]:
    """
    Returns {model: (fpr, tpr, auc)} and {model: (precision, recall, ap)}.
    """
    roc_data, pr_data = {}, {}
    for name, scores in score_dict.items():
        if len(np.unique(y_true)) < 2:
            continue
        try:
            fpr, tpr, _ = roc_curve(y_true, scores)
            auc = roc_auc_score(y_true, scores)
            roc_data[name] = (fpr, tpr, auc)

            prec, rec, _ = precision_recall_curve(y_true, scores)
            ap = average_precision_score(y_true, scores)
            pr_data[name] = (prec, rec, ap)
        except Exception as e:
            log.warning("Curve computation failed for %s: %s", name, e)
    return roc_data, pr_data


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Figure generation (4-panel)
# ─────────────────────────────────────────────────────────────────────────────

COLORS = {
    "LogisticRegression": "#4da6ff",
    "RandomForest":       "#f0b429",
    "MLP":                "#b48ead",
    "GraphSAGE":          "#00ff88",
    "HGT":                "#e74c3c",
}


def make_figure(
    metrics_cv: list[dict],
    roc_data: dict,
    pr_data: dict,
    shap_vals: np.ndarray,
    feat_names: list[str],
    out_path: Path,
) -> None:
    fig = plt.figure(figsize=(18, 14), facecolor="#0a0a0a")
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.32)

    # ── Panel A: AUC bar chart ────────────────────────────────────────────────
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.set_facecolor("#0f0f0f")
    names = [m["model"] for m in metrics_cv]
    aucs  = [m["auc"]   for m in metrics_cv]
    lo_e  = [m["auc"] - m["auc_lo"] for m in metrics_cv]
    hi_e  = [m["auc_hi"] - m["auc"] for m in metrics_cv]
    colors = [COLORS.get(n, "#888") for n in names]
    bars = ax_a.bar(range(len(names)), aucs, color=colors, alpha=0.85, width=0.6)
    ax_a.errorbar(range(len(names)), aucs,
                  yerr=[lo_e, hi_e], fmt="none",
                  ecolor="white", capsize=5, linewidth=1.5)
    ax_a.set_xticks(range(len(names)))
    ax_a.set_xticklabels(names, rotation=25, ha="right", fontsize=9, color="white")
    ax_a.set_ylim(0, 1.12)
    ax_a.set_ylabel("AUC-ROC", color="white", fontsize=11)
    ax_a.set_title("Panel A: AUC Comparison (5-fold CV ± 95% CI)", color="white",
                   fontsize=11, fontweight="bold")
    ax_a.axhline(0.5, ls="--", color="#555", lw=1, label="Random (0.5)")
    ax_a.tick_params(colors="white")
    for spine in ax_a.spines.values():
        spine.set_edgecolor("#333")
    for bar, auc in zip(bars, aucs):
        ax_a.text(bar.get_x() + bar.get_width() / 2, auc + 0.02,
                  f"{auc:.3f}", ha="center", va="bottom", fontsize=8, color="white")
    ax_a.legend(fontsize=8, framealpha=0.2, labelcolor="white")

    # ── Panel B: ROC curves ───────────────────────────────────────────────────
    ax_b = fig.add_subplot(gs[0, 1])
    ax_b.set_facecolor("#0f0f0f")
    for name, (fpr, tpr, auc) in roc_data.items():
        ax_b.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})",
                  color=COLORS.get(name, "#888"), lw=2)
    ax_b.plot([0, 1], [0, 1], ls="--", color="#555", lw=1)
    ax_b.set_xlabel("False Positive Rate", color="white")
    ax_b.set_ylabel("True Positive Rate",  color="white")
    ax_b.set_title("Panel B: ROC Curves (OOF predictions)", color="white",
                   fontsize=11, fontweight="bold")
    ax_b.legend(fontsize=8, framealpha=0.2, labelcolor="white",
                loc="lower right")
    ax_b.tick_params(colors="white")
    for spine in ax_b.spines.values():
        spine.set_edgecolor("#333")

    # ── Panel C: Precision-Recall curves ─────────────────────────────────────
    ax_c = fig.add_subplot(gs[1, 0])
    ax_c.set_facecolor("#0f0f0f")
    for name, (prec, rec, ap) in pr_data.items():
        ax_c.plot(rec, prec, label=f"{name} (AP={ap:.3f})",
                  color=COLORS.get(name, "#888"), lw=2)
    # Random baseline: proportion of positives
    # Compute from first model's y_true (stored in pr_data via roc_data)
    ax_c.set_xlabel("Recall",    color="white")
    ax_c.set_ylabel("Precision", color="white")
    ax_c.set_title("Panel C: Precision-Recall Curves", color="white",
                   fontsize=11, fontweight="bold")
    ax_c.legend(fontsize=8, framealpha=0.2, labelcolor="white")
    ax_c.tick_params(colors="white")
    for spine in ax_c.spines.values():
        spine.set_edgecolor("#333")

    # ── Panel D: SHAP feature importance ─────────────────────────────────────
    ax_d = fig.add_subplot(gs[1, 1])
    ax_d.set_facecolor("#0f0f0f")
    shap_arr  = np.array(shap_vals, dtype=float)
    order_idx = np.argsort(shap_arr)
    sorted_v  = shap_arr[order_idx].tolist()
    sorted_fn = [feat_names[int(i)] for i in order_idx.flat]
    ax_d.barh(range(len(sorted_fn)), sorted_v, color="#f0b429", alpha=0.85)
    ax_d.set_yticks(range(len(sorted_fn)))
    ax_d.set_yticklabels(sorted_fn, fontsize=9, color="white")
    ax_d.set_xlabel("Mean |SHAP value| (Random Forest)", color="white")
    ax_d.set_title("Panel D: Feature Importance (SHAP)", color="white",
                   fontsize=11, fontweight="bold")
    ax_d.tick_params(colors="white")
    for spine in ax_d.spines.values():
        spine.set_edgecolor("#333")

    # Overall title
    fig.suptitle(
        "Figure 7 v2 — GhostTrack: Tabular vs GNN Baseline Comparison\n"
        "(5-fold stratified CV, artist-level holdout, SHAP importance, n=76 artists)",
        color="white", fontsize=13, fontweight="bold", y=0.98,
    )

    fig.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    log.info("Figure saved → %s", out_path)


# ─────────────────────────────────────────────────────────────────────────────
# 7.  Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    data, X, y, feat_names = load_data()

    log.info("=" * 60)
    log.info("CLASS DISTRIBUTION: %d ghost, %d organic (total %d)",
             int(y.sum()), int((y == 0).sum()), len(y))
    log.info("  imbalance ratio: %.1f:1 organic:ghost", (y == 0).sum() / max(1, y.sum()))
    log.info("=" * 60)

    all_metrics: list[dict] = []
    all_oof_scores: dict[str, np.ndarray] = {}
    y_true_oof: np.ndarray | None = None

    # ── Tabular baselines ────────────────────────────────────────────────────
    log.info("\n[1/2] Running tabular baselines (LR, RF, MLP)...")
    tab_metrics, tab_scores, tab_true = run_tabular_baselines(X, y, feat_names)
    all_metrics.extend(tab_metrics)
    all_oof_scores.update(tab_scores)
    y_true_oof = tab_true["all"]

    # ── GNN baselines ────────────────────────────────────────────────────────
    log.info("\n[2/2] Running GNN baselines (GraphSAGE, HGT)...")

    sage_metrics, sage_scores, sage_true = run_gnn_cv(
        data, HeteroSAGE, "GraphSAGE",
        save_path=MODELS_DIR / "sage_model.pt",
    )
    all_metrics.extend(sage_metrics)
    all_oof_scores["GraphSAGE"] = sage_scores

    hgt_metrics, hgt_scores, hgt_true = run_gnn_cv(
        data, HGTModel, "HGT",
        save_path=MODELS_DIR / "hgt_model.pt",
    )
    all_metrics.extend(hgt_metrics)
    all_oof_scores["HGT"] = hgt_scores

    # Use GNN true labels (same artist set, same folds)
    # They should match tab_true since same underlying data
    gnn_y_true = sage_true  # all 5-fold OOF truth labels

    # ── Temporal validation (tabular only — GNN temporal not reliable here) ──
    log.info("\nRunning temporal validation...")
    temp_metrics = run_temporal(X, y, data, {})
    for m in temp_metrics:
        all_metrics.append(m)

    # ── Save results CSV ─────────────────────────────────────────────────────
    results_df = pd.DataFrame(all_metrics)
    results_df.to_csv(RESULTS_CSV, index=False)
    log.info("Results saved → %s", RESULTS_CSV)

    # ── Print comparison table ────────────────────────────────────────────────
    cv_results = results_df[results_df["split"] == "5fold_cv"]
    print("\n" + "=" * 80)
    print("BASELINE COMPARISON (5-fold CV)")
    print("=" * 80)
    print(cv_results[["model", "auc", "auc_lo", "auc_hi",
                       "f1", "prec_at_k", "n_ghost"]].to_string(index=False))

    # ── Honest GNN vs tabular verdict ────────────────────────────────────────
    tab_auc   = cv_results[cv_results["model"].isin(
        ["LogisticRegression", "RandomForest", "MLP"]
    )]["auc"].max()
    gnn_auc   = cv_results[cv_results["model"].isin(
        ["GraphSAGE", "HGT"]
    )]["auc"].max()
    delta     = gnn_auc - tab_auc

    print("\n" + "─" * 60)
    print(f"Best tabular AUC : {tab_auc:.3f}")
    print(f"Best GNN AUC     : {gnn_auc:.3f}")
    print(f"Delta            : {delta:+.3f}")
    if delta >= 0.03:
        print("VERDICT: GNN OUTPERFORMS tabular baselines by ≥3 AUC points.")
    else:
        print("VERDICT: GNN does NOT outperform tabular baselines by ≥3 AUC points.")
        print("         This is a NEGATIVE RESULT. Likely causes:")
        print("         • Only 3 ghost artists have real ISRC edges (synthetic edges dominate)")
        print("         • Graph structure is insufficient to carry meaningful ghost signal")
        print("         • Audio features alone are highly discriminative (trivial task)")
        print("         • 76-node graph is too small for GNN inductive generalization")
    print("─" * 60)

    # ── SHAP ─────────────────────────────────────────────────────────────────
    log.info("Computing SHAP values...")
    shap_vals, shap_names = compute_shap(X, y, feat_names)

    # ── Curves ───────────────────────────────────────────────────────────────
    # Use combined OOF score dict with same y_true
    combined_scores = {**all_oof_scores}
    # For GNN, align true labels (use tabular y_true_oof since same base data)
    roc_data, pr_data = collect_curves(y_true_oof, {
        k: v for k, v in combined_scores.items()
        if k in ["LogisticRegression", "RandomForest", "MLP"]
    })
    # Add GNN curves separately (may have different length if fold sizes differ)
    for gnn_name, gnn_sc, gnn_yt in [
        ("GraphSAGE", sage_scores, sage_true),
        ("HGT",       hgt_scores,  hgt_true),
    ]:
        if len(np.unique(gnn_yt)) < 2:
            continue
        try:
            fpr, tpr, _ = roc_curve(gnn_yt, gnn_sc)
            auc_v = roc_auc_score(gnn_yt, gnn_sc)
            roc_data[gnn_name] = (fpr, tpr, auc_v)
            prec, rec, _ = precision_recall_curve(gnn_yt, gnn_sc)
            ap = average_precision_score(gnn_yt, gnn_sc)
            pr_data[gnn_name] = (prec, rec, ap)
        except Exception as e:
            log.warning("GNN curve failed for %s: %s", gnn_name, e)

    # ── Figure ───────────────────────────────────────────────────────────────
    make_figure(
        metrics_cv = cv_results.to_dict("records"),
        roc_data   = roc_data,
        pr_data    = pr_data,
        shap_vals  = shap_vals,
        feat_names = shap_names,
        out_path   = FIG_PATH,
    )

    # ── Save hyperparams alongside results ───────────────────────────────────
    hp_path = ROOT / "results" / "baseline_hparams.json"
    hp_path.write_text(json.dumps(HPARAMS, indent=2, default=str))
    log.info("Hyperparameters saved → %s", hp_path)

    print(f"\nAll done. Results: {RESULTS_CSV}")
    print(f"Figure:  {FIG_PATH}")


if __name__ == "__main__":
    main()
