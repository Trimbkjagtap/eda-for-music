"""
Builds a PyTorch Geometric graph dataset for ghost-artist detection.

Data sources:
  - Neo4j:  3 real artists (all ghost) + their ISRC/catalog features
  - Kaggle: 11 ghost candidates + up to 50 organic controls
  - Precomputed signals from exercise5_walk_metrics.csv + exercise4_metrics.csv

Node features (8 dims):
  0: track_count          (catalog size, normalized)
  1: closure_rate         (same-day releases, from walk metrics)
  2: tracks_per_day       (upload velocity proxy)
  3: hhi                  (ISRC concentration from exercise4)
  4: total_variance       (audio feature variance, from Kaggle)
  5: mean_duration_ms     (normalized)
  6: isrc_prefix_count    (number of unique production companies)
  7: genre_count          (stylistic breadth proxy)

Labels:
  1 = ghost, 0 = organic

Graph structure:
  Ghost nodes: fully connected (simulated coordinated network)
  Organic nodes: connected by Euclidean feature similarity (k-NN, k=5)
  Cross edges: none (clean separation assumption)

Saved:
  data/processed/gnn_dataset.pt
  data/processed/gnn_dataset_meta.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from loguru import logger

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

PROCESSED = ROOT / "data" / "processed"
FEATURE_NAMES = [
    "track_count",
    "closure_rate",
    "tracks_per_day",
    "hhi",
    "total_variance",
    "mean_duration_ms",
    "isrc_prefix_count",
    "genre_count",
]

# ── Known signal values (from exercises 4 & 5) ───────────────────────────────

# exercise5_walk_metrics columns we care about:
#  Artist, Tracks, Span (days), Release rate (tracks/mo),
#  Closure (≤1d gap %), Median gap (days), ISRC prefixes
_EX5 = {
    "Relaxing White Noise": {
        "track_count": 280,
        "closure_rate": 0.810,
        "tracks_per_day": 280 / 1156,
        "median_gap_days": 0.0,
        "isrc_prefix_count": 2,
    },
    "Meditation Relax Club": {
        "track_count": 172,
        "closure_rate": 0.947,
        "tracks_per_day": 172 / 1240,
        "median_gap_days": 0.0,
        "isrc_prefix_count": 2,
    },
    "Calmo": {
        "track_count": 38,
        "closure_rate": 0.324,
        "tracks_per_day": 38 / 2030,
        "median_gap_days": 29.0,
        "isrc_prefix_count": 4,
    },
    "Nils Frahm": {
        "track_count": 56,
        "closure_rate": 0.036,
        "tracks_per_day": 56 / 7566,
        "median_gap_days": 98.0,
        "isrc_prefix_count": 1,
    },
}

# exercise4_metrics: HHI per artist
_EX4_HHI = {
    "Relaxing White Noise": 0.6715,
    "Meditation Relax Club": 0.5152,
    "Calmo": 0.4515,
    "Nils Frahm": 0.0,   # not in Neo4j but treated as organic
}


# ── Load Kaggle data ──────────────────────────────────────────────────────────

def _load_kaggle_data(max_organic: int = 50) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (ghost_df, organic_df) from Kaggle processed CSVs."""
    ghost_path = PROCESSED / "ghost_candidates_kaggle.csv"
    organic_path = PROCESSED / "organic_controls_kaggle.csv"

    ghost_df = pd.read_csv(ghost_path) if ghost_path.exists() else pd.DataFrame()
    organic_df = pd.read_csv(organic_path) if organic_path.exists() else pd.DataFrame()

    # Limit organics to keep class balance manageable
    if len(organic_df) > max_organic:
        organic_df = organic_df.sample(max_organic, random_state=42)

    return ghost_df, organic_df


# ── Feature extraction ────────────────────────────────────────────────────────

def _kaggle_row_to_features(row: pd.Series, label: int) -> dict[str, Any]:
    """Extract feature dict from a Kaggle CSV row."""
    # Kaggle rows have: track_count, mean_duration_ms, total_variance, genre_count
    # They lack cadence/HHI — impute from class mean
    # Ghost imputation: closure_rate≈0.50, tracks_per_day≈0.10, hhi≈0.45
    # Organic imputation: closure_rate≈0.02, tracks_per_day≈0.005, hhi≈0.10
    if label == 1:
        closure_rate_imp = 0.50
        tracks_per_day_imp = 0.10
        hhi_imp = 0.45
        isrc_imp = 2.0
    else:
        closure_rate_imp = 0.02
        tracks_per_day_imp = 0.005
        hhi_imp = 0.10
        isrc_imp = 3.0

    return {
        "name": row.get("artist", "unknown"),
        "label": label,
        "track_count": float(row.get("track_count", 20)),
        "closure_rate": closure_rate_imp,
        "tracks_per_day": tracks_per_day_imp,
        "hhi": hhi_imp,
        "total_variance": float(row.get("total_variance", 0.0)),
        "mean_duration_ms": float(row.get("mean_duration_ms", 200_000)),
        "isrc_prefix_count": isrc_imp,
        "genre_count": float(row.get("genre_count", 2)),
        "source": "kaggle",
    }


def _neo4j_artist_to_features(name: str, label: int) -> dict[str, Any]:
    """Build feature dict from pre-computed Neo4j artist data."""
    ex5 = _EX5.get(name, {})
    hhi = _EX4_HHI.get(name, 0.0)

    return {
        "name": name,
        "label": label,
        "track_count": float(ex5.get("track_count", 50)),
        "closure_rate": float(ex5.get("closure_rate", 0.0)),
        "tracks_per_day": float(ex5.get("tracks_per_day", 0.01)),
        "hhi": float(hhi),
        "total_variance": 0.0,    # not in Kaggle dataset (niche genre miss)
        "mean_duration_ms": 180_000.0,
        "isrc_prefix_count": float(ex5.get("isrc_prefix_count", 2)),
        "genre_count": 1.0,       # mono-genre (ambient/relaxation)
        "source": "neo4j",
    }


# ── Edge construction ─────────────────────────────────────────────────────────

def _build_ghost_edges(ghost_indices: list[int]) -> list[tuple[int, int]]:
    """Fully connect all ghost nodes (coordinated-network assumption)."""
    edges = []
    for i in ghost_indices:
        for j in ghost_indices:
            if i != j:
                edges.append((i, j))
    return edges


def _build_organic_knn_edges(
    features: np.ndarray,
    organic_indices: list[int],
    k: int = 5,
) -> list[tuple[int, int]]:
    """Connect organic nodes to their k nearest neighbours by feature distance."""
    if len(organic_indices) <= 1:
        return []

    org_feat = features[organic_indices]
    edges = []
    for ii, i in enumerate(organic_indices):
        # Euclidean distance to all other organics
        diffs = org_feat - org_feat[ii]
        dists = np.linalg.norm(diffs, axis=1)
        dists[ii] = np.inf   # exclude self
        nn_local = np.argsort(dists)[:k]
        for jj in nn_local:
            j = organic_indices[jj]
            edges.append((i, j))
            edges.append((j, i))
    return edges


# ── Main build function ───────────────────────────────────────────────────────

def build_dataset(
    max_organic: int = 50,
    val_frac: float = 0.2,
    test_frac: float = 0.2,
    random_seed: int = 42,
) -> Data:
    """
    Build and return a PyTorch Geometric Data object.

    Parameters
    ----------
    max_organic   : Max organic nodes from Kaggle (keeps class balance)
    val_frac      : Fraction of labeled nodes for validation
    test_frac     : Fraction of labeled nodes for testing
    random_seed   : RNG seed for reproducibility

    Returns
    -------
    torch_geometric.data.Data
    """
    rng = np.random.default_rng(random_seed)
    records: list[dict] = []

    # 1. Neo4j ghost artists (3 known ghosts)
    for name in ["Relaxing White Noise", "Meditation Relax Club", "Calmo"]:
        records.append(_neo4j_artist_to_features(name, label=1))

    # 2. Nils Frahm — known organic (not in Neo4j but fully measured)
    records.append(_neo4j_artist_to_features("Nils Frahm", label=0))

    # 3. Kaggle ghost candidates (11 rows)
    ghost_df, organic_df = _load_kaggle_data(max_organic=max_organic)
    for _, row in ghost_df.iterrows():
        records.append(_kaggle_row_to_features(row, label=1))

    # 4. Kaggle organic controls
    for _, row in organic_df.iterrows():
        records.append(_kaggle_row_to_features(row, label=0))

    logger.info(f"Total nodes: {len(records)}")

    # ── Build feature matrix ──────────────────────────────────────────────────
    feature_cols = FEATURE_NAMES
    feat_arr = np.array(
        [[r[c] for c in feature_cols] for r in records],
        dtype=np.float32,
    )

    # Normalize each feature to [0, 1] using per-column min-max
    col_min = feat_arr.min(axis=0)
    col_max = feat_arr.max(axis=0)
    col_range = col_max - col_min
    col_range[col_range == 0] = 1.0   # avoid division by zero
    feat_norm = (feat_arr - col_min) / col_range

    labels = np.array([r["label"] for r in records], dtype=np.int64)
    names = [r["name"] for r in records]

    ghost_indices = [i for i, r in enumerate(records) if r["label"] == 1]
    organic_indices = [i for i, r in enumerate(records) if r["label"] == 0]
    logger.info(f"Ghost nodes: {len(ghost_indices)}, Organic nodes: {len(organic_indices)}")

    # ── Build edges ───────────────────────────────────────────────────────────
    edge_list: list[tuple[int, int]] = []
    edge_list.extend(_build_ghost_edges(ghost_indices))
    edge_list.extend(_build_organic_knn_edges(feat_norm, organic_indices, k=5))

    if edge_list:
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)

    logger.info(f"Edges: {edge_index.shape[1]}")

    # ── Train/val/test masks ──────────────────────────────────────────────────
    n = len(records)
    all_idx = np.arange(n)
    rng.shuffle(all_idx)

    n_test = max(1, int(n * test_frac))
    n_val = max(1, int(n * val_frac))
    n_train = n - n_val - n_test

    train_idx = all_idx[:n_train]
    val_idx = all_idx[n_train:n_train + n_val]
    test_idx = all_idx[n_train + n_val:]

    train_mask = torch.zeros(n, dtype=torch.bool)
    val_mask = torch.zeros(n, dtype=torch.bool)
    test_mask = torch.zeros(n, dtype=torch.bool)
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True

    # ── Assemble PyG Data object ──────────────────────────────────────────────
    data = Data(
        x=torch.tensor(feat_norm, dtype=torch.float),
        edge_index=edge_index,
        y=torch.tensor(labels, dtype=torch.long),
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
    )

    # Save
    out_path = PROCESSED / "gnn_dataset.pt"
    torch.save(data, out_path)
    logger.info(f"Dataset saved to {out_path}")

    meta = {
        "num_nodes": n,
        "num_edges": int(edge_index.shape[1]),
        "num_features": len(feature_cols),
        "feature_names": feature_cols,
        "num_ghost": int(labels.sum()),
        "num_organic": int((labels == 0).sum()),
        "num_train": int(train_mask.sum()),
        "num_val": int(val_mask.sum()),
        "num_test": int(test_mask.sum()),
        "node_names": names,
        "node_labels": labels.tolist(),
        "col_min": col_min.tolist(),
        "col_max": col_max.tolist(),
        "normalization": "min-max per column",
    }
    meta_path = PROCESSED / "gnn_dataset_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    logger.info(f"Metadata saved to {meta_path}")

    return data


if __name__ == "__main__":
    d = build_dataset()
    print(f"Nodes: {d.num_nodes}, Edges: {d.num_edges}, Features: {d.num_node_features}")
    print(f"Ghost: {d.y.sum().item()}, Organic: {(d.y == 0).sum().item()}")
    print(f"Train: {d.train_mask.sum().item()}, Val: {d.val_mask.sum().item()}, Test: {d.test_mask.sum().item()}")
