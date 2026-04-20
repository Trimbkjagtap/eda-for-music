"""
scripts/02_build_hetero_graph.py

Construct a heterogeneous PyTorch Geometric graph for ghost-artist detection.

Node types:
  - 'artist'     : one node per labelled artist (ghost=1, organic=0)
  - 'registrant' : one node per ISRC prefix (production company)

Edge types:
  - ('artist', 'registered_with', 'registrant')
      weight = number of tracks registered under that prefix for that artist
  - ('registrant', 'rev_registered_with', 'artist')  [reverse, for undirected conv]

Artist features (8-dim, NO HHI, NO closure_rate — those are derived labels):
  0  track_count           normalised catalog size
  1  mean_duration_ms      normalised mean track length
  2  var_danceability      audio variance (from Kaggle)
  3  var_energy
  4  var_valence
  5  var_acousticness
  6  genre_count           stylistic breadth proxy
  7  total_audio_variance  Σ of the four audio variances

Registrant features (4-dim):
  0  total_tracks_managed  how many tracks go through this prefix
  1  n_artists_served      how many distinct artists use this prefix
  2  is_known_ghost_prefix 1 if prefix appears in Neo4j ghost tracks
  3  is_known_aggregator   1 if prefix is an aggregator (e.g. DistroKid codes)

Data sources (in priority order):
  1. data/processed/neo4j_full_graph.csv   — ISRC edges for 3 ghost artists
  2. data/ground_truth/ghost_artists.csv   — labels + names
  3. data/ground_truth/organic_artists.csv — labels + names
  4. data/processed/ghost_candidates_kaggle.csv  — Kaggle audio features (ghosts)
  5. data/processed/organic_controls_kaggle.csv  — Kaggle audio features (organics)

Output:
  data/processed/hetero_graph.pt          — PyG HeteroData object
  data/processed/hetero_graph_meta.json   — node counts, feature dims, label balance
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData

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
DATA = ROOT / "data"
OUT_PT = DATA / "processed" / "hetero_graph.pt"
OUT_META = DATA / "processed" / "hetero_graph_meta.json"

# ── known ghost ISRC prefixes (from Dagens Nyheter / Neo4j) ──────────────────
GHOST_PREFIXES: set[str] = {
    "CH654", "ITIWE", "ITI5B", "ITQV2",  # Calmo
    "ITO10", "ITLU5",                      # Meditation Relax Club
    "DEPI8", "DE1QW",                      # Relaxing White Noise
    "SEWA",  "SEWB",  "SEZZ",             # Firefly / Lucille / Tombola (DN)
    "SEXX",  "SECC",                       # Catfish / Calm+Collected (DN)
}

# Prefixes used by large distributors (not inherently suspicious, but a feature)
AGGREGATOR_PREFIXES: set[str] = {
    "QMDA5", "USUAN", "GBUM7", "USUM7", "USRC1",  # Universal
    "GBBKS", "GBAYE",                               # Sony / BMG
    "QMFME", "USSM1",                               # Columbia/Epic
    "TCACM", "TCADR",                               # DistroKid
    "SE6I9",                                        # TuneCore
}


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Load ground-truth labels
# ─────────────────────────────────────────────────────────────────────────────

def _load_labels() -> pd.DataFrame:
    """
    Combine ghost + organic CSVs.  Returns a DataFrame with columns:
      artist_id (str | None), name, label (0/1), source, confidence
    """
    ghost = pd.read_csv(DATA / "ground_truth" / "ghost_artists.csv")
    ghost["label"] = 1

    organic = pd.read_csv(DATA / "ground_truth" / "organic_artists.csv")
    organic["label"] = 0

    df = pd.concat([ghost, organic], ignore_index=True)
    df = df.rename(columns={"spotify_artist_id": "artist_id"})
    df["artist_id"] = df["artist_id"].where(df["artist_id"].notna(), None)
    log.info("Labels loaded: %d ghost, %d organic", ghost["label"].sum(), len(organic))
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Load Kaggle audio features
# ─────────────────────────────────────────────────────────────────────────────

def _load_kaggle() -> pd.DataFrame:
    """
    Merge ghost_candidates + organic_controls from Kaggle.
    Returns per-artist audio feature rows, one row per artist name.
    """
    gk = pd.read_csv(DATA / "processed" / "ghost_candidates_kaggle.csv")
    gk["kaggle_label"] = 1

    ok = pd.read_csv(DATA / "processed" / "organic_controls_kaggle.csv")
    ok["kaggle_label"] = 0

    df = pd.concat([gk, ok], ignore_index=True)
    df = df.rename(columns={"artist": "name"})
    # Deduplicate on name (keep first)
    df = df.drop_duplicates(subset="name", keep="first")
    log.info("Kaggle features: %d rows (%d ghost, %d organic)",
             len(df), (df["kaggle_label"] == 1).sum(), (df["kaggle_label"] == 0).sum())
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Build artist feature matrix
# ─────────────────────────────────────────────────────────────────────────────

AUDIO_COLS = [
    "var_danceability", "var_energy", "var_valence",
    "var_acousticness", "total_variance",
]

def _build_artist_features(labels: pd.DataFrame, kaggle: pd.DataFrame) -> pd.DataFrame:
    """
    Merge label rows with Kaggle audio features.
    For artists not in Kaggle, impute with class median.
    Returns DataFrame with columns used for node features.
    """
    # Normalise name for matching (lowercase, strip)
    labels["name_key"] = labels["name"].str.lower().str.strip()
    kaggle["name_key"] = kaggle["name"].str.lower().str.strip()

    merged = labels.merge(
        kaggle[["name_key", "track_count", "mean_duration_ms",
                "genre_count"] + AUDIO_COLS],
        on="name_key", how="left",
    )

    # Impute missing numeric cols with class-wise median
    feature_cols = ["track_count", "mean_duration_ms", "genre_count"] + AUDIO_COLS
    for col in feature_cols:
        for label_val in (0, 1):
            mask_miss = merged["label"] == label_val
            class_median = merged.loc[mask_miss & merged[col].notna(), col].median()
            if pd.isna(class_median):
                class_median = merged[col].median()
            fill_mask = mask_miss & merged[col].isna()
            merged.loc[fill_mask, col] = class_median

    log.info("Artist features built: %d artists, %d with Kaggle data",
             len(merged), merged["track_count"].notna().sum())
    return merged


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Load ISRC edges (from Neo4j export) & build registrant nodes
# ─────────────────────────────────────────────────────────────────────────────

def _load_isrc_graph() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load neo4j_full_graph.csv and compute per-registrant features.
    Returns (edge_df, registrant_df).
    """
    neo = pd.read_csv(DATA / "processed" / "neo4j_full_graph.csv")
    # edge: artist_id -> prefix, weight = track count
    edges = (
        neo.groupby(["artist_id", "prefix"])
        .size()
        .reset_index(name="track_count_edge")
    )

    # Registrant stats
    reg = neo.groupby("prefix").agg(
        total_tracks_managed=("track_id", "count"),
        n_artists_served=("artist_id", "nunique"),
    ).reset_index()
    reg["is_known_ghost_prefix"] = reg["prefix"].isin(GHOST_PREFIXES).astype(float)
    reg["is_known_aggregator"] = reg["prefix"].isin(AGGREGATOR_PREFIXES).astype(float)

    log.info("ISRC edges: %d; registrant prefixes: %d", len(edges), len(reg))
    return edges, reg


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Build synthetic registrant nodes for artists without ISRC data
#     (needed so every artist is connected to at least one registrant)
# ─────────────────────────────────────────────────────────────────────────────

def _synthetic_registrants(artists_df: pd.DataFrame,
                            isrc_edges: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Artists in the ground truth that have no ISRC edges get a 'synthetic'
    registrant based on their label (ghost-pool vs organic-pool).
    This preserves the graph structure while acknowledging the data gap.
    """
    artists_with_edge = set(isrc_edges["artist_id"].unique())

    # Build a name->artist_id map only for artists that have an ID
    id_map = dict(
        artists_df.dropna(subset=["artist_id"])
        .set_index("artist_id")["label"]
    )

    artists_without_edge = artists_df[
        ~artists_df["artist_id"].isin(artists_with_edge)
    ].copy()

    # Assign synthetic prefix by label and source
    def _synth_prefix(row: pd.Series) -> str:
        if row["label"] == 1:
            src = str(row.get("source", "unk")).lower()
            if "smith" in src:
                return "SYNTH_SMITH"
            elif "dn" in src:
                return "SYNTH_DN"
            else:
                return "SYNTH_GHOST_OTHER"
        else:
            return "SYNTH_ORGANIC"

    artists_without_edge["synth_prefix"] = artists_without_edge.apply(
        _synth_prefix, axis=1
    )

    # New synthetic edges (weight=1 — unknown track count)
    synth_edges = artists_without_edge[
        artists_without_edge["artist_id"].notna()
    ][["artist_id", "synth_prefix"]].rename(
        columns={"synth_prefix": "prefix"}
    )
    synth_edges["track_count_edge"] = 1.0

    # For artists without any spotify ID, we can't add them to the graph at all
    n_no_id = (artists_without_edge["artist_id"].isna()).sum()
    log.info("Artists without ISRC edges: %d total, %d also lack Spotify ID (skipped)",
             len(artists_without_edge), n_no_id)

    # New registrant rows for synthetic prefixes
    synth_regs = pd.DataFrame([
        {"prefix": "SYNTH_SMITH",       "total_tracks_managed": 0, "n_artists_served": 0,
         "is_known_ghost_prefix": 1.0,  "is_known_aggregator": 0.0},
        {"prefix": "SYNTH_DN",          "total_tracks_managed": 0, "n_artists_served": 0,
         "is_known_ghost_prefix": 1.0,  "is_known_aggregator": 0.0},
        {"prefix": "SYNTH_GHOST_OTHER", "total_tracks_managed": 0, "n_artists_served": 0,
         "is_known_ghost_prefix": 0.5,  "is_known_aggregator": 0.0},
        {"prefix": "SYNTH_ORGANIC",     "total_tracks_managed": 0, "n_artists_served": 0,
         "is_known_ghost_prefix": 0.0,  "is_known_aggregator": 0.0},
    ])

    return synth_edges, synth_regs


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Min-max normalise
# ─────────────────────────────────────────────────────────────────────────────

def _minmax(arr: np.ndarray) -> np.ndarray:
    lo, hi = arr.min(0), arr.max(0)
    rng = np.where(hi - lo == 0, 1.0, hi - lo)
    return (arr - lo) / rng


# ─────────────────────────────────────────────────────────────────────────────
# 7.  Assemble HeteroData
# ─────────────────────────────────────────────────────────────────────────────

def build_hetero_graph() -> HeteroData:
    labels    = _load_labels()
    kaggle    = _load_kaggle()
    artists   = _build_artist_features(labels, kaggle)
    isrc_edges, registrants = _load_isrc_graph()
    synth_edges, synth_regs = _synthetic_registrants(artists, isrc_edges)

    # Combine real + synthetic registrants
    all_regs = pd.concat([registrants, synth_regs], ignore_index=True)
    all_regs = all_regs.drop_duplicates(subset="prefix", keep="first")

    # Combine real + synthetic edges (only for artists with an ID)
    all_edges = pd.concat([isrc_edges, synth_edges], ignore_index=True)

    # ── Filter artists: keep only those with a Spotify ID (needed for edge table)
    artists_with_id = artists[artists["artist_id"].notna()].copy()
    artists_with_id = artists_with_id.drop_duplicates(subset="artist_id", keep="first")

    # Filter edges to only artists in the graph
    valid_ids = set(artists_with_id["artist_id"].tolist())
    all_edges = all_edges[all_edges["artist_id"].isin(valid_ids)]

    # ── Index maps ───────────────────────────────────────────────────────────
    artist_id_to_idx = {aid: i for i, aid in enumerate(artists_with_id["artist_id"])}
    prefix_to_idx    = {p: i for i, p in enumerate(all_regs["prefix"])}

    # ── Artist node features (8 dims) ────────────────────────────────────────
    feat_cols = ["track_count", "mean_duration_ms"] + AUDIO_COLS + ["genre_count"]
    artist_feat = artists_with_id[feat_cols].values.astype(np.float32)
    artist_feat = _minmax(artist_feat)
    artist_labels = artists_with_id["label"].values.astype(np.long)

    # ── Registrant node features (4 dims) ────────────────────────────────────
    reg_feat_cols = ["total_tracks_managed", "n_artists_served",
                     "is_known_ghost_prefix", "is_known_aggregator"]
    reg_feat = all_regs[reg_feat_cols].values.astype(np.float32)
    # Normalise first two columns only
    for c in range(2):
        lo, hi = reg_feat[:, c].min(), reg_feat[:, c].max()
        reg_feat[:, c] = (reg_feat[:, c] - lo) / max(hi - lo, 1.0)

    # ── Edge indices ─────────────────────────────────────────────────────────
    src_artist = [artist_id_to_idx[r] for r in all_edges["artist_id"]]
    dst_reg    = [prefix_to_idx[p]    for p in all_edges["prefix"]]
    edge_weight = all_edges["track_count_edge"].values.astype(np.float32)
    # Normalise edge weight
    ew_max = edge_weight.max() if edge_weight.max() > 0 else 1.0
    edge_weight = edge_weight / ew_max

    # ── Build HeteroData ─────────────────────────────────────────────────────
    data = HeteroData()

    data["artist"].x     = torch.tensor(artist_feat,   dtype=torch.float)
    data["artist"].y     = torch.tensor(artist_labels, dtype=torch.long)
    data["artist"].node_id = list(artists_with_id["artist_id"])

    data["registrant"].x = torch.tensor(reg_feat, dtype=torch.float)

    ei = torch.tensor([src_artist, dst_reg], dtype=torch.long)
    data["artist", "registered_with", "registrant"].edge_index  = ei
    data["artist", "registered_with", "registrant"].edge_weight = torch.tensor(
        edge_weight, dtype=torch.float
    )
    # Reverse edges for undirected message passing
    ei_rev = torch.tensor([dst_reg, src_artist], dtype=torch.long)
    data["registrant", "rev_registered_with", "artist"].edge_index  = ei_rev
    data["registrant", "rev_registered_with", "artist"].edge_weight = torch.tensor(
        edge_weight, dtype=torch.float
    )

    # ── Train / val / test masks (artist-level, stratified) ──────────────────
    rng = np.random.default_rng(SEED)
    n = len(artist_labels)
    idx = np.arange(n)
    ghost_idx   = idx[artist_labels == 1]
    organic_idx = idx[artist_labels == 0]

    def _split(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rng.shuffle(arr)
        n1, n2 = max(1, len(arr) // 5), max(1, len(arr) // 5)
        return arr[n2 + n1:], arr[n2:n2 + n1], arr[:n2]

    g_train, g_val, g_test = _split(ghost_idx)
    o_train, o_val, o_test = _split(organic_idx)

    for split_name, g_idx, o_idx in [
        ("train_mask", g_train, o_train),
        ("val_mask",   g_val,   o_val),
        ("test_mask",  g_test,  o_test),
    ]:
        mask = torch.zeros(n, dtype=torch.bool)
        mask[np.concatenate([g_idx, o_idx])] = True
        setattr(data["artist"], split_name, mask)

    # ── Temporal split mask (first_release ≤ 2023 → train_temporal) ──────────
    # We don't have per-artist release dates for most; flag the 3 Neo4j ghosts
    # (all have data going back to 2014-2020) as "old" and recent entries as "new".
    # This is a best-effort approximation given the data constraints.
    is_neo4j_artist = artists_with_id["source"].isin(["Smith"]) | (
        artists_with_id["confidence"] == "high"
    )
    temporal_train = torch.zeros(n, dtype=torch.bool)
    temporal_test  = torch.zeros(n, dtype=torch.bool)
    temporal_train[is_neo4j_artist.values] = True
    temporal_test[~is_neo4j_artist.values] = True
    data["artist"].temporal_train_mask = temporal_train
    data["artist"].temporal_test_mask  = temporal_test

    log.info(
        "HeteroData built: %d artist nodes (%d ghost, %d organic), "
        "%d registrant nodes, %d edges",
        n,
        int(artist_labels.sum()),
        int((artist_labels == 0).sum()),
        len(all_regs),
        ei.shape[1],
    )
    log.info(
        "Train %d | Val %d | Test %d",
        data["artist"].train_mask.sum().item(),
        data["artist"].val_mask.sum().item(),
        data["artist"].test_mask.sum().item(),
    )

    return data, artists_with_id, artist_feat, artist_labels, feat_cols


# ─────────────────────────────────────────────────────────────────────────────
# 8.  Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    data, artists_with_id, artist_feat, artist_labels, feat_cols = build_hetero_graph()

    torch.save(data, OUT_PT)
    log.info("Saved graph → %s", OUT_PT)

    n = len(artist_labels)
    meta = {
        "n_artist_nodes": n,
        "n_ghost_nodes":   int(artist_labels.sum()),
        "n_organic_nodes": int((artist_labels == 0).sum()),
        "n_registrant_nodes": data["registrant"].x.shape[0],
        "n_edges": int(data["artist", "registered_with", "registrant"].edge_index.shape[1]),
        "artist_feature_dim": int(data["artist"].x.shape[1]),
        "registrant_feature_dim": int(data["registrant"].x.shape[1]),
        "artist_feature_names": feat_cols,
        "registrant_feature_names": [
            "total_tracks_managed", "n_artists_served",
            "is_known_ghost_prefix", "is_known_aggregator",
        ],
        "train_nodes": int(data["artist"].train_mask.sum()),
        "val_nodes":   int(data["artist"].val_mask.sum()),
        "test_nodes":  int(data["artist"].test_mask.sum()),
        "seed": SEED,
        "data_caveats": [
            "Only 3 ghost artists have real ISRC edges from Neo4j; others use synthetic pooled registrants.",
            "Audio features for 97 ghost artists imputed from class median (no Kaggle match).",
            "Temporal split approximated: high-confidence/Smith ghosts = pre-2024, rest = 2024+.",
        ],
    }
    OUT_META.write_text(json.dumps(meta, indent=2))
    log.info("Saved metadata → %s", OUT_META)

    # Print summary
    print("\n" + "=" * 60)
    print("HETEROGENEOUS GRAPH SUMMARY")
    print("=" * 60)
    for k, v in meta.items():
        if k != "data_caveats":
            print(f"  {k}: {v}")
    print("\nCaveats:")
    for c in meta["data_caveats"]:
        print(f"  ⚠  {c}")


if __name__ == "__main__":
    main()
