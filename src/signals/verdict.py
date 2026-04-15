"""
Verdict Scorer: Combines all 7 signals into a single fraud probability.

Each signal returns a score in [0, 1]. The verdict combines them with weights.
Current weights: roughly equal, with a slight upweight on signals that are
independent of API availability (cadence, graph density, metadata).

Signals:
  S1: Audio Fingerprint Similarity   (Kaggle — works if tracks in dataset)
  S2: Release Cadence Synchrony      (Neo4j + pre-computed closure — always works)
  S3: Playlist Co-occurrence Density (Neo4j ISRC proxy — always works)
  S4: Follower-to-Catalog Ratio      (Neo4j catalog-only signals — always works)
  S5: Metadata Similarity            (Neo4j track names + keywords — always works)
  S6: Graph Density                  (Neo4j ISRC bipartite — always works)
  S7: Cross-Platform Discrepancy     (YouTube + Apple Music APIs — requires keys)

Verdict labels:
  LIKELY_GHOST   : overall_score > 0.70
  SUSPICIOUS     : 0.40 ≤ overall_score ≤ 0.70
  LIKELY_ORGANIC : overall_score < 0.40
"""
from __future__ import annotations

from loguru import logger

from src.graph.neo4j_client import Neo4jClient
import src.signals.audio_similarity as s1_mod
import src.signals.cadence_sync as s2_mod
import src.signals.playlist_cooccurrence as s3_mod
import src.signals.follower_ratio as s4_mod
import src.signals.metadata_similarity as s5_mod
import src.signals.graph_density as s6_mod
import src.signals.cross_platform as s7_mod

# Default signal weights (must sum to 1.0)
DEFAULT_WEIGHTS = {
    "s1_audio_similarity": 0.10,     # limited by Kaggle coverage
    "s2_cadence_sync": 0.20,         # strongest signal, pre-computed
    "s3_playlist_cooccurrence": 0.15,
    "s4_follower_ratio": 0.15,
    "s5_metadata_similarity": 0.15,
    "s6_graph_density": 0.15,        # HHI — very reliable
    "s7_cross_platform": 0.10,       # requires external API
}

_VERDICT_LABELS = {
    "LIKELY_GHOST": (0.70, 1.01),
    "SUSPICIOUS": (0.40, 0.70),
    "LIKELY_ORGANIC": (0.0, 0.40),
}


def compute_verdict(
    artist_id: str,
    artist_name: str | None = None,
    track_name: str | None = None,
    all_artist_ids: list[str] | None = None,
    weights: dict[str, float] | None = None,
    neo4j: Neo4jClient | None = None,
    run_s7: bool = True,
) -> dict:
    """
    Compute all 7 signal scores and return a combined verdict.

    Parameters
    ----------
    artist_id       : Spotify artist ID
    artist_name     : Display name (fetched from Neo4j if None)
    track_name      : Representative track for cross-platform check (optional)
    all_artist_ids  : All artist IDs in the database (for cluster signals)
    weights         : Custom signal weights (defaults to DEFAULT_WEIGHTS)
    neo4j           : Neo4jClient instance (created if None)
    run_s7          : Whether to run Signal 7 (requires external API calls)

    Returns
    -------
    dict with keys:
        artist_id, artist_name, signal_scores (dict),
        overall_score, verdict, confidence,
        signals_computed (int), signals_total (int),
        explanation (str), weights_used (dict)
    """
    _neo4j = neo4j or Neo4jClient()
    _weights = weights or DEFAULT_WEIGHTS

    # Resolve artist name
    if artist_name is None:
        rows = _neo4j.run(
            "MATCH (a:Artist {spotify_id: $id}) RETURN a.name AS name", id=artist_id
        )
        artist_name = rows[0]["name"] if rows else artist_id

    # Resolve all artist IDs for cluster signals
    if all_artist_ids is None:
        rows = _neo4j.run("MATCH (a:Artist) RETURN a.spotify_id AS id")
        all_artist_ids = [r["id"] for r in rows]

    signal_scores: dict[str, float | None] = {}
    signal_details: dict[str, dict] = {}

    # ── Signal 1: Audio Fingerprint Similarity ─────────────────────────
    try:
        r1 = s1_mod.score_artist(artist_id)
        signal_scores["s1_audio_similarity"] = r1.get("suspicion_score")
        signal_details["s1"] = r1
        logger.debug(f"S1 done: {signal_scores['s1_audio_similarity']}")
    except Exception as e:
        logger.warning(f"S1 failed for {artist_name}: {e}")
        signal_scores["s1_audio_similarity"] = None

    # ── Signal 2: Release Cadence Synchrony ───────────────────────────
    try:
        r2 = s2_mod.score_artist(artist_id, _neo4j)
        signal_scores["s2_cadence_sync"] = r2.get("suspicion_score")
        signal_details["s2"] = r2
        logger.debug(f"S2 done: {signal_scores['s2_cadence_sync']}")
    except Exception as e:
        logger.warning(f"S2 failed for {artist_name}: {e}")
        signal_scores["s2_cadence_sync"] = None

    # ── Signal 3: Playlist Co-occurrence ──────────────────────────────
    try:
        r3 = s3_mod.score_artist(artist_id, all_artist_ids, _neo4j)
        signal_scores["s3_playlist_cooccurrence"] = r3.get("suspicion_score")
        signal_details["s3"] = r3
        logger.debug(f"S3 done: {signal_scores['s3_playlist_cooccurrence']}")
    except Exception as e:
        logger.warning(f"S3 failed for {artist_name}: {e}")
        signal_scores["s3_playlist_cooccurrence"] = None

    # ── Signal 4: Follower/Catalog Ratio ──────────────────────────────
    try:
        r4 = s4_mod.score_artist(artist_id, _neo4j)
        signal_scores["s4_follower_ratio"] = r4.get("suspicion_score")
        signal_details["s4"] = r4
        logger.debug(f"S4 done: {signal_scores['s4_follower_ratio']}")
    except Exception as e:
        logger.warning(f"S4 failed for {artist_name}: {e}")
        signal_scores["s4_follower_ratio"] = None

    # ── Signal 5: Metadata Similarity ─────────────────────────────────
    try:
        r5 = s5_mod.score_artist(artist_id, all_artist_ids, _neo4j)
        signal_scores["s5_metadata_similarity"] = r5.get("suspicion_score")
        signal_details["s5"] = r5
        logger.debug(f"S5 done: {signal_scores['s5_metadata_similarity']}")
    except Exception as e:
        logger.warning(f"S5 failed for {artist_name}: {e}")
        signal_scores["s5_metadata_similarity"] = None

    # ── Signal 6: Graph Density ────────────────────────────────────────
    try:
        r6 = s6_mod.score_artist(artist_id, _neo4j)
        signal_scores["s6_graph_density"] = r6.get("suspicion_score")
        signal_details["s6"] = r6
        logger.debug(f"S6 done: {signal_scores['s6_graph_density']}")
    except Exception as e:
        logger.warning(f"S6 failed for {artist_name}: {e}")
        signal_scores["s6_graph_density"] = None

    # ── Signal 7: Cross-Platform Discrepancy ──────────────────────────
    if run_s7:
        try:
            r7 = s7_mod.score_artist(artist_name, track_name, artist_id, _neo4j)
            signal_scores["s7_cross_platform"] = r7.get("suspicion_score")
            signal_details["s7"] = r7
            logger.debug(f"S7 done: {signal_scores['s7_cross_platform']}")
        except Exception as e:
            logger.warning(f"S7 failed for {artist_name}: {e}")
            signal_scores["s7_cross_platform"] = None
    else:
        signal_scores["s7_cross_platform"] = None

    # ── Weighted average (skip None signals) ──────────────────────────
    available = {k: v for k, v in signal_scores.items() if v is not None}
    signals_computed = len(available)
    signals_total = len(signal_scores)

    if not available:
        logger.warning(f"No signals computed for {artist_name}")
        return _empty_verdict(artist_id, artist_name, signal_scores, _weights)

    # Re-normalize weights for available signals only
    total_weight = sum(_weights.get(k, 1 / 7) for k in available)
    weighted_sum = sum(
        v * _weights.get(k, 1 / 7) / total_weight
        for k, v in available.items()
    )
    overall_score = float(weighted_sum)

    # Confidence: fraction of signals computed
    confidence = signals_computed / signals_total

    verdict = _verdict_label(overall_score)

    # Explanation
    fired = [k for k, v in signal_scores.items() if v is not None and v >= 0.6]
    low = [k for k, v in signal_scores.items() if v is not None and v < 0.4]
    missing = [k for k, v in signal_scores.items() if v is None]

    explanation = _build_explanation(
        artist_name, verdict, overall_score, fired, low, missing, signal_scores, signal_details
    )

    logger.info(
        f"VERDICT {artist_name}: {verdict} (score={overall_score:.3f}, "
        f"confidence={confidence:.0%}, signals={signals_computed}/{signals_total})"
    )

    return {
        "artist_id": artist_id,
        "artist_name": artist_name,
        "signal_scores": signal_scores,
        "signal_details": signal_details,
        "overall_score": round(overall_score, 4),
        "verdict": verdict,
        "confidence": round(confidence, 2),
        "signals_computed": signals_computed,
        "signals_total": signals_total,
        "explanation": explanation,
        "weights_used": _weights,
    }


def print_report_card(verdict_dict: dict) -> None:
    """Print a formatted signal report card to stdout."""
    d = verdict_dict
    print(f"\n{'=' * 70}")
    print(f"  GHOST DETECTION REPORT: {d['artist_name']}")
    print(f"{'=' * 70}")
    print(f"  Verdict:     {d['verdict']}")
    print(f"  Score:       {d['overall_score']:.3f}  (0=organic, 1=ghost)")
    print(f"  Confidence:  {d['confidence']:.0%}  ({d['signals_computed']}/{d['signals_total']} signals computed)")
    print(f"\n  Signal Breakdown:")
    labels = {
        "s1_audio_similarity":    "S1  Audio Fingerprint  ",
        "s2_cadence_sync":        "S2  Release Cadence    ",
        "s3_playlist_cooccurrence": "S3  Playlist Co-occur ",
        "s4_follower_ratio":      "S4  Catalog Ratio      ",
        "s5_metadata_similarity": "S5  Metadata Sim       ",
        "s6_graph_density":       "S6  Graph Density (HHI)",
        "s7_cross_platform":      "S7  Cross-Platform     ",
    }
    for key, label in labels.items():
        val = d["signal_scores"].get(key)
        if val is None:
            bar = "  N/A"
            indicator = "—"
        else:
            filled = int(val * 20)
            bar = "  [" + "█" * filled + "░" * (20 - filled) + f"] {val:.2f}"
            indicator = "HIGH" if val >= 0.7 else ("MED" if val >= 0.4 else "low")
        print(f"  {label}: {bar}  {indicator}")
    print(f"\n  Explanation:")
    for line in d["explanation"].split("\n"):
        print(f"    {line}")
    print(f"{'=' * 70}\n")


# ──────────────────────────── helpers ────────────────────────────


def _verdict_label(score: float) -> str:
    if score >= 0.70:
        return "LIKELY_GHOST"
    elif score >= 0.40:
        return "SUSPICIOUS"
    return "LIKELY_ORGANIC"


def _build_explanation(
    name: str, verdict: str, score: float,
    fired: list, low: list, missing: list,
    signal_scores: dict, signal_details: dict,
) -> str:
    label_map = {
        "s1_audio_similarity": "audio fingerprint (Kaggle)",
        "s2_cadence_sync": "release cadence / bulk-upload fingerprint",
        "s3_playlist_cooccurrence": "ISRC operator co-occurrence",
        "s4_follower_ratio": "catalog density anomaly",
        "s5_metadata_similarity": "metadata / keyword pattern",
        "s6_graph_density": "ISRC concentration (HHI)",
        "s7_cross_platform": "cross-platform visibility",
    }
    lines = [f"{name} — {verdict} (score={score:.3f})"]
    if fired:
        lines.append("Signals firing HIGH (≥0.60):")
        for k in fired:
            v = signal_scores.get(k, 0.0)
            lines.append(f"  • {label_map.get(k, k)}: {v:.2f}")

    # Add specific context from signal details
    s2 = signal_details.get("s2", {})
    if s2.get("closure_rate") is not None:
        lines.append(f"  Cadence: {s2['closure_rate']:.0%} same-day releases, median gap={s2.get('median_gap_days', 0):.0f}d")

    s6 = signal_details.get("s6", {})
    if s6.get("hhi") is not None:
        lines.append(f"  ISRC HHI={s6['hhi']:.3f} (dominant company: {s6.get('dominant_share', 0):.0%} of tracks)")

    s7 = signal_details.get("s7", {})
    if s7:
        yt = s7.get("youtube_views", 0)
        apple = s7.get("apple_result_count", 0)
        lines.append(f"  Cross-platform: YouTube={yt:,} views, Apple Music={apple} results")

    if low:
        lines.append("Signals NOT firing (< 0.40):")
        for k in low:
            v = signal_scores.get(k, 0.0)
            lines.append(f"  ○ {label_map.get(k, k)}: {v:.2f}")

    if missing:
        lines.append(f"Signals unavailable: {', '.join(label_map.get(k, k) for k in missing)}")

    return "\n".join(lines)


def _empty_verdict(artist_id: str, artist_name: str,
                   signal_scores: dict, weights: dict) -> dict:
    return {
        "artist_id": artist_id,
        "artist_name": artist_name,
        "signal_scores": signal_scores,
        "signal_details": {},
        "overall_score": 0.0,
        "verdict": "UNKNOWN",
        "confidence": 0.0,
        "signals_computed": 0,
        "signals_total": len(signal_scores),
        "explanation": f"No signals could be computed for {artist_name}.",
        "weights_used": weights,
    }


# ── GNN-backed verdict ────────────────────────────────────────────────────────

def compute_verdict_gnn(
    artist_id: str,
    artist_name: str | None = None,
    track_name: str | None = None,
    gnn_weight: float = 0.6,
    rule_weight: float = 0.4,
    neo4j: Neo4jClient | None = None,
    run_s7: bool = False,
) -> dict:
    """
    Compute a hybrid verdict combining the trained GAT model with the
    rule-based signal scorer.

    Parameters
    ----------
    artist_id   : Spotify artist ID
    artist_name : Display name (resolved from Neo4j if None)
    track_name  : Optional track for cross-platform check
    gnn_weight  : Weight given to GNN score in combined verdict (default 0.6)
    rule_weight : Weight given to rule-based score (default 0.4)
    neo4j       : Neo4jClient instance (created if None)
    run_s7      : Whether to run Signal 7 in the rule-based component

    Returns
    -------
    dict with keys:
        rule_based_score, gnn_score, combined_score,
        verdict_label, confidence, explanation,
        signal_scores (from rule-based), gnn_available (bool)
    """
    from pathlib import Path
    import json as _json

    # ── Step 1: Rule-based verdict (always runs) ──────────────────────────────
    rule_result = compute_verdict(
        artist_id=artist_id,
        artist_name=artist_name,
        track_name=track_name,
        neo4j=neo4j,
        run_s7=run_s7,
    )
    rule_score = rule_result["overall_score"]
    artist_name = rule_result["artist_name"]

    # ── Step 2: GNN inference (requires trained model) ────────────────────────
    gnn_score = None
    gnn_available = False

    try:
        import torch
        from src.models.gnn_detector import GhostDetectorGAT
        from src.models.dataset_builder import FEATURE_NAMES, _EX5, _EX4_HHI

        ROOT = Path(__file__).resolve().parents[2]
        PROCESSED = ROOT / "data" / "processed"
        model_path = PROCESSED / "gat_model.pt"
        meta_path = PROCESSED / "gnn_dataset_meta.json"

        if not model_path.exists():
            logger.warning("GAT model not found at data/processed/gat_model.pt — skipping GNN inference")
            raise FileNotFoundError("gat_model.pt not found")

        meta = _json.loads(meta_path.read_text())
        col_min = torch.tensor(meta["col_min"], dtype=torch.float)
        col_max = torch.tensor(meta["col_max"], dtype=torch.float)
        col_range = (col_max - col_min).clamp(min=1e-8)

        # Build feature vector for this artist
        ex5 = _EX5.get(artist_name, {})
        hhi = _EX4_HHI.get(artist_name, 0.15)

        raw_feats = [
            float(ex5.get("track_count", rule_result.get("signal_details", {}).get("s4", {}).get("track_count", 20))),
            float(ex5.get("closure_rate", rule_result.get("signal_details", {}).get("s2", {}).get("closure_rate", 0.05))),
            float(ex5.get("tracks_per_day", 0.01)),
            float(hhi),
            0.0,                # total_variance — Kaggle miss for niche genre
            180_000.0,          # mean_duration_ms — genre default
            float(ex5.get("isrc_prefix_count", 2)),
            1.0,                # genre_count — mono-genre assumed
        ]

        x_raw = torch.tensor(raw_feats, dtype=torch.float).unsqueeze(0)
        x_norm = (x_raw - col_min) / col_range

        # Isolated inference: no graph edges (self-loop only)
        edge_index = torch.tensor([[0], [0]], dtype=torch.long)

        in_ch = len(FEATURE_NAMES)
        model = GhostDetectorGAT(in_ch)
        model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
        model.eval()

        with torch.no_grad():
            prob = model(x_norm, edge_index)
        gnn_score = float(prob[0])
        gnn_available = True
        logger.info(f"GNN score for {artist_name}: {gnn_score:.4f}")

    except Exception as e:
        logger.warning(f"GNN inference failed for {artist_name}: {e}")

    # ── Step 3: Combine scores ────────────────────────────────────────────────
    if gnn_available and gnn_score is not None:
        combined = gnn_weight * gnn_score + rule_weight * rule_score
    else:
        combined = rule_score

    verdict_label = _verdict_label(combined)

    explanation_parts = [rule_result["explanation"]]
    if gnn_available:
        explanation_parts.append(
            f"GNN score: {gnn_score:.3f} (GAT inference) · "
            f"Combined: {combined:.3f} ({gnn_weight:.0%} GNN + {rule_weight:.0%} rule-based)"
        )
    else:
        explanation_parts.append("GNN unavailable — using rule-based score only.")

    return {
        **rule_result,
        "rule_based_score": rule_score,
        "gnn_score": gnn_score,
        "gnn_available": gnn_available,
        "combined_score": round(combined, 4),
        "verdict": verdict_label,
        "overall_score": round(combined, 4),
        "explanation": "\n".join(explanation_parts),
    }
