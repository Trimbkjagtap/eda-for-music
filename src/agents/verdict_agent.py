"""
Verdict Agent — Synthesizes all signals into a final assessment.

Takes signal scores from all other agents, combines them using
weighted scoring, and produces a human-readable verdict.
"""
from __future__ import annotations

import json

import numpy as np
from crewai import Agent
from loguru import logger
from pydantic import BaseModel, Field

try:
    from crewai.tools import BaseTool
except Exception:
    try:
        from crewai.tools.base_tool import BaseTool
    except Exception:
        try:
            from crewai import BaseTool
        except Exception:
            class BaseTool(BaseModel):
                name: str = "tool"
                description: str = "compat tool"
                args_schema: type[BaseModel] | None = None

                def _run(self, *args, **kwargs) -> str:
                    raise NotImplementedError("BaseTool unavailable in current CrewAI install")

# Known ghost/organic baselines for comparison
_KNOWN_GHOSTS = {
    "Relaxing White Noise":   {"s2": 0.42, "s4": 0.76, "s6": 0.72},
    "Meditation Relax Club":  {"s2": 0.59, "s4": 0.54, "s6": 0.56},
    "Calmo":                  {"s2": 0.23, "s4": 0.14, "s6": 0.45},
}
_KNOWN_ORGANIC = {
    "Nils Frahm": {"s2": 0.02, "s4": 0.00, "s6": 0.00},
}


# ── Tool input schemas ────────────────────────────────────────────────────────

class _ScoresInput(BaseModel):
    signal_scores: dict = Field(
        description="Dict of signal key to float score, e.g. {'s2_cadence_sync': 0.42}"
    )
    artist_name: str = Field(default="Unknown Artist", description="Artist display name")


class _CompareInput(BaseModel):
    scores: dict = Field(description="Signal scores dict for the artist being analyzed")
    artist_name: str = Field(default="Unknown Artist")


# ── Tools ─────────────────────────────────────────────────────────────────────

class ComputeVerdictTool(BaseTool):
    name: str = "compute_verdict"
    description: str = (
        "Combine signal scores into a weighted verdict. "
        "Returns verdict_label (LIKELY_GHOST/SUSPICIOUS/LIKELY_ORGANIC), "
        "overall_score, confidence, and explanation."
    )
    args_schema: type[BaseModel] = _ScoresInput

    def _run(self, signal_scores: dict, artist_name: str = "Unknown Artist") -> str:
        from src.signals.verdict import DEFAULT_WEIGHTS, _verdict_label

        available = {k: v for k, v in signal_scores.items() if v is not None}
        if not available:
            return json.dumps({
                "verdict": "UNKNOWN",
                "overall_score": 0.0,
                "confidence": 0.0,
                "explanation": "No signal scores available.",
            })

        total_weight = sum(DEFAULT_WEIGHTS.get(k, 1/7) for k in available)
        weighted_sum = sum(
            v * DEFAULT_WEIGHTS.get(k, 1/7) / total_weight
            for k, v in available.items()
        )
        overall_score = float(weighted_sum)
        confidence = len(available) / 7

        verdict_label = _verdict_label(overall_score)
        explanation = _build_explanation(artist_name, verdict_label, overall_score, signal_scores)

        return json.dumps({
            "verdict": verdict_label,
            "overall_score": round(overall_score, 4),
            "confidence": round(confidence, 2),
            "signals_computed": len(available),
            "explanation": explanation,
        })


class GenerateExplanationTool(BaseTool):
    name: str = "generate_explanation"
    description: str = (
        "Generate a natural language explanation of the fraud assessment "
        "based on the signal scores and verdict."
    )
    args_schema: type[BaseModel] = _ScoresInput

    def _run(self, signal_scores: dict, artist_name: str = "Unknown Artist") -> str:
        s2 = signal_scores.get("s2_cadence_sync") or 0
        s4 = signal_scores.get("s4_follower_ratio") or 0
        s6 = signal_scores.get("s6_graph_density") or 0
        s7 = signal_scores.get("s7_cross_platform") or 0

        explanation = _build_explanation(artist_name, "ANALYSIS", 0, signal_scores)
        return json.dumps({"explanation": explanation})


class CompareToBaselineTool(BaseTool):
    name: str = "compare_to_baseline"
    description: str = (
        "Compare artist scores to known ghost and organic baselines. "
        "Returns similarity to each reference artist."
    )
    args_schema: type[BaseModel] = _CompareInput

    def _run(self, scores: dict, artist_name: str = "Unknown Artist") -> str:
        comparisons = {}
        for ref_name, ref_scores in {**_KNOWN_GHOSTS, **_KNOWN_ORGANIC}.items():
            # Compare available keys
            common_keys = set(scores.keys()) & set(ref_scores.keys())
            # Normalize key names (drop "s2_" prefix etc)
            def normalize(d):
                return {k.split("_")[0]: v for k, v in d.items() if v is not None}
            a = normalize(scores)
            b = normalize(ref_scores)
            common = set(a.keys()) & set(b.keys())
            if common:
                diffs = [abs(a[k] - b[k]) for k in common]
                similarity = 1.0 - float(np.mean(diffs))
            else:
                similarity = 0.0
            label = "ghost" if ref_name in _KNOWN_GHOSTS else "organic"
            comparisons[ref_name] = {"similarity": round(similarity, 3), "label": label}

        # Find most similar
        if comparisons:
            closest = max(comparisons, key=lambda k: comparisons[k]["similarity"])
        else:
            closest = "unknown"

        return json.dumps({
            "artist": artist_name,
            "comparisons": comparisons,
            "most_similar_to": closest,
            "most_similar_label": comparisons.get(closest, {}).get("label", "unknown"),
        })


# ── Pure Python verdict synthesis (no LLM needed) ────────────────────────────

def synthesize_verdict(
    artist_id: str,
    all_signal_scores: dict,
    signal_details: dict | None = None,
    artist_name: str = "Unknown",
) -> dict:
    """
    Combine all signal scores into a final verdict dict.
    This is the pure-Python path used by crew.py pipeline.
    """
    from src.signals.verdict import DEFAULT_WEIGHTS, _verdict_label

    available = {k: v for k, v in all_signal_scores.items() if v is not None}

    if not available:
        return {
            "artist_id": artist_id,
            "artist_name": artist_name,
            "signal_scores": all_signal_scores,
            "overall_score": 0.0,
            "verdict": "UNKNOWN",
            "confidence": 0.0,
            "explanation": f"No signal data available for {artist_name}.",
        }

    total_weight = sum(DEFAULT_WEIGHTS.get(k, 1/7) for k in available)
    weighted_sum = sum(
        v * DEFAULT_WEIGHTS.get(k, 1/7) / total_weight
        for k, v in available.items()
    )
    overall_score = float(weighted_sum)
    confidence = len(available) / 7
    verdict_label = _verdict_label(overall_score)

    explanation = _build_explanation(artist_name, verdict_label, overall_score, all_signal_scores)

    logger.info(
        f"VERDICT {artist_name}: {verdict_label} "
        f"(score={overall_score:.3f}, conf={confidence:.0%})"
    )

    return {
        "artist_id": artist_id,
        "artist_name": artist_name,
        "signal_scores": all_signal_scores,
        "overall_score": round(overall_score, 4),
        "verdict": verdict_label,
        "confidence": round(confidence, 2),
        "signals_computed": len(available),
        "signals_total": 7,
        "explanation": explanation,
    }


# ── CrewAI Agent definition ───────────────────────────────────────────────────

def build_verdict_agent() -> Agent:
    return Agent(
        role="Fraud Detection Verdict Analyst",
        goal="Synthesize all evidence from other agents into a clear, justified fraud assessment",
        backstory=(
            "Senior analyst experienced in combining multiple evidence streams into coherent "
            "threat assessments, with deep expertise in music industry fraud patterns. "
            "Weighs signal reliability, data availability, and contextual baselines to "
            "produce calibrated ghost probability scores."
        ),
        tools=[
            ComputeVerdictTool(),
            GenerateExplanationTool(),
            CompareToBaselineTool(),
        ],
        verbose=True,
        allow_delegation=False,
    )


# ── helpers ───────────────────────────────────────────────────────────────────

def _build_explanation(name: str, verdict: str, score: float, scores: dict) -> str:
    s2 = scores.get("s2_cadence_sync")
    s4 = scores.get("s4_follower_ratio")
    s6 = scores.get("s6_graph_density")
    s7 = scores.get("s7_cross_platform")

    lines = [f"{name}: {verdict} (overall_score={score:.3f})"]

    if s2 is not None:
        if s2 >= 0.6:
            lines.append(f"  HIGH cadence suspicion ({s2:.2f}): bulk-upload fingerprint detected")
        elif s2 >= 0.4:
            lines.append(f"  MEDIUM cadence suspicion ({s2:.2f}): irregular release clustering")
        else:
            lines.append(f"  LOW cadence suspicion ({s2:.2f}): natural release spacing")

    if s4 is not None:
        if s4 >= 0.6:
            lines.append(f"  HIGH catalog density ({s4:.2f}): {'>0.15'} tracks/day — suspicious")
        elif s4 >= 0.35:
            lines.append(f"  MEDIUM catalog density ({s4:.2f})")
        else:
            lines.append(f"  LOW catalog density ({s4:.2f}): organic production rate")

    if s6 is not None:
        if s6 >= 0.6:
            lines.append(f"  HIGH ISRC concentration ({s6:.2f}): monopolistic production company")
        elif s6 >= 0.35:
            lines.append(f"  MEDIUM ISRC concentration ({s6:.2f})")
        else:
            lines.append(f"  LOW ISRC concentration ({s6:.2f}): diverse production sources")

    if s7 is not None:
        if s7 >= 0.7:
            lines.append(f"  HIGH cross-platform invisibility ({s7:.2f}): not on YouTube/Apple Music")
        elif s7 >= 0.4:
            lines.append(f"  MEDIUM cross-platform score ({s7:.2f})")
        else:
            lines.append(f"  LOW cross-platform score ({s7:.2f}): present on other platforms")

    missing = [k for k, v in scores.items() if v is None]
    if missing:
        lines.append(f"  Data unavailable: {', '.join(missing)}")

    return "\n".join(lines)
