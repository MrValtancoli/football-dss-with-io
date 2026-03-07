"""
DSS Engine — Core computation module.
Pure functions: no print, no file I/O, no side effects.
Input: dicts/lists conformi allo schema JSON.
Output: dicts/lists conformi allo schema JSON.
"""

import json
import numpy as np
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

VERSION = "1.0.0"

ATTR_KEYS = [f"A{i}" for i in range(1, 15)]


# ---------------------------------------------------------------------------
# Configuration loading
# ---------------------------------------------------------------------------

def load_strategies(path: str | Path | None = None) -> list[dict]:
    """Load strategy templates from JSON config file."""
    if path is None:
        path = Path(__file__).parent / "strategy_templates.json"
    with open(path, "r") as f:
        data = json.load(f)
    return data["strategies"]


# ---------------------------------------------------------------------------
# Distance computation
# ---------------------------------------------------------------------------

def compute_semantic_distance(vector1: list[float], vector2: list[float]) -> float:
    """Euclidean distance between two attribute vectors."""
    return float(np.sqrt(np.sum((np.array(vector1) - np.array(vector2)) ** 2)))


# ---------------------------------------------------------------------------
# Strategy vector helpers (continuous traits, not binary flags)
# ---------------------------------------------------------------------------

def _intensity_score(sv: list[float]) -> float:
    """How physically demanding is this strategy? Continuous [0,1]."""
    return (sv[3] + sv[4]) / 2.0  # A4 (Transition) + A5 (Pressing)

def _offensive_score(sv: list[float]) -> float:
    """How offensively oriented? Continuous [0,1]."""
    return (sv[0] + sv[4]) / 2.0  # A1 (Offensive) + A5 (Pressing)

def _conservative_score(sv: list[float]) -> float:
    """How conservative/defensive? Continuous [0,1]."""
    return (sv[1] + sv[9]) / 2.0  # A2 (Defensive) + A10 (Time Management)

def _complexity_score(sv: list[float]) -> float:
    """How tactically complex? Continuous [0,1]."""
    return (sv[10] + sv[2]) / 2.0  # A11 (Cohesion) + A3 (Midfield)


# ---------------------------------------------------------------------------
# Sigmoid utility
# ---------------------------------------------------------------------------

def _sigmoid(x: float, center: float = 0.0, steepness: float = 1.0) -> float:
    """Sigmoid mapped to [0, 1]. center = inflection point, steepness = slope."""
    return 1.0 / (1.0 + np.exp(-steepness * (x - center)))


def _scale_factor(value_01: float, low: float, high: float) -> float:
    """Map a [0,1] value to an arbitrary [low, high] range."""
    return low + value_01 * (high - low)


# ---------------------------------------------------------------------------
# Weight axes — each returns a factor around 1.0
# Signature: (match_conditions, strategy_vector) -> float
# Convention: < 1.0 = bonus (reduces distance), > 1.0 = penalty
# ---------------------------------------------------------------------------

def axis_energy(mc: dict, sv: list[float]) -> float:
    """
    Fatigue axis.
    High fatigue penalizes high-intensity strategies, rewards low-intensity ones.
    Always contributes — no dead zones.
    """
    fatigue = float(mc.get("fatigue_level", 0.5))
    intensity = _intensity_score(sv)

    # interaction: high fatigue * high intensity → penalty up to 1.4
    #              high fatigue * low intensity  → bonus down to 0.85
    #              low fatigue → near 1.0 regardless
    interaction = fatigue * (intensity - 0.5) * 2.0  # range ~ [-1, 1]
    return _scale_factor(_sigmoid(interaction, center=0.0, steepness=3.0), 0.85, 1.40)


def axis_urgency(mc: dict, sv: list[float]) -> float:
    """
    Time pressure + score context.
    Behind with little time → bonus for offensive, penalty for conservative.
    Ahead with little time → opposite.
    Continuous: even 35 minutes left with -1 produces a mild push.
    """
    time_left = float(mc.get("time_remaining", 45))
    score_diff = float(mc.get("score_diff", 0))

    # time_pressure: 0 at 90min, 1 at 0min — sigmoid centered at 25min
    time_pressure = _sigmoid(-time_left, center=-25.0, steepness=0.12)

    # need: positive = need to score (behind), negative = need to protect (ahead)
    need = -score_diff  # behind → positive need

    # Combined urgency signal
    urgency = time_pressure * need  # range ~ [-1, 1]

    # How it interacts with strategy:
    # urgency > 0 (need goals) → reward offensive, penalize conservative
    # urgency < 0 (protect lead) → reward conservative, penalize offensive
    offensive = _offensive_score(sv)
    conservative = _conservative_score(sv)
    strategy_direction = offensive - conservative  # positive = offensive leaning

    # Alignment: urgency and strategy direction agree → bonus
    alignment = urgency * strategy_direction  # positive = aligned
    return _scale_factor(_sigmoid(-alignment, center=0.0, steepness=4.0), 0.65, 1.50)


def axis_morale(mc: dict, sv: list[float]) -> float:
    """
    Morale axis.
    Low morale penalizes complex tactics, rewards simple/structured ones.
    High morale gives slight bonus to ambitious strategies.
    Continuous across the full morale range.
    """
    morale = float(mc.get("morale", 0.7))
    complexity = _complexity_score(sv)
    offensive = _offensive_score(sv)

    # Low morale: complexity is risky
    morale_deficit = 0.7 - morale  # positive when morale is below average
    complexity_penalty = morale_deficit * complexity * 2.0

    # High morale: offensive ambition is rewarded
    morale_surplus = morale - 0.7  # positive when morale is above average
    ambition_bonus = morale_surplus * offensive * 1.5

    signal = complexity_penalty - ambition_bonus  # positive → penalty
    return _scale_factor(_sigmoid(signal, center=0.0, steepness=3.0), 0.85, 1.25)


def axis_score_context(mc: dict, sv: list[float]) -> float:
    """
    Score differential axis (independent of time).
    Large lead → mild preference for possession/conservative.
    Large deficit → mild preference for directness.
    Small or zero diff → near neutral.
    """
    score_diff = float(mc.get("score_diff", 0))
    conservative = _conservative_score(sv)
    offensive = _offensive_score(sv)

    if score_diff > 0:
        # Ahead: reward conservative proportionally to lead size
        fit = conservative * min(score_diff, 3) / 3.0
        return _scale_factor(1.0 - fit, 0.90, 1.10)
    elif score_diff < 0:
        # Behind: reward offensive proportionally to deficit
        fit = offensive * min(abs(score_diff), 3) / 3.0
        return _scale_factor(1.0 - fit, 0.90, 1.10)
    else:
        return 1.0


# ---------------------------------------------------------------------------
# Axis registry — add/remove axes here without touching anything else
# ---------------------------------------------------------------------------

WEIGHT_AXES = [
    axis_energy,
    axis_urgency,
    axis_morale,
    axis_score_context,
]


# ---------------------------------------------------------------------------
# Main dynamic weight function
# ---------------------------------------------------------------------------

def apply_dynamic_weights(
    raw_distance: float,
    match_conditions: dict,
    strategy_vector: list[float],
) -> float:
    """
    Context-aware adjustment multiplier.
    Product of all registered weight axes, clamped to [0.4, 2.0].
    Each axis is an independent, continuous function — no dead zones.
    """
    adjustment = 1.0
    for axis_fn in WEIGHT_AXES:
        adjustment *= axis_fn(match_conditions, strategy_vector)

    adjustment = max(0.4, min(adjustment, 2.0))
    return max(0.0, raw_distance * adjustment)


# ---------------------------------------------------------------------------
# Strategy selection (single scenario)
# ---------------------------------------------------------------------------

def _normalize_min_max(values: list[float], floor: float = 0.1) -> list[float]:
    """
    Min-max normalization to [floor, 1.0].
    Floor > 0 ensures the best raw strategy still gets a nonzero
    distance that dynamic weights can meaningfully adjust.
    If all values are equal, returns uniform floor values.
    """
    v_min = min(values)
    v_max = max(values)
    if v_max - v_min < 1e-9:
        return [floor] * len(values)
    return [floor + (1.0 - floor) * (v - v_min) / (v_max - v_min) for v in values]


def evaluate_strategies(
    team_profile: dict[str, float],
    opponent_profile: dict[str, float],
    strategies: list[dict],
    match_conditions: dict,
    opponent_penalty_lambda: float = 0.5,
) -> list[dict]:
    """
    Evaluate all strategies for a single scenario.
    1. Compute combined distances (team fit + opponent penalty)
    2. Normalize to [0.1, 1.0] so relative differences are preserved
       but dynamic weights can effectively reorder the ranking
    3. Apply context-aware dynamic weights
    Returns list of dicts sorted by adjusted_distance (ascending = best fit first).
    """
    team_vector = [team_profile[k] for k in ATTR_KEYS]
    opponent_vector = [opponent_profile[k] for k in ATTR_KEYS]

    # Pass 1: compute raw combined distances
    raw_data = []
    for strategy in strategies:
        sv = strategy["vector"]
        raw_dist = compute_semantic_distance(team_vector, sv)
        opp_dist = compute_semantic_distance(opponent_vector, sv)
        
        # Aggiornato in base alla sottrazione lineare: d_comb = d_adapt(team,S) - α d_adapt(opp,S)
        combined = raw_dist - opponent_penalty_lambda * opp_dist
        
        raw_data.append({
            "strategy": strategy["name"],
            "vector": sv,
            "raw_distance": raw_dist,
            "combined_distance": combined,
            "category": strategy.get("category", "unknown"),
        })

    # Pass 2: normalize combined distances
    combined_values = [d["combined_distance"] for d in raw_data]
    normalized = _normalize_min_max(combined_values)

    # Pass 3: apply dynamic weights on normalized distances
    scores = []
    for entry, norm_dist in zip(raw_data, normalized):
        adjusted = apply_dynamic_weights(norm_dist, match_conditions, entry["vector"])
        scores.append({
            "strategy": entry["strategy"],
            "adjusted_distance": round(adjusted, 4),
            "raw_distance": round(entry["raw_distance"], 4),
            "category": entry["category"],
        })

    scores.sort(key=lambda x: x["adjusted_distance"])
    return scores


def compute_baseline(
    team_profile: dict[str, float],
    strategies: list[dict],
) -> dict:
    """
    Static baseline: team fit only, no dynamic weights, no opponent.
    Returns the best strategy_score dict.
    """
    team_vector = [team_profile[k] for k in ATTR_KEYS]
    best = None
    for strategy in strategies:
        dist = compute_semantic_distance(team_vector, strategy["vector"])
        entry = {
            "strategy": strategy["name"],
            "adjusted_distance": round(dist, 4),
            "raw_distance": round(dist, 4),
            "category": strategy.get("category", "unknown"),
        }
        if best is None or dist < best["adjusted_distance"]:
            best = entry
    return best


# ---------------------------------------------------------------------------
# Batch execution (main entry point)
# ---------------------------------------------------------------------------

def run_batch(input_data: dict, strategies: list[dict] | None = None) -> dict:
    """
    Main DSS function.
    Takes input conforming to dss_input_schema.json,
    returns output conforming to dss_output_schema.json.
    """
    if strategies is None:
        strategies = load_strategies()

    # --- Parse input ---
    team = input_data["team"]
    opponent = input_data["opponent"]
    scenarios = input_data["scenarios"]
    config = input_data.get("config", {})

    opp_lambda = config.get("opponent_penalty_lambda", 0.5)
    top_n = config.get("top_n", 5)

    # --- Build team/opponent profiles (only A1-A14 keys) ---
    team_profile = {k: team[k] for k in ATTR_KEYS}
    opponent_profile = {k: opponent[k] for k in ATTR_KEYS}

    # --- Baseline (computed once, shared across scenarios) ---
    baseline = compute_baseline(team_profile, strategies)

    # --- Evaluate each scenario ---
    results = []
    for scenario in scenarios:
        mc = scenario["match_conditions"]

        ranking_full = evaluate_strategies(
            team_profile, opponent_profile, strategies, mc,
            opponent_penalty_lambda=opp_lambda,
        )

        results.append({
            "scenario_id": scenario.get("id"),
            "scenario_label": scenario.get("label"),
            "match_conditions": mc,
            "best_strategy": ranking_full[0],
            "baseline_strategy": baseline,
            "ranking": ranking_full[:top_n],
        })

    # --- Build output ---
    return {
        "meta": {
            "version": VERSION,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "config_used": {
                "opponent_penalty_lambda": opp_lambda,
                "top_n": top_n,
            },
            "team_name": team.get("name"),
            "opponent_name": opponent.get("name"),
            "total_scenarios": len(scenarios),
        },
        "results": results,
    }