"""
DSS Engine — Core computation module.
Pure functions: no print, no file I/O, no side effects.
Input: dicts/lists conformi allo schema JSON.
Output: dicts/lists conformi allo schema JSON.

Weight computation follows Section 3.6.2 of the paper:
  - Equations (5)–(7):  energy-based multipliers (m5, m10, m13)
  - Equations (8)–(11): gap-based multipliers (m1, m2, m6, m11)
  - Equations (12)–(13): time-pressure multipliers (m4, m1 additive)
  - Clamp to [0.3, 2.5], normalize to sum = 14
"""

import json
import numpy as np
from datetime import datetime, timezone
from pathlib import Path

VERSION = "2.0.0"

ATTR_KEYS = [f"A{i}" for i in range(1, 15)]

# ---------------------------------------------------------------------------
# Default parameters (Table 5 in paper)
# ---------------------------------------------------------------------------

DEFAULT_PARAMS = {
    "tau_e": 0.50,    # Energy threshold: fatigue becomes salient below this
    "gamma_e": 1.50,  # Energy sensitivity
    "gamma_g": 1.00,  # Gap sensitivity
    "tau_t": 0.25,    # Time threshold: urgency triggers in final quarter
    "gamma_t": 2.00,  # Urgency sensitivity
    "m_min": 0.30,    # Multiplier floor
    "m_max": 2.50,    # Multiplier ceiling
}


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
# Weight vector computation (Section 3.6.2, Algorithm 1)
# ---------------------------------------------------------------------------

def compute_weight_vector(
    team_profile: dict[str, float],
    opponent_profile: dict[str, float],
    match_conditions: dict,
    params: dict | None = None,
) -> list[float]:
    """
    Compute the 14-dimensional weight vector w from match context.

    Follows Algorithm 1 and equations (5)–(13) from the paper.

    Inputs from match_conditions:
      - fatigue_level: used as proxy for residual energy via e = 1 - fatigue_level
        (future: IoT sensors will provide this directly; invert with e = 1 - fatigue_level)
      - time_remaining: minutes remaining [0, 90], converted to t = time_remaining / 90
      - score_diff: integer, mapped to ternary s ∈ {-1, 0, +1}

    Inputs from profiles:
      - team A12, A13 (technical/physical base)
      - opponent A12, A13 (for gap computation)

    Returns: list of 14 floats (w1..w14), normalized to sum = 14.
    """
    p = {**DEFAULT_PARAMS, **(params or {})}

    # --- Initialize all multipliers to 1 ---
    m = [1.0] * 14  # m[0] = m1 (A1), ..., m[13] = m14 (A14)

    # --- Context indicators ---

    # Energy: e = residual energy ∈ [0,1] (high = fresh)
    # fatigue_level is inverted: high = tired, so e = 1 - fatigue_level
    # NOTE: when IoT data arrives, fatigue_level will come from heart rate monitors
    # and the inversion e = 1 - fatigue_level maps it to the paper's convention.
    # For now, fatigue_level in match_conditions serves as per-scenario proxy for A8.
    e = 1.0 - float(match_conditions.get("fatigue_level", 0.5))
    delta_e = max(0.0, p["tau_e"] - e)

    # Technical and physical gaps
    delta_tech = team_profile["A12"] - opponent_profile["A12"]
    delta_phys = team_profile["A13"] - opponent_profile["A13"]

    # Time pressure: t ∈ [0,1] fraction remaining (1 = kickoff, 0 = final whistle)
    t = float(match_conditions.get("time_remaining", 45)) / 90.0

    # Score state: ternary s ∈ {-1, 0, +1}
    raw_score_diff = int(match_conditions.get("score_diff", 0))
    if raw_score_diff > 0:
        s = 1
    elif raw_score_diff < 0:
        s = -1
    else:
        s = 0

    # Time pressure indicator: δt = max(0, τt - t) · 1[s ≤ 0]
    indicator_not_winning = 1.0 if s <= 0 else 0.0
    delta_t = max(0.0, p["tau_t"] - t) * indicator_not_winning

    # --- Energy-based adjustments (eq. 5–7) ---
    m[4]  = 1.0 - p["gamma_e"] * delta_e          # m5:  reduce High Press Capability
    m[9]  = 1.0 + p["gamma_e"] * delta_e          # m10: increase Time Management
    m[12] = 1.0 - 0.5 * p["gamma_e"] * delta_e    # m13: reduce Physical Base

    # --- Gap-based adjustments (eq. 8–11) ---
    m[1]  = 1.0 + p["gamma_g"] * max(0.0, -delta_tech)    # m2:  increase Defensive Strength if technically inferior
    m[10] = 1.0 + p["gamma_g"] * max(0.0, -delta_phys)    # m11: increase Tactical Cohesion if physically inferior
    m[0]  = 1.0 - 0.5 * p["gamma_g"] * max(0.0, -delta_tech)  # m1: reduce Offensive Strength if outmatched
    m[5]  = 1.0 - 0.5 * p["gamma_g"] * max(0.0, -delta_phys)  # m6: reduce Width Utilization if outmatched

    # --- Time pressure adjustments (eq. 12–13) ---
    m[3]  = 1.0 + p["gamma_t"] * delta_t           # m4: increase Transition Speed
    m[0]  = m[0] + p["gamma_t"] * delta_t           # m1: further increase Offensive Strength (additive)

    # --- Clamp all multipliers (Section 3.6.2) ---
    for j in range(14):
        m[j] = max(p["m_min"], min(m[j], p["m_max"]))

    # --- Normalize to sum = 14 (preserving baseline where all w_j = 1) ---
    total = sum(m)
    w = [14.0 * mj / total for mj in m]

    return w


# ---------------------------------------------------------------------------
# Distance computation
# ---------------------------------------------------------------------------

def compute_semantic_distance(
    vector1: list[float],
    vector2: list[float],
    weights: list[float] | None = None,
) -> float:
    """
    Weighted Euclidean distance: d = sqrt( sum( w_j * (x_j - y_j)^2 ) )
    If weights is None, falls back to standard Euclidean (all w_j = 1).
    """
    v1 = np.array(vector1)
    v2 = np.array(vector2)
    if weights is None:
        return float(np.sqrt(np.sum((v1 - v2) ** 2)))
    w = np.array(weights)
    return float(np.sqrt(np.sum(w * (v1 - v2) ** 2)))


# ---------------------------------------------------------------------------
# Strategy selection (single scenario)
# ---------------------------------------------------------------------------

def evaluate_strategies(
    team_profile: dict[str, float],
    opponent_profile: dict[str, float],
    strategies: list[dict],
    match_conditions: dict,
    opponent_penalty_lambda: float = 0.5,
) -> list[dict]:
    """
    Evaluate all strategies for a single scenario.

    Follows Algorithm 1 from the paper:
    1. Compute weight vector w from match context
    2. For each strategy: d_adapt(team, S; w) - α · d_adapt(opp, S; w)
    3. Sort by ascending combined distance (best fit first)

    No post-hoc normalization — the weight vector inside the distance
    is the sole mechanism for context adaptation.
    """
    team_vector = [team_profile[k] for k in ATTR_KEYS]
    opponent_vector = [opponent_profile[k] for k in ATTR_KEYS]

    # Compute weight vector once per scenario
    w = compute_weight_vector(team_profile, opponent_profile, match_conditions)

    # Compute raw (unweighted) baseline distance for diagnostics
    scores = []
    for strategy in strategies:
        sv = strategy["vector"]

        # Adapted (weighted) distances
        d_team = compute_semantic_distance(team_vector, sv, w)
        d_opp = compute_semantic_distance(opponent_vector, sv, w)

        # Combined distance: linear subtraction (Section 4.2)
        d_comb = d_team - opponent_penalty_lambda * d_opp

        # Raw (unweighted) distance for baseline comparison
        d_raw = compute_semantic_distance(team_vector, sv)

        scores.append({
            "strategy": strategy["name"],
            "adjusted_distance": round(d_comb, 4),
            "raw_distance": round(d_raw, 4),
            "category": strategy.get("category", "unknown"),
        })

    scores.sort(key=lambda x: x["adjusted_distance"])
    return scores


def compute_baseline(
    team_profile: dict[str, float],
    strategies: list[dict],
) -> dict:
    """
    Static baseline: team fit only, no dynamic weights, no opponent.
    Uses unweighted Euclidean distance.
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

        # Apply per-scenario profile overrides (A7, A8, A9)
        overrides = scenario.get("profile_overrides") or {}
        effective_team_profile = {**team_profile, **{k: v for k, v in overrides.items() if v is not None}}

        ranking_full = evaluate_strategies(
            effective_team_profile, opponent_profile, strategies, mc,
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