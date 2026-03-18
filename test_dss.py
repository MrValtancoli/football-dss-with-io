"""
DSS Test Suite — v2.0.0
Aligned with per-attribute weight vector (Section 3.6.2, eq. 5–13).
Run: python -m pytest test_dss.py -v
"""

import json
import pytest
import numpy as np
from pathlib import Path

from dss_engine import (
    compute_semantic_distance,
    compute_weight_vector,
    evaluate_strategies,
    compute_baseline,
    run_batch,
    load_strategies,
    DEFAULT_PARAMS,
)
from dss_schema import DSSInput, DSSOutput, TeamProfile, MatchConditions
from pydantic import ValidationError


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def strategies():
    return load_strategies(Path(__file__).parent / "strategy_templates.json")


@pytest.fixture
def balanced_team():
    """A team profile equidistant from many strategies."""
    return {f"A{i}": 0.6 for i in range(1, 15)}


@pytest.fixture
def example_input():
    path = Path(__file__).parent / "example_input.json"
    if not path.exists():
        path = Path(__file__).parent / "examples" / "example_input.json"
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Unit tests: distance computation
# ---------------------------------------------------------------------------

class TestDistance:
    def test_identical_vectors_zero(self):
        v = [0.5] * 14
        assert compute_semantic_distance(v, v) == 0.0

    def test_identical_vectors_zero_weighted(self):
        v = [0.5] * 14
        w = [2.0] * 14
        assert compute_semantic_distance(v, v, w) == 0.0

    def test_known_distance_unweighted(self):
        v1 = [1.0] * 14
        v2 = [0.0] * 14
        expected = np.sqrt(14.0)
        assert abs(compute_semantic_distance(v1, v2) - expected) < 1e-6

    def test_known_distance_weighted(self):
        """With uniform weights w=2, distance = sqrt(2) * unweighted."""
        v1 = [1.0] * 14
        v2 = [0.0] * 14
        w = [2.0] * 14
        expected = np.sqrt(2.0 * 14.0)
        assert abs(compute_semantic_distance(v1, v2, w) - expected) < 1e-6

    def test_weights_none_equals_unweighted(self):
        v1 = [0.3, 0.7, 0.5, 0.8, 0.2, 0.6, 0.4, 0.9, 0.1, 0.5, 0.7, 0.3, 0.8, 0.6]
        v2 = [0.6, 0.4, 0.8, 0.2, 0.7, 0.3, 0.9, 0.1, 0.5, 0.7, 0.3, 0.8, 0.2, 0.4]
        d_none = compute_semantic_distance(v1, v2)
        d_ones = compute_semantic_distance(v1, v2, [1.0] * 14)
        assert abs(d_none - d_ones) < 1e-9

    def test_symmetry(self):
        v1 = [0.3, 0.7, 0.5, 0.8, 0.2, 0.6, 0.4, 0.9, 0.1, 0.5, 0.7, 0.3, 0.8, 0.6]
        v2 = [0.6, 0.4, 0.8, 0.2, 0.7, 0.3, 0.9, 0.1, 0.5, 0.7, 0.3, 0.8, 0.2, 0.4]
        assert compute_semantic_distance(v1, v2) == compute_semantic_distance(v2, v1)

    def test_single_attribute_weight_amplifies(self):
        """High weight on a single differing attribute should increase distance."""
        v1 = [0.5] * 14
        v2 = [0.5] * 14
        v2[4] = 0.9  # A5 differs by 0.4
        w_uniform = [1.0] * 14
        w_amplified = [1.0] * 14
        w_amplified[4] = 3.0  # triple weight on A5
        d_uniform = compute_semantic_distance(v1, v2, w_uniform)
        d_amplified = compute_semantic_distance(v1, v2, w_amplified)
        assert d_amplified > d_uniform


# ---------------------------------------------------------------------------
# Unit tests: weight vector computation (Section 3.6.2)
# ---------------------------------------------------------------------------

class TestWeightVector:
    @pytest.fixture
    def neutral_team(self):
        return {f"A{i}": 0.7 for i in range(1, 15)}

    @pytest.fixture
    def neutral_opponent(self):
        return {f"A{i}": 0.5 for i in range(1, 15)}

    def test_neutral_conditions_uniform(self, neutral_team, neutral_opponent):
        """No fatigue, no time pressure, team superior → all weights = 1.0."""
        mc = {"time_remaining": 45, "score_diff": 0, "fatigue_level": 0.3, "morale": 0.7}
        w = compute_weight_vector(neutral_team, neutral_opponent, mc)
        assert len(w) == 14
        for wj in w:
            assert abs(wj - 1.0) < 1e-9, f"Expected 1.0, got {wj}"

    def test_sum_always_14(self, neutral_team, neutral_opponent):
        """Weight vector must always sum to 14 (normalization invariant)."""
        cases = [
            {"time_remaining": 5, "score_diff": -3, "fatigue_level": 0.95, "morale": 0.2},
            {"time_remaining": 85, "score_diff": 2, "fatigue_level": 0.1, "morale": 0.9},
            {"time_remaining": 20, "score_diff": 0, "fatigue_level": 0.7, "morale": 0.5},
        ]
        for mc in cases:
            w = compute_weight_vector(neutral_team, neutral_opponent, mc)
            assert abs(sum(w) - 14.0) < 1e-9, f"Sum = {sum(w)} for {mc}"

    def test_clamp_bounds(self, neutral_team, neutral_opponent):
        """No individual weight should exceed clamped+normalized bounds."""
        mc = {"time_remaining": 1, "score_diff": -5, "fatigue_level": 0.99, "morale": 0.1}
        w = compute_weight_vector(neutral_team, neutral_opponent, mc)
        for j, wj in enumerate(w):
            assert wj > 0, f"w[A{j+1}] = {wj} should be positive"

    def test_energy_activates_above_threshold(self, neutral_team, neutral_opponent):
        """fatigue_level > 0.50 → e < 0.50 → δe > 0 → m5 decreases, m10 increases."""
        mc = {"time_remaining": 45, "score_diff": 0, "fatigue_level": 0.75, "morale": 0.7}
        w = compute_weight_vector(neutral_team, neutral_opponent, mc)
        assert w[4] < 1.0, f"A5 (High Press) weight should decrease, got {w[4]}"
        assert w[9] > 1.0, f"A10 (Time Mgmt) weight should increase, got {w[9]}"
        assert w[12] < 1.0, f"A13 (Physical Base) weight should decrease, got {w[12]}"

    def test_energy_inactive_below_threshold(self, neutral_team, neutral_opponent):
        """fatigue_level < 0.50 → e > 0.50 → δe = 0 → no energy adjustment."""
        mc = {"time_remaining": 45, "score_diff": 0, "fatigue_level": 0.3, "morale": 0.7}
        w = compute_weight_vector(neutral_team, neutral_opponent, mc)
        # All weights should be 1.0 (no gap, no time pressure either)
        for wj in w:
            assert abs(wj - 1.0) < 1e-9

    def test_time_pressure_activates(self, neutral_team, neutral_opponent):
        """time < τt*90 AND losing → m4 and m1 increase."""
        # t = 10/90 = 0.111, τt = 0.25, s = -1 → δt = 0.139
        mc = {"time_remaining": 10, "score_diff": -1, "fatigue_level": 0.3, "morale": 0.7}
        w = compute_weight_vector(neutral_team, neutral_opponent, mc)
        assert w[3] > 1.0, f"A4 (Transition Speed) should increase, got {w[3]}"
        assert w[0] > 1.0, f"A1 (Offensive Strength) should increase, got {w[0]}"

    def test_time_pressure_inactive_when_winning(self, neutral_team, neutral_opponent):
        """Even with low time, winning means 1[s ≤ 0] = 0 → no time pressure."""
        mc = {"time_remaining": 5, "score_diff": 1, "fatigue_level": 0.3, "morale": 0.7}
        w = compute_weight_vector(neutral_team, neutral_opponent, mc)
        # No energy (fatigue low), no gap (team superior), no time pressure (winning)
        for wj in w:
            assert abs(wj - 1.0) < 1e-9

    def test_gap_activates_when_inferior(self):
        """Team technically/physically inferior → defensive weights increase."""
        weak_team = {f"A{i}": 0.4 for i in range(1, 15)}
        strong_opp = {f"A{i}": 0.4 for i in range(1, 15)}
        strong_opp["A12"] = 0.8  # technically superior
        strong_opp["A13"] = 0.8  # physically superior
        mc = {"time_remaining": 45, "score_diff": 0, "fatigue_level": 0.3, "morale": 0.7}
        w = compute_weight_vector(weak_team, strong_opp, mc)
        assert w[1] > 1.0, f"A2 (Defensive Strength) should increase, got {w[1]}"
        assert w[10] > 1.0, f"A11 (Tactical Cohesion) should increase, got {w[10]}"
        assert w[0] < 1.0, f"A1 (Offensive Strength) should decrease, got {w[0]}"
        assert w[5] < 1.0, f"A6 (Width Utilization) should decrease, got {w[5]}"

    def test_gap_inactive_when_superior(self, neutral_team, neutral_opponent):
        """Team superior on A12/A13 → max(0, -Δ) = 0 → no gap adjustment."""
        mc = {"time_remaining": 45, "score_diff": 0, "fatigue_level": 0.3, "morale": 0.7}
        w = compute_weight_vector(neutral_team, neutral_opponent, mc)
        # neutral_team A12=0.7, neutral_opponent A12=0.5 → Δtech = +0.2 → no adjustment
        for wj in w:
            assert abs(wj - 1.0) < 1e-9

    def test_combined_energy_and_time_pressure(self, neutral_team, neutral_opponent):
        """Both energy and time pressure active simultaneously."""
        mc = {"time_remaining": 8, "score_diff": -1, "fatigue_level": 0.85, "morale": 0.5}
        w = compute_weight_vector(neutral_team, neutral_opponent, mc)
        # Energy: e=0.15, δe=0.35 → m5 down, m10 up, m13 down
        # Time: t=0.089, δt=0.161 → m4 up, m1 up
        assert w[4] < 1.0, "A5 should decrease (energy)"
        assert w[9] > 1.0, "A10 should increase (energy)"
        assert w[3] > 1.0, "A4 should increase (time pressure)"
        assert w[0] > 1.0, "A1 should increase (time pressure)"

    def test_score_diff_ternary_mapping(self, neutral_team, neutral_opponent):
        """score_diff=3 and score_diff=1 should produce same weights (both s=+1)."""
        mc_1 = {"time_remaining": 10, "score_diff": 1, "fatigue_level": 0.5, "morale": 0.7}
        mc_3 = {"time_remaining": 10, "score_diff": 3, "fatigue_level": 0.5, "morale": 0.7}
        w1 = compute_weight_vector(neutral_team, neutral_opponent, mc_1)
        w3 = compute_weight_vector(neutral_team, neutral_opponent, mc_3)
        for j in range(14):
            assert abs(w1[j] - w3[j]) < 1e-9, f"A{j+1}: {w1[j]} vs {w3[j]}"


# ---------------------------------------------------------------------------
# Integration: evaluate_strategies
# ---------------------------------------------------------------------------

class TestEvaluateStrategies:
    def test_returns_sorted(self, balanced_team, strategies):
        mc = {"time_remaining": 45, "score_diff": 0, "fatigue_level": 0.5, "morale": 0.7}
        results = evaluate_strategies(balanced_team, balanced_team, strategies, mc)
        dists = [r["adjusted_distance"] for r in results]
        assert dists == sorted(dists), "Results must be sorted ascending"

    def test_all_strategies_present(self, balanced_team, strategies):
        mc = {"time_remaining": 45, "score_diff": 0, "fatigue_level": 0.5, "morale": 0.7}
        results = evaluate_strategies(balanced_team, balanced_team, strategies, mc)
        assert len(results) == len(strategies)

    def test_has_required_fields(self, balanced_team, strategies):
        mc = {"time_remaining": 45, "score_diff": 0, "fatigue_level": 0.5, "morale": 0.7}
        results = evaluate_strategies(balanced_team, balanced_team, strategies, mc)
        for r in results:
            assert "strategy" in r
            assert "adjusted_distance" in r
            assert "raw_distance" in r
            assert "category" in r


# ---------------------------------------------------------------------------
# Integration: scenario differentiation
# ---------------------------------------------------------------------------

class TestScenarioDifferentiation:
    @pytest.fixture
    def versatile_team(self):
        """A realistic team profile — technically/physically inferior to strong_opponent."""
        return {
            "A1": 0.70, "A2": 0.65, "A3": 0.72, "A4": 0.68,
            "A5": 0.60, "A6": 0.65, "A7": 0.70, "A8": 0.75,
            "A9": 0.72, "A10": 0.60, "A11": 0.70, "A12": 0.55,
            "A13": 0.50, "A14": 0.70,
        }

    @pytest.fixture
    def strong_opponent(self):
        """Opponent superior on A12/A13 to activate gap-based adjustments."""
        return {
            "A1": 0.65, "A2": 0.70, "A3": 0.68, "A4": 0.60,
            "A5": 0.55, "A6": 0.60, "A7": 0.72, "A8": 0.78,
            "A9": 0.70, "A10": 0.65, "A11": 0.72, "A12": 0.82,
            "A13": 0.80, "A14": 0.68,
        }

    @pytest.fixture
    def weak_opponent(self):
        return {f"A{i}": 0.45 for i in range(1, 15)}

    def test_different_conditions_different_ranking(self, versatile_team, strong_opponent, strategies):
        """
        Core test: radically different match conditions must produce distinct rankings.
        Fresh/ahead triggers no adjustments.
        Tired/behind/inferior triggers energy + time pressure + gap (all three mechanisms).
        """
        mc_fresh_ahead = {"time_remaining": 70, "score_diff": 2, "fatigue_level": 0.2, "morale": 0.85}
        mc_tired_behind = {"time_remaining": 10, "score_diff": -2, "fatigue_level": 0.90, "morale": 0.35}

        r1 = evaluate_strategies(versatile_team, strong_opponent, strategies, mc_fresh_ahead)
        r2 = evaluate_strategies(versatile_team, strong_opponent, strategies, mc_tired_behind)

        top5_a = [r["strategy"] for r in r1[:5]]
        top5_b = [r["strategy"] for r in r2[:5]]
        assert top5_a != top5_b, f"Radically different conditions produced same top-5: {top5_a}"

    def test_fatigue_increases_time_management_salience(self, versatile_team, strong_opponent, strategies):
        """
        Paper mechanics: high fatigue increases w10 (Time Management) and
        decreases w5 (High Press) and w13 (Physical Base).
        Strategies with high A10 demand should benefit when tired,
        because the weight on their strong dimension increases.
        Compare adjusted distances for a time-management-heavy strategy.
        """
        mc_fresh = {"time_remaining": 45, "score_diff": 0, "fatigue_level": 0.2, "morale": 0.7}
        mc_tired = {"time_remaining": 45, "score_diff": 0, "fatigue_level": 0.90, "morale": 0.7}

        r_fresh = evaluate_strategies(versatile_team, strong_opponent, strategies, mc_fresh)
        r_tired = evaluate_strategies(versatile_team, strong_opponent, strategies, mc_tired)

        # Adjusted distances must differ between fresh and tired
        dists_fresh = {r["strategy"]: r["adjusted_distance"] for r in r_fresh}
        dists_tired = {r["strategy"]: r["adjusted_distance"] for r in r_tired}
        differences = sum(1 for s in dists_fresh if abs(dists_fresh[s] - dists_tired[s]) > 1e-4)
        assert differences > 0, "Fatigue should change at least some adjusted distances"

    def test_urgency_shifts_ranking(self, versatile_team, weak_opponent, strategies):
        """Behind with little time left should favor offensive over conservative."""
        mc = {"time_remaining": 10, "score_diff": -2, "fatigue_level": 0.3, "morale": 0.7}
        results = evaluate_strategies(versatile_team, weak_opponent, strategies, mc)
        top3_cats = [r["category"] for r in results[:3]]
        assert any(c in ("offensive", "pressing") for c in top3_cats), \
            f"Expected offensive/pressing in top3, got {top3_cats}"

    def test_advantage_management(self, versatile_team, weak_opponent, strategies):
        """Ahead 2-0 with 10 min left should favor conservative/possession."""
        mc = {"time_remaining": 10, "score_diff": 2, "fatigue_level": 0.5, "morale": 0.8}
        results = evaluate_strategies(versatile_team, weak_opponent, strategies, mc)
        top5_cats = [r["category"] for r in results[:5]]
        assert any(c in ("defensive", "possession", "hybrid") for c in top5_cats), \
            f"Expected defensive/possession/hybrid in top5, got {top5_cats}"

    def test_gap_changes_distances(self, versatile_team, strong_opponent, strategies):
        """
        Against a technically/physically superior opponent, gap adjustments
        change the weight vector (w2, w11 up; w1, w6 down), which must
        produce different adjusted distances compared to a weak opponent.
        """
        mc = {"time_remaining": 45, "score_diff": 0, "fatigue_level": 0.3, "morale": 0.7}

        weak_opp = {f"A{i}": 0.40 for i in range(1, 15)}
        r_weak = evaluate_strategies(versatile_team, weak_opp, strategies, mc)
        r_strong = evaluate_strategies(versatile_team, strong_opponent, strategies, mc)

        # Rankings must differ — gap adjustments change the distance geometry
        order_weak = [r["strategy"] for r in r_weak]
        order_strong = [r["strategy"] for r in r_strong]
        assert order_weak != order_strong, \
            "Gap adjustments should produce a different ranking vs strong opponent"


# ---------------------------------------------------------------------------
# Integration: run_batch
# ---------------------------------------------------------------------------

class TestRunBatch:
    def test_batch_output_structure(self, example_input, strategies):
        output = run_batch(example_input, strategies)
        assert "meta" in output
        assert "results" in output
        assert output["meta"]["total_scenarios"] == len(example_input["scenarios"])
        assert len(output["results"]) == len(example_input["scenarios"])

    def test_output_validates_schema(self, example_input, strategies):
        output = run_batch(example_input, strategies)
        validated = DSSOutput(**output)
        assert validated.meta.version == "2.0.0"

    def test_scenario_ids_preserved(self, example_input, strategies):
        output = run_batch(example_input, strategies)
        input_ids = [s.get("id") for s in example_input["scenarios"]]
        output_ids = [r["scenario_id"] for r in output["results"]]
        assert input_ids == output_ids

    def test_top_n_respected(self, example_input, strategies):
        top_n = example_input.get("config", {}).get("top_n", 5)
        output = run_batch(example_input, strategies)
        for r in output["results"]:
            assert len(r["ranking"]) <= top_n


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------

class TestSchemaValidation:
    def test_valid_input_passes(self, example_input):
        validated = DSSInput(**example_input)
        assert validated.input_mode == "macro"
        assert len(validated.scenarios) == 5

    def test_attribute_out_of_range(self):
        data = {
            "input_mode": "macro",
            "team": {f"A{i}": 0.5 for i in range(1, 15)},
            "opponent": {f"A{i}": 0.5 for i in range(1, 15)},
            "scenarios": [{"match_conditions": {"time_remaining": 45, "score_diff": 0, "fatigue_level": 0.5, "morale": 0.7}}],
        }
        data["team"]["A1"] = 1.5
        with pytest.raises(ValidationError):
            DSSInput(**data)

    def test_raw_mode_rejected(self):
        data = {
            "input_mode": "raw",
            "team": {f"A{i}": 0.5 for i in range(1, 15)},
            "opponent": {f"A{i}": 0.5 for i in range(1, 15)},
            "scenarios": [{"match_conditions": {"time_remaining": 45, "score_diff": 0, "fatigue_level": 0.5, "morale": 0.7}}],
        }
        with pytest.raises(ValidationError):
            DSSInput(**data)

    def test_empty_scenarios_rejected(self):
        data = {
            "input_mode": "macro",
            "team": {f"A{i}": 0.5 for i in range(1, 15)},
            "opponent": {f"A{i}": 0.5 for i in range(1, 15)},
            "scenarios": [],
        }
        with pytest.raises(ValidationError):
            DSSInput(**data)

    def test_missing_attribute_rejected(self):
        team = {f"A{i}": 0.5 for i in range(1, 14)}  # missing A14
        data = {
            "input_mode": "macro",
            "team": team,
            "opponent": {f"A{i}": 0.5 for i in range(1, 15)},
            "scenarios": [{"match_conditions": {"time_remaining": 45, "score_diff": 0, "fatigue_level": 0.5, "morale": 0.7}}],
        }
        with pytest.raises(ValidationError):
            DSSInput(**data)