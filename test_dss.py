"""
DSS Test Suite
Run: python -m pytest test_dss.py -v
"""

import json
import pytest
import numpy as np
from pathlib import Path

from dss_engine import (
    compute_semantic_distance,
    apply_dynamic_weights,
    evaluate_strategies,
    compute_baseline,
    run_batch,
    load_strategies,
    WEIGHT_AXES,
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
    path = Path(__file__).parent / ".." / "example_input.json"
    if not path.exists():
        path = Path("/mnt/user-data/outputs/example_input.json")
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Unit tests: distance computation
# ---------------------------------------------------------------------------

class TestDistance:
    def test_identical_vectors_zero(self):
        v = [0.5] * 14
        assert compute_semantic_distance(v, v) == 0.0

    def test_known_distance(self):
        v1 = [1.0] * 14
        v2 = [0.0] * 14
        expected = np.sqrt(14.0)
        assert abs(compute_semantic_distance(v1, v2) - expected) < 1e-6

    def test_symmetry(self):
        v1 = [0.3, 0.7, 0.5, 0.8, 0.2, 0.6, 0.4, 0.9, 0.1, 0.5, 0.7, 0.3, 0.8, 0.6]
        v2 = [0.6, 0.4, 0.8, 0.2, 0.7, 0.3, 0.9, 0.1, 0.5, 0.7, 0.3, 0.8, 0.2, 0.4]
        assert compute_semantic_distance(v1, v2) == compute_semantic_distance(v2, v1)


# ---------------------------------------------------------------------------
# Unit tests: dynamic weights (continuous, no dead zones)
# ---------------------------------------------------------------------------

class TestDynamicWeights:
    def test_neutral_conditions_near_one(self):
        """Neutral match state should produce adjustment close to 1.0."""
        mc = {"time_remaining": 45, "score_diff": 0, "fatigue_level": 0.5, "morale": 0.7}
        sv = [0.6] * 14
        result = apply_dynamic_weights(1.0, mc, sv)
        assert 0.7 < result < 1.5, f"Expected near 1.0, got {result}"

    def test_clamped_bounds(self):
        """Result must always be within [0.4, 2.0] * raw_distance."""
        extreme_mc = {"time_remaining": 1, "score_diff": -5, "fatigue_level": 0.99, "morale": 0.05}
        sv = [0.9] * 14
        result = apply_dynamic_weights(1.0, extreme_mc, sv)
        assert 0.4 <= result <= 2.0

    def test_fatigue_penalizes_intensity(self, strategies):
        """High fatigue should make high-intensity strategies score worse than low-intensity."""
        mc_tired = {"time_remaining": 45, "score_diff": 0, "fatigue_level": 0.9, "morale": 0.7}
        mc_fresh = {"time_remaining": 45, "score_diff": 0, "fatigue_level": 0.1, "morale": 0.7}

        high_press = next(s for s in strategies if s["name"] == "High Press")
        adj_tired = apply_dynamic_weights(1.0, mc_tired, high_press["vector"])
        adj_fresh = apply_dynamic_weights(1.0, mc_fresh, high_press["vector"])
        assert adj_tired > adj_fresh, "High Press should be penalized more when tired"

    def test_no_dead_zone(self):
        """Slightly different conditions must produce different adjustments."""
        sv = [0.7, 0.5, 0.6, 0.8, 0.7, 0.5, 0.6, 0.7, 0.7, 0.5, 0.8, 0.6, 0.7, 0.6]
        mc_a = {"time_remaining": 45, "score_diff": 0, "fatigue_level": 0.50, "morale": 0.70}
        mc_b = {"time_remaining": 45, "score_diff": 0, "fatigue_level": 0.55, "morale": 0.70}
        a = apply_dynamic_weights(1.0, mc_a, sv)
        b = apply_dynamic_weights(1.0, mc_b, sv)
        assert a != b, "Different fatigue must produce different adjustment"

    def test_all_axes_registered(self):
        """Verify the axis registry has the expected count."""
        assert len(WEIGHT_AXES) == 4, f"Expected 4 axes, got {len(WEIGHT_AXES)}"

    def test_each_axis_returns_positive(self):
        """Each axis function must return a positive float."""
        mc = {"time_remaining": 30, "score_diff": -1, "fatigue_level": 0.6, "morale": 0.5}
        sv = [0.7] * 14
        for fn in WEIGHT_AXES:
            val = fn(mc, sv)
            assert val > 0, f"{fn.__name__} returned {val}"
            assert isinstance(val, float), f"{fn.__name__} returned {type(val)}"


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
# Integration: scenario differentiation (the original problem)
# ---------------------------------------------------------------------------

class TestScenarioDifferentiation:
    @pytest.fixture
    def versatile_team(self):
        """A realistic team profile, not perfectly balanced."""
        return {
            "A1": 0.70, "A2": 0.65, "A3": 0.72, "A4": 0.68,
            "A5": 0.60, "A6": 0.65, "A7": 0.70, "A8": 0.75,
            "A9": 0.72, "A10": 0.60, "A11": 0.70, "A12": 0.68,
            "A13": 0.65, "A14": 0.70,
        }

    @pytest.fixture
    def weak_opponent(self):
        return {f"A{i}": 0.45 for i in range(1, 15)}

    def test_different_conditions_different_ranking(self, versatile_team, weak_opponent, strategies):
        """Core test: distinct match conditions must produce distinct rankings."""
        mc_fresh_ahead = {"time_remaining": 70, "score_diff": 2, "fatigue_level": 0.2, "morale": 0.85}
        mc_tired_behind = {"time_remaining": 10, "score_diff": -2, "fatigue_level": 0.85, "morale": 0.35}

        r1 = evaluate_strategies(versatile_team, weak_opponent, strategies, mc_fresh_ahead)
        r2 = evaluate_strategies(versatile_team, weak_opponent, strategies, mc_tired_behind)

        top5_a = [r["strategy"] for r in r1[:5]]
        top5_b = [r["strategy"] for r in r2[:5]]
        assert top5_a != top5_b, f"Radically different conditions produced same top-5: {top5_a}"

    def test_urgency_shifts_ranking(self, versatile_team, weak_opponent, strategies):
        """Behind with 10 min left should favor offensive over conservative."""
        mc = {"time_remaining": 10, "score_diff": -2, "fatigue_level": 0.3, "morale": 0.7}
        results = evaluate_strategies(versatile_team, weak_opponent, strategies, mc)
        top3_cats = [r["category"] for r in results[:3]]
        # At least one offensive or pressing in top 3
        assert any(c in ("offensive", "pressing") for c in top3_cats), \
            f"Expected offensive/pressing in top3, got {top3_cats}"

    def test_advantage_management(self, versatile_team, weak_opponent, strategies):
        """Ahead 2-0 with 10 min left should favor conservative/possession."""
        mc = {"time_remaining": 10, "score_diff": 2, "fatigue_level": 0.5, "morale": 0.8}
        results = evaluate_strategies(versatile_team, weak_opponent, strategies, mc)
        top5_cats = [r["category"] for r in results[:5]]
        assert any(c in ("defensive", "possession", "hybrid") for c in top5_cats), \
            f"Expected defensive/possession/hybrid in top5, got {top5_cats}"


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
        assert validated.meta.version == "1.0.0"

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
