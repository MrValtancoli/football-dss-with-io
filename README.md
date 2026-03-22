# Extending Semantic Distance for Football Tactics with Structured I/O and Modular Engine

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![I/O: JSON](https://img.shields.io/badge/I%2FO-JSON-orange.svg)]()
[![Tests: 35 passed](https://img.shields.io/badge/tests-35%20passed-green.svg)]()

Fork of the [Semantic Distance-Based DSS for Football Tactics](https://github.com/Aribertus/football-dss-semantic-distance) by introducing structured JSON input/output and refactoring the original monolithic implementation into a modular, testable engine.

> Based on the paper: *Can Semantic Methods Enhance Team Sports Tactics? A Methodology for Football with Broader Applications*
> Di Rubbo A., Neri M., Pareschi R., Pedroni M., Valtancoli R., Zica P. — Sci 2026, 8(3), 63

## What this extension adds

The original repository provides a monolithic script (`football_strategy_generation_1_3_1.py`, ~1000 lines) that generates synthetic team data, computes strategy recommendations, and produces plots — all in a single execution flow with hardcoded scenarios and `print()` output. It serves as the research prototype behind the paper.

This extension makes the DSS consumable as a component, with a formal I/O contract:

- **Structured JSON input/output** with formal schemas and validation
- **Pure computation engine** — no print, no file I/O, no side effects
- **Strategy templates externalized** as configuration (JSON), not code constants
- **Batch processing** — multiple match scenarios in a single call, shared team profiles
- **Per-attribute dynamic weighting** inside the Euclidean distance (Section 3.6.2, eq. 5–13 of the paper), replacing the previous post-distance scalar multiplier
- **Per-scenario profile overrides** — A7, A8, A9 can be adjusted per scenario to reflect in-game state changes, independently of the static team profile
- **Integrated figure generation** from real JSON results (`--figures` flag)
- **Full test suite** (35 tests) covering distance computation, weight vector, integration, schema validation, and scenario differentiation

The original research scripts (`make_figures.py`, `compute_pilot_distances.py`) are preserved unchanged for reproducibility of the paper's experimental results.

## Repository Structure

```text
.
├── README.md
├── requirements.txt
├── dss_engine.py                  # Core computation module (pure functions)
├── dss_schema.py                  # Pydantic models for I/O validation
├── dss_run.py                     # CLI entry point (engine + optional figures)
├── dss_figures.py                 # Figure generator from JSON results
├── strategy_templates.json        # 20 strategies with categories (config)
├── dss_input_schema.json          # JSON Schema for input contract
├── dss_output_schema.json         # JSON Schema for output contract
├── test_dss.py                    # Test suite (pytest)
│
│   # Match scenario examples (Al-Najma 2 - 4 Al-Hilal, Saudi Pro League 2025-11-07)
├── examples/
|   ├── example_input.json         # Example: Al-Hilal vs Al-Najma, 5 scenarios
|   ├── my_strategies.json         # Custom 20 strategies with categories (config)
│   ├── dss_input_scenario_1.json  # S1 min 3'  — Goal conceded (0-1), cold start reaction
│   ├── dss_input_scenario_2.json  # S2 min 10' — Equalizer (1-1), positive momentum
│   ├── dss_input_scenario_3.json  # S3 min 58' — Red card Lázaro, 11v10 numerical advantage
│   ├── dss_input_scenario_4.json  # S4 min 71' — Drawn 2-2, match reopened after OG
│   └── dss_input_scenario_5.json  # S5 min 85' — Leading 3-2, Al-Najma down to 9 men
│
│   # Generated outputs and diagnostic figures
├── output/
│   ├── example_output.json
│   ├── results.json
│   ├── results_s1.json
│   ├── results_s2.json
│   ├── results_s3.json
│   ├── test_output.json
│   └── figures/
│       ├── baseline_delta.png
│       ├── cross_scenario_overview.png
│       ├── radar_S1_min03.png
│       ├── radar_S2_min10.png
│       ├── radar_S3_min58.png
│       ├── ranking_S1_min03.png
│       ├── ranking_S2_min10.png
│       └── ranking_S3_min58.png
│
│   # Original research scripts (unchanged)
├── football_strategy_generation_1_3_1.py
├── make_figures.py
├── compute_pilot_distances.py
└── LICENSE

```

## Installation

```bash
git clone https://github.com/MrValtancoli/football-dss-with-io.git
cd football-dss-with-io

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt

```

Dependencies: `numpy`, `pandas`, `matplotlib`, `pydantic`, `pytest`.

## Quick Start

Run the engine:

```bash
python dss_run.py --input examples/example_input.json --output output/results.json

```

Run the engine with figure generation:

```bash
python dss_run.py --input examples/example_input.json --output output/results.json --figures

```

Run with custom strategies and custom figure directory:

```bash
python dss_run.py --input examples/example_input.json --output output/results.json --strategies examples/my_strategies.json --figures --figdir my_figures/

```

Output:

```text
[OK] 3 scenarios processed. Output: output/results.json
     First scenario best strategy: Build-up Play

[FIGURES] Generating plots in output/figures/ ...
[OK] 8 figures generated:
  → radar_S1_min03.png
  → ranking_S1_min03.png
  → ...
  → cross_scenario_overview.png
  → baseline_delta.png

```

Custom strategy templates:

```bash
python dss_run.py --input examples/example_input.json --output output/results.json --strategies examples/my_strategies.json

```

The figure module can also run standalone against previously generated results:

```bash
python dss_figures.py --results output/results.json --input examples/example_input.json --strategies strategy_templates.json --outdir output/figures/

```

Run a specific match scenario (see `examples/`):

```bash
python dss_run.py --input examples/dss_input_scenario_3.json --output output/results_s3.json

```

## Input Format

The system accepts a JSON file with three required sections: team profile, opponent profile, and an array of match scenarios. An optional config section controls engine parameters.

```json
{
  "input_mode": "macro",
  "team": {
    "name": "Al-Hilal",
    "A1": 0.75, "A2": 0.65, "A3": 0.80, "A4": 0.50,
    "A5": 0.42, "A6": 0.65, "A7": 0.72, "A8": 0.75,
    "A9": 0.78, "A10": 0.75, "A11": 0.70, "A12": 0.78,
    "A13": 0.62, "A14": 0.72
  },
  "opponent": {
    "name": "Al-Najma",
    "A1": 0.50, "A2": 0.75, "A3": 0.55, "A4": 0.55,
    "A5": 0.45, "A6": 0.40, "A7": 0.70, "A8": 0.75,
    "A9": 0.65, "A10": 0.70, "A11": 0.72, "A12": 0.50,
    "A13": 0.65, "A14": 0.68
  },
  "scenarios": [
    {
      "id": "S1_min03",
      "label": "Goal conceded (0-1), cold start reaction",
      "match_conditions": {
        "time_remaining": 87,
        "score_diff": -1,
        "fatigue_level": 0.05,
        "morale": 0.40
      },
      "profile_overrides": {
        "A7": 0.55,
        "A9": 0.40
      }
    }
  ],
  "config": {
    "opponent_penalty_lambda": 0.2,
    "top_n": 5
  }
}

```

`input_mode` accepts `"macro"` (A1-A14 already aggregated) or `"raw"` (player-level attributes, not yet implemented — reserved for future use).

### Profile Overrides

The optional `profile_overrides` field allows per-scenario adjustment of three team attributes that change during a match:

| Field | Attribute | Typical use |
| --- | --- | --- |
| `A7` | Psychological Resilience | Lower after a goal conceded, red card, or comeback |
| `A8` | Residual Energy | Structural reserve (heavy schedule, limited rotation) |
| `A9` | Team Morale | Reflects in-game momentum shifts |

These overrides replace the corresponding value in the team vector for that scenario only. The static team profile and the baseline computation are never modified.

Note: `fatigue_level` in `match_conditions` and `A8` in `profile_overrides` serve distinct purposes. `fatigue_level` is the instantaneous in-game fatigue signal that feeds the dynamic weight vector (how much each attribute matters). `A8` is the structural energy reserve the team entered the match with. When IoT data (heart rate monitors, GPS) becomes available, `fatigue_level` will be populated in real time from physiological sensors; `A8` will remain a pre-match assessment.

Validation rejects out-of-range values, missing fields, and unsupported modes with clear error messages:

```text
[ERROR] Input validation failed:
  team → A1: Input should be less than or equal to 1

```

## Output Format

For each scenario, the engine returns the best strategy (with and without dynamic weights), plus the top-N ranking:

```json
{
  "meta": {
    "version": "2.0.0",
    "timestamp": "2026-03-18T10:14:29Z",
    "config_used": { "opponent_penalty_lambda": 0.2, "top_n": 5 },
    "team_name": "Al-Hilal",
    "opponent_name": "Al-Najma",
    "total_scenarios": 1
  },
  "results": [
    {
      "scenario_id": "S1_min03",
      "scenario_label": "Goal conceded (0-1), cold start reaction",
      "match_conditions": { "time_remaining": 87, "score_diff": -1, "fatigue_level": 0.05, "morale": 0.40 },
      "best_strategy": {
        "strategy": "Build-up Play",
        "adjusted_distance": 0.1619,
        "raw_distance": 0.2844,
        "category": "offensive"
      },
      "baseline_strategy": {
        "strategy": "Build-up Play",
        "adjusted_distance": 0.2844,
        "raw_distance": 0.2844,
        "category": "offensive"
      },
      "ranking": [ "..." ]
    }
  ]
}

```

`best_strategy` uses per-attribute weighted Euclidean distance (context-aware); `baseline_strategy` is pure unweighted geometric fit (static). When these differ, it means match conditions shifted the recommendation. Note: `adjusted_distance` may be negative because the combined formula uses linear subtraction of the opponent distance.

## Figure Generation

The `--figures` flag activates `dss_figures.py`, which reads the three JSON sources (input, output, strategy templates) and produces four types of diagnostic plots.

**Radar plots (per scenario)** — Team profile overlaid with the top-3 recommended strategies. Shows at a glance where team capabilities align or diverge from what each strategy demands. Color-coded by strategy category.

**Ranking bar charts (per scenario)** — Horizontal bars comparing adjusted (DSS) vs raw (baseline) distances for the top-N strategies. Makes the gap between first and second choice immediately visible.

**Cross-scenario overview** — Two-panel summary across all scenarios. Top panel: baseline, DSS-adjusted, and raw distances side by side. Bottom panel: match conditions (fatigue, morale, score differential, time remaining) that drove each recommendation. Designed to reveal whether the system actually differentiates across contexts.

**Baseline delta chart** — Per-scenario comparison of the context adjustment (raw − adjusted) for each ranked strategy. Green bars indicate the DSS reduced the distance (context-fit bonus), red bars indicate a penalty. Highlights which scenarios benefit most from dynamic weighting.

All figures are saved as PNG at 180 dpi. The default output directory is `output/figures/` alongside the results file, overridable with `--figdir`.

## Architecture

### Computation Pipeline

```text
Input JSON
    │
    ▼
┌─────────────────────┐
│  Pydantic Validation │  ← dss_schema.py
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Team/Opponent       │  A1-A14 vectors
│  Profile Parsing     │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  For each scenario:  │
│                      │
│  0. Apply overrides  │  A7, A8, A9 per-scenario (team vector only)
│  1. Weight vector    │  14-dim w from match context (eq. 5–13)
│  2. Adapted distance │  Weighted Euclidean: team ↔ 20 strategies
│  3. Opponent penalty │  Linear subtraction: d_team - α·d_opp
│  4. Sort & rank      │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Output JSON         │
└──────────┬──────────┘
           │  (optional: --figures)
           ▼
┌─────────────────────┐
│  dss_figures.py      │  Joins input + output + templates
│  Radar / Bars /      │  Generates diagnostic plots
│  Overview / Delta    │
└─────────────────────┘

```

### Dynamic Weight System

The engine implements the per-attribute dynamic weighting from Section 3.6.2 of the paper (equations 5–13). Instead of applying a scalar multiplier after the distance computation, each of the 14 attributes receives an individually calibrated weight *inside* the weighted Euclidean distance formula:

```
d_adapt(x, y; w) = sqrt( Σ w_j · (x_j - y_j)² )
```

The weight vector w is computed fresh for each scenario from six match-state inputs: instantaneous fatigue (via `fatigue_level`, inverted as `e = 1 - fatigue_level`), technical and physical gaps (A12/A13 team vs opponent), time remaining (converted to fraction `t = time_remaining / 90`), and score state (mapped to ternary `s ∈ {-1, 0, +1}`).

Three mechanisms modulate the weights:

**Energy adjustments (eq. 5–7)** — When `e < τe` (default 0.50), the system reduces the importance of High Press Capability (A5) and Physical Base (A13), while increasing Time Management (A10). This reflects the tactical reality that fatigued teams should avoid energy-intensive strategies.

**Gap adjustments (eq. 8–11)** — When the team is technically or physically inferior (negative Δtech or Δphys), the system increases weight on Defensive Strength (A2) and Tactical Cohesion (A11), while reducing Offensive Strength (A1) and Width Utilization (A6).

**Time pressure adjustments (eq. 12–13)** — When time is running out (`t < τt`, default 0.25) and the team is not winning (`s ≤ 0`), the system amplifies Transition Speed (A4) and Offensive Strength (A1) to favor direct, vertical play.

All multipliers are clamped to [0.3, 2.5] and normalized to sum = 14 (preserving baseline scale). Default parameters follow Table 5 of the paper and are exposed via `DEFAULT_PARAMS` in `dss_engine.py`.

## Macro-Attributes

| Code | Attribute | Description |
| --- | --- | --- |
| A1 | Offensive Strength | Capacity to create and convert scoring opportunities |
| A2 | Defensive Strength | Ability to prevent opponent attacks |
| A3 | Midfield Control | Dominance in central areas |
| A4 | Transition Speed | Pace of attack-to-defense and defense-to-attack shifts |
| A5 | High Press Capability | Ability to press opponents in advanced zones |
| A6 | Width Utilization | Exploitation of flanks and wide areas |
| A7 | Psychological Resilience | Mental robustness under pressure |
| A8 | Residual Energy | Current physical reserves |
| A9 | Team Morale | Collective confidence and motivation |
| A10 | Time Management | Ability to control match tempo |
| A11 | Tactical Cohesion | Coordination and positional discipline |
| A12 | Technical Base | Aggregate technical skill level |
| A13 | Physical Base | Aggregate physical capabilities |
| A14 | Relational Cohesion | Interpersonal chemistry and trust |

## Strategy Library

The 20 strategies are organized into five categories in `strategy_templates.json`:

**Offensive (8):** Build-up Play, Fast Counterattack, Long Ball to Target Man, Late Midfield Runners, Systematic Crossing, Overlapping Flanks, Quick Rotations in Attack, Direct Vertical Attack

**Defensive (5):** Classic Catenaccio, Positional Defense, Compact Zonal Defense, Strict Man-Marking, Offside Trap

**Pressing (4):** High Press, Gegenpressing, Midfield Pressing, Inducing Build-up Errors

**Possession (2):** Extended Possession Play, Cautious Horizontal Play

**Hybrid (1):** Central Block with Quick Breaks

Each strategy is encoded as a 14-dimensional vector representing its demands on each macro-attribute. These vectors are configuration, not code: editing `strategy_templates.json` changes the behavior without touching Python.

## Testing

```bash
python -m pytest test_dss.py -v

```

The suite covers six areas (35 tests total):

**Distance computation (7)** — zero distance for identical vectors (weighted and unweighted), known Euclidean values, symmetry, weight=None equivalence, single-attribute amplification.

**Weight vector (11)** — neutral conditions produce uniform weights, sum always = 14, clamp bounds respected, energy activation/inactivation thresholds, time pressure activation (losing) and inactivation (winning), gap activation (inferior) and inactivation (superior), combined mechanisms, ternary score mapping.

**Strategy evaluation (3)** — results sorted ascending, all 20 strategies present, required output fields.

**Scenario differentiation (5)** — radically different conditions produce different rankings, fatigue changes adjusted distances, urgency favors offensive when behind, advantage management favors conservative when ahead, gap changes ranking vs strong opponent.

**Batch execution (4)** — output structure matches schema, Pydantic validates output, scenario IDs preserved, top_n respected.

**Schema validation (5)** — valid input passes, out-of-range attributes rejected, raw mode rejected, empty scenarios rejected, missing attributes rejected.

## Match Scenario Examples

The `examples/` folder contains five input files derived from the real match **Al-Najma 2 – 4 Al-Hilal** (Saudi Pro League, 2025-11-07). Each file isolates a key match event and provides the contextual `match_conditions` for the DSS to evaluate.

| File | Minute | Event | Score | Phase |
| --- | --- | --- | --- | --- |
| `dss_input_scenario_1.json` | 3' | Lázaro scores from distance | 1-0 | Disadvantage |
| `dss_input_scenario_2.json` | 10' | Al-Dawsari equalizes | 1-1 | Equilibrium |
| `dss_input_scenario_3.json` | 58' | Lázaro red card (violent conduct) | 1-1 | Equilibrium, 11v10 |
| `dss_input_scenario_4.json` | 71' | Theo Hernández makes it 2-2 | 2-2 | Equilibrium |
| `dss_input_scenario_5.json` | 85' | Leading 3-2, Al-Najma down to 9 | 3-2 | Advantage, 11v9 |

Run any scenario:

```bash
python dss_run.py --input examples/dss_input_scenario_1.json --output output/results_s1.json

```

Or batch all five:

```bash
for f in examples/dss_input_scenario_*.json; do
  python dss_run.py --input "$f" --output "output/results_$(basename $f)"
done

```

These scenarios are designed for demonstrating how the DSS dynamically adapts recommendations as match context evolves — from early-match shock (conceding at 3') through numerical superiority management (11v10, then 11v9).

## Roadmap

**Per-scenario profile overrides (v1).** A7, A8, A9 can be adjusted per scenario via `profile_overrides` to reflect in-game state changes (morale drop after a goal conceded, energy depletion, resilience under pressure). The static team profile serves as the pre-match baseline and is never modified.

**Step 1B — Raw input mode.** Accept player-level attributes (xG, speed, stamina, etc.) per role and compute A1-A14 internally using the aggregation functions from the original codebase. The `input_mode: "raw"` field is already reserved in the schema.

**Diagnostics layer.** Per-attribute delta analysis (strategy demands vs. team capabilities) in the output, identifying capability gaps and surpluses for the recommended strategy.

**Parameter calibration.** Systematic tuning of weight parameters (τe, γe, γg, τt, γt) against expert-labeled match scenarios to ensure meaningful ranking differentiation. Currently using paper defaults from Table 5.

**IoT integration layer.** Connect `fatigue_level` to real-time physiological data (heart rate monitors, GPS units). The engine already accepts per-scenario fatigue via `match_conditions`; the inversion `e = 1 - fatigue_level` maps it to the paper's residual energy convention.

**Extended figure suite.** Sensitivity analysis (λ sweep), robustness tests (Monte Carlo noise injection), and ablation studies — currently available in the original `make_figures.py` with synthetic data, to be integrated into `dss_figures.py` using real JSON results.

## Upstream Research Scripts

The following scripts from the [original repository](https://github.com/Aribertus/football-dss-semantic-distance) are preserved unchanged for reproducibility of the paper's experimental results:

* `football_strategy_generation_1_3_1.py` — Original monolithic DSS with all 20 strategies, synthetic data generation, and visualization
* `make_figures.py` — Generates figures for the paper's experimental evaluation (sensitivity analysis, ablation, robustness)
* `compute_pilot_distances.py` — Pilot validation computations (Al-Hilal vs Al-Najma match analysis)

These scripts use `SEED = 41` for deterministic output.

## License

MIT License — see [LICENSE](https://www.google.com/search?q=LICENSE) for details.

## References

Di Rubbo, A.; Neri, M.; Pareschi, R.; Pedroni, M.; Valtancoli, R.; Zica, P. *Can Semantic Methods Enhance Team Sports Tactics? A Methodology for Football with Broader Applications.* Sci 2026, 8, 63. https://doi.org/10.3390/sci8030063
