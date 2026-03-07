# Extending Semantic Distance for Football Tactics with Structured I/O and Modular Engine

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![I/O: JSON](https://img.shields.io/badge/I%2FO-JSON-orange.svg)]()
[![Tests: 24 passed](https://img.shields.io/badge/tests-24%20passed-green.svg)]()

Fork of the [Semantic Distance-Based DSS for Football Tactics](https://github.com/Aribertus/football-dss-semantic-distance) by introducing structured JSON input/output and refactoring the original monolithic implementation into a modular, testable engine.

> Based on the paper: *Can Semantic Methods Enhance Team Sports Tactics? A Methodology for Football with Broader Applications*
> Di Rubbo A., Neri M., Pareschi R., Pedroni M., Valtancoli R., Zica P. — Sci 2025

## What this extension adds

The original repository provides a monolithic script (`football_strategy_generation_1_3_1.py`, ~1000 lines) that generates synthetic team data, computes strategy recommendations, and produces plots — all in a single execution flow with hardcoded scenarios and `print()` output. It serves as the research prototype behind the paper.

This extension makes the DSS consumable as a component, with a formal I/O contract:

- **Structured JSON input/output** with formal schemas and validation
- **Pure computation engine** — no print, no file I/O, no side effects
- **Strategy templates externalized** as configuration (JSON), not code constants
- **Batch processing** — multiple match scenarios in a single call, shared team profiles
- **Continuous dynamic weights** replacing binary if/else thresholds (no dead zones)
- **Min-max normalization** enabling context-aware ranking reordering
- **Plug-in architecture** for weight axes — add/remove/rewrite without touching core logic
- **Integrated figure generation** from real JSON results (`--figures` flag)
- **Full test suite** (24 tests) covering unit, integration, schema validation, and scenario differentiation

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
python dss_run.py --input examples/example_input.json --output output/results.json --strategies my_strategies.json --figures --figdir my_figures/

```

Output:

```text
[OK] 3 scenarios processed. Output: output/results.json
     First scenario best strategy: Quick Rotations in Attack

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
python dss_run.py --input examples/example_input.json --output output/results.json --strategies my_strategies.json

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
    "A1": 0.82, "A2": 0.55, "A3": 0.78, "A4": 0.75,
    "A5": 0.70, "A6": 0.72, "A7": 0.75, "A8": 0.80,
    "A9": 0.80, "A10": 0.65, "A11": 0.80, "A12": 0.82,
    "A13": 0.75, "A14": 0.78
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
        "morale": 0.60
      }
    }
  ],
  "config": {
    "opponent_penalty_lambda": 0.5,
    "top_n": 5
  }
}

```

`input_mode` accepts `"macro"` (A1-A14 already aggregated) or `"raw"` (player-level attributes, not yet implemented — reserved for future use).

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
    "version": "1.0.0",
    "timestamp": "2026-03-01T15:02:22Z",
    "config_used": { "opponent_penalty_lambda": 0.5, "top_n": 5 },
    "team_name": "Al-Hilal",
    "opponent_name": "Al-Najma",
    "total_scenarios": 3
  },
  "results": [
    {
      "scenario_id": "S1_min03",
      "scenario_label": "Goal conceded (0-1), cold start reaction",
      "match_conditions": { "time_remaining": 87, "score_diff": -1, "fatigue_level": 0.05, "morale": 0.60 },
      "best_strategy": {
        "strategy": "Quick Rotations in Attack",
        "adjusted_distance": 0.1213,
        "raw_distance": 0.2655,
        "category": "offensive"
      },
      "baseline_strategy": {
        "strategy": "Quick Rotations in Attack",
        "adjusted_distance": 0.2655,
        "raw_distance": 0.2655,
        "category": "offensive"
      },
      "ranking": [ "..." ]
    }
  ]
}

```

`best_strategy` uses dynamic weights (context-aware); `baseline_strategy` is pure geometric fit (static). When these differ, it means match conditions shifted the recommendation.

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
│  1. Raw distances    │  Euclidean: team ↔ 20 strategies
│  2. Opponent penalty │  Linear subtraction on opponent fit
│  3. Min-max normalize│  Relative scaling [0.1, 1.0]
│  4. Dynamic weights  │  Product of 4 continuous axes
│  5. Sort & rank      │
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

The original implementation used binary if/else thresholds: conditions either triggered or didn't, creating dead zones where different match states produced identical rankings.

This extension replaces them with four continuous sigmoid-based axes. Each axis maps match conditions to a multiplier around 1.0 (below 1.0 = bonus, above = penalty). The final adjustment is the product of all axes, clamped to [0.4, 2.0].

**Energy axis** — High fatigue penalizes high-intensity strategies proportionally. A team at fatigue 0.5 sees a *mild* penalty on pressing tactics; at 0.85 the penalty is *heavy*. No threshold, no dead zone.

**Urgency axis** — Combines time pressure and score context through a sigmoid centered at minute 25. Being behind with 30 minutes left produces a subtle push toward offensive tactics; with 10 minutes left, the push is strong. The axis also works in reverse: ahead with little time rewards conservative strategies.

**Morale axis** — Low morale penalizes tactically complex strategies (requiring high cohesion and coordination), while high morale gives a slight bonus to ambitious offensive tactics.

**Score context axis** — Independent of time, captures the pure effect of lead size. A 2-0 lead mildly favors possession/defensive strategies regardless of minute.

### Plug-in Design

Adding a new axis requires two changes:

1. Write a function with signature `(match_conditions: dict, strategy_vector: list[float]) -> float`
2. Append it to the `WEIGHT_AXES` list

```python
# Example: weather axis (hypothetical)
def axis_weather(mc: dict, sv: list[float]) -> float:
    rain = mc.get("rain_intensity", 0.0)
    technical_demand = (sv[11] + sv[2]) / 2.0  # A12 + A3
    penalty = rain * technical_demand
    return _scale_factor(_sigmoid(penalty, 0.0, 3.0), 0.90, 1.20)

WEIGHT_AXES.append(axis_weather)

```

No other code needs to change. The engine multiplies all registered axes automatically.

### Min-Max Normalization

Raw Euclidean distances vary in absolute scale depending on the team profile. A team strongly aligned with one strategy can have a raw distance gap of 34% between #1 and #2, making it impossible for dynamic weights to reorder the ranking.

Normalizing combined distances to [0.1, 1.0] before applying weights preserves relative ordering while giving the dynamic system enough leverage to shift rankings when conditions warrant it. The floor of 0.1 ensures the best raw strategy still receives a nonzero distance that weights can meaningfully adjust.

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

The suite covers six areas:

**Distance computation** — zero distance for identical vectors, known Euclidean values, symmetry.

**Dynamic weights** — neutral conditions produce near-1.0 adjustment, extreme conditions stay clamped, fatigue penalizes intensity, no dead zones (different inputs always produce different outputs), all axes return positive floats.

**Strategy evaluation** — results sorted ascending, all 20 strategies present, required output fields.

**Scenario differentiation** — radically different conditions produce different top-5 rankings, urgency favors offensive tactics when behind, advantage management favors conservative tactics when ahead.

**Batch execution** — output structure matches schema, Pydantic validates output, scenario IDs preserved, top_n respected.

**Schema validation** — valid input passes, out-of-range attributes rejected, raw mode rejected, empty scenarios rejected, missing attributes rejected.

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

**Step 1B — Raw input mode.** Accept player-level attributes (xG, speed, stamina, etc.) per role and compute A1-A14 internally using the aggregation functions from the original codebase. The `input_mode: "raw"` field is already reserved in the schema.

**Diagnostics layer.** Per-attribute delta analysis (strategy demands vs. team capabilities) in the output, identifying capability gaps and surpluses for the recommended strategy.

**Weight calibration.** Systematic tuning of axis parameters (sigmoid centers, steepness, ranges) against expert-labeled match scenarios to ensure meaningful ranking differentiation across diverse game states.

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

Di Rubbo, A.; Neri, M.; Pareschi, R.; Pedroni, M.; Valtancoli, R.; Zica, P. *Can Semantic Methods Enhance Team Sports Tactics? A Methodology for Football with Broader Applications.* Sci 2025, 1, 0.
