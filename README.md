# Semantic Distance-Based Decision Support System for Football Tactics

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the implementation code for the paper:

> **A Semantic Distance-Based Decision Support System for Football Strategy Selection**  

## Overview

The system recommends tactical strategies by computing **semantic distances** between a team's current state (represented as a 14-dimensional attribute vector) and a library of 20 canonical football strategies. Context-aware weighting adjusts recommendations based on match conditions such as energy levels, opponent characteristics, and time pressure.

### Key Features

- **14 macro-attributes** capturing offensive, defensive, physical, psychological, and relational dimensions
- **20 strategy templates** spanning the tactical spectrum from ultra-defensive to high-pressing approaches
- **Dynamic weight adaptation** based on contextual factors (fatigue, opponent gaps, time pressure)
- **Diagnostic outputs** explaining which attributes drive each recommendation
- **Reproducible experiments** with seeded random number generation

## Repository Structure

```
.
├── README.md
├── requirements.txt
├── football_strategy_generation_1_3_1.py   # Core DSS implementation
├── make_figures.py                          # Figure generation for experiments
├── compute_pilot_distances.py               # Pilot validation computations
└── figures/                                 # Generated figures (created on run)
```

## Installation

```bash
# Clone the repository
git clone https://github.com/[username]/football-dss-semantic-distance.git
cd football-dss-semantic-distance

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### 1. Core DSS System

The main implementation generates team profiles, computes semantic distances, and recommends strategies across multiple scenarios:

```bash
python football_strategy_generation_1_3_1.py
```

**Outputs:**
- Console output with strategy rankings per scenario
- Radar charts comparing team profiles with recommended strategies
- Diagnostic CSV files with per-attribute analysis
- Summary plots in `results_YYYYMMDD_HHMMSS/` directory

### 2. Experimental Figures

Generate figures for the experimental evaluation section:

```bash
python make_figures.py
```

**Outputs** (in `figures/` directory):
- `radar_scenario[1-4].png` — Team vs. top strategies for each scenario
- `sensitivity_all.png` — λ parameter sensitivity analysis
- `robustness_s2.png` — Robustness to input noise
- `ablation_s3.png` — Ablation study results
- `attribute_importance.png` — Top macro-attributes by impact

### 3. Pilot Validation

Reproduce the pilot study computations (C-Junioren match analysis):

```bash
python compute_pilot_distances.py
```

**Outputs:**
- Console output with distance tables and rankings
- `figures/pilot_radar_halftime.png` — Radar chart for pilot study
- `figures/pilot_results_summary.txt` — Summary of pilot validation

## Macro-Attributes

| Code | Attribute | Description |
|------|-----------|-------------|
| A1 | Offensive Strength | Capacity to create and convert scoring opportunities |
| A2 | Defensive Strength | Ability to prevent opponent attacks |
| A3 | Midfield Control | Dominance in central areas |
| A4 | Transition Speed | Pace of attack-to-defense and defense-to-attack shifts |
| A5 | High Press Capability | Ability to press opponents in advanced zones |
| A6 | Width Play | Exploitation of flanks and wide areas |
| A7 | Psychological Resilience | Mental robustness under pressure |
| A8 | Residual Energy | Current physical reserves |
| A9 | Team Morale | Collective confidence and motivation |
| A10 | Time Management | Ability to control match tempo |
| A11 | Tactical Cohesion | Coordination and positional discipline |
| A12 | Technical Base | Aggregate technical skill level |
| A13 | Physical Base | Aggregate physical capabilities |
| A14 | Relational Cohesion | Interpersonal chemistry and trust |

## Strategy Library

The system includes 20 strategies organized into categories:

**Offensive:** Build-up Play, Fast Counterattack, Direct Vertical Attack, Target Man Long Ball, Overlapping Flanks, Systematic Crossing, Quick Rotations, Late Midfield Runners

**Defensive:** Classic Catenaccio, Deep Block, Compact Zonal Defense, Strict Man-Marking, Offside Trap

**Pressing:** Ultra-Offensive Pressing, High-Zone Pressing, Midfield Pressing, Gegenpressing, Inducing Build-up Errors

**Possession:** Long Possession Game, Cautious Horizontal Play, Central Block + Short Bursts

## Reproducibility

All scripts use seeded random number generation (default `SEED = 41`) to ensure reproducible results. To verify reproducibility:

```bash
# Run twice and compare outputs
python football_strategy_generation_1_3_1.py > run1.txt
python football_strategy_generation_1_3_1.py > run2.txt
diff run1.txt run2.txt  # Should show no differences
```


## License

This project is licensed under the MIT License — see the LICENSE file for details.

## Contact

For questions or issues, please open a GitHub issue or contact remo-pareschi@unimol.it.
