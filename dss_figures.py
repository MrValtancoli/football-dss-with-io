"""
DSS Figures — Plot generator from JSON results.
Reads results.json + example_input.json + strategy_templates.json
and produces publication-ready figures.

Usage (standalone):
  python dss_figures.py --results results.json --input match.json --strategies strategy_templates.json --outdir figures/

Usage (from dss_run.py):
  Called programmatically via generate_all_figures()
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path
from typing import Optional

# ───────────────────────────────────────────────────────────────
# Config
# ───────────────────────────────────────────────────────────────

ATTR_KEYS = [f"A{i}" for i in range(1, 15)]
ATTR_SHORT = [
    "Off.Str", "Def.Str", "Mid.Ctrl", "Trans.Spd",
    "Hi.Press", "Width", "Psy.Res", "Energy",
    "Morale", "Time.Mgt", "Tact.Coh", "Tech.Base",
    "Phys.Base", "Rel.Coh"
]

CATEGORY_COLORS = {
    "offensive":  "#e74c3c",
    "defensive":  "#3498db",
    "pressing":   "#e67e22",
    "possession": "#2ecc71",
    "hybrid":     "#9b59b6",
    "unknown":    "#95a5a6",
}

PLT_STYLE = {
    "figure.facecolor": "#fafafa",
    "axes.facecolor":   "#ffffff",
    "axes.grid":        True,
    "grid.alpha":       0.3,
    "font.size":        10,
}


# ───────────────────────────────────────────────────────────────
# Data loading & merging
# ───────────────────────────────────────────────────────────────

def load_all(results_path: str, input_path: str, strategies_path: str) -> dict:
    """Load and merge the three JSON sources."""
    with open(results_path) as f:
        results = json.load(f)
    with open(input_path) as f:
        inp = json.load(f)
    with open(strategies_path) as f:
        strats = json.load(f)

    # Build strategy name → vector lookup
    strat_lookup = {s["name"]: s["vector"] for s in strats["strategies"]}

    # Team vector
    team_vec = [inp["team"][k] for k in ATTR_KEYS]

    return {
        "results": results,
        "input": inp,
        "strategies": strats,
        "strat_lookup": strat_lookup,
        "team_vec": team_vec,
        "team_name": inp["team"].get("name", "Team"),
        "opponent_name": inp["opponent"].get("name", "Opponent"),
    }


# ───────────────────────────────────────────────────────────────
# Fig 1: Radar per scenario — team vs top-3 strategies
# ───────────────────────────────────────────────────────────────

def fig_radar_scenario(data: dict, scenario: dict, outdir: str):
    """Radar plot: team profile overlaid with top-3 recommended strategies."""
    plt.rcParams.update(PLT_STYLE)

    sid = scenario["scenario_id"]
    label = scenario["scenario_label"]
    ranking = scenario["ranking"][:3]
    team_vec = np.array(data["team_vec"])

    N = len(ATTR_SHORT)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    def close(v):
        v = np.asarray(v, dtype=float)
        return np.concatenate([v, [v[0]]])

    fig, ax = plt.subplots(figsize=(9, 6.5), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(ATTR_SHORT, fontsize=8)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8"], fontsize=7)
    ax.set_ylim(0, 1.0)

    # Team profile
    ax.plot(angles, close(team_vec), linewidth=2.2, linestyle="--",
            color="#2c3e50", label=data["team_name"], zorder=5)
    ax.fill(angles, close(team_vec), alpha=0.06, color="#2c3e50")

    # Top strategies
    line_styles = ["-", "-", ":"]
    alphas = [0.9, 0.7, 0.5]
    for i, entry in enumerate(ranking):
        sname = entry["strategy"]
        vec = data["strat_lookup"].get(sname)
        if vec is None:
            continue
        cat = entry.get("category", "unknown")
        color = CATEGORY_COLORS.get(cat, "#95a5a6")
        dist_label = f"{sname} (d={entry['adjusted_distance']:.3f})"
        ax.plot(angles, close(vec), linewidth=1.8, linestyle=line_styles[i],
                color=color, alpha=alphas[i], label=dist_label)

    ax.legend(loc="lower left", bbox_to_anchor=(1.05, -0.05), fontsize=8,
              framealpha=0.9)
    ax.set_title(f"{sid}: {label}", fontsize=12, pad=18, fontweight="bold")

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"radar_{sid}.png"), dpi=180,
                bbox_inches="tight")
    plt.close()


# ───────────────────────────────────────────────────────────────
# Fig 2: Ranking bar chart per scenario
# ───────────────────────────────────────────────────────────────

def fig_ranking_bars(scenario: dict, outdir: str):
    """Horizontal bar chart of top-N adjusted distances."""
    plt.rcParams.update(PLT_STYLE)

    sid = scenario["scenario_id"]
    label = scenario["scenario_label"]
    ranking = scenario["ranking"]

    names = [r["strategy"] for r in ranking][::-1]
    adj = [r["adjusted_distance"] for r in ranking][::-1]
    raw = [r["raw_distance"] for r in ranking][::-1]
    cats = [r.get("category", "unknown") for r in ranking][::-1]
    colors = [CATEGORY_COLORS.get(c, "#95a5a6") for c in cats]

    fig, ax = plt.subplots(figsize=(9, 3.5))

    bars_adj = ax.barh(names, adj, height=0.45, color=colors, alpha=0.85,
                       label="Adjusted (DSS)")
    bars_raw = ax.barh(names, raw, height=0.45, color="#bdc3c7", alpha=0.4,
                       label="Raw (baseline)")

    # Value labels
    for bar, val in zip(bars_adj, adj):
        ax.text(val + 0.008, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=8, fontweight="bold")

    ax.set_xlabel("Semantic Distance (lower = better fit)")
    ax.set_title(f"{sid}: {label}", fontsize=11, fontweight="bold")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_xlim(0, max(max(adj), max(raw)) * 1.2)

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"ranking_{sid}.png"), dpi=180,
                bbox_inches="tight")
    plt.close()


# ───────────────────────────────────────────────────────────────
# Fig 3: Cross-scenario overview
# ───────────────────────────────────────────────────────────────

def fig_cross_scenario(results_data: dict, outdir: str):
    """
    Multi-panel summary: best strategy distance + match conditions
    across all scenarios.
    """
    plt.rcParams.update(PLT_STYLE)

    scenarios = results_data["results"]
    sids = [s["scenario_id"] for s in scenarios]
    best_adj = [s["best_strategy"]["adjusted_distance"] for s in scenarios]
    best_raw = [s["best_strategy"]["raw_distance"] for s in scenarios]
    best_names = [s["best_strategy"]["strategy"] for s in scenarios]
    baseline_adj = [s["baseline_strategy"]["adjusted_distance"] for s in scenarios]

    # Match conditions for secondary axis
    fatigue = [s["match_conditions"]["fatigue_level"] for s in scenarios]
    morale = [s["match_conditions"]["morale"] for s in scenarios]
    score_diff = [s["match_conditions"]["score_diff"] for s in scenarios]
    time_rem = [s["match_conditions"]["time_remaining"] for s in scenarios]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), height_ratios=[3, 2],
                                     sharex=True)

    # --- Top panel: distances ---
    x = np.arange(len(sids))
    w = 0.28
    ax1.bar(x - w, baseline_adj, w, color="#bdc3c7", alpha=0.7, label="Baseline (raw)")
    ax1.bar(x, best_adj, w, color="#e74c3c", alpha=0.85, label="DSS adjusted")
    ax1.bar(x + w, best_raw, w, color="#3498db", alpha=0.6, label="Best raw dist.")

    for i, name in enumerate(best_names):
        short = name[:20] + "…" if len(name) > 20 else name
        ax1.text(i, best_adj[i] + 0.008, short, ha="center", fontsize=7,
                 rotation=15, fontweight="bold")

    ax1.set_ylabel("Distance")
    ax1.set_title("Cross-Scenario Overview: Strategy Selection & Context",
                  fontsize=12, fontweight="bold")
    ax1.legend(fontsize=8, loc="upper left")

    # --- Bottom panel: match conditions ---
    ax2.plot(x, fatigue, "o-", color="#e67e22", label="Fatigue", linewidth=1.5)
    ax2.plot(x, morale, "s-", color="#2ecc71", label="Morale", linewidth=1.5)

    ax2_twin = ax2.twinx()
    ax2_twin.bar(x - 0.15, score_diff, 0.3, color="#9b59b6", alpha=0.4,
                 label="Score diff")
    ax2_twin.plot(x, [t / 90.0 for t in time_rem], "d--", color="#3498db",
                  alpha=0.7, label="Time left (norm.)", linewidth=1.2)

    ax2.set_xticks(x)
    ax2.set_xticklabels([f"{s}\n{scenarios[i]['scenario_label'][:25]}"
                          for i, s in enumerate(sids)], fontsize=7)
    ax2.set_ylabel("Fatigue / Morale (0-1)")
    ax2.set_ylim(0, 1.05)
    ax2_twin.set_ylabel("Score Diff / Time (norm.)")

    # Merge legends
    h1, l1 = ax2.get_legend_handles_labels()
    h2, l2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2, fontsize=7, loc="upper right", ncol=2)

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "cross_scenario_overview.png"), dpi=180,
                bbox_inches="tight")
    plt.close()


# ───────────────────────────────────────────────────────────────
# Fig 4: Baseline vs DSS delta
# ───────────────────────────────────────────────────────────────

def fig_baseline_delta(results_data: dict, outdir: str):
    """
    Shows the adjustment delta (raw - adjusted) per scenario,
    highlighting where dynamic weighting helps most.
    """
    plt.rcParams.update(PLT_STYLE)

    scenarios = results_data["results"]
    sids = [s["scenario_id"] for s in scenarios]

    # For each scenario, show delta for top-5 strategies
    fig, axes = plt.subplots(1, len(scenarios), figsize=(3.2 * len(scenarios), 4),
                              sharey=True)
    if len(scenarios) == 1:
        axes = [axes]

    for ax, sc in zip(axes, scenarios):
        ranking = sc["ranking"]
        names = [r["strategy"][:18] for r in ranking]
        raw = [r["raw_distance"] for r in ranking]
        adj = [r["adjusted_distance"] for r in ranking]
        delta = [r - a for r, a in zip(raw, adj)]

        colors = ["#2ecc71" if d > 0 else "#e74c3c" for d in delta]
        y = np.arange(len(names))

        ax.barh(y, delta, color=colors, alpha=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels(names, fontsize=7)
        ax.axvline(0, color="#2c3e50", linewidth=0.8)
        ax.set_xlabel("Δ (raw − adj)", fontsize=8)
        ax.set_title(sc["scenario_id"], fontsize=10, fontweight="bold")

    fig.suptitle("Context Adjustment Delta per Scenario\n"
                 "(green = DSS reduced distance = context-fit bonus)",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "baseline_delta.png"), dpi=180,
                bbox_inches="tight")
    plt.close()


# ───────────────────────────────────────────────────────────────
# Master generator
# ───────────────────────────────────────────────────────────────

def generate_all_figures(
    results_path: str,
    input_path: str,
    strategies_path: str,
    outdir: str = "figures",
):
    """Generate all figures from JSON data. Returns list of created files."""
    os.makedirs(outdir, exist_ok=True)
    data = load_all(results_path, input_path, strategies_path)
    results = data["results"]
    created = []

    # Per-scenario figures
    for sc in results["results"]:
        fig_radar_scenario(data, sc, outdir)
        created.append(f"radar_{sc['scenario_id']}.png")

        fig_ranking_bars(sc, outdir)
        created.append(f"ranking_{sc['scenario_id']}.png")

    # Cross-scenario
    fig_cross_scenario(results, outdir)
    created.append("cross_scenario_overview.png")

    # Baseline delta
    fig_baseline_delta(results, outdir)
    created.append("baseline_delta.png")

    return created


# ───────────────────────────────────────────────────────────────
# CLI standalone
# ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="DSS Figure Generator")
    parser.add_argument("--results", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--strategies", default="strategy_templates.json")
    parser.add_argument("--outdir", default="figures")
    args = parser.parse_args()

    files = generate_all_figures(args.results, args.input, args.strategies, args.outdir)
    print(f"[OK] {len(files)} figures generated in {args.outdir}/")
    for f in files:
        print(f"  → {f}")
