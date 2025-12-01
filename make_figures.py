# make_figures.py
# Reproducible figure generator for the Experimental Evaluation section
# Requires: Python 3.9+, numpy, pandas, matplotlib

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

# ----------------------------
# Global config
# ----------------------------
FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)

ATTRIBUTES = [
    "A1 Offensive Strength", "A2 Defensive Strength", "A3 Midfield Control",
    "A4 Transition Speed", "A5 High Press Capability", "A6 Width Play",
    "A7 Psychological Resilience", "A8 Residual Energy", "A9 Team Morale",
    "A10 Time Management", "A11 Tactical Cohesion", "A12 Technical Base",
    "A13 Physical Base", "A14 Relational Cohesion"
]

# Minimal strategy set for examples (replace with your 20-template matrix if desired)
STRATEGY_TEMPLATES = {
    "High Press":               [0.7,0.8,0.6,0.9,0.9,0.6,0.8,0.7,0.8,0.6,0.9,0.7,0.8,0.8],
    "Fast Counterattack":       [0.9,0.6,0.5,0.9,0.5,0.6,0.7,0.8,0.7,0.8,0.6,0.7,0.8,0.6],
    "Positional Defense":       [0.4,0.9,0.8,0.3,0.2,0.3,0.7,0.6,0.6,0.9,0.8,0.6,0.5,0.7],
    "Build-up Play":            [0.7,0.6,0.8,0.5,0.4,0.6,0.7,0.6,0.8,0.7,0.8,0.8,0.6,0.8],
    "Direct Vertical Attack":   [0.85,0.5,0.5,0.9,0.4,0.7,0.6,0.7,0.7,0.7,0.6,0.7,0.8,0.6],
}

# ----------------------------
# Core model utilities
# ----------------------------
def euclidean_distance(x, y, w=None):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if w is None:
        return float(np.sqrt(np.sum((x - y) ** 2)))
    w = np.asarray(w, dtype=float)
    return float(np.sqrt(np.sum(w * (x - y) ** 2)))

def make_dynamic_weights(team_vec, context):
    """
    context: dict with keys
        - energy (0..1)         -> if low, penalize A5 (press), favor A10 (time mgmt)
        - tech_gap (-1..1)      -> negative => favor A2,A11; reduce A1,A6
        - phys_gap (-1..1)      -> negative => favor A2,A11; reduce A1,A13
        - time_pressure (0..1)  -> high => favor A4,A6
    returns weights w (len=14), normalized around 1.0
    """
    w = np.ones(14)

    energy = context.get("energy", 0.7)
    tech_gap = context.get("tech_gap", 0.0)
    phys_gap = context.get("phys_gap", 0.0)
    tpress = context.get("time_pressure", 0.3)

    # Energy effects
    if energy < 0.5:
        w[4] *= 1.3  # A5 High Press Capability (penalize mismatch)
        w[9] *= 1.2  # A10 Time Management (more weight)
        w[7] *= 1.2  # A8 Residual Energy (more weight)

    # Technical/physical gaps
    if tech_gap < 0:
        w[1] *= 1.2  # A2 Defensive Strength
        w[10] *= 1.2 # A11 Tactical Cohesion
        w[0] *= 0.9  # A1 Offensive Strength
        w[5] *= 0.9  # A6 Width Play
    if phys_gap < 0:
        w[1] *= 1.1
        w[10] *= 1.1
        w[12] *= 1.1 # A13 Physical Base important
        w[0] *= 0.95

    # Time pressure
    if tpress > 0.6:
        w[3] *= 1.2  # A4 Transition Speed
        w[5] *= 1.1  # A6 Width Play

    # Keep weights within reasonable bounds
    w = np.clip(w, 0.6, 1.6)
    return w

def adapted_distance(team_vec, strat_vec, context, lam=0.5):
    """
    lam controls how strongly contextual penalties apply.
    Contextual penalty here is modeled through dynamic weights.
    """
    w = make_dynamic_weights(team_vec, context)
    base = euclidean_distance(team_vec, strat_vec)
    adapted = euclidean_distance(team_vec, strat_vec, w=w)
    return (1 - lam) * base + lam * adapted

def rank_strategies(team_vec, strategies, context, lam=0.5):
    rows = []
    for name, vec in strategies.items():
        d = adapted_distance(team_vec, vec, context, lam)
        rows.append((name, float(d)))
    rows.sort(key=lambda x: x[1])
    return rows  # list of (name, distance)

# ----------------------------
# Figure helpers
# ----------------------------
def radar_plot(team_vec, strategies_to_show, title, outfile):
    labels = [a.split(" ", 1)[1] for a in ATTRIBUTES]  # cleaner short labels
    team = np.array(team_vec, dtype=float)

    N = len(labels)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    def close_loop(arr):
        arr = np.asarray(arr, dtype=float)
        return np.concatenate([arr, [arr[0]]])

    plt.figure(figsize=(9,6))
    ax = plt.subplot(111, polar=True)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    # grid & labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_rlabel_position(0)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8])
    ax.set_yticklabels(["0.2","0.4","0.6","0.8"], fontsize=8)

    ax.plot(angles, close_loop(team), linewidth=2, linestyle="dashed", label="Team Profile")
    ax.fill(angles, close_loop(team), alpha=0.05)

    for name, vec in strategies_to_show.items():
        ax.plot(angles, close_loop(vec), linewidth=2, label=f"Strategy: {name}")

    ax.legend(loc="lower left", bbox_to_anchor=(1.02, 0.0), fontsize=9)
    ax.set_title(title, fontsize=14, pad=12)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, outfile), dpi=200)
    plt.close()

def plot_sensitivity(team_vec, strategies, context, lambda_values, outfile):
    best_dist = []
    for lam in lambda_values:
        ranked = rank_strategies(team_vec, strategies, context, lam=lam)
        best_dist.append(ranked[0][1])

    plt.figure(figsize=(7,4))
    plt.plot(lambda_values, best_dist, marker="o")
    plt.xlabel(r"$\lambda$ (contextual weight)")
    plt.ylabel("Adapted distance of best strategy")
    plt.title("Sensitivity to $\lambda$")
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, outfile), dpi=200)
    plt.close()

def robustness_test(team_vec, strategies, context, lam=0.5, N=200, noise=0.05):
    choices = []
    t = np.asarray(team_vec, dtype=float)
    for _ in range(N):
        noisy = np.clip(t + np.random.uniform(-noise, noise, size=t.shape), 0, 1)
        ranked = rank_strategies(noisy, strategies, context, lam=lam)
        choices.append(ranked[0][0])
    counts = Counter(choices)
    total = sum(counts.values())
    names = list(counts.keys())
    perc = [100.0 * counts[n] / total for n in names]
    return names, perc

def plot_robustness(names, perc, outfile):
    order = np.argsort(perc)[::-1]
    names = [names[i] for i in order]
    perc  = [perc[i] for i in order]
    plt.figure(figsize=(8,4))
    plt.bar(names, perc)
    plt.ylabel("% selection as best strategy")
    plt.title("Robustness to Input Noise")
    plt.xticks(rotation=20, ha="right")
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, outfile), dpi=200)
    plt.close()

def ablation_study(team_vec, strategies, context, lam=0.5):
    base_rank = rank_strategies(team_vec, strategies, context, lam=lam)[0]
    base_best, base_dist = base_rank
    dists = []
    for i in range(len(team_vec)):
        modified = np.array(team_vec, dtype=float)
        modified[i] = 0.0
        top = rank_strategies(modified, strategies, context, lam=lam)[0]
        dists.append((f"A{i+1}", top[0], top[1]))
    return base_best, base_dist, dists

def plot_ablation(dists, outfile):
    labels = [x[0] for x in dists]
    values = [x[2] for x in dists]
    plt.figure(figsize=(9,4))
    plt.bar(labels, values)
    plt.xlabel("Removed attribute")
    plt.ylabel("Adapted distance (best strategy)")
    plt.title("Ablation Study")
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, outfile), dpi=200)
    plt.close()

def plot_attribute_importance(importances, outfile):
    # importances: dict { "A1": value, ... }
    items = sorted(importances.items(), key=lambda kv: kv[1], reverse=True)[:5]
    names = [k for k,_ in items]
    vals  = [v for _,v in items]
    plt.figure(figsize=(7,4))
    plt.bar(names, vals)
    plt.ylabel("Relative importance (normalized)")
    plt.title("Top Macro-Attributes by Impact")
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, outfile), dpi=200)
    plt.close()

# ----------------------------
# Demo scenarios (replace with your vectors if available)
# ----------------------------
def scenario_vectors(seed=7):
    rng = np.random.default_rng(seed)
    team = rng.uniform(0.5, 0.8, 14)        # balanced team
    opp  = rng.uniform(0.5, 0.8, 14)

    # Four scenario contexts
    S1 = {"energy":0.8, "tech_gap":0.0, "phys_gap":0.0, "time_pressure":0.3}
    S2 = {"energy":0.3, "tech_gap":-0.2, "phys_gap":-0.2, "time_pressure":0.4}
    S3 = {"energy":0.5, "tech_gap":-0.1, "phys_gap":0.0, "time_pressure":0.8}
    S4 = {"energy":0.7, "tech_gap": 0.2, "phys_gap": 0.2, "time_pressure":0.3}
    return team, opp, [S1,S2,S3,S4]

# ----------------------------
# Main: generate all figures
# ----------------------------
if __name__ == "__main__":
    team_vec, opp_vec, scenarios = scenario_vectors()

    # 1) Radar for scenario 1–4: team vs best 3 strategies (by lam=0.5)
    for idx, ctx in enumerate(scenarios, start=1):
        ranked = rank_strategies(team_vec, STRATEGY_TEMPLATES, ctx, lam=0.5)
        top3 = dict((name, STRATEGY_TEMPLATES[name]) for name,_ in ranked[:3])
        radar_plot(
            team_vec,
            strategies_to_show=top3,
            title=f"Scenario {idx}: Team vs. Top Strategies",
            outfile=f"radar_scenario{idx}.png"
        )

    # 2) Sensitivity to lambda (Scenario 1)
    lambda_values = [0.1,0.2,0.3,0.4,0.5,0.6,0.7]
    plot_sensitivity(team_vec, STRATEGY_TEMPLATES, scenarios[0], lambda_values, "sensitivity_all.png")

    # 3) Robustness (Scenario 2)
    names, perc = robustness_test(team_vec, STRATEGY_TEMPLATES, scenarios[1], lam=0.5, N=200, noise=0.05)
    plot_robustness(names, perc, "robustness_s2.png")

    # 4) Ablation (Scenario 3)
    base_best, base_dist, dists = ablation_study(team_vec, STRATEGY_TEMPLATES, scenarios[2], lam=0.5)
    plot_ablation(dists, "ablation_s3.png")

    # 5) Aggregate attribute importance (toy example:
    #    rank attributes by average distance increase from ablation)
    increases = {}
    base = base_dist
    for lab, _, dist in dists:
        increases[lab] = max(0.0, dist - base)
    # normalize to [0,1]
    if increases:
        mx = max(increases.values()) or 1.0
        increases = {k: v/mx for k,v in increases.items()}
    plot_attribute_importance(increases, "attribute_importance.png")

    print(f"Done. Figures written to: {os.path.abspath(FIG_DIR)}")
