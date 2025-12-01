"""
Pilot Validation Computation Script
====================================
Computes semantic distances for the C-Junioren match (SSV Pachten vs JSG Stausee-Losheim)
to fill in the placeholders in the pilot validation section.

Based on the football_strategy_generation code and make_figures utilities.
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# Create output directory
os.makedirs("figures", exist_ok=True)

# =============================================================================
# 1. ATTRIBUTE DEFINITIONS (subset used in pilot)
# =============================================================================

# Full 14 attributes for reference
FULL_ATTRIBUTES = [
    "A1 Offensive Strength", "A2 Defensive Strength", "A3 Midfield Control",
    "A4 Transition Speed", "A5 High Press Capability", "A6 Width Play",
    "A7 Psychological Resilience", "A8 Residual Energy", "A9 Team Morale",
    "A10 Time Management", "A11 Tactical Cohesion", "A12 Technical Base",
    "A13 Physical Base", "A14 Relational Cohesion"
]

# Pilot study uses 5 unique attributes (mapped from 6 observations)
PILOT_ATTRIBUTES = ["A1", "A2", "A4", "A5", "A8"]
PILOT_LABELS = [
    "Offensive Strength", "Defensive Strength", "Transition Speed", 
    "High Press Capability", "Residual Energy"
]

# =============================================================================
# 2. OBSERVED DATA FROM THE MATCH
# =============================================================================

# Categorical to continuous mapping
def cat_to_cont(niveau):
    """Convert German categorical scale to continuous [0,1]"""
    mapping = {
        "Hoch": 0.85,
        "Mittel": 0.50,
        "Niedrig": 0.20
    }
    return mapping.get(niveau, 0.50)

# First Half observations (SSV Pachten)
first_half_obs = {
    "Offensivkraft": "Hoch",           # -> A1
    "Direkte vertikale Angriffe": "Hoch",  # -> A4 (one of two)
    "Gegenangriff": "Hoch",            # -> A4 (one of two)
    "Kompakte Defensive": "Mittel",    # -> A2
    "Restenergie": "Mittel",           # -> A8
    "Gegenpressing": "Mittel"          # -> A5
}

# Second Half observations
second_half_obs = {
    "Offensivkraft": "Hoch",
    "Direkte vertikale Angriffe": "Mittel",
    "Gegenangriff": "Hoch",
    "Kompakte Defensive": "Niedrig",
    "Restenergie": "Niedrig",
    "Gegenpressing": "Mittel"
}

def observations_to_vector(obs):
    """
    Convert observations to 5-dimensional vector [A1, A2, A4, A5, A8]
    Note: A4 uses max of Direkte vertikale Angriffe and Gegenangriff
    """
    a1 = cat_to_cont(obs["Offensivkraft"])
    a2 = cat_to_cont(obs["Kompakte Defensive"])
    # A4: take maximum of two transition-related observations
    a4_vert = cat_to_cont(obs["Direkte vertikale Angriffe"])
    a4_counter = cat_to_cont(obs["Gegenangriff"])
    a4 = max(a4_vert, a4_counter)
    a5 = cat_to_cont(obs["Gegenpressing"])
    a8 = cat_to_cont(obs["Restenergie"])
    
    return np.array([a1, a2, a4, a5, a8])

# Compute vectors
V_first_half = observations_to_vector(first_half_obs)
V_second_half = observations_to_vector(second_half_obs)

# Halftime projection: apply fatigue discount to A8
fatigue_discount = 0.15
V_halftime_projected = V_first_half.copy()
V_halftime_projected[4] = max(0.0, V_first_half[4] - fatigue_discount)  # A8 index = 4

print("=" * 60)
print("PILOT VALIDATION: Vector Computations")
print("=" * 60)
print("\n1. OBSERVED VECTORS")
print(f"   First Half:      {V_first_half}")
print(f"   Second Half:     {V_second_half}")
print(f"   HT Projected:    {V_halftime_projected}")
print(f"   Delta (H2-H1):   {V_second_half - V_first_half}")

# =============================================================================
# 3. STRATEGY TEMPLATES (reduced to 5 dimensions)
# =============================================================================

# Original 14-dim strategy templates from the main code
FULL_STRATEGY_TEMPLATES = {
    "Fast Counterattack": [0.9, 0.6, 0.5, 0.9, 0.5, 0.6, 0.7, 0.8, 0.7, 0.8, 0.6, 0.7, 0.8, 0.6],
    "Positional Defense": [0.4, 0.9, 0.8, 0.3, 0.2, 0.3, 0.7, 0.6, 0.6, 0.9, 0.8, 0.6, 0.5, 0.7],
    "High Pressing": [0.7, 0.8, 0.6, 0.9, 0.9, 0.5, 0.8, 0.7, 0.8, 0.6, 0.9, 0.7, 0.8, 0.8],
    "Gegenpressing": [0.7, 0.8, 0.6, 0.8, 0.9, 0.5, 0.8, 0.7, 0.8, 0.6, 0.9, 0.7, 0.8, 0.8],
    "Build-up Play": [0.8, 0.5, 0.7, 0.5, 0.4, 0.6, 0.7, 0.6, 0.7, 0.6, 0.8, 0.7, 0.6, 0.7],
}

def reduce_to_pilot_dims(full_vec):
    """Extract [A1, A2, A4, A5, A8] from full 14-dim vector (0-indexed: 0, 1, 3, 4, 7)"""
    indices = [0, 1, 3, 4, 7]  # A1, A2, A4, A5, A8
    return np.array([full_vec[i] for i in indices])

# Create reduced strategy templates
PILOT_STRATEGIES = {name: reduce_to_pilot_dims(vec) for name, vec in FULL_STRATEGY_TEMPLATES.items()}

print("\n2. STRATEGY TEMPLATES (5-dim)")
for name, vec in PILOT_STRATEGIES.items():
    print(f"   {name}: {vec}")

# =============================================================================
# 4. DISTANCE COMPUTATIONS
# =============================================================================

def euclidean_distance(x, y):
    """Standard Euclidean distance"""
    return float(np.sqrt(np.sum((np.array(x) - np.array(y)) ** 2)))

def make_dynamic_weights_pilot(team_vec, energy_projected):
    """
    Simplified dynamic weights for pilot study.
    team_vec: [A1, A2, A4, A5, A8]
    """
    w = np.ones(5)
    
    # Energy effects (A8 is index 4)
    if energy_projected < 0.5:
        w[3] *= 1.3  # A5 High Press - penalize mismatch when tired
        w[4] *= 1.2  # A8 Energy - more weight on energy match
    
    return w

def adapted_distance_pilot(team_vec, strat_vec, energy_projected, lam=0.5):
    """
    Adapted distance for pilot study.
    Combines base Euclidean with weighted version.
    """
    w = make_dynamic_weights_pilot(team_vec, energy_projected)
    base = euclidean_distance(team_vec, strat_vec)
    weighted = float(np.sqrt(np.sum(w * (team_vec - strat_vec) ** 2)))
    return (1 - lam) * base + lam * weighted

# Compute distances for halftime projected vector
print("\n3. SEMANTIC DISTANCES (at halftime, with fatigue projection)")
print("-" * 60)
print(f"{'Strategy':<25} {'d_eucl':>10} {'d_adapt':>10}")
print("-" * 60)

results = []
for name, strat_vec in PILOT_STRATEGIES.items():
    d_eucl = euclidean_distance(V_halftime_projected, strat_vec)
    d_adapt = adapted_distance_pilot(V_halftime_projected, strat_vec, 
                                      energy_projected=V_halftime_projected[4], lam=0.5)
    results.append((name, d_eucl, d_adapt))
    print(f"{name:<25} {d_eucl:>10.4f} {d_adapt:>10.4f}")

# Sort by adapted distance
results.sort(key=lambda x: x[2])

print("-" * 60)
print("\nRANKING (by adapted distance):")
for i, (name, d_e, d_a) in enumerate(results, 1):
    print(f"   {i}. {name}: {d_a:.4f}")

best_strategy = results[0][0]
print(f"\n*** DSS RECOMMENDATION: {best_strategy} ***")

# =============================================================================
# 5. GENERATE RADAR CHART
# =============================================================================

def radar_plot_pilot(team_vec, strategies_to_show, title, outfile):
    """Generate radar plot for pilot study (5 dimensions)"""
    labels = PILOT_LABELS
    N = len(labels)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]
    
    def close_loop(arr):
        arr = np.asarray(arr, dtype=float)
        return np.concatenate([arr, [arr[0]]])
    
    fig, ax = plt.subplots(figsize=(8, 6), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_rlabel_position(0)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8"], fontsize=8)
    ax.set_ylim(0, 1)
    
    # Team profile
    ax.plot(angles, close_loop(team_vec), linewidth=2, linestyle="solid", 
            label="Team Profile (HT projected)", color="blue")
    ax.fill(angles, close_loop(team_vec), alpha=0.1, color="blue")
    
    # Strategies
    colors = ["green", "orange", "red"]
    for i, (name, vec) in enumerate(strategies_to_show.items()):
        ax.plot(angles, close_loop(vec), linewidth=2, linestyle="--",
                label=f"Strategy: {name}", color=colors[i % len(colors)])
    
    ax.legend(loc="lower left", bbox_to_anchor=(1.05, 0.0), fontsize=9)
    ax.set_title(title, fontsize=12, pad=15)
    plt.tight_layout()
    plt.savefig(outfile, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"\n[OK] Radar chart saved: {outfile}")

# Generate radar comparing team with top 3 strategies
top3_names = [r[0] for r in results[:3]]
top3_strategies = {name: PILOT_STRATEGIES[name] for name in top3_names}

radar_plot_pilot(
    V_halftime_projected,
    top3_strategies,
    "Pilot Study: Team Profile vs. Top Strategies (Halftime)",
    "figures/pilot_radar_halftime.png"
)

# =============================================================================
# 6. DIAGNOSTIC ANALYSIS
# =============================================================================

print("\n4. DIAGNOSTIC ANALYSIS FOR RECOMMENDED STRATEGY")
print("=" * 60)

best_vec = PILOT_STRATEGIES[best_strategy]
deltas = best_vec - V_halftime_projected

print(f"\nStrategy: {best_strategy}")
print(f"{'Attribute':<25} {'Team':>8} {'Strategy':>10} {'Delta':>8}")
print("-" * 55)
for i, (attr, team_v, strat_v, delta) in enumerate(zip(PILOT_LABELS, V_halftime_projected, best_vec, deltas)):
    sign = "+" if delta > 0 else ""
    print(f"{attr:<25} {team_v:>8.2f} {strat_v:>10.2f} {sign}{delta:>7.2f}")

print("\nKey factors driving recommendation:")
# Identify strengths and constraints
strengths = []
constraints = []
for i, (attr, delta) in enumerate(zip(PILOT_LABELS, deltas)):
    if abs(delta) < 0.15:  # Good alignment
        strengths.append((attr, V_halftime_projected[i]))
    elif delta > 0.15:  # Team lacks this
        constraints.append((attr, delta))

print("\n  STRENGTHS (good alignment):")
for attr, val in strengths:
    print(f"    - {attr}: {val:.2f}")

print("\n  CONSTRAINTS (team shortfall):")
for attr, delta in constraints:
    print(f"    - {attr}: gap of {delta:+.2f}")

# =============================================================================
# 7. COMPARISON: OBSERVED SECOND HALF vs RECOMMENDATION
# =============================================================================

print("\n5. RETROSPECTIVE ANALYSIS")
print("=" * 60)

print("\nObserved second-half profile vs. DSS recommendation:")
print(f"{'Attribute':<25} {'Observed':>10} {'Recommended':>12} {'Match?':>8}")
print("-" * 60)

# What the second half actually looked like
observed_profile_desc = {
    "Offensive Strength": "High (0.85)",
    "Defensive Strength": "Low (0.20)", 
    "Transition Speed": "High (0.85)",
    "High Press Capability": "Medium (0.50)",
    "Residual Energy": "Low (0.20)"
}

for i, attr in enumerate(PILOT_LABELS):
    obs = V_second_half[i]
    rec = best_vec[i]
    match = "✓" if abs(obs - rec) < 0.2 else "✗"
    print(f"{attr:<25} {obs:>10.2f} {rec:>12.2f} {match:>8}")

# =============================================================================
# 8. OUTPUT LATEX TABLE
# =============================================================================

print("\n" + "=" * 60)
print("LATEX TABLE OUTPUT (for paper)")
print("=" * 60)

print("""
\\begin{table}[htbp]
\\centering
\\caption{Semantic distances to candidate strategies at halftime (projected second-half state).}
\\label{tab:strategy_distances}
\\begin{tabular}{lcc}
\\hline
\\textbf{Strategy} & \\textbf{$d_{\\text{eucl}}$} & \\textbf{$d_{\\text{adapt}}$} \\\\
\\hline""")

for name, d_e, d_a in results:
    print(f"{name} & {d_e:.4f} & {d_a:.4f} \\\\")

print("""\\hline
\\end{tabular}
\\end{table}
""")

print("\nDSS Recommendation statement for paper:")
print(f'\\textbf{{{best_strategy}}}')

# =============================================================================
# 9. SAVE SUMMARY
# =============================================================================

with open("figures/pilot_results_summary.txt", "w") as f:
    f.write("PILOT VALIDATION RESULTS SUMMARY\n")
    f.write("=" * 50 + "\n\n")
    f.write(f"Match: SSV Pachten vs JSG Stausee-Losheim (4:3)\n")
    f.write(f"Competition: C-Junioren German Youth Championship\n\n")
    f.write("Halftime Projected Team Vector:\n")
    f.write(f"  [A1, A2, A4, A5, A8] = {list(V_halftime_projected)}\n\n")
    f.write("Strategy Rankings (by adapted distance):\n")
    for i, (name, d_e, d_a) in enumerate(results, 1):
        f.write(f"  {i}. {name}: d_eucl={d_e:.4f}, d_adapt={d_a:.4f}\n")
    f.write(f"\nDSS RECOMMENDATION: {best_strategy}\n")

print("\n[OK] Summary saved to figures/pilot_results_summary.txt")
print("\nDone!")
