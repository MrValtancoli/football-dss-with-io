# LIBRARIES
import numpy as np
import pandas as pd
import random
import os
import csv
import json
import matplotlib.pyplot as plt
import re
from datetime import datetime

# CONFIGURATION AND CONSTANTS

# SEED configuration for reproducibility
SEED = 41
np.random.seed(SEED)
random.seed(SEED)

MACRO_ATTR_LABELS = {
    'A1': 'Offensive Strength',
    'A2': 'Defensive Strength',
    'A3': 'Midfield Control',
    'A4': 'Transition Speed',
    'A5': 'High Press Capability',
    'A6': 'Width Utilization',
    'A7': 'Psychological Resilience',
    'A8': 'Residual Energy',
    'A9': 'Team Morale',
    'A10': 'Time Management',
    'A11': 'Tactical Cohesion',
    'A12': 'Technical Base',
    'A13': 'Physical Base',
    'A14': 'Relational Cohesion'
}

"""
Index:
    Main roles:
        GK: GoalKeeper
        CB: Center Back
        FB: Full Back
        CM: Central Midfielder
        FW: Forward

    Attributes:
        reflexes: reflexes
        aerial_duels: aerial duels
        passing: passing
        speed: speed
        stamina: stamina
        tackling: tackling
        interceptions: interceptions
        xG: expected goals (statistical value indicating the quality of scoring opportunities)
        xA: expected assists (statistical value indicating the quality of assist opportunities)
        aggression: aggression
        dribbling: dribbling
"""
roles = {
    'GK': {'reflexes': (0.85, 0.05), 'aerial_duels': (0.8, 0.05), 'passing': (0.65, 0.1),
           'speed': (0.4, 0.1), 'stamina': (0.7, 0.05), 'resilience': (0.8, 0.05),
           'dribbling': (0.3, 0.1), 'tackling': (0.2, 0.1), 'interceptions': (0.3, 0.1),
           'xG': (0.0, 0.0), 'xA': (0.2, 0.1), 'aggression': (0.6, 0.1)},
    'CB': {'reflexes': (0.3, 0.1), 'aerial_duels': (0.9, 0.05), 'passing': (0.7, 0.1),
           'speed': (0.5, 0.1), 'stamina': (0.75, 0.05), 'resilience': (0.8, 0.05),
           'dribbling': (0.5, 0.1), 'tackling': (0.85, 0.05), 'interceptions': (0.8, 0.05),
           'xG': (0.1, 0.05), 'xA': (0.2, 0.1), 'aggression': (0.8, 0.05)},
    'FB': {'reflexes': (0.4, 0.1), 'aerial_duels': (0.7, 0.1), 'passing': (0.75, 0.05),
           'speed': (0.75, 0.1), 'stamina': (0.8, 0.05), 'resilience': (0.75, 0.05),
           'dribbling': (0.7, 0.05), 'tackling': (0.7, 0.1), 'interceptions': (0.7, 0.1),
           'xG': (0.2, 0.1), 'xA': (0.5, 0.1), 'aggression': (0.7, 0.1)},
    'CM': {'reflexes': (0.4, 0.1), 'aerial_duels': (0.7, 0.1), 'passing': (0.8, 0.05),
           'speed': (0.7, 0.1), 'stamina': (0.8, 0.05), 'resilience': (0.8, 0.05),
           'dribbling': (0.75, 0.05), 'tackling': (0.7, 0.1), 'interceptions': (0.75, 0.1),
           'xG': (0.4, 0.1), 'xA': (0.7, 0.1), 'aggression': (0.75, 0.1)},
    'FW': {'reflexes': (0.4, 0.1), 'aerial_duels': (0.65, 0.1), 'passing': (0.7, 0.1),
           'speed': (0.8, 0.05), 'stamina': (0.8, 0.05), 'resilience': (0.7, 0.05),
           'dribbling': (0.85, 0.05), 'tackling': (0.4, 0.1), 'interceptions': (0.4, 0.1),
           'xG': (0.85, 0.05), 'xA': (0.6, 0.1), 'aggression': (0.75, 0.1)}
}

# Strategies encoded as vectors of 14 macro-attributes for 20 tactical choices, normalized between 0 (irrelevant) and 1 (critical)
"""
Build-up Play (Possession-based): Focuses on ball possession control and slow build-up to create opportunities.
Fast Counterattack: Prioritizes rapid transition from defense to attack.
Long Ball to Target Man: A direct approach relying on long balls to bypass midfield.
Late Midfield Runners: Relies on midfielders making late runs into the box or offensive positions after the attack has already started, creating surprise for the opposing defense.
Systematic Crossing: Exploits wide areas to cross the ball into the box, aiming to beat the defense with long, precise passes from the flanks.
Overlapping Flanks: Exploits the wings, with fullbacks overlapping wingers to create numerical superiority.
Quick Rotations in Attack: Focuses on continuous position exchanges among attackers to disorganize the opposing defense.
Direct Vertical Attack: Seeks to move the ball forward as quickly as possible with direct passes and few touches.
Classic Catenaccio: An extremely solid and passive defense aimed at denying space to the opponent.
Gegenpressing: Aims to recover the ball in advanced areas to score quickly.
Positional Defense: The team deploys en masse near its own penalty area, conceding possession but denying space to penetrate and create chances.
Compact Zonal Defense: Defending specific areas of the pitch. Players move as a unified block, maintaining tight formation to close gaps between lines.
Strict Man-Marking: Each player takes responsibility for marking a specific opponent, following them across the pitch.
High Press: Immediately recover the ball when lost, pressing opponents in advanced zones to force errors or loss of possession.
Midfield Pressing: Pressure on the opponent starts in the central zone, aiming to limit their build-up and force turnovers.
Extended Possession Play: Focuses on maintaining possession to tire the opponent and find the right moment to attack.
Cautious Horizontal Play: A low-risk approach that moves the ball horizontally while waiting for penetration opportunities.
Central Block with Quick Breaks: A hybrid tactic combining solid central defense with rapid offensive incursions.
Inducing Build-up Errors: Press the opponent when building from the back, forcing passing or positioning errors.
Offside Trap: High-risk defensive tactic where the defensive line moves forward to catch opposing attackers offside.
"""
strategy_templates = [
    # Offensive systems
    {"name": "Build-up Play", "vector": [0.8, 0.5, 0.7, 0.5, 0.4, 0.6, 0.7, 0.6, 0.8, 0.7, 0.8, 0.8, 0.6, 0.8]},
    {"name": "Fast Counterattack", "vector": [0.9, 0.6, 0.5, 0.9, 0.5, 0.6, 0.7, 0.8, 0.7, 0.8, 0.6, 0.7, 0.8, 0.6]},
    {"name": "Long Ball to Target Man", "vector": [0.8, 0.6, 0.5, 0.6, 0.4, 0.4, 0.6, 0.7, 0.6, 0.7, 0.5, 0.5, 0.8, 0.5]},
    {"name": "Late Midfield Runners", "vector": [0.8, 0.5, 0.6, 0.7, 0.5, 0.5, 0.6, 0.7, 0.7, 0.6, 0.7, 0.7, 0.7, 0.6]},
    {"name": "Systematic Crossing", "vector": [0.7, 0.5, 0.6, 0.6, 0.5, 0.9, 0.7, 0.7, 0.7, 0.6, 0.7, 0.7, 0.7, 0.6]},
    {"name": "Overlapping Flanks", "vector": [0.7, 0.5, 0.7, 0.7, 0.5, 0.9, 0.7, 0.8, 0.8, 0.7, 0.8, 0.7, 0.8, 0.7]},
    {"name": "Quick Rotations in Attack", "vector": [0.8, 0.5, 0.7, 0.8, 0.6, 0.7, 0.8, 0.7, 0.8, 0.7, 0.9, 0.7, 0.8, 0.7]},
    {"name": "Direct Vertical Attack", "vector": [0.9, 0.5, 0.5, 0.8, 0.5, 0.6, 0.7, 0.7, 0.7, 0.7, 0.6, 0.7, 0.8, 0.6]},
    # Defensive structures
    {"name": "Classic Catenaccio", "vector": [0.4, 0.9, 0.7, 0.3, 0.2, 0.3, 0.8, 0.7, 0.7, 0.9, 0.8, 0.6, 0.6, 0.7]},
    {"name": "Positional Defense", "vector": [0.4, 0.9, 0.8, 0.3, 0.2, 0.3, 0.7, 0.6, 0.6, 0.9, 0.8, 0.6, 0.5, 0.7]},
    {"name": "Compact Zonal Defense", "vector": [0.5, 0.9, 0.8, 0.4, 0.4, 0.4, 0.7, 0.6, 0.7, 0.8, 0.9, 0.7, 0.6, 0.7]},
    {"name": "Strict Man-Marking", "vector": [0.5, 0.9, 0.7, 0.5, 0.5, 0.3, 0.7, 0.7, 0.6, 0.8, 0.8, 0.7, 0.7, 0.7]},
    {"name": "Offside Trap", "vector": [0.5, 0.8, 0.7, 0.5, 0.6, 0.4, 0.7, 0.7, 0.7, 0.8, 0.8, 0.7, 0.7, 0.7]},
    # Pressing variants
    {"name": "High Press", "vector": [0.7, 0.8, 0.6, 0.9, 0.9, 0.5, 0.8, 0.7, 0.8, 0.6, 0.9, 0.7, 0.8, 0.8]},
    {"name": "Gegenpressing", "vector": [0.7, 0.8, 0.6, 0.8, 0.9, 0.5, 0.8, 0.7, 0.8, 0.6, 0.9, 0.7, 0.8, 0.8]},
    {"name": "Midfield Pressing", "vector": [0.6, 0.7, 0.7, 0.7, 0.7, 0.4, 0.7, 0.7, 0.7, 0.7, 0.8, 0.7, 0.7, 0.7]},
    {"name": "Inducing Build-up Errors", "vector": [0.7, 0.8, 0.6, 0.8, 0.9, 0.4, 0.7, 0.7, 0.8, 0.6, 0.8, 0.7, 0.7, 0.8]},
    # Possession/control
    {"name": "Extended Possession Play", "vector": [0.7, 0.7, 0.9, 0.5, 0.5, 0.6, 0.8, 0.7, 0.8, 0.7, 0.9, 0.8, 0.6, 0.8]},
    {"name": "Cautious Horizontal Play", "vector": [0.5, 0.7, 0.8, 0.4, 0.3, 0.5, 0.7, 0.7, 0.8, 0.7, 0.8, 0.7, 0.5, 0.7]},
    {"name": "Central Block with Quick Breaks", "vector": [0.7, 0.8, 0.7, 0.7, 0.7, 0.5, 0.7, 0.7, 0.7, 0.7, 0.8, 0.7, 0.7, 0.7]},
]


# FUNCTIONS
"""Function to generate random values for player attributes"""
def generate_attribute(mean, std):
    return np.clip(np.random.normal(mean, std), 0.0, 1.0)

"""Functions for computing macro-attributes"""
def compute_offensive_strength(players_team):
    """Compute macro-attribute A1: Capacity to create and convert scoring opportunities"""
    relevant_roles = ['FW', 'CM']
    xG_values, dribbling_values = [], []
    for player in players_team:
        if player['role'] in relevant_roles:
            xG_values.append(player.get('xG', 0))
            dribbling_values.append(player.get('dribbling', 0))
    return 0.7 * np.mean(xG_values) + 0.3 * np.mean(dribbling_values)

def compute_defensive_strength(players_team):
    """Compute macro-attribute A2: Ability to limit opponent scoring"""
    relevant_roles = ['GK', 'CB']
    reflexes_value, tackling_value = [], []
    for player in players_team:
        if player['role'] in relevant_roles:
            reflexes_value.append(player.get('reflexes', 0))
            tackling_value.append(player.get('tackling', 0))
    return 0.7 * np.mean(reflexes_value) + 0.3 * np.mean(tackling_value)

def compute_midfield_solidity(players_team):
    """Compute macro-attribute A3: Control and filtering capability"""
    relevant_roles = ['FB', 'CM']
    xA_value, speed_value = [], []
    for player in players_team:
        if player['role'] in relevant_roles:
            xA_value.append(player.get('xA', 0))
            speed_value.append(player.get('speed', 0))
    return 0.7 * np.mean(xA_value) + 0.3 * np.mean(speed_value)

def compute_transition_speed(players_team):
    """Compute macro-attribute A4: Rapid transitions between defense and attack"""
    relevant_roles = ['CM', 'FW', 'FB']
    speed_values, stamina_values = [], []
    for player in players_team:
        if player['role'] in relevant_roles:
            speed_values.append(player.get('speed', 0))
            stamina_values.append(player.get('stamina', 0))
    return 0.7 * np.mean(speed_values) + 0.3 * np.mean(stamina_values)

def compute_high_press_capability(players_team):
    """Compute macro-attribute A5: Ball recovery in advanced zones"""
    relevant_roles = ['FW', 'CM', 'FB', 'CB']
    tackling_values, interceptions_values = [], []
    for player in players_team:
        if player['role'] in relevant_roles:
            tackling_values.append(player.get('tackling', 0))
            interceptions_values.append(player.get('interceptions', 0))
    return 0.7 * np.mean(tackling_values) + 0.3 * np.mean(interceptions_values)

def compute_width_utilization(players_team):
    """Compute macro-attribute A6: Effective use of the flanks"""
    relevant_roles = ['CM', 'FB']
    xA_values, stamina_values = [], []
    for player in players_team:
        if player['role'] in relevant_roles:
            xA_values.append(player.get('xA', 0))
            stamina_values.append(player.get('stamina', 0))
    return 0.7 * np.mean(xA_values) + 0.3 * np.mean(stamina_values)

def compute_psychological_resilience(players_team):
    """Compute macro-attribute A7: Resistance to emotional pressure"""
    resilience_values, aggression_values = [], []
    for player in players_team:
        resilience_values.append(player.get('resilience', 0))
        aggression_values.append(player.get('aggression', 0))
    return 0.7 * np.mean(resilience_values) + 0.3 * np.mean(aggression_values)

def compute_residual_energy(players_team):
    """Compute macro-attribute A8: Stamina levels across the team"""
    stamina_values, resilience_values = [], []
    for player in players_team:
        stamina_values.append(player.get('stamina', 0))
        resilience_values.append(player.get('resilience', 0))
    return 0.7 * np.mean(stamina_values) + 0.3 * np.mean(resilience_values)

def compute_team_morale(players_team):
    """Compute macro-attribute A9: Motivation and collective spirit"""
    resilience_values, aggression_values = [], []
    for player in players_team:
        resilience_values.append(player.get('resilience', 0))
        aggression_values.append(player.get('aggression', 0))
    return 0.6 * np.mean(resilience_values) + 0.4 * np.mean(aggression_values)

def compute_time_management(players_team):
    """Compute macro-attribute A10: Adaptability to clock pressure"""
    relevant_roles = ['GK', 'CM', 'FB']
    interceptions_values, passing_values = [], []
    for player in players_team:
        if player['role'] in relevant_roles:
            interceptions_values.append(player.get('interceptions', 0))
            passing_values.append(player.get('passing', 0))
    return 0.5 * np.mean(interceptions_values) + 0.5 * np.mean(passing_values)

def compute_tactical_cohesion(players_team):
    """Compute macro-attribute A11: Synchronization between units"""
    passing_values, xA_values = [], []
    for player in players_team:
        passing_values.append(player.get('passing', 0))
        xA_values.append(player.get('xA', 0))
    return 0.6 * np.mean(passing_values) + 0.4 * np.mean(xA_values)

def compute_technical_base(players_team):
    """Compute team technical base A12"""
    technical_attributes = ['reflexes', 'passing', 'dribbling', 'tackling', 'interceptions', 'xG', 'xA']
    total_values = []
    for player in players_team:
        for attr in technical_attributes:
            total_values.append(player.get(attr, 0))
    return np.mean(total_values)

def compute_physical_base(players_team):
    """Compute team physical base A13"""
    physical_attributes = ['aerial_duels', 'speed', 'stamina', 'aggression']
    total_values = []
    for player in players_team:
        for attr in physical_attributes:
            total_values.append(player.get(attr, 0))
    return np.mean(total_values)

def compute_internal_relational_cohesion():
    """Compute macro-attribute A14: Internal relationship stability"""
    return np.random.uniform(0.5, 0.9)

def aggregate_team_profile(players):
    """
    Aggregates team attributes to compute the macro-attribute profile.
    """
    profile = {
        'A1': compute_offensive_strength(players),
        'A2': compute_defensive_strength(players),
        'A3': compute_midfield_solidity(players),
        'A4': compute_transition_speed(players),
        'A5': compute_high_press_capability(players),
        'A6': compute_width_utilization(players),
        'A7': compute_psychological_resilience(players),
        'A8': compute_residual_energy(players),
        'A9': compute_team_morale(players),
        'A10': compute_time_management(players),
        'A11': compute_tactical_cohesion(players),
        'A12_base': compute_technical_base(players),
        'A13_base': compute_physical_base(players),
        'A14': compute_internal_relational_cohesion()
    }
    return profile

def generate_match_conditions_from_text(match_description, team_name):
    """
    Robust rule-based parser for scenario text.
    Extracts:
      - time_remaining (minutes)
      - score_diff (ours - theirs)
      - fatigue_level [0..1]
      - morale [0..1]
    """
    text = match_description.lower()
    conditions = {
        "time_remaining": 30,
        "score_diff": 0,
        "fatigue_level": 0.7,
        "morale": 0.7
    }

    # --- TIME ---
    # "last 10 minutes"
    m = re.search(r"last\s+(\d+)\s*min", text)
    if m:
        conditions["time_remaining"] = int(m.group(1))
    # "at 60'" -> 90 - 60 = 30
    m = re.search(r"at\s+(\d+)\s*['']", text)
    if m:
        minute = int(m.group(1))
        conditions["time_remaining"] = max(0, 90 - minute)
    # Alternative: "at the 60'" or "60th minute"
    m = re.search(r"(\d+)\s*[''](?!\w)", text)
    if m:
        minute = int(m.group(1))
        conditions["time_remaining"] = max(0, 90 - minute)

    # --- SCORE / STATUS ---
    # "winning 3-0"
    m = re.search(r"winning\s+(\d+)\s*[-:]\s*(\d+)", text)
    if m:
        conditions["score_diff"] = int(m.group(1)) - int(m.group(2))
    # "losing 1-2"
    m = re.search(r"losing\s+(\d+)\s*[-:]\s*(\d+)", text)
    if m:
        conditions["score_diff"] = int(m.group(1)) - int(m.group(2))
    # "0-0", "1:1" etc -> draw
    m = re.search(r"\b(\d+)\s*[-:]\s*(\d+)\b", text)
    if m and int(m.group(1)) == int(m.group(2)):
        conditions["score_diff"] = 0
    # "draw", "drawing", "deadlocked"
    if re.search(r"draw\w*|deadlock\w*", text):
        conditions["score_diff"] = 0
    # "down by 1 goal"
    m = re.search(r"down\s+by\s+(\w+)\s+goal", text)
    if m:
        num_word = m.group(1)
        num_map = {'one': 1, 'two': 2, 'three': 3, 'a': 1, '1': 1, '2': 2, '3': 3}
        conditions["score_diff"] = -num_map.get(num_word, 1)
    # "ahead by 2"
    m = re.search(r"ahead\s+by\s+(\d+)", text)
    if m:
        conditions["score_diff"] = int(m.group(1))
    # "behind"
    if re.search(r"behind\s+\d+-\d+", text):
        m = re.search(r"behind\s+(\d+)-(\d+)", text)
        if m:
            conditions["score_diff"] = int(m.group(1)) - int(m.group(2))
    # "we are winning/losing"
    if re.search(r"we\s+are\s+winning", text) and conditions["score_diff"] == 0:
        conditions["score_diff"] = 1
    if re.search(r"we\s+are\s+(behind|losing)", text) and conditions["score_diff"] == 0:
        conditions["score_diff"] = -1

    # --- ENERGY: "fresh", "full of energy" ---
    if re.search(r"fresh|full\s+of\s+energ\w+|energetic", text):
        conditions["fatigue_level"] = 0.35  # => high energy

    # --- TIME: "few minutes" (fallback if no explicit numbers) ---
    if re.search(r"few\s+minut\w*", text) and "time_remaining" not in conditions:
        conditions["time_remaining"] = 8

    # --- FATIGUE / ENERGY ---
    # high -> tired, exhausted, fatigued
    if re.search(r"tired|tiredness|exhaust\w*|fatigu\w*|dropping", text):
        conditions["fatigue_level"] = 0.85
    # low -> fresh, energetic, fit
    if re.search(r"fresh|energetic|fit|rested", text):
        conditions["fatigue_level"] = 0.35

    # --- MORALE ---
    if re.search(r"demorali[sz]\w*|discourag\w*|low\s+morale|unfocus\w*", text):
        conditions["morale"] = 0.4
    if re.search(r"high\s+morale|motivat\w*|confident", text):
        conditions["morale"] = 0.85

    return conditions


"""Function to compute Euclidean distance between two vectors"""
def compute_semantic_distance_updated(vector1, vector2):
    return np.sqrt(np.sum((np.array(vector1) - np.array(vector2)) ** 2))

"""Function to aggregate team macro-attributes. Corresponds to 'aggregate_context_tree' in pseudocode."""
def aggregate_context_tree(team_profile):
    """
    Aggregates team macro-attributes to obtain a single vector.
    In this case, the team profile is already the desired vector.
    """
    return list(team_profile.values())

"""
Function to apply dynamic weights based on match conditions.
This function is the core of the dynamic simulation module. Its purpose is to take a "static" distance score
(computed from fixed team and strategy profiles) and modify it based on factors evolving during the match.
It works as a penalty or bonus multiplier that adapts to the game context.
Input:
raw_distance: The raw Euclidean distance between the team profile and the strategy vector.
match_conditions: A dictionary containing dynamic match data such as energy level, time remaining, score difference,
resilience and morale.
gap_technical and gap_physical: Values measuring the technical and physical gap between your team and the opponent.
"""
### MODIFICATION ###
# Dynamic weights function v2: more incisive and strategy-aware.
def apply_dynamic_weights_v2(raw_distance, match_conditions, strategy_vector):
    """
    Applies a more incisive and "strategy-aware" adjustment factor.
    """
    energy = 1.0 - float(match_conditions.get("fatigue_level", 0.7))
    time_left = float(match_conditions.get("time_remaining", 30))
    score_diff = float(match_conditions.get("score_diff", 0))
    morale = float(match_conditions.get("morale", 0.7))

    adjustment = 1.0

    # Analysis of strategy nature based on its vector
    is_high_intensity = strategy_vector[4] > 0.7 or strategy_vector[3] > 0.7  # A5 (Pressing), A4 (Transition)
    is_ultra_offensive = strategy_vector[0] > 0.8 or strategy_vector[4] > 0.8  # A1 (Offensive), A5 (Pressing)
    is_conservative = strategy_vector[1] > 0.8 and strategy_vector[9] > 0.7  # A2 (Defensive), A10 (Time Management)
    is_possession_based = strategy_vector[10] > 0.7 and strategy_vector[2] > 0.7  # A11 (Cohesion), A3 (Midfield)

    # --- Enhanced adjustment logic ---

    # CASE 1: Desperation (behind in score, little time)
    if score_diff < 0 and time_left < 25:
        if is_ultra_offensive or is_high_intensity:
            adjustment *= 0.65  # Massive bonus (distance reduced by 35%)
        elif is_conservative or is_possession_based:
            adjustment *= 1.50  # Heavy penalty (distance increased by 50%)

    # CASE 2: Advantage management (ahead in score, little time)
    elif score_diff > 0 and time_left < 20:
        if is_conservative:
            adjustment *= 0.70  # Massive bonus for conservative tactics
        elif is_ultra_offensive:
            adjustment *= 1.60  # Heavy penalty for risky tactics

    # CASE 3: Tired team
    if energy < 0.45:
        if is_high_intensity:
            adjustment *= 1.40  # Heavy penalty for energy-demanding strategies
        else:
            adjustment *= 0.90  # Slight bonus for low-energy strategies

    # CASE 4: Morale
    if morale < 0.4:  # Demoralized team
        if strategy_vector[10] > 0.8:  # A11 (Tactical Cohesion)
            adjustment *= 1.20  # Penalty for complex strategies requiring concentration
    elif morale > 0.8:  # Euphoric team
        if is_ultra_offensive:
            adjustment *= 0.85  # Bonus to capitalize on positive momentum

    # Clamp to avoid extreme values
    adjustment = max(0.4, min(adjustment, 2.0))
    return max(0.0, raw_distance * adjustment)

# Function to select the most compatible strategy
"""
Central function of the model.
Examines all available strategies and using information from YOUR team, the opponent, and match conditions, selects the most effective tactic.
Input:
final_profile_team1: The complete vector profile of your team.
final_profile_team2: The complete vector profile of the opposing team.
strategy_templates: The list of all predefined strategies.
match_conditions: The dictionary of dynamic match parameters.
"""
def select_best_strategy_v2(final_profile_team1, final_profile_team2, strategy_templates, match_conditions, opponent_penalty_lambda=0.5):
    team_vector = list(final_profile_team1.values())
    opponent_vector = list(final_profile_team2.values())

    strategy_scores = []
    for strategy in strategy_templates:
        strategy_name = strategy["name"]
        strategy_vector = strategy["vector"]

        raw_distance_team = compute_semantic_distance_updated(team_vector, strategy_vector)
        raw_distance_opponent = compute_semantic_distance_updated(opponent_vector, strategy_vector)
        
        fit_opponent = np.exp(-raw_distance_opponent)
        combined_distance = raw_distance_team + opponent_penalty_lambda * fit_opponent

        ### MODIFICATION: Using the new weighting function ###
        adjusted_distance = apply_dynamic_weights_v2(combined_distance, match_conditions, strategy_vector)
        
        strategy_scores.append((strategy_name, adjusted_distance))

    strategy_scores.sort(key=lambda item: item[1])
    
    # Compute diagnostics only for the best
    best_strategy_name = strategy_scores[0][0]
    best_strategy_vector = next(s['vector'] for s in strategy_templates if s['name'] == best_strategy_name)
    diagnostics_data = {
        'team_attributes': final_profile_team1,
        'strategy_attributes': dict(zip(final_profile_team1.keys(), best_strategy_vector)),
    }

    return strategy_scores[0], strategy_scores, diagnostics_data

def _build_attr_rows(team_attrs: dict, strat_attrs: dict, order_keys):
    """Creates rows with (code, label, team, strategy, delta = strategy - team)."""
    rows = []
    for k in order_keys:
        team_v = float(team_attrs.get(k, 0.0))
        strat_v = float(strat_attrs.get(k, 0.0))
        delta = strat_v - team_v     # >0 = deficiency; <0 = surplus
        rows.append({
            'code': k,
            'label': MACRO_ATTR_LABELS.get(k, k),
            'team': team_v,
            'strategy': strat_v,
            'delta': delta
        })
    return rows

def print_attribute_diagnostics(best_strategy_name: str, diagnostics_data: dict, top_k: int = 3, export_csv_path: str = None):
    """
    diagnostics_data: a dictionary containing attributes for the single best strategy.
    """
    if not diagnostics_data:
        print("[WARN] Diagnostics data not available.")
        return

    # No need to search, we already have the correct block.
    team_attrs = diagnostics_data['team_attributes']
    strat_attrs = diagnostics_data['strategy_attributes']
    order_keys = [f'A{i}' for i in range(1, 15)]

    rows = _build_attr_rows(team_attrs, strat_attrs, order_keys)

    shortages = sorted(rows, key=lambda r: r['delta'], reverse=True)
    surpluses = sorted(rows, key=lambda r: r['delta'])

    print(f"\n=== Attribute Diagnostics — Strategy: {best_strategy_name} ===")
    print("\n-- Main Deficiencies (team lacks these attributes required by the strategy):")
    for r in shortages[:top_k]:
        if r['delta'] > 0.05:
            print(f"  {r['code']} {r['label']}: Team={r['team']:.3f}, Required={r['strategy']:.3f}, Delta={r['delta']:.3f}")

    print("\n-- Main Surpluses (team excels in these attributes relative to the strategy):")
    for r in surpluses[:top_k]:
        if r['delta'] < -0.05:
            print(f"  {r['code']} {r['label']}: Team={r['team']:.3f}, Required={r['strategy']:.3f}, Delta={r['delta']:.3f}")

    if export_csv_path:
        with open(export_csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['code', 'label', 'team', 'strategy', 'delta'])
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n[OK] Diagnostics exported to: {export_csv_path}")


def plot_radar_chart(team_profile, strategy_vectors, strategy_names, title, save_path=None, show=False):
    """
    Generates a radar chart to visualize and compare team and strategy profiles.
    :param team_profile: Dictionary or list of team macro-attributes.
    :param strategy_vectors: List of strategy vectors to compare.
    :param strategy_names: List of strategy names.
    :param title: Chart title.
    """
    # Labels in order A1..A14
    labels = [MACRO_ATTR_LABELS[f'A{i}'] for i in range(1, 15)]
    num_vars = len(labels)

    # Angles
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    # Team data
    team_data = [team_profile[f'A{i}'] for i in range(1, 15)]
    team_data += team_data[:1]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    ax.set_title(title, fontsize=16, y=1.1)

    # Team profile line
    line_team, = ax.plot(angles, team_data, linewidth=2, linestyle='--', label='Team Profile')
    ax.fill(angles, team_data, alpha=0.12, color=line_team.get_color())

    # Strategies (simple lines)
    for i, strategy_vector in enumerate(strategy_vectors):
        data = strategy_vector + strategy_vector[:1]
        ax.plot(angles, data, linewidth=2, label=f"Strategy: {strategy_names[i]}")

    # Polar setup
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_rlabel_position(0)
    ax.set_ylim(0, 1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=10)

    ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1.05))

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    plt.close(fig)

def make_results_dir(base="results"):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = os.path.join(base, ts)
    os.makedirs(outdir, exist_ok=True)
    return outdir

# More differentiated scenarios to test model robustness.
def generate_scenarios_v2():
    """
    Returns a list of varied match situation descriptions.
    """
    return [
        "We are down by one goal at 80', the team is tired but morale is still high after nearly equalizing.",
        "Match deadlocked at 0-0 at 60', we are fresh and playing against a technically superior opponent.",
        "We are winning 2-0 at 70', but the opponent is attacking persistently. Energy levels are dropping.",
        "We are behind 1-0 at 55', the team appears demoralized and unfocused after conceding.",
        "Balanced match at 1-1 at halftime (45'), both teams seem energetic and motivated for the second half."
    ]

def test_lambda_sensitivity(final_profile_team1, final_profile_team2, strategy_templates, scenario_desc, lambdas=None):
    """
    Runs strategy selection for a set of lambda values and shows how the ranking changes.
    """
    if lambdas is None:
        lambdas = [0.0, 0.3, 0.7, 1.0]  # typical values to test

    match_conditions = generate_match_conditions_from_text(scenario_desc, "Milan")
    print(f"\n=== Lambda Sensitivity for scenario: \"{scenario_desc}\" ===")

    results = []
    for lam in lambdas:
        best, ranking, _ = select_best_strategy_v2(
            final_profile_team1,
            final_profile_team2,
            strategy_templates,
            match_conditions,
            opponent_penalty_lambda=lam
        )
        results.append((lam, best[0], best[1]))
        print(f"lambda={lam:.2f} -> Best: {best[0]} (distance {best[1]:.4f})")

    # Print summary table
    print("\n--- Summary ---")
    for lam, name, dist in results:
        print(f"lambda={lam:.2f} : {name} (score {dist:.4f})")

    return results

def plot_lambda_sensitivity(results, scenario_idx, results_dir):
    lambdas = [r[0] for r in results]
    distances = [r[2] for r in results]

    plt.figure()
    plt.plot(lambdas, distances, marker='o')
    plt.title(f"Lambda Sensitivity - Scenario {scenario_idx}")
    plt.xlabel("lambda (opponent fit penalty)")
    plt.ylabel("Adjusted Distance")
    plt.grid(True)

    path = os.path.join(results_dir, f"sensitivity_lambda_scenario{scenario_idx}.png")
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Lambda sensitivity plot saved to: {path}")

def ablation_study(final_profile_team1, final_profile_team2, strategy_templates, scenario_desc):
    """
    True ablation: progressively remove attributes A1..A14 and observe how much
    the baseline strategy distance worsens and/or if the best strategy changes.
    """
    match_conditions = generate_match_conditions_from_text(scenario_desc, "Milan")
    print(f"\n=== Ablation Study (attribute removal) for scenario: \"{scenario_desc}\" ===")

    # --- BASELINE ---
    baseline_best, baseline_ranking, _ = select_best_strategy_v2(
        final_profile_team1, final_profile_team2, strategy_templates, match_conditions, opponent_penalty_lambda=0.5
    )
    baseline_name, baseline_score = baseline_best
    print(f"[Baseline] Best: {baseline_name} (score={baseline_score:.4f})")

    results = [("Baseline", baseline_name, baseline_score, "")]

    attr_keys = [f"A{i}" for i in range(1, 15)]

    # --- SINGLE ATTRIBUTE REMOVAL ---
    for k in attr_keys:
        mod_team = final_profile_team1.copy()
        mod_oppo = final_profile_team2.copy()
        # zero out attribute k
        mod_team[k] = 0.0
        mod_oppo[k] = 0.0

        best, _, _ = select_best_strategy_v2(mod_team, mod_oppo, strategy_templates, match_conditions, opponent_penalty_lambda=0.5)
        best_name, best_score = best
        change = "" if best_name == baseline_name else f"[CHANGED] from {baseline_name} -> {best_name}"
        print(f"- Removed {k}: best={best_name} (score={best_score:.4f}) {change}")
        results.append((k, best_name, best_score, change))

    return results

def plot_ablation_study(results, scenario_idx, results_dir):
    """
    Chart: for each removed attribute, shows how much the baseline score worsens.
    """
    baseline_score = results[0][2]
    labels = [r[0] for r in results[1:]]  # skip baseline
    scores = [r[2] - baseline_score for r in results[1:]]

    plt.figure(figsize=(10, 6))
    bars = plt.barh(labels, scores, color="skyblue")
    plt.axvline(0, color="black", lw=1)
    plt.xlabel("Δ distance vs baseline (positive = worse)")
    plt.title(f"Attribute Ablation - Scenario {scenario_idx}")
    plt.gca().invert_yaxis()

    for bar, s in zip(bars, scores):
        plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, f"{s:+.3f}", va='center')

    path = os.path.join(results_dir, f"ablation_attributes_scenario{scenario_idx}.png")
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Attribute ablation plot saved to: {path}")

def test_robustness_noise(final_profile_team1, final_profile_team2, strategy_templates, scenario_desc, n_sim=100, noise_level=0.05, opponent_penalty_lambda=0.5):
    match_conditions = generate_match_conditions_from_text(scenario_desc, "Milan")
    base_vector = list(final_profile_team1.values())
    opponent_vector = list(final_profile_team2.values())
    counts = {}
    for _ in range(n_sim):
        noisy_vector = np.clip(
            np.array(base_vector) + np.random.uniform(-noise_level, noise_level, size=len(base_vector)),
            0, 1
        )
        strategy_scores = []
        for strat in strategy_templates:
            raw_team = compute_semantic_distance_updated(noisy_vector, strat["vector"])
            raw_opp = compute_semantic_distance_updated(opponent_vector, strat["vector"])
            fit_opp = np.exp(-raw_opp)
            combined = raw_team + opponent_penalty_lambda * fit_opp
            # Correct call with strategy_vector
            combined = apply_dynamic_weights_v2(combined, match_conditions, strat["vector"])
            strategy_scores.append((strat["name"], combined))
        strategy_scores.sort(key=lambda x: x[1])
        best = strategy_scores[0][0]
        counts[best] = counts.get(best, 0) + 1
    sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    print(f"\n=== Robustness to Noise Test for scenario: \"{scenario_desc}\" ===")
    for name, cnt in sorted_counts:
        print(f"{name:35s}: selected {cnt}/{n_sim} times ({cnt/n_sim*100:.1f}%)")
    return sorted_counts

def plot_robustness_noise(sorted_counts, scenario_idx, results_dir):
    """
    Creates a bar chart with strategy selection frequency under noise.
    """
    strategies = [x[0] for x in sorted_counts]
    counts = [x[1] for x in sorted_counts]

    plt.figure(figsize=(10, 5))
    bars = plt.barh(strategies, counts, color="skyblue")
    bars[0].set_color("green")
    plt.xlabel("Number of times selected in simulations")
    plt.title(f"Robustness to Noise - Scenario {scenario_idx}")
    plt.gca().invert_yaxis()

    for bar, c in zip(bars, counts):
        plt.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, f"{c}", va='center')

    path = os.path.join(results_dir, f"robustness_scenario{scenario_idx}.png")
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Robustness plot saved to: {path}")

def export_summary(results_summary, results_dir):
    """
    Saves a CSV file with a global summary of results from all scenarios.
    """
    if not results_summary:
        print("[WARN] No data to export to summary_results.csv")
        return

    summary_path = os.path.join(results_dir, "summary_results.csv")
    with open(summary_path, "w", newline='', encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results_summary[0].keys())
        writer.writeheader()
        writer.writerows(results_summary)
    print(f"[OK] General summary saved to: {summary_path}")

def plot_summary(summary_data, results_dir):
    """
    Creates a summary chart: distance of the best strategy for each scenario.
    """
    scenarios = [f"S{d['scenario_id']}" for d in summary_data]
    distances = [d['best_distance'] for d in summary_data]
    strategies = [d['best_strategy'] for d in summary_data]

    plt.figure(figsize=(12, 6))
    bars = plt.bar(scenarios, distances, color="skyblue", edgecolor='black')
    min_idx = distances.index(min(distances))
    bars[min_idx].set_color("green")

    plt.xlabel("Scenario", fontsize=12)
    plt.ylabel("Distance (lower = better)", fontsize=12)
    plt.title("Best Strategy per Scenario (distance)", fontsize=14, weight='bold')

    # Readable labels above each bar
    for bar, strategy, dist in zip(bars, strategies, distances):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.015,
            f"{strategy}\n({dist:.2f})",
            ha='center',
            va='bottom',
            fontsize=9,
            rotation=0,
            wrap=True
        )

    # Improve spacing and appearance
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.ylim(0, max(distances) * 1.2)  # leave some space above

    path = os.path.join(results_dir, "summary_overview.png")
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Summary plot saved to: {path}")

# MAIN FUNCTION
def main():
    """1. Create a dataset for two football teams"""
    # Team 1 (our team)
    positions_team1 = ['GK', 'CB', 'CB', 'CB', 'FB', 'CM', 'CM', 'FB', 'FW', 'FW', 'FW']
    players_team1 = []
    for idx, role in enumerate(positions_team1):
        player = {'name': f'Player_{idx + 1}', 'role': role}
        for attr, (mean, std) in roles[role].items():
            player[attr] = generate_attribute(mean, std)
        players_team1.append(player)

    # Team 2 (opponent)
    positions_team2 = ['GK', 'CB', 'CB', 'CB', 'CB', 'CB', 'CM', 'FB', 'FB', 'FW', 'FW']
    players_team2 = []
    for idx, role in enumerate(positions_team2):
        player = {'name': f'Opponent_{idx + 1}', 'role': role}
        for attr, (mean, std) in roles[role].items():
            player[attr] = generate_attribute(mean, std)
        players_team2.append(player)

    results_dir = make_results_dir()

    """2. Aggregation of base profiles for both teams"""
    profile_team1 = aggregate_team_profile(players_team1)
    profile_team2 = aggregate_team_profile(players_team2)

    """3. Gap computation"""
    gap_technical = profile_team1['A12_base'] - profile_team2['A12_base']
    gap_physical  = profile_team1['A13_base'] - profile_team2['A13_base']

    # --- BUILDING AND PRINTING FINAL PROFILES ---
    final_profile_team1 = {
        'A1': profile_team1['A1'], 'A2': profile_team1['A2'],
        'A3': profile_team1['A3'], 'A4': profile_team1['A4'],
        'A5': profile_team1['A5'], 'A6': profile_team1['A6'],
        'A7': profile_team1['A7'], 'A8': profile_team1['A8'],
        'A9': profile_team1['A9'], 'A10': profile_team1['A10'],
        'A11': profile_team1['A11'], 'A12': profile_team1['A12_base'],
        'A13': profile_team1['A13_base'], 'A14': profile_team1['A14']
    }

    print("--- Final Macro-Attribute Profile Team 1 ---")
    for key, value in final_profile_team1.items():
        print(f"{key}: {value:.4f}")

    final_profile_team2 = {
        'A1': profile_team2['A1'], 'A2': profile_team2['A2'],
        'A3': profile_team2['A3'], 'A4': profile_team2['A4'],
        'A5': profile_team2['A5'], 'A6': profile_team2['A6'],
        'A7': profile_team2['A7'], 'A8': profile_team2['A8'],
        'A9': profile_team2['A9'], 'A10': profile_team2['A10'],
        'A11': profile_team2['A11'], 'A12': profile_team2['A12_base'],
        'A13': profile_team2['A13_base'], 'A14': profile_team2['A14']
    }

    print("\n--- Final Macro-Attribute Profile Team 2 ---")
    for key, value in final_profile_team2.items():
        print(f"{key}: {value:.4f}")

    """SECTION: DYNAMIC CONDITIONS GENERATION"""
    summary_data = []  # ← empty list to accumulate data

    scenarios = generate_scenarios_v2()
    for idx, desc in enumerate(scenarios, start=1):
        print(f"\n=== Scenario {idx} ===")
        match_conditions = generate_match_conditions_from_text(desc, "Milan")

        ### MODIFICATION: Baseline insertion ###
        # Baseline computation: static team fit only, without context or opponent
        baseline_scores = []
        team_vector = list(final_profile_team1.values())
        for strategy in strategy_templates:
            dist = compute_semantic_distance_updated(team_vector, strategy["vector"])
            baseline_scores.append((strategy["name"], dist))
        baseline_scores.sort(key=lambda item: item[1])
        print(f"\n--- Baseline Strategy (static fit only): {baseline_scores[0][0]} (Distance: {baseline_scores[0][1]:.4f}) ---")

        # --- Strategy selection (with dynamic weights) ---
        best_strategy, ranking, diagnostics = select_best_strategy_v2(final_profile_team1, final_profile_team2, strategy_templates, match_conditions, opponent_penalty_lambda=0.7)
        print(f"--- Best Strategy Selected (Dynamic): {best_strategy[0]} (Score: {best_strategy[1]:.4f}) ---")

        # Extract vectors of top 3 strategies (if available)
        top_n = min(3, len(ranking))
        top_names = [ranking[i][0] for i in range(top_n)]
        top_vectors = []
        for nm in top_names:
            vec = next((s['vector'] for s in strategy_templates if s['name'] == nm), None)
            if vec is not None:
                top_vectors.append(vec)

        # Save key information for this scenario
        summary_data.append({
            "scenario_id": idx,
            "scenario_desc": desc,
            "best_strategy": best_strategy[0],
            "best_distance": round(best_strategy[1], 4),
            "match_conditions": match_conditions,
            "top3_strategies": ", ".join([r[0] for r in ranking[:3]])
        })

        # Plot radar comparison (team profile vs top-N strategies)
        if top_vectors:
            slug = "_".join(desc.lower().split()[:3])
            radar_path = os.path.join(results_dir, f"radar_{idx}_{slug}.png")
            plot_radar_chart(
                team_profile=final_profile_team1,
                strategy_vectors=top_vectors,
                strategy_names=top_names,
                title='Best Strategy vs. Team Profile Comparison',
                save_path=radar_path,
                show=False
            )
            print(f"[OK] Radar chart saved to: {radar_path}")

        print("\n--- Best Strategy Selected (Dynamic) ---")
        print(f"The best strategy to adopt is: {best_strategy[0]}")
        print(f"Adjusted distance: {best_strategy[1]:.4f}")

        print("\n--- All Adjusted Distance Scores ---")
        for name, dist in ranking:
            print(f"{name}: {dist:.4f}")

        # >>> ATTRIBUTE DIAGNOSTICS <<<
        diag_csv_path = os.path.join(
            results_dir,
            f"diagnostics_scenario{idx}_{best_strategy[0].replace(' ', '_')}.csv"
        )
        print_attribute_diagnostics(
            best_strategy_name=best_strategy[0],
            diagnostics_data=diagnostics,
            top_k=3,
            export_csv_path=diag_csv_path
        )
    
    # === Lambda sensitivity for each scenario ===
    for idx, desc in enumerate(scenarios, start=1):
        results = test_lambda_sensitivity(
            final_profile_team1,
            final_profile_team2,
            strategy_templates,
            desc,
            lambdas=[0.1, 0.3, 0.5, 0.7]
        )
        plot_lambda_sensitivity(results, idx, results_dir)

    # === Ablation Study for each scenario ===
    for idx, desc in enumerate(scenarios, start=1):
        results = ablation_study(final_profile_team1, final_profile_team2, strategy_templates, desc)
        plot_ablation_study(results, idx, results_dir)

    # === Robustness to Noise Test for each scenario ===
    for idx, desc in enumerate(scenarios, start=1):
        sorted_counts = test_robustness_noise(
            final_profile_team1,
            final_profile_team2,
            strategy_templates,
            desc,
            n_sim=100,          # number of simulations
            noise_level=0.05,    # noise ±0.05
            opponent_penalty_lambda=0.5
        )
        plot_robustness_noise(sorted_counts, idx, results_dir)

    # === Export unique summary and chart ===
    export_summary(summary_data, results_dir)
    plot_summary(summary_data, results_dir)

if __name__ == "__main__":
    main()
