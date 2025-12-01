# LIBRERIE
import numpy as np
import pandas as pd
import random
import os
import csv
import json
import matplotlib.pyplot as plt
import re
from datetime import datetime

# CONFIGURAZIONE E COSTANTI

# Configurazione SEED per riproducibilità
SEED = 41
np.random.seed(SEED)
random.seed(SEED)

MACRO_ATTR_LABELS = {
    'A1': 'Forza Offensiva',
    'A2': 'Forza Difensiva',
    'A3': 'Solidità Centrocampo',
    'A4': 'Velocità di Transizione',
    'A5': 'Pressione Alta',
    'A6': 'Capacità di Ampiezza',
    'A7': 'Resilienza Psicologica',
    'A8': 'Energia Residua',
    'A9': 'Morale di Squadra',
    'A10': 'Gestione del Tempo',
    'A11': 'Coesione Tattica',
    'A12': 'Base Tecnica (team)',
    'A13': 'Base Fisica (team)',
    'A14': 'Coesione Relazionale Interna'
}

"""
Indice:
    Ruoli principali:
        GK: GoalKeeper (Portiere)
        CB: Center Back (Difensore Centrale)
        FB: Full Back (Terzino)
        CM: Central Midfelder (Centrocampista)
        FW: Forward (Attaccante)

    Attributi:
        reflexes: riflessi
        aereal_duels: duelli aerei
        passing: passaggio
        speed: velocità
        stamina: resistenza
        tackling: contrasto
        interceptions: intercettazioni
        xG: probabilità di fare gola (valore statistico che indica la qualità delle occasioni per effettuare goal)
        xA: probabilità di fare assist (valore statistico che indica la qualità delle occasioni per aiutare i compagni di squadra a fare goal)
        aggression: aggressività
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

# Esempio: strategie codificate come vettori di 14 macro-attributi per 20 scelte tattiche, normalizzati tra 0 (irrilevante) e 1 (critico)
"""
Attacco di accumulo (Possession-based): Si concentra sul controllo del possesso palla e sulla costruzione lenta del gioco per creare opportunità.
Contropiede veloce: Privilegia la transizione rapida dalla difesa all'attacco.
Palla lunga per l'uomo bersaglio: Un approccio diretto che si basa su lanci lunghi per bypassare il centrocampo.
Corridori in ritardo da centrocampo: si basa sulla capacità dei centrocampisti di inserirsi in area di rigore o in posizioni offensive solo dopo che l'azione 
d'attacco è già iniziata, creando un elemento sorpresa per la difesa avversaria
Attraversamento sistematico: sfruttare le corsie esterne del campo per crossare la palla in area, mirando a superare la difesa avversaria con passaggi lunghi e precisi dalle fasce.
Fianchi sovrapposti: Sfrutta le fasce, con i terzini che si sovrappongono agli attaccanti per creare superiorità numerica.
Rotazioni rapide in attacco: si concentra sullo scambio continuo di posizioni tra gli attaccanti per disorganizzare la difesa avversaria.
Attacco diretto verticale: cerca di portare la palla in avanti il più velocemente possibile, con passaggi diretti e pochi tocchi.
Catenaccio classico: Una difesa estremamente solida e passiva, volta a negare spazi all'avversario.
Pressing ultra offensivo: Mira a recuperare la palla in zone avanzate del campo per segnare rapidamente.
Difesa a blocchi profondi: La squadra si schiera in massa vicino alla propria area di rigore, concedendo il possesso all'avversario ma negando spazi per penetrare e creare occasioni da gol.
Difesa zonale compatta: difendere le aree specifiche del campo. I giocatori si muovono come un blocco unito, mantenendo una formazione stretta per chiudere gli spazi tra le linee e negare all'avversario la possibilità di giocare palla.
Difesa a blocchi profondi: La squadra si ritira a ridosso della propria area, difendendo in massa e concedendo all'avversario l'iniziativa.
Difesa zonale compatta: I giocatori si muovono insieme come un'unità per difendere una zona del campo in modo organizzato.
Marcatura rigorosa dell'uomo: Ogni giocatore si prende la responsabilità di marcare un avversario specifico, seguendolo in ogni zona del campo.
Pressing ad alta zona: recuperare immediatamente la palla non appena viene persa, pressando gli avversari in zone avanzate del campo per costringerli a commettere errori o a perdere il possesso.
Pressing a centrocampo:  La pressione sull'avversario inizia nella zona centrale del campo, con l'obiettivo di limitare la costruzione del gioco avversario e forzare la perdita di possesso.
Gioco di possesso lungo: Si concentra sul mantenimento del possesso palla per stancare l'avversario e trovare il momento giusto per attaccare.
Gioco orizzontale cauto: Un approccio a basso rischio che sposta la palla orizzontalmente in attesa di un'opportunità di penetrazione.
Blocco centrale + brevi pause: Una tattica ibrida che combina una difesa solida al centro con rapide incursioni offensive.
Indurre errori di accumulo: pressare l'avversario quando sta costruendo l'azione dalle retrovie, spingendolo a commettere errori di passaggio o di posizionamento.
Forzare fuorigioco: tattica difensiva ad alto rischio in cui la linea difensiva si sposta in avanti in modo che gli attaccanti avversari siano in posizione irregolare.
"""
strategy_templates = [
    {"name": "Attacco di accumulo", "vector": [0.8, 0.5, 0.7, 0.5, 0.4, 0.6, 0.7, 0.6, 0.7, 0.6, 0.8, 0.7, 0.6, 0.7]},
    {"name": "Contropiede veloce", "vector": [0.9, 0.6, 0.5, 0.9, 0.5, 0.6, 0.7, 0.8, 0.7, 0.8, 0.6, 0.7, 0.8, 0.6]},
    {"name": "Palla lunga per l'uomo bersaglio", "vector": [0.8, 0.6, 0.5, 0.6, 0.4, 0.4, 0.6, 0.7, 0.6, 0.7, 0.5, 0.5, 0.8, 0.5]},
    {"name": "Corridori in ritardo da centrocampo", "vector": [0.8, 0.5, 0.6, 0.7, 0.5, 0.5, 0.6, 0.7, 0.7, 0.6, 0.7, 0.7, 0.7, 0.6]},
    {"name": "Attraversamento sistematico", "vector": [0.7, 0.5, 0.6, 0.6, 0.5, 0.9, 0.7, 0.7, 0.7, 0.6, 0.7, 0.7, 0.7, 0.6]},
    {"name": "Fianchi sovrapposti", "vector": [0.7, 0.5, 0.7, 0.7, 0.5, 0.9, 0.7, 0.8, 0.8, 0.7, 0.8, 0.7, 0.8, 0.7]},
    {"name": "Rotazioni rapide in attacco", "vector": [0.8, 0.5, 0.7, 0.8, 0.6, 0.7, 0.8, 0.7, 0.8, 0.7, 0.9, 0.7, 0.8, 0.7]},
    {"name": "Attacco diretto verticale", "vector": [0.9, 0.5, 0.5, 0.8, 0.5, 0.6, 0.7, 0.7, 0.7, 0.7, 0.6, 0.7, 0.8, 0.6]},
    {"name": "Catenaccio classico", "vector": [0.4, 0.9, 0.7, 0.3, 0.2, 0.3, 0.8, 0.7, 0.7, 0.9, 0.8, 0.6, 0.6, 0.7]},
    {"name": "Pressing ultra offensivo", "vector": [0.7, 0.8, 0.6, 0.9, 0.9, 0.5, 0.8, 0.7, 0.8, 0.6, 0.9, 0.7, 0.8, 0.8]},
    {"name": "Difesa a blocchi profondi", "vector": [0.4, 0.9, 0.8, 0.3, 0.2, 0.3, 0.7, 0.6, 0.6, 0.9, 0.8, 0.6, 0.5, 0.7]},
    {"name": "Difesa zonale compatta", "vector": [0.5, 0.9, 0.8, 0.4, 0.4, 0.4, 0.7, 0.6, 0.7, 0.8, 0.9, 0.7, 0.6, 0.7]},
    {"name": "Marcatura rigorosa dell'uomo", "vector": [0.5, 0.9, 0.7, 0.5, 0.5, 0.3, 0.7, 0.7, 0.6, 0.8, 0.8, 0.7, 0.7, 0.7]},
    {"name": "Pressing ad alta zona", "vector": [0.7, 0.8, 0.6, 0.8, 0.9, 0.5, 0.8, 0.7, 0.8, 0.6, 0.9, 0.7, 0.8, 0.8]},
    {"name": "Pressing a centrocampo", "vector": [0.6, 0.7, 0.7, 0.7, 0.7, 0.4, 0.7, 0.7, 0.7, 0.7, 0.8, 0.7, 0.7, 0.7]},
    {"name": "Gioco di possesso lungo", "vector": [0.7, 0.7, 0.9, 0.5, 0.5, 0.6, 0.8, 0.7, 0.8, 0.7, 0.9, 0.8, 0.6, 0.8]},
    {"name": "Gioco orizzontale cauto", "vector": [0.5, 0.7, 0.8, 0.4, 0.3, 0.5, 0.7, 0.7, 0.8, 0.7, 0.8, 0.7, 0.5, 0.7]},
    {"name": "Blocco centrale + brevi pause", "vector": [0.7, 0.8, 0.7, 0.7, 0.7, 0.5, 0.7, 0.7, 0.7, 0.7, 0.8, 0.7, 0.7, 0.7]},
    {"name": "Indurre errori di accumulo", "vector": [0.7, 0.8, 0.6, 0.8, 0.9, 0.4, 0.7, 0.7, 0.8, 0.6, 0.8, 0.7, 0.7, 0.8]},
    {"name": "Forzare fuorigioco", "vector": [0.5, 0.8, 0.7, 0.5, 0.6, 0.4, 0.7, 0.7, 0.7, 0.8, 0.8, 0.7, 0.7, 0.7]},
]


# FUNZIONI
""" Funzione di generazione di valori casuali per gli attributi dei giocatori """
def generate_attribute(mean, std):
    return np.clip(np.random.normal(mean, std), 0.0, 1.0)

""" Funzioni per il calcolo dei macro-attributi """
def compute_offensive_strength(players_team):
    """ Calcolo del valore della macro-attributo A1: Capacità di creare e finalizzare opportunità """
    relevant_roles = ['FW', 'CM']
    xG_values, dribbling_values = [], []
    for player in players_team:
        if player['role'] in relevant_roles:
            xG_values.append(player.get('xG', 0))
            dribbling_values.append(player.get('dribbling', 0))
    return 0.7 * np.mean(xG_values) + 0.3 * np.mean(dribbling_values)

def compute_defensive_strength(players_team):
    """ Calcolo del valore della macro-attributo A2: Possibilità di limitare il punteggio avversario """
    relevant_roles = ['GK', 'CB']
    reflexes_value, tackling_value = [], []
    for player in players_team:
        if player['role'] in relevant_roles:
            reflexes_value.append(player.get('reflexes', 0))
            tackling_value.append(player.get('tackling', 0))
    return 0.7 * np.mean(reflexes_value) + 0.3 * np.mean(tackling_value)

def compute_midfield_solidity(players_team):
    """ Calcolo del valore della macro-attributo A3: Capacità di controllo e filtraggio """
    relevant_roles = ['FB', 'CM']
    xA_value, speed_value = [], []
    for player in players_team:
        if player['role'] in relevant_roles:
            xA_value.append(player.get('xA', 0))
            speed_value.append(player.get('speed', 0))
    return 0.7 * np.mean(xA_value) + 0.3 * np.mean(speed_value)

def compute_transition_speed(players_team):
    """ Calcolo del valore della macro-attributo A4: Rapidi passaggi tra difesa e attacco """
    relevant_roles = ['CM', 'FW', 'FB']
    speed_values, stamina_values = [], []
    for player in players_team:
        if player['role'] in relevant_roles:
            speed_values.append(player.get('speed', 0))
            stamina_values.append(player.get('stamina', 0))
    return 0.7 * np.mean(speed_values) + 0.3 * np.mean(stamina_values)

def compute_high_press_capability(players_team):
    """ Calcolo del valore della macro-attributo A5: Recupero del possesso nelle zone avanzate """
    relevant_roles = ['FW', 'CM', 'FB', 'CB']
    tackling_values, interceptions_values = [], []
    for player in players_team:
        if player['role'] in relevant_roles:
            tackling_values.append(player.get('tackling', 0))
            interceptions_values.append(player.get('interceptions', 0))
    return 0.7 * np.mean(tackling_values) + 0.3 * np.mean(interceptions_values)

def compute_width_utilization(players_team):
    """ Calcolo del valore della macro-attributo A6: Uso efficace dei fianchi """
    relevant_roles = ['CM', 'FB']
    xA_values, stamina_values = [], []
    for player in players_team:
        if player['role'] in relevant_roles:
            xA_values.append(player.get('xA', 0))
            stamina_values.append(player.get('stamina', 0))
    return 0.7 * np.mean(xA_values) + 0.3 * np.mean(stamina_values)

def compute_psychological_resilience(players_team):
    """ Calcolo del valore della macro-attributo A7: Resistenza alla pressione emotiva """
    resilience_values, aggression_values = [], []
    for player in players_team:
        resilience_values.append(player.get('resilience', 0))
        aggression_values.append(player.get('aggression', 0))
    return 0.7 * np.mean(resilience_values) + 0.3 * np.mean(aggression_values)

def compute_residual_energy(players_team):
    """ Calcolo del valore della macro-attributo A8: Livelli di resistenza in tutta la squadra """
    stamina_values, resilience_values = [], []
    for player in players_team:
        stamina_values.append(player.get('stamina', 0))
        resilience_values.append(player.get('resilience', 0))
    return 0.7 * np.mean(stamina_values) + 0.3 * np.mean(resilience_values)

def compute_team_morale(players_team):
    """ Calcolo del valore della macro-attributo A9: Motivazione e spirito collettivo """
    resilience_values, aggression_values = [], []
    for player in players_team:
        resilience_values.append(player.get('resilience', 0))
        aggression_values.append(player.get('aggression', 0))
    return 0.6 * np.mean(resilience_values) + 0.4 * np.mean(aggression_values)

def compute_time_management(players_team):
    """ Calcolo del valore della macro-attributo A10: Adattabilità alla pressione dell'orologio """
    relevant_roles = ['GK', 'CM', 'FB']
    interceptions_values, passing_values = [], []
    for player in players_team:
        if player['role'] in relevant_roles:
            interceptions_values.append(player.get('interceptions', 0))
            passing_values.append(player.get('passing', 0))
    return 0.5 * np.mean(interceptions_values) + 0.5 * np.mean(passing_values)

def compute_tactical_cohesion(players_team):
    """ Calcolo del valore della macro-attributo A11: Sincronizzazione tra le unità """
    passing_values, xA_values = [], []
    for player in players_team:
        passing_values.append(player.get('passing', 0))
        xA_values.append(player.get('xA', 0))
    return 0.6 * np.mean(passing_values) + 0.4 * np.mean(xA_values)

def compute_technical_base(players_team):
    """Calcolo del valore della base tecnica di una squadra A12 base"""
    technical_attributes = ['reflexes', 'passing', 'dribbling', 'tackling', 'interceptions', 'xG', 'xA']
    total_values = []
    for player in players_team:
        for attr in technical_attributes:
            total_values.append(player.get(attr, 0))
    return np.mean(total_values)

def compute_physical_base(players_team):
    """Calcolo del valore della base fisica di una squadra A13 base"""
    physical_attributes = ['aerial_duels', 'speed', 'stamina', 'aggression']
    total_values = []
    for player in players_team:
        for attr in physical_attributes:
            total_values.append(player.get(attr, 0))
    return np.mean(total_values)

def compute_internal_relational_cohesion():
    """ Calcolo del valore della macro-attributo A14: Stabilità delle relazioni interne """
    return np.random.uniform(0.5, 0.9)

def aggregate_team_profile(players):
    """
    Aggrega gli attributi di una squadra per calcolare il suo profilo di macro-attributi.
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
    Parser rule-based robusto del testo scenario.
    Estrae:
      - time_remaining (minuti)
      - score_diff (nostri - loro)
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

    # --- TEMPO ---
    # "ultimi 10 minuti"
    m = re.search(r"ultimi\s+(\d+)\s*min", text)
    if m:
        conditions["time_remaining"] = int(m.group(1))
    # "siamo al 60'" -> 90 - 60 = 30
    m = re.search(r"(\d+)\s*['’](?!\w)", text)
    if m:
        minute = int(m.group(1))
        conditions["time_remaining"] = max(0, 90 - minute)

    # --- PUNTEGGIO / STATO ---
    # "vinciamo 3-0"
    m = re.search(r"vinciamo\s+(\d+)\s*[-:]\s*(\d+)", text)
    if m:
        conditions["score_diff"] = int(m.group(1)) - int(m.group(2))
    # "perdiamo 1-2"
    m = re.search(r"perdiamo\s+(\d+)\s*[-:]\s*(\d+)", text)
    if m:
        conditions["score_diff"] = int(m.group(1)) - int(m.group(2))
    # "0-0", "1:1" etc -> pareggio
    m = re.search(r"\b(\d+)\s*[-:]\s*(\d+)\b", text)
    if m and int(m.group(1)) == int(m.group(2)):
        conditions["score_diff"] = 0
    # "pareggio", "pareggiando"
    if re.search(r"paregg\w*", text):
        conditions["score_diff"] = 0
    # "sotto di 1 gol"
    m = re.search(r"sotto\s+di\s+(\d+)\s+gol", text)
    if m:
        conditions["score_diff"] = -int(m.group(1))
    # "sopra di 2", "in vantaggio"
    m = re.search(r"(sopra|vantaggio)\s+di\s+(\d+)", text)
    if m:
        conditions["score_diff"] = int(m.group(2))
    if re.search(r"in\s+vantaggio", text):
        # se non ci sono numeri, assumo +1
        if conditions["score_diff"] == 0:
            conditions["score_diff"] = +1
    if re.search(r"in\s+svantaggio", text):
        if conditions["score_diff"] == 0:
            conditions["score_diff"] = -1
    
    # --- PUNTEGGIO: "avanti di 1 gol" ---
    m = re.search(r"avanti\s+di\s+(\d+)\s+gol", text)
    if m:
        conditions["score_diff"] = int(m.group(1))

    # --- PUNTEGGIO: "parità"/"pari" ---
    if re.search(r"\bparit[aà]\b|\bpari\b", text):
        conditions["score_diff"] = 0

    # --- ENERGIA: "fresca/fresche", "piena di energie" ---
    if re.search(r"fresc\w*|pien[oa]\s+di\s+energi\w+", text):
        conditions["fatigue_level"] = 0.35  # ⇒ energia alta

    # --- TEMPO: "pochi minuti" (fallback se non ho numeri espliciti) ---
    if re.search(r"pochi\s+minut[oi]", text) and "time_remaining" not in conditions:
        conditions["time_remaining"] = 8

    # --- FATICA / ENERGIA ---
    # alti -> stanchi, esausti, fatica
    if re.search(r"stanchi|stanchezza|esaust[oi]|cotti|fatica", text):
        conditions["fatigue_level"] = 0.85
    # bassi -> freschi, energici, in forma
    if re.search(r"fresch[io]|energic[io]|in\s+forma|riposat[io]", text):
        conditions["fatigue_level"] = 0.35

    # --- MORALE ---
    if re.search(r"demoralizzat\w*|scoraggiat\w*|morale\s+basso|testa\s+gi[ùu]", text):
        conditions["morale"] = 0.4
    if re.search(r"morale\s+alto|motivazion\w+\s+alta|molto\s+motiv", text):
        conditions["morale"] = 0.85

    return conditions


""" Funzione per calcolare la distanza euclidea tra due vettori """
def compute_semantic_distance_updated(vector1, vector2):
    return np.sqrt(np.sum((np.array(vector1) - np.array(vector2)) ** 2))

""" Funzione per l'aggregazione dei macro-attributi del team. Corrisponde a 'aggregate_context_tree' nello pseudocodice. """
def aggregate_context_tree(team_profile):
    """
    Aggrega i macro-attributi del team per ottenere un singolo vettore.
    In questo caso, il profilo del team è già il vettore desiderato.
    """
    return list(team_profile.values())

"""
Funzione per applicare i pesi dinamici in base alle condizioni della partita.
Questa funzione è il cuore del modulo di simulazione dinamica. Il suo scopo è prendere un punteggio di distanza "statico"
(calcolato sulla base dei profili fissi di squadra e strategia) e modificarlo in base a fattori in evoluzione durante la partita. 
Funziona come un moltiplicatore di penalità o bonus che si adatta al contesto di gioco.
Input:
raw_distance: La distanza euclidea grezza tra il profilo della squadra e il vettore della strategia.
match_conditions: Un dizionario che contiene i dati dinamici della partita, come il livello di energia, il tempo rimanente, la differenza di punteggio, 
la resilienza e il morale.
gap_technical e gap_physical: Valori che misurano il divario tecnico e fisico tra la tua squadra e l'avversario.

"""
### MODIFICA ###
# Funzione dei pesi dinamici v2: più incisiva e consapevole della strategia.
def apply_dynamic_weights_v2(raw_distance, match_conditions, strategy_vector):
    """
    Applica un fattore di aggiustamento più incisivo e "consapevole" della strategia.
    """
    energy = 1.0 - float(match_conditions.get("fatigue_level", 0.7))
    time_left = float(match_conditions.get("time_remaining", 30))
    score_diff = float(match_conditions.get("score_diff", 0))
    morale = float(match_conditions.get("morale", 0.7))

    adjustment = 1.0

    # Analisi della natura della strategia basata sul suo vettore
    is_high_intensity = strategy_vector[4] > 0.7 or strategy_vector[3] > 0.7 # A5 (Pressing), A4 (Transizione)
    is_ultra_offensive = strategy_vector[0] > 0.8 or strategy_vector[4] > 0.8 # A1 (Offensiva), A5 (Pressing)
    is_conservative = strategy_vector[1] > 0.8 and strategy_vector[9] > 0.7 # A2 (Difensiva), A10 (Gestione Tempo)
    is_possession_based = strategy_vector[10] > 0.7 and strategy_vector[2] > 0.7 # A11 (Coesione), A3 (Centrocampo)

    # --- Logica di aggiustamento potenziata ---

    # CASO 1: Disperazione (sotto nel punteggio, poco tempo)
    if score_diff < 0 and time_left < 25:
        if is_ultra_offensive or is_high_intensity:
            adjustment *= 0.65  # Bonus massiccio (distanza ridotta del 35%)
        elif is_conservative or is_possession_based:
            adjustment *= 1.50  # Malus pesante (distanza aumentata del 50%)

    # CASO 2: Gestione del vantaggio (sopra nel punteggio, poco tempo)
    elif score_diff > 0 and time_left < 20:
        if is_conservative:
            adjustment *= 0.70  # Bonus massiccio per tattiche conservative
        elif is_ultra_offensive:
            adjustment *= 1.60  # Malus pesante per tattiche rischiose

    # CASO 3: Squadra stanca
    if energy < 0.45:
        if is_high_intensity:
            adjustment *= 1.40  # Malus pesante per strategie dispendiose
        else:
            adjustment *= 0.90  # Leggero bonus per strategie a basso consumo energetico

    # CASO 4: Morale
    if morale < 0.4: # Squadra demoralizzata
        if strategy_vector[10] > 0.8: # A11 (Coesione Tattica)
            adjustment *= 1.20 # Malus a strategie complesse che richiedono concentrazione
    elif morale > 0.8: # Squadra euforica
        if is_ultra_offensive:
            adjustment *= 0.85 # Bonus per capitalizzare il momento positivo

    # Clamp per evitare valori estremi
    adjustment = max(0.4, min(adjustment, 2.0))
    return max(0.0, raw_distance * adjustment)

# Funzione per selezionare la strategia più compatibile
"""
Funzione centrale del modello. 
Esaminare tutte le strategie disponibili e utilizzando le informazioni della TUA squadra, dell'avversario e sulle condizioni della partita, seleziona la tattica più efficace.
Input:
final_profile_team1: Il profilo vettoriale completo della tua squadra.
final_profile_team2: Il profilo vettoriale completo della squadra avversaria.
strategy_templates: La lista di tutte le strategie predefinite.
match_conditions: Il dizionario dei parametri dinamici della partita.
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

        ### MODIFICA: Utilizzo della nuova funzione di pesatura ###
        adjusted_distance = apply_dynamic_weights_v2(combined_distance, match_conditions, strategy_vector)
        
        strategy_scores.append((strategy_name, adjusted_distance))

    strategy_scores.sort(key=lambda item: item[1])
    
    # Calcolo diagnostica solo per la migliore
    best_strategy_name = strategy_scores[0][0]
    best_strategy_vector = next(s['vector'] for s in strategy_templates if s['name'] == best_strategy_name)
    diagnostics_data = {
        'team_attributes': final_profile_team1,
        'strategy_attributes': dict(zip(final_profile_team1.keys(), best_strategy_vector)),
    }

    return strategy_scores[0], strategy_scores, diagnostics_data

def _build_attr_rows(team_attrs: dict, strat_attrs: dict, order_keys):
    """Crea righe con (code, label, team, strategy, delta = strategy - team)."""
    rows = []
    for k in order_keys:
        team_v = float(team_attrs.get(k, 0.0))
        strat_v = float(strat_attrs.get(k, 0.0))
        delta = strat_v - team_v     # >0 = carenza; <0 = surplus
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
    diagnostics_data: un dizionario contenente gli attributi per la singola strategia migliore.
    """
    if not diagnostics_data:
        print("[WARN] Dati di diagnostica non disponibili.")
        return

    # Non c'è bisogno di cercare, abbiamo già il blocco corretto.
    team_attrs = diagnostics_data['team_attributes']
    strat_attrs = diagnostics_data['strategy_attributes']
    order_keys = [f'A{i}' for i in range(1, 15)]

    rows = _build_attr_rows(team_attrs, strat_attrs, order_keys)

    shortages = sorted(rows, key=lambda r: r['delta'], reverse=True)
    surpluses = sorted(rows, key=lambda r: r['delta'])

    print(f"\n=== Diagnostica per Attributo — Strategia: {best_strategy_name} ===")
    print("\n-- Carenze Principali (la squadra manca di questi attributi richiesti dalla strategia):")
    for r in shortages[:top_k]:
        if r['delta'] > 0.05:
            print(f"  {r['code']} {r['label']}: Team={r['team']:.3f}, Richiesto={r['strategy']:.3f}, Delta={r['delta']:.3f}")

    print("\n-- Surplus Principali (la squadra eccelle in questi attributi rispetto alla strategia):")
    for r in surpluses[:top_k]:
        if r['delta'] < -0.05:
            print(f"  {r['code']} {r['label']}: Team={r['team']:.3f}, Richiesto={r['strategy']:.3f}, Delta={r['delta']:.3f}")

    if export_csv_path:
        with open(export_csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['code', 'label', 'team', 'strategy', 'delta'])
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n[OK] Diagnostica esportata in: {export_csv_path}")


def plot_radar_chart(team_profile, strategy_vectors, strategy_names, title, save_path=None, show=False):
    """
    Genera un radar chart per visualizzare e confrontare i profili della squadra e delle strategie.
    :param team_profile: Dizionario o lista dei macro-attributi della squadra.
    :param strategy_vectors: Lista dei vettori delle strategie da confrontare.
    :param strategy_names: Lista dei nomi delle strategie.
    :param title: Titolo del grafico.
    """
    # Etichette in ordine A1..A14
    labels = [MACRO_ATTR_LABELS[f'A{i}'] for i in range(1, 15)]
    num_vars = len(labels)

    # Angoli
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    # Dati team
    team_data = [team_profile[f'A{i}'] for i in range(1, 15)]
    team_data += team_data[:1]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    ax.set_title(title, fontsize=16, y=1.1)

    # Linea profilo squadra (nessun colore posizionale!)
    line_team, = ax.plot(angles, team_data, linewidth=2, linestyle='--', label='Profilo Squadra')
    ax.fill(angles, team_data, alpha=0.12, color=line_team.get_color())

    # Strategie (linee semplici, niente colori posizionali)
    for i, strategy_vector in enumerate(strategy_vectors):
        data = strategy_vector + strategy_vector[:1]
        ax.plot(angles, data, linewidth=2, label=f"Strategia: {strategy_names[i]}")

    # Setup polare
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

# Scenari più differenziati per testare la robustezza del modello.
def generate_scenarios_v2():
    """
    Restituisce una lista di descrizioni di partite più variegate.
    """
    return [
        "Siamo sotto di un gol al 80', la squadra è stanca ma il morale è ancora alto dopo aver quasi pareggiato.",
        "Partita bloccata sullo 0-0 al 60', siamo freschi e giochiamo contro un avversario tecnicamente superiore.",
        "Vinciamo 2-0 al 70', ma l'avversario sta attaccando con insistenza. Le energie iniziano a calare.",
        "Siamo in svantaggio 1-0 al 55', la squadra appare demoralizzata e poco lucida dopo aver subito gol.",
        "Partita equilibrata 1-1 all'intervallo (45'), entrambe le squadre sembrano energiche e motivate per il secondo tempo."
    ]

def test_lambda_sensitivity(final_profile_team1, final_profile_team2, strategy_templates, scenario_desc, lambdas=None):
    """
    Esegue la selezione strategie per un insieme di valori di lambda e mostra come cambia il ranking.
    """
    if lambdas is None:
        lambdas = [0.0, 0.3, 0.7, 1.0]  # valori tipici da testare

    match_conditions = generate_match_conditions_from_text(scenario_desc, "Milan")
    print(f"\n=== Sensibilità lambda per scenario: \"{scenario_desc}\" ===")

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
        print(f"lambda={lam:.2f} -> Migliore: {best[0]} (distanza {best[1]:.4f})")

    # Stampa tabella sintetica
    print("\n--- Sintesi ---")
    for lam, name, dist in results:
        print(f"lambda={lam:.2f} : {name} (score {dist:.4f})")

    return results

def plot_lambda_sensitivity(results, scenario_idx, results_dir):
    lambdas = [r[0] for r in results]
    distances = [r[2] for r in results]

    plt.figure()
    plt.plot(lambdas, distances, marker='o')
    plt.title(f"Sensibilità lambda - Scenario {scenario_idx}")
    plt.xlabel("lambda (penalità fit avversario)")
    plt.ylabel("Distanza aggiustata")
    plt.grid(True)

    path = os.path.join(results_dir, f"sensitivity_lambda_scenario{scenario_idx}.png")
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Grafico sensibilità lambda salvato in: {path}")

def ablation_study(final_profile_team1, final_profile_team2, strategy_templates, scenario_desc):
    """
    Ablation vera: togli progressivamente attributi A1..A14 e osserva quanto peggiora
    la distanza della strategia baseline e/o se cambia la best strategy.
    """
    match_conditions = generate_match_conditions_from_text(scenario_desc, "Milan")
    print(f"\n=== Ablation Study (rimozione attributi) per scenario: \"{scenario_desc}\" ===")

    # --- BASELINE ---
    baseline_best, baseline_ranking, _ = select_best_strategy_v2(
        final_profile_team1, final_profile_team2, strategy_templates, match_conditions, opponent_penalty_lambda=0.5
    )
    baseline_name, baseline_score = baseline_best
    print(f"[Baseline] Migliore: {baseline_name} (score={baseline_score:.4f})")

    results = [("Baseline", baseline_name, baseline_score, "")]

    attr_keys = [f"A{i}" for i in range(1, 15)]

    # --- RIMOZIONE SINGOLI ATTRIBUTI ---
    for k in attr_keys:
        mod_team = final_profile_team1.copy()
        mod_oppo = final_profile_team2.copy()
        # azzera attributo k
        mod_team[k] = 0.0
        mod_oppo[k] = 0.0

        best, _, _ = select_best_strategy_v2(mod_team, mod_oppo, strategy_templates, match_conditions, opponent_penalty_lambda=0.5)
        best_name, best_score = best
        change = "" if best_name == baseline_name else f"[CAMBIATO] da {baseline_name} -> {best_name}"
        print(f"- Rimosso {k}: best={best_name} (score={best_score:.4f}) {change}")
        results.append((k, best_name, best_score, change))

    return results

def plot_ablation_study(results, scenario_idx, results_dir):
    """
    Grafico: per ogni attributo rimosso mostra di quanto peggiora il punteggio della baseline.
    """
    baseline_score = results[0][2]
    labels = [r[0] for r in results[1:]]  # salta baseline
    scores = [r[2] - baseline_score for r in results[1:]]

    plt.figure(figsize=(10, 6))
    bars = plt.barh(labels, scores, color="skyblue")
    plt.axvline(0, color="black", lw=1)
    plt.xlabel("Δ distanza rispetto a baseline (positivo = peggiora)")
    plt.title(f"Ablation sugli attributi - Scenario {scenario_idx}")
    plt.gca().invert_yaxis()

    for bar, s in zip(bars, scores):
        plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, f"{s:+.3f}", va='center')

    path = os.path.join(results_dir, f"ablation_attributes_scenario{scenario_idx}.png")
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Grafico ablation attributi salvato in: {path}")

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
            # Chiamata corretta con strategy_vector
            combined = apply_dynamic_weights_v2(combined, match_conditions, strat["vector"])
            strategy_scores.append((strat["name"], combined))
        strategy_scores.sort(key=lambda x: x[1])
        best = strategy_scores[0][0]
        counts[best] = counts.get(best, 0) + 1
    sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    print(f"\n=== Test Robustezza al Rumore per scenario: \"{scenario_desc}\" ===")
    for name, cnt in sorted_counts:
        print(f"{name:35s}: scelto {cnt}/{n_sim} volte ({cnt/n_sim*100:.1f}%)")
    return sorted_counts

def plot_robustness_noise(sorted_counts, scenario_idx, results_dir):
    """
    Crea un grafico a barre con la frequenza di scelta delle strategie sotto rumore.
    """
    strategies = [x[0] for x in sorted_counts]
    counts = [x[1] for x in sorted_counts]

    plt.figure(figsize=(10, 5))
    bars = plt.barh(strategies, counts, color="skyblue")
    bars[0].set_color("green")
    plt.xlabel("Numero di volte scelta su simulazioni")
    plt.title(f"Robustezza al Rumore - Scenario {scenario_idx}")
    plt.gca().invert_yaxis()

    for bar, c in zip(bars, counts):
        plt.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, f"{c}", va='center')

    path = os.path.join(results_dir, f"robustness_scenario{scenario_idx}.png")
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Grafico robustezza salvato in: {path}")

def export_summary(results_summary, results_dir):
    """
    Salva un file CSV con un riepilogo globale dei risultati di tutti gli scenari.
    """
    if not results_summary:
        print("[WARN] Nessun dato da esportare in summary_results.csv")
        return

    summary_path = os.path.join(results_dir, "summary_results.csv")
    with open(summary_path, "w", newline='', encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results_summary[0].keys())
        writer.writeheader()
        writer.writerows(results_summary)
    print(f"[OK] Riepilogo generale salvato in: {summary_path}")

def plot_summary(summary_data, results_dir):
    """
    Crea un grafico riepilogativo: distanza della miglior strategia per ogni scenario.
    """
    scenarios = [f"S{d['scenario_id']}" for d in summary_data]
    distances = [d['best_distance'] for d in summary_data]
    strategies = [d['best_strategy'] for d in summary_data]

    plt.figure(figsize=(12, 6))
    bars = plt.bar(scenarios, distances, color="skyblue", edgecolor='black')
    min_idx = distances.index(min(distances))
    bars[min_idx].set_color("green")

    plt.xlabel("Scenario", fontsize=12)
    plt.ylabel("Distanza (più bassa = migliore)", fontsize=12)
    plt.title("Strategia migliore per scenario (distanza)", fontsize=14, weight='bold')

    # Etichette leggibili sopra ogni barra
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

    # Migliora la spaziatura e l'aspetto
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.ylim(0, max(distances) * 1.2)  # lascia un po’ di spazio sopra

    path = os.path.join(results_dir, "summary_overview.png")
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Grafico riepilogativo salvato in: {path}")

# FUNZIONE D'INIZIO
def main():
    """ 1. Creare un dataset per due squadre di calcio """
    # Squadra 1 (la nostra squadra)
    positions_team1 = ['GK', 'CB', 'CB', 'CB', 'FB', 'CM', 'CM', 'FB', 'FW', 'FW', 'FW']
    players_team1 = []
    for idx, role in enumerate(positions_team1):
        player = {'name': f'Player_{idx + 1}', 'role': role}
        for attr, (mean, std) in roles[role].items():
            player[attr] = generate_attribute(mean, std)
        players_team1.append(player)

    # Squadra 2 (avversario)
    positions_team2 = ['GK', 'CB', 'CB', 'CB', 'CB', 'CB', 'CM', 'FB', 'FB', 'FW', 'FW']
    players_team2 = []
    for idx, role in enumerate(positions_team2):
        player = {'name': f'Opponent_{idx + 1}', 'role': role}
        for attr, (mean, std) in roles[role].items():
            player[attr] = generate_attribute(mean, std)
        players_team2.append(player)

    results_dir = make_results_dir()

    """ 2. Aggregazione dei profili di base per entrambe le squadre """
    profile_team1 = aggregate_team_profile(players_team1)
    profile_team2 = aggregate_team_profile(players_team2)

    """ 3. Calcolo dei Gap """
    gap_technical = profile_team1['A12_base'] - profile_team2['A12_base']
    gap_physical  = profile_team1['A13_base'] - profile_team2['A13_base']

    # --- COSTRUZIONE E STAMPA DEI PROFILI FINALI ---
    final_profile_team1 = {
        'A1': profile_team1['A1'], 'A2': profile_team1['A2'],
        'A3': profile_team1['A3'], 'A4': profile_team1['A4'],
        'A5': profile_team1['A5'], 'A6': profile_team1['A6'],
        'A7': profile_team1['A7'], 'A8': profile_team1['A8'],
        'A9': profile_team1['A9'], 'A10': profile_team1['A10'],
        'A11': profile_team1['A11'], 'A12': profile_team1['A12_base'],
        'A13': profile_team1['A13_base'], 'A14': profile_team1['A14']
    }

    print("--- Profilo Finale Macro-Attributi Squadra 1 ---")
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

    print("\n--- Profilo Finale Macro-Attributi Squadra 2 ---")
    for key, value in final_profile_team2.items():
        print(f"{key}: {value:.4f}")

    """ SEZIONE: GENERAZIONE DINAMICA DELLE CONDIZIONI """
    summary_data = []  # ← lista vuota per accumulare i dati

    scenarios = generate_scenarios_v2()
    for idx, desc in enumerate(scenarios, start=1):
        print(f"\n=== Scenario {idx} ===")
        match_conditions = generate_match_conditions_from_text(desc, "Milan")

        ### MODIFICA: Inserimento della baseline ###
        # Calcolo Baseline: solo fit statico della squadra, senza contesto o avversario
        baseline_scores = []
        team_vector = list(final_profile_team1.values())
        for strategy in strategy_templates:
            dist = compute_semantic_distance_updated(team_vector, strategy["vector"])
            baseline_scores.append((strategy["name"], dist))
        baseline_scores.sort(key=lambda item: item[1])
        print(f"\n--- Strategia Baseline (solo fit statico): {baseline_scores[0][0]} (Distanza: {baseline_scores[0][1]:.4f}) ---")

        # --- Selezione strategia (con pesi dinamici) ---
        best_strategy, ranking, diagnostics = select_best_strategy_v2(final_profile_team1, final_profile_team2, strategy_templates, match_conditions, opponent_penalty_lambda=0.7)
        print(f"--- Strategia Migliore Selezionata (Dinamica): {best_strategy[0]} (Score: {best_strategy[1]:.4f}) ---")

        # Estrarre i vettori delle prime 3 strategie (se disponibili)
        top_n = min(3, len(ranking))
        top_names = [ranking[i][0] for i in range(top_n)]
        top_vectors = []
        for nm in top_names:
            vec = next((s['vector'] for s in strategy_templates if s['name'] == nm), None)
            if vec is not None:
                top_vectors.append(vec)

        # Salva informazioni chiave di questo scenario
        summary_data.append({
            "scenario_id": idx,
            "scenario_desc": desc,
            "best_strategy": best_strategy[0],
            "best_distance": round(best_strategy[1], 4),
            "match_conditions": match_conditions,
            "top3_strategies": ", ".join([r[0] for r in ranking[:3]])
        })

        # Plot radar confronto (profilo squadra vs top-N strategie)
        if top_vectors:
            slug = "_".join(desc.lower().split()[:3])
            radar_path = os.path.join(results_dir, f"radar_{idx}_{slug}.png")
            plot_radar_chart(
                team_profile=final_profile_team1,
                strategy_vectors=top_vectors,
                strategy_names=top_names,
                title='Confronto Strategia Migliore vs. Profilo Squadra',
                save_path=radar_path,
                show=False
            )
            print(f"[OK] Radar salvato in: {radar_path}")

        print("\n--- Strategia Migliore Selezionata (Dinamica) ---")
        print(f"La strategia migliore da adottare e': {best_strategy[0]}")
        print(f"Distanza aggiustata: {best_strategy[1]:.4f}")

        print("\n--- Tutti i Punteggi di Distanza Aggiustati ---")
        for name, dist in ranking:
            print(f"{name}: {dist:.4f}")

        # >>> DIAGNOSTICA PER ATTRIBUTO <<<
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
    
    # === 
    for idx, desc in enumerate(scenarios, start=1):
        results = test_lambda_sensitivity(
            final_profile_team1,
            final_profile_team2,
            strategy_templates,
            desc,
            lambdas=[0.1, 0.3, 0.5, 0.7]
        )
        plot_lambda_sensitivity(results, idx, results_dir)

    # === Ablation Study per ogni scenario ===
    for idx, desc in enumerate(scenarios, start=1):
        results = ablation_study(final_profile_team1, final_profile_team2, strategy_templates, desc)
        plot_ablation_study(results, idx, results_dir)

    # === Test di Robustezza al Rumore per ogni scenario ===
    for idx, desc in enumerate(scenarios, start=1):
        sorted_counts = test_robustness_noise(
            final_profile_team1,
            final_profile_team2,
            strategy_templates,
            desc,
            n_sim=100,          # numero simulazioni
            noise_level=0.05,    # rumore ±0.05
            opponent_penalty_lambda=0.5
        )
        plot_robustness_noise(sorted_counts, idx, results_dir)

    # === Esportazione del riepilogo unico e del grafico ===
    export_summary(summary_data, results_dir)
    plot_summary(summary_data, results_dir)

if __name__ == "__main__":
    main()