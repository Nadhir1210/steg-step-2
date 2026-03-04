#!/usr/bin/env python3
"""
⭐ 06 - PD Severity Score - Score Global de Sévérité des Décharges
===================================================================
Créer un score agrégé de 0 à 100 pour quantifier la sévérité PD.

Formule:
PD_Score = w1×Intensity + w2×Energy + w3×Asymmetry + w4×Trend

Où chaque composante est normalisée entre 0 et 100.

Interprétation:
- 0-25: Excellent (Normal)
- 25-50: Bon (Surveillance légère)
- 50-75: Moyen (Surveillance accrue)
- 75-100: Critique (Intervention recommandée)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')

# Configuration
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "LAST_DATA"
PLOTS_DIR = Path(__file__).parent / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

print("=" * 80)
print("⭐ PD SEVERITY SCORE - Score de Sévérité Global")
print("=" * 80)

# ============================================================================
# 1. CONFIGURATION DES POIDS
# ============================================================================
# Poids pour chaque composante du score (total = 1)
WEIGHTS = {
    'intensity': 0.35,    # Importance de l'intensité des décharges
    'energy': 0.25,       # Énergie totale des décharges
    'asymmetry': 0.15,    # Déséquilibre entre canaux
    'trend': 0.15,        # Tendance (augmentation récente)
    'stability': 0.10     # Stabilité (inverse de la variance)
}

# Vérification que les poids somment à 1
assert abs(sum(WEIGHTS.values()) - 1.0) < 0.01, "Les poids doivent sommer à 1"

# ============================================================================
# 2. CHARGEMENT DES DONNÉES
# ============================================================================
print("\n[1/6] Chargement des données PD enrichies...")

feature_file = DATA_DIR / "TG1_Sousse_PD_Features.csv"
if feature_file.exists():
    df = pd.read_csv(feature_file)
    print(f"   ✓ Dataset enrichi: {df.shape[0]:,} × {df.shape[1]}")
else:
    print("   ⚠️ Fichier enrichi non trouvé. Exécutez d'abord 01_PD_Feature_Engineering.py")
    exit(1)

# ============================================================================
# 3. EXTRACTION DES COMPOSANTES
# ============================================================================
print("\n[2/6] Extraction des composantes du score...")

# Dictionnaire pour stocker les composantes normalisées
components = {}

# 1. INTENSITÉ
print("   → Composante Intensité...")
if 'PD_INTENSITY_TOTAL' in df.columns:
    intensity = df['PD_INTENSITY_TOTAL'].values
else:
    # Calculer si non disponible
    current_cols = [c for c in df.columns if 'CURRENT' in c and 'ABS' in c]
    pulse_cols = [c for c in df.columns if 'PULSE_COUNT' in c]
    if current_cols and pulse_cols:
        intensity = df[current_cols].sum(axis=1).values * df[pulse_cols].sum(axis=1).values
    else:
        intensity = np.ones(len(df))

# Normaliser [0, 100] avec percentile pour éviter les outliers
p5, p95 = np.percentile(intensity, [5, 95])
intensity_norm = np.clip((intensity - p5) / (p95 - p5 + 1e-10), 0, 1) * 100
components['intensity'] = intensity_norm
print(f"      ✓ Moyenne: {intensity_norm.mean():.2f} / 100")

# 2. ÉNERGIE
print("   → Composante Énergie...")
if 'PD_ENERGY_TOTAL' in df.columns:
    energy = df['PD_ENERGY_TOTAL'].values
else:
    charge_cols = [c for c in df.columns if 'MEAN_CHARGE' in c]
    rate_cols = [c for c in df.columns if 'DISCHARGE_RATE' in c]
    if charge_cols and rate_cols:
        energy = df[charge_cols].sum(axis=1).values * df[rate_cols].sum(axis=1).values
    else:
        energy = np.ones(len(df))

p5, p95 = np.percentile(energy, [5, 95])
energy_norm = np.clip((energy - p5) / (p95 - p5 + 1e-10), 0, 1) * 100
components['energy'] = energy_norm
print(f"      ✓ Moyenne: {energy_norm.mean():.2f} / 100")

# 3. ASYMÉTRIE
print("   → Composante Asymétrie...")
if 'INTENSITY_ASYMMETRY' in df.columns:
    asymmetry = df['INTENSITY_ASYMMETRY'].values
elif 'ENERGY_ASYMMETRY' in df.columns:
    asymmetry = df['ENERGY_ASYMMETRY'].values
else:
    # Calculer l'asymétrie entre canaux
    ch_cols = {}
    for ch in ['CH1', 'CH2', 'CH3', 'CH4']:
        ch_intensity = [c for c in df.columns if ch in c and 'INTENSITY' in c]
        if ch_intensity:
            ch_cols[ch] = df[ch_intensity[0]].values if ch_intensity else np.zeros(len(df))
    
    if ch_cols:
        ch_values = np.array(list(ch_cols.values()))
        asymmetry = np.max(ch_values, axis=0) - np.min(ch_values, axis=0)
    else:
        asymmetry = np.zeros(len(df))

p5, p95 = np.percentile(asymmetry, [5, 95])
asymmetry_norm = np.clip((asymmetry - p5) / (p95 - p5 + 1e-10), 0, 1) * 100
components['asymmetry'] = asymmetry_norm
print(f"      ✓ Moyenne: {asymmetry_norm.mean():.2f} / 100")

# 4. TENDANCE (basée sur rolling mean)
print("   → Composante Tendance...")
if 'PD_INTENSITY_TREND_1h' in df.columns:
    trend = df['PD_INTENSITY_TREND_1h'].values
elif 'PD_INTENSITY_ROLL_MEAN_1h' in df.columns:
    # Calculer la tendance comme variation par rapport à la moyenne
    roll_mean = df['PD_INTENSITY_ROLL_MEAN_1h'].values
    current = df['PD_INTENSITY_TOTAL'].values if 'PD_INTENSITY_TOTAL' in df.columns else intensity
    trend = (current - roll_mean) / (roll_mean + 1e-10)
else:
    # Tendance simple sur un rolling
    trend = pd.Series(intensity).diff(30).fillna(0).values / (intensity + 1e-10)

# Normaliser le trend (peut être négatif = bon, positif = mauvais)
# Une tendance positive (augmentation) est mauvaise
trend = np.nan_to_num(trend, nan=0, posinf=100, neginf=-100)
trend_norm = np.clip((trend + 1) / 2, 0, 1) * 100  # Shift et scale
components['trend'] = trend_norm
print(f"      ✓ Moyenne: {trend_norm.mean():.2f} / 100")

# 5. STABILITÉ (inverse du coefficient de variation rolling)
print("   → Composante Stabilité...")
if 'INTENSITY_CV' in df.columns:
    cv = df['INTENSITY_CV'].values
    instability = np.abs(cv)
elif 'PD_INTENSITY_ROLL_STD_30min' in df.columns:
    roll_std = df['PD_INTENSITY_ROLL_STD_30min'].values
    roll_mean = df.get('PD_INTENSITY_ROLL_MEAN_30min', pd.Series(intensity).rolling(30, min_periods=1).mean()).values
    instability = roll_std / (roll_mean + 1e-10)
else:
    # CV simple
    instability = pd.Series(intensity).rolling(30, min_periods=1).std().fillna(0).values / (intensity + 1e-10)

p5, p95 = np.percentile(instability, [5, 95])
stability_norm = np.clip((instability - p5) / (p95 - p5 + 1e-10), 0, 1) * 100
components['stability'] = stability_norm
print(f"      ✓ Moyenne: {stability_norm.mean():.2f} / 100")

# ============================================================================
# 4. CALCUL DU SCORE GLOBAL
# ============================================================================
print("\n[3/6] Calcul du score de sévérité...")

# Score pondéré
df['PD_SEVERITY_SCORE'] = (
    WEIGHTS['intensity'] * components['intensity'] +
    WEIGHTS['energy'] * components['energy'] +
    WEIGHTS['asymmetry'] * components['asymmetry'] +
    WEIGHTS['trend'] * components['trend'] +
    WEIGHTS['stability'] * components['stability']
)

# Ajouter les composantes au dataframe
for name, values in components.items():
    df[f'Score_{name.capitalize()}'] = values

print(f"   ✓ Score calculé pour {len(df):,} observations")

# Statistiques du score
print(f"\n   📊 STATISTIQUES DU SCORE:")
print("   " + "-" * 40)
print(f"   • Minimum: {df['PD_SEVERITY_SCORE'].min():.2f}")
print(f"   • Maximum: {df['PD_SEVERITY_SCORE'].max():.2f}")
print(f"   • Moyenne: {df['PD_SEVERITY_SCORE'].mean():.2f}")
print(f"   • Médiane: {df['PD_SEVERITY_SCORE'].median():.2f}")
print(f"   • Écart-type: {df['PD_SEVERITY_SCORE'].std():.2f}")

# ============================================================================
# 5. CLASSIFICATION PAR SEUILS
# ============================================================================
print("\n[4/6] Classification par sévérité...")

# Définir les catégories
def classify_severity(score):
    if score < 25:
        return 'Excellent'
    elif score < 50:
        return 'Bon'
    elif score < 75:
        return 'Moyen'
    else:
        return 'Critique'

df['Severity_Class'] = df['PD_SEVERITY_SCORE'].apply(classify_severity)

# Distribution des classes
class_counts = df['Severity_Class'].value_counts()
print(f"\n   📊 DISTRIBUTION DES CLASSES:")
print("   " + "-" * 50)
severity_order = ['Excellent', 'Bon', 'Moyen', 'Critique']
for cls in severity_order:
    if cls in class_counts:
        count = class_counts[cls]
        pct = 100 * count / len(df)
        bar = '█' * int(pct / 2)
        print(f"   {cls:<12} {count:>10,} ({pct:>5.1f}%) {bar}")

# ============================================================================
# 6. VISUALISATIONS
# ============================================================================
print("\n[5/6] Création des visualisations...")

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 1. Distribution du score
ax1 = axes[0, 0]
ax1.hist(df['PD_SEVERITY_SCORE'], bins=50, edgecolor='black', alpha=0.7)
ax1.axvline(25, color='green', linestyle='--', linewidth=2, label='Excellent/Bon')
ax1.axvline(50, color='orange', linestyle='--', linewidth=2, label='Bon/Moyen')
ax1.axvline(75, color='red', linestyle='--', linewidth=2, label='Moyen/Critique')
ax1.set_xlabel('PD Severity Score')
ax1.set_ylabel('Fréquence')
ax1.set_title('Distribution du Score de Sévérité')
ax1.legend(fontsize=8)

# 2. Pie chart des classes
ax2 = axes[0, 1]
colors_pie = {'Excellent': '#2ecc71', 'Bon': '#f1c40f', 'Moyen': '#e67e22', 'Critique': '#e74c3c'}
ordered_counts = [class_counts.get(c, 0) for c in severity_order]
ax2.pie(ordered_counts, labels=severity_order, autopct='%1.1f%%', 
        colors=[colors_pie[c] for c in severity_order],
        explode=[0.05, 0, 0, 0.1])
ax2.set_title('Répartition par Classe de Sévérité')

# 3. Évolution temporelle du score (échantillon)
ax3 = axes[0, 2]
sample_size = min(5000, len(df))
sample_indices = np.linspace(0, len(df)-1, sample_size, dtype=int)
sample_scores = df['PD_SEVERITY_SCORE'].values[sample_indices]
ax3.plot(sample_scores, alpha=0.7, linewidth=0.5)
ax3.axhline(25, color='green', linestyle='--', alpha=0.5)
ax3.axhline(50, color='orange', linestyle='--', alpha=0.5)
ax3.axhline(75, color='red', linestyle='--', alpha=0.5)
ax3.fill_between(range(len(sample_scores)), 0, 25, alpha=0.2, color='green')
ax3.fill_between(range(len(sample_scores)), 25, 50, alpha=0.2, color='yellow')
ax3.fill_between(range(len(sample_scores)), 50, 75, alpha=0.2, color='orange')
ax3.fill_between(range(len(sample_scores)), 75, 100, alpha=0.2, color='red')
ax3.set_xlabel('Temps')
ax3.set_ylabel('Score')
ax3.set_title('Évolution Temporelle du Score')
ax3.set_ylim(0, 100)

# 4. Radar chart des composantes (moyenne)
ax4 = axes[1, 0]
component_names = ['Intensité', 'Énergie', 'Asymétrie', 'Tendance', 'Stabilité']
component_values = [
    components['intensity'].mean(),
    components['energy'].mean(),
    components['asymmetry'].mean(),
    components['trend'].mean(),
    components['stability'].mean()
]
# Polar plot
angles = np.linspace(0, 2 * np.pi, len(component_names), endpoint=False).tolist()
component_values_plot = component_values + [component_values[0]]
angles += [angles[0]]

ax4 = fig.add_subplot(2, 3, 4, projection='polar')
ax4.plot(angles, component_values_plot, 'b-', linewidth=2)
ax4.fill(angles, component_values_plot, alpha=0.25)
ax4.set_xticks(angles[:-1])
ax4.set_xticklabels(component_names, fontsize=9)
ax4.set_ylim(0, 100)
ax4.set_title('Profil des Composantes\n(Moyennes)', y=1.1)

# 5. Boxplot par classe
ax5 = axes[1, 1]
box_data = [df[df['Severity_Class'] == cls]['PD_SEVERITY_SCORE'].values for cls in severity_order]
bp = ax5.boxplot(box_data, labels=severity_order, patch_artist=True)
for patch, cls in zip(bp['boxes'], severity_order):
    patch.set_facecolor(colors_pie[cls])
ax5.set_ylabel('Score')
ax5.set_title('Distribution par Classe')

# 6. Corrélation composantes/score
ax6 = axes[1, 2]
component_cols = [f'Score_{name.capitalize()}' for name in ['intensity', 'energy', 'asymmetry', 'trend', 'stability']]
component_cols_available = [c for c in component_cols if c in df.columns]
corr_data = df[component_cols_available + ['PD_SEVERITY_SCORE']].corr()
sns.heatmap(corr_data, annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=ax6)
ax6.set_title('Corrélation des Composantes')

plt.tight_layout()
plt.savefig(PLOTS_DIR / "06_pd_severity_score.png", dpi=150, bbox_inches='tight')
plt.close()
print("   ✓ 06_pd_severity_score.png")

# ============================================================================
# 7. SAUVEGARDE
# ============================================================================
print("\n[6/6] Sauvegarde des résultats...")

# Sauvegarder le dataset avec score
output_file = DATA_DIR / "TG1_Sousse_PD_WithScore.csv"
df.to_csv(output_file, index=False)
print(f"   ✓ Dataset avec score: {output_file}")

# Sauvegarder les paramètres du score
score_params = {
    'weights': WEIGHTS,
    'thresholds': {
        'excellent': (0, 25),
        'bon': (25, 50),
        'moyen': (50, 75),
        'critique': (75, 100)
    },
    'component_stats': {
        name: {
            'mean': components[name].mean(),
            'std': components[name].std(),
            'min': components[name].min(),
            'max': components[name].max()
        } for name in components.keys()
    },
    'overall_stats': {
        'mean': df['PD_SEVERITY_SCORE'].mean(),
        'std': df['PD_SEVERITY_SCORE'].std(),
        'min': df['PD_SEVERITY_SCORE'].min(),
        'max': df['PD_SEVERITY_SCORE'].max(),
        'median': df['PD_SEVERITY_SCORE'].median()
    }
}
joblib.dump(score_params, PLOTS_DIR / "06_severity_score_params.pkl")
print("   ✓ 06_severity_score_params.pkl")

# Résumé CSV
summary = pd.DataFrame({
    'Class': severity_order,
    'Count': [class_counts.get(c, 0) for c in severity_order],
    'Percentage': [100 * class_counts.get(c, 0) / len(df) for c in severity_order]
})
summary.to_csv(PLOTS_DIR / "06_severity_distribution.csv", index=False)
print("   ✓ 06_severity_distribution.csv")

print("\n" + "=" * 80)
print("✅ PD SEVERITY SCORE - TERMINÉ")
print("=" * 80)
print(f"""
⭐ SCORE DE SÉVÉRITÉ PD

📊 Formule:
   PD_Score = {WEIGHTS['intensity']:.0%}×Intensité + {WEIGHTS['energy']:.0%}×Énergie + 
              {WEIGHTS['asymmetry']:.0%}×Asymétrie + {WEIGHTS['trend']:.0%}×Tendance + {WEIGHTS['stability']:.0%}×Stabilité

📊 Interprétation:
   🟢 [0-25]   Excellent: Fonctionnement optimal
   🟡 [25-50]  Bon: Surveillance légère recommandée
   🟠 [50-75]  Moyen: Surveillance accrue requise
   🔴 [75-100] Critique: Intervention recommandée

📊 Statistiques:
   • Score moyen: {df['PD_SEVERITY_SCORE'].mean():.2f} / 100
   • Score médian: {df['PD_SEVERITY_SCORE'].median():.2f} / 100

📊 Distribution:""")

for cls in severity_order:
    if cls in class_counts:
        count = class_counts[cls]
        pct = 100 * count / len(df)
        symbol = {'Excellent': '🟢', 'Bon': '🟡', 'Moyen': '🟠', 'Critique': '🔴'}[cls]
        print(f"   {symbol} {cls:<12}: {pct:.1f}%")

print(f"""
📁 Fichiers générés:
   • {PLOTS_DIR / '06_pd_severity_score.png'}
   • {DATA_DIR / 'TG1_Sousse_PD_WithScore.csv'}
   • {PLOTS_DIR / '06_severity_score_params.pkl'}
   • {PLOTS_DIR / '06_severity_distribution.csv'}
""")
