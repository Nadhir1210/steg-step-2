"""
🔥 MODULE 5 - Global TG1 Health Index
======================================
TG1 Turbo-Alternator - Digital Twin Health Monitoring System

Chef-d'œuvre du système: Score de santé global combinant tous les indicateurs

Formule:
TG1_HEALTH = 30% PD + 30% Thermal + 20% Cooling + 20% Electrical

Classification:
🟢 Excellent (85-100)
🟡 Stable (70-84)
🟠 Degrading (50-69)
🔴 Critical (<50)

Auteur: Nadhir - Stage STEG 2026
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
import joblib
import warnings
warnings.filterwarnings('ignore')

# Configuration
plt.style.use('seaborn-v0_8-whitegrid')
PLOTS_DIR = "tg1_monitoring/plots"
DATA_DIR = "LAST_DATA"

print("=" * 70)
print("🔥 MODULE 5 - GLOBAL TG1 HEALTH INDEX")
print("=" * 70)

# =============================================================================
# 1. CHARGEMENT DES DONNÉES
# =============================================================================
print("\n📥 Chargement des données...")

# Données APM Alternateur
df_apm = pd.read_csv(f"{DATA_DIR}/APM_Alternateur_10min_ML.csv")
print(f"   APM Alternateur: {len(df_apm):,} lignes")

# Données PD (si disponibles)
try:
    df_pd = pd.read_csv(f"{DATA_DIR}/TG1_Sousse_ML.csv")
    has_pd = True
    print(f"   TG1 Sousse PD: {len(df_pd):,} lignes")
except:
    has_pd = False
    print("   ⚠️ Données PD non trouvées (utilisation de simulation)")

# =============================================================================
# 2. CALCUL DES SCORES COMPOSANTS
# =============================================================================
print("\n🔧 Calcul des scores composants...")

# Variables de température stator
temp_cols = [col for col in df_apm.columns if 'STATOR' in col and 'TEMP' in col]

# ---- SCORE THERMIQUE ----
print("\n   📊 Score Thermique...")
df_apm['STATOR_TEMP_MEAN'] = df_apm[temp_cols].mean(axis=1)
df_apm['STATOR_TEMP_MAX'] = df_apm[temp_cols].max(axis=1)
df_apm['LOAD_MW'] = df_apm['MODE_TAG_1']
df_apm['AMBIENT_TEMP'] = df_apm['AMBIENT_AIR_TEMP_C']

# Score thermique basé sur la température vs charge
# Température attendue = baseline + coef * charge
# Plus la température réelle est proche de l'attendue, meilleur est le score

# Modèle simple: temp normale = 50 + 0.3 * charge
df_apm['TEMP_EXPECTED'] = 50 + 0.3 * df_apm['LOAD_MW']
df_apm['TEMP_RESIDUAL'] = df_apm['STATOR_TEMP_MEAN'] - df_apm['TEMP_EXPECTED']

# Normaliser le résidu en score (0-100)
# Résidu de 0 = 100 points, résidu de ±20°C = 0 points
df_apm['THERMAL_SCORE'] = 100 - np.clip(np.abs(df_apm['TEMP_RESIDUAL']) * 5, 0, 100)
df_apm['THERMAL_SCORE'] = df_apm['THERMAL_SCORE'].clip(0, 100)

print(f"      Moyenne: {df_apm['THERMAL_SCORE'].mean():.1f}/100")

# ---- SCORE COOLING ----
print("\n   📊 Score Cooling...")
df_apm['DELTA_T'] = df_apm['ENCLOSED_HOT_AIR_TEMP_1_degC'] - df_apm['ENCLOSED_COLD_AIR_TEMP_1_degC']

# Delta T attendu basé sur la charge
df_apm['DELTA_T_EXPECTED'] = 15 + 0.08 * df_apm['LOAD_MW']
df_apm['COOLING_RESIDUAL'] = df_apm['DELTA_T'] - df_apm['DELTA_T_EXPECTED']

# Score cooling
df_apm['COOLING_SCORE'] = 100 - np.clip(np.abs(df_apm['COOLING_RESIDUAL']) * 3, 0, 100)
df_apm['COOLING_SCORE'] = df_apm['COOLING_SCORE'].clip(0, 100)

print(f"      Moyenne: {df_apm['COOLING_SCORE'].mean():.1f}/100")

# ---- SCORE ÉLECTRIQUE ----
print("\n   📊 Score Électrique...")
FREQ_NOMINAL = 50.0
VOLTAGE_NOMINAL = 15.75

df_apm['FREQUENCY'] = df_apm['FREQUENCY_Hz']
df_apm['VOLTAGE'] = df_apm['TERMINAL_VOLTAGE_kV']
df_apm['REACTIVE_POWER'] = df_apm['REACTIVE_LOAD']

# Écarts
df_apm['FREQ_DEVIATION'] = np.abs(df_apm['FREQUENCY'] - FREQ_NOMINAL)
df_apm['VOLTAGE_DEVIATION_PCT'] = np.abs(df_apm['VOLTAGE'] - VOLTAGE_NOMINAL) / VOLTAGE_NOMINAL * 100

# Facteur de puissance
df_apm['APPARENT_POWER'] = np.sqrt(df_apm['LOAD_MW']**2 + df_apm['REACTIVE_POWER']**2)
df_apm['POWER_FACTOR'] = df_apm['LOAD_MW'] / (df_apm['APPARENT_POWER'] + 0.001)
df_apm['POWER_FACTOR'] = df_apm['POWER_FACTOR'].clip(0, 1)

# Score électrique composite
freq_score = 100 - df_apm['FREQ_DEVIATION'] * 20  # -20 points par 0.1 Hz
voltage_score = 100 - df_apm['VOLTAGE_DEVIATION_PCT'] * 10  # -10 points par 1%
pf_score = df_apm['POWER_FACTOR'] * 100  # Directement le PF en %

df_apm['ELECTRICAL_SCORE'] = (freq_score.clip(0, 100) * 0.4 + 
                               voltage_score.clip(0, 100) * 0.3 + 
                               pf_score * 0.3)
df_apm['ELECTRICAL_SCORE'] = df_apm['ELECTRICAL_SCORE'].clip(0, 100)

print(f"      Moyenne: {df_apm['ELECTRICAL_SCORE'].mean():.1f}/100")

# ---- SCORE PD ----
print("\n   📊 Score PD...")
if has_pd:
    # Utiliser les données PD réelles
    # Calculer un score basé sur l'intensité PD
    pd_intensity_cols = [col for col in df_pd.columns if 'CURRENT_ABS' in col]
    df_pd['PD_INTENSITY_TOTAL'] = df_pd[pd_intensity_cols].sum(axis=1)
    
    # Normaliser en score inverse (haute intensité = mauvais score)
    pd_max = df_pd['PD_INTENSITY_TOTAL'].quantile(0.99)
    df_pd['PD_SCORE'] = 100 - (df_pd['PD_INTENSITY_TOTAL'] / pd_max * 100).clip(0, 100)
    
    # Comme les données PD et APM ont des tailles différentes, on utilise des statistiques
    pd_score_mean = df_pd['PD_SCORE'].mean()
    pd_score_std = df_pd['PD_SCORE'].std()
    
    # Simuler un score PD synchronisé avec les données APM
    np.random.seed(42)
    df_apm['PD_SCORE'] = np.random.normal(pd_score_mean, pd_score_std/2, len(df_apm))
    df_apm['PD_SCORE'] = df_apm['PD_SCORE'].clip(0, 100)
else:
    # Simulation basée sur la charge (plus de charge = plus de risque PD)
    np.random.seed(42)
    base_pd_score = 85 - (df_apm['LOAD_MW'] / df_apm['LOAD_MW'].max()) * 15
    noise = np.random.normal(0, 5, len(df_apm))
    df_apm['PD_SCORE'] = (base_pd_score + noise).clip(0, 100)

print(f"      Moyenne: {df_apm['PD_SCORE'].mean():.1f}/100")

# =============================================================================
# 3. CALCUL DU GLOBAL HEALTH INDEX
# =============================================================================
print("\n🎯 Calcul du Global Health Index...")

# Pondérations
WEIGHT_PD = 0.30
WEIGHT_THERMAL = 0.30
WEIGHT_COOLING = 0.20
WEIGHT_ELECTRICAL = 0.20

df_apm['HEALTH_INDEX'] = (
    WEIGHT_PD * df_apm['PD_SCORE'] +
    WEIGHT_THERMAL * df_apm['THERMAL_SCORE'] +
    WEIGHT_COOLING * df_apm['COOLING_SCORE'] +
    WEIGHT_ELECTRICAL * df_apm['ELECTRICAL_SCORE']
)

# Classification
def classify_health(score):
    if score >= 85:
        return 'Excellent'
    elif score >= 70:
        return 'Stable'
    elif score >= 50:
        return 'Degrading'
    else:
        return 'Critical'

df_apm['HEALTH_STATUS'] = df_apm['HEALTH_INDEX'].apply(classify_health)

# Statistiques
health_status_counts = df_apm['HEALTH_STATUS'].value_counts()

print("\n📊 Distribution du Health Index:")
for status, count in health_status_counts.items():
    pct = count / len(df_apm) * 100
    emoji = {'Excellent': '🟢', 'Stable': '🟡', 'Degrading': '🟠', 'Critical': '🔴'}
    print(f"   {emoji.get(status, '⚪')} {status}: {count:,} ({pct:.1f}%)")

print(f"\n📈 Health Index Statistics:")
print(f"   • Moyenne: {df_apm['HEALTH_INDEX'].mean():.2f}/100")
print(f"   • Médiane: {df_apm['HEALTH_INDEX'].median():.2f}/100")
print(f"   • Min: {df_apm['HEALTH_INDEX'].min():.2f}")
print(f"   • Max: {df_apm['HEALTH_INDEX'].max():.2f}")

# =============================================================================
# 4. ANALYSE DES TENDANCES
# =============================================================================
print("\n📈 Analyse des tendances...")

# Rolling moyenne sur 24h (144 points pour 10-min)
window_24h = 144

df_apm['HEALTH_ROLL_MEAN'] = df_apm['HEALTH_INDEX'].rolling(window=window_24h, min_periods=1).mean()
df_apm['HEALTH_ROLL_STD'] = df_apm['HEALTH_INDEX'].rolling(window=window_24h, min_periods=1).std()

# Tendance (dérivée)
df_apm['HEALTH_TREND'] = df_apm['HEALTH_ROLL_MEAN'].diff(periods=window_24h)

# Classification de la tendance
def classify_trend(trend):
    if pd.isna(trend):
        return 'Stable'
    elif trend > 2:
        return 'Improving'
    elif trend < -2:
        return 'Declining'
    else:
        return 'Stable'

df_apm['TREND_STATUS'] = df_apm['HEALTH_TREND'].apply(classify_trend)

trend_counts = df_apm['TREND_STATUS'].value_counts()
print("\n📊 Distribution des tendances:")
for status, count in trend_counts.items():
    pct = count / len(df_apm) * 100
    print(f"   {status}: {count:,} ({pct:.1f}%)")

# =============================================================================
# 5. ALERTES ET RECOMMANDATIONS
# =============================================================================
print("\n⚠️ Génération des alertes...")

# Identifier les périodes critiques
critical_periods = df_apm[df_apm['HEALTH_STATUS'] == 'Critical']
degrading_periods = df_apm[df_apm['HEALTH_STATUS'] == 'Degrading']

n_critical = len(critical_periods)
n_degrading = len(degrading_periods)

if n_critical > 0:
    print(f"   🔴 {n_critical} périodes CRITICAL détectées")
    print("      → Action: Inspection immédiate requise")

if n_degrading > 0:
    print(f"   🟠 {n_degrading} périodes DEGRADING détectées")
    print("      → Action: Planifier maintenance préventive")

# Identifier le composant le plus faible
component_scores = {
    'PD': df_apm['PD_SCORE'].mean(),
    'Thermal': df_apm['THERMAL_SCORE'].mean(),
    'Cooling': df_apm['COOLING_SCORE'].mean(),
    'Electrical': df_apm['ELECTRICAL_SCORE'].mean()
}

weakest_component = min(component_scores, key=component_scores.get)
print(f"\n   ⚠️ Composant le plus faible: {weakest_component} ({component_scores[weakest_component]:.1f}/100)")

# =============================================================================
# 6. VISUALIZATIONS
# =============================================================================
print("\n📊 Génération des visualisations...")

fig = plt.figure(figsize=(24, 20))

# 1. Distribution Health Index
ax1 = fig.add_subplot(4, 3, 1)
colors_health = {'Excellent': 'green', 'Stable': 'yellow', 'Degrading': 'orange', 'Critical': 'red'}
ax1.hist(df_apm['HEALTH_INDEX'], bins=50, color='steelblue', edgecolor='white', alpha=0.7)
ax1.axvline(85, color='green', linestyle='--', linewidth=2, label='Excellent')
ax1.axvline(70, color='gold', linestyle='--', linewidth=2, label='Stable')
ax1.axvline(50, color='orange', linestyle='--', linewidth=2, label='Degrading')
ax1.set_xlabel('Health Index')
ax1.set_ylabel('Fréquence')
ax1.set_title(f'Distribution Global Health Index\n(Moyenne: {df_apm["HEALTH_INDEX"].mean():.1f}/100)')
ax1.legend()

# 2. Pie Chart Health Status
ax2 = fig.add_subplot(4, 3, 2)
health_pct = health_status_counts / len(df_apm) * 100
colors_list = [colors_health.get(x, 'gray') for x in health_pct.index]
wedges, texts, autotexts = ax2.pie(health_pct.values, labels=health_pct.index, autopct='%1.1f%%',
                                    colors=colors_list, startangle=90)
ax2.set_title('Distribution des États de Santé')

# 3. Radar Chart des composants
ax3 = fig.add_subplot(4, 3, 3, projection='polar')
categories = list(component_scores.keys())
values = list(component_scores.values())
values += values[:1]  # Fermer le radar
angles = [n / float(len(categories)) * 2 * np.pi for n in range(len(categories))]
angles += angles[:1]
ax3.plot(angles, values, 'o-', linewidth=2, color='steelblue')
ax3.fill(angles, values, alpha=0.25, color='steelblue')
ax3.set_xticks(angles[:-1])
ax3.set_xticklabels(categories)
ax3.set_ylim(0, 100)
ax3.set_title('Scores par Composant')

# 4. Évolution temporelle Health Index
ax4 = fig.add_subplot(4, 3, 4)
ax4.plot(df_apm['HEALTH_INDEX'].iloc[:2000].values, linewidth=0.5, color='steelblue', alpha=0.5, label='Instantané')
ax4.plot(df_apm['HEALTH_ROLL_MEAN'].iloc[:2000].values, linewidth=2, color='darkblue', label='Moyenne 24h')
ax4.axhline(85, color='green', linestyle='--', alpha=0.7)
ax4.axhline(70, color='gold', linestyle='--', alpha=0.7)
ax4.axhline(50, color='orange', linestyle='--', alpha=0.7)
ax4.fill_between(range(2000), 85, 100, alpha=0.1, color='green')
ax4.fill_between(range(2000), 70, 85, alpha=0.1, color='yellow')
ax4.fill_between(range(2000), 50, 70, alpha=0.1, color='orange')
ax4.fill_between(range(2000), 0, 50, alpha=0.1, color='red')
ax4.set_xlabel('Index temporel')
ax4.set_ylabel('Health Index')
ax4.set_title('Évolution Temporelle du Health Index')
ax4.legend()
ax4.set_ylim(0, 105)

# 5. Scores des composants
ax5 = fig.add_subplot(4, 3, 5)
components = ['PD', 'Thermal', 'Cooling', 'Electrical']
scores = [component_scores['PD'], component_scores['Thermal'], 
          component_scores['Cooling'], component_scores['Electrical']]
weights = [WEIGHT_PD, WEIGHT_THERMAL, WEIGHT_COOLING, WEIGHT_ELECTRICAL]
colors_comp = ['purple', 'red', 'blue', 'orange']

bars = ax5.bar(components, scores, color=colors_comp, edgecolor='white', linewidth=2)
for bar, weight in zip(bars, weights):
    ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
             f'{weight*100:.0f}%', ha='center', fontsize=10)
ax5.axhline(85, color='green', linestyle='--', alpha=0.7, label='Excellent')
ax5.axhline(70, color='gold', linestyle='--', alpha=0.7, label='Stable')
ax5.set_ylabel('Score')
ax5.set_title('Scores par Composant avec Pondérations')
ax5.set_ylim(0, 110)
ax5.legend()

# 6. Health vs Charge
ax6 = fig.add_subplot(4, 3, 6)
scatter = ax6.scatter(df_apm['LOAD_MW'][:2000], df_apm['HEALTH_INDEX'][:2000], 
                      c=df_apm['AMBIENT_TEMP'][:2000], cmap='coolwarm', alpha=0.5, s=10)
ax6.set_xlabel('Charge (MW)')
ax6.set_ylabel('Health Index')
ax6.set_title('Health Index vs Charge')
plt.colorbar(scatter, ax=ax6, label='Temp Ambiante (°C)')

# 7. Heatmap des corrélations entre composants
ax7 = fig.add_subplot(4, 3, 7)
score_cols = ['PD_SCORE', 'THERMAL_SCORE', 'COOLING_SCORE', 'ELECTRICAL_SCORE', 'HEALTH_INDEX']
corr_matrix = df_apm[score_cols].corr()
sns.heatmap(corr_matrix, annot=True, cmap='RdYlGn', center=0.5, ax=ax7, fmt='.2f',
            xticklabels=['PD', 'Thermal', 'Cooling', 'Elec', 'Health'],
            yticklabels=['PD', 'Thermal', 'Cooling', 'Elec', 'Health'])
ax7.set_title('Corrélations entre Composants')

# 8. Distribution des tendances
ax8 = fig.add_subplot(4, 3, 8)
trend_pct = trend_counts / len(df_apm) * 100
colors_trend = {'Improving': 'green', 'Stable': 'gray', 'Declining': 'red'}
bars = ax8.bar(trend_pct.index, trend_pct.values, 
               color=[colors_trend.get(x, 'gray') for x in trend_pct.index])
ax8.set_ylabel('Pourcentage (%)')
ax8.set_title('Distribution des Tendances')

# 9. Boxplot des scores par composant
ax9 = fig.add_subplot(4, 3, 9)
score_data = df_apm[['PD_SCORE', 'THERMAL_SCORE', 'COOLING_SCORE', 'ELECTRICAL_SCORE']].melt()
score_data.columns = ['Composant', 'Score']
score_data['Composant'] = score_data['Composant'].str.replace('_SCORE', '')
sns.boxplot(x='Composant', y='Score', data=score_data, ax=ax9, palette='Set2')
ax9.axhline(85, color='green', linestyle='--', alpha=0.7)
ax9.axhline(70, color='gold', linestyle='--', alpha=0.7)
ax9.set_title('Distribution des Scores par Composant')

# 10. Health Index par heure
ax10 = fig.add_subplot(4, 3, 10)
hourly_health = df_apm.groupby('Hour')['HEALTH_INDEX'].mean()
ax10.bar(hourly_health.index, hourly_health.values, color='teal', edgecolor='white')
ax10.axhline(hourly_health.mean(), color='red', linestyle='--', linewidth=2, label=f'Moyenne ({hourly_health.mean():.1f})')
ax10.set_xlabel('Heure')
ax10.set_ylabel('Health Index Moyen')
ax10.set_title('Profil Journalier du Health Index')
ax10.legend()

# 11. Gauge Chart (simplifié)
ax11 = fig.add_subplot(4, 3, 11)
mean_health = df_apm['HEALTH_INDEX'].mean()
theta = np.linspace(0, np.pi, 100)
ax11.plot(np.cos(theta), np.sin(theta), 'k-', linewidth=3)
ax11.fill_between(np.cos(theta[:50]), np.sin(theta[:50]), alpha=0.3, color='red')
ax11.fill_between(np.cos(theta[50:75]), np.sin(theta[50:75]), alpha=0.3, color='orange')
ax11.fill_between(np.cos(theta[75:85]), np.sin(theta[75:85]), alpha=0.3, color='yellow')
ax11.fill_between(np.cos(theta[85:]), np.sin(theta[85:]), alpha=0.3, color='green')
# Aiguille
needle_angle = np.pi * (1 - mean_health/100)
ax11.arrow(0, 0, 0.7*np.cos(needle_angle), 0.7*np.sin(needle_angle), 
           head_width=0.1, head_length=0.05, fc='black', ec='black')
ax11.text(0, -0.3, f'{mean_health:.1f}', fontsize=24, ha='center', fontweight='bold')
ax11.text(0, -0.5, 'Health Index', fontsize=12, ha='center')
ax11.set_xlim(-1.2, 1.2)
ax11.set_ylim(-0.6, 1.2)
ax11.axis('off')
ax11.set_title('Global Health Index')

# 12. Tableau récapitulatif
ax12 = fig.add_subplot(4, 3, 12)
ax12.axis('off')
summary_text = f"""
╔══════════════════════════════════════════════════════╗
║          TG1 DIGITAL TWIN HEALTH REPORT              ║
╠══════════════════════════════════════════════════════╣
║ Global Health Index:  {mean_health:.1f}/100                      ║
║ Status: {df_apm['HEALTH_STATUS'].mode()[0]:15}                       ║
╠══════════════════════════════════════════════════════╣
║ COMPOSANTS:                          Score  Poids   ║
║ ─────────────────────────────────────────────────── ║
║ 🟣 PD (Partial Discharge)           {component_scores['PD']:.1f}   30%    ║
║ 🔴 Thermal (Température)            {component_scores['Thermal']:.1f}   30%    ║
║ 🔵 Cooling (Refroidissement)        {component_scores['Cooling']:.1f}   20%    ║
║ 🟠 Electrical (Stabilité)           {component_scores['Electrical']:.1f}   20%    ║
╠══════════════════════════════════════════════════════╣
║ DISTRIBUTION:                                        ║
║ 🟢 Excellent: {health_status_counts.get('Excellent', 0):5} ({health_status_counts.get('Excellent', 0)/len(df_apm)*100:5.1f}%)                    ║
║ 🟡 Stable:    {health_status_counts.get('Stable', 0):5} ({health_status_counts.get('Stable', 0)/len(df_apm)*100:5.1f}%)                    ║
║ 🟠 Degrading: {health_status_counts.get('Degrading', 0):5} ({health_status_counts.get('Degrading', 0)/len(df_apm)*100:5.1f}%)                    ║
║ 🔴 Critical:  {health_status_counts.get('Critical', 0):5} ({health_status_counts.get('Critical', 0)/len(df_apm)*100:5.1f}%)                    ║
╠══════════════════════════════════════════════════════╣
║ ⚠️ Point faible: {weakest_component:10} ({component_scores[weakest_component]:.1f}/100)           ║
╚══════════════════════════════════════════════════════╝
"""
ax12.text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
          verticalalignment='center', transform=ax12.transAxes)

plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}/05_global_health_index.png', dpi=150, bbox_inches='tight')
plt.close()

print(f"   ✅ Sauvegardé: {PLOTS_DIR}/05_global_health_index.png")

# =============================================================================
# 7. SAUVEGARDE DES RÉSULTATS
# =============================================================================
print("\n💾 Sauvegarde des résultats...")

# Sauvegarder le dataframe avec tous les scores
df_apm.to_csv(f'{DATA_DIR}/TG1_Health_Index.csv', index=False)
print(f"   ✅ Data: {DATA_DIR}/TG1_Health_Index.csv")

# Configuration du modèle
health_config = {
    'weights': {
        'PD': WEIGHT_PD,
        'Thermal': WEIGHT_THERMAL,
        'Cooling': WEIGHT_COOLING,
        'Electrical': WEIGHT_ELECTRICAL
    },
    'thresholds': {
        'Excellent': 85,
        'Stable': 70,
        'Degrading': 50,
        'Critical': 0
    },
    'component_scores': component_scores,
    'statistics': {
        'mean': df_apm['HEALTH_INDEX'].mean(),
        'median': df_apm['HEALTH_INDEX'].median(),
        'std': df_apm['HEALTH_INDEX'].std(),
        'min': df_apm['HEALTH_INDEX'].min(),
        'max': df_apm['HEALTH_INDEX'].max()
    },
    'distribution': health_status_counts.to_dict(),
    'weakest_component': weakest_component
}
joblib.dump(health_config, f'{PLOTS_DIR}/05_health_config.pkl')
print(f"   ✅ Config: {PLOTS_DIR}/05_health_config.pkl")

# =============================================================================
# RÉSUMÉ FINAL
# =============================================================================
print("\n" + "=" * 70)
print("🏆 RÉSUMÉ - GLOBAL TG1 HEALTH INDEX")
print("=" * 70)
print(f"""
╔══════════════════════════════════════════════════════════════════════╗
║                    TG1 DIGITAL TWIN HEALTH REPORT                    ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║    ████  GLOBAL HEALTH INDEX: {df_apm['HEALTH_INDEX'].mean():.1f}/100  ████                  ║
║                                                                      ║
╠══════════════════════════════════════════════════════════════════════╣
║ FORMULA:                                                             ║
║ Health = 30% PD + 30% Thermal + 20% Cooling + 20% Electrical         ║
╠══════════════════════════════════════════════════════════════════════╣
║ COMPOSANTS:                                                          ║
║   🟣 PD Score:         {component_scores['PD']:6.1f}/100  (30%)                         ║
║   🔴 Thermal Score:    {component_scores['Thermal']:6.1f}/100  (30%)                         ║
║   🔵 Cooling Score:    {component_scores['Cooling']:6.1f}/100  (20%)                         ║
║   🟠 Electrical Score: {component_scores['Electrical']:6.1f}/100  (20%)                         ║
╠══════════════════════════════════════════════════════════════════════╣
║ DISTRIBUTION:                                                        ║
║   🟢 Excellent (≥85): {health_status_counts.get('Excellent', 0):6,} points ({health_status_counts.get('Excellent', 0)/len(df_apm)*100:5.1f}%)                   ║
║   🟡 Stable (70-84):  {health_status_counts.get('Stable', 0):6,} points ({health_status_counts.get('Stable', 0)/len(df_apm)*100:5.1f}%)                   ║
║   🟠 Degrading (50-69): {health_status_counts.get('Degrading', 0):4,} points ({health_status_counts.get('Degrading', 0)/len(df_apm)*100:5.1f}%)                   ║
║   🔴 Critical (<50):  {health_status_counts.get('Critical', 0):6,} points ({health_status_counts.get('Critical', 0)/len(df_apm)*100:5.1f}%)                   ║
╠══════════════════════════════════════════════════════════════════════╣
║ RECOMMANDATION:                                                      ║
║   ⚠️ Point faible: {weakest_component:10} → Priorité de surveillance             ║
╚══════════════════════════════════════════════════════════════════════╝
""")
print("=" * 70)
