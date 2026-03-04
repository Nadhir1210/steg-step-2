"""
🔥 MODULE 3 - Electrical Stability Analysis
============================================
TG1 Turbo-Alternator - Digital Twin Health Monitoring System

Objectif: Analyser la stabilité électrique du turbo-alternateur
- Frequency Stability: Variation de fréquence autour de 50 Hz
- Voltage Stability: Écart par rapport à 15.75 kV nominal
- Reactive Load Analysis: Oscillations de puissance réactive
- Power Factor Analysis: cos(φ) = P / √(P² + Q²)

Auteur: Nadhir - Stage STEG 2026
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy import stats
import joblib
import warnings
warnings.filterwarnings('ignore')

# Configuration
plt.style.use('seaborn-v0_8-whitegrid')
PLOTS_DIR = "tg1_monitoring/plots"
DATA_DIR = "LAST_DATA"

print("=" * 70)
print("🔥 MODULE 3 - ELECTRICAL STABILITY ANALYSIS")
print("=" * 70)

# =============================================================================
# 1. CHARGEMENT DES DONNÉES
# =============================================================================
print("\n📥 Chargement des données APM Alternateur...")

df = pd.read_csv(f"{DATA_DIR}/APM_Alternateur_10min_ML.csv")
print(f"   Dataset: {len(df):,} lignes × {len(df.columns)} colonnes")

# =============================================================================
# 2. FEATURE ENGINEERING - ELECTRICAL
# =============================================================================
print("\n🔧 Feature Engineering - Electrical...")

# Variables principales
df['FREQUENCY'] = df['FREQUENCY_Hz']
df['VOLTAGE'] = df['TERMINAL_VOLTAGE_kV']
df['ACTIVE_POWER'] = df['MODE_TAG_1']  # MW
df['REACTIVE_POWER'] = df['REACTIVE_LOAD']  # MVAR

# Vitesse: utiliser SPEED_rpm ou SPEED_CTRL_pct selon disponibilité
if 'SPEED_rpm' in df.columns:
    df['SPEED_PERCENT'] = df['SPEED_rpm'] / 3000 * 100
elif 'SPEED_CTRL_pct' in df.columns:
    df['SPEED_PERCENT'] = df['SPEED_CTRL_pct']
else:
    df['SPEED_PERCENT'] = 100  # Default

# Valeurs nominales
FREQ_NOMINAL = 50.0  # Hz
VOLTAGE_NOMINAL = 15.75  # kV

# Écarts par rapport aux nominaux
df['FREQ_DEVIATION'] = df['FREQUENCY'] - FREQ_NOMINAL
df['FREQ_DEVIATION_ABS'] = np.abs(df['FREQ_DEVIATION'])
df['VOLTAGE_DEVIATION'] = df['VOLTAGE'] - VOLTAGE_NOMINAL
df['VOLTAGE_DEVIATION_ABS'] = np.abs(df['VOLTAGE_DEVIATION'])
df['VOLTAGE_DEVIATION_PCT'] = (df['VOLTAGE_DEVIATION'] / VOLTAGE_NOMINAL) * 100

# Puissance apparente et facteur de puissance
df['APPARENT_POWER'] = np.sqrt(df['ACTIVE_POWER']**2 + df['REACTIVE_POWER']**2)
df['POWER_FACTOR'] = df['ACTIVE_POWER'] / (df['APPARENT_POWER'] + 0.001)
df['POWER_FACTOR'] = df['POWER_FACTOR'].clip(0, 1)

# Rolling statistics (stabilité temporelle)
window = 6  # 1 heure (6 x 10min)

df['FREQ_ROLL_STD'] = df['FREQUENCY'].rolling(window=window, min_periods=1).std()
df['FREQ_ROLL_MEAN'] = df['FREQUENCY'].rolling(window=window, min_periods=1).mean()
df['VOLTAGE_ROLL_STD'] = df['VOLTAGE'].rolling(window=window, min_periods=1).std()
df['REACTIVE_ROLL_STD'] = df['REACTIVE_POWER'].rolling(window=window, min_periods=1).std()

# Coefficient de variation
df['FREQ_CV'] = df['FREQ_ROLL_STD'] / (df['FREQ_ROLL_MEAN'] + 0.001) * 100

# Nettoyer les données (machine en service)
df_clean = df[(df['ACTIVE_POWER'] > 5) & (df['FREQUENCY'] > 0)].copy()
print(f"   Dataset nettoyé: {len(df_clean):,} lignes (machine en service)")

# =============================================================================
# 3. FREQUENCY STABILITY ANALYSIS
# =============================================================================
print("\n📊 Frequency Stability Analysis...")

# Statistiques fréquence
freq_mean = df_clean['FREQUENCY'].mean()
freq_std = df_clean['FREQUENCY'].std()
freq_min = df_clean['FREQUENCY'].min()
freq_max = df_clean['FREQUENCY'].max()

print(f"   Fréquence moyenne: {freq_mean:.4f} Hz")
print(f"   Écart-type: {freq_std:.4f} Hz")
print(f"   Plage: [{freq_min:.2f} - {freq_max:.2f}] Hz")

# Classification de la stabilité fréquence
# Normes: ±0.2 Hz normal, ±0.5 Hz acceptable, >0.5 Hz problématique
def classify_frequency(deviation):
    if abs(deviation) < 0.2:
        return 'Excellent'
    elif abs(deviation) < 0.5:
        return 'Normal'
    elif abs(deviation) < 1.0:
        return 'Warning'
    else:
        return 'Critical'

df_clean['FREQ_STATUS'] = df_clean['FREQ_DEVIATION'].apply(classify_frequency)

freq_status_counts = df_clean['FREQ_STATUS'].value_counts()
print("\n📈 Distribution Stabilité Fréquence:")
for status, count in freq_status_counts.items():
    pct = count / len(df_clean) * 100
    print(f"   {status}: {count:,} ({pct:.1f}%)")

# =============================================================================
# 4. VOLTAGE STABILITY ANALYSIS
# =============================================================================
print("\n📊 Voltage Stability Analysis...")

# Statistiques tension
voltage_mean = df_clean['VOLTAGE'].mean()
voltage_std = df_clean['VOLTAGE'].std()

print(f"   Tension moyenne: {voltage_mean:.3f} kV (nominal: {VOLTAGE_NOMINAL} kV)")
print(f"   Écart-type: {voltage_std:.4f} kV")
print(f"   Écart moyen: {df_clean['VOLTAGE_DEVIATION'].mean():.4f} kV")

# Classification tension (±5% acceptable)
def classify_voltage(deviation_pct):
    if abs(deviation_pct) < 2:
        return 'Excellent'
    elif abs(deviation_pct) < 5:
        return 'Normal'
    elif abs(deviation_pct) < 10:
        return 'Warning'
    else:
        return 'Critical'

df_clean['VOLTAGE_STATUS'] = df_clean['VOLTAGE_DEVIATION_PCT'].apply(classify_voltage)

voltage_status_counts = df_clean['VOLTAGE_STATUS'].value_counts()
print("\n📈 Distribution Stabilité Tension:")
for status, count in voltage_status_counts.items():
    pct = count / len(df_clean) * 100
    print(f"   {status}: {count:,} ({pct:.1f}%)")

# =============================================================================
# 5. REACTIVE POWER ANALYSIS
# =============================================================================
print("\n📊 Reactive Power Analysis...")

reactive_mean = df_clean['REACTIVE_POWER'].mean()
reactive_std = df_clean['REACTIVE_POWER'].std()
reactive_min = df_clean['REACTIVE_POWER'].min()
reactive_max = df_clean['REACTIVE_POWER'].max()

print(f"   Réactive moyenne: {reactive_mean:.2f} MVAR")
print(f"   Écart-type: {reactive_std:.2f} MVAR")
print(f"   Plage: [{reactive_min:.2f} - {reactive_max:.2f}] MVAR")

# Oscillations réactives
df_clean['REACTIVE_OSCILLATION'] = np.abs(df_clean['REACTIVE_POWER'] - reactive_mean)

# =============================================================================
# 6. POWER FACTOR ANALYSIS
# =============================================================================
print("\n📊 Power Factor Analysis...")

pf_mean = df_clean['POWER_FACTOR'].mean()
pf_std = df_clean['POWER_FACTOR'].std()

print(f"   Facteur de puissance moyen: {pf_mean:.4f}")
print(f"   Écart-type: {pf_std:.4f}")

# Classification facteur de puissance
def classify_power_factor(pf):
    if pf >= 0.95:
        return 'Excellent'
    elif pf >= 0.90:
        return 'Bon'
    elif pf >= 0.85:
        return 'Acceptable'
    elif pf >= 0.80:
        return 'Faible'
    else:
        return 'Critique'

df_clean['PF_STATUS'] = df_clean['POWER_FACTOR'].apply(classify_power_factor)

pf_status_counts = df_clean['PF_STATUS'].value_counts()
print("\n📈 Distribution Facteur de Puissance:")
for status, count in pf_status_counts.items():
    pct = count / len(df_clean) * 100
    print(f"   {status}: {count:,} ({pct:.1f}%)")

# =============================================================================
# 7. ELECTRICAL STABILITY SCORE
# =============================================================================
print("\n🎯 Electrical Stability Score...")

# Score composite de stabilité électrique (0-100)
def calculate_electrical_score(row):
    score = 100
    
    # Pénalité fréquence (max -30 points)
    freq_penalty = min(30, abs(row['FREQ_DEVIATION']) * 30)
    score -= freq_penalty
    
    # Pénalité tension (max -30 points)
    voltage_penalty = min(30, abs(row['VOLTAGE_DEVIATION_PCT']) * 3)
    score -= voltage_penalty
    
    # Pénalité facteur de puissance (max -20 points)
    pf_penalty = max(0, (0.95 - row['POWER_FACTOR']) * 100)
    score -= min(20, pf_penalty)
    
    # Pénalité variabilité (max -20 points)
    if row['FREQ_ROLL_STD'] > 0.1:
        score -= min(10, row['FREQ_ROLL_STD'] * 50)
    if row['VOLTAGE_ROLL_STD'] > 0.1:
        score -= min(10, row['VOLTAGE_ROLL_STD'] * 50)
    
    return max(0, min(100, score))

df_clean['ELECTRICAL_SCORE'] = df_clean.apply(calculate_electrical_score, axis=1)

# Classification du score
def classify_electrical_score(score):
    if score >= 90:
        return 'Excellent'
    elif score >= 75:
        return 'Bon'
    elif score >= 60:
        return 'Acceptable'
    elif score >= 40:
        return 'Warning'
    else:
        return 'Critical'

df_clean['ELECTRICAL_STATUS'] = df_clean['ELECTRICAL_SCORE'].apply(classify_electrical_score)

elec_status_counts = df_clean['ELECTRICAL_STATUS'].value_counts()
print("\n📊 Distribution Stabilité Électrique:")
for status, count in elec_status_counts.items():
    pct = count / len(df_clean) * 100
    print(f"   {status}: {count:,} ({pct:.1f}%)")

# Statistiques du score
print(f"\n📈 Score Électrique:")
print(f"   Moyenne: {df_clean['ELECTRICAL_SCORE'].mean():.1f}/100")
print(f"   Médiane: {df_clean['ELECTRICAL_SCORE'].median():.1f}/100")
print(f"   Min: {df_clean['ELECTRICAL_SCORE'].min():.1f} | Max: {df_clean['ELECTRICAL_SCORE'].max():.1f}")

# =============================================================================
# 8. VISUALIZATIONS
# =============================================================================
print("\n📊 Génération des visualisations...")

fig = plt.figure(figsize=(20, 16))

# 1. Distribution Fréquence
ax1 = fig.add_subplot(3, 3, 1)
ax1.hist(df_clean['FREQUENCY'], bins=50, color='steelblue', edgecolor='white', alpha=0.7)
ax1.axvline(FREQ_NOMINAL, color='green', linestyle='--', linewidth=2, label=f'Nominal ({FREQ_NOMINAL} Hz)')
ax1.axvline(FREQ_NOMINAL - 0.5, color='orange', linestyle='--', alpha=0.7)
ax1.axvline(FREQ_NOMINAL + 0.5, color='orange', linestyle='--', alpha=0.7)
ax1.set_xlabel('Fréquence (Hz)')
ax1.set_ylabel('Fréquence')
ax1.set_title('Distribution Fréquence')
ax1.legend()

# 2. Distribution Tension
ax2 = fig.add_subplot(3, 3, 2)
ax2.hist(df_clean['VOLTAGE'], bins=50, color='coral', edgecolor='white', alpha=0.7)
ax2.axvline(VOLTAGE_NOMINAL, color='green', linestyle='--', linewidth=2, label=f'Nominal ({VOLTAGE_NOMINAL} kV)')
ax2.axvline(VOLTAGE_NOMINAL * 0.95, color='orange', linestyle='--', alpha=0.7, label='±5%')
ax2.axvline(VOLTAGE_NOMINAL * 1.05, color='orange', linestyle='--', alpha=0.7)
ax2.set_xlabel('Tension (kV)')
ax2.set_ylabel('Fréquence')
ax2.set_title('Distribution Tension')
ax2.legend()

# 3. Facteur de Puissance
ax3 = fig.add_subplot(3, 3, 3)
ax3.hist(df_clean['POWER_FACTOR'], bins=50, color='mediumpurple', edgecolor='white', alpha=0.7)
ax3.axvline(0.95, color='green', linestyle='--', linewidth=2, label='Excellent (>0.95)')
ax3.axvline(0.90, color='orange', linestyle='--', alpha=0.7, label='Bon (>0.90)')
ax3.axvline(0.85, color='red', linestyle='--', alpha=0.7, label='Acceptable (>0.85)')
ax3.set_xlabel('Facteur de Puissance (cos φ)')
ax3.set_ylabel('Fréquence')
ax3.set_title('Distribution Facteur de Puissance')
ax3.legend()

# 4. P vs Q (Diagramme PQ)
ax4 = fig.add_subplot(3, 3, 4)
scatter = ax4.scatter(df_clean['ACTIVE_POWER'][:2000], df_clean['REACTIVE_POWER'][:2000], 
                      c=df_clean['POWER_FACTOR'][:2000], cmap='RdYlGn', alpha=0.6, s=10)
ax4.axhline(0, color='gray', linestyle='-', alpha=0.5)
ax4.axvline(0, color='gray', linestyle='-', alpha=0.5)
ax4.set_xlabel('Puissance Active P (MW)')
ax4.set_ylabel('Puissance Réactive Q (MVAR)')
ax4.set_title('Diagramme P-Q')
plt.colorbar(scatter, ax=ax4, label='cos φ')

# 5. Fréquence temporelle
ax5 = fig.add_subplot(3, 3, 5)
ax5.plot(df_clean['FREQUENCY'].iloc[:500].values, linewidth=0.5, color='steelblue')
ax5.axhline(FREQ_NOMINAL, color='green', linestyle='--', linewidth=1.5)
ax5.axhline(FREQ_NOMINAL + 0.5, color='orange', linestyle='--', alpha=0.7)
ax5.axhline(FREQ_NOMINAL - 0.5, color='orange', linestyle='--', alpha=0.7)
ax5.fill_between(range(500), FREQ_NOMINAL - 0.2, FREQ_NOMINAL + 0.2, alpha=0.2, color='green')
ax5.set_xlabel('Index temporel')
ax5.set_ylabel('Fréquence (Hz)')
ax5.set_title('Évolution Temporelle Fréquence')

# 6. Tension temporelle
ax6 = fig.add_subplot(3, 3, 6)
ax6.plot(df_clean['VOLTAGE'].iloc[:500].values, linewidth=0.5, color='coral')
ax6.axhline(VOLTAGE_NOMINAL, color='green', linestyle='--', linewidth=1.5)
ax6.axhline(VOLTAGE_NOMINAL * 1.05, color='orange', linestyle='--', alpha=0.7)
ax6.axhline(VOLTAGE_NOMINAL * 0.95, color='orange', linestyle='--', alpha=0.7)
ax6.set_xlabel('Index temporel')
ax6.set_ylabel('Tension (kV)')
ax6.set_title('Évolution Temporelle Tension')

# 7. Score Électrique Distribution
ax7 = fig.add_subplot(3, 3, 7)
ax7.hist(df_clean['ELECTRICAL_SCORE'], bins=50, color='teal', edgecolor='white', alpha=0.7)
ax7.axvline(90, color='green', linestyle='--', linewidth=2, label='Excellent')
ax7.axvline(75, color='yellow', linestyle='--', linewidth=1.5, label='Bon')
ax7.axvline(60, color='orange', linestyle='--', linewidth=1.5, label='Acceptable')
ax7.axvline(40, color='red', linestyle='--', linewidth=1.5, label='Warning')
ax7.set_xlabel('Score Électrique')
ax7.set_ylabel('Fréquence')
ax7.set_title('Distribution Score Électrique')
ax7.legend(loc='upper left')

# 8. Statut Électrique
ax8 = fig.add_subplot(3, 3, 8)
colors_elec = {'Excellent': 'green', 'Bon': 'lightgreen', 'Acceptable': 'yellow', 
               'Warning': 'orange', 'Critical': 'red'}
elec_pct = elec_status_counts / len(df_clean) * 100
ax8.pie(elec_pct.values, labels=elec_pct.index, autopct='%1.1f%%',
        colors=[colors_elec.get(x, 'gray') for x in elec_pct.index], startangle=90)
ax8.set_title('Statut Stabilité Électrique')

# 9. Heatmap corrélations
ax9 = fig.add_subplot(3, 3, 9)
corr_cols = ['FREQUENCY', 'VOLTAGE', 'ACTIVE_POWER', 'REACTIVE_POWER', 
             'POWER_FACTOR', 'ELECTRICAL_SCORE']
corr_matrix = df_clean[corr_cols].corr()
sns.heatmap(corr_matrix, annot=True, cmap='RdYlBu_r', center=0, ax=ax9, fmt='.2f')
ax9.set_title('Corrélations Électriques')

plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}/03_electrical_stability.png', dpi=150, bbox_inches='tight')
plt.close()

print(f"   ✅ Sauvegardé: {PLOTS_DIR}/03_electrical_stability.png")

# =============================================================================
# 9. SAUVEGARDE DES RÉSULTATS
# =============================================================================
print("\n💾 Sauvegarde des résultats...")

electrical_config = {
    'nominal_values': {
        'frequency_hz': FREQ_NOMINAL,
        'voltage_kv': VOLTAGE_NOMINAL
    },
    'statistics': {
        'freq_mean': freq_mean,
        'freq_std': freq_std,
        'voltage_mean': voltage_mean,
        'voltage_std': voltage_std,
        'pf_mean': pf_mean
    },
    'thresholds': {
        'freq_excellent': 0.2,
        'freq_normal': 0.5,
        'freq_warning': 1.0,
        'voltage_excellent_pct': 2,
        'voltage_normal_pct': 5,
        'voltage_warning_pct': 10
    },
    'score_distribution': elec_status_counts.to_dict()
}
joblib.dump(electrical_config, f'{PLOTS_DIR}/03_electrical_config.pkl')

print(f"   ✅ Config: {PLOTS_DIR}/03_electrical_config.pkl")

# =============================================================================
# RÉSUMÉ
# =============================================================================
print("\n" + "=" * 70)
print("📊 RÉSUMÉ - ELECTRICAL STABILITY ANALYSIS")
print("=" * 70)
print(f"""
⚡ Fréquence:
   • Moyenne: {freq_mean:.4f} Hz (nominal: {FREQ_NOMINAL} Hz)
   • Écart-type: {freq_std:.4f} Hz
   • Excellent (<±0.2 Hz): {freq_status_counts.get('Excellent', 0):,}
   • Warning/Critical: {freq_status_counts.get('Warning', 0) + freq_status_counts.get('Critical', 0):,}

🔌 Tension:
   • Moyenne: {voltage_mean:.3f} kV (nominal: {VOLTAGE_NOMINAL} kV)
   • Écart moyen: {df_clean['VOLTAGE_DEVIATION'].mean():.4f} kV
   • Dans les limites (±5%): {voltage_status_counts.get('Excellent', 0) + voltage_status_counts.get('Normal', 0):,}

⚙️ Facteur de Puissance:
   • Moyen: {pf_mean:.4f}
   • Excellent (>0.95): {pf_status_counts.get('Excellent', 0):,} ({pf_status_counts.get('Excellent', 0)/len(df_clean)*100:.1f}%)

📈 Score Stabilité Électrique:
   • Moyenne: {df_clean['ELECTRICAL_SCORE'].mean():.1f}/100
   • Excellent (>90): {elec_status_counts.get('Excellent', 0):,} ({elec_status_counts.get('Excellent', 0)/len(df_clean)*100:.1f}%)
   • Bon (75-90): {elec_status_counts.get('Bon', 0):,} ({elec_status_counts.get('Bon', 0)/len(df_clean)*100:.1f}%)
   • Warning/Critical: {elec_status_counts.get('Warning', 0) + elec_status_counts.get('Critical', 0):,}
""")
print("=" * 70)
