"""
🔥 MODULE 2 - Cooling Efficiency Model
=======================================
TG1 Turbo-Alternator - Digital Twin Health Monitoring System

Objectif: Monitorer l'efficacité du système de refroidissement
- Cooling Efficiency Index: ΔT = Hot Air - Cold Air
- Anomaly Detection: Si ΔT anormal → ventilation dégradée
- Control Chart (SPC): Statistical Process Control

Auteur: Nadhir - Stage STEG 2026
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
import joblib
import warnings
warnings.filterwarnings('ignore')

# Configuration
plt.style.use('seaborn-v0_8-whitegrid')
PLOTS_DIR = "tg1_monitoring/plots"
DATA_DIR = "LAST_DATA"

print("=" * 70)
print("🔥 MODULE 2 - COOLING EFFICIENCY MODEL")
print("=" * 70)

# =============================================================================
# 1. CHARGEMENT DES DONNÉES
# =============================================================================
print("\n📥 Chargement des données APM Alternateur...")

df = pd.read_csv(f"{DATA_DIR}/APM_Alternateur_10min_ML.csv")
print(f"   Dataset: {len(df):,} lignes × {len(df.columns)} colonnes")

# =============================================================================
# 2. FEATURE ENGINEERING - COOLING
# =============================================================================
print("\n🔧 Feature Engineering - Cooling...")

# Delta température (efficacité refroidissement)
df['DELTA_T_1'] = df['ENCLOSED_HOT_AIR_TEMP_1_degC'] - df['ENCLOSED_COLD_AIR_TEMP_1_degC']
df['DELTA_T_2'] = df['ENCLOSED_HOT_AIR_TEMP_2_degC'] - df['ENCLOSED_COLD_AIR_TEMP_2_degC']
df['DELTA_T_MEAN'] = (df['DELTA_T_1'] + df['DELTA_T_2']) / 2

# Température moyenne air chaud/froid
df['HOT_AIR_MEAN'] = (df['ENCLOSED_HOT_AIR_TEMP_1_degC'] + df['ENCLOSED_HOT_AIR_TEMP_2_degC']) / 2
df['COLD_AIR_MEAN'] = (df['ENCLOSED_COLD_AIR_TEMP_1_degC'] + df['ENCLOSED_COLD_AIR_TEMP_2_degC']) / 2

# Charge et ambiant
df['LOAD_MW'] = df['MODE_TAG_1']
df['AMBIENT_TEMP'] = df['AMBIENT_AIR_TEMP_C']

# Ratio efficacité: ΔT / Charge (plus élevé = meilleure dissipation)
df['COOLING_EFFICIENCY'] = df['DELTA_T_MEAN'] / (df['LOAD_MW'] + 1)

# Température stator
temp_cols = [col for col in df.columns if 'STATOR' in col and 'TEMP' in col]
df['STATOR_TEMP_MEAN'] = df[temp_cols].mean(axis=1)

# Ratio stator / ambiant (stress thermique)
df['THERMAL_STRESS'] = df['STATOR_TEMP_MEAN'] - df['AMBIENT_TEMP']

# Nettoyer les données
df_clean = df[(df['LOAD_MW'] > 5) & (df['DELTA_T_MEAN'] > 0)].copy()
print(f"   Dataset nettoyé: {len(df_clean):,} lignes (charge > 5 MW)")

# =============================================================================
# 3. STATISTICAL PROCESS CONTROL (SPC)
# =============================================================================
print("\n📊 Statistical Process Control (SPC)...")

# Calculer les limites de contrôle pour DELTA_T
delta_t_mean = df_clean['DELTA_T_MEAN'].mean()
delta_t_std = df_clean['DELTA_T_MEAN'].std()

# Limites de contrôle (3-sigma)
UCL = delta_t_mean + 3 * delta_t_std  # Upper Control Limit
LCL = max(0, delta_t_mean - 3 * delta_t_std)  # Lower Control Limit
UWL = delta_t_mean + 2 * delta_t_std  # Upper Warning Limit
LWL = max(0, delta_t_mean - 2 * delta_t_std)  # Lower Warning Limit

print(f"   DELTA_T statistiques:")
print(f"   • Mean: {delta_t_mean:.2f}°C")
print(f"   • Std: {delta_t_std:.2f}°C")
print(f"   • UCL (Upper Control): {UCL:.2f}°C")
print(f"   • LCL (Lower Control): {LCL:.2f}°C")

# Classification SPC
def classify_spc(value):
    if value > UCL or value < LCL:
        return 'Out of Control'
    elif value > UWL or value < LWL:
        return 'Warning'
    else:
        return 'Normal'

df_clean['SPC_STATUS'] = df_clean['DELTA_T_MEAN'].apply(classify_spc)

spc_counts = df_clean['SPC_STATUS'].value_counts()
print("\n📈 Distribution SPC:")
for status, count in spc_counts.items():
    pct = count / len(df_clean) * 100
    print(f"   {status}: {count:,} ({pct:.1f}%)")

# =============================================================================
# 4. COOLING EFFICIENCY INDEX
# =============================================================================
print("\n🎯 Cooling Efficiency Index...")

# Index d'efficacité normalisé (0-100)
# Plus le DELTA_T est élevé pour une charge donnée, plus le refroidissement est efficace

# Modèle attendu: DELTA_T attendu = a * LOAD + b
from sklearn.linear_model import LinearRegression

# Entraîner un modèle linéaire simple
X_load = df_clean[['LOAD_MW', 'AMBIENT_TEMP']].values
y_delta = df_clean['DELTA_T_MEAN'].values

lr_model = LinearRegression()
lr_model.fit(X_load, y_delta)
delta_t_expected = lr_model.predict(X_load)

# Efficacité = Réel / Attendu (en %)
df_clean['DELTA_T_EXPECTED'] = delta_t_expected
df_clean['COOLING_INDEX'] = (df_clean['DELTA_T_MEAN'] / df_clean['DELTA_T_EXPECTED']) * 100
df_clean['COOLING_INDEX'] = df_clean['COOLING_INDEX'].clip(0, 200)  # Limiter les valeurs extrêmes

# Classification de l'efficacité
def classify_efficiency(index):
    if index < 70:
        return 'Dégradé'
    elif index < 90:
        return 'Modéré'
    elif index < 110:
        return 'Normal'
    elif index < 130:
        return 'Bon'
    else:
        return 'Excellent'

df_clean['EFFICIENCY_CLASS'] = df_clean['COOLING_INDEX'].apply(classify_efficiency)

eff_counts = df_clean['EFFICIENCY_CLASS'].value_counts()
print("\n📈 Distribution Efficacité:")
for status, count in eff_counts.items():
    pct = count / len(df_clean) * 100
    print(f"   {status}: {count:,} ({pct:.1f}%)")

# =============================================================================
# 5. ISOLATION FOREST ANOMALY DETECTION
# =============================================================================
print("\n🔴 Isolation Forest Anomaly Detection...")

# Features pour détection d'anomalies
cooling_features = ['DELTA_T_MEAN', 'LOAD_MW', 'AMBIENT_TEMP', 'HOT_AIR_MEAN', 
                    'COLD_AIR_MEAN', 'THERMAL_STRESS']

X_cooling = df_clean[cooling_features].values

# Normalisation
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_cooling)

# Isolation Forest
iso_forest = IsolationForest(
    n_estimators=200,
    contamination=0.05,  # 5% d'anomalies attendues
    random_state=42,
    n_jobs=-1
)

df_clean['ISO_ANOMALY'] = iso_forest.fit_predict(X_scaled)
df_clean['ISO_ANOMALY'] = df_clean['ISO_ANOMALY'].map({1: 'Normal', -1: 'Anomaly'})

iso_counts = df_clean['ISO_ANOMALY'].value_counts()
print(f"   Points normaux: {iso_counts.get('Normal', 0):,}")
print(f"   Anomalies: {iso_counts.get('Anomaly', 0):,}")

# =============================================================================
# 6. STATUT FINAL COOLING
# =============================================================================
print("\n🎯 Classification finale du refroidissement...")

def get_final_cooling_status(row):
    """Combine tous les indicateurs pour un statut final"""
    score = 0
    
    # SPC
    if row['SPC_STATUS'] == 'Out of Control':
        score += 2
    elif row['SPC_STATUS'] == 'Warning':
        score += 1
    
    # Efficacité
    if row['EFFICIENCY_CLASS'] == 'Dégradé':
        score += 2
    elif row['EFFICIENCY_CLASS'] == 'Modéré':
        score += 1
    
    # Isolation Forest
    if row['ISO_ANOMALY'] == 'Anomaly':
        score += 1
    
    # Classification finale
    if score >= 4:
        return 'Critical'
    elif score >= 2:
        return 'Warning'
    else:
        return 'Normal'

df_clean['COOLING_STATUS'] = df_clean.apply(get_final_cooling_status, axis=1)

cooling_status_counts = df_clean['COOLING_STATUS'].value_counts()
print("\n📊 Distribution statut refroidissement:")
for status, count in cooling_status_counts.items():
    pct = count / len(df_clean) * 100
    print(f"   {status}: {count:,} ({pct:.2f}%)")

# =============================================================================
# 7. VISUALIZATIONS
# =============================================================================
print("\n📊 Génération des visualisations...")

fig = plt.figure(figsize=(20, 16))

# 1. Control Chart - Delta T
ax1 = fig.add_subplot(3, 3, 1)
x_range = range(min(1000, len(df_clean)))
ax1.plot(df_clean['DELTA_T_MEAN'].iloc[:1000].values, linewidth=0.5, color='steelblue')
ax1.axhline(delta_t_mean, color='green', linestyle='-', linewidth=2, label=f'Mean ({delta_t_mean:.1f}°C)')
ax1.axhline(UCL, color='red', linestyle='--', linewidth=1.5, label=f'UCL ({UCL:.1f}°C)')
ax1.axhline(LCL, color='red', linestyle='--', linewidth=1.5, label=f'LCL ({LCL:.1f}°C)')
ax1.axhline(UWL, color='orange', linestyle='--', alpha=0.7)
ax1.axhline(LWL, color='orange', linestyle='--', alpha=0.7)
ax1.fill_between(x_range, LWL, UWL, alpha=0.2, color='green')
ax1.set_xlabel('Index temporel')
ax1.set_ylabel('ΔT (°C)')
ax1.set_title('Control Chart - Delta Température')
ax1.legend(loc='upper right')

# 2. Distribution Delta T
ax2 = fig.add_subplot(3, 3, 2)
ax2.hist(df_clean['DELTA_T_MEAN'], bins=50, color='steelblue', edgecolor='white', alpha=0.7)
ax2.axvline(UCL, color='red', linestyle='--', linewidth=2, label='UCL/LCL')
ax2.axvline(LCL, color='red', linestyle='--', linewidth=2)
ax2.axvline(delta_t_mean, color='green', linestyle='-', linewidth=2, label='Mean')
ax2.set_xlabel('ΔT (°C)')
ax2.set_ylabel('Fréquence')
ax2.set_title('Distribution Delta Température')
ax2.legend()

# 3. Efficacité vs Charge
ax3 = fig.add_subplot(3, 3, 3)
scatter = ax3.scatter(df_clean['LOAD_MW'][:2000], df_clean['COOLING_INDEX'][:2000], 
                      c=df_clean['AMBIENT_TEMP'][:2000], cmap='coolwarm', alpha=0.6, s=10)
ax3.axhline(100, color='green', linestyle='--', linewidth=2, label='Référence (100%)')
ax3.axhline(70, color='red', linestyle='--', alpha=0.7, label='Seuil dégradé')
ax3.set_xlabel('Charge (MW)')
ax3.set_ylabel('Cooling Index (%)')
ax3.set_title('Efficacité Refroidissement vs Charge')
plt.colorbar(scatter, ax=ax3, label='Temp Ambiante')
ax3.legend()

# 4. Hot vs Cold Air
ax4 = fig.add_subplot(3, 3, 4)
ax4.scatter(df_clean['COLD_AIR_MEAN'][:2000], df_clean['HOT_AIR_MEAN'][:2000], 
            c=df_clean['LOAD_MW'][:2000], cmap='viridis', alpha=0.6, s=10)
ax4.plot([30, 70], [30, 70], 'r--', linewidth=2, label='y=x (pas de refroidissement)')
ax4.set_xlabel('Température Air Froid (°C)')
ax4.set_ylabel('Température Air Chaud (°C)')
ax4.set_title('Hot Air vs Cold Air')
plt.colorbar(ax4.collections[0], ax=ax4, label='Charge (MW)')
ax4.legend()

# 5. Distribution Efficacité
ax5 = fig.add_subplot(3, 3, 5)
colors_eff = {'Dégradé': 'red', 'Modéré': 'orange', 'Normal': 'yellow', 
              'Bon': 'lightgreen', 'Excellent': 'green'}
eff_pct = eff_counts / len(df_clean) * 100
bars = ax5.bar(eff_pct.index, eff_pct.values, 
               color=[colors_eff.get(x, 'gray') for x in eff_pct.index])
ax5.set_xlabel('Classe Efficacité')
ax5.set_ylabel('Pourcentage (%)')
ax5.set_title('Distribution Classes Efficacité')
ax5.tick_params(axis='x', rotation=45)

# 6. Distribution SPC
ax6 = fig.add_subplot(3, 3, 6)
colors_spc = {'Normal': 'green', 'Warning': 'orange', 'Out of Control': 'red'}
spc_pct = spc_counts / len(df_clean) * 100
bars = ax6.bar(spc_pct.index, spc_pct.values, 
               color=[colors_spc.get(x, 'gray') for x in spc_pct.index])
ax6.set_xlabel('Statut SPC')
ax6.set_ylabel('Pourcentage (%)')
ax6.set_title('Statistical Process Control')

# 7. Delta T vs Charge avec régression
ax7 = fig.add_subplot(3, 3, 7)
ax7.scatter(df_clean['LOAD_MW'][:2000], df_clean['DELTA_T_MEAN'][:2000], 
            alpha=0.4, s=10, color='steelblue', label='Données')
# Ligne de régression
load_range = np.linspace(df_clean['LOAD_MW'].min(), df_clean['LOAD_MW'].max(), 100)
ambient_mean = df_clean['AMBIENT_TEMP'].mean()
delta_pred = lr_model.predict(np.column_stack([load_range, np.full(100, ambient_mean)]))
ax7.plot(load_range, delta_pred, 'r-', linewidth=2, label='Modèle attendu')
ax7.set_xlabel('Charge (MW)')
ax7.set_ylabel('ΔT (°C)')
ax7.set_title('Delta T vs Charge')
ax7.legend()

# 8. Évolution temporelle des statuts
ax8 = fig.add_subplot(3, 3, 8)
status_map = {'Normal': 0, 'Warning': 1, 'Critical': 2}
status_numeric = df_clean['COOLING_STATUS'].map(status_map)
ax8.scatter(range(min(2000, len(df_clean))), status_numeric.iloc[:2000], 
            c=status_numeric.iloc[:2000], cmap='RdYlGn_r', alpha=0.6, s=5)
ax8.set_yticks([0, 1, 2])
ax8.set_yticklabels(['Normal', 'Warning', 'Critical'])
ax8.set_xlabel('Index temporel')
ax8.set_ylabel('Statut')
ax8.set_title('Évolution du Statut Refroidissement')

# 9. Statut Final
ax9 = fig.add_subplot(3, 3, 9)
colors_final = {'Normal': 'green', 'Warning': 'orange', 'Critical': 'red'}
final_pct = cooling_status_counts / len(df_clean) * 100
bars = ax9.pie(final_pct.values, labels=final_pct.index, autopct='%1.1f%%',
               colors=[colors_final.get(x, 'gray') for x in final_pct.index])
ax9.set_title('Statut Final Refroidissement')

plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}/02_cooling_efficiency.png', dpi=150, bbox_inches='tight')
plt.close()

print(f"   ✅ Sauvegardé: {PLOTS_DIR}/02_cooling_efficiency.png")

# =============================================================================
# 8. SAUVEGARDE DES MODÈLES
# =============================================================================
print("\n💾 Sauvegarde des modèles...")

joblib.dump(iso_forest, f'{PLOTS_DIR}/02_cooling_iso_forest.pkl')
joblib.dump(lr_model, f'{PLOTS_DIR}/02_cooling_lr_model.pkl')
joblib.dump(scaler, f'{PLOTS_DIR}/02_cooling_scaler.pkl')

cooling_config = {
    'control_limits': {
        'mean': delta_t_mean,
        'std': delta_t_std,
        'UCL': UCL,
        'LCL': LCL,
        'UWL': UWL,
        'LWL': LWL
    },
    'features': cooling_features,
    'efficiency_thresholds': {
        'degraded': 70,
        'moderate': 90,
        'normal': 110,
        'good': 130
    }
}
joblib.dump(cooling_config, f'{PLOTS_DIR}/02_cooling_config.pkl')

print(f"   ✅ Isolation Forest: {PLOTS_DIR}/02_cooling_iso_forest.pkl")
print(f"   ✅ Config: {PLOTS_DIR}/02_cooling_config.pkl")

# =============================================================================
# RÉSUMÉ
# =============================================================================
print("\n" + "=" * 70)
print("📊 RÉSUMÉ - COOLING EFFICIENCY MODEL")
print("=" * 70)
print(f"""
📉 Statistical Process Control:
   • Mean ΔT: {delta_t_mean:.2f}°C
   • UCL: {UCL:.2f}°C | LCL: {LCL:.2f}°C
   • Normal: {spc_counts.get('Normal', 0):,} ({spc_counts.get('Normal', 0)/len(df_clean)*100:.1f}%)
   • Warning: {spc_counts.get('Warning', 0):,}
   • Out of Control: {spc_counts.get('Out of Control', 0):,}

🎯 Cooling Efficiency Index:
   • Index moyen: {df_clean['COOLING_INDEX'].mean():.1f}%
   • Index min: {df_clean['COOLING_INDEX'].min():.1f}%
   • Index max: {df_clean['COOLING_INDEX'].max():.1f}%

🔴 Isolation Forest:
   • Anomalies détectées: {iso_counts.get('Anomaly', 0):,} ({iso_counts.get('Anomaly', 0)/len(df_clean)*100:.1f}%)

📊 Statut Final:
   • Normal: {cooling_status_counts.get('Normal', 0):,} ({cooling_status_counts.get('Normal', 0)/len(df_clean)*100:.1f}%)
   • Warning: {cooling_status_counts.get('Warning', 0):,} ({cooling_status_counts.get('Warning', 0)/len(df_clean)*100:.1f}%)
   • Critical: {cooling_status_counts.get('Critical', 0):,} ({cooling_status_counts.get('Critical', 0)/len(df_clean)*100:.1f}%)
""")
print("=" * 70)
