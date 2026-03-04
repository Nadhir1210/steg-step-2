#!/usr/bin/env python3
"""
⭐ Health Index (HI) - Indice de Santé Alternateur
==================================================
Calcul d'un indice composite de santé basé sur:
- Erreur de prédiction (XGBoost/RF)
- Asymétrie entre phases
- Ratio Température/Puissance
- Score d'anomalies

HI = f(Erreur prédiction, Asymétrie phases, Ratio Temp/Puissance, Score anomalies)

Excellent pour soutenance - synthèse de tous les modèles!
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import IsolationForest, RandomForestRegressor
import joblib
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

# Configuration
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "LAST_DATA"
PLOTS_DIR = Path(__file__).parent / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

print("=" * 80)
print("⭐ HEALTH INDEX - Indice de Santé Alternateur")
print("=" * 80)

# ============================================================================
# 1. CHARGEMENT DES DONNÉES
# ============================================================================
print("\n[1/6] Chargement des données...")
df = pd.read_csv(DATA_DIR / "APM_Alternateur_ML.csv")
print(f"   ✓ Dataset: {df.shape[0]:,} × {df.shape[1]}")

# Nettoyer
df_clean = df.select_dtypes(include=[np.number]).dropna()

# Échantillonnage
sample_size = min(100000, len(df_clean))
df_sample = df_clean.sample(n=sample_size, random_state=42).copy()
print(f"   ✓ Échantillon: {len(df_sample):,}")

# ============================================================================
# 2. COMPOSANTE 1: ERREUR DE PRÉDICTION
# ============================================================================
print("\n[2/6] Calcul de l'erreur de prédiction...")

TARGET = 'TEMP_STATOR_MEAN_degC'
EXCLUDE_COLS = [
    'Datetime', 'Year', 'Month', 'Day', 'Hour', 'Minute', 'DayOfWeek', 'Quarter',
    TARGET, 'TEMP_STATOR_MAX_degC', 'TEMP_PHASE_IMBALANCE_degC',
    'TEMP_PHASE_A_MEAN_degC', 'TEMP_PHASE_B_MEAN_degC', 'TEMP_PHASE_C_MEAN_degC'
]

feature_cols = [col for col in df_sample.columns if col not in EXCLUDE_COLS and col in df_sample.select_dtypes(include=[np.number]).columns]

X = df_sample[feature_cols].values
y = df_sample[TARGET].values

# Entraîner un modèle rapide (Random Forest)
rf = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
rf.fit(X, y)

# Prédiction et erreur
y_pred = rf.predict(X)
prediction_error = np.abs(y - y_pred)

# Normaliser 0-1 (plus l'erreur est grande, plus le score est mauvais)
scaler = MinMaxScaler()
prediction_score = scaler.fit_transform(prediction_error.reshape(-1, 1)).flatten()

print(f"   ✓ Erreur moyenne: {prediction_error.mean():.2f}°C")
print(f"   ✓ Score prédiction (moy): {prediction_score.mean():.3f}")

# ============================================================================
# 3. COMPOSANTE 2: ASYMÉTRIE DES PHASES
# ============================================================================
print("\n[3/6] Calcul de l'asymétrie des phases...")

# Asymétrie de courant (déjà calculée ou à calculer)
if 'CURRENT_IMBALANCE_pct' in df_sample.columns:
    current_imbalance = df_sample['CURRENT_IMBALANCE_pct'].values
else:
    # Calculer manuellement si pas disponible
    if all(c in df_sample.columns for c in ['COURANT_A', 'COURANT_B', 'COURANT_C']):
        courants = df_sample[['COURANT_A', 'COURANT_B', 'COURANT_C']].values
        mean_current = np.mean(courants, axis=1)
        max_dev = np.max(np.abs(courants - mean_current.reshape(-1, 1)), axis=1)
        current_imbalance = 100 * max_dev / np.maximum(mean_current, 1)
    else:
        current_imbalance = np.zeros(len(df_sample))

# Asymétrie thermique
if 'TEMP_PHASE_IMBALANCE_degC' in df_sample.columns:
    temp_imbalance = df_sample['TEMP_PHASE_IMBALANCE_degC'].values
else:
    temp_imbalance = np.zeros(len(df_sample))

# Combinaison pondérée des asymétries
asymmetry_raw = 0.5 * current_imbalance + 0.5 * temp_imbalance
asymmetry_score = scaler.fit_transform(asymmetry_raw.reshape(-1, 1)).flatten()

print(f"   ✓ Asymétrie courant (moy): {current_imbalance.mean():.2f}%")
print(f"   ✓ Asymétrie thermique (moy): {temp_imbalance.mean():.2f}°C")

# ============================================================================
# 4. COMPOSANTE 3: RATIO TEMPÉRATURE/PUISSANCE
# ============================================================================
print("\n[4/6] Calcul du ratio Température/Puissance...")

if 'PUISSANCE_MW' in df_sample.columns:
    power = df_sample['PUISSANCE_MW'].values
    temp = df_sample[TARGET].values
    
    # Ratio: température par MW (éviter division par zéro)
    power_safe = np.maximum(power, 0.1)  # Min 0.1 MW
    temp_power_ratio = temp / power_safe
    
    # Une température élevée par MW = problème potentiel
    ratio_score = scaler.fit_transform(temp_power_ratio.reshape(-1, 1)).flatten()
    print(f"   ✓ Ratio Temp/Puissance (moy): {temp_power_ratio.mean():.3f} °C/MW")
else:
    ratio_score = np.zeros(len(df_sample))
    print("   ⚠️ Puissance non disponible, ratio = 0")

# ============================================================================
# 5. COMPOSANTE 4: SCORE D'ANOMALIES
# ============================================================================
print("\n[5/6] Calcul du score d'anomalies...")

# Features pour Isolation Forest
anomaly_features = [
    TARGET, 'PUISSANCE_MW', 'COURANT_A', 'COURANT_B', 'COURANT_C',
    'TEMP_HYDROGENE_degC', 'CURRENT_IMBALANCE_pct'
]
available_anomaly_features = [f for f in anomaly_features if f in df_sample.columns]

X_anomaly = df_sample[available_anomaly_features].values
X_anomaly_scaled = StandardScaler().fit_transform(X_anomaly)

# Isolation Forest
iso_forest = IsolationForest(n_estimators=100, contamination=0.05, random_state=42, n_jobs=-1)
iso_forest.fit(X_anomaly_scaled)

# Decision function: plus négatif = plus anormal
anomaly_raw_score = -iso_forest.decision_function(X_anomaly_scaled)  # Inverser pour que + = anomalie
anomaly_score = scaler.fit_transform(anomaly_raw_score.reshape(-1, 1)).flatten()

print(f"   ✓ Score anomalie (moy): {anomaly_score.mean():.3f}")

# ============================================================================
# 6. CALCUL DU HEALTH INDEX
# ============================================================================
print("\n[6/6] Calcul du Health Index composite...")

# Pondérations (ajustables selon l'expertise métier)
WEIGHTS = {
    'prediction_error': 0.30,   # 30% - Erreur de prédiction
    'asymmetry': 0.25,          # 25% - Asymétrie des phases
    'temp_power_ratio': 0.20,   # 20% - Ratio température/puissance
    'anomaly_score': 0.25       # 25% - Score d'anomalies
}

# Calcul du HI (0 = parfait, 1 = très mauvais)
health_index_bad = (
    WEIGHTS['prediction_error'] * prediction_score +
    WEIGHTS['asymmetry'] * asymmetry_score +
    WEIGHTS['temp_power_ratio'] * ratio_score +
    WEIGHTS['anomaly_score'] * anomaly_score
)

# Inverser pour avoir HI: 100 = parfait, 0 = critique
health_index = 100 * (1 - health_index_bad)

# Statistiques
print("\n   📊 STATISTIQUES HEALTH INDEX:")
print("   " + "-" * 50)
print(f"   Moyenne:     {health_index.mean():.1f}/100")
print(f"   Médiane:     {np.median(health_index):.1f}/100")
print(f"   Min:         {health_index.min():.1f}/100")
print(f"   Max:         {health_index.max():.1f}/100")
print(f"   Écart-type:  {health_index.std():.1f}")
print("   " + "-" * 50)

# Classification
def classify_health(hi):
    if hi >= 80: return 'Excellent'
    elif hi >= 60: return 'Bon'
    elif hi >= 40: return 'Attention'
    elif hi >= 20: return 'Critique'
    else: return 'Défaillance'

health_classes = np.array([classify_health(h) for h in health_index])
class_counts = pd.Series(health_classes).value_counts()

print("\n   📊 DISTRIBUTION SANTÉ:")
for cls, count in class_counts.items():
    pct = 100 * count / len(health_index)
    bar = '█' * int(pct / 2)
    print(f"   {cls:<12} {count:>6,} ({pct:>5.1f}%) {bar}")

# Ajouter au DataFrame
df_sample['health_index'] = health_index
df_sample['health_class'] = health_classes
df_sample['prediction_score'] = prediction_score
df_sample['asymmetry_score'] = asymmetry_score
df_sample['ratio_score'] = ratio_score
df_sample['anomaly_score'] = anomaly_score

# ============================================================================
# 7. VISUALISATIONS
# ============================================================================
print("\n   Création des visualisations...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Distribution du Health Index
ax1 = axes[0, 0]
colors = ['green' if h >= 80 else 'limegreen' if h >= 60 else 'orange' if h >= 40 else 'red' if h >= 20 else 'darkred' 
          for h in health_index]
ax1.hist(health_index, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
ax1.axvline(80, color='green', linestyle='--', lw=2, label='Excellent (≥80)')
ax1.axvline(60, color='orange', linestyle='--', lw=2, label='Attention (<60)')
ax1.axvline(40, color='red', linestyle='--', lw=2, label='Critique (<40)')
ax1.set_xlabel('Health Index (0-100)')
ax1.set_ylabel('Fréquence')
ax1.set_title(f'Distribution du Health Index\nMoyenne: {health_index.mean():.1f}/100')
ax1.legend()

# 2. Contribution des composantes
ax2 = axes[0, 1]
components = ['Erreur\nPrédiction', 'Asymétrie\nPhases', 'Ratio\nTemp/Puiss.', 'Score\nAnomalies']
means = [prediction_score.mean(), asymmetry_score.mean(), ratio_score.mean(), anomaly_score.mean()]
weights_list = list(WEIGHTS.values())
weighted_contrib = [m * w for m, w in zip(means, weights_list)]

bars = ax2.bar(components, weighted_contrib, color=['#3498db', '#e74c3c', '#2ecc71', '#9b59b6'])
ax2.set_ylabel('Contribution au Score de Dégradation')
ax2.set_title('Contribution de chaque Composante')
for bar, val in zip(bars, weighted_contrib):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
             f'{val:.3f}', ha='center', fontsize=10)

# 3. Health Index vs Puissance
ax3 = axes[1, 0]
if 'PUISSANCE_MW' in df_sample.columns:
    sample_plot = df_sample.sample(min(5000, len(df_sample)))
    scatter = ax3.scatter(sample_plot['PUISSANCE_MW'], sample_plot['health_index'],
                          c=sample_plot['health_index'], cmap='RdYlGn', 
                          alpha=0.5, s=5, vmin=0, vmax=100)
    plt.colorbar(scatter, ax=ax3, label='Health Index')
    ax3.set_xlabel('Puissance (MW)')
    ax3.set_ylabel('Health Index')
    ax3.set_title('Health Index vs Puissance')
    ax3.axhline(60, color='orange', linestyle='--', alpha=0.7)
    ax3.axhline(40, color='red', linestyle='--', alpha=0.7)

# 4. Pie Chart des classes
ax4 = axes[1, 1]
colors_pie = {'Excellent': '#27ae60', 'Bon': '#2ecc71', 'Attention': '#f1c40f', 
              'Critique': '#e74c3c', 'Défaillance': '#c0392b'}
ax4.pie(class_counts.values, labels=class_counts.index, autopct='%1.1f%%',
        colors=[colors_pie.get(c, 'gray') for c in class_counts.index],
        startangle=90, explode=[0.05 if c in ['Critique', 'Défaillance'] else 0 for c in class_counts.index])
ax4.set_title('Répartition des États de Santé')

plt.tight_layout()
plt.savefig(PLOTS_DIR / "07_health_index_results.png", dpi=150, bbox_inches='tight')
plt.close()
print("   ✓ 07_health_index_results.png")

# Exporter les données avec HI
export_cols = ['health_index', 'health_class', 'prediction_score', 'asymmetry_score', 
               'ratio_score', 'anomaly_score', TARGET, 'PUISSANCE_MW']
export_cols = [c for c in export_cols if c in df_sample.columns]
df_sample[export_cols].to_csv(PLOTS_DIR / "07_health_index_data.csv", index=False)
print("   ✓ 07_health_index_data.csv")
# Sauvegarder les modèles en PKL
joblib.dump(rf, PLOTS_DIR / "07_health_index_rf_model.pkl")
joblib.dump(iso_forest, PLOTS_DIR / "07_health_index_isoforest_model.pkl")
joblib.dump(scaler, PLOTS_DIR / "07_health_index_scaler.pkl")
print("   \u2713 07_health_index_models.pkl")
# Métriques
metrics = {
    'mean_health_index': health_index.mean(),
    'median_health_index': np.median(health_index),
    'std_health_index': health_index.std(),
    'pct_excellent': (health_index >= 80).mean() * 100,
    'pct_attention': ((health_index >= 40) & (health_index < 60)).mean() * 100,
    'pct_critical': (health_index < 40).mean() * 100,
    'weights': str(WEIGHTS)
}
pd.DataFrame([metrics]).to_csv(PLOTS_DIR / "07_health_index_metrics.csv", index=False)

print("\n" + "=" * 80)
print("⭐ HEALTH INDEX - TERMINÉ")
print(f"   Score moyen: {health_index.mean():.1f}/100")
print(f"   Excellent: {(health_index >= 80).mean()*100:.1f}%")
print(f"   Nécessite attention: {(health_index < 60).mean()*100:.1f}%")
print("=" * 80)
print("\n📝 FORMULE HEALTH INDEX:")
print("   HI = 100 - (0.30×Err_Pred + 0.25×Asymétrie + 0.20×Ratio_T/P + 0.25×Anomalie)")
print("   Où chaque composante est normalisée [0,1]")
print("=" * 80)
