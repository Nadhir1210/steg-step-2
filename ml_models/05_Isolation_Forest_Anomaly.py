#!/usr/bin/env python3
"""
🔴 Isolation Forest - Détection d'Anomalies
=============================================
Détecte les dérives thermiques anormales de l'alternateur.
Méthode non-supervisée, excellente pour la maintenance prédictive.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
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
print("🔴 Isolation Forest - Détection d'Anomalies")
print("=" * 80)

# ============================================================================
# 1. CHARGEMENT DES DONNÉES
# ============================================================================
print("\n[1/5] Chargement des données...")
df = pd.read_csv(DATA_DIR / "APM_Alternateur_ML.csv")
print(f"   ✓ Dataset: {df.shape[0]:,} × {df.shape[1]}")

# ============================================================================
# 2. PRÉPARATION
# ============================================================================
print("\n[2/5] Préparation des données...")

# Features pour détection d'anomalies (focus sur comportement thermique)
FEATURES = [
    'TEMP_STATOR_MEAN_degC',
    'TEMP_STATOR_MAX_degC',
    'TEMP_PHASE_IMBALANCE_degC',
    'TEMP_HYDROGENE_degC',
    'PUISSANCE_MW',
    'PUISSANCE_REACTIVE_MVAR',
    'COURANT_A', 'COURANT_B', 'COURANT_C',
    'TENSION_AB_V', 'TENSION_BC_V', 'TENSION_CA_V',
    'CURRENT_IMBALANCE_pct'
]

# Filtrer colonnes disponibles
available_features = [f for f in FEATURES if f in df.columns]
print(f"   ✓ Features: {len(available_features)}")

# Nettoyer
df_clean = df[available_features].dropna()
print(f"   ✓ Données après nettoyage: {df_clean.shape[0]:,}")

# Échantillonnage (pour rapidité)
sample_size = min(200000, len(df_clean))
df_sample = df_clean.sample(n=sample_size, random_state=42)

# Normalisation
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_sample)

# ============================================================================
# 3. ISOLATION FOREST
# ============================================================================
print("\n[3/5] Entraînement Isolation Forest...")

# Contamination = proportion estimée d'anomalies (2-5% typique)
CONTAMINATION = 0.03  # 3% d'anomalies attendues

model = IsolationForest(
    n_estimators=200,
    contamination=CONTAMINATION,
    max_samples='auto',
    random_state=42,
    n_jobs=-1
)

# Fit et prédiction
# -1 = anomalie, 1 = normal
predictions = model.fit_predict(X_scaled)
anomaly_scores = model.decision_function(X_scaled)

# Ajouter les résultats
df_results = df_sample.copy()
df_results['anomaly'] = predictions
df_results['anomaly_score'] = anomaly_scores

# Statistiques
n_anomalies = (predictions == -1).sum()
n_normal = (predictions == 1).sum()

print(f"   ✓ Données normales: {n_normal:,} ({100*n_normal/len(predictions):.1f}%)")
print(f"   ✓ Anomalies: {n_anomalies:,} ({100*n_anomalies/len(predictions):.1f}%)")

# ============================================================================
# 4. ANALYSE DES ANOMALIES
# ============================================================================
print("\n[4/5] Analyse des anomalies...")

# Statistiques comparatives
normal_data = df_results[df_results['anomaly'] == 1]
anomaly_data = df_results[df_results['anomaly'] == -1]

print("\n   📊 Comparaison Normal vs Anomalies:")
print("   " + "-" * 60)
print(f"   {'Variable':<30} {'Normal (moy)':<15} {'Anomalie (moy)':<15}")
print("   " + "-" * 60)

for col in available_features[:6]:  # Top 6 features
    normal_mean = normal_data[col].mean()
    anomaly_mean = anomaly_data[col].mean()
    diff_pct = 100 * (anomaly_mean - normal_mean) / normal_mean if normal_mean != 0 else 0
    print(f"   {col:<30} {normal_mean:<15.2f} {anomaly_mean:<15.2f} ({diff_pct:+.1f}%)")

# ============================================================================
# 5. VISUALISATIONS
# ============================================================================
print("\n[5/5] Visualisations...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Distribution des scores d'anomalie
ax1 = axes[0, 0]
ax1.hist(anomaly_scores, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
threshold = np.percentile(anomaly_scores, CONTAMINATION * 100)
ax1.axvline(threshold, color='red', linestyle='--', lw=2, label=f'Seuil ({CONTAMINATION*100:.0f}%)')
ax1.set_xlabel('Score d\'Anomalie')
ax1.set_ylabel('Fréquence')
ax1.set_title('Distribution des Scores (Isolation Forest)')
ax1.legend()

# 2. PCA Visualisation
ax2 = axes[0, 1]
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

normal_mask = predictions == 1
anomaly_mask = predictions == -1

ax2.scatter(X_pca[normal_mask, 0], X_pca[normal_mask, 1], 
            c='green', alpha=0.3, s=5, label='Normal')
ax2.scatter(X_pca[anomaly_mask, 0], X_pca[anomaly_mask, 1], 
            c='red', alpha=0.8, s=20, label='Anomalie')
ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
ax2.set_title('Projection PCA - Anomalies en Rouge')
ax2.legend()

# 3. Température vs Puissance
ax3 = axes[1, 0]
if 'TEMP_STATOR_MEAN_degC' in df_results.columns and 'PUISSANCE_MW' in df_results.columns:
    ax3.scatter(df_results.loc[normal_mask, 'PUISSANCE_MW'], 
                df_results.loc[normal_mask, 'TEMP_STATOR_MEAN_degC'],
                c='green', alpha=0.3, s=5, label='Normal')
    ax3.scatter(df_results.loc[anomaly_mask, 'PUISSANCE_MW'],
                df_results.loc[anomaly_mask, 'TEMP_STATOR_MEAN_degC'],
                c='red', alpha=0.8, s=20, label='Anomalie')
    ax3.set_xlabel('Puissance (MW)')
    ax3.set_ylabel('Température Stator (°C)')
    ax3.set_title('Dérives Thermiques Anormales')
    ax3.legend()

# 4. Proportion par feature
ax4 = axes[1, 1]
feature_importance = []
for col in available_features:
    if col in anomaly_data.columns:
        normal_std = normal_data[col].std()
        anomaly_std = anomaly_data[col].std()
        ratio = anomaly_std / normal_std if normal_std > 0 else 1
        feature_importance.append({'feature': col, 'variability_ratio': ratio})

fi_df = pd.DataFrame(feature_importance).sort_values('variability_ratio', ascending=True)
ax4.barh(fi_df['feature'], fi_df['variability_ratio'], color='coral')
ax4.axvline(1.0, color='black', linestyle='--', alpha=0.5)
ax4.set_xlabel('Ratio de Variabilité (Anomalie/Normal)')
ax4.set_title('Features Discriminantes')

plt.tight_layout()
plt.savefig(PLOTS_DIR / "05_isolation_forest_results.png", dpi=150, bbox_inches='tight')
plt.close()
print("   ✓ 05_isolation_forest_results.png")

# Exporter les anomalies
anomaly_export = df_results[df_results['anomaly'] == -1].copy()
anomaly_export.to_csv(PLOTS_DIR / "05_anomalies_detected.csv", index=False)
print(f"   ✓ 05_anomalies_detected.csv ({len(anomaly_export):,} anomalies)")
# Sauvegarder le modèle en PKL
joblib.dump(model, PLOTS_DIR / "05_isolation_forest_model.pkl")
joblib.dump(scaler, PLOTS_DIR / "05_isolation_forest_scaler.pkl")
print("   \u2713 05_isolation_forest_model.pkl")
# Métriques
metrics = {
    'model': 'Isolation Forest',
    'contamination': CONTAMINATION,
    'n_samples': len(df_sample),
    'n_anomalies': n_anomalies,
    'anomaly_rate': n_anomalies / len(predictions),
    'n_features': len(available_features)
}
pd.DataFrame([metrics]).to_csv(PLOTS_DIR / "05_isolation_forest_metrics.csv", index=False)

print("\n" + "=" * 80)
print("✅ Isolation Forest - TERMINÉ")
print(f"   Anomalies détectées: {n_anomalies:,} ({100*n_anomalies/len(predictions):.1f}%)")
print("=" * 80)
