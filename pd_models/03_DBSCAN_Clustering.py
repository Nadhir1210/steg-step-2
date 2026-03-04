#!/usr/bin/env python3
"""
🟣 03 - DBSCAN Clustering - Détection des Événements Extrêmes PD
=================================================================
DBSCAN (Density-Based Spatial Clustering) pour détecter:
- Points aberrants (outliers)
- Événements extrêmes de décharge
- Patterns de défaillance rares

Avantages DBSCAN:
- Pas besoin de spécifier K
- Détecte automatiquement les outliers (noise points)
- Identifie des clusters de forme arbitraire
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score
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
print("🟣 DBSCAN CLUSTERING - Détection des Anomalies PD")
print("=" * 80)

# ============================================================================
# 1. CHARGEMENT DES DONNÉES
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
# 2. SÉLECTION DES FEATURES POUR DÉTECTION D'ANOMALIES
# ============================================================================
print("\n[2/6] Sélection des features...")

# Features sensibles aux anomalies
DBSCAN_FEATURES = [
    'PD_INTENSITY_TOTAL',
    'PD_ENERGY_TOTAL',
    'INTENSITY_ASYMMETRY',
    'ENERGY_ASYMMETRY',
    'PULSE_TOTAL',
    'PD_INTENSITY_ROLL_STD_30min',
    'MAX_CHARGE_TOTAL'
]

# Alternative si les features ne sont pas disponibles
FALLBACK_FEATURES = [
    'CURRENT_TOTAL', 'PULSE_TOTAL', 'PD_INTENSITY_TOTAL', 'PD_ENERGY_TOTAL'
]

available_features = [f for f in DBSCAN_FEATURES if f in df.columns]
if len(available_features) < 3:
    available_features = [f for f in FALLBACK_FEATURES if f in df.columns]

print(f"   ✓ Features sélectionnées: {len(available_features)}")
for f in available_features:
    print(f"      - {f}")

# Préparer les données
df_cluster = df[available_features].copy()
df_cluster = df_cluster.replace([np.inf, -np.inf], np.nan)
df_cluster = df_cluster.dropna()

print(f"   ✓ Données après nettoyage: {len(df_cluster):,}")

# Échantillonnage (DBSCAN est O(n²))
MAX_SAMPLES = 50000
if len(df_cluster) > MAX_SAMPLES:
    df_sample = df_cluster.sample(n=MAX_SAMPLES, random_state=42)
    print(f"   ✓ Échantillon: {MAX_SAMPLES:,}")
else:
    df_sample = df_cluster

X = df_sample.values
original_indices = df_sample.index.tolist()

# ============================================================================
# 3. NORMALISATION ROBUSTE
# ============================================================================
print("\n[3/6] Normalisation robuste...")

# RobustScaler est moins sensible aux outliers
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)
print("   ✓ RobustScaler appliqué")

joblib.dump(scaler, PLOTS_DIR / "03_dbscan_scaler.pkl")

# ============================================================================
# 4. ESTIMATION DES PARAMÈTRES (eps, min_samples)
# ============================================================================
print("\n[4/6] Estimation des paramètres optimaux...")

# Méthode du coude pour eps (k-NN distance)
min_samples = max(5, int(np.log(len(X_scaled))))  # Règle empirique
print(f"   → min_samples = {min_samples}")

# Calculer les distances k-NN
neigh = NearestNeighbors(n_neighbors=min_samples)
neigh.fit(X_scaled)
distances, _ = neigh.kneighbors(X_scaled)

# Distance au k-ème voisin (triée)
k_distances = np.sort(distances[:, -1])

# Trouver le coude automatiquement (différence seconde)
# Chercher où la courbe commence à augmenter rapidement
gradient = np.gradient(k_distances)
acceleration = np.gradient(gradient)
elbow_idx = np.argmax(acceleration > 0.1 * acceleration.max())
eps_optimal = k_distances[elbow_idx]

# Si eps trop petit ou trop grand, utiliser le percentile 90
if eps_optimal < 0.1 or eps_optimal > 5:
    eps_optimal = np.percentile(k_distances, 90)

print(f"   → eps optimal = {eps_optimal:.3f}")

# ============================================================================
# 5. DBSCAN CLUSTERING
# ============================================================================
print("\n[5/6] Clustering DBSCAN...")

dbscan = DBSCAN(eps=eps_optimal, min_samples=min_samples, n_jobs=-1)
clusters = dbscan.fit_predict(X_scaled)

# Statistiques des clusters
n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
n_noise = (clusters == -1).sum()
noise_pct = 100 * n_noise / len(clusters)

print(f"   ✓ Nombre de clusters: {n_clusters}")
print(f"   ✓ Points Noise: {n_noise:,} ({noise_pct:.2f}%)")

df_sample['Cluster'] = clusters

# ============================================================================
# 6. ANALYSE DES ANOMALIES
# ============================================================================
print("\n[6/6] Analyse des anomalies...")

# Les anomalies sont les points de bruit (cluster = -1)
anomalies = df_sample[df_sample['Cluster'] == -1]
normal = df_sample[df_sample['Cluster'] != -1]

print(f"\n   📊 ANALYSE DES ANOMALIES:")
print("   " + "-" * 60)
print(f"   • Points normaux: {len(normal):,} ({100-noise_pct:.1f}%)")
print(f"   • Anomalies: {len(anomalies):,} ({noise_pct:.1f}%)")

# Caractéristiques des anomalies vs normaux
print(f"\n   {'Feature':<35} {'Normal':>12} {'Anomalie':>12} {'Ratio':>8}")
print("   " + "-" * 70)
for feat in available_features:
    normal_mean = normal[feat].mean()
    anomaly_mean = anomalies[feat].mean() if len(anomalies) > 0 else 0
    ratio = anomaly_mean / (normal_mean + 1e-10)
    print(f"   {feat:<35} {normal_mean:>12.2f} {anomaly_mean:>12.2f} {ratio:>7.1f}x")

# Distribution par cluster
print(f"\n   📊 DISTRIBUTION DES CLUSTERS:")
print("   " + "-" * 40)
cluster_counts = pd.Series(clusters).value_counts().sort_index()
for cluster_id, count in cluster_counts.items():
    pct = 100 * count / len(clusters)
    if cluster_id == -1:
        label = "ANOMALIES (noise)"
    else:
        label = f"Cluster {cluster_id}"
    print(f"   {label:<25} {count:>10,} ({pct:>5.1f}%)")

# ============================================================================
# 7. VISUALISATIONS
# ============================================================================
print("\n   Création des visualisations...")

# PCA pour visualisation
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 1. K-distance plot (méthode du coude)
ax1 = axes[0, 0]
ax1.plot(range(len(k_distances)), k_distances, 'b-', linewidth=1)
ax1.axhline(eps_optimal, color='red', linestyle='--', label=f'eps = {eps_optimal:.3f}')
ax1.set_xlabel('Points (triés)')
ax1.set_ylabel(f'Distance au {min_samples}-ème voisin')
ax1.set_title('K-Distance Plot (Méthode du Coude)')
ax1.legend()
ax1.grid(True)

# 2. Clusters en 2D (PCA)
ax2 = axes[0, 1]
scatter = ax2.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', 
                      alpha=0.5, s=10)
# Marquer les anomalies en rouge
anomaly_mask = clusters == -1
if anomaly_mask.any():
    ax2.scatter(X_pca[anomaly_mask, 0], X_pca[anomaly_mask, 1], 
                c='red', alpha=0.7, s=30, marker='x', label='Anomalies')
ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
ax2.set_title('DBSCAN Clusters (PCA)')
ax2.legend()

# 3. Distribution des clusters
ax3 = axes[0, 2]
colors = ['red' if c == -1 else plt.cm.viridis(c / max(max(clusters), 1)) 
          for c in cluster_counts.index]
bars = ax3.bar([str(c) for c in cluster_counts.index], cluster_counts.values, color=colors)
ax3.set_xlabel('Cluster (-1 = Anomalies)')
ax3.set_ylabel('Nombre d\'observations')
ax3.set_title('Distribution des Clusters')
for bar, count in zip(bars, cluster_counts.values):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50, 
             f'{count:,}', ha='center', fontsize=9)

# 4. Intensité: Normal vs Anomalies
ax4 = axes[1, 0]
if 'PD_INTENSITY_TOTAL' in df_sample.columns:
    ax4.boxplot([normal['PD_INTENSITY_TOTAL'].values, 
                 anomalies['PD_INTENSITY_TOTAL'].values if len(anomalies) > 0 else [0]],
                labels=['Normal', 'Anomalies'])
    ax4.set_ylabel('PD Intensity Total')
    ax4.set_title('Distribution Intensité')
    ax4.set_yscale('log')

# 5. Énergie: Normal vs Anomalies
ax5 = axes[1, 1]
if 'PD_ENERGY_TOTAL' in df_sample.columns:
    ax5.boxplot([normal['PD_ENERGY_TOTAL'].values, 
                 anomalies['PD_ENERGY_TOTAL'].values if len(anomalies) > 0 else [0]],
                labels=['Normal', 'Anomalies'])
    ax5.set_ylabel('PD Energy Total')
    ax5.set_title('Distribution Énergie')
    ax5.set_yscale('log')

# 6. Heatmap comparatif Normal vs Anomalies
ax6 = axes[1, 2]
if len(anomalies) > 0:
    comparison = pd.DataFrame({
        'Normal': normal[available_features].mean(),
        'Anomalie': anomalies[available_features].mean()
    }).T
    # Normaliser pour la heatmap
    comparison_norm = (comparison.T - comparison.T.min()) / (comparison.T.max() - comparison.T.min() + 1e-10)
    sns.heatmap(comparison_norm, annot=True, fmt='.2f', cmap='YlOrRd', ax=ax6)
    ax6.set_xlabel('Feature')
    ax6.set_ylabel('')
    ax6.set_title('Comparaison Normal vs Anomalie')
else:
    ax6.text(0.5, 0.5, 'Pas d\'anomalies détectées', ha='center', va='center')
    ax6.set_title('Comparaison')

plt.tight_layout()
plt.savefig(PLOTS_DIR / "03_dbscan_clustering.png", dpi=150, bbox_inches='tight')
plt.close()
print("   ✓ 03_dbscan_clustering.png")

# Sauvegarder le modèle et les résultats
joblib.dump(dbscan, PLOTS_DIR / "03_dbscan_model.pkl")
print("   ✓ 03_dbscan_model.pkl")

# Sauvegarder les anomalies
if len(anomalies) > 0:
    # Retourner aux indices originaux
    anomaly_indices = anomalies.index.tolist()
    pd.DataFrame({'original_index': anomaly_indices}).to_csv(
        PLOTS_DIR / "03_anomalies_indices.csv", index=False
    )

# Métriques
metrics = {
    'eps': eps_optimal,
    'min_samples': min_samples,
    'n_clusters': n_clusters,
    'n_anomalies': n_noise,
    'anomaly_pct': noise_pct,
    'n_samples': len(df_sample),
    'features': available_features
}
pd.DataFrame([metrics]).to_csv(PLOTS_DIR / "03_dbscan_metrics.csv", index=False)
joblib.dump(metrics, PLOTS_DIR / "03_dbscan_params.pkl")

print("\n" + "=" * 80)
print("✅ DBSCAN CLUSTERING - TERMINÉ")
print("=" * 80)
print(f"""
📊 Résultats:
   • Paramètres: eps={eps_optimal:.3f}, min_samples={min_samples}
   • Clusters trouvés: {n_clusters}
   • Anomalies détectées: {n_noise:,} ({noise_pct:.2f}%)

🎯 Interprétation:
   Les points marqués comme "noise" (cluster -1) représentent:
   - Événements de décharge extrêmes
   - Comportements anormaux isolés
   - Potentiels débuts de défaillance

📁 Fichiers générés:
   • {PLOTS_DIR / '03_dbscan_clustering.png'}
   • {PLOTS_DIR / '03_dbscan_model.pkl'}
   • {PLOTS_DIR / '03_dbscan_scaler.pkl'}
   • {PLOTS_DIR / '03_anomalies_indices.csv'}
""")
