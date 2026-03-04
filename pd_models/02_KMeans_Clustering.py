#!/usr/bin/env python3
"""
🔵 02 - KMeans Clustering - Segmentation des Décharges Partielles
==================================================================
Identifier différents comportements de décharge par clustering.

Pipeline:
1. Chargement des features PD
2. Normalisation (StandardScaler)
3. Réduction de dimension (PCA)
4. Méthode du coude pour choisir K
5. KMeans clustering
6. Interprétation des clusters

Clusters attendus:
- Cluster 0: Faible activité (normal)
- Cluster 1: Activité modérée
- Cluster 2: Forte activité
- Cluster 3: Comportement instable
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
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
print("🔵 KMEANS CLUSTERING - Segmentation PD")
print("=" * 80)

# ============================================================================
# 1. CHARGEMENT DES DONNÉES
# ============================================================================
print("\n[1/6] Chargement des données PD enrichies...")

# Essayer de charger le fichier enrichi, sinon le fichier original
feature_file = DATA_DIR / "TG1_Sousse_PD_Features.csv"
if feature_file.exists():
    df = pd.read_csv(feature_file)
    print(f"   ✓ Dataset enrichi: {df.shape[0]:,} × {df.shape[1]}")
else:
    print("   ⚠️ Fichier enrichi non trouvé. Exécutez d'abord 01_PD_Feature_Engineering.py")
    exit(1)

# ============================================================================
# 2. SÉLECTION DES FEATURES POUR CLUSTERING
# ============================================================================
print("\n[2/6] Sélection des features...")

# Features principales pour le clustering
CLUSTER_FEATURES = [
    'PD_INTENSITY_TOTAL',
    'PD_ENERGY_TOTAL',
    'INTENSITY_ASYMMETRY',
    'ENERGY_ASYMMETRY',
    'CURRENT_TOTAL',
    'PULSE_TOTAL',
    'PD_INTENSITY_ROLL_STD_30min',
    'INTENSITY_CV'
]

# Filtrer les colonnes disponibles
available_features = [f for f in CLUSTER_FEATURES if f in df.columns]
print(f"   ✓ Features disponibles: {len(available_features)}")
for f in available_features:
    print(f"      - {f}")

# Préparer les données
df_cluster = df[available_features].copy()
df_cluster = df_cluster.replace([np.inf, -np.inf], np.nan)
df_cluster = df_cluster.dropna()

print(f"   ✓ Données après nettoyage: {len(df_cluster):,}")

# Échantillonnage si trop de données
MAX_SAMPLES = 100000
if len(df_cluster) > MAX_SAMPLES:
    df_sample = df_cluster.sample(n=MAX_SAMPLES, random_state=42)
    print(f"   ✓ Échantillon: {MAX_SAMPLES:,}")
else:
    df_sample = df_cluster

X = df_sample.values

# ============================================================================
# 3. NORMALISATION
# ============================================================================
print("\n[3/6] Normalisation...")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("   ✓ StandardScaler appliqué")

# Sauvegarder le scaler
joblib.dump(scaler, PLOTS_DIR / "02_kmeans_scaler.pkl")

# ============================================================================
# 4. RÉDUCTION DE DIMENSION (PCA)
# ============================================================================
print("\n[4/6] Réduction de dimension (PCA)...")

# PCA pour visualisation (2D)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# PCA avec plus de composantes pour le clustering
pca_full = PCA(n_components=min(5, len(available_features)))
X_pca_full = pca_full.fit_transform(X_scaled)

explained_var = pca.explained_variance_ratio_.sum() * 100
print(f"   ✓ Variance expliquée (2D): {explained_var:.1f}%")
print(f"   ✓ Variance expliquée (5D): {pca_full.explained_variance_ratio_.sum()*100:.1f}%")

# Sauvegarder PCA
joblib.dump(pca, PLOTS_DIR / "02_kmeans_pca.pkl")

# ============================================================================
# 5. MÉTHODE DU COUDE + KMEANS
# ============================================================================
print("\n[5/6] Clustering KMeans...")

# Méthode du coude
K_range = range(2, 10)
inertias = []
silhouettes = []

print("   → Recherche du K optimal...")
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)
    sil = silhouette_score(X_scaled, kmeans.labels_)
    silhouettes.append(sil)
    print(f"      K={k}: Inertia={kmeans.inertia_:.0f}, Silhouette={sil:.3f}")

# Choisir K optimal (basé sur silhouette)
optimal_k = K_range[np.argmax(silhouettes)]
print(f"\n   ✓ K optimal (silhouette): {optimal_k}")

# Entraînement final
kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
clusters = kmeans_final.fit_predict(X_scaled)

print(f"   ✓ KMeans entraîné avec K={optimal_k}")

# Ajouter les clusters aux données
df_sample['Cluster'] = clusters

# ============================================================================
# 6. INTERPRÉTATION DES CLUSTERS
# ============================================================================
print("\n[6/6] Interprétation des clusters...")

# Statistiques par cluster
cluster_stats = df_sample.groupby('Cluster')[available_features].agg(['mean', 'std'])
print("\n   📊 PROFIL DES CLUSTERS:")
print("   " + "-" * 70)

# Calculer les moyennes normalisées pour chaque cluster
cluster_means = df_sample.groupby('Cluster')[available_features].mean()

# Classer les clusters par intensité
cluster_ranking = cluster_means['PD_INTENSITY_TOTAL'].sort_values()

cluster_labels = {}
labels_desc = ['Faible activité (Normal)', 'Activité modérée', 'Forte activité', 'Comportement instable']

for i, (cluster_id, _) in enumerate(cluster_ranking.items()):
    if i < len(labels_desc):
        cluster_labels[cluster_id] = labels_desc[i]
    else:
        cluster_labels[cluster_id] = f'Cluster {cluster_id}'

print(f"\n   {'Cluster':<10} {'Label':<30} {'Count':>10} {'%':>8}")
print("   " + "-" * 60)
for cluster_id in range(optimal_k):
    count = (clusters == cluster_id).sum()
    pct = 100 * count / len(clusters)
    label = cluster_labels.get(cluster_id, f'Cluster {cluster_id}')
    print(f"   {cluster_id:<10} {label:<30} {count:>10,} {pct:>7.1f}%")

# Caractéristiques moyennes par cluster
print("\n   📊 CARACTÉRISTIQUES MOYENNES PAR CLUSTER:")
print("   " + "-" * 70)
for cluster_id in range(optimal_k):
    print(f"\n   Cluster {cluster_id}: {cluster_labels.get(cluster_id, '')}")
    cluster_data = df_sample[df_sample['Cluster'] == cluster_id]
    for feat in available_features[:4]:  # Top 4 features
        mean_val = cluster_data[feat].mean()
        print(f"      {feat}: {mean_val:.2f}")

# ============================================================================
# 7. VISUALISATIONS
# ============================================================================
print("\n   Création des visualisations...")

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 1. Méthode du coude
ax1 = axes[0, 0]
ax1.plot(K_range, inertias, 'bo-', linewidth=2, markersize=8)
ax1.axvline(optimal_k, color='red', linestyle='--', label=f'K optimal = {optimal_k}')
ax1.set_xlabel('Nombre de clusters (K)')
ax1.set_ylabel('Inertie')
ax1.set_title('Méthode du Coude')
ax1.legend()
ax1.grid(True)

# 2. Silhouette Score
ax2 = axes[0, 1]
ax2.plot(K_range, silhouettes, 'go-', linewidth=2, markersize=8)
ax2.axvline(optimal_k, color='red', linestyle='--', label=f'K optimal = {optimal_k}')
ax2.set_xlabel('Nombre de clusters (K)')
ax2.set_ylabel('Silhouette Score')
ax2.set_title('Score de Silhouette')
ax2.legend()
ax2.grid(True)

# 3. Clusters en 2D (PCA)
ax3 = axes[0, 2]
scatter = ax3.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', 
                      alpha=0.5, s=10)
plt.colorbar(scatter, ax=ax3, label='Cluster')
ax3.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
ax3.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
ax3.set_title('Clusters (Projection PCA)')

# 4. Distribution des clusters
ax4 = axes[1, 0]
cluster_counts = pd.Series(clusters).value_counts().sort_index()
colors = plt.cm.viridis(np.linspace(0, 1, len(cluster_counts)))
bars = ax4.bar(cluster_counts.index, cluster_counts.values, color=colors)
ax4.set_xlabel('Cluster')
ax4.set_ylabel('Nombre d\'observations')
ax4.set_title('Distribution des Clusters')
for bar, count in zip(bars, cluster_counts.values):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100, 
             f'{count:,}', ha='center', fontsize=9)

# 5. Intensité par cluster
ax5 = axes[1, 1]
if 'PD_INTENSITY_TOTAL' in df_sample.columns:
    boxdata = [df_sample[df_sample['Cluster'] == c]['PD_INTENSITY_TOTAL'].values 
               for c in range(optimal_k)]
    bp = ax5.boxplot(boxdata, labels=[f'C{c}' for c in range(optimal_k)])
    ax5.set_xlabel('Cluster')
    ax5.set_ylabel('PD Intensity Total')
    ax5.set_title('Distribution Intensité par Cluster')
    ax5.set_yscale('log')

# 6. Heatmap des caractéristiques
ax6 = axes[1, 2]
# Normaliser les moyennes pour la heatmap
cluster_means_norm = (cluster_means - cluster_means.min()) / (cluster_means.max() - cluster_means.min() + 1e-10)
sns.heatmap(cluster_means_norm.T, annot=True, fmt='.2f', cmap='YlOrRd', ax=ax6,
            xticklabels=[f'C{c}' for c in range(optimal_k)])
ax6.set_xlabel('Cluster')
ax6.set_ylabel('Feature')
ax6.set_title('Profil des Clusters (Normalisé)')

plt.tight_layout()
plt.savefig(PLOTS_DIR / "02_kmeans_clustering.png", dpi=150, bbox_inches='tight')
plt.close()
print("   ✓ 02_kmeans_clustering.png")

# Sauvegarder le modèle et les résultats
joblib.dump(kmeans_final, PLOTS_DIR / "02_kmeans_model.pkl")
print("   ✓ 02_kmeans_model.pkl")

# Sauvegarder les métriques
metrics = {
    'optimal_k': optimal_k,
    'silhouette_score': silhouettes[optimal_k - 2],
    'inertia': inertias[optimal_k - 2],
    'n_samples': len(df_sample),
    'features': available_features,
    'cluster_labels': cluster_labels
}
pd.DataFrame([metrics]).to_csv(PLOTS_DIR / "02_kmeans_metrics.csv", index=False)
joblib.dump(cluster_labels, PLOTS_DIR / "02_cluster_labels.pkl")

print("\n" + "=" * 80)
print("✅ KMEANS CLUSTERING - TERMINÉ")
print("=" * 80)
print(f"""
📊 Résultats:
   • K optimal: {optimal_k}
   • Silhouette Score: {silhouettes[optimal_k - 2]:.3f}
   • Clusters identifiés:
""")
for cluster_id in range(optimal_k):
    count = (clusters == cluster_id).sum()
    pct = 100 * count / len(clusters)
    label = cluster_labels.get(cluster_id, '')
    print(f"      - Cluster {cluster_id}: {label} ({pct:.1f}%)")

print(f"""
📁 Fichiers générés:
   • {PLOTS_DIR / '02_kmeans_clustering.png'}
   • {PLOTS_DIR / '02_kmeans_model.pkl'}
   • {PLOTS_DIR / '02_kmeans_scaler.pkl'}
   • {PLOTS_DIR / '02_kmeans_pca.pkl'}
""")
