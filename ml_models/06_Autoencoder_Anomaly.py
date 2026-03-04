#!/usr/bin/env python3
"""
🟠 Autoencoder - Détection d'Anomalies Deep Learning
=====================================================
Détecte les anomalies en mesurant l'erreur de reconstruction.
Très scientifique → excellent pour soutenance!
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

# Essayer d'importer TensorFlow/Keras
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, callbacks
    TF_AVAILABLE = True
    
    # Configuration GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"\u2713 GPU détecté: {len(gpus)} GPU(s) disponible(s)")
        except RuntimeError as e:
            print(f"\u26a0 Erreur config GPU: {e}")
    else:
        print("\u26a0 Pas de GPU détecté - utilisation CPU")
except ImportError:
    TF_AVAILABLE = False
    print("\u26a0 TensorFlow non installé. Installation: pip install tensorflow")

# Configuration
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "LAST_DATA"
PLOTS_DIR = Path(__file__).parent / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

if not TF_AVAILABLE:
    print("Arrêt du script - TensorFlow requis.")
    exit(1)

print("=" * 80)
print("🟠 Autoencoder - Détection d'Anomalies Deep Learning")
print("=" * 80)

# ============================================================================
# 1. CHARGEMENT DES DONNÉES
# ============================================================================
print("\n[1/6] Chargement des données...")
df = pd.read_csv(DATA_DIR / "APM_Alternateur_ML.csv")
print(f"   ✓ Dataset: {df.shape[0]:,} × {df.shape[1]}")

# ============================================================================
# 2. PRÉPARATION
# ============================================================================
print("\n[2/6] Préparation des données...")

# Features pour l'autoencoder
FEATURES = [
    'TEMP_STATOR_MEAN_degC',
    'TEMP_STATOR_MAX_degC',
    'TEMP_PHASE_IMBALANCE_degC',
    'TEMP_HYDROGENE_degC',
    'PUISSANCE_MW',
    'PUISSANCE_REACTIVE_MVAR',
    'COURANT_A', 'COURANT_B', 'COURANT_C',
    'TENSION_AB_V', 'TENSION_BC_V', 'TENSION_CA_V',
    'CURRENT_IMBALANCE_pct',
    'VOLTAGE_IMBALANCE_pct'
]

available_features = [f for f in FEATURES if f in df.columns]
print(f"   ✓ Features: {len(available_features)}")

df_clean = df[available_features].dropna()
sample_size = min(150000, len(df_clean))
df_sample = df_clean.sample(n=sample_size, random_state=42)

# Normalisation
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_sample)

# Split pour validation
X_train, X_test = train_test_split(X_scaled, test_size=0.2, random_state=42)
print(f"   ✓ Train: {len(X_train):,} | Test: {len(X_test):,}")

# ============================================================================
# 3. ARCHITECTURE AUTOENCODER
# ============================================================================
print("\n[3/6] Construction de l'Autoencoder...")

n_features = X_train.shape[1]
encoding_dim = 4  # Dimension de l'espace latent

# Encoder
encoder_input = keras.Input(shape=(n_features,), name='encoder_input')
x = layers.Dense(32, activation='relu')(encoder_input)
x = layers.BatchNormalization()(x)
x = layers.Dense(16, activation='relu')(x)
x = layers.BatchNormalization()(x)
encoded = layers.Dense(encoding_dim, activation='relu', name='encoded')(x)

# Decoder
x = layers.Dense(16, activation='relu')(encoded)
x = layers.BatchNormalization()(x)
x = layers.Dense(32, activation='relu')(x)
x = layers.BatchNormalization()(x)
decoded = layers.Dense(n_features, activation='linear', name='decoded')(x)

# Modèle complet
autoencoder = keras.Model(encoder_input, decoded, name='autoencoder')

# Modèle encoder seul (pour visualisation)
encoder = keras.Model(encoder_input, encoded, name='encoder')

autoencoder.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='mse'
)

autoencoder.summary()

# ============================================================================
# 4. ENTRAÎNEMENT
# ============================================================================
print("\n[4/6] Entraînement...")

early_stop = callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

history = autoencoder.fit(
    X_train, X_train,  # Input = Output (reconstruction)
    validation_data=(X_test, X_test),
    epochs=100,
    batch_size=256,
    callbacks=[early_stop],
    verbose=1
)

# ============================================================================
# 5. DÉTECTION D'ANOMALIES
# ============================================================================
print("\n[5/6] Détection d'anomalies...")

# Reconstruction sur toutes les données
X_all_scaled = scaler.transform(df_sample)
X_reconstructed = autoencoder.predict(X_all_scaled, verbose=0)

# Erreur de reconstruction (MSE par sample)
reconstruction_error = np.mean((X_all_scaled - X_reconstructed) ** 2, axis=1)

# Définir le seuil (percentile 97 = 3% anomalies)
PERCENTILE_THRESHOLD = 97
threshold = np.percentile(reconstruction_error, PERCENTILE_THRESHOLD)

# Classifier
anomaly_mask = reconstruction_error > threshold
n_anomalies = anomaly_mask.sum()
n_normal = (~anomaly_mask).sum()

print(f"   ✓ Seuil (P{PERCENTILE_THRESHOLD}): {threshold:.6f}")
print(f"   ✓ Données normales: {n_normal:,} ({100*n_normal/len(reconstruction_error):.1f}%)")
print(f"   ✓ Anomalies: {n_anomalies:,} ({100*n_anomalies/len(reconstruction_error):.1f}%)")

# Résultats
df_results = df_sample.copy()
df_results['reconstruction_error'] = reconstruction_error
df_results['anomaly'] = anomaly_mask.astype(int)

# Statistiques des anomalies
print("\n   📊 Analyse des Anomalies:")
print("   " + "-" * 60)
normal_data = df_results[~anomaly_mask]
anomaly_data = df_results[anomaly_mask]

for col in available_features[:5]:
    normal_mean = normal_data[col].mean()
    anomaly_mean = anomaly_data[col].mean()
    print(f"   {col:<30} Normal: {normal_mean:.2f} | Anomalie: {anomaly_mean:.2f}")

# ============================================================================
# 6. VISUALISATIONS
# ============================================================================
print("\n[6/6] Visualisations...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Learning Curves
ax1 = axes[0, 0]
ax1.plot(history.history['loss'], label='Train Loss')
ax1.plot(history.history['val_loss'], label='Val Loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Reconstruction Loss (MSE)')
ax1.set_title('Courbes d\'Apprentissage Autoencoder')
ax1.legend()
ax1.grid(True)

# 2. Distribution des erreurs de reconstruction
ax2 = axes[0, 1]
ax2.hist(reconstruction_error[~anomaly_mask], bins=50, alpha=0.7, 
         label='Normal', color='green', edgecolor='black')
ax2.hist(reconstruction_error[anomaly_mask], bins=50, alpha=0.7, 
         label='Anomalie', color='red', edgecolor='black')
ax2.axvline(threshold, color='black', linestyle='--', lw=2, 
            label=f'Seuil (P{PERCENTILE_THRESHOLD})')
ax2.set_xlabel('Erreur de Reconstruction')
ax2.set_ylabel('Fréquence')
ax2.set_title('Distribution des Erreurs')
ax2.legend()

# 3. Espace latent (2D projection)
ax3 = axes[1, 0]
encoded_data = encoder.predict(X_all_scaled, verbose=0)
# Prendre les 2 premières dimensions de l'espace latent
ax3.scatter(encoded_data[~anomaly_mask, 0], encoded_data[~anomaly_mask, 1],
            c='green', alpha=0.3, s=5, label='Normal')
ax3.scatter(encoded_data[anomaly_mask, 0], encoded_data[anomaly_mask, 1],
            c='red', alpha=0.8, s=20, label='Anomalie')
ax3.set_xlabel('Dimension Latente 1')
ax3.set_ylabel('Dimension Latente 2')
ax3.set_title(f'Espace Latent ({encoding_dim}D → 2D)')
ax3.legend()

# 4. Erreur par feature
ax4 = axes[1, 1]
feature_errors = np.mean((X_all_scaled[anomaly_mask] - X_reconstructed[anomaly_mask]) ** 2, axis=0)
feature_df = pd.DataFrame({
    'feature': available_features,
    'error': feature_errors
}).sort_values('error', ascending=True)
ax4.barh(feature_df['feature'], feature_df['error'], color='orange')
ax4.set_xlabel('Erreur de Reconstruction Moyenne')
ax4.set_title('Contribution par Feature (Anomalies)')

plt.tight_layout()
plt.savefig(PLOTS_DIR / "06_autoencoder_results.png", dpi=150, bbox_inches='tight')
plt.close()
print("   ✓ 06_autoencoder_results.png")

# Exporter anomalies
anomaly_export = df_results[df_results['anomaly'] == 1].copy()
anomaly_export.to_csv(PLOTS_DIR / "06_autoencoder_anomalies.csv", index=False)
print(f"   ✓ 06_autoencoder_anomalies.csv ({len(anomaly_export):,} anomalies)")

# Métriques
metrics = {
    'model': 'Autoencoder',
    'encoding_dim': encoding_dim,
    'threshold': threshold,
    'threshold_percentile': PERCENTILE_THRESHOLD,
    'n_anomalies': n_anomalies,
    'anomaly_rate': n_anomalies / len(reconstruction_error),
    'epochs_trained': len(history.history['loss'])
}
pd.DataFrame([metrics]).to_csv(PLOTS_DIR / "06_autoencoder_metrics.csv", index=False)

# Sauvegarder modèles
autoencoder.save(PLOTS_DIR / "06_autoencoder_model.keras")
encoder.save(PLOTS_DIR / "06_encoder_model.keras")
print("   ✓ 06_autoencoder_model.keras")
# Sauvegarder le scaler en PKL
joblib.dump(scaler, PLOTS_DIR / "06_autoencoder_scaler.pkl")
print("   \u2713 06_autoencoder_scaler.pkl")
print("\n" + "=" * 80)
print("✅ Autoencoder - TERMINÉ")
print(f"   Anomalies détectées: {n_anomalies:,} ({100*n_anomalies/len(reconstruction_error):.1f}%)")
print("=" * 80)
