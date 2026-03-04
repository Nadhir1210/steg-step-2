#!/usr/bin/env python3
"""
🔵 ANN (Artificial Neural Network) - Deep Learning Regression
===============================================================
Réseau de neurones pour la prédiction de température stator.
Approche académique avec architecture personnalisable.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
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
print("🔵 ANN (Artificial Neural Network) - Deep Learning")
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

TARGET = 'TEMP_STATOR_MEAN_degC'
EXCLUDE_COLS = [
    'Datetime', 'Year', 'Month', 'Day', 'Hour', 'Minute', 'DayOfWeek', 'Quarter',
    TARGET, 'TEMP_STATOR_MAX_degC', 'TEMP_PHASE_IMBALANCE_degC',
    'TEMP_PHASE_A_MEAN_degC', 'TEMP_PHASE_B_MEAN_degC', 'TEMP_PHASE_C_MEAN_degC'
]

feature_cols = [col for col in df.select_dtypes(include=[np.number]).columns 
                if col not in EXCLUDE_COLS]

df_clean = df[feature_cols + [TARGET]].dropna()

# Échantillonnage
sample_size = min(100000, len(df_clean))
df_sample = df_clean.sample(n=sample_size, random_state=42)

X = df_sample[feature_cols].values
y = df_sample[TARGET].values

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Normalisation
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train = scaler_X.fit_transform(X_train)
X_val = scaler_X.transform(X_val)
X_test = scaler_X.transform(X_test)

y_train = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
y_val = scaler_y.transform(y_val.reshape(-1, 1)).flatten()
y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()

print(f"   ✓ Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")
print(f"   ✓ Features: {len(feature_cols)}")

# ============================================================================
# 3. ARCHITECTURE DU RÉSEAU
# ============================================================================
print("\n[3/6] Construction du modèle ANN...")

n_features = X_train.shape[1]

model = keras.Sequential([
    layers.Input(shape=(n_features,)),
    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.2),
    layers.Dense(32, activation='relu'),
    layers.Dense(16, activation='relu'),
    layers.Dense(1)  # Sortie régression
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='mse',
    metrics=['mae']
)

model.summary()

# ============================================================================
# 4. ENTRAÎNEMENT
# ============================================================================
print("\n[4/6] Entraînement...")

early_stop = callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

reduce_lr = callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-6
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=256,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

# ============================================================================
# 5. ÉVALUATION
# ============================================================================
print("\n[5/6] Évaluation...")

# Prédictions (dénormaliser)
y_pred_scaled = model.predict(X_test, verbose=0).flatten()
y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()

# Métriques
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
test_mae = mean_absolute_error(y_test, y_pred)
test_r2 = r2_score(y_test, y_pred)

print("\n   📊 RÉSULTATS ANN:")
print("   " + "-" * 40)
print(f"   RMSE: {test_rmse:.4f} °C")
print(f"   MAE:  {test_mae:.4f} °C")
print(f"   R²:   {test_r2:.4f}")
print("   " + "-" * 40)

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
ax1.set_ylabel('Loss (MSE)')
ax1.set_title('Courbes d\'Apprentissage')
ax1.legend()
ax1.grid(True)

# 2. Prédictions vs Réel
ax2 = axes[0, 1]
ax2.scatter(y_test, y_pred, alpha=0.3, s=5, c='purple')
ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
ax2.set_xlabel('Température Réelle (°C)')
ax2.set_ylabel('Température Prédite (°C)')
ax2.set_title(f'ANN: Prédictions vs Réel\nR² = {test_r2:.4f}, RMSE = {test_rmse:.2f}°C')

# 3. Distribution des erreurs
ax3 = axes[1, 0]
errors = y_test - y_pred
ax3.hist(errors, bins=50, color='purple', edgecolor='black', alpha=0.7)
ax3.axvline(0, color='red', linestyle='--')
ax3.set_xlabel('Erreur (°C)')
ax3.set_ylabel('Fréquence')
ax3.set_title(f'Distribution des Erreurs\nMoy: {errors.mean():.3f}°C, Std: {errors.std():.3f}°C')

# 4. MAE par Epoch
ax4 = axes[1, 1]
ax4.plot(history.history['mae'], label='Train MAE')
ax4.plot(history.history['val_mae'], label='Val MAE')
ax4.set_xlabel('Epoch')
ax4.set_ylabel('MAE')
ax4.set_title('MAE par Epoch')
ax4.legend()
ax4.grid(True)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "03_ann_results.png", dpi=150, bbox_inches='tight')
plt.close()
print("   ✓ 03_ann_results.png")

# Sauvegarder métriques
metrics = {
    'model': 'ANN (Deep Learning)',
    'target': TARGET,
    'test_rmse': test_rmse,
    'test_mae': test_mae,
    'test_r2': test_r2,
    'epochs_trained': len(history.history['loss']),
    'architecture': '128-64-32-16-1'
}
pd.DataFrame([metrics]).to_csv(PLOTS_DIR / "03_ann_metrics.csv", index=False)

# Sauvegarder le modèle
model.save(PLOTS_DIR / "03_ann_model.keras")
print("   ✓ 03_ann_model.keras")
# Sauvegarder les scalers en PKL
joblib.dump(scaler_X, PLOTS_DIR / "03_ann_scaler_X.pkl")
joblib.dump(scaler_y, PLOTS_DIR / "03_ann_scaler_y.pkl")
print("   \u2713 03_ann_scalers.pkl")
print("\n" + "=" * 80)
print("✅ ANN (Deep Learning) - TERMINÉ")
print(f"   R² = {test_r2:.4f} | RMSE = {test_rmse:.2f}°C")
print("=" * 80)
