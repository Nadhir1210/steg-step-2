#!/usr/bin/env python3
"""
🟣 LSTM (Long Short-Term Memory) - Prédiction Séries Temporelles
=================================================================
Prédit la température stator à t+10 minutes en utilisant l'historique.
Parfait pour soutenance - approche état-de-l'art pour séries temporelles.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.preprocessing import MinMaxScaler
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
print("🟣 LSTM - Prédiction Température à t+10 min")
print("=" * 80)

# ============================================================================
# 1. CHARGEMENT DES DONNÉES
# ============================================================================
print("\n[1/6] Chargement des données...")

# Utiliser les données 10min pour l'LSTM (plus adapté)
df = pd.read_csv(DATA_DIR / "APM_Alternateur_10min_ML.csv")
print(f"   ✓ Dataset 10min: {df.shape[0]:,} × {df.shape[1]}")

# ============================================================================
# 2. PRÉPARATION SÉRIES TEMPORELLES
# ============================================================================
print("\n[2/6] Préparation des séquences...")

TARGET = 'TEMP_STATOR_MEAN_degC'

# Features pour LSTM
FEATURES = [
    TARGET,  # La cible elle-même (pour l'historique)
    'PUISSANCE_MW',
    'COURANT_A', 'COURANT_B', 'COURANT_C',
    'TENSION_AB_V', 'TENSION_BC_V', 'TENSION_CA_V',
    'TEMP_HYDROGENE_degC'
]

# Filtrer les colonnes disponibles
available_features = [f for f in FEATURES if f in df.columns]
print(f"   ✓ Features disponibles: {len(available_features)}")

# Nettoyer les données
df_clean = df[available_features].dropna()
print(f"   ✓ Données après nettoyage: {df_clean.shape[0]:,}")

# Normalisation MinMax (recommandée pour LSTM)
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(df_clean)

# Paramètres de séquence
WINDOW_SIZE = 10  # 10 pas de temps (= 100 minutes d'historique)
HORIZON = 1       # Prédire le prochain pas (= +10 minutes)

def create_sequences(data, window_size, horizon):
    """Crée les séquences X et y pour LSTM."""
    X, y = [], []
    target_idx = 0  # TARGET est la première colonne
    
    for i in range(len(data) - window_size - horizon + 1):
        X.append(data[i:i + window_size])
        y.append(data[i + window_size + horizon - 1, target_idx])
    
    return np.array(X), np.array(y)

X, y = create_sequences(data_scaled, WINDOW_SIZE, HORIZON)
print(f"   ✓ Séquences créées: {X.shape[0]:,}")
print(f"   ✓ Shape X: {X.shape} (samples, timesteps, features)")

# Split temporel (pas de shuffle pour séries temporelles!)
train_size = int(len(X) * 0.7)
val_size = int(len(X) * 0.15)

X_train, y_train = X[:train_size], y[:train_size]
X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]

print(f"   ✓ Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")

# ============================================================================
# 3. ARCHITECTURE LSTM
# ============================================================================
print("\n[3/6] Construction du modèle LSTM...")

n_features = X.shape[2]

model = keras.Sequential([
    layers.Input(shape=(WINDOW_SIZE, n_features)),
    layers.LSTM(64, return_sequences=True),
    layers.Dropout(0.2),
    layers.LSTM(32, return_sequences=False),
    layers.Dropout(0.2),
    layers.Dense(16, activation='relu'),
    layers.Dense(1)  # Prédiction unique
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
    patience=15,
    restore_best_weights=True
)

reduce_lr = callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=7,
    min_lr=1e-6
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=64,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

# ============================================================================
# 5. ÉVALUATION
# ============================================================================
print("\n[5/6] Évaluation...")

# Prédictions
y_pred_scaled = model.predict(X_test, verbose=0).flatten()

# Dénormalisation
# Créer un array avec la bonne shape pour inverse_transform
temp_array = np.zeros((len(y_test), len(available_features)))
temp_array[:, 0] = y_test  # TARGET à l'index 0
y_test_original = scaler.inverse_transform(temp_array)[:, 0]

temp_array[:, 0] = y_pred_scaled
y_pred_original = scaler.inverse_transform(temp_array)[:, 0]

# Métriques
test_rmse = np.sqrt(mean_squared_error(y_test_original, y_pred_original))
test_mae = mean_absolute_error(y_test_original, y_pred_original)
test_r2 = r2_score(y_test_original, y_pred_original)

print("\n   📊 RÉSULTATS LSTM:")
print("   " + "-" * 40)
print(f"   RMSE: {test_rmse:.4f} °C")
print(f"   MAE:  {test_mae:.4f} °C")
print(f"   R²:   {test_r2:.4f}")
print(f"   Horizon: +10 minutes")
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
ax1.set_title('Courbes d\'Apprentissage LSTM')
ax1.legend()
ax1.grid(True)

# 2. Prédiction vs Réel (série temporelle)
ax2 = axes[0, 1]
n_plot = min(500, len(y_test_original))
ax2.plot(range(n_plot), y_test_original[:n_plot], label='Réel', alpha=0.8)
ax2.plot(range(n_plot), y_pred_original[:n_plot], label='Prédit', alpha=0.8)
ax2.set_xlabel('Index temporel')
ax2.set_ylabel('Température (°C)')
ax2.set_title('LSTM: Prédiction Séquentielle (500 points)')
ax2.legend()
ax2.grid(True)

# 3. Scatter Prédit vs Réel
ax3 = axes[1, 0]
ax3.scatter(y_test_original, y_pred_original, alpha=0.3, s=5, c='magenta')
ax3.plot([y_test_original.min(), y_test_original.max()], 
         [y_test_original.min(), y_test_original.max()], 'r--', lw=2)
ax3.set_xlabel('Température Réelle (°C)')
ax3.set_ylabel('Température Prédite (°C)')
ax3.set_title(f'R² = {test_r2:.4f}, RMSE = {test_rmse:.2f}°C')

# 4. Distribution des erreurs
ax4 = axes[1, 1]
errors = y_test_original - y_pred_original
ax4.hist(errors, bins=50, color='magenta', edgecolor='black', alpha=0.7)
ax4.axvline(0, color='red', linestyle='--')
ax4.set_xlabel('Erreur de Prédiction (°C)')
ax4.set_ylabel('Fréquence')
ax4.set_title(f'Distribution des Erreurs\nMoy: {errors.mean():.3f}°C, Std: {errors.std():.3f}°C')

plt.tight_layout()
plt.savefig(PLOTS_DIR / "04_lstm_results.png", dpi=150, bbox_inches='tight')
plt.close()
print("   ✓ 04_lstm_results.png")

# Sauvegarder métriques
metrics = {
    'model': 'LSTM (Time Series)',
    'target': f'{TARGET} @ t+10min',
    'test_rmse': test_rmse,
    'test_mae': test_mae,
    'test_r2': test_r2,
    'window_size': WINDOW_SIZE,
    'horizon': '10 minutes',
    'epochs_trained': len(history.history['loss'])
}
pd.DataFrame([metrics]).to_csv(PLOTS_DIR / "04_lstm_metrics.csv", index=False)

# Sauvegarder le modèle
model.save(PLOTS_DIR / "04_lstm_model.keras")
print("   ✓ 04_lstm_model.keras")
# Sauvegarder le scaler en PKL
joblib.dump(scaler, PLOTS_DIR / "04_lstm_scaler.pkl")
print("   \u2713 04_lstm_scaler.pkl")
print("\n" + "=" * 80)
print("✅ LSTM (Time Series) - TERMINÉ")
print(f"   R² = {test_r2:.4f} | RMSE = {test_rmse:.2f}°C")
print(f"   Prédiction à: t+10 minutes")
print("=" * 80)
