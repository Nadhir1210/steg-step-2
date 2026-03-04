#!/usr/bin/env python3
"""
🔮 05 - LSTM PD Prediction - Prédiction Temporelle de l'Intensité PD
=====================================================================
Prédire l'évolution de PD_INTENSITY_TOTAL à t+30min ou t+1h.

Pipeline:
1. Chargement des features PD
2. Création des séquences temporelles
3. Architecture LSTM
4. Entraînement avec validation
5. Prédiction et évaluation

Objectif: Anticiper les pics de décharge pour maintenance préventive.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from pathlib import Path
import warnings
import os

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Import TensorFlow
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

plt.style.use('seaborn-v0_8-whitegrid')

# Configuration
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "LAST_DATA"
PLOTS_DIR = Path(__file__).parent / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

print("=" * 80)
print("🔮 LSTM PD PREDICTION - Prédiction Temporelle")
print("=" * 80)

# ============================================================================
# 1. CONFIGURATION
# ============================================================================
SEQUENCE_LENGTH = 60       # 60 points d'historique
PREDICTION_HORIZON = 30    # Prédire 30 points dans le futur (~30 min si 1 point/min)
FEATURES_TO_USE = [
    'PD_INTENSITY_TOTAL',
    'PD_ENERGY_TOTAL',
    'INTENSITY_ASYMMETRY',
    'CURRENT_TOTAL',
    'PULSE_TOTAL'
]
TARGET = 'PD_INTENSITY_TOTAL'

# ============================================================================
# 2. CHARGEMENT DES DONNÉES
# ============================================================================
print("\n[1/7] Chargement des données PD enrichies...")

feature_file = DATA_DIR / "TG1_Sousse_PD_Features.csv"
if feature_file.exists():
    df = pd.read_csv(feature_file)
    print(f"   ✓ Dataset enrichi: {df.shape[0]:,} × {df.shape[1]}")
else:
    print("   ⚠️ Fichier enrichi non trouvé. Exécutez d'abord 01_PD_Feature_Engineering.py")
    exit(1)

# ============================================================================
# 3. PRÉPARATION DES FEATURES
# ============================================================================
print("\n[2/7] Préparation des features...")

available_features = [f for f in FEATURES_TO_USE if f in df.columns]

# S'assurer que la target est disponible
if TARGET not in available_features:
    print(f"   ⚠️ Target '{TARGET}' non disponible. Recherche d'alternative...")
    intensity_cols = [c for c in df.columns if 'INTENSITY' in c]
    if intensity_cols:
        TARGET = intensity_cols[0]
        available_features = [TARGET] + [f for f in available_features if f != TARGET]
    else:
        print("   ❌ Impossible de trouver une colonne d'intensité")
        exit(1)

print(f"   ✓ Target: {TARGET}")
print(f"   ✓ Features: {len(available_features)}")
for f in available_features:
    print(f"      - {f}")

# Nettoyage
df_seq = df[available_features].copy()
df_seq = df_seq.replace([np.inf, -np.inf], np.nan)
df_seq = df_seq.interpolate(method='linear', limit_direction='both')
df_seq = df_seq.dropna()

print(f"   ✓ Données séquentielles: {len(df_seq):,}")

# Limiter pour performance
MAX_SAMPLES = 200000
if len(df_seq) > MAX_SAMPLES:
    df_seq = df_seq.iloc[-MAX_SAMPLES:]  # Garder les données récentes
    print(f"   ✓ Limité aux {MAX_SAMPLES:,} derniers points")

# ============================================================================
# 4. NORMALISATION
# ============================================================================
print("\n[3/7] Normalisation...")

scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(df_seq.values)

# Scaler pour la target uniquement (pour inverse transform)
target_idx = available_features.index(TARGET)
target_scaler = MinMaxScaler(feature_range=(0, 1))
target_scaler.fit(df_seq[[TARGET]].values)

print("   ✓ MinMaxScaler appliqué")

# ============================================================================
# 5. CRÉATION DES SÉQUENCES
# ============================================================================
print("\n[4/7] Création des séquences temporelles...")

def create_sequences(data, seq_length, pred_horizon, target_idx):
    """Créer des séquences pour LSTM."""
    X, y = [], []
    for i in range(len(data) - seq_length - pred_horizon + 1):
        X.append(data[i:i+seq_length])
        # Prédire la target à t+pred_horizon
        y.append(data[i+seq_length+pred_horizon-1, target_idx])
    return np.array(X), np.array(y)

X, y = create_sequences(data_scaled, SEQUENCE_LENGTH, PREDICTION_HORIZON, target_idx)

print(f"   ✓ X shape: {X.shape}")
print(f"   ✓ y shape: {y.shape}")
print(f"   → Séquences de {SEQUENCE_LENGTH} points")
print(f"   → Prédiction à t+{PREDICTION_HORIZON}")

# Split temporel (pas de shuffle pour données séquentielles)
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

print(f"   ✓ Train: {len(X_train):,} | Test: {len(X_test):,}")

# ============================================================================
# 6. ARCHITECTURE LSTM
# ============================================================================
print("\n[5/7] Construction du modèle LSTM...")

model = Sequential([
    # Couche LSTM bidirectionnelle
    Bidirectional(LSTM(64, return_sequences=True), 
                  input_shape=(SEQUENCE_LENGTH, len(available_features))),
    Dropout(0.2),
    
    # Deuxième couche LSTM
    LSTM(32, return_sequences=False),
    Dropout(0.2),
    
    # Couches denses
    Dense(16, activation='relu'),
    Dense(1, activation='linear')
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='mse',
    metrics=['mae']
)

print("\n   📊 Architecture:")
model.summary()

# ============================================================================
# 7. ENTRAÎNEMENT
# ============================================================================
print("\n[6/7] Entraînement du modèle...")

callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )
]

history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=64,
    validation_split=0.2,
    callbacks=callbacks,
    verbose=1
)

print("   ✓ Entraînement terminé")

# ============================================================================
# 8. ÉVALUATION
# ============================================================================
print("\n[7/7] Évaluation du modèle...")

# Prédictions
y_pred_scaled = model.predict(X_test, verbose=0)

# Inverse transform
y_test_orig = target_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
y_pred_orig = target_scaler.inverse_transform(y_pred_scaled).flatten()

# Métriques
rmse = np.sqrt(mean_squared_error(y_test_orig, y_pred_orig))
mae = mean_absolute_error(y_test_orig, y_pred_orig)
r2 = r2_score(y_test_orig, y_pred_orig)
mape = np.mean(np.abs((y_test_orig - y_pred_orig) / (y_test_orig + 1e-10))) * 100

print(f"\n   📊 PERFORMANCES:")
print("   " + "-" * 50)
print(f"   • RMSE: {rmse:.4f}")
print(f"   • MAE: {mae:.4f}")
print(f"   • R²: {r2:.4f}")
print(f"   • MAPE: {mape:.2f}%")

# ============================================================================
# 9. VISUALISATIONS
# ============================================================================
print("\n   Création des visualisations...")

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 1. Historique d'entraînement - Loss
ax1 = axes[0, 0]
ax1.plot(history.history['loss'], label='Train Loss', linewidth=2)
ax1.plot(history.history['val_loss'], label='Val Loss', linewidth=2)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss (MSE)')
ax1.set_title('Historique d\'Entraînement')
ax1.legend()
ax1.grid(True)

# 2. Prédictions vs Réel (segment)
ax2 = axes[0, 1]
n_plot = min(500, len(y_test_orig))
ax2.plot(y_test_orig[:n_plot], label='Réel', alpha=0.7, linewidth=1)
ax2.plot(y_pred_orig[:n_plot], label='Prédit', alpha=0.7, linewidth=1)
ax2.set_xlabel('Time Step')
ax2.set_ylabel('PD Intensity')
ax2.set_title(f'Prédiction vs Réel (t+{PREDICTION_HORIZON})')
ax2.legend()
ax2.grid(True)

# 3. Scatter plot
ax3 = axes[0, 2]
ax3.scatter(y_test_orig, y_pred_orig, alpha=0.3, s=5)
ax3.plot([y_test_orig.min(), y_test_orig.max()], 
         [y_test_orig.min(), y_test_orig.max()], 
         'r--', linewidth=2, label='Ligne parfaite')
ax3.set_xlabel('Valeurs Réelles')
ax3.set_ylabel('Valeurs Prédites')
ax3.set_title(f'Scatter: R² = {r2:.4f}')
ax3.legend()
ax3.grid(True)

# 4. Distribution des erreurs
ax4 = axes[1, 0]
errors = y_test_orig - y_pred_orig
ax4.hist(errors, bins=50, edgecolor='black', alpha=0.7)
ax4.axvline(0, color='red', linestyle='--', linewidth=2)
ax4.axvline(errors.mean(), color='blue', linestyle='--', linewidth=2, label=f'Moyenne: {errors.mean():.2f}')
ax4.set_xlabel('Erreur de Prédiction')
ax4.set_ylabel('Fréquence')
ax4.set_title('Distribution des Erreurs')
ax4.legend()

# 5. QQ Plot des erreurs
ax5 = axes[1, 1]
sorted_errors = np.sort(errors)
theoretical = np.random.normal(errors.mean(), errors.std(), len(errors))
theoretical = np.sort(theoretical)
ax5.scatter(theoretical, sorted_errors, alpha=0.3, s=5)
ax5.plot([theoretical.min(), theoretical.max()], 
         [theoretical.min(), theoretical.max()], 'r--', linewidth=2)
ax5.set_xlabel('Quantiles Théoriques')
ax5.set_ylabel('Quantiles Observés')
ax5.set_title('QQ Plot des Erreurs')

# 6. Métriques résumé
ax6 = axes[1, 2]
ax6.axis('off')
metrics_text = f"""
╔════════════════════════════════════╗
║   LSTM PD PREDICTION - RÉSUMÉ     ║
╠════════════════════════════════════╣
║                                    ║
║   📊 Configuration:                 ║
║   • Séquence: {SEQUENCE_LENGTH} points             ║
║   • Horizon: t+{PREDICTION_HORIZON} points           ║
║   • Features: {len(available_features)}                   ║
║                                    ║
║   📈 Performances:                  ║
║   • RMSE: {rmse:.4f}                  ║
║   • MAE: {mae:.4f}                   ║
║   • R²: {r2:.4f}                   ║
║   • MAPE: {mape:.2f}%                 ║
║                                    ║
║   📁 Train: {len(X_train):,}              ║
║   📁 Test: {len(X_test):,}               ║
║                                    ║
╚════════════════════════════════════╝
"""
ax6.text(0.1, 0.5, metrics_text, fontsize=10, family='monospace', 
         verticalalignment='center', transform=ax6.transAxes)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "05_lstm_pd_prediction.png", dpi=150, bbox_inches='tight')
plt.close()
print("   ✓ 05_lstm_pd_prediction.png")

# Sauvegarder le modèle
model.save(PLOTS_DIR / "05_lstm_pd_model.keras")
print("   ✓ 05_lstm_pd_model.keras")

# Sauvegarder les scalers
joblib.dump({
    'features_scaler': scaler,
    'target_scaler': target_scaler,
    'features': available_features,
    'target': TARGET,
    'target_idx': target_idx,
    'sequence_length': SEQUENCE_LENGTH,
    'prediction_horizon': PREDICTION_HORIZON
}, PLOTS_DIR / "05_lstm_pd_config.pkl")

# Métriques
metrics = {
    'rmse': rmse,
    'mae': mae,
    'r2': r2,
    'mape': mape,
    'train_samples': len(X_train),
    'test_samples': len(X_test),
    'sequence_length': SEQUENCE_LENGTH,
    'prediction_horizon': PREDICTION_HORIZON,
    'features': available_features
}
pd.DataFrame([metrics]).to_csv(PLOTS_DIR / "05_lstm_pd_metrics.csv", index=False)

print("\n" + "=" * 80)
print("✅ LSTM PD PREDICTION - TERMINÉ")
print("=" * 80)
print(f"""
📊 Résultats:
   • RMSE: {rmse:.4f}
   • MAE: {mae:.4f}
   • R²: {r2:.4f}
   • MAPE: {mape:.2f}%

🔮 Capacité de Prédiction:
   Le modèle peut prédire PD_INTENSITY à t+{PREDICTION_HORIZON} points
   basé sur les {SEQUENCE_LENGTH} dernières observations.

📁 Fichiers générés:
   • {PLOTS_DIR / '05_lstm_pd_prediction.png'}
   • {PLOTS_DIR / '05_lstm_pd_model.keras'}
   • {PLOTS_DIR / '05_lstm_pd_config.pkl'}
   • {PLOTS_DIR / '05_lstm_pd_metrics.csv'}
""")
