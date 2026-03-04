#!/usr/bin/env python3
"""
🔮 05 - LSTM PD Classification - Prédiction d'Événements Critiques
===================================================================
AMÉLIORATION: Au lieu de prédire une valeur exacte, on prédit:
1. Probabilité d'événement critique dans 30 min
2. Classification temporelle (Normal → Warning → Critical)
3. Augmentation significative (> seuil)

Avantages:
- Plus robuste que la régression
- Actionnable pour la maintenance
- Métriques interprétables

⚠️ VALIDATION TEMPORELLE:
- Split chronologique (pas de shuffle)
- Pas de data leakage
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import (accuracy_score, classification_report, 
                             confusion_matrix, f1_score, roc_auc_score,
                             precision_recall_curve, roc_curve)
from pathlib import Path
import warnings
import os

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Import TensorFlow
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

plt.style.use('seaborn-v0_8-whitegrid')

# Configuration
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "LAST_DATA"
PLOTS_DIR = Path(__file__).parent / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

print("=" * 80)
print("🔮 LSTM PD CLASSIFICATION - Prédiction d'Événements Critiques")
print("=" * 80)
print("   ✓ Amélioration: Classification au lieu de régression")
print("   ✓ Validation temporelle chronologique")
print("   ✓ Pas de data leakage")

# ============================================================================
# 1. CONFIGURATION
# ============================================================================
SEQUENCE_LENGTH = 60       # 60 points d'historique
PREDICTION_HORIZON = 30    # Prédire 30 points dans le futur
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
print("\n[1/8] Chargement des données PD enrichies...")

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
print("\n[2/8] Préparation des features...")

available_features = [f for f in FEATURES_TO_USE if f in df.columns]

if TARGET not in available_features:
    intensity_cols = [c for c in df.columns if 'INTENSITY' in c]
    if intensity_cols:
        TARGET = intensity_cols[0]
        available_features = [TARGET] + [f for f in available_features if f != TARGET]

print(f"   ✓ Target: {TARGET}")
print(f"   ✓ Features: {len(available_features)}")

# Nettoyage
df_seq = df[available_features].copy()
df_seq = df_seq.replace([np.inf, -np.inf], np.nan)
df_seq = df_seq.interpolate(method='linear', limit_direction='both')
df_seq = df_seq.dropna()

print(f"   ✓ Données séquentielles: {len(df_seq):,}")

# ============================================================================
# 4. CRÉATION DES LABELS DE CLASSIFICATION
# ============================================================================
print("\n[3/8] Création des labels de classification...")

# Calculer les seuils basés sur les quantiles
Q75 = df_seq[TARGET].quantile(0.75)
Q90 = df_seq[TARGET].quantile(0.90)
Q95 = df_seq[TARGET].quantile(0.95)

print(f"   → Q75: {Q75:.2f}")
print(f"   → Q90: {Q90:.2f}")
print(f"   → Q95 (Critical threshold): {Q95:.2f}")

# Label: Est-ce qu'il y aura un événement critique dans les 30 prochains points?
def create_future_labels(data, target_col, horizon, threshold):
    """
    Pour chaque point t, regarder si max(t+1 : t+horizon) > threshold
    """
    labels = []
    values = data[target_col].values
    
    for i in range(len(values) - horizon):
        future_max = np.max(values[i+1:i+horizon+1])
        if future_max >= threshold:
            labels.append(2)  # Critical event coming
        elif future_max >= Q75:
            labels.append(1)  # Warning event coming
        else:
            labels.append(0)  # Normal
    
    return np.array(labels)

labels = create_future_labels(df_seq, TARGET, PREDICTION_HORIZON, Q95)
df_seq = df_seq.iloc[:len(labels)]  # Aligner les données

print(f"\n   📊 Distribution des labels:")
unique, counts = np.unique(labels, return_counts=True)
label_names = {0: 'Normal', 1: 'Warning', 2: 'Critical'}
for u, c in zip(unique, counts):
    pct = 100 * c / len(labels)
    print(f"      {label_names[u]:<10}: {c:>8,} ({pct:>5.1f}%)")

# ============================================================================
# 5. CRÉATION DES SÉQUENCES (VALIDATION TEMPORELLE)
# ============================================================================
print("\n[4/8] Création des séquences (validation temporelle)...")

# Normalisation AVANT le split (mais on fit uniquement sur train après)
scaler = StandardScaler()

def create_sequences_classification(data, labels, seq_length):
    """Créer des séquences pour classification LSTM."""
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(labels[i+seq_length-1])  # Label au dernier point de la séquence
    return np.array(X), np.array(y)

# Données brutes (on normalise après le split pour éviter data leakage)
data_values = df_seq.values

# SPLIT CHRONOLOGIQUE (pas de shuffle!)
train_size = int(len(data_values) * 0.7)
val_size = int(len(data_values) * 0.15)

print(f"   ⚠️ VALIDATION TEMPORELLE:")
print(f"      Train: indices 0 → {train_size:,}")
print(f"      Val: indices {train_size:,} → {train_size + val_size:,}")
print(f"      Test: indices {train_size + val_size:,} → {len(data_values):,}")

# Split chronologique
train_data = data_values[:train_size]
val_data = data_values[train_size:train_size+val_size]
test_data = data_values[train_size+val_size:]

train_labels = labels[:train_size]
val_labels = labels[train_size:train_size+val_size]
test_labels = labels[train_size+val_size:]

# Fit scaler UNIQUEMENT sur train (pas de data leakage)
scaler.fit(train_data)
train_scaled = scaler.transform(train_data)
val_scaled = scaler.transform(val_data)
test_scaled = scaler.transform(test_data)

print(f"   ✓ Scaler fitted sur train uniquement (pas de leakage)")

# Créer les séquences
X_train, y_train = create_sequences_classification(train_scaled, train_labels, SEQUENCE_LENGTH)
X_val, y_val = create_sequences_classification(val_scaled, val_labels, SEQUENCE_LENGTH)
X_test, y_test = create_sequences_classification(test_scaled, test_labels, SEQUENCE_LENGTH)

print(f"\n   ✓ X_train: {X_train.shape}")
print(f"   ✓ X_val: {X_val.shape}")
print(f"   ✓ X_test: {X_test.shape}")

# Class weights pour gérer le déséquilibre
class_counts = np.bincount(y_train)
total = len(y_train)
class_weights = {i: total / (len(class_counts) * count) for i, count in enumerate(class_counts) if count > 0}
print(f"\n   ✓ Class weights: {class_weights}")

# ============================================================================
# 6. ARCHITECTURE LSTM CLASSIFICATION
# ============================================================================
print("\n[5/8] Construction du modèle LSTM Classification...")

model = Sequential([
    # Couche LSTM bidirectionnelle
    Bidirectional(LSTM(64, return_sequences=True), 
                  input_shape=(SEQUENCE_LENGTH, len(available_features))),
    BatchNormalization(),
    Dropout(0.3),
    
    # Deuxième couche LSTM
    LSTM(32, return_sequences=False),
    BatchNormalization(),
    Dropout(0.3),
    
    # Couches denses
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(16, activation='relu'),
    
    # Sortie: 3 classes (Normal, Warning, Critical)
    Dense(3, activation='softmax')
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("\n   📊 Architecture:")
model.summary()

# ============================================================================
# 7. ENTRAÎNEMENT
# ============================================================================
print("\n[6/8] Entraînement du modèle...")

callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=8,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=4,
        min_lr=1e-6,
        verbose=1
    )
]

history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=64,
    validation_data=(X_val, y_val),
    class_weight=class_weights,
    callbacks=callbacks,
    verbose=1
)

print("   ✓ Entraînement terminé")

# ============================================================================
# 8. ÉVALUATION
# ============================================================================
print("\n[7/8] Évaluation du modèle...")

# Prédictions
y_pred_proba = model.predict(X_test, verbose=0)
y_pred = np.argmax(y_pred_proba, axis=1)

# Métriques
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')

# ROC AUC pour chaque classe (one-vs-rest)
try:
    roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')
except:
    roc_auc = 0

print(f"\n   📊 PERFORMANCES:")
print("   " + "-" * 50)
print(f"   • Accuracy: {accuracy*100:.2f}%")
print(f"   • F1-Score (weighted): {f1:.4f}")
print(f"   • ROC-AUC (weighted): {roc_auc:.4f}")

print(f"\n   📊 RAPPORT DE CLASSIFICATION:")
print("   " + "-" * 50)
# Use labels present in data
unique_labels = sorted(set(y_test) | set(y_pred))
target_names_present = [label_names[l] for l in unique_labels]
report = classification_report(y_test, y_pred, labels=unique_labels, target_names=target_names_present)
for line in report.split('\n'):
    print(f"   {line}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred, labels=unique_labels)
print(f"\n   📊 MATRICE DE CONFUSION:")
print("   " + "-" * 50)
print(f"              Prédit →")
header = "   Réel ↓     " + "  ".join([f"{label_names[l]:>7}" for l in unique_labels])
print(header)
for i, label in enumerate(unique_labels):
    row_name = label_names[label]
    row = cm[i]
    print(f"   {row_name:<10}", end="")
    for val in row:
        print(f" {val:>7}", end="")
    print()

# ============================================================================
# 9. ANALYSE SPÉCIFIQUE: PRÉDICTION D'ÉVÉNEMENTS CRITIQUES
# ============================================================================
print("\n   📊 ANALYSE DES ÉVÉNEMENTS CRITIQUES:")
print("   " + "-" * 50)

# Probabilité d'événement critique
critical_proba = y_pred_proba[:, 2]  # Classe 2 = Critical

# Seuil optimal pour alertes
thresholds = [0.3, 0.5, 0.7, 0.9]
print(f"\n   Seuil de probabilité pour alerte critique:")
for thresh in thresholds:
    predicted_critical = (critical_proba >= thresh).astype(int)
    actual_critical = (y_test == 2).astype(int)
    
    # True positives, false positives
    tp = ((predicted_critical == 1) & (actual_critical == 1)).sum()
    fp = ((predicted_critical == 1) & (actual_critical == 0)).sum()
    fn = ((predicted_critical == 0) & (actual_critical == 1)).sum()
    
    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    
    print(f"      P(critical) ≥ {thresh}: Precision={precision:.2%}, Recall={recall:.2%}")

# ============================================================================
# 10. VISUALISATIONS
# ============================================================================
print("\n[8/8] Création des visualisations...")

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 1. Historique d'entraînement - Loss
ax1 = axes[0, 0]
ax1.plot(history.history['loss'], label='Train Loss', linewidth=2)
ax1.plot(history.history['val_loss'], label='Val Loss', linewidth=2)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Historique d\'Entraînement')
ax1.legend()
ax1.grid(True)

# 2. Historique - Accuracy
ax2 = axes[0, 1]
ax2.plot(history.history['accuracy'], label='Train Acc', linewidth=2)
ax2.plot(history.history['val_accuracy'], label='Val Acc', linewidth=2)
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy')
ax2.set_title('Accuracy')
ax2.legend()
ax2.grid(True)

# 3. Confusion Matrix
ax3 = axes[0, 2]
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax3,
            xticklabels=['Normal', 'Warning', 'Critical'],
            yticklabels=['Normal', 'Warning', 'Critical'])
ax3.set_xlabel('Prédit')
ax3.set_ylabel('Réel')
ax3.set_title(f'Confusion Matrix\nAccuracy: {accuracy*100:.2f}%')

# 4. Distribution des probabilités de Critical
ax4 = axes[1, 0]
for label_val, label_name in zip([0, 1, 2], ['Normal', 'Warning', 'Critical']):
    mask = y_test == label_val
    if mask.any():
        ax4.hist(critical_proba[mask], bins=30, alpha=0.5, label=label_name)
ax4.set_xlabel('P(Critical)')
ax4.set_ylabel('Fréquence')
ax4.set_title('Distribution P(Critical) par Classe Réelle')
ax4.legend()

# 5. ROC Curve pour Critical
ax5 = axes[1, 1]
actual_critical = (y_test == 2).astype(int)
if actual_critical.sum() > 0:
    fpr, tpr, _ = roc_curve(actual_critical, critical_proba)
    ax5.plot(fpr, tpr, linewidth=2, label=f'ROC (AUC={roc_auc:.3f})')
    ax5.plot([0, 1], [0, 1], 'k--', linewidth=1)
    ax5.set_xlabel('False Positive Rate')
    ax5.set_ylabel('True Positive Rate')
    ax5.set_title('ROC Curve - Détection Critical')
    ax5.legend()
else:
    ax5.text(0.5, 0.5, 'Pas d\'événements critiques\ndans le test set', 
             ha='center', va='center')

# 6. Timeline des prédictions
ax6 = axes[1, 2]
n_plot = min(300, len(y_test))
ax6.scatter(range(n_plot), y_test[:n_plot], c='blue', alpha=0.5, s=20, label='Réel')
ax6.scatter(range(n_plot), y_pred[:n_plot] + 0.1, c='red', alpha=0.5, s=20, label='Prédit')
ax6.set_yticks([0, 1, 2])
ax6.set_yticklabels(['Normal', 'Warning', 'Critical'])
ax6.set_xlabel('Time Step')
ax6.set_ylabel('Classe')
ax6.set_title('Timeline: Réel vs Prédit')
ax6.legend()

plt.tight_layout()
plt.savefig(PLOTS_DIR / "05_lstm_pd_classification.png", dpi=150, bbox_inches='tight')
plt.close()
print("   ✓ 05_lstm_pd_classification.png")

# Sauvegarder le modèle
model.save(PLOTS_DIR / "05_lstm_pd_classifier.keras")
print("   ✓ 05_lstm_pd_classifier.keras")

# Sauvegarder les configurations
config = {
    'scaler': scaler,
    'features': available_features,
    'target': TARGET,
    'sequence_length': SEQUENCE_LENGTH,
    'prediction_horizon': PREDICTION_HORIZON,
    'thresholds': {'Q75': Q75, 'Q90': Q90, 'Q95_critical': Q95},
    'label_names': label_names
}
joblib.dump(config, PLOTS_DIR / "05_lstm_pd_classifier_config.pkl")

# Métriques
metrics = {
    'accuracy': accuracy,
    'f1_score': f1,
    'roc_auc': roc_auc,
    'train_samples': len(X_train),
    'val_samples': len(X_val),
    'test_samples': len(X_test),
    'sequence_length': SEQUENCE_LENGTH,
    'prediction_horizon': PREDICTION_HORIZON,
    'validation_type': 'chronological_split'
}
pd.DataFrame([metrics]).to_csv(PLOTS_DIR / "05_lstm_pd_classifier_metrics.csv", index=False)

print("\n" + "=" * 80)
print("✅ LSTM PD CLASSIFICATION - TERMINÉ")
print("=" * 80)
print(f"""
📊 AMÉLIORATION: Classification au lieu de Régression

🎯 Objectif: Prédire si un événement critique arrivera dans les 30 prochains points

📊 Performances:
   • Accuracy: {accuracy*100:.2f}%
   • F1-Score: {f1:.4f}
   • ROC-AUC: {roc_auc:.4f}

⚠️ Validation Temporelle:
   • Split chronologique (pas de shuffle)
   • Scaler fit sur train uniquement
   • Pas de data leakage

🔮 Utilisation:
   P(Critical) ≥ 0.5 → Alerte critique
   P(Warning) ≥ 0.5 → Surveillance accrue

📁 Fichiers générés:
   • {PLOTS_DIR / '05_lstm_pd_classification.png'}
   • {PLOTS_DIR / '05_lstm_pd_classifier.keras'}
   • {PLOTS_DIR / '05_lstm_pd_classifier_config.pkl'}
""")
