#!/usr/bin/env python3
"""
🔴 04 - XGBoost Classifier - Classification Supervisée PD
==========================================================
Classifier supervisé pour prédire l'état PD:
- Normal: PD_INTENSITY < Q25
- Warning: Q25 <= PD_INTENSITY < Q75
- Critical: PD_INTENSITY >= Q75

Pipeline:
1. Feature Engineering labels
2. Train/Test split
3. XGBoost Classification
4. Évaluation (Accuracy, Confusion Matrix, Feature Importance)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, classification_report, 
                             confusion_matrix, f1_score, precision_recall_curve)
import xgboost as xgb
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
print("🔴 XGBOOST CLASSIFIER - Classification PD")
print("=" * 80)

# ============================================================================
# 1. CHARGEMENT DES DONNÉES
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
# 2. CRÉATION DES LABELS (SUPERVISÉ)
# ============================================================================
print("\n[2/7] Création des labels de classification...")

# Calculer PD_INTENSITY_TOTAL si pas présent
if 'PD_INTENSITY_TOTAL' not in df.columns:
    # Chercher les colonnes d'intensité par canal
    intensity_cols = [c for c in df.columns if 'CURRENT' in c and 'ABS' in c]
    pulse_cols = [c for c in df.columns if 'PULSE' in c]
    if intensity_cols and pulse_cols:
        df['PD_INTENSITY_TOTAL'] = df[intensity_cols].sum(axis=1) * df[pulse_cols].sum(axis=1)

# Définir les seuils basés sur les quantiles
Q25 = df['PD_INTENSITY_TOTAL'].quantile(0.25)
Q75 = df['PD_INTENSITY_TOTAL'].quantile(0.75)
Q90 = df['PD_INTENSITY_TOTAL'].quantile(0.90)

print(f"   → Q25 (Normal): {Q25:.2f}")
print(f"   → Q75 (Warning): {Q75:.2f}")
print(f"   → Q90 (Critical): {Q90:.2f}")

# Créer les labels
def create_pd_label(intensity):
    """Classifier l'état PD basé sur l'intensité."""
    if intensity < Q25:
        return 0  # Normal
    elif intensity < Q75:
        return 1  # Warning
    else:
        return 2  # Critical

df['PD_Label'] = df['PD_INTENSITY_TOTAL'].apply(create_pd_label)

# Alternative: seuils plus discriminants
# Normal: < Q50, Warning: Q50-Q90, Critical: > Q90

label_names = {0: 'Normal', 1: 'Warning', 2: 'Critical'}
label_counts = df['PD_Label'].value_counts().sort_index()

print("\n   📊 Distribution des labels:")
for label, count in label_counts.items():
    pct = 100 * count / len(df)
    print(f"      {label_names[label]:<10}: {count:>10,} ({pct:>5.1f}%)")

# ============================================================================
# 3. PRÉPARATION DES FEATURES
# ============================================================================
print("\n[3/7] Préparation des features...")

# Features pour la classification
FEATURES = [
    'PD_INTENSITY_TOTAL',
    'PD_ENERGY_TOTAL',
    'INTENSITY_ASYMMETRY',
    'ENERGY_ASYMMETRY',
    'CURRENT_TOTAL',
    'PULSE_TOTAL',
    'MAX_CHARGE_TOTAL',
    'MEAN_CHARGE_TOTAL',
    'PD_INTENSITY_ROLL_MEAN_10min',
    'PD_INTENSITY_ROLL_STD_30min',
    'INTENSITY_CV'
]

# Note: On exclut PD_INTENSITY_TOTAL car c'est la base du label
# En pratique, on utiliserait des features indépendantes
FEATURES_FOR_MODEL = [f for f in FEATURES if f in df.columns and f != 'PD_INTENSITY_TOTAL']

# Si pas assez de features, utiliser des colonnes alternatives
if len(FEATURES_FOR_MODEL) < 3:
    FEATURES_FOR_MODEL = [c for c in df.columns if any(x in c for x in ['CURRENT', 'PULSE', 'ENERGY', 'CHARGE'])][:10]

print(f"   ✓ {len(FEATURES_FOR_MODEL)} features sélectionnées")
for f in FEATURES_FOR_MODEL[:8]:
    print(f"      - {f}")
if len(FEATURES_FOR_MODEL) > 8:
    print(f"      ... et {len(FEATURES_FOR_MODEL) - 8} autres")

# Préparer X et y
df_clean = df[FEATURES_FOR_MODEL + ['PD_Label']].copy()
df_clean = df_clean.replace([np.inf, -np.inf], np.nan)
df_clean = df_clean.dropna()

X = df_clean[FEATURES_FOR_MODEL].values
y = df_clean['PD_Label'].values

print(f"   ✓ Dataset final: {len(X):,} échantillons")

# ============================================================================
# 4. SPLIT ET NORMALISATION
# ============================================================================
print("\n[4/7] Split et normalisation...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"   ✓ Train: {len(X_train):,} | Test: {len(X_test):,}")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

joblib.dump(scaler, PLOTS_DIR / "04_xgboost_scaler.pkl")

# ============================================================================
# 5. ENTRAÎNEMENT XGBOOST
# ============================================================================
print("\n[5/7] Entraînement XGBoost Classifier...")

# Configuration XGBoost
try:
    # Essayer avec GPU
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=8,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='multi:softmax',
        num_class=3,
        device='cuda',
        random_state=42,
        n_jobs=-1
    )
    print("   → GPU CUDA activé")
except:
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=8,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='multi:softmax',
        num_class=3,
        random_state=42,
        n_jobs=-1
    )
    print("   → CPU mode")

model.fit(
    X_train_scaled, y_train,
    eval_set=[(X_test_scaled, y_test)],
    verbose=False
)

print("   ✓ Modèle entraîné")

# ============================================================================
# 6. ÉVALUATION
# ============================================================================
print("\n[6/7] Évaluation du modèle...")

y_pred = model.predict(X_test_scaled)
y_pred_proba = model.predict_proba(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"\n   📊 PERFORMANCES:")
print("   " + "-" * 50)
print(f"   • Accuracy: {accuracy*100:.2f}%")
print(f"   • F1-Score (weighted): {f1:.4f}")

print(f"\n   📊 RAPPORT DE CLASSIFICATION:")
print("   " + "-" * 50)
report = classification_report(y_test, y_pred, target_names=['Normal', 'Warning', 'Critical'])
for line in report.split('\n'):
    print(f"   {line}")

# Matrice de confusion
cm = confusion_matrix(y_test, y_pred)
print(f"\n   📊 MATRICE DE CONFUSION:")
print("   " + "-" * 50)
print(f"              Prédit →")
print(f"   Réel ↓     Normal  Warning  Critical")
for i, row_name in enumerate(['Normal', 'Warning', 'Critical']):
    row = cm[i]
    print(f"   {row_name:<10} {row[0]:>7} {row[1]:>8} {row[2]:>9}")

# Feature Importance
feature_importance = pd.DataFrame({
    'feature': FEATURES_FOR_MODEL,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\n   📊 TOP 10 FEATURES IMPORTANTES:")
print("   " + "-" * 50)
for _, row in feature_importance.head(10).iterrows():
    print(f"   {row['feature']:<35} {row['importance']:.4f}")

# ============================================================================
# 7. VISUALISATIONS
# ============================================================================
print("\n[7/7] Création des visualisations...")

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 1. Matrice de confusion
ax1 = axes[0, 0]
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
            xticklabels=['Normal', 'Warning', 'Critical'],
            yticklabels=['Normal', 'Warning', 'Critical'])
ax1.set_xlabel('Prédit')
ax1.set_ylabel('Réel')
ax1.set_title(f'Matrice de Confusion\nAccuracy: {accuracy*100:.2f}%')

# 2. Feature Importance
ax2 = axes[0, 1]
top_features = feature_importance.head(10)
colors = plt.cm.Blues(np.linspace(0.8, 0.3, len(top_features)))
ax2.barh(range(len(top_features)), top_features['importance'].values, color=colors)
ax2.set_yticks(range(len(top_features)))
ax2.set_yticklabels(top_features['feature'].values, fontsize=8)
ax2.invert_yaxis()
ax2.set_xlabel('Importance')
ax2.set_title('Feature Importance (Top 10)')

# 3. Distribution des classes
ax3 = axes[0, 2]
class_colors = ['green', 'orange', 'red']
bars = ax3.bar(['Normal', 'Warning', 'Critical'], label_counts.values, color=class_colors)
for bar, count in zip(bars, label_counts.values):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 500, 
             f'{count:,}', ha='center', fontsize=9)
ax3.set_ylabel('Nombre d\'échantillons')
ax3.set_title('Distribution des Classes')

# 4. Precision-Recall per class
ax4 = axes[1, 0]
for i, class_name in enumerate(['Normal', 'Warning', 'Critical']):
    y_true_binary = (y_test == i).astype(int)
    precision, recall, _ = precision_recall_curve(y_true_binary, y_pred_proba[:, i])
    ax4.plot(recall, precision, label=f'{class_name}', linewidth=2)
ax4.set_xlabel('Recall')
ax4.set_ylabel('Precision')
ax4.set_title('Courbes Precision-Recall')
ax4.legend()
ax4.grid(True)

# 5. Confusion Matrix Normalisée
ax5 = axes[1, 1]
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Blues', ax=ax5,
            xticklabels=['Normal', 'Warning', 'Critical'],
            yticklabels=['Normal', 'Warning', 'Critical'])
ax5.set_xlabel('Prédit')
ax5.set_ylabel('Réel')
ax5.set_title('Confusion Matrix (Normalisée)')

# 6. Métriques par classe
ax6 = axes[1, 2]
metrics_per_class = []
for i, class_name in enumerate(['Normal', 'Warning', 'Critical']):
    precision = cm[i, i] / cm[:, i].sum() if cm[:, i].sum() > 0 else 0
    recall = cm[i, i] / cm[i, :].sum() if cm[i, :].sum() > 0 else 0
    metrics_per_class.append({'class': class_name, 'precision': precision, 'recall': recall})

metrics_df = pd.DataFrame(metrics_per_class)
x = np.arange(len(metrics_df))
width = 0.35

bars1 = ax6.bar(x - width/2, metrics_df['precision'], width, label='Precision', color='steelblue')
bars2 = ax6.bar(x + width/2, metrics_df['recall'], width, label='Recall', color='coral')

ax6.set_ylabel('Score')
ax6.set_title('Precision/Recall par Classe')
ax6.set_xticks(x)
ax6.set_xticklabels(metrics_df['class'])
ax6.legend()
ax6.set_ylim(0, 1.1)

for bar in bars1:
    ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
             f'{bar.get_height():.2f}', ha='center', fontsize=8)
for bar in bars2:
    ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
             f'{bar.get_height():.2f}', ha='center', fontsize=8)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "04_xgboost_classifier.png", dpi=150, bbox_inches='tight')
plt.close()
print("   ✓ 04_xgboost_classifier.png")

# Sauvegarder le modèle
model.save_model(str(PLOTS_DIR / "04_xgboost_classifier.json"))
joblib.dump(model, PLOTS_DIR / "04_xgboost_classifier.pkl")
print("   ✓ 04_xgboost_classifier.pkl")

# Sauvegarder les métadonnées
metadata = {
    'accuracy': accuracy,
    'f1_score': f1,
    'n_train': len(X_train),
    'n_test': len(X_test),
    'n_features': len(FEATURES_FOR_MODEL),
    'features': FEATURES_FOR_MODEL,
    'label_names': label_names,
    'thresholds': {'Q25': Q25, 'Q75': Q75, 'Q90': Q90}
}
joblib.dump(metadata, PLOTS_DIR / "04_xgboost_metadata.pkl")
feature_importance.to_csv(PLOTS_DIR / "04_feature_importance.csv", index=False)

print("\n" + "=" * 80)
print("✅ XGBOOST CLASSIFIER - TERMINÉ")
print("=" * 80)
print(f"""
📊 Résultats:
   • Accuracy: {accuracy*100:.2f}%
   • F1-Score: {f1:.4f}
   
   Classification:
   • Normal (Intensité < {Q25:.0f}): Fonctionnement sain
   • Warning ({Q25:.0f} <= Intensité < {Q75:.0f}): Surveillance requise
   • Critical (Intensité >= {Q75:.0f}): Intervention recommandée

📁 Fichiers générés:
   • {PLOTS_DIR / '04_xgboost_classifier.png'}
   • {PLOTS_DIR / '04_xgboost_classifier.pkl'}
   • {PLOTS_DIR / '04_xgboost_scaler.pkl'}
   • {PLOTS_DIR / '04_feature_importance.csv'}
""")
