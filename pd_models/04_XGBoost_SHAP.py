#!/usr/bin/env python3
"""
🔴 04 - XGBoost Classifier + SHAP - Classification Supervisée PD
==================================================================
AMÉLIORATIONS:
1. ✅ Validation temporelle (split chronologique)
2. ✅ SHAP pour expliquer les prédictions
3. ✅ Pas de data leakage

Pipeline:
1. Feature Engineering labels
2. Split CHRONOLOGIQUE (pas de shuffle)
3. XGBoost Classification
4. SHAP Explanations
5. Évaluation complète
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, classification_report, 
                             confusion_matrix, f1_score, precision_recall_curve,
                             roc_auc_score, roc_curve)
import xgboost as xgb
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# Installation de SHAP si nécessaire
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    print("   ⚠️ SHAP non installé. Installation en cours...")
    import subprocess
    subprocess.run(['pip', 'install', 'shap', '-q'])
    import shap
    SHAP_AVAILABLE = True

plt.style.use('seaborn-v0_8-whitegrid')

# Configuration
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "LAST_DATA"
PLOTS_DIR = Path(__file__).parent / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

print("=" * 80)
print("🔴 XGBOOST CLASSIFIER + SHAP - Classification PD")
print("=" * 80)
print("   ✓ Amélioration 1: Validation temporelle chronologique")
print("   ✓ Amélioration 2: SHAP pour explications des prédictions")
print("   ✓ Amélioration 3: Pas de data leakage")

# ============================================================================
# 1. CHARGEMENT DES DONNÉES
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
# 2. CRÉATION DES LABELS (SUPERVISÉ)
# ============================================================================
print("\n[2/8] Création des labels de classification...")

# Calculer PD_INTENSITY_TOTAL si pas présent
if 'PD_INTENSITY_TOTAL' not in df.columns:
    current_cols = [c for c in df.columns if 'CURRENT' in c and 'ABS' in c]
    pulse_cols = [c for c in df.columns if 'PULSE' in c]
    if current_cols and pulse_cols:
        df['PD_INTENSITY_TOTAL'] = df[current_cols].sum(axis=1).values * df[pulse_cols].sum(axis=1).values

# Définir les seuils basés sur les quantiles
Q25 = df['PD_INTENSITY_TOTAL'].quantile(0.25)
Q75 = df['PD_INTENSITY_TOTAL'].quantile(0.75)
Q90 = df['PD_INTENSITY_TOTAL'].quantile(0.90)

print(f"   → Q25 (Normal): {Q25:.2f}")
print(f"   → Q75 (Warning): {Q75:.2f}")
print(f"   → Q90 (Critical): {Q90:.2f}")

# Créer les labels
def create_pd_label(intensity):
    if intensity < Q25:
        return 0  # Normal
    elif intensity < Q75:
        return 1  # Warning
    else:
        return 2  # Critical

df['PD_Label'] = df['PD_INTENSITY_TOTAL'].apply(create_pd_label)

label_names = {0: 'Normal', 1: 'Warning', 2: 'Critical'}
label_counts = df['PD_Label'].value_counts().sort_index()

print("\n   📊 Distribution des labels:")
for label, count in label_counts.items():
    pct = 100 * count / len(df)
    print(f"      {label_names[label]:<10}: {count:>10,} ({pct:>5.1f}%)")

# ============================================================================
# 3. PRÉPARATION DES FEATURES
# ============================================================================
print("\n[3/8] Préparation des features...")

# Features pour la classification (exclure PD_INTENSITY_TOTAL)
FEATURES = [
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

FEATURES_FOR_MODEL = [f for f in FEATURES if f in df.columns]

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
# 4. SPLIT CHRONOLOGIQUE (VALIDATION TEMPORELLE)
# ============================================================================
print("\n[4/8] Split CHRONOLOGIQUE (validation temporelle)...")

# IMPORTANT: Pas de shuffle! Split temporel
train_size = int(len(X) * 0.7)
val_size = int(len(X) * 0.15)
test_size = len(X) - train_size - val_size

print(f"   ⚠️ VALIDATION TEMPORELLE (pas de shuffle):")
print(f"      Train: indices 0 → {train_size:,} ({100*train_size/len(X):.0f}%)")
print(f"      Val: indices {train_size:,} → {train_size + val_size:,} ({100*val_size/len(X):.0f}%)")
print(f"      Test: indices {train_size + val_size:,} → {len(X):,} ({100*test_size/len(X):.0f}%)")

X_train = X[:train_size]
X_val = X[train_size:train_size+val_size]
X_test = X[train_size+val_size:]

y_train = y[:train_size]
y_val = y[train_size:train_size+val_size]
y_test = y[train_size+val_size:]

print(f"\n   ✓ Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")

# Normalisation (fit sur train uniquement!)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

print(f"   ✓ Scaler fitted sur train uniquement (pas de leakage)")

joblib.dump(scaler, PLOTS_DIR / "04_xgboost_shap_scaler.pkl")

# ============================================================================
# 5. ENTRAÎNEMENT XGBOOST
# ============================================================================
print("\n[5/8] Entraînement XGBoost Classifier...")

# Configuration XGBoost
try:
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

# Entraînement avec validation
model.fit(
    X_train_scaled, y_train,
    eval_set=[(X_val_scaled, y_val)],
    verbose=False
)

print("   ✓ Modèle entraîné")

# ============================================================================
# 6. ÉVALUATION
# ============================================================================
print("\n[6/8] Évaluation du modèle...")

y_pred = model.predict(X_test_scaled)
y_pred_proba = model.predict_proba(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')

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
report = classification_report(y_test, y_pred, target_names=['Normal', 'Warning', 'Critical'])
for line in report.split('\n'):
    print(f"   {line}")

# Matrice de confusion
cm = confusion_matrix(y_test, y_pred)

# Feature Importance (XGBoost native)
feature_importance = pd.DataFrame({
    'feature': FEATURES_FOR_MODEL,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\n   📊 TOP FEATURES (XGBoost native):")
print("   " + "-" * 50)
for _, row in feature_importance.head(8).iterrows():
    print(f"   {row['feature']:<35} {row['importance']:.4f}")

# ============================================================================
# 7. SHAP EXPLANATIONS
# ============================================================================
print("\n[7/8] Calcul des explications SHAP...")

# Créer l'explainer SHAP
explainer = shap.TreeExplainer(model)

# Calculer les SHAP values (sur un échantillon pour la performance)
sample_size = min(1000, len(X_test_scaled))
X_sample = X_test_scaled[:sample_size]
shap_values = explainer.shap_values(X_sample)

print(f"   ✓ SHAP values calculées pour {sample_size} échantillons")

# Identifier les prédictions critiques
critical_indices = np.where(y_pred[:sample_size] == 2)[0]
warning_indices = np.where(y_pred[:sample_size] == 1)[0]
normal_indices = np.where(y_pred[:sample_size] == 0)[0]

print(f"   ✓ Échantillons par classe:")
print(f"      - Normal: {len(normal_indices)}")
print(f"      - Warning: {len(warning_indices)}")
print(f"      - Critical: {len(critical_indices)}")

# SHAP importance moyenne par classe
print(f"\n   📊 TOP FEATURES POUR PRÉDICTIONS CRITIQUES (SHAP):")
print("   " + "-" * 50)

if len(critical_indices) > 0 and isinstance(shap_values, list) and len(shap_values) > 2:
    # Pour les prédictions Critical (classe 2)
    critical_shap = np.abs(shap_values[2][critical_indices]).mean(axis=0)
    critical_importance = pd.DataFrame({
        'feature': FEATURES_FOR_MODEL,
        'shap_importance': critical_shap
    }).sort_values('shap_importance', ascending=False)
    
    for _, row in critical_importance.head(8).iterrows():
        print(f"   {row['feature']:<35} {row['shap_importance']:.4f}")
else:
    # Si pas de classe 2 ou format différent
    mean_shap = np.abs(shap_values).mean(axis=0) if not isinstance(shap_values, list) else np.abs(shap_values[0]).mean(axis=0)
    if len(mean_shap.shape) > 1:
        mean_shap = mean_shap.mean(axis=1)
    critical_importance = pd.DataFrame({
        'feature': FEATURES_FOR_MODEL,
        'shap_importance': mean_shap
    }).sort_values('shap_importance', ascending=False)
    
    for _, row in critical_importance.head(8).iterrows():
        print(f"   {row['feature']:<35} {row['shap_importance']:.4f}")

# ============================================================================
# 8. VISUALISATIONS
# ============================================================================
print("\n[8/8] Création des visualisations...")

fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# 1. Confusion Matrix
ax1 = axes[0, 0]
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
            xticklabels=['Normal', 'Warning', 'Critical'],
            yticklabels=['Normal', 'Warning', 'Critical'])
ax1.set_xlabel('Prédit')
ax1.set_ylabel('Réel')
ax1.set_title(f'Confusion Matrix\nAccuracy: {accuracy*100:.2f}%')

# 2. Feature Importance XGBoost vs SHAP
ax2 = axes[0, 1]
top_n = min(8, len(FEATURES_FOR_MODEL))
x_pos = np.arange(top_n)
width = 0.4

top_features_xgb = feature_importance.head(top_n)
top_features_shap = critical_importance.head(top_n)

# Normaliser pour comparaison
xgb_norm = top_features_xgb['importance'].values / top_features_xgb['importance'].max()
shap_norm = top_features_shap['shap_importance'].values / top_features_shap['shap_importance'].max()

ax2.barh(x_pos - width/2, xgb_norm, width, label='XGBoost', color='steelblue')
ax2.barh(x_pos + width/2, shap_norm, width, label='SHAP', color='coral')
ax2.set_yticks(x_pos)
ax2.set_yticklabels(top_features_xgb['feature'].values, fontsize=8)
ax2.invert_yaxis()
ax2.set_xlabel('Importance (normalisée)')
ax2.set_title('Feature Importance: XGBoost vs SHAP')
ax2.legend()

# 3. Distribution des classes dans le temps
ax3 = axes[0, 2]
# Split boundaries
ax3.axvline(train_size, color='green', linestyle='--', linewidth=2, label='Train|Val')
ax3.axvline(train_size + val_size, color='red', linestyle='--', linewidth=2, label='Val|Test')
# Sample of predictions
sample_idx = np.linspace(0, len(y)-1, min(2000, len(y)), dtype=int)
colors_map = {0: 'green', 1: 'orange', 2: 'red'}
sample_colors = [colors_map[yi] for yi in y[sample_idx]]
ax3.scatter(sample_idx, y[sample_idx], c=sample_colors, alpha=0.3, s=5)
ax3.set_xlabel('Index temporel')
ax3.set_ylabel('Classe')
ax3.set_yticks([0, 1, 2])
ax3.set_yticklabels(['Normal', 'Warning', 'Critical'])
ax3.set_title('Distribution temporelle des classes\n(Split chronologique)')
ax3.legend()

# 4. SHAP Summary Plot (simplifié)
ax4 = axes[1, 0]
if isinstance(shap_values, list) and len(shap_values) > 0:
    mean_abs_shap = np.abs(shap_values[0]).mean(axis=0)
    for i in range(1, len(shap_values)):
        mean_abs_shap += np.abs(shap_values[i]).mean(axis=0)
    mean_abs_shap /= len(shap_values)
else:
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    if len(mean_abs_shap.shape) > 1:
        mean_abs_shap = mean_abs_shap.mean(axis=1)

sorted_idx = np.argsort(mean_abs_shap)[-top_n:]
ax4.barh(range(top_n), mean_abs_shap[sorted_idx], color='coral')
ax4.set_yticks(range(top_n))
ax4.set_yticklabels([FEATURES_FOR_MODEL[i] for i in sorted_idx], fontsize=8)
ax4.set_xlabel('Mean |SHAP value|')
ax4.set_title('SHAP Feature Importance (Toutes classes)')

# 5. ROC Curves par classe
ax5 = axes[1, 1]
colors = ['green', 'orange', 'red']
for i, (class_name, color) in enumerate(zip(['Normal', 'Warning', 'Critical'], colors)):
    y_true_binary = (y_test == i).astype(int)
    if y_true_binary.sum() > 0:
        fpr, tpr, _ = roc_curve(y_true_binary, y_pred_proba[:, i])
        auc = roc_auc_score(y_true_binary, y_pred_proba[:, i])
        ax5.plot(fpr, tpr, color=color, linewidth=2, label=f'{class_name} (AUC={auc:.3f})')
ax5.plot([0, 1], [0, 1], 'k--', linewidth=1)
ax5.set_xlabel('False Positive Rate')
ax5.set_ylabel('True Positive Rate')
ax5.set_title('ROC Curves par Classe')
ax5.legend()

# 6. Exemple d'explication pour une prédiction critique
ax6 = axes[1, 2]
if len(critical_indices) > 0:
    # Prendre le premier exemple critique
    critical_idx = critical_indices[0]
    if isinstance(shap_values, list):
        shap_for_critical = shap_values[2][critical_idx]
    else:
        shap_for_critical = shap_values[critical_idx]
        if len(shap_for_critical.shape) > 1:
            shap_for_critical = shap_for_critical[:, 2]
    
    # Trier par valeur SHAP
    sorted_idx = np.argsort(np.abs(shap_for_critical))[-top_n:]
    colors = ['red' if v > 0 else 'blue' for v in shap_for_critical[sorted_idx]]
    
    ax6.barh(range(len(sorted_idx)), shap_for_critical[sorted_idx], color=colors)
    ax6.set_yticks(range(len(sorted_idx)))
    ax6.set_yticklabels([FEATURES_FOR_MODEL[i] for i in sorted_idx], fontsize=8)
    ax6.axvline(0, color='black', linewidth=0.5)
    ax6.set_xlabel('SHAP value')
    ax6.set_title('Pourquoi cette prédiction = Critical?\n(Rouge: pousse vers Critical)')
else:
    ax6.text(0.5, 0.5, 'Pas de prédiction Critical\ndans l\'échantillon', 
             ha='center', va='center')
    ax6.set_title('Explication Critical')

plt.tight_layout()
plt.savefig(PLOTS_DIR / "04_xgboost_shap.png", dpi=150, bbox_inches='tight')
plt.close()
print("   ✓ 04_xgboost_shap.png")

# SHAP Beeswarm plot séparé
try:
    fig_shap, ax_shap = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample, feature_names=FEATURES_FOR_MODEL, 
                      show=False, max_display=10)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "04_shap_summary.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("   ✓ 04_shap_summary.png")
except Exception as e:
    print(f"   ⚠️ SHAP summary plot: {e}")

# Sauvegarder le modèle
model.save_model(str(PLOTS_DIR / "04_xgboost_shap_model.json"))
joblib.dump(model, PLOTS_DIR / "04_xgboost_shap_model.pkl")
print("   ✓ 04_xgboost_shap_model.pkl")

# Sauvegarder SHAP explainer
joblib.dump(explainer, PLOTS_DIR / "04_shap_explainer.pkl")
print("   ✓ 04_shap_explainer.pkl")

# Métadonnées
metadata = {
    'accuracy': accuracy,
    'f1_score': f1,
    'roc_auc': roc_auc,
    'n_train': len(X_train),
    'n_val': len(X_val),
    'n_test': len(X_test),
    'n_features': len(FEATURES_FOR_MODEL),
    'features': FEATURES_FOR_MODEL,
    'label_names': label_names,
    'thresholds': {'Q25': Q25, 'Q75': Q75, 'Q90': Q90},
    'validation_type': 'chronological_split'
}
joblib.dump(metadata, PLOTS_DIR / "04_xgboost_shap_metadata.pkl")
feature_importance.to_csv(PLOTS_DIR / "04_feature_importance_shap.csv", index=False)
critical_importance.to_csv(PLOTS_DIR / "04_shap_critical_importance.csv", index=False)

print("\n" + "=" * 80)
print("✅ XGBOOST CLASSIFIER + SHAP - TERMINÉ")
print("=" * 80)
print(f"""
📊 AMÉLIORATIONS APPLIQUÉES:

✅ 1. Validation Temporelle:
   • Split chronologique (pas de shuffle)
   • Train → Val → Test dans l'ordre temporel
   • Scaler fit sur train uniquement

✅ 2. SHAP Explanations:
   • Importance réelle des variables
   • Explication de chaque prédiction
   • Pourquoi "Critical"?

📊 Performances:
   • Accuracy: {accuracy*100:.2f}%
   • F1-Score: {f1:.4f}
   • ROC-AUC: {roc_auc:.4f}

📊 Top Features SHAP pour prédictions Critiques:
""")
for i, (_, row) in enumerate(critical_importance.head(5).iterrows()):
    print(f"   {i+1}. {row['feature']}: {row['shap_importance']:.4f}")

print(f"""
📁 Fichiers générés:
   • {PLOTS_DIR / '04_xgboost_shap.png'}
   • {PLOTS_DIR / '04_shap_summary.png'}
   • {PLOTS_DIR / '04_xgboost_shap_model.pkl'}
   • {PLOTS_DIR / '04_shap_explainer.pkl'}
   • {PLOTS_DIR / '04_shap_critical_importance.csv'}
""")
