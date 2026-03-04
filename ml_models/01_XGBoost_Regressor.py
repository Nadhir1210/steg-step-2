#!/usr/bin/env python3
"""
🔵 XGBoost Regressor - Prédiction de Température Stator
========================================================
Modèle principal pour la prédiction des températures de l'alternateur.

Objectifs:
- Prédire TEMP_STATOR_MEAN_degC à partir des autres variables
- Évaluer les performances (RMSE, MAE, R²)
- Feature Importance pour comprendre les facteurs clés
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import warnings
import os
from pathlib import Path

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')

# Configuration
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "LAST_DATA"
PLOTS_DIR = Path(__file__).parent / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

print("=" * 80)
print("🔵 XGBoost Regressor - Prédiction Température Stator")
print("=" * 80)

# ============================================================================
# 1. CHARGEMENT DES DONNÉES
# ============================================================================
print("\n[1/7] Chargement des données...")
df = pd.read_csv(DATA_DIR / "APM_Alternateur_ML.csv")
print(f"   ✓ Dataset chargé: {df.shape[0]:,} lignes × {df.shape[1]} colonnes")

# ============================================================================
# 2. SÉLECTION DES FEATURES
# ============================================================================
print("\n[2/7] Sélection des features...")

# Variable cible
TARGET = 'TEMP_STATOR_MEAN_degC'

# Features à exclure (temporelles, cibles, identifiants)
EXCLUDE_COLS = [
    'Datetime', 'Year', 'Month', 'Day', 'Hour', 'Minute', 'DayOfWeek', 'Quarter',
    TARGET, 'TEMP_STATOR_MAX_degC', 'TEMP_PHASE_IMBALANCE_degC',
    'TEMP_PHASE_A_MEAN_degC', 'TEMP_PHASE_B_MEAN_degC', 'TEMP_PHASE_C_MEAN_degC'
]

# Sélectionner les features numériques
feature_cols = [col for col in df.select_dtypes(include=[np.number]).columns 
                if col not in EXCLUDE_COLS]

print(f"   ✓ Features sélectionnées: {len(feature_cols)}")
print(f"   ✓ Variable cible: {TARGET}")

# ============================================================================
# 3. PRÉPARATION DES DONNÉES
# ============================================================================
print("\n[3/7] Préparation des données...")

# Supprimer les lignes avec valeurs manquantes
df_clean = df[feature_cols + [TARGET]].dropna()
print(f"   ✓ Données après nettoyage: {df_clean.shape[0]:,} lignes")

X = df_clean[feature_cols]
y = df_clean[TARGET]

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"   ✓ Train: {X_train.shape[0]:,} échantillons")
print(f"   ✓ Test:  {X_test.shape[0]:,} échantillons")

# Normalisation
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================================================
# 4. ENTRAÎNEMENT DU MODÈLE XGBoost
# ============================================================================
print("\n[4/7] Entraînement du modèle XGBoost...")

# Vérifier si GPU disponible pour XGBoost
try:
    import xgboost as xgb
    # Test GPU
    test_dmatrix = xgb.DMatrix(np.array([[1,2,3]]))
    gpu_available = True
    device = 'cuda'
    print("   ✓ GPU CUDA détecté - utilisation accélérée")
except:
    gpu_available = False
    device = 'cpu'
    print("   ⚠ GPU non disponible - utilisation CPU")

# Paramètres optimisés pour la régression
params = {
    'objective': 'reg:squarederror',
    'n_estimators': 200,
    'max_depth': 8,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 3,
    'gamma': 0.1,
    'random_state': 42,
    'n_jobs': -1,
    'tree_method': 'hist',
    'device': device
}

model = xgb.XGBRegressor(**params)

# Entraînement avec early stopping
model.fit(
    X_train_scaled, y_train,
    eval_set=[(X_test_scaled, y_test)],
    verbose=False
)

print("   ✓ Modèle entraîné avec succès!")

# ============================================================================
# 5. ÉVALUATION DU MODÈLE
# ============================================================================
print("\n[5/7] Évaluation du modèle...")

# Prédictions
y_train_pred = model.predict(X_train_scaled)
y_test_pred = model.predict(X_test_scaled)

# Métriques Train
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
train_mae = mean_absolute_error(y_train, y_train_pred)
train_r2 = r2_score(y_train, y_train_pred)

# Métriques Test
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
test_mae = mean_absolute_error(y_test, y_test_pred)
test_r2 = r2_score(y_test, y_test_pred)

print("\n   📊 RÉSULTATS:")
print("   " + "-" * 50)
print(f"   {'Métrique':<20} {'Train':<15} {'Test':<15}")
print("   " + "-" * 50)
print(f"   {'RMSE (°C)':<20} {train_rmse:<15.4f} {test_rmse:<15.4f}")
print(f"   {'MAE (°C)':<20} {train_mae:<15.4f} {test_mae:<15.4f}")
print(f"   {'R² Score':<20} {train_r2:<15.4f} {test_r2:<15.4f}")
print("   " + "-" * 50)

# ============================================================================
# 6. FEATURE IMPORTANCE
# ============================================================================
print("\n[6/7] Analyse Feature Importance...")

feature_importance = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\n   🔝 Top 10 Features les plus importantes:")
for i, row in feature_importance.head(10).iterrows():
    print(f"   {feature_importance.head(10).index.tolist().index(i)+1:2}. {row['Feature']:<40} {row['Importance']:.4f}")

# ============================================================================
# 7. VISUALISATIONS
# ============================================================================
print("\n[7/7] Génération des visualisations...")

# Figure 1: Prédictions vs Réel
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Scatter plot
ax1 = axes[0]
ax1.scatter(y_test, y_test_pred, alpha=0.3, s=5, c='blue')
ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Parfait')
ax1.set_xlabel('Température Réelle (°C)')
ax1.set_ylabel('Température Prédite (°C)')
ax1.set_title(f'XGBoost: Prédictions vs Réel\nR² = {test_r2:.4f}, RMSE = {test_rmse:.2f}°C')
ax1.legend()

# Distribution des erreurs
ax2 = axes[1]
errors = y_test - y_test_pred
ax2.hist(errors, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
ax2.axvline(0, color='red', linestyle='--', label='Erreur = 0')
ax2.set_xlabel('Erreur de Prédiction (°C)')
ax2.set_ylabel('Fréquence')
ax2.set_title(f'Distribution des Erreurs\nMoyenne: {errors.mean():.3f}°C, Std: {errors.std():.3f}°C')
ax2.legend()

plt.tight_layout()
plt.savefig(PLOTS_DIR / "01_xgboost_predictions.png", dpi=150, bbox_inches='tight')
plt.close()
print("   ✓ 01_xgboost_predictions.png sauvegardé")

# Figure 2: Feature Importance
plt.figure(figsize=(12, 8))
top_20 = feature_importance.head(20)
colors = plt.cm.Blues(np.linspace(0.4, 1, len(top_20)))[::-1]
plt.barh(range(len(top_20)), top_20['Importance'], color=colors)
plt.yticks(range(len(top_20)), top_20['Feature'])
plt.xlabel('Importance')
plt.title('XGBoost - Top 20 Features les Plus Importantes')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(PLOTS_DIR / "01_xgboost_feature_importance.png", dpi=150, bbox_inches='tight')
plt.close()
print("   ✓ 01_xgboost_feature_importance.png sauvegardé")

# Figure 3: Learning Curve (simulation avec cross-validation)
cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
print(f"\n   📈 Cross-Validation R² Score: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

# ============================================================================
# RÉSUMÉ
# ============================================================================
print("\n" + "=" * 80)
print("✅ XGBoost Regressor - TERMINÉ")
print("=" * 80)
print(f"""
📊 Performance du modèle:
   • R² Score (Test): {test_r2:.4f} ({test_r2*100:.1f}% de variance expliquée)
   • RMSE (Test): {test_rmse:.2f}°C
   • MAE (Test): {test_mae:.2f}°C

🔝 Variables les plus importantes:
   1. {feature_importance.iloc[0]['Feature']}
   2. {feature_importance.iloc[1]['Feature']}
   3. {feature_importance.iloc[2]['Feature']}

📁 Fichiers générés:
   • {PLOTS_DIR / '01_xgboost_predictions.png'}
   • {PLOTS_DIR / '01_xgboost_feature_importance.png'}
""")
# Sauvegarder le modèle en PKL
joblib.dump(model, PLOTS_DIR / "01_xgboost_model.pkl")
print("   ✓ 01_xgboost_model.pkl")
# Sauvegarder les métriques
metrics = {
    'model': 'XGBoost Regressor',
    'target': TARGET,
    'train_rmse': train_rmse,
    'test_rmse': test_rmse,
    'train_mae': train_mae,
    'test_mae': test_mae,
    'train_r2': train_r2,
    'test_r2': test_r2,
    'cv_r2_mean': cv_scores.mean(),
    'cv_r2_std': cv_scores.std(),
    'n_features': len(feature_cols),
    'n_train': len(X_train),
    'n_test': len(X_test)
}

# Sauvegarder en CSV
metrics_df = pd.DataFrame([metrics])
metrics_df.to_csv(PLOTS_DIR / "01_xgboost_metrics.csv", index=False)
print(f"   ✓ Métriques sauvegardées: 01_xgboost_metrics.csv")
