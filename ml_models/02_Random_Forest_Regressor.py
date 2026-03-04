#!/usr/bin/env python3
"""
🔵 Random Forest Regressor - Prédiction de Température Stator
==============================================================
Alternative à XGBoost, plus simple et interprétable.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')

# Configuration
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "LAST_DATA"
PLOTS_DIR = Path(__file__).parent / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

print("=" * 80)
print("🔵 Random Forest Regressor - Prédiction Température Stator")
print("=" * 80)

# ============================================================================
# 1. CHARGEMENT DES DONNÉES
# ============================================================================
print("\n[1/6] Chargement des données...")
df = pd.read_csv(DATA_DIR / "APM_Alternateur_ML.csv")
print(f"   ✓ Dataset chargé: {df.shape[0]:,} lignes × {df.shape[1]} colonnes")

# ============================================================================
# 2. PRÉPARATION DES FEATURES
# ============================================================================
print("\n[2/6] Préparation des features...")

TARGET = 'TEMP_STATOR_MEAN_degC'
EXCLUDE_COLS = [
    'Datetime', 'Year', 'Month', 'Day', 'Hour', 'Minute', 'DayOfWeek', 'Quarter',
    TARGET, 'TEMP_STATOR_MAX_degC', 'TEMP_PHASE_IMBALANCE_degC',
    'TEMP_PHASE_A_MEAN_degC', 'TEMP_PHASE_B_MEAN_degC', 'TEMP_PHASE_C_MEAN_degC'
]

feature_cols = [col for col in df.select_dtypes(include=[np.number]).columns 
                if col not in EXCLUDE_COLS]

df_clean = df[feature_cols + [TARGET]].dropna()
X = df_clean[feature_cols]
y = df_clean[TARGET]

# Échantillonnage pour accélérer (RF est lent sur gros dataset)
sample_size = min(100000, len(df_clean))
indices = np.random.choice(len(df_clean), sample_size, replace=False)
X_sample = X.iloc[indices]
y_sample = y.iloc[indices]

X_train, X_test, y_train, y_test = train_test_split(
    X_sample, y_sample, test_size=0.2, random_state=42
)

print(f"   ✓ Échantillon: {sample_size:,} (sur {len(df_clean):,})")
print(f"   ✓ Train: {len(X_train):,} | Test: {len(X_test):,}")

# Normalisation
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================================================
# 3. ENTRAÎNEMENT RANDOM FOREST
# ============================================================================
print("\n[3/6] Entraînement Random Forest...")

model = RandomForestRegressor(
    n_estimators=100,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    n_jobs=-1,
    random_state=42
)

model.fit(X_train_scaled, y_train)
print("   ✓ Modèle entraîné!")

# ============================================================================
# 4. ÉVALUATION
# ============================================================================
print("\n[4/6] Évaluation...")

y_train_pred = model.predict(X_train_scaled)
y_test_pred = model.predict(X_test_scaled)

# Métriques
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
train_mae = mean_absolute_error(y_train, y_train_pred)
train_r2 = r2_score(y_train, y_train_pred)

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
# 5. FEATURE IMPORTANCE
# ============================================================================
print("\n[5/6] Feature Importance...")

feature_importance = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\n   🔝 Top 10 Features:")
for idx, row in enumerate(feature_importance.head(10).itertuples()):
    print(f"   {idx+1:2}. {row.Feature:<40} {row.Importance:.4f}")

# ============================================================================
# 6. VISUALISATIONS
# ============================================================================
print("\n[6/6] Génération des visualisations...")

# Figure 1: Prédictions
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].scatter(y_test, y_test_pred, alpha=0.3, s=5, c='forestgreen')
axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[0].set_xlabel('Température Réelle (°C)')
axes[0].set_ylabel('Température Prédite (°C)')
axes[0].set_title(f'Random Forest: Prédictions vs Réel\nR² = {test_r2:.4f}, RMSE = {test_rmse:.2f}°C')

errors = y_test - y_test_pred
axes[1].hist(errors, bins=50, color='forestgreen', edgecolor='black', alpha=0.7)
axes[1].axvline(0, color='red', linestyle='--')
axes[1].set_xlabel('Erreur (°C)')
axes[1].set_ylabel('Fréquence')
axes[1].set_title(f'Distribution des Erreurs\nMoy: {errors.mean():.3f}°C, Std: {errors.std():.3f}°C')

plt.tight_layout()
plt.savefig(PLOTS_DIR / "02_random_forest_predictions.png", dpi=150, bbox_inches='tight')
plt.close()
print("   ✓ 02_random_forest_predictions.png")

# Figure 2: Feature Importance
plt.figure(figsize=(12, 8))
top_20 = feature_importance.head(20)
colors = plt.cm.Greens(np.linspace(0.4, 1, len(top_20)))[::-1]
plt.barh(range(len(top_20)), top_20['Importance'], color=colors)
plt.yticks(range(len(top_20)), top_20['Feature'])
plt.xlabel('Importance')
plt.title('Random Forest - Top 20 Features')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(PLOTS_DIR / "02_random_forest_feature_importance.png", dpi=150, bbox_inches='tight')
plt.close()
print("   ✓ 02_random_forest_feature_importance.png")
# Sauvegarder le modèle en PKL
joblib.dump(model, PLOTS_DIR / "02_random_forest_model.pkl")
print("   \u2713 02_random_forest_model.pkl")
# Sauvegarder métriques
metrics = {
    'model': 'Random Forest',
    'target': TARGET,
    'test_rmse': test_rmse,
    'test_mae': test_mae,
    'test_r2': test_r2,
    'n_estimators': 100,
    'max_depth': 15
}
pd.DataFrame([metrics]).to_csv(PLOTS_DIR / "02_random_forest_metrics.csv", index=False)

print("\n" + "=" * 80)
print("✅ Random Forest - TERMINÉ")
print(f"   R² = {test_r2:.4f} | RMSE = {test_rmse:.2f}°C")
print("=" * 80)
