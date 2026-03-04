"""
🔥 MODULE 1 - Thermal Health Modeling
=====================================
TG1 Turbo-Alternator - Digital Twin Health Monitoring System

Objectif: Modéliser le comportement thermique normal et détecter les anomalies
- Thermal Baseline Model: Temp = f(Load, Reactive, Ambient)
- Residual Anomaly Detection: Si résiduel ↑↑ → problème refroidissement

Auteur: Nadhir - Stage STEG 2026
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb
import joblib
import warnings
warnings.filterwarnings('ignore')

# Configuration
plt.style.use('seaborn-v0_8-whitegrid')
PLOTS_DIR = "tg1_monitoring/plots"
DATA_DIR = "LAST_DATA"

print("=" * 70)
print("🔥 MODULE 1 - THERMAL HEALTH MODELING")
print("=" * 70)

# =============================================================================
# 1. CHARGEMENT DES DONNÉES
# =============================================================================
print("\n📥 Chargement des données APM Alternateur...")

# Utiliser le dataset 10-min pour éviter les problèmes de mémoire
df = pd.read_csv(f"{DATA_DIR}/APM_Alternateur_10min_ML.csv")
print(f"   Dataset: {len(df):,} lignes × {len(df.columns)} colonnes")

# Colonnes de température stator
temp_cols = [col for col in df.columns if 'STATOR' in col and 'TEMP' in col]
print(f"   Colonnes température: {len(temp_cols)}")

# =============================================================================
# 2. FEATURE ENGINEERING
# =============================================================================
print("\n🔧 Feature Engineering...")

# Créer les features principales
df['LOAD_MW'] = df['MODE_TAG_1']  # Puissance active
df['REACTIVE_MVAR'] = df['REACTIVE_LOAD']  # Puissance réactive
df['AMBIENT_TEMP'] = df['AMBIENT_AIR_TEMP_C']

# Température moyenne du stator (toutes les phases)
df['STATOR_TEMP_MEAN'] = df[temp_cols].mean(axis=1)
df['STATOR_TEMP_MAX'] = df[temp_cols].max(axis=1)
df['STATOR_TEMP_MIN'] = df[temp_cols].min(axis=1)
df['STATOR_TEMP_STD'] = df[temp_cols].std(axis=1)

# Température par phase
phase_a_cols = [col for col in temp_cols if 'PHASE_A' in col]
phase_b_cols = [col for col in temp_cols if 'PHASE_B' in col]
phase_c_cols = [col for col in temp_cols if 'PHASE_C' in col]

df['PHASE_A_MEAN'] = df[phase_a_cols].mean(axis=1)
df['PHASE_B_MEAN'] = df[phase_b_cols].mean(axis=1)
df['PHASE_C_MEAN'] = df[phase_c_cols].mean(axis=1)

# Déséquilibre entre phases
df['PHASE_IMBALANCE'] = df['STATOR_TEMP_MAX'] - df['STATOR_TEMP_MIN']

# Delta température refroidissement
df['COOLING_DELTA'] = df['ENCLOSED_HOT_AIR_TEMP_1_degC'] - df['ENCLOSED_COLD_AIR_TEMP_1_degC']

# Température excès par rapport à l'ambiant
df['TEMP_RISE'] = df['STATOR_TEMP_MEAN'] - df['AMBIENT_TEMP']

# Ratio charge / température
df['LOAD_TEMP_RATIO'] = df['LOAD_MW'] / (df['STATOR_TEMP_MEAN'] + 1)

# Features temporelles
df['HOUR'] = df['Hour']
df['IS_PEAK_HOUR'] = df['Hour'].apply(lambda x: 1 if 8 <= x <= 20 else 0)

# Supprimer les lignes avec valeurs manquantes ou zéros
df_clean = df[(df['LOAD_MW'] > 0) & (df['STATOR_TEMP_MEAN'] > 0)].copy()
print(f"   Dataset nettoyé: {len(df_clean):,} lignes")

# =============================================================================
# 3. THERMAL BASELINE MODEL
# =============================================================================
print("\n🎯 Entraînement du Thermal Baseline Model...")
print("   Modèle: STATOR_TEMP = f(Load, Reactive, Ambient, Cooling)")

# Features et target
feature_cols = ['LOAD_MW', 'REACTIVE_MVAR', 'AMBIENT_TEMP', 'COOLING_DELTA', 
                'HOUR', 'IS_PEAK_HOUR']
target = 'STATOR_TEMP_MEAN'

X = df_clean[feature_cols].values
y = df_clean[target].values

# Split chronologique (validation temporelle)
train_size = int(len(X) * 0.7)
val_size = int(len(X) * 0.15)

X_train = X[:train_size]
y_train = y[:train_size]
X_val = X[train_size:train_size+val_size]
y_val = y[train_size:train_size+val_size]
X_test = X[train_size+val_size:]
y_test = y[train_size+val_size:]

print(f"   Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")

# Normalisation (fit sur train uniquement)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Modèle XGBoost
print("\n   Training XGBoost Regressor...")
xgb_model = xgb.XGBRegressor(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    tree_method='hist',  # Pour GPU si disponible
    n_jobs=-1
)

xgb_model.fit(
    X_train_scaled, y_train,
    eval_set=[(X_val_scaled, y_val)],
    verbose=False
)

# Prédictions
y_pred_train = xgb_model.predict(X_train_scaled)
y_pred_val = xgb_model.predict(X_val_scaled)
y_pred_test = xgb_model.predict(X_test_scaled)

# Métriques
print("\n📊 Performances du Thermal Baseline Model:")
print("-" * 50)

metrics = {
    'Train': (y_train, y_pred_train),
    'Validation': (y_val, y_pred_val),
    'Test': (y_test, y_pred_test)
}

results = {}
for name, (y_true, y_pred) in metrics.items():
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    results[name] = {'RMSE': rmse, 'MAE': mae, 'R²': r2}
    print(f"   {name:12} → RMSE: {rmse:.2f}°C | MAE: {mae:.2f}°C | R²: {r2:.4f}")

# =============================================================================
# 4. RESIDUAL ANOMALY DETECTION
# =============================================================================
print("\n🔴 Residual Anomaly Detection...")

# Calculer les résidus sur tout le dataset
X_all_scaled = scaler.transform(X)
y_pred_all = xgb_model.predict(X_all_scaled)
residuals = y - y_pred_all

# Statistiques des résidus
residual_mean = np.mean(residuals)
residual_std = np.std(residuals)

print(f"   Résidus: mean = {residual_mean:.2f}°C, std = {residual_std:.2f}°C")

# Seuils d'anomalie (3 sigma rule)
threshold_warning = residual_mean + 2 * residual_std
threshold_critical = residual_mean + 3 * residual_std

print(f"   Seuil Warning: > {threshold_warning:.2f}°C")
print(f"   Seuil Critical: > {threshold_critical:.2f}°C")

# Classification des anomalies
df_clean['THERMAL_RESIDUAL'] = residuals
df_clean['THERMAL_STATUS'] = pd.cut(
    residuals,
    bins=[-np.inf, -threshold_critical, -threshold_warning, 
          threshold_warning, threshold_critical, np.inf],
    labels=['Sous-refroidi', 'Normal-', 'Normal', 'Warning', 'Critical']
)

# Distribution des statuts
status_counts = df_clean['THERMAL_STATUS'].value_counts()
print("\n📈 Distribution des statuts thermiques:")
for status, count in status_counts.items():
    pct = count / len(df_clean) * 100
    print(f"   {status}: {count:,} ({pct:.1f}%)")

# =============================================================================
# 5. FEATURE IMPORTANCE
# =============================================================================
print("\n🔍 Feature Importance:")


importance = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': xgb_model.feature_importances_
}).sort_values('Importance', ascending=False)

for _, row in importance.iterrows():
    print(f"   {row['Feature']:20} → {row['Importance']:.4f}")

# =============================================================================
# 6. VISUALIZATIONS
# =============================================================================
print("\n📊 Génération des visualisations...")

fig = plt.figure(figsize=(20, 16))

# 1. Prédiction vs Réalité
ax1 = fig.add_subplot(3, 3, 1)
ax1.scatter(y_test[:500], y_pred_test[:500], alpha=0.5, s=10)
ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
ax1.set_xlabel('Température Réelle (°C)')
ax1.set_ylabel('Température Prédite (°C)')
ax1.set_title(f'Thermal Baseline Model (R² = {results["Test"]["R²"]:.4f})')

# 2. Distribution des résidus
ax2 = fig.add_subplot(3, 3, 2)
ax2.hist(residuals, bins=50, color='steelblue', edgecolor='white', alpha=0.7)
ax2.axvline(threshold_warning, color='orange', linestyle='--', label=f'Warning ({threshold_warning:.1f}°C)')
ax2.axvline(threshold_critical, color='red', linestyle='--', label=f'Critical ({threshold_critical:.1f}°C)')
ax2.axvline(-threshold_warning, color='orange', linestyle='--')
ax2.axvline(-threshold_critical, color='red', linestyle='--')
ax2.set_xlabel('Résidu (°C)')
ax2.set_ylabel('Fréquence')
ax2.set_title('Distribution des Résidus Thermiques')
ax2.legend()

# 3. Feature Importance
ax3 = fig.add_subplot(3, 3, 3)
colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(importance)))
ax3.barh(importance['Feature'], importance['Importance'], color=colors)
ax3.set_xlabel('Importance')
ax3.set_title('Feature Importance - Thermal Model')
ax3.invert_yaxis()

# 4. Température vs Charge
ax4 = fig.add_subplot(3, 3, 4)
scatter = ax4.scatter(df_clean['LOAD_MW'][:2000], df_clean['STATOR_TEMP_MEAN'][:2000], 
                      c=df_clean['AMBIENT_TEMP'][:2000], cmap='coolwarm', alpha=0.6, s=10)
ax4.set_xlabel('Charge (MW)')
ax4.set_ylabel('Température Stator (°C)')
ax4.set_title('Temp vs Charge (coloré par Ambiant)')
plt.colorbar(scatter, ax=ax4, label='Temp Ambiante (°C)')

# 5. Série temporelle des résidus
ax5 = fig.add_subplot(3, 3, 5)
ax5.plot(residuals[:1000], linewidth=0.5, color='steelblue')
ax5.axhline(threshold_warning, color='orange', linestyle='--', alpha=0.7)
ax5.axhline(threshold_critical, color='red', linestyle='--', alpha=0.7)
ax5.axhline(-threshold_warning, color='orange', linestyle='--', alpha=0.7)
ax5.axhline(-threshold_critical, color='red', linestyle='--', alpha=0.7)
ax5.fill_between(range(1000), -threshold_warning, threshold_warning, alpha=0.2, color='green')
ax5.set_xlabel('Index temporel')
ax5.set_ylabel('Résidu (°C)')
ax5.set_title('Évolution temporelle des résidus')

# 6. Heatmap des corrélations
ax6 = fig.add_subplot(3, 3, 6)
corr_cols = ['LOAD_MW', 'REACTIVE_MVAR', 'AMBIENT_TEMP', 'STATOR_TEMP_MEAN', 
             'COOLING_DELTA', 'PHASE_IMBALANCE']
corr_matrix = df_clean[corr_cols].corr()
sns.heatmap(corr_matrix, annot=True, cmap='RdYlBu_r', center=0, ax=ax6, fmt='.2f')
ax6.set_title('Corrélations Thermiques')

# 7. Température par phase
ax7 = fig.add_subplot(3, 3, 7)
phase_data = df_clean[['PHASE_A_MEAN', 'PHASE_B_MEAN', 'PHASE_C_MEAN']].melt()
sns.boxplot(x='variable', y='value', data=phase_data, ax=ax7, palette='Set2')
ax7.set_xlabel('Phase')
ax7.set_ylabel('Température (°C)')
ax7.set_title('Distribution par Phase')

# 8. Température vs Heure
ax8 = fig.add_subplot(3, 3, 8)
hourly_temp = df_clean.groupby('HOUR')['STATOR_TEMP_MEAN'].mean()
ax8.bar(hourly_temp.index, hourly_temp.values, color='coral', edgecolor='white')
ax8.set_xlabel('Heure')
ax8.set_ylabel('Température Moyenne (°C)')
ax8.set_title('Profil Thermique Journalier')

# 9. Distribution des statuts
ax9 = fig.add_subplot(3, 3, 9)
colors_status = {'Sous-refroidi': 'blue', 'Normal-': 'lightgreen', 
                 'Normal': 'green', 'Warning': 'orange', 'Critical': 'red'}
status_pct = status_counts / len(df_clean) * 100
bars = ax9.bar(status_pct.index, status_pct.values, 
               color=[colors_status.get(x, 'gray') for x in status_pct.index])
ax9.set_xlabel('Statut Thermique')
ax9.set_ylabel('Pourcentage (%)')
ax9.set_title('Distribution des États Thermiques')
ax9.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}/01_thermal_health_model.png', dpi=150, bbox_inches='tight')
plt.close()

print(f"   ✅ Sauvegardé: {PLOTS_DIR}/01_thermal_health_model.png")

# =============================================================================
# 7. SAUVEGARDE DES MODÈLES
# =============================================================================
print("\n💾 Sauvegarde des modèles...")

# Sauvegarder le modèle XGBoost
joblib.dump(xgb_model, f'{PLOTS_DIR}/01_thermal_xgb_model.pkl')
joblib.dump(scaler, f'{PLOTS_DIR}/01_thermal_scaler.pkl')

# Sauvegarder les paramètres
thermal_config = {
    'feature_cols': feature_cols,
    'target': target,
    'residual_mean': residual_mean,
    'residual_std': residual_std,
    'threshold_warning': threshold_warning,
    'threshold_critical': threshold_critical,
    'metrics': results
}
joblib.dump(thermal_config, f'{PLOTS_DIR}/01_thermal_config.pkl')

print(f"   ✅ Modèle: {PLOTS_DIR}/01_thermal_xgb_model.pkl")
print(f"   ✅ Scaler: {PLOTS_DIR}/01_thermal_scaler.pkl")
print(f"   ✅ Config: {PLOTS_DIR}/01_thermal_config.pkl")

# =============================================================================
# RÉSUMÉ
# =============================================================================
print("\n" + "=" * 70)
print("📊 RÉSUMÉ - THERMAL HEALTH MODEL")
print("=" * 70)
print(f"""
🔥 Thermal Baseline Model:
   • R² Score (Test): {results['Test']['R²']:.4f}
   • RMSE: {results['Test']['RMSE']:.2f}°C
   • MAE: {results['Test']['MAE']:.2f}°C

🔴 Residual Anomaly Detection:
   • Seuil Warning: > {threshold_warning:.2f}°C
   • Seuil Critical: > {threshold_critical:.2f}°C
   • Anomalies Warning: {status_counts.get('Warning', 0):,} ({status_counts.get('Warning', 0)/len(df_clean)*100:.2f}%)
   • Anomalies Critical: {status_counts.get('Critical', 0):,} ({status_counts.get('Critical', 0)/len(df_clean)*100:.2f}%)

🔍 Top Features:
   1. {importance.iloc[0]['Feature']}: {importance.iloc[0]['Importance']:.4f}
   2. {importance.iloc[1]['Feature']}: {importance.iloc[1]['Importance']:.4f}
   3. {importance.iloc[2]['Feature']}: {importance.iloc[2]['Importance']:.4f}
""")
print("=" * 70)
