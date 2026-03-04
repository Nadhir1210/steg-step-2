"""
🔵 MODULE 4 - LOAD VS TEMPERATURE COUPLING
==========================================
Analyse du couplage Charge (MW) vs Température Stator

Contenu:
- Feature Interaction Modeling
- SHAP Analysis (explainability)
- Sensitivity Curves
- Thermal Response Characterization

Auteur: Nadhir - Stage STEG 2026
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import GradientBoostingRegressor
import xgboost as xgb
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configuration
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 10

# Chemins
BASE_DIR = Path(__file__).parent
PLOTS_DIR = BASE_DIR / "plots"
DATA_DIR = BASE_DIR.parent / "LAST_DATA"
PLOTS_DIR.mkdir(exist_ok=True)

print("=" * 70)
print("🔵 MODULE 4 - LOAD VS TEMPERATURE COUPLING")
print("=" * 70)

# =============================================================================
# 1. CHARGEMENT DES DONNÉES
# =============================================================================
print("\n📥 Chargement des données APM Alternateur...")

df = pd.read_csv(DATA_DIR / "APM_Alternateur_10min_ML.csv")
print(f"   Dataset: {len(df):,} lignes × {len(df.columns)} colonnes")

# =============================================================================
# 2. FEATURE ENGINEERING
# =============================================================================
print("\n🔧 Feature Engineering - Load vs Temperature...")

# Variables de base
df['LOAD_MW'] = df['MODE_TAG_1']
df['REACTIVE_MVAR'] = df['REACTIVE_LOAD']

# Température stator (moyenne des phases)
temp_cols = [col for col in df.columns if 'STATOR' in col and 'TEMP' in col and 'degC' in col]
df['STATOR_TEMP'] = df[temp_cols].mean(axis=1)
df['STATOR_TEMP_MAX'] = df[temp_cols].max(axis=1)
df['STATOR_TEMP_MIN'] = df[temp_cols].min(axis=1)
df['TEMP_SPREAD'] = df['STATOR_TEMP_MAX'] - df['STATOR_TEMP_MIN']

# Température ambiante et refroidissement
df['AMBIENT_TEMP'] = df['AMBIENT_AIR_TEMP_C']
df['HOT_AIR'] = df[['ENCLOSED_HOT_AIR_TEMP_1_degC', 'ENCLOSED_HOT_AIR_TEMP_2_degC']].mean(axis=1)
df['COLD_AIR'] = df[['ENCLOSED_COLD_AIR_TEMP_1_degC', 'ENCLOSED_COLD_AIR_TEMP_2_degC']].mean(axis=1)
df['COOLING_DELTA'] = df['HOT_AIR'] - df['COLD_AIR']

# Features d'interaction
df['LOAD_SQUARED'] = df['LOAD_MW'] ** 2
df['LOAD_CUBED'] = df['LOAD_MW'] ** 3
df['LOAD_X_AMBIENT'] = df['LOAD_MW'] * df['AMBIENT_TEMP']
df['LOAD_X_COOLING'] = df['LOAD_MW'] * df['COOLING_DELTA']
df['REACTIVE_X_LOAD'] = df['REACTIVE_MVAR'] * df['LOAD_MW']
df['APPARENT_POWER'] = np.sqrt(df['LOAD_MW']**2 + df['REACTIVE_MVAR']**2)

# Température relative
df['TEMP_ABOVE_AMBIENT'] = df['STATOR_TEMP'] - df['AMBIENT_TEMP']
df['THERMAL_EFFICIENCY'] = df['TEMP_ABOVE_AMBIENT'] / (df['LOAD_MW'] + 1)

# Features temporelles
if 'Hour' in df.columns:
    df['IS_PEAK'] = df['Hour'].apply(lambda x: 1 if 9 <= x <= 18 else 0)
    df['IS_NIGHT'] = df['Hour'].apply(lambda x: 1 if x < 6 or x > 22 else 0)

# Nettoyage
df_clean = df[(df['LOAD_MW'] > 5) & (df['STATOR_TEMP'] > 30)].copy()
df_clean = df_clean.dropna(subset=['LOAD_MW', 'STATOR_TEMP', 'AMBIENT_TEMP', 'COOLING_DELTA'])
print(f"   Dataset nettoyé: {len(df_clean):,} lignes")

# =============================================================================
# 3. ANALYSE DE CORRÉLATION
# =============================================================================
print("\n📊 Analyse de Corrélation...")

# Corrélations avec la température
corr_features = ['LOAD_MW', 'REACTIVE_MVAR', 'AMBIENT_TEMP', 'COOLING_DELTA', 
                 'LOAD_SQUARED', 'LOAD_X_AMBIENT', 'APPARENT_POWER', 'HOT_AIR', 'COLD_AIR']

correlations = df_clean[corr_features + ['STATOR_TEMP']].corr()['STATOR_TEMP'].drop('STATOR_TEMP').sort_values(ascending=False)

print("   Corrélations avec STATOR_TEMP:")
for feat, corr in correlations.items():
    print(f"      {feat}: {corr:.4f}")

# =============================================================================
# 4. FEATURE INTERACTION MODELING
# =============================================================================
print("\n🔬 Feature Interaction Modeling...")

# Features pour le modèle d'interaction
interaction_features = ['LOAD_MW', 'REACTIVE_MVAR', 'AMBIENT_TEMP', 'COOLING_DELTA']
X = df_clean[interaction_features].copy()
y = df_clean['STATOR_TEMP'].copy()

# Split chronologique
train_size = int(len(X) * 0.7)
val_size = int(len(X) * 0.15)

X_train = X.iloc[:train_size]
X_val = X.iloc[train_size:train_size+val_size]
X_test = X.iloc[train_size+val_size:]
y_train = y.iloc[:train_size]
y_val = y.iloc[train_size:train_size+val_size]
y_test = y.iloc[train_size+val_size:]

print(f"   Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

# Scaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# 4.1 Modèle Polynomial (interactions)
print("\n   📐 Modèle Polynomial (degree=2)...")
poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(X_train_scaled)
X_test_poly = poly.transform(X_test_scaled)

poly_model = Ridge(alpha=1.0)
poly_model.fit(X_train_poly, y_train)
poly_pred = poly_model.predict(X_test_poly)

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
poly_r2 = r2_score(y_test, poly_pred)
poly_rmse = np.sqrt(mean_squared_error(y_test, poly_pred))
print(f"      R² = {poly_r2:.4f}, RMSE = {poly_rmse:.2f}°C")

# Coefficients d'interaction
feature_names = poly.get_feature_names_out(interaction_features)
coef_df = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': poly_model.coef_
}).sort_values('Coefficient', key=abs, ascending=False)

print("   Top interactions:")
for _, row in coef_df.head(10).iterrows():
    print(f"      {row['Feature']}: {row['Coefficient']:.4f}")

# 4.2 XGBoost pour interactions non-linéaires
print("\n   🌳 XGBoost Interaction Model...")
xgb_model = xgb.XGBRegressor(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbosity=0
)
xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
xgb_pred = xgb_model.predict(X_test)

xgb_r2 = r2_score(y_test, xgb_pred)
xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_pred))
print(f"      R² = {xgb_r2:.4f}, RMSE = {xgb_rmse:.2f}°C")

# Feature importance
importance = pd.DataFrame({
    'Feature': interaction_features,
    'Importance': xgb_model.feature_importances_
}).sort_values('Importance', ascending=False)

print("   Feature Importance (XGBoost):")
for _, row in importance.iterrows():
    print(f"      {row['Feature']}: {row['Importance']:.4f}")

# =============================================================================
# 5. SHAP ANALYSIS
# =============================================================================
print("\n🔍 SHAP Analysis...")

try:
    import shap
    
    # Créer un explainer
    explainer = shap.TreeExplainer(xgb_model)
    
    # Calculer les valeurs SHAP sur un échantillon
    n_samples = min(2000, len(X_test))
    X_shap = X_test.iloc[:n_samples]
    shap_values = explainer.shap_values(X_shap)
    
    print(f"   SHAP values calculées sur {n_samples} échantillons")
    
    # Importance SHAP
    shap_importance = np.abs(shap_values).mean(axis=0)
    shap_df = pd.DataFrame({
        'Feature': interaction_features,
        'SHAP_Importance': shap_importance
    }).sort_values('SHAP_Importance', ascending=False)
    
    print("   SHAP Feature Importance:")
    for _, row in shap_df.iterrows():
        print(f"      {row['Feature']}: {row['SHAP_Importance']:.4f}")
    
    # Sauvegarder l'explainer
    joblib.dump(explainer, PLOTS_DIR / "04_shap_explainer.pkl")
    shap_available = True
    
except ImportError:
    print("   ⚠️ SHAP non installé. Installation: pip install shap")
    shap_available = False

# =============================================================================
# 6. SENSITIVITY CURVES
# =============================================================================
print("\n📈 Génération des Sensitivity Curves...")

# Valeurs de référence (médianes)
reference = {
    'LOAD_MW': df_clean['LOAD_MW'].median(),
    'REACTIVE_MVAR': df_clean['REACTIVE_MVAR'].median(),
    'AMBIENT_TEMP': df_clean['AMBIENT_TEMP'].median(),
    'COOLING_DELTA': df_clean['COOLING_DELTA'].median()
}

# Range de variation pour chaque feature
ranges = {
    'LOAD_MW': np.linspace(df_clean['LOAD_MW'].quantile(0.05), df_clean['LOAD_MW'].quantile(0.95), 50),
    'REACTIVE_MVAR': np.linspace(df_clean['REACTIVE_MVAR'].quantile(0.05), df_clean['REACTIVE_MVAR'].quantile(0.95), 50),
    'AMBIENT_TEMP': np.linspace(df_clean['AMBIENT_TEMP'].quantile(0.05), df_clean['AMBIENT_TEMP'].quantile(0.95), 50),
    'COOLING_DELTA': np.linspace(df_clean['COOLING_DELTA'].quantile(0.05), df_clean['COOLING_DELTA'].quantile(0.95), 50)
}

# Calculer les courbes de sensibilité
sensitivity_curves = {}

for feature in interaction_features:
    curve_x = ranges[feature]
    curve_y = []
    
    for val in curve_x:
        # Créer un point avec valeurs de référence
        point = reference.copy()
        point[feature] = val
        
        # Prédire
        X_point = pd.DataFrame([point])
        pred = xgb_model.predict(X_point)[0]
        curve_y.append(pred)
    
    sensitivity_curves[feature] = (curve_x, curve_y)

# Calculer la sensibilité (dérivée)
print("   Sensibilité (ΔTemp/ΔFeature):")
for feature in interaction_features:
    curve_x, curve_y = sensitivity_curves[feature]
    sensitivity = (curve_y[-1] - curve_y[0]) / (curve_x[-1] - curve_x[0])
    print(f"      {feature}: {sensitivity:.4f} °C/unité")

# =============================================================================
# 7. THERMAL RESPONSE CHARACTERIZATION
# =============================================================================
print("\n🌡️ Caractérisation de la Réponse Thermique...")

# Régime par plage de charge
load_bins = [0, 50, 100, 150, 200]
df_clean['LOAD_BIN'] = pd.cut(df_clean['LOAD_MW'], bins=load_bins, labels=['0-50', '50-100', '100-150', '150-200'])

thermal_response = df_clean.groupby('LOAD_BIN').agg({
    'STATOR_TEMP': ['mean', 'std', 'min', 'max'],
    'TEMP_ABOVE_AMBIENT': 'mean',
    'COOLING_DELTA': 'mean',
    'LOAD_MW': 'count'
}).round(2)

print("   Réponse thermique par régime de charge:")
print(thermal_response.to_string())

# Coefficient de couplage thermique par régime
print("\n   Coefficient de couplage thermique (dT/dP) par régime:")
for regime in ['0-50', '50-100', '100-150', '150-200']:
    subset = df_clean[df_clean['LOAD_BIN'] == regime]
    if len(subset) > 10:
        corr = subset['LOAD_MW'].corr(subset['STATOR_TEMP'])
        # Régression linéaire simple
        coef = np.polyfit(subset['LOAD_MW'], subset['STATOR_TEMP'], 1)[0]
        print(f"      {regime} MW: Corr = {corr:.3f}, dT/dP = {coef:.4f} °C/MW")

# =============================================================================
# 8. VISUALISATIONS
# =============================================================================
print("\n📊 Génération des visualisations...")

fig = plt.figure(figsize=(20, 24))

# 1. Scatter Load vs Temperature avec régression
ax1 = fig.add_subplot(4, 3, 1)
sample = df_clean.sample(min(5000, len(df_clean)))
scatter = ax1.scatter(sample['LOAD_MW'], sample['STATOR_TEMP'], 
                      c=sample['AMBIENT_TEMP'], cmap='RdYlBu_r', 
                      alpha=0.5, s=10)
# Ligne de tendance
z = np.polyfit(df_clean['LOAD_MW'], df_clean['STATOR_TEMP'], 2)
p = np.poly1d(z)
x_line = np.linspace(df_clean['LOAD_MW'].min(), df_clean['LOAD_MW'].max(), 100)
ax1.plot(x_line, p(x_line), 'r-', linewidth=2, label='Tendance quadratique')
ax1.set_xlabel('Charge (MW)')
ax1.set_ylabel('Température Stator (°C)')
ax1.set_title('Load vs Temperature Coupling')
ax1.legend()
plt.colorbar(scatter, ax=ax1, label='Temp Ambiante (°C)')

# 2. Heatmap des corrélations
ax2 = fig.add_subplot(4, 3, 2)
corr_matrix = df_clean[['LOAD_MW', 'REACTIVE_MVAR', 'AMBIENT_TEMP', 'COOLING_DELTA', 
                        'STATOR_TEMP', 'TEMP_SPREAD']].corr()
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=ax2)
ax2.set_title('Matrice de Corrélation')

# 3. Feature Importance (XGBoost)
ax3 = fig.add_subplot(4, 3, 3)
importance.plot(kind='barh', x='Feature', y='Importance', ax=ax3, color='steelblue', legend=False)
ax3.set_xlabel('Importance')
ax3.set_title('XGBoost Feature Importance')
ax3.invert_yaxis()

# 4-7. Sensitivity Curves
for idx, feature in enumerate(interaction_features):
    ax = fig.add_subplot(4, 3, 4 + idx)
    curve_x, curve_y = sensitivity_curves[feature]
    ax.plot(curve_x, curve_y, 'b-', linewidth=2)
    ax.fill_between(curve_x, min(curve_y), curve_y, alpha=0.3)
    ax.axvline(reference[feature], color='r', linestyle='--', label='Référence')
    ax.set_xlabel(feature)
    ax.set_ylabel('Température Prédite (°C)')
    ax.set_title(f'Sensitivity: {feature}')
    ax.legend()
    ax.grid(True, alpha=0.3)

# 8. Polynomial Interactions
ax8 = fig.add_subplot(4, 3, 8)
top_interactions = coef_df.head(10)
colors = ['green' if c > 0 else 'red' for c in top_interactions['Coefficient']]
ax8.barh(range(len(top_interactions)), top_interactions['Coefficient'], color=colors)
ax8.set_yticks(range(len(top_interactions)))
ax8.set_yticklabels(top_interactions['Feature'], fontsize=8)
ax8.set_xlabel('Coefficient')
ax8.set_title('Top 10 Polynomial Interactions')
ax8.axvline(0, color='black', linewidth=0.5)

# 9. Réponse thermique par régime
ax9 = fig.add_subplot(4, 3, 9)
df_plot = df_clean.dropna(subset=['LOAD_BIN'])
boxprops = dict(facecolor='steelblue', alpha=0.7)
df_plot.boxplot(column='STATOR_TEMP', by='LOAD_BIN', ax=ax9, 
                patch_artist=True, boxprops=boxprops)
ax9.set_xlabel('Régime de Charge (MW)')
ax9.set_ylabel('Température Stator (°C)')
ax9.set_title('Distribution Température par Régime')
plt.suptitle('')

# 10. Prédictions vs Réel (XGBoost)
ax10 = fig.add_subplot(4, 3, 10)
ax10.scatter(y_test, xgb_pred, alpha=0.3, s=5)
ax10.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)
ax10.set_xlabel('Température Réelle (°C)')
ax10.set_ylabel('Température Prédite (°C)')
ax10.set_title(f'XGBoost: R²={xgb_r2:.4f}, RMSE={xgb_rmse:.2f}°C')

# 11. Résidus par charge
ax11 = fig.add_subplot(4, 3, 11)
residuals = y_test.values - xgb_pred
ax11.scatter(X_test['LOAD_MW'], residuals, alpha=0.3, s=5)
ax11.axhline(0, color='r', linestyle='--')
ax11.axhline(2, color='orange', linestyle='--', alpha=0.5)
ax11.axhline(-2, color='orange', linestyle='--', alpha=0.5)
ax11.set_xlabel('Charge (MW)')
ax11.set_ylabel('Résidu (°C)')
ax11.set_title('Résidus vs Charge')

# 12. SHAP Summary (si disponible)
ax12 = fig.add_subplot(4, 3, 12)
if shap_available:
    shap_df.plot(kind='barh', x='Feature', y='SHAP_Importance', ax=ax12, 
                 color='purple', legend=False)
    ax12.set_xlabel('SHAP Importance')
    ax12.set_title('SHAP Feature Importance')
    ax12.invert_yaxis()
else:
    ax12.text(0.5, 0.5, 'SHAP non disponible\nInstaller: pip install shap', 
              ha='center', va='center', fontsize=12)
    ax12.set_title('SHAP Analysis')

plt.tight_layout()
plt.savefig(PLOTS_DIR / "04_load_temperature_coupling.png", dpi=150, bbox_inches='tight')
print(f"   ✅ Sauvegardé: {PLOTS_DIR / '04_load_temperature_coupling.png'}")

# =============================================================================
# 9. SHAP DETAILED PLOTS (si disponible)
# =============================================================================
if shap_available:
    print("\n📊 Génération des SHAP plots détaillés...")
    
    fig2, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    # 1. SHAP Summary Bar
    ax = axes[0, 0]
    shap.summary_plot(shap_values, X_shap, plot_type="bar", show=False)
    plt.sca(ax)
    ax.set_title('SHAP Summary (Bar)')
    
    # 2. SHAP Summary Beeswarm
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_shap, show=False)
    plt.title('SHAP Summary (Beeswarm)')
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "04_shap_beeswarm.png", dpi=150, bbox_inches='tight')
    print(f"   ✅ Sauvegardé: {PLOTS_DIR / '04_shap_beeswarm.png'}")
    plt.close()
    
    # 3. SHAP Dependence Plots
    fig3, axes3 = plt.subplots(2, 2, figsize=(14, 12))
    
    for idx, feature in enumerate(interaction_features):
        ax = axes3[idx // 2, idx % 2]
        shap.dependence_plot(feature, shap_values, X_shap, ax=ax, show=False)
        ax.set_title(f'SHAP Dependence: {feature}')
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "04_shap_dependence.png", dpi=150, bbox_inches='tight')
    print(f"   ✅ Sauvegardé: {PLOTS_DIR / '04_shap_dependence.png'}")
    plt.close()

plt.close('all')

# =============================================================================
# 10. SAUVEGARDE DES MODÈLES
# =============================================================================
print("\n💾 Sauvegarde des modèles et configurations...")

# Modèle XGBoost
joblib.dump(xgb_model, PLOTS_DIR / "04_xgb_coupling_model.pkl")

# Scaler
joblib.dump(scaler, PLOTS_DIR / "04_coupling_scaler.pkl")

# Configuration
config = {
    'features': interaction_features,
    'reference_values': reference,
    'sensitivity_curves': sensitivity_curves,
    'xgb_r2': xgb_r2,
    'xgb_rmse': xgb_rmse,
    'poly_r2': poly_r2,
    'poly_rmse': poly_rmse,
    'correlations': correlations.to_dict(),
    'shap_available': shap_available
}
joblib.dump(config, PLOTS_DIR / "04_coupling_config.pkl")

# Interactions polynomiales
coef_df.to_csv(PLOTS_DIR / "04_polynomial_interactions.csv", index=False)

print(f"   ✅ XGBoost Model: {PLOTS_DIR / '04_xgb_coupling_model.pkl'}")
print(f"   ✅ Config: {PLOTS_DIR / '04_coupling_config.pkl'}")
print(f"   ✅ Interactions: {PLOTS_DIR / '04_polynomial_interactions.csv'}")

# =============================================================================
# RÉSUMÉ
# =============================================================================
print("\n" + "=" * 70)
print("📊 RÉSUMÉ - LOAD VS TEMPERATURE COUPLING")
print("=" * 70)

print(f"""
╔══════════════════════════════════════════════════════════════════════╗
║              LOAD VS TEMPERATURE COUPLING ANALYSIS                   ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  📐 POLYNOMIAL INTERACTION MODEL                                     ║
║     • R² Score: {poly_r2:.4f}                                        ║
║     • RMSE: {poly_rmse:.2f}°C                                        ║
║                                                                      ║
║  🌳 XGBOOST INTERACTION MODEL                                        ║
║     • R² Score: {xgb_r2:.4f}                                         ║
║     • RMSE: {xgb_rmse:.2f}°C                                         ║
║                                                                      ║
╠══════════════════════════════════════════════════════════════════════╣
║  🔍 KEY CORRELATIONS WITH TEMPERATURE:                               ║
""")

for feat, corr in correlations.head(5).items():
    print(f"║     • {feat}: {corr:.4f}")

print(f"""║                                                                      ║
╠══════════════════════════════════════════════════════════════════════╣
║  📈 SENSITIVITY ANALYSIS:                                            ║
""")

for feature in interaction_features:
    curve_x, curve_y = sensitivity_curves[feature]
    sensitivity = (curve_y[-1] - curve_y[0]) / (curve_x[-1] - curve_x[0])
    print(f"║     • {feature}: {sensitivity:+.4f} °C/unit")

print(f"""║                                                                      ║
╠══════════════════════════════════════════════════════════════════════╣
║  🏆 TOP POLYNOMIAL INTERACTIONS:                                     ║
""")

for _, row in coef_df.head(5).iterrows():
    print(f"║     • {row['Feature']}: {row['Coefficient']:.4f}")

print(f"""║                                                                      ║
╠══════════════════════════════════════════════════════════════════════╣
║  🔬 SHAP ANALYSIS: {'✅ Disponible' if shap_available else '❌ Non installé'}                                      ║
╚══════════════════════════════════════════════════════════════════════╝
""")

print("=" * 70)
