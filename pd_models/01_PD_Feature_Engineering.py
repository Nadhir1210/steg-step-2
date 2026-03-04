#!/usr/bin/env python3
"""
🔵 01 - Feature Engineering PD (Décharges Partielles)
======================================================
Création de variables intelligentes à partir des données PD brutes.

Variables créées:
- PD_INTENSITY: Intensité des décharges (CURRENT × PULSE_COUNT)
- PD_ENERGY: Énergie des décharges (MEAN_CHARGE × DISCHARGE_RATE)
- CHANNEL_ASYMMETRY: Asymétrie entre les 4 canaux
- Rolling features: Moyennes mobiles, max, std

⚠️ CETTE ÉTAPE EST OBLIGATOIRE AVANT TOUT AUTRE MODÈLE
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
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
print("🔵 FEATURE ENGINEERING PD - Décharges Partielles")
print("=" * 80)

# ============================================================================
# 1. CHARGEMENT DES DONNÉES
# ============================================================================
print("\n[1/5] Chargement des données PD...")

df = pd.read_csv(DATA_DIR / "TG1_Sousse_ML.csv")
print(f"   ✓ Dataset: {df.shape[0]:,} lignes × {df.shape[1]} colonnes")

# Parser datetime
if 'Datetime' in df.columns:
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df = df.sort_values('Datetime').reset_index(drop=True)

# ============================================================================
# 2. IDENTIFICATION DES COLONNES PD
# ============================================================================
print("\n[2/5] Identification des colonnes PD...")

# Canaux disponibles
CHANNELS = ['CH1', 'CH2', 'CH3', 'CH4']

# Types de mesures par canal
MEASURES = {
    'CURRENT_ABS': '_CURRENT_ABS_uA',
    'CURRENT_NEG': '_CURRENT_NEG_uA',
    'CURRENT_POS': '_CURRENT_POS_uA',
    'DISCHARGE_RATE': '_DISCHARGE_RATE_ABS_nC2_per_s',
    'MAX_CHARGE': '_MAX_CHARGE_ABS_nC',
    'MEAN_CHARGE': '_MEAN_CHARGE_ABS_nC',
    'PULSE_COUNT': '_PULSE_COUNT_ABS'
}

# Vérifier les colonnes existantes
pd_cols = []
for ch in CHANNELS:
    for measure, suffix in MEASURES.items():
        col_name = f"{ch}{suffix}"
        if col_name in df.columns:
            pd_cols.append(col_name)

print(f"   ✓ Colonnes PD trouvées: {len(pd_cols)}")

# ============================================================================
# 3. CRÉATION DES FEATURES INTELLIGENTES
# ============================================================================
print("\n[3/5] Création des features intelligentes...")

# -------------------------------------------------------------------------
# 3.1 PD INTENSITY INDEX (par canal)
# PD_INTENSITY = CURRENT_ABS × PULSE_COUNT
# -------------------------------------------------------------------------
print("   → PD Intensity Index...")
for ch in CHANNELS:
    current_col = f"{ch}_CURRENT_ABS_uA"
    pulse_col = f"{ch}_PULSE_COUNT_ABS"
    
    if current_col in df.columns and pulse_col in df.columns:
        df[f'{ch}_PD_INTENSITY'] = df[current_col] * df[pulse_col]
        df[f'{ch}_PD_INTENSITY'] = df[f'{ch}_PD_INTENSITY'].fillna(0)

# PD Intensity Total (somme des 4 canaux)
intensity_cols = [f'{ch}_PD_INTENSITY' for ch in CHANNELS if f'{ch}_PD_INTENSITY' in df.columns]
if intensity_cols:
    df['PD_INTENSITY_TOTAL'] = df[intensity_cols].sum(axis=1)
    df['PD_INTENSITY_MEAN'] = df[intensity_cols].mean(axis=1)
    df['PD_INTENSITY_MAX'] = df[intensity_cols].max(axis=1)
    print(f"      ✓ PD_INTENSITY créé pour {len(intensity_cols)} canaux")

# -------------------------------------------------------------------------
# 3.2 PD ENERGY INDEX (par canal)
# PD_ENERGY = MEAN_CHARGE × DISCHARGE_RATE
# -------------------------------------------------------------------------
print("   → PD Energy Index...")
for ch in CHANNELS:
    charge_col = f"{ch}_MEAN_CHARGE_ABS_nC"
    discharge_col = f"{ch}_DISCHARGE_RATE_ABS_nC2_per_s"
    
    if charge_col in df.columns and discharge_col in df.columns:
        df[f'{ch}_PD_ENERGY'] = df[charge_col] * df[discharge_col]
        df[f'{ch}_PD_ENERGY'] = df[f'{ch}_PD_ENERGY'].fillna(0)

# PD Energy Total
energy_cols = [f'{ch}_PD_ENERGY' for ch in CHANNELS if f'{ch}_PD_ENERGY' in df.columns]
if energy_cols:
    df['PD_ENERGY_TOTAL'] = df[energy_cols].sum(axis=1)
    df['PD_ENERGY_MEAN'] = df[energy_cols].mean(axis=1)
    df['PD_ENERGY_MAX'] = df[energy_cols].max(axis=1)
    print(f"      ✓ PD_ENERGY créé pour {len(energy_cols)} canaux")

# -------------------------------------------------------------------------
# 3.3 CHANNEL ASYMMETRY
# ASYMMETRY = max(CH1,CH2,CH3,CH4) - min(CH1,CH2,CH3,CH4)
# -------------------------------------------------------------------------
print("   → Channel Asymmetry...")

# Asymétrie de l'intensité
if intensity_cols:
    df['INTENSITY_ASYMMETRY'] = df[intensity_cols].max(axis=1) - df[intensity_cols].min(axis=1)
    df['INTENSITY_STD'] = df[intensity_cols].std(axis=1)

# Asymétrie de l'énergie
if energy_cols:
    df['ENERGY_ASYMMETRY'] = df[energy_cols].max(axis=1) - df[energy_cols].min(axis=1)
    df['ENERGY_STD'] = df[energy_cols].std(axis=1)

# Asymétrie du courant
current_cols = [f'{ch}_CURRENT_ABS_uA' for ch in CHANNELS if f'{ch}_CURRENT_ABS_uA' in df.columns]
if current_cols:
    df['CURRENT_ASYMMETRY'] = df[current_cols].max(axis=1) - df[current_cols].min(axis=1)
    df['CURRENT_TOTAL'] = df[current_cols].sum(axis=1)

# Asymétrie des pulses
pulse_cols = [f'{ch}_PULSE_COUNT_ABS' for ch in CHANNELS if f'{ch}_PULSE_COUNT_ABS' in df.columns]
if pulse_cols:
    df['PULSE_ASYMMETRY'] = df[pulse_cols].max(axis=1) - df[pulse_cols].min(axis=1)
    df['PULSE_TOTAL'] = df[pulse_cols].sum(axis=1)

print("      ✓ Asymétries calculées")

# -------------------------------------------------------------------------
# 3.4 ROLLING FEATURES (Moyennes mobiles)
# -------------------------------------------------------------------------
print("   → Rolling Features...")

# Définir les fenêtres de temps (en nombre d'observations)
# Supposons 1 observation par minute
WINDOW_10MIN = 10
WINDOW_30MIN = 30
WINDOW_1H = 60

# Rolling sur PD_INTENSITY_TOTAL
if 'PD_INTENSITY_TOTAL' in df.columns:
    # Rolling mean
    df['PD_INTENSITY_ROLL_MEAN_10min'] = df['PD_INTENSITY_TOTAL'].rolling(window=WINDOW_10MIN, min_periods=1).mean()
    df['PD_INTENSITY_ROLL_MEAN_30min'] = df['PD_INTENSITY_TOTAL'].rolling(window=WINDOW_30MIN, min_periods=1).mean()
    df['PD_INTENSITY_ROLL_MEAN_1h'] = df['PD_INTENSITY_TOTAL'].rolling(window=WINDOW_1H, min_periods=1).mean()
    
    # Rolling max
    df['PD_INTENSITY_ROLL_MAX_1h'] = df['PD_INTENSITY_TOTAL'].rolling(window=WINDOW_1H, min_periods=1).max()
    
    # Rolling std
    df['PD_INTENSITY_ROLL_STD_30min'] = df['PD_INTENSITY_TOTAL'].rolling(window=WINDOW_30MIN, min_periods=1).std()
    
    # Trend (différence avec rolling mean)
    df['PD_INTENSITY_TREND'] = df['PD_INTENSITY_TOTAL'] - df['PD_INTENSITY_ROLL_MEAN_1h']
    df['PD_INTENSITY_TREND'] = df['PD_INTENSITY_TREND'].fillna(0)

# Rolling sur PD_ENERGY_TOTAL
if 'PD_ENERGY_TOTAL' in df.columns:
    df['PD_ENERGY_ROLL_MEAN_10min'] = df['PD_ENERGY_TOTAL'].rolling(window=WINDOW_10MIN, min_periods=1).mean()
    df['PD_ENERGY_ROLL_MEAN_1h'] = df['PD_ENERGY_TOTAL'].rolling(window=WINDOW_1H, min_periods=1).mean()
    df['PD_ENERGY_ROLL_MAX_1h'] = df['PD_ENERGY_TOTAL'].rolling(window=WINDOW_1H, min_periods=1).max()

print("      ✓ Rolling features calculées")

# -------------------------------------------------------------------------
# 3.5 RATIOS ET INDICATEURS AVANCÉS
# -------------------------------------------------------------------------
print("   → Ratios et indicateurs avancés...")

# Ratio Énergie/Intensité
if 'PD_ENERGY_TOTAL' in df.columns and 'PD_INTENSITY_TOTAL' in df.columns:
    df['ENERGY_INTENSITY_RATIO'] = df['PD_ENERGY_TOTAL'] / (df['PD_INTENSITY_TOTAL'] + 1e-10)

# Ratio Charge Max / Charge Mean (par canal)
for ch in CHANNELS:
    max_col = f"{ch}_MAX_CHARGE_ABS_nC"
    mean_col = f"{ch}_MEAN_CHARGE_ABS_nC"
    if max_col in df.columns and mean_col in df.columns:
        df[f'{ch}_CHARGE_RATIO'] = df[max_col] / (df[mean_col] + 1e-10)

# Coefficient de variation de l'intensité
if 'PD_INTENSITY_MEAN' in df.columns and 'INTENSITY_STD' in df.columns:
    df['INTENSITY_CV'] = df['INTENSITY_STD'] / (df['PD_INTENSITY_MEAN'] + 1e-10)

print("      ✓ Ratios calculés")

# ============================================================================
# 4. STATISTIQUES DES NOUVELLES FEATURES
# ============================================================================
print("\n[4/5] Statistiques des nouvelles features...")

new_features = [
    'PD_INTENSITY_TOTAL', 'PD_INTENSITY_MEAN', 'PD_INTENSITY_MAX',
    'PD_ENERGY_TOTAL', 'PD_ENERGY_MEAN', 'PD_ENERGY_MAX',
    'INTENSITY_ASYMMETRY', 'ENERGY_ASYMMETRY', 'CURRENT_ASYMMETRY',
    'PD_INTENSITY_TREND', 'INTENSITY_CV'
]

existing_features = [f for f in new_features if f in df.columns]

print("\n   📊 STATISTIQUES DES FEATURES PD:")
print("   " + "-" * 70)
print(f"   {'Feature':<35} {'Mean':>12} {'Std':>12} {'Max':>12}")
print("   " + "-" * 70)

for feat in existing_features:
    mean_val = df[feat].mean()
    std_val = df[feat].std()
    max_val = df[feat].max()
    print(f"   {feat:<35} {mean_val:>12.2f} {std_val:>12.2f} {max_val:>12.2f}")

# ============================================================================
# 5. VISUALISATIONS ET SAUVEGARDE
# ============================================================================
print("\n[5/5] Visualisations et sauvegarde...")

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 1. Distribution PD_INTENSITY_TOTAL
ax1 = axes[0, 0]
if 'PD_INTENSITY_TOTAL' in df.columns:
    ax1.hist(df['PD_INTENSITY_TOTAL'].dropna(), bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    ax1.set_xlabel('PD Intensity Total')
    ax1.set_ylabel('Fréquence')
    ax1.set_title('Distribution PD Intensity')
    ax1.set_yscale('log')

# 2. Distribution PD_ENERGY_TOTAL
ax2 = axes[0, 1]
if 'PD_ENERGY_TOTAL' in df.columns:
    ax2.hist(df['PD_ENERGY_TOTAL'].dropna(), bins=50, color='coral', edgecolor='black', alpha=0.7)
    ax2.set_xlabel('PD Energy Total')
    ax2.set_ylabel('Fréquence')
    ax2.set_title('Distribution PD Energy')
    ax2.set_yscale('log')

# 3. Asymétrie entre canaux
ax3 = axes[0, 2]
if 'INTENSITY_ASYMMETRY' in df.columns:
    ax3.hist(df['INTENSITY_ASYMMETRY'].dropna(), bins=50, color='green', edgecolor='black', alpha=0.7)
    ax3.set_xlabel('Intensity Asymmetry')
    ax3.set_ylabel('Fréquence')
    ax3.set_title('Asymétrie entre Canaux')
    ax3.set_yscale('log')

# 4. Comparaison des 4 canaux (Intensité)
ax4 = axes[1, 0]
intensity_data = [df[f'{ch}_PD_INTENSITY'].dropna().values for ch in CHANNELS if f'{ch}_PD_INTENSITY' in df.columns]
if intensity_data:
    bp = ax4.boxplot(intensity_data, labels=CHANNELS[:len(intensity_data)])
    ax4.set_ylabel('PD Intensity')
    ax4.set_title('Comparaison Intensité par Canal')
    ax4.set_yscale('log')

# 5. Évolution temporelle (échantillon)
ax5 = axes[1, 1]
if 'Datetime' in df.columns and 'PD_INTENSITY_TOTAL' in df.columns:
    sample = df.iloc[::100]  # Sous-échantillon
    ax5.plot(sample['Datetime'], sample['PD_INTENSITY_TOTAL'], alpha=0.7, linewidth=0.5)
    ax5.set_xlabel('Temps')
    ax5.set_ylabel('PD Intensity Total')
    ax5.set_title('Évolution Temporelle')
    ax5.tick_params(axis='x', rotation=45)

# 6. Corrélation Intensity vs Energy
ax6 = axes[1, 2]
if 'PD_INTENSITY_TOTAL' in df.columns and 'PD_ENERGY_TOTAL' in df.columns:
    sample = df.sample(min(10000, len(df)))
    ax6.scatter(sample['PD_INTENSITY_TOTAL'], sample['PD_ENERGY_TOTAL'], 
                alpha=0.3, s=5, c='purple')
    ax6.set_xlabel('PD Intensity')
    ax6.set_ylabel('PD Energy')
    ax6.set_title('Intensité vs Énergie')
    ax6.set_xscale('log')
    ax6.set_yscale('log')

plt.tight_layout()
plt.savefig(PLOTS_DIR / "01_feature_engineering.png", dpi=150, bbox_inches='tight')
plt.close()
print("   ✓ 01_feature_engineering.png")

# Sauvegarder le dataset enrichi
output_path = DATA_DIR / "TG1_Sousse_PD_Features.csv"
df.to_csv(output_path, index=False)
print(f"   ✓ Dataset enrichi sauvegardé: {output_path.name}")
print(f"      → {df.shape[0]:,} lignes × {df.shape[1]} colonnes")

# Lister les nouvelles colonnes
all_new_cols = [c for c in df.columns if any(x in c for x in ['PD_INTENSITY', 'PD_ENERGY', 'ASYMMETRY', 'ROLL', 'TREND', 'RATIO', 'CV'])]
print(f"\n   📦 Nouvelles features créées: {len(all_new_cols)}")

# Sauvegarder la liste des features
feature_info = {
    'intensity_cols': intensity_cols if intensity_cols else [],
    'energy_cols': energy_cols if energy_cols else [],
    'new_features': all_new_cols,
    'channels': CHANNELS
}
joblib.dump(feature_info, PLOTS_DIR / "01_feature_info.pkl")

print("\n" + "=" * 80)
print("✅ FEATURE ENGINEERING PD - TERMINÉ")
print("=" * 80)
print(f"""
📊 Features créées:
   • PD_INTENSITY (par canal + total)
   • PD_ENERGY (par canal + total)
   • Asymétries (intensité, énergie, courant, pulses)
   • Rolling features (10min, 30min, 1h)
   • Ratios et indicateurs avancés

📁 Fichiers générés:
   • {PLOTS_DIR / '01_feature_engineering.png'}
   • {output_path}
   • {PLOTS_DIR / '01_feature_info.pkl'}
""")
