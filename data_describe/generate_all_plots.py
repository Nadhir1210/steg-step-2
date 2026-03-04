#!/usr/bin/env python3
"""
Script pour générer et sauvegarder tous les plots des datasets
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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
print("GÉNÉRATION DES PLOTS - ANALYSE EDA")
print("=" * 80)

# ============================================================================
# 1. APM ALTERNATEUR (1-min)
# ============================================================================
print("\n[1/6] APM Alternateur (1-min)...")
df = pd.read_csv(DATA_DIR / "APM_Alternateur_ML.csv")

# Distribution variables principales
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()
main_vars = ['MODE_TAG_1', 'REACTIVE_LOAD', 'SPEED_CTRL_pct', 'TERMINAL_VOLTAGE_kV', 'FREQUENCY_Hz']
for i, var in enumerate(main_vars):
    ax = axes[i]
    sns.histplot(df[var], kde=True, ax=ax, color='steelblue', alpha=0.7)
    ax.set_title(f'Distribution: {var}', fontsize=11, fontweight='bold')
    ax.axvline(df[var].mean(), color='red', linestyle='--', label=f'Moy: {df[var].mean():.2f}')
    ax.legend(fontsize=8)
axes[-1].set_visible(False)
plt.suptitle('APM Alternateur - Distribution des Variables Principales', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "01_apm_alternateur_distribution.png", dpi=150, bbox_inches='tight')
plt.close()

# Températures stator
temp_cols = [col for col in df.columns if 'STATOR_PHASE' in col and 'WINDING_TEMP' in col]
if temp_cols:
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()
    for i, col in enumerate(temp_cols[:9]):
        ax = axes[i]
        sns.histplot(df[col], kde=True, ax=ax, color='coral', alpha=0.7)
        ax.set_title(col.replace('STATOR_', '').replace('_degC', ''), fontsize=9)
    plt.suptitle('APM Alternateur - Températures du Stator', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "01_apm_alternateur_temperatures.png", dpi=150, bbox_inches='tight')
    plt.close()

# Matrice de corrélation
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
exclude_cols = ['Year', 'Month', 'Day', 'Hour', 'Minute', 'DayOfWeek', 'Quarter']
numeric_cols = [c for c in numeric_cols if c not in exclude_cols]
corr = df[numeric_cols].corr()
plt.figure(figsize=(16, 14))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            square=True, linewidths=0.5, annot_kws={'size': 7})
plt.title('APM Alternateur - Matrice de Corrélation', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(PLOTS_DIR / "01_apm_alternateur_correlation.png", dpi=150, bbox_inches='tight')
plt.close()

# Box plots
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()
boxplot_vars = ['MODE_TAG_1', 'REACTIVE_LOAD', 'TERMINAL_VOLTAGE_kV', 
                'FREQUENCY_Hz', 'AMBIENT_AIR_TEMP_C', 'SPEED_CTRL_pct']
sample = df.sample(n=min(50000, len(df)), random_state=42)
for i, var in enumerate(boxplot_vars):
    if var in df.columns:
        axes[i].boxplot(sample[var].dropna(), patch_artist=True,
                        boxprops=dict(facecolor='lightblue'))
        axes[i].set_title(var, fontsize=11, fontweight='bold')
plt.suptitle('APM Alternateur - Box Plots', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "01_apm_alternateur_boxplots.png", dpi=150, bbox_inches='tight')
plt.close()

print("  ✓ 4 plots générés")

# ============================================================================
# 2. APM ALTERNATEUR (10-min)
# ============================================================================
print("[2/6] APM Alternateur (10-min)...")
df = pd.read_csv(DATA_DIR / "APM_Alternateur_10min_ML.csv")

# Distribution
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()
for i, var in enumerate(main_vars):
    if var in df.columns:
        ax = axes[i]
        sns.histplot(df[var], kde=True, ax=ax, color='steelblue', alpha=0.7)
        ax.set_title(f'Distribution: {var}', fontsize=11, fontweight='bold')
axes[-1].set_visible(False)
plt.suptitle('APM Alternateur 10min - Distribution', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "02_apm_alternateur_10min_distribution.png", dpi=150, bbox_inches='tight')
plt.close()

# Corrélation
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols = [c for c in numeric_cols if c not in exclude_cols]
corr = df[numeric_cols].corr()
plt.figure(figsize=(16, 14))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r', center=0, square=True, linewidths=0.5, annot_kws={'size': 7})
plt.title('APM Alternateur 10min - Matrice de Corrélation', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(PLOTS_DIR / "02_apm_alternateur_10min_correlation.png", dpi=150, bbox_inches='tight')
plt.close()

print("  ✓ 2 plots générés")

# ============================================================================
# 3. APM CHART (1-min)
# ============================================================================
print("[3/6] APM Chart (1-min)...")
df = pd.read_csv(DATA_DIR / "APM_Chart_ML.csv")

# Distribution
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()
for i, var in enumerate(main_vars):
    if var in df.columns:
        ax = axes[i]
        sns.histplot(df[var], kde=True, ax=ax, color='forestgreen', alpha=0.7)
        ax.set_title(f'Distribution: {var}', fontsize=11, fontweight='bold')
axes[-1].set_visible(False)
plt.suptitle('APM Chart - Distribution', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "03_apm_chart_distribution.png", dpi=150, bbox_inches='tight')
plt.close()

# Corrélation
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols = [c for c in numeric_cols if c not in exclude_cols]
corr = df[numeric_cols].corr()
plt.figure(figsize=(16, 14))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r', center=0, square=True, linewidths=0.5, annot_kws={'size': 7})
plt.title('APM Chart - Matrice de Corrélation', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(PLOTS_DIR / "03_apm_chart_correlation.png", dpi=150, bbox_inches='tight')
plt.close()

# Températures
temp_cols = [col for col in df.columns if 'STATOR_PHASE' in col and 'WINDING_TEMP' in col]
if temp_cols:
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()
    for i, col in enumerate(temp_cols[:9]):
        ax = axes[i]
        sns.histplot(df[col], kde=True, ax=ax, color='coral', alpha=0.7)
        ax.set_title(col.replace('STATOR_', '').replace('_degC', ''), fontsize=9)
    plt.suptitle('APM Chart - Températures du Stator', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "03_apm_chart_temperatures.png", dpi=150, bbox_inches='tight')
    plt.close()

print("  ✓ 3 plots générés")

# ============================================================================
# 4. APM CHART (10-min)
# ============================================================================
print("[4/6] APM Chart (10-min)...")
df = pd.read_csv(DATA_DIR / "APM_Chart_10min_ML.csv")

# Distribution
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()
for i, var in enumerate(main_vars):
    if var in df.columns:
        ax = axes[i]
        sns.histplot(df[var], kde=True, ax=ax, color='forestgreen', alpha=0.7)
        ax.set_title(f'Distribution: {var}', fontsize=11, fontweight='bold')
axes[-1].set_visible(False)
plt.suptitle('APM Chart 10min - Distribution', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "04_apm_chart_10min_distribution.png", dpi=150, bbox_inches='tight')
plt.close()

# Corrélation
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols = [c for c in numeric_cols if c not in exclude_cols]
corr = df[numeric_cols].corr()
plt.figure(figsize=(16, 14))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r', center=0, square=True, linewidths=0.5, annot_kws={'size': 7})
plt.title('APM Chart 10min - Matrice de Corrélation', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(PLOTS_DIR / "04_apm_chart_10min_correlation.png", dpi=150, bbox_inches='tight')
plt.close()

print("  ✓ 2 plots générés")

# ============================================================================
# 5. TG1 SOUSSE (original)
# ============================================================================
print("[5/6] TG1 Sousse...")
df = pd.read_csv(DATA_DIR / "TG1_Sousse_ML.csv")

# Distribution des variables
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols = [c for c in numeric_cols if c not in exclude_cols][:6]

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()
for i, var in enumerate(numeric_cols):
    ax = axes[i]
    sns.histplot(df[var], kde=True, ax=ax, color='purple', alpha=0.7)
    ax.set_title(f'Distribution: {var}', fontsize=10, fontweight='bold')
plt.suptitle('TG1 Sousse - Distribution', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "05_tg1_sousse_distribution.png", dpi=150, bbox_inches='tight')
plt.close()

# Corrélation
numeric_cols_all = df.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols_all = [c for c in numeric_cols_all if c not in exclude_cols]
corr = df[numeric_cols_all].corr()
plt.figure(figsize=(16, 14))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r', center=0, square=True, linewidths=0.5, annot_kws={'size': 7})
plt.title('TG1 Sousse - Matrice de Corrélation', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(PLOTS_DIR / "05_tg1_sousse_correlation.png", dpi=150, bbox_inches='tight')
plt.close()

print("  ✓ 2 plots générés")

# ============================================================================
# 6. TG1 SOUSSE (1-min)
# ============================================================================
print("[6/6] TG1 Sousse (1-min)...")
df = pd.read_csv(DATA_DIR / "TG1_Sousse_1min_ML.csv")

# Distribution
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols = [c for c in numeric_cols if c not in exclude_cols][:6]

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()
for i, var in enumerate(numeric_cols):
    ax = axes[i]
    sns.histplot(df[var], kde=True, ax=ax, color='purple', alpha=0.7)
    ax.set_title(f'Distribution: {var}', fontsize=10, fontweight='bold')
plt.suptitle('TG1 Sousse 1min - Distribution', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "06_tg1_sousse_1min_distribution.png", dpi=150, bbox_inches='tight')
plt.close()

# Corrélation
numeric_cols_all = df.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols_all = [c for c in numeric_cols_all if c not in exclude_cols]
corr = df[numeric_cols_all].corr()
plt.figure(figsize=(16, 14))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r', center=0, square=True, linewidths=0.5, annot_kws={'size': 7})
plt.title('TG1 Sousse 1min - Matrice de Corrélation', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(PLOTS_DIR / "06_tg1_sousse_1min_correlation.png", dpi=150, bbox_inches='tight')
plt.close()

print("  ✓ 2 plots générés")

# ============================================================================
# RÉSUMÉ
# ============================================================================
print("\n" + "=" * 80)
print("GÉNÉRATION TERMINÉE!")
print("=" * 80)
print(f"Plots sauvegardés dans: {PLOTS_DIR}")
print(f"Total: {len(list(PLOTS_DIR.glob('*.png')))} images PNG")
