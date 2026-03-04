# 🔥 TG1 Digital Twin - Health Monitoring System

> **Système complet de monitoring industriel pour le Turbo-Alternateur TG1 - STEG**

![Version](https://img.shields.io/badge/Version-1.0-blue)
![Python](https://img.shields.io/badge/Python-3.11-green)
![Status](https://img.shields.io/badge/Status-Production-brightgreen)

## 📊 Vue d'ensemble

Ce système implémente un **Digital Twin** pour le monitoring en temps réel du turbo-alternateur TG1. Il combine plusieurs axes d'analyse pour fournir un **Global Health Index** complet.

### 🎯 Formule du Health Index

```
TG1_HEALTH = 30% × PD + 30% × Thermal + 20% × Cooling + 20% × Electrical
```

| Score | État | Action |
|-------|------|--------|
| 🟢 85-100 | Excellent | Fonctionnement optimal |
| 🟡 70-84 | Stable | Surveillance normale |
| 🟠 50-69 | Degrading | Maintenance préventive |
| 🔴 0-49 | Critical | Intervention immédiate |

---

## 📁 Structure du Projet

```
tg1_monitoring/
├── 01_Thermal_Health_Model.py      # Modélisation thermique + anomalies
├── 02_Cooling_Efficiency.py        # Efficacité du refroidissement
├── 03_Electrical_Stability.py      # Stabilité électrique
├── 04_Load_Temperature_Coupling.py # ⭐ Couplage Charge-Température + SHAP
├── 05_Global_Health_Index.py       # Score de santé global
├── app_streamlit.py                # Dashboard interactif
├── README.md                        # Cette documentation
└── plots/                           # Visualisations et modèles
```

---

## 🔵 Module 1 - Thermal Health Model

### Objectif
Modéliser le comportement thermique normal et détecter les anomalies.

### Approche
1. **Thermal Baseline Model**: `Temp_Stator = f(Load, Reactive, Ambient, Cooling)`
2. **Residual Analysis**: `Residual = Temp_réelle - Temp_prédite`
3. **Anomaly Detection**: Seuils 2σ (Warning) et 3σ (Critical)

### Modèle
- **Algorithme**: XGBoost Regressor
- **Features**: Load MW, Reactive MVAR, Ambient Temp, Cooling ΔT, Hour
- **Validation**: Chronologique (70/15/15)

### Métriques Attendues
- R² > 0.85
- RMSE < 5°C
- MAE < 3°C

---

## 🔵 Module 2 - Cooling Efficiency

### Objectif
Monitorer l'efficacité du système de refroidissement.

### Indicateurs
1. **ΔT = Hot Air - Cold Air**: Efficacité du refroidissement
2. **Control Chart (SPC)**: Limites UCL/LCL (±3σ)
3. **Cooling Efficiency Index**: Ratio ΔT_réel / ΔT_attendu

### Méthodes
- Statistical Process Control (SPC)
- Isolation Forest Anomaly Detection
- Régression linéaire pour baseline

### Seuils
- **Normal**: 90-110% d'efficacité
- **Warning**: 70-90% ou 110-130%
- **Critical**: <70%

---

## 🔵 Module 3 - Electrical Stability

### Objectif
Analyser la stabilité électrique du turbo-alternateur.

### Indicateurs
1. **Fréquence**: Écart par rapport à 50 Hz nominal
2. **Tension**: Écart par rapport à 15.75 kV nominal
3. **Facteur de Puissance**: cos(φ) = P / √(P² + Q²)

### Seuils Fréquence
- Excellent: ±0.2 Hz
- Normal: ±0.5 Hz
- Warning: ±1.0 Hz
- Critical: >±1.0 Hz

### Seuils Tension
- Excellent: ±2%
- Normal: ±5%
- Warning: ±10%
- Critical: >±10%

---

## 🔵 Module 4 - Load vs Temperature Coupling

### Objectif
Analyser le couplage entre la charge (MW) et la température du stator.

### Méthodes
1. **Feature Interaction Modeling**: Interactions polynomiales et non-linéaires
2. **SHAP Analysis**: Explication des prédictions avec SHAP values
3. **Sensitivity Curves**: Courbes de sensibilité température vs features

### Corrélations Clés
| Variable | Corrélation avec Temp |
|----------|----------------------|
| HOT_AIR | 0.95 |
| COLD_AIR | 0.87 |
| COOLING_DELTA | 0.85 |
| LOAD_MW | 0.60 |

### Sensibilité
| Feature | Sensibilité (°C/unit) |
|---------|----------------------|
| COOLING_DELTA | +2.67 |
| LOAD_MW | -0.17 |
| AMBIENT_TEMP | -0.11 |

### SHAP Feature Importance
| Feature | SHAP Importance |
|---------|----------------|
| COOLING_DELTA | 4.98 |
| LOAD_MW | 1.42 |
| AMBIENT_TEMP | 1.03 |
| REACTIVE_MVAR | 0.82 |

### Fichiers Générés
- `plots/04_load_temperature_coupling.png`
- `plots/04_shap_beeswarm.png`
- `plots/04_shap_dependence.png`
- `plots/04_xgb_coupling_model.pkl`

---

## ⭐ Module 5 - Global Health Index

### Objectif
Combiner tous les indicateurs en un score unique 0-100.

### Pondérations
| Composant | Poids | Justification |
|-----------|-------|---------------|
| PD (Partial Discharge) | 30% | Indicateur critique isolation |
| Thermal | 30% | Risque surchauffe stator |
| Cooling | 20% | Efficacité ventilation |
| Electrical | 20% | Stabilité réseau |

### Classification
```
if score >= 85: "Excellent"  → 🟢 Aucune action
elif score >= 70: "Stable"   → 🟡 Surveillance normale
elif score >= 50: "Degrading"→ 🟠 Planifier maintenance
else: "Critical"             → 🔴 Intervention immédiate
```

---

## 🚀 Quick Start

### 1. Exécuter le pipeline complet

```bash
cd "c:\Users\Nadhir bh\Documents\stage\stage   step  2"

# Activer l'environnement
.\.venv\Scripts\activate

# Exécuter les modules
python tg1_monitoring/01_Thermal_Health_Model.py
python tg1_monitoring/02_Cooling_Efficiency.py
python tg1_monitoring/03_Electrical_Stability.py
python tg1_monitoring/04_Load_Temperature_Coupling.py
python tg1_monitoring/05_Global_Health_Index.py
```

### 2. Lancer le Dashboard

```bash
python -m streamlit run tg1_monitoring/app_streamlit.py
```

---

## 📊 Données Sources

### APM Alternateur (10-min)
- **Fichier**: `LAST_DATA/APM_Alternateur_10min_ML.csv`
- **Lignes**: ~52,000
- **Variables**: 26 colonnes

| Variable | Description | Unité |
|----------|-------------|-------|
| MODE_TAG_1 | Puissance active | MW |
| REACTIVE_LOAD | Puissance réactive | MVAR |
| STATOR_PHASE_X_TEMP | Températures stator | °C |
| TERMINAL_VOLTAGE_kV | Tension | kV |
| FREQUENCY_Hz | Fréquence | Hz |
| ENCLOSED_HOT_AIR_TEMP | Air chaud | °C |
| ENCLOSED_COLD_AIR_TEMP | Air froid | °C |

### TG1 Sousse PD (15K/2.2M lignes)
- **Fichier**: `LAST_DATA/TG1_Sousse_ML.csv`
- **Canaux**: 4 canaux de mesure PD
- **Variables**: CURRENT, DISCHARGE_RATE, PULSE_COUNT, CHARGE

---

## 📦 Dépendances

```python
pandas>=2.0
numpy>=1.24
scikit-learn>=1.0
xgboost>=2.0
tensorflow>=2.10
matplotlib>=3.5
seaborn>=0.12
plotly>=5.0
streamlit>=1.30
joblib>=1.0
```

---

## 📈 Fichiers Générés

| Fichier | Description |
|---------|-------------|
| `plots/01_thermal_health_model.png` | Visualisations thermiques |
| `plots/01_thermal_xgb_model.pkl` | Modèle XGBoost thermique |
| `plots/02_cooling_efficiency.png` | Control charts refroidissement |
| `plots/02_cooling_iso_forest.pkl` | Isolation Forest cooling |
| `plots/03_electrical_stability.png` | Analyse électrique |
| `plots/04_load_temperature_coupling.png` | Couplage Charge-Température |
| `plots/04_shap_beeswarm.png` | SHAP Beeswarm Plot |
| `plots/04_shap_dependence.png` | SHAP Dependence Plots |
| `plots/04_xgb_coupling_model.pkl` | Modèle XGBoost couplage |
| `plots/05_global_health_index.png` | Dashboard Health Index |
| `LAST_DATA/TG1_Health_Index.csv` | Données avec scores |

---

## 🏭 Contexte Industriel

**TG1** = Turbo-alternateur synchrone de la centrale STEG
- Machine synchronisée réseau 50 Hz (3000 rpm)
- Tension nominale: 15.75 kV
- Surveillance par 4 canaux de décharges partielles (PD)
- Refroidissement par air forcé

### Ce que ce système permet:
1. **Détection précoce** des anomalies thermiques
2. **Monitoring** de l'efficacité du refroidissement
3. **Analyse** de la stabilité électrique
4. **Score de santé global** pour aide à la décision
5. **Recommandations** de maintenance préventive

---

## 📋 Changelog

| Version | Date | Modifications |
|---------|------|---------------|
| v1.1 | 01/03/2026 | ✅ Ajout Module 4: Load-Temperature Coupling + SHAP Analysis |
| v1.0 | 01/03/2026 | Pipeline complet: Thermal, Cooling, Electrical, Health Index |

---

*Développé par Nadhir - Stage STEG 2026*
