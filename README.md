# 🏭 TG1 Digital Twin - Système de Monitoring Intelligent

## Projet STEG - Intelligence Artificielle pour la Maintenance Prédictive

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.31-red.svg)](https://streamlit.io)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange.svg)](https://tensorflow.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0-green.svg)](https://xgboost.ai)

---

## 📋 Table des Matières

- [Vue d'Ensemble](#-vue-densemble)
- [Architecture](#-architecture)
- [Modules](#-modules)
- [Installation](#-installation)
- [Dashboards](#-dashboards)
- [Modèles ML](#-modèles-ml)
- [Smart Ticketing](#-smart-ticketing)
- [Structure du Projet](#-structure-du-projet)

---

## 🎯 Vue d'Ensemble

Système complet de **Digital Twin** pour la turbine à gaz TG1 de Sousse, intégrant:

| Composant | Description |
|-----------|-------------|
| 🤖 **ML Models** | 7 modèles (XGBoost, Random Forest, LSTM, Autoencoder, etc.) |
| ⚡ **PD Models** | 6 modèles pour décharges partielles (Classification, Clustering) |
| 🏭 **TG1 Monitoring** | 5 modules (Thermal, Cooling, Electrical, Coupling, Health) |
| 🎫 **Smart Ticketing** | Génération automatique ML + RAG + LLM |
| 📊 **Drift Control** | Control Charts pour tous les modèles |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           TG1 DIGITAL TWIN                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐     │
│  │  LAST_DATA  │   │  ML Models  │   │  PD Models  │   │ TG1 Monitor │     │
│  │   (CSV)     │──▶│  (7 models) │   │  (6 models) │   │  (5 modules)│     │
│  └─────────────┘   └──────┬──────┘   └──────┬──────┘   └──────┬──────┘     │
│                           │                 │                  │            │
│                           ▼                 ▼                  ▼            │
│                    ┌─────────────────────────────────────────────┐          │
│                    │           🎫 SMART TICKETING                │          │
│                    │       ML + RAG + LLM Integration            │          │
│                    └─────────────────────────────────────────────┘          │
│                                        │                                    │
│                                        ▼                                    │
│                    ┌─────────────────────────────────────────────┐          │
│                    │          📊 DASHBOARDS (Streamlit)          │          │
│                    │  Unified | Drift Control | Ticketing        │          │
│                    └─────────────────────────────────────────────┘          │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 📦 Modules

### 1. 🤖 ML Models (`ml_models/`)

| Script | Modèle | Type | Métriques |
|--------|--------|------|-----------|
| `01_XGBoost_Regressor.py` | XGBoost | Régression | R² > 0.95 |
| `02_Random_Forest.py` | Random Forest | Régression | R² > 0.94 |
| `03_ANN_Regression.py` | Neural Network | Régression | R² > 0.93 |
| `04_LSTM_TimeSeries.py` | LSTM | Time Series | MAE < 2.0 |
| `05_Isolation_Forest_Anomaly.py` | Isolation Forest | Anomalies | - |
| `06_Autoencoder_Anomaly.py` | Autoencoder | Anomalies | - |
| `07_Health_Index.py` | RF + IsoForest | Santé globale | - |

### 2. ⚡ PD Models (`pd_models/`)

| Script | Modèle | Type | Application |
|--------|--------|------|-------------|
| `01_Feature_Engineering.py` | Feature Extraction | Preprocessing | 50+ features |
| `02_KMeans_Clustering.py` | K-Means | Clustering | 3 clusters |
| `03_DBSCAN_Clustering.py` | DBSCAN | Clustering | Outliers |
| `04_XGBoost_Classifier.py` | XGBoost + SHAP | Classification | 3 classes |
| `05_LSTM_PD_Prediction.py` | LSTM | Prédiction | Forecast |
| `06_PD_Severity_Score.py` | Scoring | Index | 0-100 |

### 3. 🏭 TG1 Monitoring (`tg1_monitoring/`)

| Script | Module | Fonction |
|--------|--------|----------|
| `01_Thermal_Health.py` | Thermique | Prédiction température |
| `02_Cooling_Efficiency.py` | Refroidissement | Efficacité ΔT |
| `03_Electrical_Stability.py` | Électrique | Stabilité fréquence |
| `04_Load_Temperature_Coupling.py` | Couplage | SHAP Explainability |
| `05_Global_Health_Index.py` | Santé Globale | Index composite |

### 4. 🎫 Smart Ticketing (`ticketing/`)

```
┌──────────────────────────────────────────────────────────────────┐
│                     🧠 SMART TICKET ENGINE                       │
├──────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐         │
│  │🤖 ML Detection│──▶│📚 RAG Retrieval│──▶│🧠 LLM Generate│        │
│  │ XGBoost/LSTM │   │ Knowledge Base│   │ Smart Content│         │
│  │ SHAP Analysis│   │ Procedures   │   │ Descriptions │         │
│  └──────────────┘   └──────────────┘   └──────────────┘         │
│          ↓                 ↓                  ↓                  │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                    🎫 SMART TICKET                         │  │
│  │ • Priorité automatique (ML)    • Procédures RAG           │  │
│  │ • Root Cause Analysis (SHAP)   • Recommandations (LLM)    │  │
│  │ • Description intelligente     • Prevention (LLM)         │  │
│  └───────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
```

---

## 💻 Installation

```bash
# Clone
git clone https://github.com/Nadhir1210/steg-step-2.git
cd steg-step-2

# Virtual environment
python -m venv .venv
.\.venv\Scripts\activate  # Windows
source .venv/bin/activate # Linux/Mac

# Dependencies
pip install -r requirements.txt
```

### Dépendances principales

```
streamlit>=1.31.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
xgboost>=2.0.0
tensorflow>=2.15.0
plotly>=5.18.0
shap>=0.44.0
fpdf2>=2.7.0
joblib>=1.3.0
```

---

## 📊 Dashboards

### Lancement rapide

```bash
# Dashboard unifié (port 8510)
streamlit run visual/app_unified.py --server.port 8510

# Drift Control (port 8520)
streamlit run visual/drift_control_dashboard.py --server.port 8520

# Smart Ticketing (port 8526)
streamlit run ticketing/app_smart_ticketing.py --server.port 8526

# Integrated ML Ticketing (port 8530)
streamlit run ticketing/app_integrated_ticketing.py --server.port 8530

# ML Models Dashboard (port 8511)
streamlit run ml_models/app_streamlit.py --server.port 8511

# PD Models Dashboard (port 8512)
streamlit run pd_models/app_streamlit.py --server.port 8512

# TG1 Monitoring Dashboard (port 8513)
streamlit run tg1_monitoring/app_streamlit.py --server.port 8513
```

### URLs

| Dashboard | Port | URL |
|-----------|------|-----|
| Unified | 8510 | http://localhost:8510 |
| Drift Control | 8520 | http://localhost:8520 |
| Smart Ticketing | 8526 | http://localhost:8526 |
| Integrated Ticketing | 8530 | http://localhost:8530 |
| ML Models | 8511 | http://localhost:8511 |
| PD Models | 8512 | http://localhost:8512 |
| TG1 Monitoring | 8513 | http://localhost:8513 |

---

## 🤖 Modèles ML

### Modèles entraînés disponibles

```
ml_models/plots/
├── 01_xgboost_model.pkl
├── 02_random_forest_model.pkl
├── 03_ann_model.keras
├── 04_lstm_model.keras
├── 05_isolation_forest_model.pkl
├── 06_autoencoder_model.keras
└── 07_health_index_rf_model.pkl

pd_models/plots/
├── 02_kmeans_model.pkl
├── 03_dbscan_model.pkl
├── 04_xgboost_classifier.pkl
├── 04_shap_explainer.pkl
└── 05_lstm_pd_model.keras

tg1_monitoring/plots/
├── 01_thermal_xgb_model.pkl
├── 02_cooling_lr_model.pkl
├── 02_cooling_iso_forest.pkl
├── 04_xgb_coupling_model.pkl
└── 04_shap_explainer.pkl
```

---

## 🎫 Smart Ticketing

### Architecture ML + RAG + LLM

| Composant | Classe | Rôle |
|-----------|--------|------|
| **ML** | Detection | Anomaly detection, severity scoring, SHAP |
| **RAG** | `KnowledgeBase` | Procédures, manuels, historique incidents |
| **LLM** | `LLMGenerator` | Descriptions, root cause, recommandations |

### Fichiers

```
ticketing/
├── smart_ticket_engine.py      # Moteur ML + RAG + LLM
├── app_smart_ticketing.py      # Dashboard Smart Ticketing
├── app_integrated_ticketing.py # Dashboard avec vrais modèles ML
├── ticket_engine.py            # Moteur basique
├── app_ticketing.py            # Dashboard basique
├── knowledge_base.json         # Base RAG
├── smart_tickets_db.json       # Stockage tickets
└── README.md                   # Documentation
```

### Exemple de ticket généré

```json
{
  "ticket_id": "SMART-20260305-5139A178",
  "module": "THERMAL",
  "priority": "HIGH",
  "severity_score": 85.0,
  "ml_confidence": 0.92,
  "llm_description": "Anomalie thermique détectée...",
  "llm_root_cause": "Analyse SHAP: charge thermique 45%...",
  "llm_recommendation": "Intervention recommandée 24-72h...",
  "retrieved_docs": ["Procédure thermique TG1..."],
  "estimated_rul": "24-72 heures"
}
```

---

## 📁 Structure du Projet

```
stage   step  2/
├── 📂 LAST_DATA/               # Données sources
│   ├── APM_Alternateur_ML.csv
│   ├── APM_Chart_ML.csv
│   ├── TG1_Sousse_ML.csv
│   └── TG1_Sousse_1min_ML.csv
│
├── 📂 ml_models/               # 7 modèles ML
│   ├── 01_XGBoost_Regressor.py
│   ├── 02_Random_Forest.py
│   ├── ...
│   ├── app_streamlit.py
│   └── plots/                  # Modèles sauvegardés
│
├── 📂 pd_models/               # 6 modèles PD
│   ├── 01_Feature_Engineering.py
│   ├── ...
│   ├── app_streamlit.py
│   └── plots/
│
├── 📂 tg1_monitoring/          # 5 modules TG1
│   ├── 01_Thermal_Health.py
│   ├── ...
│   ├── app_streamlit.py
│   └── plots/
│
├── 📂 ticketing/               # Smart Ticketing
│   ├── smart_ticket_engine.py
│   ├── app_smart_ticketing.py
│   ├── app_integrated_ticketing.py
│   └── README.md
│
├── 📂 visual/                  # Dashboards visuels
│   ├── app_unified.py
│   └── drift_control_dashboard.py
│
├── 📂 data_describe/           # EDA notebooks
│   └── *.ipynb
│
├── requirements.txt
└── README.md                   # Ce fichier
```

---

## 📊 Données

### Sources

| Fichier | Description | Rows |
|---------|-------------|------|
| `APM_Alternateur_ML.csv` | Données alternateur | ~50K |
| `APM_Chart_ML.csv` | Données Chart | ~50K |
| `TG1_Sousse_ML.csv` | TG1 données complètes | ~100K |
| `TG1_Sousse_1min_ML.csv` | TG1 haute résolution | ~500K |

---

## 🚀 Quick Start

```bash
# 1. Activer l'environnement
.\.venv\Scripts\activate

# 2. Lancer le dashboard intégré
streamlit run ticketing/app_integrated_ticketing.py --server.port 8530

# 3. Ouvrir http://localhost:8530

# 4. Aller sur "Live Analysis" pour détecter anomalies et générer tickets
```

---

## 📈 Performances

| Modèle | Métrique | Valeur |
|--------|----------|--------|
| XGBoost Regressor | R² | 0.96 |
| Random Forest | R² | 0.95 |
| LSTM TimeSeries | MAE | 1.8 |
| XGBoost PD Classifier | Accuracy | 94% |
| Health Index RF | R² | 0.92 |

---

## 👤 Auteur

**Nadhir** - Stage STEG 2026

### Contact

- 📧 Email: nadhir@example.com
- 🔗 GitHub: [Nadhir1210](https://github.com/Nadhir1210)

---

## 📄 Licence

Projet interne STEG - Usage réservé

---

## 🔮 Roadmap

- [x] ML Models (7 modèles)
- [x] PD Models (6 modèles)
- [x] TG1 Monitoring (5 modules)
- [x] Dashboards Streamlit
- [x] Smart Ticketing (ML + RAG + LLM)
- [x] Drift Control Dashboard
- [x] Integrated ML Ticketing
- [ ] Real LLM Integration (OpenAI/Claude)
- [ ] Email/SMS Notifications
- [ ] SAP PM Integration
- [ ] Real-time Data Pipeline

---

*Dernière mise à jour: Mars 2026*
