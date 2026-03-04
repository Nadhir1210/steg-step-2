# 🏭 STEG Industrial Analytics Platform

> **Dashboard Unifié** regroupant tous les modèles ML et analyses du projet

---

## 📊 Description

Cette plateforme Streamlit rassemble l'ensemble des travaux d'analyse et de modélisation réalisés sur les données industrielles du **Turbo-Alternateur TG1** de la centrale STEG.

---

## 🚀 Lancement

```bash
cd "c:\Users\Nadhir bh\Documents\stage\stage   step  2"
.\.venv\Scripts\activate
python -m streamlit run visual\app_unified.py
```

Accès: **http://localhost:8501**

---

## 📁 Structure des Données

| Dataset | Lignes | Colonnes | Description |
|---------|--------|----------|-------------|
| APM_Alternateur_ML.csv | 200K | 26 | Données alternateur (1min) |
| APM_Alternateur_10min_ML.csv | 52K | 42 | Données agrégées (10min) |
| APM_Chart_ML.csv | 200K | 20+ | Données de tendance |
| APM_Chart_10min_ML.csv | 52K | 25+ | Tendances agrégées |
| TG1_Sousse_ML.csv | 15K | 91 | Décharges partielles (PD) |
| TG1_Sousse_1min_ML.csv | 2.2M | 91 | PD haute résolution |

**Total:** 2.5M+ lignes, 200+ variables

---

## 📦 Modules Intégrés

### 🟣 PD Analysis (pd_models/)

| Script | Description | Performance |
|--------|-------------|-------------|
| 01_PD_Feature_Engineering | 33 features créées | - |
| 02_KMeans_Clustering | 2 clusters | Silhouette: 0.857 |
| 03_DBSCAN_Clustering | Détection anomalies | 9% outliers |
| 04_XGBoost_SHAP | Classification + SHAP | **97.99%** |
| 05_LSTM_PD_Classification | Deep Learning | 96.15% |
| 06_PD_Severity_Score | Score 0-100 | - |

### 🟢 ML Models (ml_models/)

| Script | Description | Performance |
|--------|-------------|-------------|
| 01_XGBoost_Regressor | Prédiction température | **R² = 1.0000** |
| 02_Random_Forest | Alternative | R² = 1.0000 |
| 03_ANN_Neural_Network | Deep Learning | R² = 0.9989 |
| 04_LSTM_TimeSeries | Prédiction future | R² = 0.9792 |
| 05_Isolation_Forest | Détection anomalies | 6K anomalies |
| 06_Autoencoder | Anomalies DL | P97 seuil |
| 07_Health_Index | Score composite | 0-100 |

### 🟠 TG1 Digital Twin (tg1_monitoring/)

| Script | Description | Résultat |
|--------|-------------|----------|
| 01_Thermal_Health_Model | Modèle thermique | XGBoost |
| 02_Cooling_Efficiency | Efficacité refroidissement | SPC ±3σ |
| 03_Electrical_Stability | Stabilité électrique | 90.3/100 |
| 04_Load_Temperature_Coupling | SHAP Analysis | Top: COOLING_DELTA |
| 05_Global_Health_Index | Score global | 67.0/100 |

---

## 🖥️ Pages du Dashboard

| Page | Description |
|------|-------------|
| 🏠 Accueil | Vue d'ensemble et architecture |
| 📊 Datasets | Description détaillée des données |
| 🟣 PD Models | Analyse des décharges partielles |
| 🟢 ML Models | Modèles de prédiction |
| 🟠 TG1 Digital Twin | Monitoring santé |
| 📈 Résultats | Tableau comparatif |
| ℹ️ À Propos | Documentation |

---

## 📊 Métriques Clés

| Module | Meilleur Modèle | Performance |
|--------|-----------------|-------------|
| PD Classification | XGBoost + SHAP | 97.99% |
| Température | XGBoost | R² = 1.0000 |
| Health Index | Composite | 67.0/100 |

---

## 🔧 Dépendances

```python
streamlit>=1.30
pandas>=2.0
numpy>=1.24
plotly>=5.0
```

---

## 📋 Dashboards Spécialisés

Pour des analyses plus détaillées:

- **PD Analysis:** `streamlit run pd_models/app_streamlit.py`
- **ML Models:** `streamlit run ml_models/app_streamlit.py`
- **TG1 Digital Twin:** `streamlit run tg1_monitoring/app_streamlit.py`

---

*Développé par Nadhir - Stage STEG 2026*
