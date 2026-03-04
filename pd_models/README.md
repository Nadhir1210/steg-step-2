# 📊 PD Models - Analyse des Décharges Partielles

> **Pipeline complet de Machine Learning pour l'analyse des Décharges Partielles (PD)**

| 🎯 Modèle | 📈 Performance | ✅ Améliorations |
|-----------|----------------|------------------|
| **Feature Engineering** | 33 features créées | Intensité, Énergie, Asymétrie |
| **KMeans** | Silhouette: 0.857 | 2 clusters identifiés |
| **DBSCAN** | 9.0% anomalies | 1,347 événements extrêmes |
| **XGBoost + SHAP** | Accuracy: 97.99% | ✅ SHAP + Validation Temporelle |
| **LSTM Classification** | Accuracy: 96.15% | ✅ Classification + Validation Temporelle |
| **Severity Score** | Score 0-100 | 75.9% Excellent, 5.1% Critique |

Ce dossier contient une collection complète de modèles d'analyse des **Décharges Partielles (PD)** pour la surveillance et le diagnostic des équipements électriques haute tension.

## 🚀 Quick Start

```bash
# Exécuter tout le pipeline PD en une seule commande
cd "c:\Users\Nadhir bh\Documents\stage\stage   step  2"
.\.venv\Scripts\python.exe pd_models\01_PD_Feature_Engineering.py
.\.venv\Scripts\python.exe pd_models\02_KMeans_Clustering.py
.\.venv\Scripts\python.exe pd_models\03_DBSCAN_Clustering.py
.\.venv\Scripts\python.exe pd_models\04_XGBoost_SHAP.py
.\.venv\Scripts\python.exe pd_models\05_LSTM_PD_Classification.py
.\.venv\Scripts\python.exe pd_models\06_PD_Severity_Score.py
```

## 📁 Structure du Dossier

```
pd_models/
├── 01_PD_Feature_Engineering.py    # Création des features intelligentes
├── 02_KMeans_Clustering.py         # Segmentation par clustering
├── 03_DBSCAN_Clustering.py         # Détection d'anomalies
├── 04_XGBoost_Classifier.py        # Classification (version sans SHAP)
├── 04_XGBoost_SHAP.py              # ⭐ RECOMMANDÉ: Classification + SHAP
├── 05_LSTM_PD_Prediction.py        # Prédiction régression (obsolète)
├── 05_LSTM_PD_Classification.py    # ⭐ RECOMMANDÉ: Classification d'événements
├── 06_PD_Severity_Score.py         # Score de sévérité global
├── README.md                        # Cette documentation
└── plots/                           # Visualisations et modèles sauvegardés (40 fichiers)
```

> 💡 **Conseil**: Utilisez les versions avec ⭐ pour bénéficier des améliorations (SHAP, validation temporelle, classification)

---

## 🔵 01 - Feature Engineering PD

**Objectif**: Créer des variables intelligentes à partir des données brutes de décharge.

### Features Créées

| Feature | Formule | Description |
|---------|---------|-------------|
| `PD_INTENSITY` | CURRENT_ABS × PULSE_COUNT | Intensité des décharges par canal |
| `PD_ENERGY` | MEAN_CHARGE × DISCHARGE_RATE | Énergie totale des décharges |
| `INTENSITY_ASYMMETRY` | max(CH1-4) - min(CH1-4) | Déséquilibre entre canaux |
| `ENERGY_ASYMMETRY` | max(Energy) - min(Energy) | Asymétrie énergétique |
| `Rolling Features` | mean/std sur 10min, 30min, 1h | Tendances temporelles |
| `INTENSITY_CV` | std/mean | Coefficient de variation |

### Résultats

- **Dataset enrichi**: 14,956 lignes × 129 colonnes
- **Nouvelles features**: 33 variables créées

### Fichiers Générés
- `plots/01_feature_engineering.png`
- `LAST_DATA/TG1_Sousse_PD_Features.csv`

---

## 🔵 02 - KMeans Clustering

**Objectif**: Identifier différents comportements de décharge par segmentation non-supervisée.

### Configuration
- **Méthode**: StandardScaler → PCA → Elbow Method → KMeans
- **K optimal**: 2 (basé sur silhouette score)
- **Silhouette Score**: 0.857

### Résultats

| Cluster | Label | Count | % |
|---------|-------|-------|---|
| 0 | Activité modérée | 299 | 2.0% |
| 1 | Faible activité (Normal) | 14,656 | 98.0% |

### Profil des Clusters

| Caractéristique | Cluster 0 (Modéré) | Cluster 1 (Normal) |
|-----------------|--------------------|--------------------|
| PD_INTENSITY_TOTAL | 15,933,028 | 101,184 |
| PD_ENERGY_TOTAL | 4,837,529 | 1,564,326 |
| INTENSITY_ASYMMETRY | 15,922,685 | 92,719 |

### Fichiers Générés
- `plots/02_kmeans_clustering.png`
- `plots/02_kmeans_model.pkl`
- `plots/02_kmeans_scaler.pkl`
- `plots/02_kmeans_pca.pkl`

---

## 🟣 03 - DBSCAN Clustering

**Objectif**: Détecter les événements extrêmes et anomalies de décharge.

### Configuration
- **Algorithme**: DBSCAN (Density-Based Spatial Clustering)
- **eps**: 0.853 (estimé par k-distance)
- **min_samples**: 9

### Résultats

| Catégorie | Count | % |
|-----------|-------|---|
| Points normaux | 13,608 | 91.0% |
| **Anomalies** | 1,347 | 9.0% |

### Comparaison Normal vs Anomalies

| Feature | Normal | Anomalie | Ratio |
|---------|--------|----------|-------|
| PD_INTENSITY_TOTAL | 21,090 | 4,424,591 | **209.8x** |
| INTENSITY_ASYMMETRY | 12,811 | 4,413,830 | **344.5x** |
| PD_INTENSITY_ROLL_STD | 5,705 | 2,865,737 | **502.3x** |
| PULSE_TOTAL | 3,472 | 31,819 | 9.2x |

### Interprétation
Les anomalies détectées représentent:
- ⚡ Événements de décharge extrêmes
- 🔴 Comportements anormaux isolés
- ⚠️ Potentiels débuts de défaillance

### Fichiers Générés
- `plots/03_dbscan_clustering.png`
- `plots/03_dbscan_model.pkl`
- `plots/03_anomalies_indices.csv`

---

## 🔴 04 - XGBoost Classifier + SHAP

**Objectif**: Classification supervisée pour prédire l'état PD (Normal/Warning/Critical).

### ✅ AMÉLIORATIONS APPLIQUÉES

1. **Validation Temporelle**: Split chronologique (pas de shuffle)
2. **SHAP Explanations**: Importance réelle des variables et explications des prédictions
3. **Pas de Data Leakage**: Scaler fit sur train uniquement

### Configuration
- **Modèle**: XGBoost avec GPU CUDA
- **n_estimators**: 200
- **max_depth**: 8
- **Split**: 70% Train → 15% Val → 15% Test (chronologique)

### Labels de Classification

| Classe | Seuil (PD_INTENSITY) | Description |
|--------|----------------------|-------------|
| Normal | < 12,027 | Fonctionnement sain |
| Warning | 12,027 - 32,765 | Surveillance requise |
| Critical | ≥ 32,765 | Intervention recommandée |

### Performances (Validation Temporelle)

| Métrique | Valeur |
|----------|--------|
| **Accuracy** | 97.99% |
| **F1-Score** | 0.9803 |
| **ROC-AUC** | 0.9989 |

### SHAP Feature Importance

Les features les plus importantes pour les prédictions **Critical**:

| Feature | SHAP Importance |
|---------|-----------------|
| CURRENT_TOTAL | 1.9410 |
| PULSE_TOTAL | 1.8272 |
| INTENSITY_ASYMMETRY | 1.6874 |
| PD_INTENSITY_ROLL_MEAN_10min | 0.1717 |
| INTENSITY_CV | 0.1509 |

### 🔍 Interprétation SHAP

SHAP permet de répondre à: **"Pourquoi le modèle classe en Critical?"**

- Valeurs SHAP **positives** (rouge): poussent vers Critical
- Valeurs SHAP **négatives** (bleu): poussent vers Normal
- Plus l'importance est élevée, plus la feature influence la décision

### Fichiers Générés
- `plots/04_xgboost_shap.png` - Visualisations complètes
- `plots/04_shap_summary.png` - SHAP Beeswarm plot
- `plots/04_xgboost_shap_model.pkl`
- `plots/04_shap_explainer.pkl` - Pour expliquer de nouvelles prédictions
- `plots/04_shap_critical_importance.csv`

---

## 🔮 05 - LSTM PD Classification

**Objectif**: Prédire si un événement critique arrivera dans les 30 prochains points.

### ✅ AMÉLIORATION: Classification au lieu de Régression

Au lieu de prédire une valeur exacte (difficile), on prédit:
- 🔹 **Probabilité d'événement critique** dans 30 min
- 🔹 **Classification temporelle**: Normal → Warning → Critical
- 🔹 **Augmentation significative** (> seuil)

### Architecture

```
Bidirectional LSTM (64 units) → BatchNorm → Dropout(0.3)
↓
LSTM (32 units) → BatchNorm → Dropout(0.3)
↓
Dense (32) → Dropout(0.2) → Dense (16)
↓
Dense (3, softmax) → Normal/Warning/Critical
```

### Configuration
- **Séquence**: 60 points d'historique
- **Horizon**: Prédiction à t+30
- **Features**: 5 (Intensity, Energy, Asymmetry, Current, Pulse)
- **Validation**: Chronologique (pas de shuffle)

### ⚠️ Validation Temporelle

```
┌─────────────────────────────────────────────────────────┐
│  Train (70%)    │   Val (15%)   │   Test (15%)        │
│  0 → 10,448     │  10,448→12,686 │  12,686 → fin      │
└─────────────────────────────────────────────────────────┘
                  ↑               ↑
              Pas de shuffle (ordre chronologique)
```

- ✅ Scaler fit sur **train uniquement**
- ✅ Pas de data leakage
- ✅ Représente la situation réelle de prédiction

### Performances

| Métrique | Valeur |
|----------|--------|
| **Accuracy** | 96.15% |
| **F1-Score** | 0.9706 |

### Distribution des Labels

| Classe | Count | % |
|--------|-------|---|
| Normal | 7,839 | 52.5% |
| Warning | 6,047 | 40.5% |
| Critical | 1,040 | 7.0% |

### 🔮 Utilisation

```python
# Charger le modèle
model = tf.keras.models.load_model('plots/05_lstm_pd_classifier.keras')
config = joblib.load('plots/05_lstm_pd_classifier_config.pkl')

# Prédire sur une nouvelle séquence
proba = model.predict(sequence)  # [P(Normal), P(Warning), P(Critical)]

if proba[2] >= 0.5:
    print("⚠️ ALERTE: Événement critique prévu dans 30 min!")
elif proba[1] >= 0.5:
    print("⚠️ Surveillance accrue recommandée")
```

### Fichiers Générés
- `plots/05_lstm_pd_classification.png`
- `plots/05_lstm_pd_classifier.keras`
- `plots/05_lstm_pd_classifier_config.pkl`

---

## ⭐ 06 - PD Severity Score

**Objectif**: Créer un score agrégé de 0 à 100 pour quantifier la sévérité PD.

### Formule du Score

```
PD_Score = 35%×Intensity + 25%×Energy + 15%×Asymmetry + 15%×Trend + 10%×Stability
```

### Composantes

| Composante | Poids | Moyenne |
|------------|-------|---------|
| Intensité | 35% | 7.51/100 |
| Énergie | 25% | 34.03/100 |
| Asymétrie | 15% | 7.04/100 |
| Tendance | 15% | 49.39/100 |
| Stabilité | 10% | 28.86/100 |

### Statistiques du Score

| Statistique | Valeur |
|-------------|--------|
| Minimum | 0.00 |
| Maximum | 100.00 |
| **Moyenne** | 22.49 |
| **Médiane** | 17.27 |
| Écart-type | 18.29 |

### Distribution des Classes

| Classe | Plage | Count | % |
|--------|-------|-------|---|
| 🟢 Excellent | 0-25 | 11,353 | 75.9% |
| 🟡 Bon | 25-50 | 2,692 | 18.0% |
| 🟠 Moyen | 50-75 | 144 | 1.0% |
| 🔴 Critique | 75-100 | 767 | 5.1% |

### Interprétation

| Score | État | Action |
|-------|------|--------|
| 🟢 0-25 | Excellent | Fonctionnement optimal |
| 🟡 25-50 | Bon | Surveillance légère recommandée |
| 🟠 50-75 | Moyen | Surveillance accrue requise |
| 🔴 75-100 | Critique | Intervention recommandée |

### Fichiers Générés
- `plots/06_pd_severity_score.png`
- `LAST_DATA/TG1_Sousse_PD_WithScore.csv`
- `plots/06_severity_score_params.pkl`

---

## 🚀 Utilisation

### Exécution Complète

```bash
# 1. Feature Engineering (OBLIGATOIRE en premier)
python pd_models/01_PD_Feature_Engineering.py

# 2. Clustering
python pd_models/02_KMeans_Clustering.py
python pd_models/03_DBSCAN_Clustering.py

# 3. Classification (avec améliorations)
python pd_models/04_XGBoost_SHAP.py        # ✅ Avec SHAP + Validation Temporelle

# 4. Prédiction d'événements (avec améliorations)
python pd_models/05_LSTM_PD_Classification.py  # ✅ Classification + Validation Temporelle

# 5. Score de Sévérité
python pd_models/06_PD_Severity_Score.py
```

### Ordre d'Exécution

```
┌─────────────────────────┐
│ 01_Feature_Engineering  │  ← OBLIGATOIRE (crée les features)
└──────────┬──────────────┘
           │
     ┌─────┴─────┐
     │           │
     ▼           ▼
┌─────────┐ ┌─────────┐
│02_KMeans│ │03_DBSCAN│  ← Clustering (parallélisable)
└────┬────┘ └────┬────┘
     │           │
     └─────┬─────┘
           │
     ┌─────┴─────┐
     │           │
     ▼           ▼
┌───────────────┐ ┌───────────────┐
│04_XGBoost_SHAP│ │05_LSTM_Classif│  ← ML avec améliorations
│ ✅ + SHAP     │ │ ✅ classif    │
└───────┬───────┘ └───────┬───────┘
        │                 │
        └────────┬────────┘
                 │
                 ▼
┌─────────────────────────┐
│ 06_PD_Severity_Score    │  ← Score final
└─────────────────────────┘
```

---

## 📊 Résumé des Performances

| Modèle | Type | Métrique Clé | Performance | Amélioration |
|--------|------|--------------|-------------|--------------|
| KMeans | Clustering | Silhouette | **0.857** | - |
| DBSCAN | Anomaly Detection | Anomalies | **9.0%** | - |
| XGBoost + SHAP | Classification | Accuracy | **97.99%** | ✅ SHAP + Validation Temporelle |
| LSTM Classification | Event Prediction | Accuracy | **96.15%** | ✅ Classification + Validation Temporelle |
| Severity Score | Scoring | Couverture | **100%** | - |

### ⚠️ Bonnes Pratiques Appliquées

| Amélioration | Description |
|-------------|-------------|
| ✅ **Validation Temporelle** | Split chronologique (70/15/15), pas de shuffle |
| ✅ **SHAP** | Explication des prédictions XGBoost |
| ✅ **LSTM Classification** | Prédire événement au lieu de valeur exacte |
| ✅ **Pas de Data Leakage** | Scaler fit sur train uniquement |

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
joblib>=1.0
shap>=0.50        # Pour les explications SHAP
```

---

## 📝 Notes Techniques

### Données Sources
- **Fichier**: `LAST_DATA/TG1_Sousse_ML.csv`
- **4 canaux de mesure**: CH1, CH2, CH3, CH4
- **Mesures par canal**: CURRENT_ABS/NEG/POS, DISCHARGE_RATE, MAX_CHARGE, MEAN_CHARGE, PULSE_COUNT

### Fichiers de Sortie
- **Features enrichies**: `LAST_DATA/TG1_Sousse_PD_Features.csv` (129 colonnes)
- **Avec score**: `LAST_DATA/TG1_Sousse_PD_WithScore.csv`

### 📁 Tous les Fichiers Générés (plots/)

| Fichier | Taille | Description |
|---------|--------|-------------|
| `01_feature_engineering.png` | 180 KB | Visualisation des features |
| `02_kmeans_clustering.png` | 276 KB | Résultats KMeans |
| `02_kmeans_model.pkl` | 61 KB | Modèle KMeans sauvegardé |
| `03_dbscan_clustering.png` | 300 KB | Résultats DBSCAN |
| `03_dbscan_model.pkl` | 874 KB | Modèle DBSCAN sauvegardé |
| `03_anomalies_indices.csv` | 8 KB | Indices des anomalies |
| `04_xgboost_shap.png` | 224 KB | Visualisation XGBoost + SHAP |
| `04_shap_summary.png` | 100 KB | SHAP Beeswarm plot |
| `04_xgboost_shap_model.pkl` | 1.5 MB | Modèle XGBoost amélioré |
| `04_shap_explainer.pkl` | 10 MB | SHAP Explainer |
| `05_lstm_pd_classification.png` | 186 KB | Résultats LSTM Classification |
| `05_lstm_pd_classifier.keras` | 773 KB | Modèle LSTM Classification |
| `06_pd_severity_score.png` | 313 KB | Visualisation Score de Sévérité |

---

*Dernière mise à jour: 1 Mars 2026 - Analyse PD pour la maintenance prédictive*

---

## 📋 Changelog

| Version | Date | Modifications |
|---------|------|---------------|
| v2.0 | 01/03/2026 | ✅ Ajout SHAP pour XGBoost, ✅ LSTM Classification, ✅ Validation temporelle |
| v1.0 | 01/03/2026 | Pipeline initial: Feature Engineering, Clustering, Classification, Score |
