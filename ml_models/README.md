# 🤖 Machine Learning Models - Alternateur APM

## 📋 Vue d'ensemble

Ce dossier contient **7 modèles de Machine Learning** développés pour la prédiction de température et la maintenance prédictive de l'alternateur APM.

### Résultats Globaux

| # | Modèle | Type | Performance | Fichier PKL |
|---|--------|------|-------------|-------------|
| 01 | **XGBoost** | Régression | R² = 1.0000, RMSE = 0.06°C | `01_xgboost_model.pkl` |
| 02 | **Random Forest** | Régression | R² = 1.0000, RMSE = 0.07°C | `02_random_forest_model.pkl` |
| 03 | **ANN** | Deep Learning | R² = 0.9989, RMSE = 0.81°C | `03_ann_model.keras` |
| 04 | **LSTM** | Séries Temporelles | R² = 0.9792, RMSE = 2.95°C | `04_lstm_model.keras` |
| 05 | **Isolation Forest** | Anomalies | 3.0% détectées (6,000 pts) | `05_isolation_forest_model.pkl` |
| 06 | **Autoencoder** | Anomalies DL | 3.0% détectées (4,500 pts) | `06_autoencoder_model.keras` |
| 07 | **Health Index** | Composite | Score moyen: 79.6/100 | `07_health_index_*.pkl` |

---

## 🔵 A. MODÈLES DE PRÉDICTION

### 1️⃣ XGBoost Regressor ⭐ RECOMMANDÉ

**Fichier:** `01_XGBoost_Regressor.py`

**Description:** Modèle de gradient boosting pour prédire la température moyenne du stator. État de l'art pour les données tabulaires.

**Configuration:**
```python
params = {
    'n_estimators': 200,
    'max_depth': 8,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'device': 'cuda'  # GPU activé
}
```

**Résultats:**
- **R² Score:** 1.0000 (100% variance expliquée)
- **RMSE:** 0.06°C
- **MAE:** 0.04°C
- **CV R²:** 1.0000 (±0.0000)

**Top Features:**
1. `STATOR_PHASE_C_WINDING_TEMP_1_degC` (75.96%)
2. `STATOR_PHASE_A_WINDING_TEMP_2_degC` (16.87%)
3. `STATOR_PHASE_B_WINDING_TEMP_2_degC` (2.30%)

**Fichiers générés:**
- `plots/01_xgboost_model.pkl` - Modèle sauvegardé
- `plots/01_xgboost_predictions.png` - Visualisation
- `plots/01_xgboost_feature_importance.png` - Importance des features
- `plots/01_xgboost_metrics.csv` - Métriques

---

### 2️⃣ Random Forest Regressor

**Fichier:** `02_Random_Forest_Regressor.py`

**Description:** Modèle d'ensemble basé sur des arbres de décision. Plus simple et interprétable que XGBoost.

**Configuration:**
```python
params = {
    'n_estimators': 100,
    'max_depth': 15,
    'random_state': 42,
    'n_jobs': -1
}
```

**Résultats:**
- **R² Score:** 1.0000
- **RMSE:** 0.07°C
- **MAE:** 0.02°C

**Top Features:**
1. `STATOR_PHASE_B_WINDING_TEMP_1_degC` (55.19%)
2. `STATOR_PHASE_B_WINDING_TEMP_2_degC` (40.30%)

**Fichiers générés:**
- `plots/02_random_forest_model.pkl`
- `plots/02_random_forest_predictions.png`
- `plots/02_random_forest_feature_importance.png`

---

### 3️⃣ ANN (Artificial Neural Network)

**Fichier:** `03_ANN_Neural_Network.py`

**Description:** Réseau de neurones profond avec architecture Dense pour la prédiction de température.

**Architecture:**
```
Input (28) → Dense(128) → BatchNorm → Dropout(0.3)
          → Dense(64)  → BatchNorm → Dropout(0.2)
          → Dense(32)  → Dense(16) → Dense(1)
```

**Résultats:**
- **R² Score:** 0.9989
- **RMSE:** 0.81°C
- **MAE:** 0.59°C
- **Epochs:** 37 (early stopping)

**Fichiers générés:**
- `plots/03_ann_model.keras` - Modèle TensorFlow
- `plots/03_ann_scaler_X.pkl` - Scaler des features
- `plots/03_ann_scaler_y.pkl` - Scaler de la target
- `plots/03_ann_results.png`

---

### 4️⃣ LSTM (Long Short-Term Memory)

**Fichier:** `04_LSTM_TimeSeries.py`

**Description:** Réseau récurrent pour la prédiction temporelle. Prédit la température à **t+10 minutes**.

**Architecture:**
```
Input (10, 1) → LSTM(64) → Dropout(0.2)
             → LSTM(32) → Dropout(0.2)
             → Dense(16) → Dense(1)
```

**Configuration:**
- **Window size:** 10 pas (100 minutes d'historique)
- **Horizon:** +10 minutes
- **Dataset:** APM_Alternateur_10min_ML.csv

**Résultats:**
- **R² Score:** 0.9792
- **RMSE:** 2.95°C
- **MAE:** 1.97°C

**Fichiers générés:**
- `plots/04_lstm_model.keras`
- `plots/04_lstm_scaler.pkl`
- `plots/04_lstm_results.png`

---

## 🔴 B. MODÈLES DE DÉTECTION D'ANOMALIES

### 5️⃣ Isolation Forest

**Fichier:** `05_Isolation_Forest_Anomaly.py`

**Description:** Algorithme non-supervisé basé sur l'isolation pour détecter les comportements anormaux.

**Configuration:**
```python
params = {
    'n_estimators': 200,
    'contamination': 0.03,  # 3% d'anomalies attendues
    'max_samples': 'auto'
}
```

**Résultats:**
- **Anomalies détectées:** 6,000 (3.0%)
- **Données normales:** 194,000 (97.0%)

**Analyse des anomalies:**
| Variable | Normal (moy) | Anomalie (moy) | Écart |
|----------|-------------|----------------|-------|
| TEMP_STATOR_MEAN | 61.75°C | 48.27°C | -21.8% |
| TEMP_STATOR_MAX | 69.66°C | 51.51°C | -26.0% |
| TEMP_PHASE_IMBALANCE | 1.67°C | 0.70°C | -58.0% |

**Fichiers générés:**
- `plots/05_isolation_forest_model.pkl`
- `plots/05_isolation_forest_scaler.pkl`
- `plots/05_anomalies_detected.csv` - Liste des anomalies
- `plots/05_isolation_forest_results.png`

---

### 6️⃣ Autoencoder

**Fichier:** `06_Autoencoder_Anomaly.py`

**Description:** Réseau de neurones encoder-decoder qui apprend la représentation "normale" des données. Les anomalies sont détectées par une erreur de reconstruction élevée.

**Architecture:**
```
Encoder: Input(3) → Dense(32) → Dense(16) → Dense(4) [Latent Space]
Decoder: Dense(4) → Dense(16) → Dense(32) → Dense(3) [Reconstruction]
```

**Résultats:**
- **Seuil (P97):** 0.001717
- **Anomalies détectées:** 4,500 (3.0%)
- **Données normales:** 145,500 (97.0%)

**Fichiers générés:**
- `plots/06_autoencoder_model.keras`
- `plots/06_encoder_model.keras`
- `plots/06_autoencoder_scaler.pkl`
- `plots/06_autoencoder_anomalies.csv`
- `plots/06_autoencoder_results.png`

---

## ⭐ C. HEALTH INDEX

### 7️⃣ Indice de Santé Composite

**Fichier:** `07_Health_Index.py`

**Description:** Calcule un score de santé global de l'alternateur en combinant plusieurs indicateurs.

**Formule:**
```
HI = 100 - (0.30 × Erreur_Prédiction + 
            0.25 × Asymétrie_Phases + 
            0.20 × Ratio_Temp/Puissance + 
            0.25 × Score_Anomalies)
```

**Composantes:**

| Composante | Poids | Description |
|------------|-------|-------------|
| Erreur de prédiction | 30% | Écart entre température réelle et prédite (RF) |
| Asymétrie des phases | 25% | Déséquilibre courant + température |
| Ratio Temp/Puissance | 20% | °C par MW produit |
| Score d'anomalies | 25% | Score Isolation Forest |

**Résultats:**
- **Score moyen:** 79.6/100
- **Médiane:** 78.7/100
- **Min:** 28.5/100
- **Max:** 98.1/100

**Distribution:**
| État | Quantité | Pourcentage |
|------|----------|-------------|
| 🟢 Excellent (≥80) | 35,719 | 35.7% |
| 🟡 Bon (60-79) | 64,227 | 64.2% |
| 🟠 Attention (40-59) | 52 | 0.1% |
| 🔴 Critique (<40) | 2 | 0.0% |

**Fichiers générés:**
- `plots/07_health_index_rf_model.pkl` - Modèle Random Forest
- `plots/07_health_index_isoforest_model.pkl` - Modèle Isolation Forest
- `plots/07_health_index_scaler.pkl` - Scaler
- `plots/07_health_index_data.csv` - Données avec HI
- `plots/07_health_index_results.png`

---

## 🚀 Utilisation

### Prérequis
```bash
pip install pandas numpy scikit-learn xgboost tensorflow keras joblib matplotlib seaborn
```

### Exécuter un modèle
```bash
cd ml_models
python 01_XGBoost_Regressor.py
```

### Exécuter tous les modèles
```bash
python run_all_models.py
```

### Charger un modèle sauvegardé
```python
import joblib

# Charger XGBoost
model = joblib.load('plots/01_xgboost_model.pkl')

# Faire une prédiction
prediction = model.predict(X_new)
```

### Charger un modèle TensorFlow
```python
from tensorflow import keras

# Charger ANN
model = keras.models.load_model('plots/03_ann_model.keras')

# Charger les scalers
scaler_X = joblib.load('plots/03_ann_scaler_X.pkl')
scaler_y = joblib.load('plots/03_ann_scaler_y.pkl')

# Prédiction
X_scaled = scaler_X.transform(X_new)
y_pred_scaled = model.predict(X_scaled)
y_pred = scaler_y.inverse_transform(y_pred_scaled)
```

---

## 📊 Structure des fichiers

```
ml_models/
├── 01_XGBoost_Regressor.py
├── 02_Random_Forest_Regressor.py
├── 03_ANN_Neural_Network.py
├── 04_LSTM_TimeSeries.py
├── 05_Isolation_Forest_Anomaly.py
├── 06_Autoencoder_Anomaly.py
├── 07_Health_Index.py
├── run_all_models.py
├── README.md
└── plots/
    ├── 01_xgboost_model.pkl
    ├── 01_xgboost_predictions.png
    ├── 01_xgboost_feature_importance.png
    ├── 01_xgboost_metrics.csv
    ├── 02_random_forest_model.pkl
    ├── 02_random_forest_*.png
    ├── 03_ann_model.keras
    ├── 03_ann_scaler_*.pkl
    ├── 03_ann_results.png
    ├── 04_lstm_model.keras
    ├── 04_lstm_scaler.pkl
    ├── 04_lstm_results.png
    ├── 05_isolation_forest_model.pkl
    ├── 05_anomalies_detected.csv
    ├── 05_isolation_forest_results.png
    ├── 06_autoencoder_model.keras
    ├── 06_autoencoder_anomalies.csv
    ├── 06_autoencoder_results.png
    ├── 07_health_index_*.pkl
    ├── 07_health_index_data.csv
    └── 07_health_index_results.png
```

---

## 📝 Notes pour la Soutenance

### Points forts à présenter:

1. **XGBoost** - Performance exceptionnelle (R² = 1.0), GPU accéléré
2. **LSTM** - Prédiction temporelle innovante (t+10 min)
3. **Health Index** - Vision business actionnable

### Comparaison des approches:

| Approche | Avantages | Inconvénients |
|----------|-----------|---------------|
| ML Classique (XGBoost, RF) | Rapide, interprétable | Pas de notion temporelle |
| Deep Learning (ANN, LSTM) | Capture patterns complexes | Plus lent à entraîner |
| Non-supervisé (IF, AE) | Pas besoin de labels | Difficile à valider |

### Applications industrielles:

- **Maintenance prédictive** : Alertes précoces sur anomalies
- **Optimisation** : Ajustement des paramètres opérationnels  
- **Surveillance** : Tableau de bord Health Index en temps réel

---

*Développé pour APM Alternateur - Stage 2024*
