"""
🎫 INTEGRATED TICKETING DASHBOARD - ML Models + Smart Ticketing
================================================================
Dashboard intégré avec tous les modèles ML réels + Génération intelligente de tickets

Modèles intégrés:
- ML Models: XGBoost, Random Forest, LSTM, Isolation Forest, Autoencoder
- PD Models: XGBoost Classifier, LSTM PD, K-Means, DBSCAN
- TG1 Monitoring: Thermal, Cooling, Electrical, Coupling

Auteur: Nadhir - Stage STEG 2026
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from datetime import datetime, timedelta
import sys
import json
import joblib
import warnings
warnings.filterwarnings('ignore')

# Paths
BASE_DIR = Path(__file__).parent.parent
ML_MODELS_DIR = BASE_DIR / "ml_models" / "plots"
PD_MODELS_DIR = BASE_DIR / "pd_models" / "plots"
TG1_MODELS_DIR = BASE_DIR / "tg1_monitoring" / "plots"
DATA_DIR = BASE_DIR / "LAST_DATA"

# Add module paths
sys.path.insert(0, str(Path(__file__).parent))
from smart_ticket_engine import (
    SmartTicketEngine, SmartTicket, KnowledgeBase, LLMGenerator,
    Priority, TicketStatus, Module, AnomalyType
)

# Try loading TensorFlow/Keras
try:
    import tensorflow as tf
    from tensorflow import keras
    KERAS_AVAILABLE = True
except ImportError:
    KERAS_AVAILABLE = False

# Configuration
st.set_page_config(
    page_title="🎫 Integrated ML Ticketing",
    page_icon="🎫",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.8rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .model-card {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        padding: 20px;
        border-radius: 15px;
        margin: 10px 0;
        box-shadow: 0 8px 25px rgba(30, 60, 114, 0.3);
    }
    .anomaly-card {
        background: linear-gradient(135deg, #f5365c 0%, #f56036 100%);
        color: white;
        padding: 20px;
        border-radius: 15px;
        margin: 10px 0;
    }
    .normal-card {
        background: linear-gradient(135deg, #2dce89 0%, #2dcecc 100%);
        color: white;
        padding: 20px;
        border-radius: 15px;
        margin: 10px 0;
    }
    .warning-card {
        background: linear-gradient(135deg, #fb6340 0%, #fbb140 100%);
        color: white;
        padding: 20px;
        border-radius: 15px;
        margin: 10px 0;
    }
    .ml-badge {
        display: inline-block;
        padding: 5px 12px;
        border-radius: 20px;
        font-weight: bold;
        margin: 3px;
        font-size: 0.8rem;
    }
    .badge-xgb { background: #4CAF50; color: white; }
    .badge-lstm { background: #2196F3; color: white; }
    .badge-iso { background: #FF5722; color: white; }
    .badge-ae { background: #9C27B0; color: white; }
    .badge-rf { background: #607D8B; color: white; }
    .status-box {
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .status-critical { background: #ffe6e6; border-left: 5px solid #f5365c; }
    .status-warning { background: #fff3e0; border-left: 5px solid #fb6340; }
    .status-healthy { background: #e8f5e9; border-left: 5px solid #2dce89; }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# MODEL LOADERS
# =============================================================================

@st.cache_resource
def load_ml_models():
    """Load all ML models"""
    models = {}
    
    # XGBoost Regressor
    try:
        models['xgboost'] = {
            'model': joblib.load(ML_MODELS_DIR / "01_xgboost_model.pkl"),
            'name': 'XGBoost Regressor',
            'type': 'regression'
        }
    except Exception as e:
        models['xgboost'] = {'error': str(e)}
    
    # Random Forest
    try:
        models['random_forest'] = {
            'model': joblib.load(ML_MODELS_DIR / "02_random_forest_model.pkl"),
            'name': 'Random Forest',
            'type': 'regression'
        }
    except Exception as e:
        models['random_forest'] = {'error': str(e)}
    
    # Isolation Forest
    try:
        models['isolation_forest'] = {
            'model': joblib.load(ML_MODELS_DIR / "05_isolation_forest_model.pkl"),
            'scaler': joblib.load(ML_MODELS_DIR / "05_isolation_forest_scaler.pkl"),
            'name': 'Isolation Forest',
            'type': 'anomaly_detection'
        }
    except Exception as e:
        models['isolation_forest'] = {'error': str(e)}
    
    # Health Index Models
    try:
        models['health_rf'] = {
            'model': joblib.load(ML_MODELS_DIR / "07_health_index_rf_model.pkl"),
            'scaler': joblib.load(ML_MODELS_DIR / "07_health_index_scaler.pkl"),
            'name': 'Health Index RF',
            'type': 'health'
        }
    except Exception as e:
        models['health_rf'] = {'error': str(e)}
    
    try:
        models['health_isoforest'] = {
            'model': joblib.load(ML_MODELS_DIR / "07_health_index_isoforest_model.pkl"),
            'name': 'Health IsoForest',
            'type': 'anomaly_detection'
        }
    except Exception as e:
        models['health_isoforest'] = {'error': str(e)}
    
    # LSTM (if Keras available)
    if KERAS_AVAILABLE:
        try:
            models['lstm'] = {
                'model': keras.models.load_model(ML_MODELS_DIR / "04_lstm_model.keras"),
                'scaler': joblib.load(ML_MODELS_DIR / "04_lstm_scaler.pkl"),
                'name': 'LSTM TimeSeries',
                'type': 'timeseries'
            }
        except Exception as e:
            models['lstm'] = {'error': str(e)}
        
        try:
            models['autoencoder'] = {
                'model': keras.models.load_model(ML_MODELS_DIR / "06_autoencoder_model.keras"),
                'scaler': joblib.load(ML_MODELS_DIR / "06_autoencoder_scaler.pkl"),
                'name': 'Autoencoder',
                'type': 'anomaly_detection'
            }
        except Exception as e:
            models['autoencoder'] = {'error': str(e)}
    
    return models


@st.cache_resource
def load_pd_models():
    """Load PD models"""
    models = {}
    
    # XGBoost Classifier
    try:
        models['xgb_classifier'] = {
            'model': joblib.load(PD_MODELS_DIR / "04_xgboost_classifier.pkl"),
            'scaler': joblib.load(PD_MODELS_DIR / "04_xgboost_scaler.pkl"),
            'metadata': joblib.load(PD_MODELS_DIR / "04_xgboost_metadata.pkl"),
            'name': 'XGBoost PD Classifier',
            'type': 'classification'
        }
    except Exception as e:
        models['xgb_classifier'] = {'error': str(e)}
    
    # K-Means
    try:
        models['kmeans'] = {
            'model': joblib.load(PD_MODELS_DIR / "02_kmeans_model.pkl"),
            'scaler': joblib.load(PD_MODELS_DIR / "02_kmeans_scaler.pkl"),
            'pca': joblib.load(PD_MODELS_DIR / "02_kmeans_pca.pkl"),
            'name': 'K-Means Clustering',
            'type': 'clustering'
        }
    except Exception as e:
        models['kmeans'] = {'error': str(e)}
    
    # DBSCAN
    try:
        models['dbscan'] = {
            'model': joblib.load(PD_MODELS_DIR / "03_dbscan_model.pkl"),
            'scaler': joblib.load(PD_MODELS_DIR / "03_dbscan_scaler.pkl"),
            'params': joblib.load(PD_MODELS_DIR / "03_dbscan_params.pkl"),
            'name': 'DBSCAN',
            'type': 'clustering'
        }
    except Exception as e:
        models['dbscan'] = {'error': str(e)}
    
    # SHAP Explainer
    try:
        models['shap_explainer'] = {
            'explainer': joblib.load(PD_MODELS_DIR / "04_shap_explainer.pkl"),
            'name': 'SHAP Explainer',
            'type': 'explainability'
        }
    except Exception as e:
        models['shap_explainer'] = {'error': str(e)}
    
    # Severity Score Params
    try:
        models['severity_params'] = joblib.load(PD_MODELS_DIR / "06_severity_score_params.pkl")
    except:
        models['severity_params'] = None
    
    # LSTM PD (if Keras available)
    if KERAS_AVAILABLE:
        try:
            models['lstm_pd'] = {
                'model': keras.models.load_model(PD_MODELS_DIR / "05_lstm_pd_model.keras"),
                'config': joblib.load(PD_MODELS_DIR / "05_lstm_pd_config.pkl"),
                'name': 'LSTM PD Predictor',
                'type': 'timeseries'
            }
        except Exception as e:
            models['lstm_pd'] = {'error': str(e)}
    
    return models


@st.cache_resource
def load_tg1_models():
    """Load TG1 monitoring models"""
    models = {}
    
    # Thermal
    try:
        models['thermal'] = {
            'model': joblib.load(TG1_MODELS_DIR / "01_thermal_xgb_model.pkl"),
            'scaler': joblib.load(TG1_MODELS_DIR / "01_thermal_scaler.pkl"),
            'config': joblib.load(TG1_MODELS_DIR / "01_thermal_config.pkl"),
            'name': 'Thermal XGBoost',
            'type': 'regression'
        }
    except Exception as e:
        models['thermal'] = {'error': str(e)}
    
    # Cooling
    try:
        models['cooling'] = {
            'model': joblib.load(TG1_MODELS_DIR / "02_cooling_lr_model.pkl"),
            'iso_forest': joblib.load(TG1_MODELS_DIR / "02_cooling_iso_forest.pkl"),
            'scaler': joblib.load(TG1_MODELS_DIR / "02_cooling_scaler.pkl"),
            'config': joblib.load(TG1_MODELS_DIR / "02_cooling_config.pkl"),
            'name': 'Cooling Model',
            'type': 'regression'
        }
    except Exception as e:
        models['cooling'] = {'error': str(e)}
    
    # Coupling/Load-Temperature
    try:
        models['coupling'] = {
            'model': joblib.load(TG1_MODELS_DIR / "04_xgb_coupling_model.pkl"),
            'scaler': joblib.load(TG1_MODELS_DIR / "04_coupling_scaler.pkl"),
            'config': joblib.load(TG1_MODELS_DIR / "04_coupling_config.pkl"),
            'shap_explainer': joblib.load(TG1_MODELS_DIR / "04_shap_explainer.pkl"),
            'name': 'Load-Temp Coupling XGBoost',
            'type': 'regression'
        }
    except Exception as e:
        models['coupling'] = {'error': str(e)}
    
    # Electrical config
    try:
        models['electrical'] = {
            'config': joblib.load(TG1_MODELS_DIR / "03_electrical_config.pkl"),
            'name': 'Electrical Stability',
            'type': 'analysis'
        }
    except Exception as e:
        models['electrical'] = {'error': str(e)}
    
    # Health config
    try:
        models['health'] = {
            'config': joblib.load(TG1_MODELS_DIR / "05_health_config.pkl"),
            'name': 'Global Health Index',
            'type': 'health'
        }
    except Exception as e:
        models['health'] = {'error': str(e)}
    
    return models


@st.cache_data(ttl=60)
def load_data():
    """Load all available data"""
    data = {}
    
    try:
        data['apm_alternateur'] = pd.read_csv(DATA_DIR / "APM_Alternateur_ML.csv")
    except:
        data['apm_alternateur'] = None
    
    try:
        data['apm_chart'] = pd.read_csv(DATA_DIR / "APM_Chart_ML.csv")
    except:
        data['apm_chart'] = None
    
    try:
        data['tg1_sousse'] = pd.read_csv(DATA_DIR / "TG1_Sousse_ML.csv")
    except:
        data['tg1_sousse'] = None
    
    try:
        data['tg1_sousse_1min'] = pd.read_csv(DATA_DIR / "TG1_Sousse_1min_ML.csv")
    except:
        data['tg1_sousse_1min'] = None
    
    return data


@st.cache_resource
def get_ticket_engine():
    """Get smart ticket engine"""
    return SmartTicketEngine()


# =============================================================================
# PREDICTION FUNCTIONS
# =============================================================================

def predict_anomalies_isolation_forest(models, data, threshold=-0.1):
    """Detect anomalies using Isolation Forest"""
    if 'isolation_forest' not in models or 'error' in models.get('isolation_forest', {}):
        return None, None
    
    model = models['isolation_forest']['model']
    scaler = models['isolation_forest']['scaler']
    
    # Select numeric columns
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    X = data[numeric_cols].dropna()
    
    if len(X) == 0:
        return None, None
    
    try:
        X_scaled = scaler.transform(X)
        scores = model.decision_function(X_scaled)
        predictions = model.predict(X_scaled)
        
        # -1 = anomaly, 1 = normal
        anomalies = X.iloc[predictions == -1]
        
        return anomalies, scores
    except Exception as e:
        st.error(f"Isolation Forest error: {e}")
        return None, None


def predict_pd_severity(pd_models, data):
    """Predict PD severity using XGBoost classifier"""
    if 'xgb_classifier' not in pd_models or 'error' in pd_models.get('xgb_classifier', {}):
        return None
    
    model_info = pd_models['xgb_classifier']
    model = model_info['model']
    scaler = model_info['scaler']
    
    try:
        # Get feature columns from metadata
        if 'metadata' in model_info:
            features = model_info['metadata'].get('features', None)
        else:
            features = None
        
        if features:
            available_features = [f for f in features if f in data.columns]
            if len(available_features) < len(features) * 0.5:
                return None
            X = data[available_features].dropna()
        else:
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            X = data[numeric_cols].dropna()
        
        if len(X) == 0:
            return None
        
        X_scaled = scaler.transform(X)
        predictions = model.predict(X_scaled)
        probas = model.predict_proba(X_scaled)
        
        return {
            'predictions': predictions,
            'probabilities': probas,
            'classes': model.classes_ if hasattr(model, 'classes_') else [0, 1, 2]
        }
    except Exception as e:
        st.error(f"PD XGBoost error: {e}")
        return None


def calculate_health_score(tg1_models, metrics):
    """Calculate overall health score"""
    score = 100
    issues = []
    
    # Check thermal
    if 'temperature' in metrics:
        temp = metrics['temperature']
        if temp > 90:
            score -= 30
            issues.append(f"Critical temperature: {temp}°C")
        elif temp > 80:
            score -= 15
            issues.append(f"High temperature: {temp}°C")
    
    # Check cooling
    if 'delta_t' in metrics:
        delta = metrics['delta_t']
        if delta < 5:
            score -= 20
            issues.append(f"Low cooling efficiency: ΔT={delta}°C")
    
    # Check electrical
    if 'frequency_deviation' in metrics:
        freq_dev = metrics['frequency_deviation']
        if abs(freq_dev) > 0.5:
            score -= 25
            issues.append(f"Frequency deviation: {freq_dev} Hz")
    
    # Check PD
    if 'pd_severity' in metrics:
        pd_sev = metrics['pd_severity']
        if pd_sev > 75:
            score -= 30
            issues.append(f"Critical PD: {pd_sev}")
        elif pd_sev > 50:
            score -= 15
            issues.append(f"High PD activity: {pd_sev}")
    
    return max(0, score), issues


# =============================================================================
# PAGES
# =============================================================================

def page_dashboard():
    """Main dashboard"""
    st.markdown('<h1 class="main-header">🎫 Integrated ML Ticketing System</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align:center;color:#666;">Real ML Models + Smart Ticket Generation</p>', unsafe_allow_html=True)
    
    # Load models
    ml_models = load_ml_models()
    pd_models = load_pd_models()
    tg1_models = load_tg1_models()
    engine = get_ticket_engine()
    
    # Model status
    st.subheader("🤖 Loaded Models Status")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### ML Models")
        for name, info in ml_models.items():
            if 'error' in info:
                st.error(f"❌ {name}: Load failed")
            else:
                st.success(f"✅ {info.get('name', name)}")
    
    with col2:
        st.markdown("### PD Models")
        for name, info in pd_models.items():
            if isinstance(info, dict):
                if 'error' in info:
                    st.error(f"❌ {name}: Load failed")
                else:
                    st.success(f"✅ {info.get('name', name)}")
    
    with col3:
        st.markdown("### TG1 Models")
        for name, info in tg1_models.items():
            if isinstance(info, dict):
                if 'error' in info:
                    st.error(f"❌ {name}: Load failed")
                else:
                    st.success(f"✅ {info.get('name', name)}")
    
    st.markdown("---")
    
    # Quick Stats
    stats = engine.get_statistics()
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("🎫 Total Tickets", stats.get('total', 0))
    with col2:
        st.metric("🔴 Open", stats.get('open', 0))
    with col3:
        st.metric("⚠️ Critical", stats.get('by_priority', {}).get('CRITICAL', 0))
    with col4:
        st.metric("🟠 High", stats.get('by_priority', {}).get('HIGH', 0))
    with col5:
        healthy = 'health_rf' in ml_models and 'error' not in ml_models.get('health_rf', {})
        st.metric("💚 System", "Healthy" if healthy else "Degraded")
    
    st.markdown("---")
    
    # Recent tickets
    st.subheader("🎫 Recent Smart Tickets")
    df = engine.export_to_dataframe()
    if not df.empty:
        st.dataframe(
            df[['ticket_id', 'timestamp', 'module', 'severity_score', 'priority', 'status']].tail(5),
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("No tickets generated yet. Use 'Live Analysis' to detect anomalies and generate tickets.")


def page_live_analysis():
    """Real-time analysis with ML models"""
    st.header("🔍 Live Analysis with ML Models")
    
    ml_models = load_ml_models()
    pd_models = load_pd_models()
    tg1_models = load_tg1_models()
    data = load_data()
    engine = get_ticket_engine()
    
    # Data source selection
    st.subheader("📊 Select Data Source")
    
    available_data = {k: v for k, v in data.items() if v is not None}
    if not available_data:
        st.error("No data available")
        return
    
    data_source = st.selectbox("Dataset", list(available_data.keys()))
    df = available_data[data_source]
    
    st.info(f"📁 {data_source}: {len(df)} rows, {len(df.columns)} columns")
    
    # Model selection
    st.subheader("🤖 Select Analysis Models")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        use_isolation = st.checkbox("Isolation Forest (Anomaly)", value=True)
        use_health = st.checkbox("Health Index", value=True)
    
    with col2:
        use_xgb_pd = st.checkbox("XGBoost PD Classifier", value='apm' in data_source.lower() or 'pd' in data_source.lower())
        use_kmeans = st.checkbox("K-Means Clustering", value=False)
    
    with col3:
        use_thermal = st.checkbox("Thermal Analysis", value='tg1' in data_source.lower())
        auto_ticket = st.checkbox("🎫 Auto-generate Tickets", value=True)
    
    severity_threshold = st.slider("Severity threshold for tickets", 40, 90, 60)
    
    if st.button("🚀 Run Analysis", type="primary"):
        results = {}
        anomalies_detected = []
        
        with st.spinner("Running ML Analysis..."):
            progress = st.progress(0)
            
            # 1. Isolation Forest
            if use_isolation and 'isolation_forest' in ml_models and 'error' not in ml_models.get('isolation_forest', {}):
                progress.progress(20, "Running Isolation Forest...")
                anomalies, scores = predict_anomalies_isolation_forest(ml_models, df)
                if anomalies is not None:
                    results['isolation_forest'] = {
                        'anomalies': len(anomalies),
                        'total': len(df),
                        'ratio': len(anomalies) / len(df) * 100 if len(df) > 0 else 0
                    }
                    if len(anomalies) > 0:
                        severity = min(100, 50 + len(anomalies) / len(df) * 100)
                        anomalies_detected.append({
                            'module': Module.GLOBAL,
                            'severity': severity,
                            'type': 'Isolation Forest Anomalies',
                            'metrics': {'anomaly_count': len(anomalies), 'ratio': len(anomalies) / len(df)}
                        })
            
            # 2. Health Index
            if use_health and 'health_rf' in ml_models and 'error' not in ml_models.get('health_rf', {}):
                progress.progress(40, "Calculating Health Index...")
                # Simulate health calculation
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    avg_values = df[numeric_cols].mean()
                    # Simple health score simulation
                    health_score = 100 - np.clip(avg_values.std() / 10, 0, 50)
                    results['health'] = {'score': health_score}
            
            # 3. PD Classifier
            if use_xgb_pd and 'xgb_classifier' in pd_models and 'error' not in pd_models.get('xgb_classifier', {}):
                progress.progress(60, "Running PD Classification...")
                pd_result = predict_pd_severity(pd_models, df)
                if pd_result:
                    critical_count = np.sum(pd_result['predictions'] == 2)  # Class 2 = critical
                    high_count = np.sum(pd_result['predictions'] == 1)      # Class 1 = high
                    results['pd_classification'] = {
                        'critical': critical_count,
                        'high': high_count,
                        'normal': len(pd_result['predictions']) - critical_count - high_count
                    }
                    if critical_count > 0:
                        severity = 90
                        anomalies_detected.append({
                            'module': Module.PD,
                            'severity': severity,
                            'type': 'Critical PD Activity',
                            'metrics': {'critical_samples': int(critical_count), 'high_samples': int(high_count)}
                        })
                    elif high_count > 0:
                        severity = 70
                        anomalies_detected.append({
                            'module': Module.PD,
                            'severity': severity,
                            'type': 'High PD Activity',
                            'metrics': {'high_samples': int(high_count)}
                        })
            
            # 4. Thermal Analysis
            if use_thermal and 'thermal' in tg1_models and 'error' not in tg1_models.get('thermal', {}):
                progress.progress(80, "Running Thermal Analysis...")
                # Check for temperature columns
                temp_cols = [c for c in df.columns if 'temp' in c.lower() or 'T_' in c]
                if temp_cols:
                    max_temp = df[temp_cols].max().max()
                    avg_temp = df[temp_cols].mean().mean()
                    results['thermal'] = {'max_temp': max_temp, 'avg_temp': avg_temp}
                    
                    if max_temp > 90:
                        anomalies_detected.append({
                            'module': Module.THERMAL,
                            'severity': 85,
                            'type': 'Temperature Critical',
                            'metrics': {'max_temperature': float(max_temp), 'avg_temperature': float(avg_temp)}
                        })
                    elif max_temp > 80:
                        anomalies_detected.append({
                            'module': Module.THERMAL,
                            'severity': 65,
                            'type': 'Temperature High',
                            'metrics': {'max_temperature': float(max_temp)}
                        })
            
            progress.progress(100, "Analysis Complete!")
        
        # Display Results
        st.success(f"✅ Analysis complete!")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if 'isolation_forest' in results:
                r = results['isolation_forest']
                st.metric("🔍 Anomalies Detected", r['anomalies'], f"{r['ratio']:.1f}%")
        
        with col2:
            if 'health' in results:
                st.metric("💚 Health Score", f"{results['health']['score']:.0f}")
        
        with col3:
            if 'pd_classification' in results:
                r = results['pd_classification']
                st.metric("⚡ PD Critical", r['critical'])
        
        with col4:
            if 'thermal' in results:
                r = results['thermal']
                st.metric("🌡️ Max Temp", f"{r['max_temp']:.1f}°C")
        
        # Generate tickets
        if auto_ticket and anomalies_detected:
            st.markdown("---")
            st.subheader("🎫 Generated Smart Tickets")
            
            generated_tickets = []
            for anomaly in anomalies_detected:
                if anomaly['severity'] >= severity_threshold:
                    ticket = engine.generate_smart_ticket(
                        module=anomaly['module'],
                        severity_score=anomaly['severity'],
                        metrics=anomaly['metrics'],
                        ml_confidence=0.85
                    )
                    generated_tickets.append(ticket)
            
            if generated_tickets:
                for ticket in generated_tickets:
                    priority_color = {
                        'CRITICAL': '#f5365c',
                        'HIGH': '#fb6340',
                        'MEDIUM': '#ffd600',
                        'LOW': '#2dce89'
                    }.get(ticket.priority, '#667eea')
                    
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, {priority_color} 0%, {priority_color}99 100%); 
                                color: white; padding: 15px; border-radius: 15px; margin: 10px 0;">
                        <h4>🎫 {ticket.ticket_id}</h4>
                        <p><strong>Module:</strong> {ticket.module} | <strong>Priority:</strong> {ticket.priority} | 
                           <strong>Severity:</strong> {ticket.severity_score:.0f}/100</p>
                        <p><strong>Type:</strong> {ticket.anomaly_type}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    with st.expander("📋 Ticket Details"):
                        st.markdown(f"**Description:** {ticket.llm_description[:500]}...")
                        st.markdown(f"**Recommendation:** {ticket.llm_recommendation[:500]}...")
            else:
                st.info(f"No anomalies above threshold ({severity_threshold})")
        elif not anomalies_detected:
            st.success("✅ No anomalies detected - System healthy!")


def page_model_details():
    """Model details and performance"""
    st.header("📊 Model Details & Performance")
    
    ml_models = load_ml_models()
    pd_models = load_pd_models()
    tg1_models = load_tg1_models()
    
    tabs = st.tabs(["ML Models", "PD Models", "TG1 Models"])
    
    with tabs[0]:
        st.subheader("🔬 ML Models")
        
        for name, info in ml_models.items():
            if isinstance(info, dict) and 'error' not in info:
                with st.expander(f"✅ {info.get('name', name)} ({info.get('type', 'unknown')})", expanded=False):
                    st.write(f"**Type:** {info.get('type', 'N/A')}")
                    if 'model' in info:
                        model = info['model']
                        st.write(f"**Class:** {type(model).__name__}")
                        if hasattr(model, 'n_estimators'):
                            st.write(f"**Estimators:** {model.n_estimators}")
                        if hasattr(model, 'feature_importances_'):
                            st.write("**Has Feature Importance:** ✅")
    
    with tabs[1]:
        st.subheader("⚡ PD Models")
        
        for name, info in pd_models.items():
            if isinstance(info, dict) and 'error' not in info:
                with st.expander(f"✅ {info.get('name', name)}", expanded=False):
                    st.write(f"**Type:** {info.get('type', 'N/A')}")
                    if 'metadata' in info:
                        st.json(info['metadata'])
    
    with tabs[2]:
        st.subheader("🏭 TG1 Models")
        
        for name, info in tg1_models.items():
            if isinstance(info, dict) and 'error' not in info:
                with st.expander(f"✅ {info.get('name', name)}", expanded=False):
                    st.write(f"**Type:** {info.get('type', 'N/A')}")
                    if 'config' in info:
                        st.json(info['config'])


def page_tickets():
    """Ticket management"""
    st.header("🎫 Smart Tickets Management")
    
    engine = get_ticket_engine()
    df = engine.export_to_dataframe()
    
    if df.empty:
        st.info("No tickets yet. Run 'Live Analysis' to generate tickets.")
        return
    
    # Filters
    col1, col2, col3 = st.columns(3)
    
    # Get available options
    available_statuses = df['status'].unique().tolist()
    default_statuses = [s for s in ["OPEN", "IN_PROGRESS"] if s in available_statuses]
    if not default_statuses:
        default_statuses = available_statuses[:2] if len(available_statuses) >= 2 else available_statuses
    
    with col1:
        filter_priority = st.multiselect("Priority", ["CRITICAL", "HIGH", "MEDIUM", "LOW"], 
                                        default=["CRITICAL", "HIGH"])
    with col2:
        filter_module = st.multiselect("Module", df['module'].unique().tolist(), 
                                      default=df['module'].unique().tolist())
    with col3:
        filter_status = st.multiselect("Status", available_statuses,
                                      default=default_statuses)
    
    # Apply filters
    filtered = df.copy()
    if filter_priority:
        filtered = filtered[filtered['priority'].isin(filter_priority)]
    if filter_module:
        filtered = filtered[filtered['module'].isin(filter_module)]
    if filter_status:
        filtered = filtered[filtered['status'].isin(filter_status)]
    
    st.info(f"📊 {len(filtered)} tickets")
    
    # Display
    display_cols = ['ticket_id', 'timestamp', 'module', 'anomaly_type', 'severity_score', 'priority', 'status']
    st.dataframe(
        filtered[display_cols].style.background_gradient(subset=['severity_score'], cmap='RdYlGn_r'),
        use_container_width=True,
        hide_index=True
    )
    
    # Export
    csv = filtered.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Export CSV", csv, "tickets.csv", "text/csv")


def page_manual_ticket():
    """Manual ticket creation"""
    st.header("✏️ Manual Ticket Creation")
    
    engine = get_ticket_engine()
    
    col1, col2 = st.columns(2)
    
    with col1:
        module = st.selectbox("Module", [m.value for m in Module])
        severity = st.slider("Severity Score", 0, 100, 70)
        confidence = st.slider("ML Confidence", 0.5, 1.0, 0.85)
    
    with col2:
        st.subheader("Metrics")
        if module == "THERMAL":
            temp = st.number_input("Temperature (°C)", 60, 120, 85)
            load = st.number_input("Load (MW)", 50, 130, 100)
            metrics = {"temperature": temp, "load": load}
        elif module == "PD":
            pd_intensity = st.number_input("PD Intensity", 10000, 1000000, 150000)
            metrics = {"pd_intensity": pd_intensity, "pd_severity": severity}
        elif module == "COOLING":
            delta_t = st.number_input("Delta T (°C)", 1, 30, 10)
            metrics = {"delta_t": delta_t}
        else:
            metrics = {"value": severity}
    
    if st.button("🎫 Generate Smart Ticket", type="primary"):
        ticket = engine.generate_smart_ticket(
            module=Module[module],
            severity_score=severity,
            metrics=metrics,
            ml_confidence=confidence
        )
        
        st.success(f"✅ Ticket created: {ticket.ticket_id}")
        
        st.markdown(f"""
        **Priority:** {ticket.priority}  
        **RUL:** {ticket.estimated_rul}  
        **Service:** {ticket.assigned_service}
        """)
        
        with st.expander("📋 Full Details"):
            st.markdown(f"**Description:** {ticket.llm_description}")
            st.markdown(f"**Root Cause:** {ticket.llm_root_cause}")
            st.markdown(f"**Recommendation:** {ticket.llm_recommendation}")


def page_settings():
    """Settings"""
    st.header("⚙️ Settings")
    
    tabs = st.tabs(["🎚️ Thresholds", "🤖 Models", "🗑️ Maintenance"])
    
    with tabs[0]:
        st.subheader("Severity Thresholds")
        critical = st.slider("Critical (≥)", 80, 100, 90)
        high = st.slider("High (≥)", 60, 80, 70)
        medium = st.slider("Medium (≥)", 30, 60, 40)
        st.info(f"LOW: < {medium}")
    
    with tabs[1]:
        st.subheader("Model Configuration")
        ml_models = load_ml_models()
        pd_models = load_pd_models()
        tg1_models = load_tg1_models()
        
        total_ml = len([m for m in ml_models.values() if isinstance(m, dict) and 'error' not in m])
        total_pd = len([m for m in pd_models.values() if isinstance(m, dict) and 'error' not in m])
        total_tg1 = len([m for m in tg1_models.values() if isinstance(m, dict) and 'error' not in m])
        
        col1, col2, col3 = st.columns(3)
        col1.metric("ML Models", total_ml)
        col2.metric("PD Models", total_pd)
        col3.metric("TG1 Models", total_tg1)
    
    with tabs[2]:
        st.subheader("Maintenance")
        engine = get_ticket_engine()
        st.metric("Tickets in DB", len(engine.tickets))
        
        if st.button("🗑️ Clear All Tickets"):
            if st.checkbox("Confirm deletion"):
                engine.tickets = []
                engine._save_tickets()
                st.success("Cleared!")
                st.rerun()


# =============================================================================
# MAIN
# =============================================================================

def main():
    st.sidebar.image("https://img.icons8.com/nolan/96/maintenance.png", width=80)
    st.sidebar.title("🎫 ML Ticketing")
    st.sidebar.markdown("Integrated System")
    st.sidebar.markdown("---")
    
    pages = {
        "🏠 Dashboard": page_dashboard,
        "🔍 Live Analysis": page_live_analysis,
        "📊 Model Details": page_model_details,
        "🎫 Tickets": page_tickets,
        "✏️ Manual Ticket": page_manual_ticket,
        "⚙️ Settings": page_settings,
    }
    
    selection = st.sidebar.radio("Navigation", list(pages.keys()))
    
    # Stats
    st.sidebar.markdown("---")
    engine = get_ticket_engine()
    stats = engine.get_statistics()
    
    st.sidebar.markdown("### 📈 Quick Stats")
    col1, col2 = st.sidebar.columns(2)
    col1.metric("Tickets", stats.get('total', 0))
    col2.metric("Open", stats.get('open', 0))
    
    # Execute
    pages[selection]()
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.caption(f"🕐 {datetime.now().strftime('%H:%M:%S')}")
    st.sidebar.caption("STEG 2026 - Nadhir")


if __name__ == "__main__":
    main()
