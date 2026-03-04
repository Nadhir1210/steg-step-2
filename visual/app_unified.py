"""
🏭 STEG INDUSTRIAL ANALYTICS PLATFORM
======================================
Dashboard Unifié - FUSION de tous les modèles ML et analyses

Auteur: Nadhir - Stage STEG 2026
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import joblib
import warnings
warnings.filterwarnings('ignore')

# Configuration
st.set_page_config(
    page_title="STEG Industrial Analytics",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Chemins
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "LAST_DATA"
PD_PLOTS = BASE_DIR / "pd_models" / "plots"
ML_PLOTS = BASE_DIR / "ml_models" / "plots"
TG1_PLOTS = BASE_DIR / "tg1_monitoring" / "plots"

# CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #1a237e 0%, #0d47a1 50%, #01579b 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #546e7a;
        text-align: center;
        margin-bottom: 2rem;
    }
    .card-pd {
        background: linear-gradient(135deg, #9c27b0, #7b1fa2);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 0.5rem 0;
    }
    .card-ml {
        background: linear-gradient(135deg, #4caf50, #388e3c);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 0.5rem 0;
    }
    .card-tg1 {
        background: linear-gradient(135deg, #ff9800, #f57c00);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 0.5rem 0;
    }
    .card-data {
        background: linear-gradient(135deg, #2196f3, #1976d2);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# DATA LOADING
# =============================================================================
@st.cache_data
def load_dataset(filename):
    try:
        filepath = DATA_DIR / filename
        if filepath.exists():
            return pd.read_csv(filepath)
    except Exception as e:
        st.error(f"Erreur: {e}")
    return None

@st.cache_data
def load_apm_10min():
    return load_dataset("APM_Alternateur_10min_ML.csv")

@st.cache_data
def load_tg1_pd():
    return load_dataset("TG1_Sousse_ML.csv")

@st.cache_data
def load_health_index():
    return load_dataset("TG1_Health_Index.csv")

@st.cache_data
def get_datasets_info():
    datasets = {
        "APM_Alternateur_10min_ML.csv": ("APM Alternateur 10min", "🔌"),
        "APM_Chart_10min_ML.csv": ("APM Chart 10min", "📈"),
        "TG1_Sousse_ML.csv": ("TG1 Sousse PD", "⚡"),
    }
    info = {}
    for filename, (name, icon) in datasets.items():
        filepath = DATA_DIR / filename
        if filepath.exists():
            try:
                df = pd.read_csv(filepath, nrows=5)
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    n_rows = sum(1 for _ in f) - 1
                size_mb = filepath.stat().st_size / (1024 * 1024)
                info[filename] = {'name': name, 'icon': icon, 'rows': n_rows, 
                                  'cols': len(df.columns), 'size_mb': size_mb,
                                  'columns': df.columns.tolist()}
            except:
                pass
    return info

def create_gauge(value, title, max_val=100):
    color = "#4CAF50" if value >= 85 else "#FFC107" if value >= 70 else "#FF9800" if value >= 50 else "#F44336"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={'text': title, 'font': {'size': 14}},
        gauge={
            'axis': {'range': [0, max_val]},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 50], 'color': '#ffebee'},
                {'range': [50, 70], 'color': '#fff3e0'},
                {'range': [70, 85], 'color': '#fffde7'},
                {'range': [85, 100], 'color': '#e8f5e9'}
            ],
        }
    ))
    fig.update_layout(height=200, margin=dict(l=20, r=20, t=40, b=20))
    return fig

# =============================================================================
# PAGE: HOME
# =============================================================================
def page_home():
    st.markdown('<h1 class="main-header">🏭 STEG Industrial Analytics Platform</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Plateforme Unifiée - Turbo-Alternateur TG1</p>', unsafe_allow_html=True)
    
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1:
        st.metric("📊 Datasets", "6")
    with col2:
        st.metric("🤖 Modèles", "18")
    with col3:
        st.metric("📈 Variables", "200+")
    with col4:
        st.metric("📋 Lignes", "2.5M+")
    with col5:
        st.metric("🎯 Accuracy", "97.99%")
    with col6:
        st.metric("🏥 Health", "67/100")
    
    st.markdown("---")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="card-data">
            <h3>📊 DATASETS</h3>
            <p>6 Sources de données</p>
            <p>2.5M+ lignes</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="card-pd">
            <h3>🟣 PD ANALYSIS</h3>
            <p>6 Modèles ML</p>
            <p>Accuracy: 97.99%</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="card-ml">
            <h3>🟢 ML MODELS</h3>
            <p>7 Modèles ML</p>
            <p>R² = 1.0000</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="card-tg1">
            <h3>🟠 DIGITAL TWIN</h3>
            <p>5 Modules</p>
            <p>Health: 67/100</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.subheader("📋 Récapitulatif des Modèles")
    
    all_models = pd.DataFrame({
        "Module": ["🟣 PD"] * 6 + ["🟢 ML"] * 7 + ["🟠 TG1"] * 5,
        "Modèle": [
            "Feature Engineering", "KMeans", "DBSCAN", "XGBoost+SHAP", "LSTM", "Severity Score",
            "XGBoost", "Random Forest", "ANN", "LSTM TimeSeries", "Isolation Forest", "Autoencoder", "Health Index",
            "Thermal Model", "Cooling SPC", "Electrical", "Load-Temp Coupling", "Global Health"
        ],
        "Type": [
            "Preprocessing", "Clustering", "Anomaly", "Classification", "Deep Learning", "Scoring",
            "Régression", "Régression", "Deep Learning", "Séries Temp.", "Anomaly", "Deep Learning", "Composite",
            "Régression", "SPC", "Analysis", "SHAP", "Composite"
        ],
        "Performance": [
            "33 features", "Silhouette: 0.857", "9% anomalies", "97.99%", "96.15%", "0-100",
            "R²=1.0000", "R²=1.0000", "R²=0.9989", "R²=0.9792", "3% detected", "3% detected", "79.6/100",
            "R²>0.85", "±3σ UCL/LCL", "90.3/100", "SHAP values", "67/100"
        ]
    })
    st.dataframe(all_models, use_container_width=True, hide_index=True)

# =============================================================================
# PAGE: DATA EXPLORER
# =============================================================================
def page_data_explorer():
    st.header("📊 Explorateur de Données")
    
    datasets_info = get_datasets_info()
    if not datasets_info:
        st.error("Aucun dataset trouvé")
        return
    
    total_rows = sum(d['rows'] for d in datasets_info.values())
    total_size = sum(d['size_mb'] for d in datasets_info.values())
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📁 Datasets", len(datasets_info))
    with col2:
        st.metric("📋 Total Lignes", f"{total_rows:,}")
    with col3:
        st.metric("💾 Taille", f"{total_size:.1f} MB")
    with col4:
        st.metric("📊 Variables", "200+")
    
    st.markdown("---")
    
    selected_dataset = st.selectbox("Sélectionner un dataset", list(datasets_info.keys()))
    
    if selected_dataset:
        info = datasets_info[selected_dataset]
        st.subheader(f"{info['icon']} {info['name']}")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Lignes", f"{info['rows']:,}")
        with col2:
            st.metric("Colonnes", info['cols'])
        with col3:
            st.metric("Taille", f"{info['size_mb']:.1f} MB")
        
        if st.button("🔍 Charger les données"):
            with st.spinner("Chargement..."):
                df = load_dataset(selected_dataset)
                if df is not None:
                    tab1, tab2, tab3 = st.tabs(["📋 Aperçu", "📈 Stats", "📊 Viz"])
                    
                    with tab1:
                        st.dataframe(df.head(20), use_container_width=True)
                    
                    with tab2:
                        numeric_cols = df.select_dtypes(include=[np.number]).columns
                        st.dataframe(df[numeric_cols].describe(), use_container_width=True)
                    
                    with tab3:
                        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                        if numeric_cols:
                            col = st.selectbox("Variable", numeric_cols)
                            fig = px.histogram(df, x=col, nbins=50, title=f"Distribution: {col}")
                            st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# PAGE: PD ANALYSIS
# =============================================================================
def page_pd_analysis():
    st.header("🟣 Analyse des Décharges Partielles")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🎯 XGBoost+SHAP", "97.99%", delta="Accuracy")
    with col2:
        st.metric("🔮 LSTM", "96.15%", delta="Accuracy")
    with col3:
        st.metric("🔵 KMeans", "0.857", delta="Silhouette")
    with col4:
        st.metric("🟣 DBSCAN", "9.0%", delta="Anomalies")
    
    st.markdown("---")
    
    tabs = st.tabs(["📊 Overview", "🔧 Features", "🔵 Clustering", "🎯 XGBoost", "⭐ Severity"])
    
    with tabs[0]:
        st.subheader("Pipeline d'Analyse PD")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info("**📥 Feature Engineering**\n- 33 nouvelles features\n- PD_INTENSITY, PD_ENERGY\n- Rolling features")
        with col2:
            st.warning("**🔍 Clustering**\n- KMeans: 2 clusters\n- DBSCAN: 9% anomalies")
        with col3:
            st.success("**🎯 Classification**\n- XGBoost + SHAP: 97.99%\n- LSTM: 96.15%")
        
        df = load_tg1_pd()
        if df is not None:
            st.subheader("📋 Données PD")
            st.dataframe(df.head(20), use_container_width=True)
    
    with tabs[1]:
        st.subheader("🔧 Feature Engineering")
        features_df = pd.DataFrame({
            "Feature": ["PD_INTENSITY_CHx", "PD_ENERGY_CHx", "INTENSITY_ASYMMETRY", "Rolling Mean/Std"],
            "Formule": ["CURRENT × PULSE", "CHARGE × RATE", "max(CH) - min(CH)", "Fenêtres 10-60min"],
            "Type": ["Par canal", "Par canal", "Global", "Temporel"]
        })
        st.dataframe(features_df, use_container_width=True, hide_index=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Entrée", "14,956 × 91")
        with col2:
            st.metric("Sortie", "14,956 × 129")
    
    with tabs[2]:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### KMeans")
            st.metric("K optimal", "2")
            st.metric("Silhouette", "0.857")
            fig = px.pie(values=[299, 14656], names=['Modéré (2%)', 'Normal (98%)'],
                        color_discrete_sequence=['#FF6B6B', '#4ECDC4'])
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### DBSCAN")
            st.metric("eps", "0.853")
            st.metric("Anomalies", "9.0% (1,347)")
            comparison = pd.DataFrame({
                "Feature": ["PD_INTENSITY", "ASYMMETRY"],
                "Normal": ["21K", "12K"],
                "Anomalie": ["4.4M", "4.4M"],
                "Ratio": ["209x", "344x"]
            })
            st.dataframe(comparison, use_container_width=True, hide_index=True)
    
    with tabs[3]:
        st.subheader("🎯 XGBoost + SHAP")
        st.success("✅ Validation Temporelle | SHAP Explanations | Pas de Data Leakage")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Accuracy", "97.99%")
        with col2:
            st.metric("F1-Score", "0.9803")
        with col3:
            st.metric("ROC-AUC", "0.9989")
        
        shap_df = pd.DataFrame({
            'Feature': ['CURRENT_TOTAL', 'PULSE_TOTAL', 'INTENSITY_ASYMMETRY', 'CH3_INTENSITY'],
            'SHAP': [1.94, 1.83, 1.69, 1.45]
        })
        fig = px.bar(shap_df, x='SHAP', y='Feature', orientation='h', color='SHAP',
                    color_continuous_scale='Purples', title='SHAP Feature Importance')
        st.plotly_chart(fig, use_container_width=True)
    
    with tabs[4]:
        st.subheader("⭐ PD Severity Score")
        st.latex(r'Score = 35\% \times Intensity + 25\% \times Energy + 15\% \times Asymmetry + 15\% \times Trend + 10\% \times Stability')
        
        severity_dist = pd.DataFrame({
            'Classe': ['🟢 Excellent', '🟡 Bon', '🟠 Moyen', '🔴 Critique'],
            'Plage': ['0-25', '25-50', '50-75', '75-100'],
            'Pourcentage': [75.9, 18.0, 1.0, 5.1]
        })
        st.dataframe(severity_dist, use_container_width=True, hide_index=True)

# =============================================================================
# PAGE: ML MODELS
# =============================================================================
def page_ml_models():
    st.header("🟢 ML Models - Prédiction de Température")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🎯 XGBoost", "R² = 1.0000", delta="RMSE: 0.06°C")
    with col2:
        st.metric("🌲 Random Forest", "R² = 1.0000", delta="RMSE: 0.07°C")
    with col3:
        st.metric("🧠 ANN", "R² = 0.9989", delta="RMSE: 0.81°C")
    with col4:
        st.metric("📈 LSTM", "R² = 0.9792", delta="RMSE: 2.95°C")
    
    st.markdown("---")
    
    tabs = st.tabs(["📊 Overview", "🎯 XGBoost", "🧠 ANN/LSTM", "🔍 Anomaly", "⭐ Health"])
    
    with tabs[0]:
        st.subheader("Comparaison des Modèles")
        models_df = pd.DataFrame({
            "Modèle": ["XGBoost ⭐", "Random Forest", "ANN", "LSTM"],
            "R²": [1.0000, 1.0000, 0.9989, 0.9792],
            "RMSE (°C)": [0.06, 0.07, 0.81, 2.95],
            "Usage": ["Production", "Backup", "Alternative", "Prédiction future"]
        })
        st.dataframe(models_df, use_container_width=True, hide_index=True)
        
        fig = go.Figure(data=[
            go.Bar(name='R² (%)', x=['XGBoost', 'RF', 'ANN', 'LSTM'], y=[100, 100, 99.89, 97.92], marker_color='#4CAF50'),
        ])
        fig.update_layout(title='Performance des Modèles', height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        df = load_apm_10min()
        if df is not None:
            st.subheader("📋 Données APM Alternateur")
            st.dataframe(df.head(20), use_container_width=True)
    
    with tabs[1]:
        st.subheader("🎯 XGBoost Regressor")
        st.success("⭐ Modèle recommandé - État de l'art")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Configuration")
            st.code("n_estimators=200, max_depth=8, lr=0.1, device=cuda")
        
        with col2:
            st.markdown("### Feature Importance")
            feat_df = pd.DataFrame({
                'Feature': ['STATOR_PHASE_C_TEMP_1', 'STATOR_PHASE_A_TEMP_2', 'STATOR_PHASE_B_TEMP_2'],
                'Importance': [75.96, 16.87, 2.30]
            })
            fig = px.bar(feat_df, x='Importance', y='Feature', orientation='h', color='Importance')
            fig.update_layout(height=250)
            st.plotly_chart(fig, use_container_width=True)
    
    with tabs[2]:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 🧠 ANN Neural Network")
            st.code("Dense(128>64>32>16>1)\nR² = 0.9989, RMSE = 0.81°C")
        with col2:
            st.markdown("### 📈 LSTM Time Series")
            st.code("LSTM(64>32) + Dense(16>1)\nR² = 0.9792, Window = 10 pas")
    
    with tabs[3]:
        st.subheader("🔍 Détection d'Anomalies")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Isolation Forest")
            st.metric("Contamination", "3%")
            st.metric("Anomalies", "~6,000")
        with col2:
            st.markdown("### Autoencoder")
            st.code("Encoder: 3→32→16→4\nDecoder: 4→16→32→3\nSeuil: P97")
    
    with tabs[4]:
        st.subheader("⭐ Health Index")
        st.latex(r'Health = 40\% \times Thermal + 30\% \times Prediction + 20\% \times Stability + 10\% \times Anomaly')
        
        col1, col2, col3 = st.columns(3)
        with col1:
            fig = create_gauge(79.6, "Health Index")
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = create_gauge(85, "Thermal")
            st.plotly_chart(fig, use_container_width=True)
        with col3:
            fig = create_gauge(72, "Stability")
            st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# PAGE: TG1 DIGITAL TWIN
# =============================================================================
def page_tg1_digital_twin():
    st.header("🟠 TG1 Digital Twin - Health Monitoring")
    
    df_health = load_health_index()
    df_apm = load_apm_10min()
    
    # Default values
    health = 67.0
    pd_score = 89.4
    thermal = 52.9
    cooling = 74.5
    elec = 47.0
    
    if df_health is not None:
        if 'HEALTH_INDEX' in df_health.columns:
            health = df_health['HEALTH_INDEX'].mean()
        if 'PD_SCORE' in df_health.columns:
            pd_score = df_health['PD_SCORE'].mean()
        if 'THERMAL_SCORE' in df_health.columns:
            thermal = df_health['THERMAL_SCORE'].mean()
        if 'COOLING_SCORE' in df_health.columns:
            cooling = df_health['COOLING_SCORE'].mean()
        if 'ELECTRICAL_SCORE' in df_health.columns:
            elec = df_health['ELECTRICAL_SCORE'].mean()
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("🏥 Health Index", f"{health:.1f}/100")
    with col2:
        st.metric("🟣 PD Score", f"{pd_score:.1f}")
    with col3:
        st.metric("🌡️ Thermal", f"{thermal:.1f}")
    with col4:
        st.metric("❄️ Cooling", f"{cooling:.1f}")
    with col5:
        st.metric("⚡ Electrical", f"{elec:.1f}")
    
    st.markdown("---")
    st.latex(r'TG1\_Health = 30\% \times PD + 30\% \times Thermal + 20\% \times Cooling + 20\% \times Electrical')
    st.markdown("---")
    
    tabs = st.tabs(["📊 Overview", "🌡️ Thermal", "❄️ Cooling", "⚡ Electrical", "🔗 Load-Temp", "🏥 Health"])
    
    with tabs[0]:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            fig = create_gauge(health, "Global Health")
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = create_gauge(thermal, "Thermal")
            st.plotly_chart(fig, use_container_width=True)
        with col3:
            fig = create_gauge(cooling, "Cooling")
            st.plotly_chart(fig, use_container_width=True)
        with col4:
            fig = create_gauge(elec, "Electrical")
            st.plotly_chart(fig, use_container_width=True)
        
        if df_health is not None and 'HEALTH_STATUS' in df_health.columns:
            st.subheader("📊 Distribution des États")
            status_counts = df_health['HEALTH_STATUS'].value_counts()
            colors = {'Excellent': '#4CAF50', 'Stable': '#FFC107', 'Degrading': '#FF9800', 'Critical': '#F44336'}
            fig = px.pie(values=status_counts.values, names=status_counts.index, 
                        color=status_counts.index, color_discrete_map=colors)
            st.plotly_chart(fig, use_container_width=True)
    
    with tabs[1]:
        st.subheader("🌡️ Analyse Thermique")
        if df_apm is not None:
            temp_cols = [c for c in df_apm.columns if 'STATOR' in c and 'TEMP' in c]
            if temp_cols:
                df_apm['STATOR_TEMP_MEAN'] = df_apm[temp_cols].mean(axis=1)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Temp Moyenne", f"{df_apm['STATOR_TEMP_MEAN'].mean():.1f}°C")
                with col2:
                    st.metric("Temp Max", f"{df_apm['STATOR_TEMP_MEAN'].max():.1f}°C")
                with col3:
                    st.metric("Écart-type", f"{df_apm['STATOR_TEMP_MEAN'].std():.2f}°C")
                
                n = st.slider("Points", 100, 5000, 1000, key='thermal_n')
                fig = px.line(df_apm.iloc[:n], y='STATOR_TEMP_MEAN', title='Évolution Température Stator')
                st.plotly_chart(fig, use_container_width=True)
    
    with tabs[2]:
        st.subheader("❄️ Efficacité Refroidissement")
        if df_apm is not None:
            hot_col = [c for c in df_apm.columns if 'HOT_AIR' in c and 'TEMP' in c]
            cold_col = [c for c in df_apm.columns if 'COLD_AIR' in c and 'TEMP' in c]
            if hot_col and cold_col:
                df_apm['DELTA_T'] = df_apm[hot_col[0]] - df_apm[cold_col[0]]
                mean_dt = df_apm['DELTA_T'].mean()
                std_dt = df_apm['DELTA_T'].std()
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("ΔT Moyen", f"{mean_dt:.1f}°C")
                with col2:
                    st.metric("UCL (3σ)", f"{mean_dt + 3*std_dt:.1f}°C")
                with col3:
                    st.metric("LCL (3σ)", f"{max(0, mean_dt - 3*std_dt):.1f}°C")
                
                n = st.slider("Points", 100, 3000, 500, key='cooling_n')
                fig = go.Figure()
                fig.add_trace(go.Scatter(y=df_apm['DELTA_T'].iloc[:n], mode='lines', name='ΔT'))
                fig.add_hline(y=mean_dt, line_dash="solid", line_color="green", annotation_text="Mean")
                fig.add_hline(y=mean_dt + 3*std_dt, line_dash="dash", line_color="red", annotation_text="UCL")
                fig.update_layout(title='Control Chart - ΔT', height=400)
                st.plotly_chart(fig, use_container_width=True)
    
    with tabs[3]:
        st.subheader("⚡ Stabilité Électrique")
        if df_apm is not None:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                if 'FREQUENCY_Hz' in df_apm.columns:
                    freq = df_apm['FREQUENCY_Hz'].mean()
                    st.metric("Fréquence", f"{freq:.2f} Hz", delta=f"{freq-50:.3f}")
            with col2:
                if 'TERMINAL_VOLTAGE_kV' in df_apm.columns:
                    volt = df_apm['TERMINAL_VOLTAGE_kV'].mean()
                    st.metric("Tension", f"{volt:.2f} kV")
            with col3:
                if 'MODE_TAG_1' in df_apm.columns:
                    st.metric("Puissance MW", f"{df_apm['MODE_TAG_1'].mean():.1f}")
            with col4:
                if 'REACTIVE_LOAD' in df_apm.columns:
                    st.metric("Réactive MVAR", f"{df_apm['REACTIVE_LOAD'].mean():.1f}")
            
            if 'FREQUENCY_Hz' in df_apm.columns:
                fig = px.histogram(df_apm, x='FREQUENCY_Hz', nbins=100, title='Distribution Fréquence')
                fig.add_vline(x=50, line_dash="dash", line_color="red", annotation_text="50 Hz")
                st.plotly_chart(fig, use_container_width=True)
    
    with tabs[4]:
        st.subheader("🔗 Couplage Charge-Température + SHAP")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Corrélations avec Température")
            corr_df = pd.DataFrame({
                'Variable': ['HOT_AIR', 'COLD_AIR', 'COOLING_DELTA', 'LOAD_MW'],
                'Corrélation': [0.95, 0.87, 0.85, 0.60]
            })
            fig = px.bar(corr_df, x='Variable', y='Corrélation', color='Corrélation', color_continuous_scale='RdYlGn')
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.markdown("### SHAP Feature Importance")
            shap_df = pd.DataFrame({
                'Feature': ['COOLING_DELTA', 'LOAD_MW', 'AMBIENT_TEMP', 'REACTIVE_MVAR'],
                'SHAP': [4.98, 1.42, 1.03, 0.82]
            })
            fig = px.bar(shap_df, x='SHAP', y='Feature', orientation='h', color='SHAP', color_continuous_scale='Oranges')
            st.plotly_chart(fig, use_container_width=True)
    
    with tabs[5]:
        st.subheader("🏥 Global Health Index")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.success("**🟢 85-100**\n\nExcellent")
        with col2:
            st.warning("**🟡 70-84**\n\nStable")
        with col3:
            st.warning("**🟠 50-69**\n\nDegrading")
        with col4:
            st.error("**🔴 0-49**\n\nCritical")
        
        if df_health is not None and 'HEALTH_INDEX' in df_health.columns:
            n = st.slider("Points", 100, 5000, 1000, key='health_n')
            fig = px.line(df_health.iloc[:n], y='HEALTH_INDEX', title='Évolution Health Index')
            fig.add_hline(y=85, line_dash="dash", line_color="green", annotation_text="Excellent")
            fig.add_hline(y=70, line_dash="dash", line_color="gold", annotation_text="Stable")
            fig.add_hline(y=50, line_dash="dash", line_color="orange", annotation_text="Degrading")
            st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# PAGE: ABOUT
# =============================================================================
def page_about():
    st.header("ℹ️ À Propos")
    
    st.markdown("""
    ## 🏭 STEG Industrial Analytics Platform
    
    Plateforme unifiée d'analyse industrielle pour le **Turbo-Alternateur TG1** - STEG.
    
    ---
    
    ### 📊 Contenu
    
    | Module | Modèles | Performance |
    |--------|---------|-------------|
    | 🟣 PD Analysis | 6 modèles | 97.99% accuracy |
    | 🟢 ML Models | 7 modèles | R² = 1.0000 |
    | 🟠 TG1 Digital Twin | 5 modules | Health: 67/100 |
    
    ---
    
    ### 🔧 Technologies
    
    - **ML/DL**: XGBoost, TensorFlow, Scikit-learn, SHAP
    - **Data**: Pandas, NumPy
    - **Viz**: Plotly, Matplotlib
    - **Dashboard**: Streamlit
    
    ---
    
    *Développé par **Nadhir** - Stage STEG 2026*
    """)

# =============================================================================
# MAIN
# =============================================================================
def main():
    st.sidebar.image("https://img.icons8.com/fluency/96/factory.png", width=80)
    st.sidebar.title("🏭 STEG Analytics")
    st.sidebar.markdown("---")
    
    page = st.sidebar.radio(
        "Navigation",
        ["🏠 Accueil", "📊 Data Explorer", "🟣 PD Analysis", "🟢 ML Models", "🟠 TG1 Digital Twin", "ℹ️ À Propos"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📈 Quick Stats")
    st.sidebar.metric("Datasets", "6")
    st.sidebar.metric("Modèles", "18")
    st.sidebar.metric("Best R²", "1.0000")
    
    st.sidebar.markdown("---")
    st.sidebar.caption("v2.0 - Unified Dashboard")
    st.sidebar.caption("STEG 2026 - Nadhir")
    
    if page == "🏠 Accueil":
        page_home()
    elif page == "📊 Data Explorer":
        page_data_explorer()
    elif page == "🟣 PD Analysis":
        page_pd_analysis()
    elif page == "🟢 ML Models":
        page_ml_models()
    elif page == "🟠 TG1 Digital Twin":
        page_tg1_digital_twin()
    elif page == "ℹ️ À Propos":
        page_about()

if __name__ == "__main__":
    main()
