"""
🔥 TG1 DIGITAL TWIN - DASHBOARD STREAMLIT
==========================================
Interface complète pour le monitoring du Turbo-Alternateur TG1

Auteur: Nadhir - Stage STEG 2026
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configuration de la page
st.set_page_config(
    page_title="TG1 Digital Twin",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Chemins
BASE_DIR = Path(__file__).parent
PLOTS_DIR = BASE_DIR / "plots"
DATA_DIR = BASE_DIR.parent / "LAST_DATA"

# Style CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(90deg, #1E88E5, #7B1FA2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-excellent { color: #4CAF50; }
    .metric-stable { color: #FFC107; }
    .metric-degrading { color: #FF9800; }
    .metric-critical { color: #F44336; }
    .stMetric > div { background-color: #f0f2f6; border-radius: 10px; padding: 10px; }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_health_data():
    """Charger les données Health Index"""
    try:
        return pd.read_csv(DATA_DIR / "TG1_Health_Index.csv")
    except:
        return None


@st.cache_data  
def load_apm_data():
    """Charger les données APM"""
    try:
        return pd.read_csv(DATA_DIR / "APM_Alternateur_10min_ML.csv")
    except:
        return None


@st.cache_data
def load_pd_data():
    """Charger les données PD"""
    try:
        return pd.read_csv(DATA_DIR / "TG1_Sousse_ML.csv")
    except:
        return None


def get_health_color(score):
    """Retourne la couleur selon le score"""
    if score >= 85:
        return "#4CAF50"  # Vert
    elif score >= 70:
        return "#FFC107"  # Jaune
    elif score >= 50:
        return "#FF9800"  # Orange
    else:
        return "#F44336"  # Rouge


def create_gauge(value, title, max_val=100):
    """Créer un gauge chart avec Plotly"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value,
        title={'text': title, 'font': {'size': 16}},
        delta={'reference': 80, 'increasing': {'color': "green"}, 'decreasing': {'color': "red"}},
        gauge={
            'axis': {'range': [0, max_val], 'tickwidth': 1},
            'bar': {'color': get_health_color(value)},
            'bgcolor': "white",
            'borderwidth': 2,
            'steps': [
                {'range': [0, 50], 'color': '#ffebee'},
                {'range': [50, 70], 'color': '#fff3e0'},
                {'range': [70, 85], 'color': '#fffde7'},
                {'range': [85, 100], 'color': '#e8f5e9'}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': 85
            }
        }
    ))
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20))
    return fig


def show_overview():
    """Page d'accueil avec vue d'ensemble"""
    st.markdown('<h1 class="main-header">⚡ TG1 Digital Twin</h1>', unsafe_allow_html=True)
    st.markdown("### Système de Monitoring du Turbo-Alternateur TG1 - STEG")
    
    df = load_health_data()
    
    if df is None:
        st.warning("⚠️ Exécutez d'abord le script 05_Global_Health_Index.py pour générer les données")
        df = load_apm_data()
        if df is not None:
            st.info("Affichage des données APM brutes")
    
    if df is not None:
        # Métriques principales
        col1, col2, col3, col4 = st.columns(4)
        
        if 'HEALTH_INDEX' in df.columns:
            health_mean = df['HEALTH_INDEX'].mean()
            health_status = "Excellent" if health_mean >= 85 else "Stable" if health_mean >= 70 else "Degrading" if health_mean >= 50 else "Critical"
            
            with col1:
                st.metric("🏥 Health Index", f"{health_mean:.1f}/100", 
                         delta=f"{health_status}")
            
            with col2:
                if 'THERMAL_SCORE' in df.columns:
                    st.metric("🌡️ Thermal Score", f"{df['THERMAL_SCORE'].mean():.1f}/100")
                else:
                    st.metric("🌡️ Temp Max", f"{df[df.columns[df.columns.str.contains('STATOR_TEMP', case=False)].tolist()[0] if any(df.columns.str.contains('STATOR_TEMP', case=False)) else df.columns[0]].max():.1f}°C" if len(df) > 0 else "N/A")
            
            with col3:
                if 'COOLING_SCORE' in df.columns:
                    st.metric("❄️ Cooling Score", f"{df['COOLING_SCORE'].mean():.1f}/100")
                else:
                    st.metric("⚡ Load MW", f"{df['MODE_TAG_1'].mean():.1f} MW" if 'MODE_TAG_1' in df.columns else "N/A")
            
            with col4:
                if 'ELECTRICAL_SCORE' in df.columns:
                    st.metric("⚡ Electrical Score", f"{df['ELECTRICAL_SCORE'].mean():.1f}/100")
                else:
                    st.metric("📊 Points", f"{len(df):,}")
        
        st.markdown("---")
        
        # Graphiques principaux
        col1, col2 = st.columns(2)
        
        with col1:
            if 'HEALTH_INDEX' in df.columns:
                st.subheader("📈 Évolution du Health Index")
                fig = px.line(df.iloc[:1000], y='HEALTH_INDEX', 
                             title='Health Index (derniers 1000 points)')
                fig.add_hline(y=85, line_dash="dash", line_color="green", 
                             annotation_text="Excellent")
                fig.add_hline(y=70, line_dash="dash", line_color="gold",
                             annotation_text="Stable")
                fig.add_hline(y=50, line_dash="dash", line_color="orange",
                             annotation_text="Degrading")
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.subheader("📈 Charge Active (MW)")
                if 'MODE_TAG_1' in df.columns:
                    fig = px.line(df.iloc[:1000], y='MODE_TAG_1', 
                                 title='Puissance Active (MW)')
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if 'HEALTH_INDEX' in df.columns:
                st.subheader("📊 Distribution des États")
                if 'HEALTH_STATUS' in df.columns:
                    status_counts = df['HEALTH_STATUS'].value_counts()
                    colors = {'Excellent': '#4CAF50', 'Stable': '#FFC107', 
                             'Degrading': '#FF9800', 'Critical': '#F44336'}
                    fig = px.pie(values=status_counts.values, names=status_counts.index,
                                color=status_counts.index, color_discrete_map=colors,
                                title='Distribution des États de Santé')
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.subheader("📊 Distribution des Températures")
                temp_cols = [col for col in df.columns if 'TEMP' in col and 'STATOR' in col]
                if temp_cols:
                    fig = px.box(df[temp_cols[:6]], title='Températures Stator par Phase')
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)


def show_thermal_analysis():
    """Page d'analyse thermique"""
    st.header("🌡️ Analyse Thermique")
    
    df = load_apm_data()
    if df is None:
        st.error("Données non disponibles")
        return
    
    # Colonnes de température
    temp_cols = [col for col in df.columns if 'STATOR' in col and 'TEMP' in col]
    
    if not temp_cols:
        st.warning("Pas de colonnes de température trouvées")
        return
    
    # Statistiques
    col1, col2, col3, col4 = st.columns(4)
    
    df['STATOR_TEMP_MEAN'] = df[temp_cols].mean(axis=1)
    df['STATOR_TEMP_MAX'] = df[temp_cols].max(axis=1)
    
    with col1:
        st.metric("Temp Moyenne", f"{df['STATOR_TEMP_MEAN'].mean():.1f}°C")
    with col2:
        st.metric("Temp Max", f"{df['STATOR_TEMP_MAX'].max():.1f}°C")
    with col3:
        st.metric("Écart-type", f"{df['STATOR_TEMP_MEAN'].std():.2f}°C")
    with col4:
        st.metric("Points", f"{len(df):,}")
    
    st.markdown("---")
    
    # Graphiques
    tab1, tab2, tab3 = st.tabs(["📈 Évolution", "📊 Distribution", "🔥 Heatmap"])
    
    with tab1:
        # Sélection des données
        n_points = st.slider("Nombre de points à afficher", 100, 5000, 1000)
        
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                           subplot_titles=('Température Stator', 'Charge (MW)'))
        
        fig.add_trace(go.Scatter(y=df['STATOR_TEMP_MEAN'].iloc[:n_points], 
                                 name='Temp Moyenne', line=dict(color='red')), row=1, col=1)
        fig.add_trace(go.Scatter(y=df['STATOR_TEMP_MAX'].iloc[:n_points], 
                                 name='Temp Max', line=dict(color='darkred', dash='dot')), row=1, col=1)
        
        if 'MODE_TAG_1' in df.columns:
            fig.add_trace(go.Scatter(y=df['MODE_TAG_1'].iloc[:n_points], 
                                     name='Charge MW', line=dict(color='blue')), row=2, col=1)
        
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.histogram(df, x='STATOR_TEMP_MEAN', nbins=50, 
                              title='Distribution Température Moyenne')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if 'MODE_TAG_1' in df.columns:
                fig = px.scatter(df.iloc[:2000], x='MODE_TAG_1', y='STATOR_TEMP_MEAN',
                                color='AMBIENT_AIR_TEMP_C' if 'AMBIENT_AIR_TEMP_C' in df.columns else None,
                                title='Température vs Charge')
                st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        # Heatmap par heure et jour
        if 'Hour' in df.columns and 'DayOfWeek' in df.columns:
            pivot = df.groupby(['DayOfWeek', 'Hour'])['STATOR_TEMP_MEAN'].mean().unstack()
            fig = px.imshow(pivot, title='Heatmap Température (Jour × Heure)',
                           labels=dict(x='Heure', y='Jour de la semaine', color='Temp (°C)'),
                           color_continuous_scale='RdYlBu_r')
            st.plotly_chart(fig, use_container_width=True)


def show_cooling_analysis():
    """Page d'analyse du refroidissement"""
    st.header("❄️ Analyse Refroidissement")
    
    df = load_apm_data()
    if df is None:
        st.error("Données non disponibles")
        return
    
    # Calcul Delta T
    if 'ENCLOSED_HOT_AIR_TEMP_1_degC' in df.columns and 'ENCLOSED_COLD_AIR_TEMP_1_degC' in df.columns:
        df['DELTA_T'] = df['ENCLOSED_HOT_AIR_TEMP_1_degC'] - df['ENCLOSED_COLD_AIR_TEMP_1_degC']
        
        # Statistiques
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("ΔT Moyen", f"{df['DELTA_T'].mean():.1f}°C")
        with col2:
            st.metric("Air Chaud", f"{df['ENCLOSED_HOT_AIR_TEMP_1_degC'].mean():.1f}°C")
        with col3:
            st.metric("Air Froid", f"{df['ENCLOSED_COLD_AIR_TEMP_1_degC'].mean():.1f}°C")
        with col4:
            st.metric("Efficacité", f"{(df['DELTA_T'].mean() / 25 * 100):.0f}%")
        
        st.markdown("---")
        
        # Control Chart
        st.subheader("📉 Control Chart - Delta Température")
        
        mean_dt = df['DELTA_T'].mean()
        std_dt = df['DELTA_T'].std()
        ucl = mean_dt + 3 * std_dt
        lcl = max(0, mean_dt - 3 * std_dt)
        
        n_points = st.slider("Points à afficher", 100, 3000, 500, key='cooling_slider')
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=df['DELTA_T'].iloc[:n_points], mode='lines',
                                 name='ΔT', line=dict(color='steelblue', width=1)))
        fig.add_hline(y=mean_dt, line_dash="solid", line_color="green",
                     annotation_text=f"Mean ({mean_dt:.1f}°C)")
        fig.add_hline(y=ucl, line_dash="dash", line_color="red",
                     annotation_text=f"UCL ({ucl:.1f}°C)")
        fig.add_hline(y=lcl, line_dash="dash", line_color="red",
                     annotation_text=f"LCL ({lcl:.1f}°C)")
        
        fig.update_layout(title='Control Chart - Efficacité Refroidissement',
                         xaxis_title='Index', yaxis_title='ΔT (°C)', height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        # Hot vs Cold Air
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.scatter(df.iloc[:2000], x='ENCLOSED_COLD_AIR_TEMP_1_degC', 
                            y='ENCLOSED_HOT_AIR_TEMP_1_degC',
                            color='MODE_TAG_1' if 'MODE_TAG_1' in df.columns else None,
                            title='Air Chaud vs Air Froid')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.histogram(df, x='DELTA_T', nbins=50, title='Distribution ΔT')
            fig.add_vline(x=ucl, line_dash="dash", line_color="red")
            fig.add_vline(x=lcl, line_dash="dash", line_color="red")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Colonnes de refroidissement non trouvées")


def show_electrical_analysis():
    """Page d'analyse électrique"""
    st.header("⚡ Analyse Électrique")
    
    df = load_apm_data()
    if df is None:
        st.error("Données non disponibles")
        return
    
    # Statistiques
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if 'FREQUENCY_Hz' in df.columns:
            freq_mean = df['FREQUENCY_Hz'].mean()
            st.metric("Fréquence", f"{freq_mean:.2f} Hz", delta=f"{freq_mean - 50:.3f}")
    
    with col2:
        if 'TERMINAL_VOLTAGE_kV' in df.columns:
            voltage_mean = df['TERMINAL_VOLTAGE_kV'].mean()
            st.metric("Tension", f"{voltage_mean:.2f} kV", delta=f"{(voltage_mean - 15.75):.3f}")
    
    with col3:
        if 'MODE_TAG_1' in df.columns:
            st.metric("Puissance Active", f"{df['MODE_TAG_1'].mean():.1f} MW")
    
    with col4:
        if 'REACTIVE_LOAD' in df.columns:
            st.metric("Puissance Réactive", f"{df['REACTIVE_LOAD'].mean():.1f} MVAR")
    
    st.markdown("---")
    
    tab1, tab2, tab3 = st.tabs(["📊 Fréquence", "🔌 Tension", "⚡ Puissance"])
    
    with tab1:
        if 'FREQUENCY_Hz' in df.columns:
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.histogram(df, x='FREQUENCY_Hz', nbins=50,
                                  title='Distribution Fréquence')
                fig.add_vline(x=50, line_dash="dash", line_color="green",
                             annotation_text="Nominal 50 Hz")
                fig.add_vline(x=50.5, line_dash="dot", line_color="orange")
                fig.add_vline(x=49.5, line_dash="dot", line_color="orange")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.line(df.iloc[:500], y='FREQUENCY_Hz',
                             title='Évolution Fréquence')
                fig.add_hline(y=50, line_dash="dash", line_color="green")
                st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        if 'TERMINAL_VOLTAGE_kV' in df.columns:
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.histogram(df, x='TERMINAL_VOLTAGE_kV', nbins=50,
                                  title='Distribution Tension')
                fig.add_vline(x=15.75, line_dash="dash", line_color="green",
                             annotation_text="Nominal 15.75 kV")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.line(df.iloc[:500], y='TERMINAL_VOLTAGE_kV',
                             title='Évolution Tension')
                fig.add_hline(y=15.75, line_dash="dash", line_color="green")
                st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        if 'MODE_TAG_1' in df.columns and 'REACTIVE_LOAD' in df.columns:
            # Diagramme P-Q
            df['APPARENT_POWER'] = np.sqrt(df['MODE_TAG_1']**2 + df['REACTIVE_LOAD']**2)
            df['POWER_FACTOR'] = df['MODE_TAG_1'] / (df['APPARENT_POWER'] + 0.001)
            
            fig = px.scatter(df.iloc[:2000], x='MODE_TAG_1', y='REACTIVE_LOAD',
                            color='POWER_FACTOR', color_continuous_scale='RdYlGn',
                            title='Diagramme P-Q',
                            labels={'MODE_TAG_1': 'P (MW)', 'REACTIVE_LOAD': 'Q (MVAR)'})
            fig.add_hline(y=0, line_dash="dash", line_color="gray")
            fig.add_vline(x=0, line_dash="dash", line_color="gray")
            st.plotly_chart(fig, use_container_width=True)


def show_load_temp_coupling():
    """Page Load vs Temperature Coupling avec SHAP"""
    st.header("🔗 Load vs Temperature Coupling")
    
    st.info("📊 **Analyse du couplage** entre la charge (MW) et la température du stator avec SHAP explainability")
    
    df = load_apm_data()
    if df is None:
        st.error("Données non disponibles")
        return
    
    # Calcul des features
    temp_cols = [col for col in df.columns if 'STATOR' in col and 'TEMP' in col and 'degC' in col]
    if temp_cols:
        df['STATOR_TEMP'] = df[temp_cols].mean(axis=1)
    
    df['HOT_AIR'] = df[['ENCLOSED_HOT_AIR_TEMP_1_degC', 'ENCLOSED_HOT_AIR_TEMP_2_degC']].mean(axis=1)
    df['COLD_AIR'] = df[['ENCLOSED_COLD_AIR_TEMP_1_degC', 'ENCLOSED_COLD_AIR_TEMP_2_degC']].mean(axis=1)
    df['COOLING_DELTA'] = df['HOT_AIR'] - df['COLD_AIR']
    
    # Nettoyage
    df_clean = df[(df['MODE_TAG_1'] > 5) & (df['STATOR_TEMP'] > 30)].dropna(subset=['MODE_TAG_1', 'STATOR_TEMP'])
    
    # Corrélation principale
    corr = df_clean['MODE_TAG_1'].corr(df_clean['STATOR_TEMP'])
    
    # Métriques
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Corrélation Load-Temp", f"{corr:.3f}")
    with col2:
        st.metric("Load Moyen", f"{df_clean['MODE_TAG_1'].mean():.1f} MW")
    with col3:
        st.metric("Temp Moyenne", f"{df_clean['STATOR_TEMP'].mean():.1f}°C")
    with col4:
        st.metric("Points", f"{len(df_clean):,}")
    
    st.markdown("---")
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Corrélations", "🔬 Sensibilité", "🔍 SHAP", "📊 Régimes"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Scatter: Load vs Temperature")
            sample = df_clean.sample(min(3000, len(df_clean)))
            fig = px.scatter(sample, x='MODE_TAG_1', y='STATOR_TEMP',
                            color='AMBIENT_AIR_TEMP_C' if 'AMBIENT_AIR_TEMP_C' in df.columns else None,
                            color_continuous_scale='RdYlBu_r',
                            labels={'MODE_TAG_1': 'Charge (MW)', 'STATOR_TEMP': 'Temp Stator (°C)'},
                            title=f'Load vs Temperature (r={corr:.3f})')
            # Ligne de tendance
            z = np.polyfit(df_clean['MODE_TAG_1'], df_clean['STATOR_TEMP'], 2)
            p = np.poly1d(z)
            x_line = np.linspace(df_clean['MODE_TAG_1'].min(), df_clean['MODE_TAG_1'].max(), 100)
            fig.add_trace(go.Scatter(x=x_line, y=p(x_line), mode='lines', 
                                    name='Tendance', line=dict(color='red', width=2)))
            fig.update_layout(height=450)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Matrice de Corrélation")
            corr_cols = ['MODE_TAG_1', 'REACTIVE_LOAD', 'STATOR_TEMP', 'COOLING_DELTA', 'AMBIENT_AIR_TEMP_C']
            corr_cols = [c for c in corr_cols if c in df_clean.columns]
            corr_matrix = df_clean[corr_cols].corr()
            
            fig = px.imshow(corr_matrix, text_auto='.2f', color_continuous_scale='RdBu_r',
                           title='Corrélations')
            fig.update_layout(height=450)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("📈 Courbes de Sensibilité")
        
        st.markdown("""
        Les courbes de sensibilité montrent comment la température prédite varie 
        quand on modifie une seule variable (les autres restant à leur valeur médiane).
        """)
        
        # Charger config si disponible
        try:
            config = joblib.load(PLOTS_DIR / "04_coupling_config.pkl")
            sensitivity_curves = config['sensitivity_curves']
            
            col1, col2 = st.columns(2)
            
            for idx, (feature, (curve_x, curve_y)) in enumerate(sensitivity_curves.items()):
                with col1 if idx % 2 == 0 else col2:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=curve_x, y=curve_y, mode='lines', 
                                            fill='tozeroy', line=dict(color='steelblue', width=2)))
                    sensitivity = (curve_y[-1] - curve_y[0]) / (curve_x[-1] - curve_x[0])
                    fig.update_layout(title=f'{feature} (Sens: {sensitivity:.4f} °C/unit)',
                                     xaxis_title=feature, yaxis_title='Temp Prédite (°C)', height=300)
                    st.plotly_chart(fig, use_container_width=True)
            
            # Tableau de sensibilité
            st.subheader("📋 Résumé des Sensibilités")
            sens_data = []
            for feature, (curve_x, curve_y) in sensitivity_curves.items():
                sensitivity = (curve_y[-1] - curve_y[0]) / (curve_x[-1] - curve_x[0])
                sens_data.append({'Feature': feature, 'Sensibilité (°C/unit)': f"{sensitivity:.4f}"})
            st.dataframe(pd.DataFrame(sens_data), use_container_width=True, hide_index=True)
            
        except:
            st.warning("⚠️ Exécutez d'abord `04_Load_Temperature_Coupling.py` pour générer les données")
    
    with tab3:
        st.subheader("🔍 SHAP Feature Importance")
        
        st.markdown("""
        **SHAP** (SHapley Additive exPlanations) explique l'impact de chaque feature 
        sur les prédictions du modèle.
        """)
        
        # Afficher les images SHAP si disponibles
        col1, col2 = st.columns(2)
        
        shap_beeswarm = PLOTS_DIR / "04_shap_beeswarm.png"
        shap_dependence = PLOTS_DIR / "04_shap_dependence.png"
        
        with col1:
            if shap_beeswarm.exists():
                st.image(str(shap_beeswarm), caption="SHAP Beeswarm Plot")
            else:
                st.info("SHAP Beeswarm non disponible")
        
        with col2:
            if shap_dependence.exists():
                st.image(str(shap_dependence), caption="SHAP Dependence Plots")
            else:
                st.info("SHAP Dependence non disponible")
        
        # SHAP importance
        try:
            config = joblib.load(PLOTS_DIR / "04_coupling_config.pkl")
            if config.get('shap_available'):
                st.success("✅ SHAP Analysis disponible")
                
                st.markdown("""
                **Interprétation SHAP:**
                - **COOLING_DELTA** (4.98): Variable la plus influente - l'efficacité du refroidissement
                - **LOAD_MW** (1.42): La charge a un impact modéré
                - **AMBIENT_TEMP** (1.03): La température ambiante affecte aussi
                - **REACTIVE_MVAR** (0.82): La puissance réactive a un faible impact
                """)
        except:
            pass
    
    with tab4:
        st.subheader("📊 Réponse Thermique par Régime de Charge")
        
        # Créer des bins de charge
        df_clean['LOAD_BIN'] = pd.cut(df_clean['MODE_TAG_1'], 
                                      bins=[0, 50, 100, 150, 200],
                                      labels=['0-50 MW', '50-100 MW', '100-150 MW', '150-200 MW'])
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.box(df_clean.dropna(subset=['LOAD_BIN']), 
                        x='LOAD_BIN', y='STATOR_TEMP',
                        color='LOAD_BIN',
                        title='Distribution Températures par Régime',
                        labels={'LOAD_BIN': 'Régime de Charge', 'STATOR_TEMP': 'Temp Stator (°C)'})
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Stats par régime
            stats = df_clean.groupby('LOAD_BIN').agg({
                'STATOR_TEMP': ['mean', 'std', 'min', 'max'],
                'MODE_TAG_1': 'count'
            }).round(2)
            stats.columns = ['Temp Moy', 'Temp Std', 'Temp Min', 'Temp Max', 'Count']
            st.dataframe(stats, use_container_width=True)
        
        # Coefficient de couplage
        st.subheader("📐 Coefficient de Couplage dT/dP")
        
        st.markdown("Le coefficient **dT/dP** indique combien la température augmente par MW de charge supplémentaire.")
        
        coef_data = []
        for regime in ['0-50 MW', '50-100 MW', '100-150 MW', '150-200 MW']:
            subset = df_clean[df_clean['LOAD_BIN'] == regime]
            if len(subset) > 10:
                corr = subset['MODE_TAG_1'].corr(subset['STATOR_TEMP'])
                coef = np.polyfit(subset['MODE_TAG_1'], subset['STATOR_TEMP'], 1)[0]
                coef_data.append({
                    'Régime': regime,
                    'Points': len(subset),
                    'Corrélation': f"{corr:.3f}",
                    'dT/dP (°C/MW)': f"{coef:.4f}"
                })
        
        if coef_data:
            st.dataframe(pd.DataFrame(coef_data), use_container_width=True, hide_index=True)


def show_health_dashboard():
    """Dashboard Health Index complet"""
    st.header("🏥 Global Health Index Dashboard")
    
    df = load_health_data()
    
    if df is None or 'HEALTH_INDEX' not in df.columns:
        st.warning("⚠️ Exécutez d'abord le script 05_Global_Health_Index.py")
        st.code("""
# Exécuter dans le terminal:
cd tg1_monitoring
python 01_Thermal_Health_Model.py
python 02_Cooling_Efficiency.py
python 03_Electrical_Stability.py
python 05_Global_Health_Index.py
        """)
        return
    
    # Gauges des scores
    st.subheader("📊 Scores des Composants")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if 'PD_SCORE' in df.columns:
            fig = create_gauge(df['PD_SCORE'].mean(), "PD Score")
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if 'THERMAL_SCORE' in df.columns:
            fig = create_gauge(df['THERMAL_SCORE'].mean(), "Thermal Score")
            st.plotly_chart(fig, use_container_width=True)
    
    with col3:
        if 'COOLING_SCORE' in df.columns:
            fig = create_gauge(df['COOLING_SCORE'].mean(), "Cooling Score")
            st.plotly_chart(fig, use_container_width=True)
    
    with col4:
        if 'ELECTRICAL_SCORE' in df.columns:
            fig = create_gauge(df['ELECTRICAL_SCORE'].mean(), "Electrical Score")
            st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Health Index principal
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📈 Évolution du Health Index")
        
        n_points = st.slider("Points à afficher", 100, 5000, 1000, key='health_slider')
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=df['HEALTH_INDEX'].iloc[:n_points], 
                                 mode='lines', name='Health Index',
                                 line=dict(color='steelblue', width=1)))
        
        if 'HEALTH_ROLL_MEAN' in df.columns:
            fig.add_trace(go.Scatter(y=df['HEALTH_ROLL_MEAN'].iloc[:n_points],
                                     mode='lines', name='Moyenne Mobile',
                                     line=dict(color='darkblue', width=2)))
        
        fig.add_hrect(y0=85, y1=100, fillcolor="green", opacity=0.1,
                     annotation_text="Excellent", annotation_position="right")
        fig.add_hrect(y0=70, y1=85, fillcolor="yellow", opacity=0.1,
                     annotation_text="Stable")
        fig.add_hrect(y0=50, y1=70, fillcolor="orange", opacity=0.1,
                     annotation_text="Degrading")
        fig.add_hrect(y0=0, y1=50, fillcolor="red", opacity=0.1,
                     annotation_text="Critical")
        
        fig.update_layout(height=500, xaxis_title='Index', yaxis_title='Health Index')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📊 Distribution")
        
        if 'HEALTH_STATUS' in df.columns:
            status_counts = df['HEALTH_STATUS'].value_counts()
            colors = {'Excellent': '#4CAF50', 'Stable': '#FFC107', 
                     'Degrading': '#FF9800', 'Critical': '#F44336'}
            
            fig = px.pie(values=status_counts.values, names=status_counts.index,
                        color=status_counts.index, color_discrete_map=colors,
                        hole=0.4)
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Statistiques
        st.markdown("**Statistiques:**")
        st.write(f"- Moyenne: **{df['HEALTH_INDEX'].mean():.1f}**/100")
        st.write(f"- Médiane: **{df['HEALTH_INDEX'].median():.1f}**/100")
        st.write(f"- Min: **{df['HEALTH_INDEX'].min():.1f}**")
        st.write(f"- Max: **{df['HEALTH_INDEX'].max():.1f}**")


def show_alerts():
    """Page des alertes et recommandations"""
    st.header("⚠️ Alertes et Recommandations")
    
    df = load_health_data()
    
    if df is None or 'HEALTH_INDEX' not in df.columns:
        st.warning("Données Health Index non disponibles")
        return
    
    # Alertes actives
    st.subheader("🚨 Alertes Actives")
    
    if 'HEALTH_STATUS' in df.columns:
        critical_count = (df['HEALTH_STATUS'] == 'Critical').sum()
        degrading_count = (df['HEALTH_STATUS'] == 'Degrading').sum()
        
        if critical_count > 0:
            st.error(f"🔴 **{critical_count}** périodes CRITICAL détectées - Intervention requise")
        
        if degrading_count > 0:
            st.warning(f"🟠 **{degrading_count}** périodes DEGRADING - Surveillance accrue recommandée")
        
        if critical_count == 0 and degrading_count == 0:
            st.success("✅ Aucune alerte active - Système en bon état")
    
    st.markdown("---")
    
    # Analyse des composants faibles
    st.subheader("📊 Analyse des Points Faibles")
    
    scores = {}
    if 'PD_SCORE' in df.columns:
        scores['PD'] = df['PD_SCORE'].mean()
    if 'THERMAL_SCORE' in df.columns:
        scores['Thermal'] = df['THERMAL_SCORE'].mean()
    if 'COOLING_SCORE' in df.columns:
        scores['Cooling'] = df['COOLING_SCORE'].mean()
    if 'ELECTRICAL_SCORE' in df.columns:
        scores['Electrical'] = df['ELECTRICAL_SCORE'].mean()
    
    if scores:
        sorted_scores = sorted(scores.items(), key=lambda x: x[1])
        
        for i, (component, score) in enumerate(sorted_scores):
            if i == 0:
                st.warning(f"⚠️ **{component}**: {score:.1f}/100 - Point le plus faible")
            elif score < 70:
                st.info(f"📌 **{component}**: {score:.1f}/100 - À surveiller")
            else:
                st.success(f"✅ **{component}**: {score:.1f}/100 - OK")
    
    st.markdown("---")
    
    # Recommandations
    st.subheader("💡 Recommandations")
    
    recommendations = []
    
    if 'THERMAL_SCORE' in df.columns and df['THERMAL_SCORE'].mean() < 70:
        recommendations.append("🌡️ **Thermal**: Vérifier le système de ventilation et les filtres à air")
    
    if 'COOLING_SCORE' in df.columns and df['COOLING_SCORE'].mean() < 70:
        recommendations.append("❄️ **Cooling**: Inspecter les échangeurs de chaleur")
    
    if 'ELECTRICAL_SCORE' in df.columns and df['ELECTRICAL_SCORE'].mean() < 70:
        recommendations.append("⚡ **Electrical**: Vérifier les connexions et la synchronisation réseau")
    
    if 'PD_SCORE' in df.columns and df['PD_SCORE'].mean() < 70:
        recommendations.append("🟣 **PD**: Planifier une inspection de l'isolation")
    
    if recommendations:
        for rec in recommendations:
            st.markdown(f"- {rec}")
    else:
        st.success("✅ Aucune recommandation particulière - Continuer la surveillance normale")


def show_about():
    """Page À propos"""
    st.header("ℹ️ À Propos")
    
    st.markdown("""
    ### 🔥 TG1 Digital Twin - Système de Monitoring
    
    Ce dashboard fait partie d'un système complet de **Digital Twin** pour le monitoring
    du Turbo-Alternateur TG1 de la centrale STEG.
    
    ---
    
    ### 📊 Modules Disponibles
    
    | Module | Description | Fichier |
    |--------|-------------|---------|
    | 🌡️ Thermal Health | Modélisation thermique + anomalies | `01_Thermal_Health_Model.py` |
    | ❄️ Cooling Efficiency | Efficacité refroidissement | `02_Cooling_Efficiency.py` |
    | ⚡ Electrical Stability | Stabilité électrique | `03_Electrical_Stability.py` |
    | 🔗 Load-Temp Coupling | Couplage Charge-Température + SHAP | `04_Load_Temperature_Coupling.py` |
    | 🏥 Global Health Index | Score santé global | `05_Global_Health_Index.py` |
    
    ---
    
    ### 🎯 Formule Health Index
    
    ```
    HEALTH_INDEX = 30% × PD + 30% × Thermal + 20% × Cooling + 20% × Electrical
    ```
    
    | Score | État | Couleur |
    |-------|------|---------|
    | 85-100 | Excellent | 🟢 Vert |
    | 70-84 | Stable | 🟡 Jaune |
    | 50-69 | Degrading | 🟠 Orange |
    | 0-49 | Critical | 🔴 Rouge |
    
    ---
    
    ### 📁 Structure
    
    ```
    tg1_monitoring/
    ├── 01_Thermal_Health_Model.py
    ├── 02_Cooling_Efficiency.py
    ├── 03_Electrical_Stability.py
    ├── 04_Load_Temperature_Coupling.py  ⭐ NEW
    ├── 05_Global_Health_Index.py
    ├── app_streamlit.py (ce dashboard)
    ├── README.md
    └── plots/
    ```
    
    ---
    
    *Développé par Nadhir - Stage STEG 2026*
    """)


# Navigation principale
def main():
    # Sidebar
    st.sidebar.image("https://img.icons8.com/color/96/000000/wind-turbine.png", width=80)
    st.sidebar.title("Navigation")
    
    pages = {
        "🏠 Vue d'ensemble": show_overview,
        "🏥 Health Index": show_health_dashboard,
        "🌡️ Analyse Thermique": show_thermal_analysis,
        "❄️ Refroidissement": show_cooling_analysis,
        "⚡ Électrique": show_electrical_analysis,
        "🔗 Load-Temp Coupling": show_load_temp_coupling,
        "⚠️ Alertes": show_alerts,
        "ℹ️ À Propos": show_about
    }
    
    selection = st.sidebar.radio("Aller à", list(pages.keys()))
    
    # Afficher la page sélectionnée
    pages[selection]()
    
    # Footer sidebar
    st.sidebar.markdown("---")
    
    df = load_health_data()
    if df is not None and 'HEALTH_INDEX' in df.columns:
        health = df['HEALTH_INDEX'].mean()
        color = get_health_color(health)
        st.sidebar.markdown(f"""
        **Health Index:**
        <h2 style='color: {color}'>{health:.1f}/100</h2>
        """, unsafe_allow_html=True)
    
    st.sidebar.markdown("---")
    st.sidebar.caption("TG1 Digital Twin v1.1 - STEG 2026")


if __name__ == "__main__":
    main()
