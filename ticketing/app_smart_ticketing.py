"""
🧠 SMART TICKETING DASHBOARD - ML + RAG + LLM
==============================================
Dashboard intelligent avec génération automatique de tickets

Auteur: Nadhir - Stage STEG 2026
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from datetime import datetime
import sys
import json

# Ajouter le chemin du module
sys.path.insert(0, str(Path(__file__).parent))
from smart_ticket_engine import (
    SmartTicketEngine, SmartTicket, KnowledgeBase, LLMGenerator,
    Priority, TicketStatus, Module, AnomalyType
)

# Configuration
st.set_page_config(
    page_title="🧠 Smart Ticketing - ML+RAG+LLM",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Chemins
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "LAST_DATA"

# CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .smart-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 25px;
        border-radius: 20px;
        margin: 15px 0;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.4);
    }
    .rag-card {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        padding: 20px;
        border-radius: 15px;
        margin: 10px 0;
    }
    .llm-card {
        background: linear-gradient(135deg, #ee0979 0%, #ff6a00 100%);
        color: white;
        padding: 20px;
        border-radius: 15px;
        margin: 10px 0;
    }
    .ml-card {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        padding: 20px;
        border-radius: 15px;
        margin: 10px 0;
    }
    .priority-critical {
        background: linear-gradient(135deg, #f5365c 0%, #f56036 100%);
        color: white;
        padding: 20px;
        border-radius: 15px;
        margin: 10px 0;
    }
    .priority-high {
        background: linear-gradient(135deg, #fb6340 0%, #fbb140 100%);
        color: white;
        padding: 20px;
        border-radius: 15px;
        margin: 10px 0;
    }
    .priority-medium {
        background: linear-gradient(135deg, #ffd600 0%, #ffab00 100%);
        color: #333;
        padding: 20px;
        border-radius: 15px;
        margin: 10px 0;
    }
    .priority-low {
        background: linear-gradient(135deg, #2dce89 0%, #2dcecc 100%);
        color: white;
        padding: 20px;
        border-radius: 15px;
        margin: 10px 0;
    }
    .tech-badge {
        display: inline-block;
        padding: 5px 15px;
        border-radius: 20px;
        font-weight: bold;
        margin: 5px;
        font-size: 0.85rem;
    }
    .badge-ml { background: #4facfe; color: white; }
    .badge-rag { background: #11998e; color: white; }
    .badge-llm { background: #ee0979; color: white; }
    .knowledge-box {
        background: #f8f9fa;
        border-left: 4px solid #11998e;
        padding: 15px;
        margin: 10px 0;
        border-radius: 0 10px 10px 0;
    }
    .llm-output {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        padding: 20px;
        border-radius: 15px;
        margin: 10px 0;
        color: #333;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# CACHE & DATA
# =============================================================================

@st.cache_resource
def get_engine():
    return SmartTicketEngine()

@st.cache_resource
def get_knowledge_base():
    return KnowledgeBase()

@st.cache_data(ttl=60)
def load_health_data():
    try:
        return pd.read_csv(DATA_DIR / "TG1_Health_Index.csv")
    except:
        return None

@st.cache_data(ttl=60)
def load_pd_data():
    try:
        return pd.read_csv(DATA_DIR / "TG1_Sousse_PD_WithScore.csv")
    except:
        return None


# =============================================================================
# PAGES
# =============================================================================

def page_dashboard():
    """Page principale"""
    st.markdown('<h1 class="main-header">🧠 Smart Ticketing System</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">ML + RAG + LLM - Génération Intelligente de Tickets</p>', unsafe_allow_html=True)
    
    # Tech badges
    st.markdown("""
    <div style="text-align: center; margin-bottom: 20px;">
        <span class="tech-badge badge-ml">🤖 Machine Learning</span>
        <span class="tech-badge badge-rag">📚 RAG Knowledge Base</span>
        <span class="tech-badge badge-llm">🧠 LLM Generation</span>
    </div>
    """, unsafe_allow_html=True)
    
    engine = get_engine()
    stats = engine.get_statistics()
    
    # KPIs
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        st.metric("📊 Total Tickets", stats.get('total', 0))
    with col2:
        st.metric("🔴 Ouverts", stats.get('open', 0))
    with col3:
        st.metric("⚠️ Critical", stats.get('by_priority', {}).get('CRITICAL', 0))
    with col4:
        st.metric("🟠 High", stats.get('by_priority', {}).get('HIGH', 0))
    with col5:
        avg_severity = stats.get('avg_severity', 0)
        st.metric("📈 Sévérité Moy.", f"{avg_severity:.1f}" if avg_severity else "N/A")
    with col6:
        avg_time = stats.get('avg_processing_time_ms', 0)
        st.metric("⚡ Temps ML", f"{avg_time:.0f}ms" if avg_time else "N/A")
    
    st.markdown("---")
    
    # Architecture
    st.subheader("🏗️ Architecture ML + RAG + LLM")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="ml-card">
            <h3>🤖 ML Detection</h3>
            <p>XGBoost, LSTM, Isolation Forest</p>
            <p>Détection anomalies</p>
            <p>SHAP Analysis</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="rag-card">
            <h3>📚 RAG Retrieval</h3>
            <p>Base de connaissances</p>
            <p>Manuels techniques</p>
            <p>Historique incidents</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="llm-card">
            <h3>🧠 LLM Generation</h3>
            <p>Descriptions intelligentes</p>
            <p>Root Cause Analysis</p>
            <p>Recommandations</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="smart-card">
            <h3>🎫 Smart Ticket</h3>
            <p>Ticket enrichi</p>
            <p>Actions prioritisées</p>
            <p>Prévention</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Workflow
    st.subheader("📈 Pipeline de Traitement")
    
    workflow_data = {
        "Étape": ["1. Détection", "2. Classification", "3. RAG Retrieval", "4. LLM Generation", "5. Ticket"],
        "Composant": ["ML Models", "Priority Engine", "Knowledge Base", "LLM Generator", "Smart Ticket"],
        "Output": ["Anomaly Score", "CRITICAL/HIGH/MED/LOW", "Procedures + History", "Description + Recommandation", "Ticket Complet"],
        "Temps (ms)": [50, 5, 30, 100, 10]
    }
    
    st.dataframe(pd.DataFrame(workflow_data), use_container_width=True, hide_index=True)
    
    # Last tickets
    st.markdown("---")
    st.subheader("🎫 Derniers Smart Tickets")
    
    df = engine.export_to_dataframe()
    if not df.empty:
        for _, ticket in df.tail(3).iterrows():
            priority_class = f"priority-{ticket['priority'].lower()}"
            st.markdown(f"""
            <div class="{priority_class}">
                <h4>🎫 {ticket['ticket_id']}</h4>
                <p><strong>Module:</strong> {ticket['module']} | <strong>Priorité:</strong> {ticket['priority']} | <strong>Sévérité:</strong> {ticket['severity_score']:.1f}/100</p>
                <p><strong>Type:</strong> {ticket['anomaly_type']}</p>
                <p><strong>⚡ Traitement:</strong> {ticket['processing_time_ms']:.1f}ms | <strong>🤖 Confiance ML:</strong> {ticket['ml_confidence']*100:.0f}%</p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("Aucun ticket généré. Utilisez la page 'Générer' pour créer un smart ticket.")


def page_generate():
    """Page de génération de tickets"""
    st.header("➕ Générer un Smart Ticket")
    
    st.markdown("""
    <div style="text-align: center; margin-bottom: 20px;">
        <span class="tech-badge badge-ml">🤖 ML</span>
        <span class="tech-badge badge-rag">📚 RAG</span>
        <span class="tech-badge badge-llm">🧠 LLM</span>
        → 🎫 Smart Ticket
    </div>
    """, unsafe_allow_html=True)
    
    engine = get_engine()
    
    tabs = st.tabs(["🤖 Génération Automatique", "✍️ Génération Manuelle", "📊 Depuis Données"])
    
    with tabs[0]:
        st.subheader("🤖 Détection & Génération Automatique")
        
        health_df = load_health_data()
        pd_df = load_pd_data()
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Données disponibles:**")
            st.write(f"- Health Index: {'✅' if health_df is not None else '❌'}")
            st.write(f"- PD Data: {'✅' if pd_df is not None else '❌'}")
        
        with col2:
            threshold = st.slider("Seuil de sévérité", 40, 90, 70)
            auto_generate = st.checkbox("Générer pour toutes les anomalies", value=False)
        
        if st.button("🔍 Analyser & Générer Smart Tickets", type="primary"):
            with st.spinner("🤖 ML Detection + 📚 RAG Retrieval + 🧠 LLM Generation..."):
                generated = []
                
                # Analyse Health Index
                if health_df is not None:
                    latest_health = health_df.iloc[-1] if len(health_df) > 0 else None
                    
                    if latest_health is not None:
                        # Check global health
                        if 'HEALTH_INDEX' in health_df.columns:
                            health_val = latest_health.get('HEALTH_INDEX', 100)
                            if health_val < threshold:
                                severity = 100 - health_val
                                ticket = engine.generate_smart_ticket(
                                    module=Module.GLOBAL,
                                    severity_score=severity,
                                    metrics={"health_index": health_val},
                                    ml_confidence=0.90
                                )
                                generated.append(ticket)
                        
                        # Check thermal
                        if 'THERMAL_SCORE' in health_df.columns:
                            thermal_val = latest_health.get('THERMAL_SCORE', 100)
                            if thermal_val < threshold:
                                severity = 100 - thermal_val
                                ticket = engine.generate_smart_ticket(
                                    module=Module.THERMAL,
                                    severity_score=severity,
                                    metrics={
                                        "temperature": 85 + (100 - thermal_val) * 0.3,
                                        "thermal_score": thermal_val,
                                        "load": 100,
                                        "delta_t": 15
                                    },
                                    shap_features={"load": 0.4, "cooling": 0.3, "ambient": 0.2},
                                    ml_confidence=0.88
                                )
                                generated.append(ticket)
                
                # Analyse PD
                if pd_df is not None and 'PD_SEVERITY_SCORE' in pd_df.columns:
                    critical_pd = pd_df[pd_df['PD_SEVERITY_SCORE'] >= threshold]
                    if len(critical_pd) > 0:
                        latest_pd = critical_pd.iloc[-1]
                        ticket = engine.generate_smart_ticket(
                            module=Module.PD,
                            severity_score=latest_pd['PD_SEVERITY_SCORE'],
                            metrics={
                                "pd_severity": latest_pd['PD_SEVERITY_SCORE'],
                                "pd_intensity": latest_pd.get('PD_INTENSITY_TOTAL', 100000),
                                "asymmetry": latest_pd.get('INTENSITY_ASYMMETRY', 20)
                            },
                            shap_features={
                                "CURRENT_TOTAL": 0.35,
                                "INTENSITY_ASYMMETRY": 0.30,
                                "PULSE_TOTAL": 0.20
                            },
                            ml_confidence=0.92
                        )
                        generated.append(ticket)
            
            if generated:
                st.success(f"✅ {len(generated)} Smart Ticket(s) généré(s)!")
                
                for ticket in generated:
                    with st.expander(f"🎫 {ticket.ticket_id} - {ticket.module} - {ticket.priority}", expanded=True):
                        display_smart_ticket(ticket)
            else:
                st.info("Aucune anomalie détectée au-dessus du seuil.")
    
    with tabs[1]:
        st.subheader("✍️ Génération Manuelle")
        
        col1, col2 = st.columns(2)
        
        with col1:
            module = st.selectbox("Module", [m.value for m in Module])
            severity = st.slider("Score de sévérité", 0, 100, 75)
            confidence = st.slider("Confiance ML", 0.5, 1.0, 0.90)
        
        with col2:
            st.write("**Métriques du module:**")
            
            if module == "THERMAL":
                temp = st.number_input("Température (°C)", 60, 120, 85)
                load = st.number_input("Charge (MW)", 50, 130, 100)
                delta_t = st.number_input("Delta T (°C)", 5, 30, 15)
                metrics = {"temperature": temp, "load": load, "delta_t": delta_t}
            elif module == "PD":
                pd_intensity = st.number_input("Intensité PD", 10000, 1000000, 200000)
                asymmetry = st.number_input("Asymétrie (%)", 0, 100, 30)
                metrics = {"pd_severity": severity, "pd_intensity": pd_intensity, "asymmetry": asymmetry}
            elif module == "COOLING":
                delta_t = st.number_input("Delta T (°C)", 1, 30, 10)
                efficiency = st.number_input("Efficacité (%)", 50, 100, 75)
                metrics = {"delta_t": delta_t, "efficiency": efficiency/100}
            else:
                freq_dev = st.number_input("Dév. Fréquence (Hz)", 0.0, 2.0, 0.3)
                asymmetry = st.number_input("Asymétrie phases (%)", 0, 20, 8)
                metrics = {"frequency_deviation": freq_dev, "asymmetry": asymmetry}
        
        st.write("**SHAP Features:**")
        shap1 = st.slider("Feature 1 contribution", 0.0, 1.0, 0.4)
        shap2 = st.slider("Feature 2 contribution", 0.0, 1.0, 0.3)
        shap3 = st.slider("Feature 3 contribution", 0.0, 1.0, 0.2)
        
        if st.button("🧠 Générer Smart Ticket", type="primary"):
            with st.spinner("🤖 ML + 📚 RAG + 🧠 LLM en cours..."):
                ticket = engine.generate_smart_ticket(
                    module=Module[module],
                    severity_score=severity,
                    metrics=metrics,
                    shap_features={"feature1": shap1, "feature2": shap2, "feature3": shap3},
                    ml_confidence=confidence
                )
            
            st.success(f"✅ Smart Ticket créé: {ticket.ticket_id}")
            display_smart_ticket(ticket)
    
    with tabs[2]:
        st.subheader("📊 Génération depuis fichier")
        st.info("Uploadez un CSV avec les colonnes: module, severity_score, metrics")
        
        uploaded = st.file_uploader("Charger CSV", type=['csv'])
        if uploaded:
            df = pd.read_csv(uploaded)
            st.dataframe(df.head())
            st.warning("Fonctionnalité en développement")


def display_smart_ticket(ticket):
    """Afficher un smart ticket complet"""
    
    # Header avec priorité
    priority_colors = {
        "CRITICAL": "#f5365c",
        "HIGH": "#fb6340",
        "MEDIUM": "#ffd600",
        "LOW": "#2dce89"
    }
    color = priority_colors.get(ticket.priority, "#667eea")
    
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, {color} 0%, {color}99 100%); 
                color: white; padding: 15px; border-radius: 15px; margin-bottom: 15px;">
        <h3>🎫 {ticket.ticket_id}</h3>
        <p><strong>Module:</strong> {ticket.module} | <strong>Priorité:</strong> {ticket.priority} | 
           <strong>Sévérité:</strong> {ticket.severity_score:.1f}/100 | <strong>RUL:</strong> {ticket.estimated_rul}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Tabs pour les différentes sections
    t1, t2, t3, t4, t5 = st.tabs(["🤖 ML Analysis", "🧠 LLM Description", "📚 RAG Context", "🛠️ Recommandations", "🔮 Prévention"])
    
    with t1:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 🎯 Détection ML")
            st.metric("Confiance", f"{ticket.ml_confidence*100:.0f}%")
            st.metric("Temps traitement", f"{ticket.processing_time_ms:.1f} ms")
        
        with col2:
            st.markdown("### 📊 SHAP Analysis")
            if ticket.shap_analysis:
                shap_df = pd.DataFrame([
                    {"Feature": k, "Contribution": v}
                    for k, v in ticket.shap_analysis.items()
                ])
                fig = px.bar(shap_df, x='Contribution', y='Feature', orientation='h',
                            color='Contribution', color_continuous_scale='RdYlGn_r')
                fig.update_layout(height=200, margin=dict(l=0, r=0, t=0, b=0))
                st.plotly_chart(fig, use_container_width=True)
    
    with t2:
        st.markdown("### 📝 Description Générée (LLM)")
        st.markdown(f"""
        <div class="llm-output">
            {ticket.llm_description}
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 🔍 Analyse Cause Racine (LLM + SHAP)")
        st.markdown(ticket.llm_root_cause)
    
    with t3:
        st.markdown("### 📚 Documents Récupérés (RAG)")
        
        if ticket.retrieved_docs:
            for i, doc in enumerate(ticket.retrieved_docs):
                st.markdown(f"""
                <div class="knowledge-box">
                    <strong>📄 Document {i+1}</strong><br>
                    {doc}
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("### 📋 Procédures Pertinentes")
        if ticket.relevant_procedures:
            for proc in ticket.relevant_procedures:
                with st.expander("📋 Procédure"):
                    st.text(proc)
        
        st.markdown("### 📜 Incidents Similaires")
        if ticket.similar_incidents:
            for inc in ticket.similar_incidents:
                with st.expander("📜 Incident historique"):
                    st.text(inc)
    
    with t4:
        st.markdown("### 🛠️ Recommandations (LLM)")
        st.markdown(ticket.llm_recommendation)
        
        st.markdown("### 👥 Assignation")
        st.info(f"**Service:** {ticket.assigned_service}")
    
    with t5:
        st.markdown("### 🔮 Mesures Préventives (LLM)")
        st.markdown(ticket.llm_prevention)


def page_tickets_list():
    """Liste des tickets"""
    st.header("📋 Liste des Smart Tickets")
    
    engine = get_engine()
    df = engine.export_to_dataframe()
    
    if df.empty:
        st.info("Aucun ticket disponible")
        return
    
    # Filtres
    col1, col2, col3 = st.columns(3)
    with col1:
        filter_priority = st.multiselect("Priorité", ["CRITICAL", "HIGH", "MEDIUM", "LOW"], 
                                        default=["CRITICAL", "HIGH", "MEDIUM", "LOW"])
    with col2:
        filter_module = st.multiselect("Module", ["THERMAL", "COOLING", "ELECTRICAL", "PD", "GLOBAL"],
                                      default=["THERMAL", "COOLING", "ELECTRICAL", "PD", "GLOBAL"])
    with col3:
        filter_status = st.multiselect("Statut", ["OPEN", "IN_PROGRESS", "RESOLVED", "CLOSED"],
                                      default=["OPEN", "IN_PROGRESS"])
    
    # Appliquer filtres
    if filter_priority:
        df = df[df['priority'].isin(filter_priority)]
    if filter_module:
        df = df[df['module'].isin(filter_module)]
    if filter_status:
        df = df[df['status'].isin(filter_status)]
    
    st.info(f"📊 {len(df)} ticket(s) trouvé(s)")
    
    # Colonnes à afficher
    display_cols = ['ticket_id', 'timestamp', 'module', 'anomaly_type', 
                   'severity_score', 'priority', 'status', 'ml_confidence', 'processing_time_ms']
    
    st.dataframe(
        df[display_cols].style.background_gradient(subset=['severity_score'], cmap='RdYlGn_r'),
        use_container_width=True,
        hide_index=True
    )
    
    # Export
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Exporter CSV", csv, "smart_tickets.csv", "text/csv")


def page_ticket_detail():
    """Détail d'un ticket"""
    st.header("🔍 Détail Smart Ticket")
    
    engine = get_engine()
    df = engine.export_to_dataframe()
    
    if df.empty:
        st.warning("Aucun ticket disponible")
        return
    
    ticket_id = st.selectbox("Sélectionner un ticket", df['ticket_id'].tolist())
    
    if ticket_id:
        ticket_data = df[df['ticket_id'] == ticket_id].iloc[0]
        
        # Recréer l'objet ticket pour l'affichage
        class TicketDisplay:
            pass
        
        ticket = TicketDisplay()
        for col in df.columns:
            setattr(ticket, col, ticket_data[col])
        
        display_smart_ticket(ticket)


def page_knowledge_base():
    """Exploration de la base de connaissances"""
    st.header("📚 Base de Connaissances (RAG)")
    
    kb = get_knowledge_base()
    
    st.markdown("""
    <div class="rag-card">
        <h3>📚 Knowledge Base pour RAG</h3>
        <p>Contient les manuels techniques, procédures et historique des incidents TG1</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Stats
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📄 Total Chunks", len(kb.chunks))
    with col2:
        thermal = len([c for c in kb.chunks if c.category == "THERMAL"])
        st.metric("🌡️ Thermal", thermal)
    with col3:
        pd_chunks = len([c for c in kb.chunks if c.category == "PD"])
        st.metric("⚡ PD", pd_chunks)
    with col4:
        incidents = len([c for c in kb.chunks if c.category == "INCIDENTS"])
        st.metric("📜 Incidents", incidents)
    
    st.markdown("---")
    
    # Recherche
    st.subheader("🔍 Recherche dans la base")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        query = st.text_input("Requête de recherche", "température surchauffe stator")
    with col2:
        category = st.selectbox("Catégorie", ["Toutes", "THERMAL", "COOLING", "ELECTRICAL", "PD", "INCIDENTS", "MAINTENANCE"])
    
    if st.button("🔍 Rechercher"):
        cat = None if category == "Toutes" else category
        results = kb.search(query, category=cat, top_k=5)
        
        st.success(f"✅ {len(results)} résultat(s) trouvé(s)")
        
        for i, chunk in enumerate(results):
            with st.expander(f"📄 {chunk.source} ({chunk.category})", expanded=i==0):
                st.markdown(f"**Keywords:** {', '.join(chunk.keywords)}")
                st.text(chunk.content)
    
    st.markdown("---")
    
    # Explorer les chunks
    st.subheader("📋 Explorer la base complète")
    
    chunks_df = pd.DataFrame([
        {
            "ID": c.chunk_id,
            "Source": c.source,
            "Catégorie": c.category,
            "Keywords": ", ".join(c.keywords[:3]),
            "Taille": len(c.content)
        }
        for c in kb.chunks
    ])
    
    st.dataframe(chunks_df, use_container_width=True, hide_index=True)


def page_analytics():
    """Analytiques"""
    st.header("📊 Analytiques Smart Ticketing")
    
    engine = get_engine()
    df = engine.export_to_dataframe()
    
    if df.empty:
        st.warning("Pas assez de données")
        return
    
    # KPIs
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Tickets", len(df))
    with col2:
        st.metric("Sévérité Moyenne", f"{df['severity_score'].mean():.1f}")
    with col3:
        st.metric("Confiance ML Moy.", f"{df['ml_confidence'].mean()*100:.0f}%")
    with col4:
        st.metric("Temps Moyen", f"{df['processing_time_ms'].mean():.0f}ms")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Distribution par Priorité")
        priority_counts = df['priority'].value_counts()
        fig = px.pie(values=priority_counts.values, names=priority_counts.index,
                    color=priority_counts.index,
                    color_discrete_map={'CRITICAL': '#f5365c', 'HIGH': '#fb6340', 
                                       'MEDIUM': '#ffd600', 'LOW': '#2dce89'})
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📊 Distribution par Module")
        module_counts = df['module'].value_counts()
        fig = px.bar(x=module_counts.index, y=module_counts.values, color=module_counts.index)
        st.plotly_chart(fig, use_container_width=True)
    
    # Sévérité par module
    st.subheader("📈 Sévérité par Module")
    fig = px.box(df, x='module', y='severity_score', color='priority',
                color_discrete_map={'CRITICAL': '#f5365c', 'HIGH': '#fb6340', 
                                   'MEDIUM': '#ffd600', 'LOW': '#2dce89'})
    st.plotly_chart(fig, use_container_width=True)
    
    # Performance ML
    st.subheader("⚡ Performance du Pipeline")
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.histogram(df, x='processing_time_ms', nbins=20, title="Distribution temps de traitement")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.histogram(df, x='ml_confidence', nbins=20, title="Distribution confiance ML")
        st.plotly_chart(fig, use_container_width=True)


def page_settings():
    """Paramètres"""
    st.header("⚙️ Paramètres")
    
    tabs = st.tabs(["🎚️ Seuils", "🧠 LLM Config", "📚 RAG Config", "🔧 Maintenance"])
    
    with tabs[0]:
        st.subheader("Configuration des Seuils de Priorité")
        
        col1, col2 = st.columns(2)
        with col1:
            critical = st.slider("Critical (≥)", 80, 100, 90)
            high = st.slider("High (≥)", 60, 80, 70)
            medium = st.slider("Medium (≥)", 30, 60, 40)
            st.caption(f"Low: < {medium}")
        
        with col2:
            st.markdown("""
            | Score | Priorité |
            |-------|----------|
            | ≥ {} | CRITICAL |
            | ≥ {} | HIGH |
            | ≥ {} | MEDIUM |
            | < {} | LOW |
            """.format(critical, high, medium, medium))
    
    with tabs[1]:
        st.subheader("Configuration LLM")
        st.info("En production: OpenAI API, Claude, ou LLM local (Llama, Mistral)")
        
        llm_provider = st.selectbox("Provider", ["Template (local)", "OpenAI GPT-4", "Anthropic Claude", "Local Llama"])
        
        if llm_provider != "Template (local)":
            api_key = st.text_input("API Key", type="password")
            temperature = st.slider("Temperature", 0.0, 1.0, 0.7)
            max_tokens = st.slider("Max Tokens", 100, 2000, 500)
    
    with tabs[2]:
        st.subheader("Configuration RAG")
        st.info("Base de connaissances pour le retrieval")
        
        kb = get_knowledge_base()
        st.metric("Chunks en base", len(kb.chunks))
        
        st.file_uploader("Ajouter un document PDF", type=['pdf'])
        st.caption("En production: Parser PDF + embedding avec sentence-transformers")
    
    with tabs[3]:
        st.subheader("Maintenance")
        
        engine = get_engine()
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Tickets en base", len(engine.tickets))
            st.caption(f"Fichier: {engine.storage_path.name}")
        
        with col2:
            if st.button("🗑️ Réinitialiser la base"):
                if st.checkbox("Confirmer suppression"):
                    engine.tickets = []
                    engine._save_tickets()
                    st.success("Base réinitialisée")
                    st.rerun()


# =============================================================================
# MAIN
# =============================================================================

def main():
    st.sidebar.image("https://img.icons8.com/nolan/96/artificial-intelligence.png", width=80)
    st.sidebar.title("🧠 Smart Ticketing")
    st.sidebar.markdown("**ML + RAG + LLM**")
    st.sidebar.markdown("---")
    
    pages = {
        "🏠 Dashboard": page_dashboard,
        "➕ Générer Ticket": page_generate,
        "📋 Liste Tickets": page_tickets_list,
        "🔍 Détail Ticket": page_ticket_detail,
        "📚 Knowledge Base": page_knowledge_base,
        "📊 Analytiques": page_analytics,
        "⚙️ Paramètres": page_settings,
    }
    
    selection = st.sidebar.radio("Navigation", list(pages.keys()))
    
    # Quick stats
    st.sidebar.markdown("---")
    engine = get_engine()
    stats = engine.get_statistics()
    
    st.sidebar.markdown("### 📈 Stats")
    col1, col2 = st.sidebar.columns(2)
    col1.metric("Total", stats.get('total', 0))
    col2.metric("Open", stats.get('open', 0))
    
    # Execute page
    pages[selection]()
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.caption(f"🕐 {datetime.now().strftime('%H:%M:%S')}")
    st.sidebar.caption("STEG 2026 - Nadhir")


if __name__ == "__main__":
    main()
