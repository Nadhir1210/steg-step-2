"""
🎫 MAINTENANCE TICKETING DASHBOARD
===================================
Dashboard Streamlit pour la gestion des tickets de maintenance

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

# Ajouter le chemin du module
sys.path.insert(0, str(Path(__file__).parent))
from ticket_engine import (
    TicketEngine, MaintenanceTicket, Priority, TicketStatus, 
    Module, AnomalyType, AnomalyTicketIntegrator
)

# Configuration
st.set_page_config(
    page_title="🎫 Maintenance Ticketing - STEG",
    page_icon="🎫",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Chemins
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "LAST_DATA"

# CSS personnalisé
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(90deg, #E53935, #FF9800, #4CAF50);
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
    .ticket-critical {
        background: linear-gradient(135deg, #f44336, #c62828);
        color: white;
        padding: 20px;
        border-radius: 15px;
        margin: 10px 0;
        box-shadow: 0 4px 15px rgba(244, 67, 54, 0.4);
    }
    .ticket-high {
        background: linear-gradient(135deg, #ff9800, #f57c00);
        color: white;
        padding: 20px;
        border-radius: 15px;
        margin: 10px 0;
        box-shadow: 0 4px 15px rgba(255, 152, 0, 0.4);
    }
    .ticket-medium {
        background: linear-gradient(135deg, #ffc107, #ffa000);
        color: #333;
        padding: 20px;
        border-radius: 15px;
        margin: 10px 0;
    }
    .ticket-low {
        background: linear-gradient(135deg, #4caf50, #388e3c);
        color: white;
        padding: 20px;
        border-radius: 15px;
        margin: 10px 0;
    }
    .stat-card {
        background: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        text-align: center;
    }
    .priority-badge {
        display: inline-block;
        padding: 5px 15px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 0.9rem;
    }
    .priority-critical { background-color: #f44336; color: white; }
    .priority-high { background-color: #ff9800; color: white; }
    .priority-medium { background-color: #ffc107; color: #333; }
    .priority-low { background-color: #4caf50; color: white; }
    .status-open { background-color: #2196f3; color: white; padding: 3px 10px; border-radius: 10px; }
    .status-progress { background-color: #ff9800; color: white; padding: 3px 10px; border-radius: 10px; }
    .status-resolved { background-color: #4caf50; color: white; padding: 3px 10px; border-radius: 10px; }
    .status-closed { background-color: #9e9e9e; color: white; padding: 3px 10px; border-radius: 10px; }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# DATA LOADING
# =============================================================================

@st.cache_resource
def get_ticket_engine():
    """Obtenir l'instance du moteur de tickets"""
    return TicketEngine()

@st.cache_data(ttl=60)
def load_health_data():
    """Charger les données de santé"""
    try:
        return pd.read_csv(DATA_DIR / "TG1_Health_Index.csv")
    except:
        return None

@st.cache_data(ttl=60)
def load_pd_data():
    """Charger les données PD"""
    try:
        return pd.read_csv(DATA_DIR / "TG1_Sousse_PD_WithScore.csv")
    except:
        return None


# =============================================================================
# HELPERS
# =============================================================================

def get_priority_color(priority: str) -> str:
    """Obtenir la couleur selon la priorité"""
    colors = {
        "CRITICAL": "#f44336",
        "HIGH": "#ff9800",
        "MEDIUM": "#ffc107",
        "LOW": "#4caf50"
    }
    return colors.get(priority, "#9e9e9e")

def get_status_color(status: str) -> str:
    """Obtenir la couleur selon le statut"""
    colors = {
        "OPEN": "#2196f3",
        "IN_PROGRESS": "#ff9800",
        "PENDING": "#9c27b0",
        "RESOLVED": "#4caf50",
        "CLOSED": "#9e9e9e",
        "AUTO_CLOSED": "#607d8b"
    }
    return colors.get(status, "#9e9e9e")

def format_ticket_card(ticket: MaintenanceTicket) -> str:
    """Formater un ticket en HTML"""
    priority_class = f"ticket-{ticket.priority.lower()}"
    return f"""
    <div class="{priority_class}">
        <h4>🎫 {ticket.ticket_id}</h4>
        <p><strong>Module:</strong> {ticket.module} | <strong>Type:</strong> {ticket.anomaly_type}</p>
        <p><strong>Sévérité:</strong> {ticket.severity_score:.1f}/100 | <strong>RUL:</strong> {ticket.estimated_rul}</p>
        <p><strong>Action:</strong> {ticket.recommended_action}</p>
        <p><strong>Service:</strong> {ticket.assigned_service}</p>
    </div>
    """


# =============================================================================
# PAGES
# =============================================================================

def page_dashboard():
    """Page principale - Dashboard"""
    st.markdown('<h1 class="main-header">🎫 Maintenance Ticketing System</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Système automatisé de gestion des incidents - TG1 STEG</p>', unsafe_allow_html=True)
    
    engine = get_ticket_engine()
    stats = engine.get_statistics()
    
    # KPIs principaux
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        st.metric("📊 Total Tickets", stats.get('total', 0))
    with col2:
        st.metric("🔴 Ouverts", stats.get('open', 0), 
                 delta=f"-{stats.get('resolved', 0)} résolus" if stats.get('resolved', 0) > 0 else None)
    with col3:
        st.metric("⚠️ Critical", stats.get('by_priority', {}).get('CRITICAL', 0),
                 delta="Urgence" if stats.get('by_priority', {}).get('CRITICAL', 0) > 0 else None,
                 delta_color="inverse")
    with col4:
        st.metric("🟠 High", stats.get('by_priority', {}).get('HIGH', 0))
    with col5:
        st.metric("🟡 Medium", stats.get('by_priority', {}).get('MEDIUM', 0))
    with col6:
        st.metric("✅ Fermés", stats.get('closed', 0))
    
    st.markdown("---")
    
    # Tickets par priorité et module
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Distribution par Priorité")
        priority_data = stats.get('by_priority', {})
        if sum(priority_data.values()) > 0:
            fig = go.Figure(data=[go.Pie(
                labels=list(priority_data.keys()),
                values=list(priority_data.values()),
                hole=0.4,
                marker_colors=['#f44336', '#ff9800', '#ffc107', '#4caf50']
            )])
            fig.update_layout(height=300, margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Aucun ticket")
    
    with col2:
        st.subheader("🏭 Distribution par Module")
        module_data = stats.get('by_module', {})
        if sum(module_data.values()) > 0:
            fig = go.Figure(data=[go.Bar(
                x=list(module_data.keys()),
                y=list(module_data.values()),
                marker_color=['#f44336', '#2196f3', '#9c27b0', '#ff9800', '#4caf50']
            )])
            fig.update_layout(height=300, margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Aucun ticket")
    
    st.markdown("---")
    
    # Tickets critiques et urgents
    st.subheader("🚨 Tickets Urgents (Critical + High)")
    
    critical_tickets = engine.get_tickets_by_priority(Priority.CRITICAL)
    high_tickets = engine.get_tickets_by_priority(Priority.HIGH)
    urgent_tickets = critical_tickets + high_tickets
    
    if urgent_tickets:
        for ticket in urgent_tickets[:5]:  # Top 5
            st.markdown(format_ticket_card(ticket), unsafe_allow_html=True)
    else:
        st.success("✅ Aucun ticket urgent en cours")
    
    # Workflow
    st.markdown("---")
    st.subheader("📈 Workflow du Système")
    
    col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
    with col1:
        st.markdown("### 1️⃣\n**Détection**\nAnomalies ML")
    with col2:
        st.markdown("### →")
    with col3:
        st.markdown("### 2️⃣\n**Scoring**\n0-100")
    with col4:
        st.markdown("### →")
    with col5:
        st.markdown("### 3️⃣\n**Ticket**\nAuto-généré")
    with col6:
        st.markdown("### →")
    with col7:
        st.markdown("### 4️⃣\n**Action**\nMaintenance")


def page_tickets_list():
    """Page liste des tickets"""
    st.header("📋 Liste des Tickets")
    
    engine = get_ticket_engine()
    
    # Filtres
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        filter_status = st.multiselect(
            "Statut",
            ["OPEN", "IN_PROGRESS", "PENDING", "RESOLVED", "CLOSED", "AUTO_CLOSED"],
            default=["OPEN", "IN_PROGRESS"]
        )
    
    with col2:
        filter_priority = st.multiselect(
            "Priorité",
            ["CRITICAL", "HIGH", "MEDIUM", "LOW"],
            default=["CRITICAL", "HIGH", "MEDIUM", "LOW"]
        )
    
    with col3:
        filter_module = st.multiselect(
            "Module",
            ["THERMAL", "COOLING", "ELECTRICAL", "PD", "GLOBAL"],
            default=["THERMAL", "COOLING", "ELECTRICAL", "PD", "GLOBAL"]
        )
    
    with col4:
        sort_by = st.selectbox(
            "Trier par",
            ["timestamp", "severity_score", "priority"]
        )
    
    # Récupérer et filtrer les tickets
    df = engine.export_to_dataframe()
    
    if not df.empty:
        # Appliquer les filtres
        if filter_status:
            df = df[df['status'].isin(filter_status)]
        if filter_priority:
            df = df[df['priority'].isin(filter_priority)]
        if filter_module:
            df = df[df['module'].isin(filter_module)]
        
        # Trier
        ascending = sort_by != 'severity_score'
        df = df.sort_values(sort_by, ascending=ascending)
        
        # Afficher le nombre de résultats
        st.info(f"📊 {len(df)} ticket(s) trouvé(s)")
        
        # Styliser le DataFrame
        def style_priority(val):
            colors = {
                'CRITICAL': 'background-color: #ffebee; color: #c62828;',
                'HIGH': 'background-color: #fff3e0; color: #e65100;',
                'MEDIUM': 'background-color: #fffde7; color: #f57f17;',
                'LOW': 'background-color: #e8f5e9; color: #2e7d32;'
            }
            return colors.get(val, '')
        
        def style_status(val):
            colors = {
                'OPEN': 'background-color: #e3f2fd; color: #1565c0;',
                'IN_PROGRESS': 'background-color: #fff3e0; color: #e65100;',
                'RESOLVED': 'background-color: #e8f5e9; color: #2e7d32;',
                'CLOSED': 'background-color: #f5f5f5; color: #616161;',
                'AUTO_CLOSED': 'background-color: #eceff1; color: #546e7a;'
            }
            return colors.get(val, '')
        
        # Sélectionner les colonnes à afficher
        display_cols = ['ticket_id', 'timestamp', 'module', 'anomaly_type', 
                       'severity_score', 'priority', 'status', 'assigned_service']
        
        styled_df = df[display_cols].style\
            .applymap(style_priority, subset=['priority'])\
            .applymap(style_status, subset=['status'])\
            .format({'severity_score': '{:.1f}'})
        
        st.dataframe(styled_df, use_container_width=True, hide_index=True, height=500)
        
        # Export
        col1, col2 = st.columns([1, 4])
        with col1:
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Exporter CSV",
                csv,
                "tickets_export.csv",
                "text/csv"
            )
    else:
        st.info("Aucun ticket disponible")


def page_ticket_detail():
    """Page détail d'un ticket"""
    st.header("🔍 Détail du Ticket")
    
    engine = get_ticket_engine()
    df = engine.export_to_dataframe()
    
    if df.empty:
        st.warning("Aucun ticket disponible")
        return
    
    # Sélection du ticket
    ticket_ids = df['ticket_id'].tolist()
    selected_id = st.selectbox("Sélectionner un ticket", ticket_ids)
    
    if selected_id:
        ticket_data = df[df['ticket_id'] == selected_id].iloc[0]
        
        # En-tête
        priority = ticket_data['priority']
        priority_color = get_priority_color(priority)
        
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, {priority_color}, {priority_color}99); 
                    color: white; padding: 20px; border-radius: 15px; margin-bottom: 20px;">
            <h2>🎫 {ticket_data['ticket_id']}</h2>
            <p style="font-size: 1.2rem;">
                <strong>Priorité:</strong> {priority} | 
                <strong>Statut:</strong> {ticket_data['status']} |
                <strong>Sévérité:</strong> {ticket_data['severity_score']:.1f}/100
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Détails
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📋 Informations")
            st.write(f"**Module:** {ticket_data['module']}")
            st.write(f"**Type d'anomalie:** {ticket_data['anomaly_type']}")
            st.write(f"**Service assigné:** {ticket_data['assigned_service']}")
            st.write(f"**RUL estimé:** {ticket_data['estimated_rul']}")
            st.write(f"**Créé le:** {ticket_data['timestamp']}")
            st.write(f"**Mis à jour:** {ticket_data['updated_at']}")
        
        with col2:
            st.subheader("🔧 Actions")
            st.write(f"**Description:** {ticket_data['description']}")
            st.write(f"**Action recommandée:** {ticket_data['recommended_action']}")
            
            st.subheader("🔍 Cause Racine")
            st.text(ticket_data.get('root_cause', 'Non définie'))
        
        # Actions
        st.markdown("---")
        st.subheader("✏️ Mettre à jour le ticket")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            new_status = st.selectbox(
                "Nouveau statut",
                ["OPEN", "IN_PROGRESS", "PENDING", "RESOLVED", "CLOSED"],
                index=["OPEN", "IN_PROGRESS", "PENDING", "RESOLVED", "CLOSED"].index(
                    ticket_data['status']) if ticket_data['status'] in 
                    ["OPEN", "IN_PROGRESS", "PENDING", "RESOLVED", "CLOSED"] else 0
            )
        
        with col2:
            resolution_notes = st.text_area("Notes de résolution", 
                                           value=ticket_data.get('resolution_notes', ''))
        
        with col3:
            if st.button("💾 Sauvegarder"):
                engine.update_ticket_status(
                    selected_id, 
                    TicketStatus[new_status],
                    resolution_notes
                )
                st.success("✅ Ticket mis à jour!")
                st.rerun()


def page_generate_tickets():
    """Page de génération de tickets"""
    st.header("➕ Générer des Tickets")
    
    engine = get_ticket_engine()
    
    tabs = st.tabs(["🤖 Auto-détection", "✍️ Manuel", "📊 Depuis Données"])
    
    with tabs[0]:
        st.subheader("🤖 Détection Automatique")
        st.info("Génère des tickets basés sur l'analyse des données actuelles")
        
        health_df = load_health_data()
        pd_df = load_pd_data()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Données Health Index:**", "✅ Chargées" if health_df is not None else "❌ Non disponibles")
            st.write("**Données PD:**", "✅ Chargées" if pd_df is not None else "❌ Non disponibles")
        
        with col2:
            threshold = st.slider("Seuil d'alerte", 40, 90, 70)
        
        if st.button("🔍 Analyser et Générer"):
            integrator = AnomalyTicketIntegrator(engine)
            generated = []
            
            with st.spinner("Analyse en cours..."):
                if health_df is not None:
                    tickets = integrator.process_health_data(health_df, threshold)
                    generated.extend(tickets)
                
                if pd_df is not None:
                    tickets = integrator.process_pd_anomalies(pd_df)
                    generated.extend(tickets)
            
            if generated:
                st.success(f"✅ {len(generated)} ticket(s) généré(s)")
                for ticket in generated:
                    st.markdown(format_ticket_card(ticket), unsafe_allow_html=True)
            else:
                st.info("Aucune anomalie détectée au-dessus du seuil")
    
    with tabs[1]:
        st.subheader("✍️ Création Manuelle")
        
        col1, col2 = st.columns(2)
        
        with col1:
            module = st.selectbox("Module", [m.value for m in Module])
            severity = st.slider("Score de sévérité", 0, 100, 50)
            anomaly_type = st.selectbox("Type d'anomalie", [a.value for a in AnomalyType])
        
        with col2:
            description = st.text_area("Description", height=100)
            metrics = st.text_input("Métriques (JSON)", '{"temperature": 85}')
        
        if st.button("📝 Créer le Ticket"):
            try:
                import json
                metrics_dict = json.loads(metrics)
                
                ticket = engine.generate_ticket(
                    module=Module[module],
                    severity_score=severity,
                    metrics=metrics_dict
                )
                
                st.success(f"✅ Ticket créé: {ticket.ticket_id}")
                st.markdown(format_ticket_card(ticket), unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Erreur: {e}")
    
    with tabs[2]:
        st.subheader("📊 Génération depuis Fichier")
        
        uploaded = st.file_uploader("Charger un CSV avec les anomalies", type=['csv'])
        
        if uploaded:
            df = pd.read_csv(uploaded)
            st.dataframe(df.head())
            
            if st.button("🔄 Traiter le fichier"):
                st.info("Fonctionnalité à implémenter selon le format du fichier")


def page_analytics():
    """Page analytiques"""
    st.header("📊 Analytiques")
    
    engine = get_ticket_engine()
    df = engine.export_to_dataframe()
    
    if df.empty:
        st.warning("Pas assez de données pour les analytiques")
        return
    
    # Convertir timestamp
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['date'] = df['timestamp'].dt.date
    
    # KPIs
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_severity = df['severity_score'].mean()
        st.metric("Sévérité Moyenne", f"{avg_severity:.1f}/100")
    
    with col2:
        critical_rate = len(df[df['priority'] == 'CRITICAL']) / len(df) * 100
        st.metric("Taux Critical", f"{critical_rate:.1f}%")
    
    with col3:
        open_rate = len(df[df['status'] == 'OPEN']) / len(df) * 100
        st.metric("Taux Ouvert", f"{open_rate:.1f}%")
    
    with col4:
        auto_closed = len(df[df['status'] == 'AUTO_CLOSED'])
        st.metric("Auto-fermés", auto_closed)
    
    st.markdown("---")
    
    # Graphiques
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Évolution Temporelle")
        daily = df.groupby('date').size().reset_index(name='count')
        fig = px.line(daily, x='date', y='count', markers=True)
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📊 Distribution Sévérité")
        fig = px.histogram(df, x='severity_score', nbins=20, color='priority',
                          color_discrete_map={'CRITICAL': '#f44336', 'HIGH': '#ff9800',
                                             'MEDIUM': '#ffc107', 'LOW': '#4caf50'})
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    # Heatmap Module vs Priorité
    st.subheader("🗺️ Heatmap Module × Priorité")
    pivot = pd.crosstab(df['module'], df['priority'])
    fig = px.imshow(pivot, aspect='auto', color_continuous_scale='RdYlGn_r',
                   labels=dict(x='Priorité', y='Module', color='Nombre'))
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    # Top anomalies
    st.subheader("🔥 Top Types d'Anomalies")
    top_anomalies = df['anomaly_type'].value_counts().head(10)
    fig = px.bar(x=top_anomalies.values, y=top_anomalies.index, orientation='h',
                color=top_anomalies.values, color_continuous_scale='Reds')
    fig.update_layout(height=350)
    st.plotly_chart(fig, use_container_width=True)


def page_settings():
    """Page paramètres"""
    st.header("⚙️ Paramètres")
    
    engine = get_ticket_engine()
    
    tabs = st.tabs(["🎚️ Seuils", "📧 Notifications", "🔧 Maintenance"])
    
    with tabs[0]:
        st.subheader("Configuration des Seuils")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Seuils de Priorité")
            critical_threshold = st.slider("Critical (>)", 80, 100, 90)
            high_threshold = st.slider("High (>)", 60, 80, 70)
            medium_threshold = st.slider("Medium (>)", 30, 60, 40)
            st.caption(f"Low: ≤ {medium_threshold}")
        
        with col2:
            st.markdown("### Seuils par Module")
            
            st.write("**🌡️ Thermal**")
            temp_warning = st.number_input("Temp Warning (°C)", value=80)
            temp_critical = st.number_input("Temp Critical (°C)", value=100)
            
            st.write("**⚡ PD**")
            pd_warning = st.number_input("PD Severity Warning", value=50)
            pd_critical = st.number_input("PD Severity Critical", value=75)
        
        if st.button("💾 Sauvegarder Seuils"):
            st.success("✅ Seuils sauvegardés (simulation)")
    
    with tabs[1]:
        st.subheader("Configuration Notifications")
        
        st.write("**📧 Email**")
        email_enabled = st.checkbox("Activer notifications email", value=True)
        email_recipients = st.text_input("Destinataires", "maintenance@steg.com.tn")
        
        st.write("**📱 SMS**")
        sms_enabled = st.checkbox("Activer SMS (Critical seulement)", value=False)
        sms_number = st.text_input("Numéro", "+216 XX XXX XXX")
        
        st.write("**🔔 Intégrations**")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.checkbox("Microsoft Teams", value=True)
        with col2:
            st.checkbox("SAP PM", value=False)
        with col3:
            st.checkbox("ServiceNow", value=False)
        
        if st.button("💾 Sauvegarder Notifications"):
            st.success("✅ Configuration sauvegardée (simulation)")
    
    with tabs[2]:
        st.subheader("🔧 Maintenance du Système")
        
        stats = engine.get_statistics()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Tickets en base", stats.get('total', 0))
            st.metric("Fichier DB", str(engine.storage_path.name))
        
        with col2:
            if st.button("🗑️ Purger tickets fermés (> 30 jours)"):
                st.warning("Fonctionnalité de purge (simulation)")
            
            if st.button("🔄 Réinitialiser la base"):
                if st.checkbox("Je confirme la suppression"):
                    engine.tickets = []
                    engine._save_tickets()
                    st.success("✅ Base réinitialisée")
                    st.rerun()


# =============================================================================
# MAIN
# =============================================================================

def main():
    # Sidebar
    st.sidebar.image("https://img.icons8.com/color/96/000000/maintenance.png", width=80)
    st.sidebar.title("🎫 Ticketing System")
    st.sidebar.markdown("---")
    
    # Navigation
    pages = {
        "🏠 Dashboard": page_dashboard,
        "📋 Liste Tickets": page_tickets_list,
        "🔍 Détail Ticket": page_ticket_detail,
        "➕ Générer Tickets": page_generate_tickets,
        "📊 Analytiques": page_analytics,
        "⚙️ Paramètres": page_settings,
    }
    
    selection = st.sidebar.radio("Navigation", list(pages.keys()))
    
    # Quick stats
    st.sidebar.markdown("---")
    engine = get_ticket_engine()
    stats = engine.get_statistics()
    
    st.sidebar.markdown("### 📈 Quick Stats")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.sidebar.metric("Open", stats.get('open', 0))
    with col2:
        critical = stats.get('by_priority', {}).get('CRITICAL', 0)
        st.sidebar.metric("Critical", critical)
    
    # Exécuter la page
    pages[selection]()
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.caption(f"Dernière MàJ: {datetime.now().strftime('%H:%M:%S')}")
    st.sidebar.caption("STEG Industrial - 2026")


if __name__ == "__main__":
    main()
