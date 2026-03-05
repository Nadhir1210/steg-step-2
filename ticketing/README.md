# 🎫 Maintenance Ticketing System

## Système Automatisé de Gestion des Incidents - TG1 STEG

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.31-red.svg)](https://streamlit.io)

---

## 🧠 NOUVEAU: Smart Ticketing (ML + RAG + LLM)

### Architecture Intelligente

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

### Lancement Smart Dashboard

```bash
streamlit run ticketing/app_smart_ticketing.py --server.port 8526
# URL: http://localhost:8526
```

---

## 📋 Table des Matières

- [Objectif](#-objectif)
- [Architecture](#-architecture)
- [Fonctionnalités](#-fonctionnalités)
- [Installation](#-installation)
- [Utilisation](#-utilisation)
- [API du Moteur](#-api-du-moteur)
- [Configuration](#-configuration)

---

## 🎯 Objectif

Après détection d'anomalie par les modèles ML, le système génère automatiquement :

| Élément | Description |
|---------|-------------|
| 📩 **Ticket** | ID unique + timestamp |
| ⚠️ **Criticité** | LOW / MEDIUM / HIGH / CRITICAL |
| 🛠 **Action** | Recommandation technique |
| 📅 **Priorité** | Basée sur le score de sévérité |
| 👤 **Service** | Équipe maintenance assignée |
| ⏱️ **RUL** | Durée de vie restante estimée |

---

## 🏗️ Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Détection     │────▶│    Scoring      │────▶│  Classification │
│   Anomalie ML   │     │    0-100        │     │  Criticité      │
└─────────────────┘     └─────────────────┘     └────────┬────────┘
                                                         │
┌─────────────────┐     ┌─────────────────┐     ┌────────▼────────┐
│    Clôture      │◀────│    Suivi        │◀────│   Génération    │
│   Auto/Manuel   │     │   Dashboard     │     │   Ticket        │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

---

## 📊 Classification de Criticité

| Score | Niveau | Couleur | Action Requise |
|-------|--------|---------|----------------|
| 90-100 | 🔴 CRITICAL | Rouge | Intervention urgente (< 24h) |
| 70-89 | 🟠 HIGH | Orange | Intervention rapide (24-72h) |
| 40-69 | 🟡 MEDIUM | Jaune | Inspection planifiée (1-2 sem) |
| 0-39 | 🟢 LOW | Vert | Surveillance continue |

---

## 🎫 Contenu d'un Ticket

```json
{
  "ticket_id": "TKT-20260305-487B6C05",
  "timestamp": "2026-03-05T07:04:23",
  "module": "THERMAL",
  "anomaly_type": "Température élevée",
  "severity_score": 85.0,
  "priority": "HIGH",
  "status": "OPEN",
  "description": "Température élevée: 92.0°C (seuil: 80°C)",
  "root_cause": "Basé sur analyse SHAP:\n  - LOAD: contribution 1.42\n  - Charge élevée détectée",
  "recommended_action": "Vérifier système de refroidissement et isolation thermique",
  "assigned_service": "Maintenance Mécanique",
  "estimated_rul": "24-72 heures",
  "auto_close_eligible": true
}
```

---

## ⚡ Fonctionnalités

### 🤖 Détection Automatique
- Intégration avec les modèles ML (XGBoost, LSTM, Isolation Forest)
- Analyse des données Health Index en temps réel
- Détection multi-modules (Thermal, Cooling, Electrical, PD)

### 🔍 Root Cause Analysis
- Suggestions basées sur **SHAP** feature importance
- Analyse contextuelle (charge, température ambiante, etc.)
- Corrélation multi-signaux

### ⏱️ Remaining Useful Life (RUL)
- Estimation basée sur le score de sévérité
- Prise en compte des tendances (LSTM)
- Alertes proactives

### 🔄 Auto-Closing
- Fermeture automatique si l'anomalie disparaît
- Historique des résolutions
- Statistiques de récurrence

---

## 💻 Installation

```bash
# Depuis le dossier racine du projet
cd ticketing

# Les dépendances sont déjà dans requirements.txt du projet principal
pip install streamlit pandas numpy plotly
```

---

## 🚀 Utilisation

### Lancer le Dashboard

```bash
# Depuis la racine du projet
streamlit run ticketing/app_ticketing.py --server.port 8525
```

**Accès:** http://localhost:8525

### Utiliser le Moteur (API Python)

```python
from ticketing.ticket_engine import TicketEngine, Module

# Initialiser le moteur
engine = TicketEngine()

# Générer un ticket
ticket = engine.generate_ticket(
    module=Module.THERMAL,
    severity_score=85,
    metrics={"temperature": 92, "load": 110}
)

print(f"Ticket créé: {ticket.ticket_id}")
print(f"Priorité: {ticket.priority}")
print(f"Action: {ticket.recommended_action}")
```

---

## 📱 Pages du Dashboard

| Page | Description |
|------|-------------|
| 🏠 **Dashboard** | Vue d'ensemble, KPIs, tickets urgents |
| 📋 **Liste Tickets** | Tableau filtrable, tri, export CSV |
| 🔍 **Détail Ticket** | Vue complète, mise à jour statut |
| ➕ **Générer** | Auto-détection ou création manuelle |
| 📊 **Analytiques** | Graphiques, heatmaps, tendances |
| ⚙️ **Paramètres** | Seuils, notifications, maintenance |

---

## 🔧 API du Moteur

### Classes Principales

```python
# Énumérations
class Priority(Enum):
    LOW, MEDIUM, HIGH, CRITICAL

class TicketStatus(Enum):
    OPEN, IN_PROGRESS, PENDING, RESOLVED, CLOSED, AUTO_CLOSED

class Module(Enum):
    THERMAL, COOLING, ELECTRICAL, PD, GLOBAL

class AnomalyType(Enum):
    TEMPERATURE_HIGH, TEMPERATURE_CRITICAL, COOLING_INEFFICIENCY,
    ELECTRICAL_INSTABILITY, PD_ACTIVITY_HIGH, PD_CRITICAL,
    PHASE_ASYMMETRY, DRIFT_DETECTED, HEALTH_DEGRADATION, MULTI_FAULT
```

### Méthodes TicketEngine

| Méthode | Description |
|---------|-------------|
| `generate_ticket()` | Créer un nouveau ticket |
| `update_ticket_status()` | Mettre à jour le statut |
| `auto_close_tickets()` | Fermeture automatique |
| `get_open_tickets()` | Tickets ouverts |
| `get_tickets_by_priority()` | Filtrer par priorité |
| `get_tickets_by_module()` | Filtrer par module |
| `get_statistics()` | Statistiques globales |
| `export_to_dataframe()` | Export en DataFrame |

---

## ⚙️ Configuration

### Seuils par Module

```python
MODULE_THRESHOLDS = {
    Module.THERMAL: {
        "temp_max": 90,        # °C
        "temp_warning": 80,    # °C
        "temp_critical": 100,  # °C
    },
    Module.PD: {
        "severity_warning": 50,
        "severity_critical": 75,
    },
    Module.ELECTRICAL: {
        "frequency_tolerance": 0.5,  # Hz
        "asymmetry_max": 10,         # %
    },
    Module.COOLING: {
        "delta_t_min": 5,       # °C
        "efficiency_min": 0.7,  # ratio
    }
}
```

### Services Responsables

| Module | Service Assigné |
|--------|-----------------|
| THERMAL | Maintenance Mécanique |
| COOLING | Maintenance Mécanique |
| ELECTRICAL | Maintenance Électrique |
| PD | Maintenance Électrique - Isolation |
| GLOBAL | Direction Maintenance |

---

## 🔔 Intégrations (Futures)

- 📧 **Email**: Notifications automatiques
- 📱 **SMS**: Alertes critiques
- 💬 **Teams**: Intégration Microsoft Teams
- 🏢 **SAP PM**: Export vers SAP
- 🎫 **ServiceNow**: Synchronisation tickets

---

## 📁 Structure des Fichiers

```
ticketing/
├── ticket_engine.py      # Moteur de génération de tickets
├── app_ticketing.py      # Dashboard Streamlit
├── tickets_db.json       # Base de données JSON
└── README.md             # Cette documentation
```

---

## 📊 Exemple de Sortie

```
============================================================
🎫 MAINTENANCE TICKETING ENGINE - TEST
============================================================

✅ Ticket créé: TKT-20260305-487B6C05
   Module: THERMAL
   Type: Température élevée
   Priorité: HIGH
   Sévérité: 85/100
   RUL: 24-72 heures
   Service: Maintenance Mécanique

📊 STATISTIQUES
----------------------------------------
Total tickets: 4
Ouverts: 4
Par priorité: {'CRITICAL': 0, 'HIGH': 2, 'MEDIUM': 2, 'LOW': 0}
```

---

## 👤 Auteur

**Nadhir** - Stage STEG 2026

---

## 📄 Licence

Projet interne STEG - Usage réservé
