"""
🧠 SMART TICKETING ENGINE - ML + RAG + LLM
==========================================
Système intelligent de gestion de tickets avec:
- ML: Détection d'anomalies et classification
- RAG: Retrieval Augmented Generation pour documentation
- LLM: Génération intelligente de descriptions et recommandations

Auteur: Nadhir - Stage STEG 2026
"""

import uuid
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict, field
from enum import Enum
import hashlib
import re
import requests


# =============================================================================
# OLLAMA CONFIGURATION
# =============================================================================

OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "llama3.2:1b"
OLLAMA_TIMEOUT = 120.0


# =============================================================================
# ENUMS & DATA CLASSES
# =============================================================================

class Priority(Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class TicketStatus(Enum):
    OPEN = "OPEN"
    IN_PROGRESS = "IN_PROGRESS"
    PENDING = "PENDING"
    RESOLVED = "RESOLVED"
    CLOSED = "CLOSED"
    AUTO_CLOSED = "AUTO_CLOSED"


class Module(Enum):
    THERMAL = "THERMAL"
    COOLING = "COOLING"
    ELECTRICAL = "ELECTRICAL"
    PD = "PD"
    GLOBAL = "GLOBAL"


class AnomalyType(Enum):
    TEMPERATURE_HIGH = "Température élevée"
    TEMPERATURE_CRITICAL = "Température critique"
    COOLING_INEFFICIENCY = "Inefficacité refroidissement"
    ELECTRICAL_INSTABILITY = "Instabilité électrique"
    PD_ACTIVITY_HIGH = "Activité PD élevée"
    PD_CRITICAL = "Décharges partielles critiques"
    PHASE_ASYMMETRY = "Asymétrie des phases"
    DRIFT_DETECTED = "Drift modèle détecté"
    HEALTH_DEGRADATION = "Dégradation santé globale"
    MULTI_FAULT = "Défaut multiple"


@dataclass
class KnowledgeChunk:
    """Chunk de connaissance pour RAG"""
    chunk_id: str
    source: str
    category: str
    content: str
    keywords: List[str]
    embedding: Optional[List[float]] = None


@dataclass
class SmartTicket:
    """Ticket intelligent avec RAG + LLM"""
    ticket_id: str
    timestamp: str
    module: str
    anomaly_type: str
    severity_score: float
    priority: str
    status: str
    
    # ML Analysis
    ml_confidence: float
    detected_features: Dict[str, float]
    shap_analysis: Dict[str, float]
    
    # RAG Retrieved Context
    retrieved_docs: List[str]
    relevant_procedures: List[str]
    similar_incidents: List[str]
    
    # LLM Generated Content (Ollama-powered)
    llm_description: str
    llm_root_cause: str
    llm_recommendation: str
    llm_prevention: str
    
    # Standard Fields
    assigned_service: str
    estimated_rul: str
    auto_close_eligible: bool
    created_by: str
    updated_at: str
    resolution_notes: str
    
    # Metadata
    processing_time_ms: float
    model_versions: Dict[str, str]
    
    # Optional LLM Resolution (added later, has default)
    llm_resolution: str = ""  # Step-by-step resolution guide


# =============================================================================
# KNOWLEDGE BASE (RAG)
# =============================================================================

class KnowledgeBase:
    """
    Base de connaissances pour RAG
    Contient les procédures, manuels et historique des incidents
    """
    
    def __init__(self, kb_path: Optional[Path] = None):
        self.kb_path = kb_path or Path(__file__).parent / "knowledge_base.json"
        self.chunks: List[KnowledgeChunk] = []
        self._initialize_knowledge()
    
    def _initialize_knowledge(self):
        """Initialiser la base de connaissances avec les données techniques"""
        
        # Charger depuis fichier si existe
        if self.kb_path.exists():
            try:
                with open(self.kb_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.chunks = [KnowledgeChunk(**c) for c in data]
                    return
            except:
                pass
        
        # Sinon, créer la base de connaissances par défaut
        self._create_default_knowledge()
        self._save_knowledge()
    
    def _create_default_knowledge(self):
        """Créer la base de connaissances par défaut"""
        
        knowledge_data = [
            # THERMAL PROCEDURES
            {
                "source": "Manuel Turbine TG1 - Section 4.2",
                "category": "THERMAL",
                "content": """
                PROCÉDURE DE SURCHAUFFE STATOR
                ================================
                1. Vérifier la température ambiante et les conditions de charge
                2. Contrôler le système de refroidissement (ventilateurs, échangeurs)
                3. Inspecter l'isolation thermique des enroulements
                4. Mesurer la résistance d'isolement
                5. Vérifier l'équilibre des phases
                
                SEUILS CRITIQUES:
                - Température stator max: 130°C (alarme), 140°C (déclenchement)
                - Gradient max: 5°C/minute
                - Température paliers: 80°C max
                """,
                "keywords": ["température", "stator", "surchauffe", "refroidissement", "isolation"]
            },
            {
                "source": "Manuel Turbine TG1 - Section 4.3",
                "category": "THERMAL",
                "content": """
                DIAGNOSTIC THERMIQUE AVANCÉ
                ===========================
                Causes fréquentes de surchauffe:
                - Surcharge prolongée (>110% charge nominale)
                - Défaut ventilation (obstruction, panne ventilateur)
                - Court-circuit inter-spires (point chaud localisé)
                - Déséquilibre de charge entre phases
                - Dégradation isolation (vieillissement, humidité)
                
                Actions correctives:
                - Réduire la charge si >100%
                - Nettoyer les filtres à air
                - Vérifier le débit d'eau de refroidissement
                - Contrôler l'étanchéité du circuit de refroidissement
                """,
                "keywords": ["diagnostic", "thermique", "surchauffe", "ventilation", "surcharge"]
            },
            
            # COOLING PROCEDURES
            {
                "source": "Manuel Turbine TG1 - Section 5.1",
                "category": "COOLING",
                "content": """
                SYSTÈME DE REFROIDISSEMENT
                ==========================
                Le système de refroidissement comprend:
                - Refroidissement à air (ventilateurs axiaux)
                - Refroidissement à eau (échangeurs thermiques)
                - Circuit d'hydrogène (alternateurs haute puissance)
                
                PARAMÈTRES NOMINAUX:
                - Delta T air (entrée/sortie): 15-25°C
                - Débit d'eau: 500 m³/h
                - Pression circuit: 3-4 bar
                - Efficacité minimale: 85%
                
                MAINTENANCE PRÉVENTIVE:
                - Nettoyage filtres: mensuel
                - Contrôle étanchéité: trimestriel
                - Analyse eau: semestriel
                """,
                "keywords": ["refroidissement", "ventilation", "échangeur", "débit", "efficacité"]
            },
            {
                "source": "Manuel Turbine TG1 - Section 5.2",
                "category": "COOLING",
                "content": """
                DIAGNOSTIC DÉFAUT REFROIDISSEMENT
                =================================
                Symptômes d'inefficacité:
                - Delta T < 10°C (insuffisant)
                - Delta T > 30°C (surcharge thermique)
                - Température sortie air > 60°C
                
                Causes probables:
                1. Filtres encrassés → Nettoyage/remplacement
                2. Ventilateur défaillant → Contrôle moteur et pales
                3. Fuite circuit eau → Recherche et colmatage
                4. Encrassement échangeur → Nettoyage chimique
                5. Pompe défaillante → Vérification débit et pression
                """,
                "keywords": ["diagnostic", "refroidissement", "delta", "filtre", "ventilateur"]
            },
            
            # ELECTRICAL PROCEDURES
            {
                "source": "Manuel Turbine TG1 - Section 6.1",
                "category": "ELECTRICAL",
                "content": """
                STABILITÉ ÉLECTRIQUE
                ====================
                Paramètres nominaux TG1 Sousse:
                - Puissance: 124 MW
                - Tension: 15.75 kV
                - Fréquence: 50 Hz ± 0.1%
                - Facteur de puissance: 0.85
                
                LIMITES D'EXPLOITATION:
                - Fréquence: 49.5 - 50.5 Hz
                - Tension: ± 5% nominal
                - Asymétrie phases: < 3%
                - Courant rotor max: 2500 A
                
                PROTECTIONS:
                - Surtension: déclenchement à 110%
                - Sous-fréquence: déclenchement à 47.5 Hz
                - Surintensité: alarme à 105%, décl. à 120%
                """,
                "keywords": ["électrique", "fréquence", "tension", "puissance", "protection"]
            },
            {
                "source": "Manuel Turbine TG1 - Section 6.2",
                "category": "ELECTRICAL",
                "content": """
                DIAGNOSTIC INSTABILITÉ ÉLECTRIQUE
                =================================
                Causes d'asymétrie des phases:
                - Déséquilibre de charge réseau
                - Défaut connexion (serrage, oxydation)
                - Court-circuit partiel enroulement
                - Défaut transformateur élévateur
                
                Actions correctives:
                1. Vérifier équilibre charges par phase
                2. Contrôler serrages connexions HT
                3. Mesurer résistances d'enroulements
                4. Thermographie infrarouge des connexions
                5. Analyse vibratoire (origine mécanique)
                
                ATTENTION: Toute intervention HT nécessite consignation!
                """,
                "keywords": ["asymétrie", "phases", "déséquilibre", "connexion", "court-circuit"]
            },
            
            # PD (Partial Discharge) PROCEDURES
            {
                "source": "Manuel Turbine TG1 - Section 7.1",
                "category": "PD",
                "content": """
                DÉCHARGES PARTIELLES (PD)
                =========================
                Les décharges partielles indiquent une dégradation de l'isolation.
                
                TYPES DE PD:
                - PD internes (cavités dans isolation)
                - PD de surface (tracking)
                - PD corona (champ électrique intense)
                
                SEUILS D'ALERTE:
                - Niveau bas: < 100 pC (normal)
                - Niveau moyen: 100-500 pC (surveillance)
                - Niveau élevé: 500-1000 pC (maintenance planifiée)
                - Niveau critique: > 1000 pC (intervention urgente)
                
                FACTEURS AGGRAVANTS:
                - Humidité > 60%
                - Température élevée
                - Surtensions répétées
                - Vibrations mécaniques
                """,
                "keywords": ["décharge", "partielle", "isolation", "PD", "corona", "cavité"]
            },
            {
                "source": "Manuel Turbine TG1 - Section 7.2",
                "category": "PD",
                "content": """
                DIAGNOSTIC ET MAINTENANCE PD
                ============================
                Procédure de diagnostic:
                1. Mesure PD en ligne (capteurs HFCT)
                2. Localisation par triangulation
                3. Analyse pattern reconnaissance (PRPD)
                4. Corrélation avec charge et température
                
                Actions selon niveau:
                - Niveau moyen: Surveillance renforcée, analyse tendance
                - Niveau élevé: Planifier arrêt pour inspection
                - Niveau critique: Réduire charge, préparer intervention
                
                Réparations possibles:
                - Réinjection résine époxy (cavités)
                - Revêtement semi-conducteur (tracking)
                - Remplacement barres (cas sévère)
                """,
                "keywords": ["diagnostic", "PD", "mesure", "localisation", "réparation"]
            },
            
            # HISTORICAL INCIDENTS
            {
                "source": "Historique Incidents TG1 - 2024",
                "category": "INCIDENTS",
                "content": """
                INCIDENT #2024-047: Surchauffe Phase B
                Date: 15/07/2024
                Symptômes: Température phase B +15°C vs A et C
                Cause: Connexion desserrée sortie alternateur
                Résolution: Resserrage couple 120 Nm, nettoyage contacts
                Durée arrêt: 4 heures
                
                INCIDENT #2024-089: PD élevé sur barres stator
                Date: 23/09/2024
                Symptômes: PD > 800 pC, asymétrie canaux
                Cause: Dégradation isolation barre #23 (fissure)
                Résolution: Réinjection résine + surveillance
                Prochaine action: Remplacement planifié révision 2025
                """,
                "keywords": ["incident", "historique", "surchauffe", "PD", "résolution"]
            },
            {
                "source": "Historique Incidents TG1 - 2025",
                "category": "INCIDENTS",
                "content": """
                INCIDENT #2025-012: Baisse efficacité refroidissement
                Date: 08/02/2025
                Symptômes: Delta T réduit à 8°C, températures en hausse
                Cause: Filtres air colmatés + encrassement échangeur
                Résolution: Nettoyage complet circuit refroidissement
                Durée arrêt: 8 heures
                
                INCIDENT #2025-028: Instabilité fréquence
                Date: 14/03/2025
                Symptômes: Oscillations ±0.3 Hz
                Cause: Défaut régulateur vitesse
                Résolution: Recalibration régulateur + firmware update
                Durée arrêt: 2 heures
                """,
                "keywords": ["incident", "2025", "refroidissement", "fréquence", "régulateur"]
            },
            
            # MAINTENANCE SCHEDULES
            {
                "source": "Planning Maintenance TG1",
                "category": "MAINTENANCE",
                "content": """
                PLANNING MAINTENANCE PRÉVENTIVE
                ===============================
                
                QUOTIDIEN:
                - Relevé températures (auto)
                - Contrôle visuel fuites
                - Vérification alarmes
                
                HEBDOMADAIRE:
                - Analyse vibratoire
                - Test protection
                - Relevé compteurs énergie
                
                MENSUEL:
                - Prélèvement huile (analyse)
                - Contrôle filtres air
                - Vérification serrages accessibles
                
                TRIMESTRIEL:
                - Thermographie infrarouge
                - Mesure isolement enroulements
                - Contrôle système refroidissement
                
                ANNUEL (arrêt programmé):
                - Inspection complète stator/rotor
                - Remplacement filtres
                - Étalonnage capteurs
                - Test PD complet
                """,
                "keywords": ["maintenance", "préventive", "planning", "inspection", "test"]
            }
        ]
        
        # Créer les chunks
        for idx, item in enumerate(knowledge_data):
            chunk = KnowledgeChunk(
                chunk_id=f"KB-{idx:04d}",
                source=item["source"],
                category=item["category"],
                content=item["content"].strip(),
                keywords=item["keywords"]
            )
            self.chunks.append(chunk)
    
    def _save_knowledge(self):
        """Sauvegarder la base de connaissances"""
        with open(self.kb_path, 'w', encoding='utf-8') as f:
            json.dump([asdict(c) for c in self.chunks], f, indent=2, ensure_ascii=False)
    
    def search(self, query: str, category: Optional[str] = None, top_k: int = 3) -> List[KnowledgeChunk]:
        """
        Recherche simple par mots-clés (sans embeddings pour simplicité)
        En production: utiliser sentence-transformers + FAISS
        """
        query_words = set(query.lower().split())
        
        scored_chunks = []
        for chunk in self.chunks:
            # Filtrer par catégorie si spécifié
            if category and chunk.category != category:
                continue
            
            # Score basé sur keywords + contenu
            score = 0
            chunk_text = chunk.content.lower()
            
            for word in query_words:
                if word in chunk.keywords:
                    score += 3
                if word in chunk_text:
                    score += 1
            
            if score > 0:
                scored_chunks.append((score, chunk))
        
        # Trier par score et retourner top_k
        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        return [chunk for _, chunk in scored_chunks[:top_k]]
    
    def get_similar_incidents(self, anomaly_type: str, module: str) -> List[str]:
        """Trouver des incidents similaires dans l'historique"""
        incidents = self.search(f"{anomaly_type} {module}", category="INCIDENTS", top_k=2)
        return [chunk.content for chunk in incidents]
    
    def get_procedures(self, module: str, anomaly_type: str) -> List[str]:
        """Récupérer les procédures pertinentes"""
        procedures = self.search(f"{anomaly_type}", category=module, top_k=2)
        return [chunk.content for chunk in procedures]


# =============================================================================
# OLLAMA LLM GENERATOR - Professional Problem & Resolution Descriptions
# =============================================================================

class OllamaLLMGenerator:
    """
    Générateur de texte intelligent utilisant Ollama LLM
    Génère des descriptions professionnelles des problèmes et solutions
    """
    
    def __init__(self):
        self.ollama_url = OLLAMA_BASE_URL
        self.model = OLLAMA_MODEL
        self.timeout = OLLAMA_TIMEOUT
        self.templates = self._load_fallback_templates()
    
    def _call_ollama(self, prompt: str) -> Optional[str]:
        """Appeler l'API Ollama"""
        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.6,
                        "top_p": 0.9,
                        "num_predict": 400
                    }
                },
                timeout=self.timeout
            )
            if response.status_code == 200:
                return response.json().get("response", "")
            return None
        except Exception as e:
            print(f"Ollama error: {e}")
            return None
    
    def _load_fallback_templates(self) -> Dict:
        """Charger les templates de génération"""
        return {
            "description": {
                "THERMAL": [
                    "Anomalie thermique détectée sur le module {module}. La température mesurée de {temp:.1f}°C dépasse le seuil de {threshold}°C, indiquant une condition de {severity}. L'analyse ML a identifié cette déviation avec une confiance de {confidence:.1f}%.",
                    "Alerte thermique: Le système de monitoring a relevé une élévation anormale de température ({temp:.1f}°C) sur {module}. Cette situation nécessite une attention {priority_text}."
                ],
                "COOLING": [
                    "Dégradation de l'efficacité du système de refroidissement détectée. Le différentiel de température (ΔT) mesuré de {delta_t:.1f}°C est inférieur au minimum requis, suggérant une perte d'efficacité de {efficiency_loss:.0f}%.",
                    "Alerte système de refroidissement: Performance en dessous des seuils nominaux. Efficacité actuelle estimée à {efficiency:.0f}% contre {nominal}% nominal."
                ],
                "ELECTRICAL": [
                    "Instabilité électrique détectée sur le réseau TG1. {symptom}. Les mesures indiquent une déviation de {deviation:.2f}% par rapport aux paramètres nominaux.",
                    "Anomalie électrique: {symptom}. Cette condition peut affecter la qualité de l'énergie produite et nécessite une investigation."
                ],
                "PD": [
                    "Activité de décharges partielles élevée détectée. Score de sévérité PD: {severity:.1f}/100. L'analyse multi-canaux révèle une asymétrie de {asymmetry:.1f}%, suggérant une dégradation localisée de l'isolation.",
                    "Alerte isolation stator: Le système de monitoring PD a identifié une activité anormale. Niveau: {level}. Tendance: {trend}."
                ]
            },
            "root_cause": {
                "THERMAL": [
                    "Analyse des causes probables basée sur SHAP et corrélations:\n\n1. **Charge élevée** ({load_contribution:.0f}% contribution): La puissance de {load}MW représente {load_pct:.0f}% de la capacité nominale.\n\n2. **Inefficacité refroidissement** ({cooling_contribution:.0f}% contribution): Le ΔT de {delta_t:.1f}°C est sous les normes.\n\n3. **Conditions ambiantes** ({ambient_contribution:.0f}% contribution): Température ambiante de {ambient}°C au-dessus de la normale saisonnière.",
                    "Les principaux facteurs identifiés par l'analyse ML sont:\n- {factor1}\n- {factor2}\n- {factor3}"
                ],
                "PD": [
                    "Analyse causale des décharges partielles:\n\n1. **Localisation**: Canal(aux) {channels} présentant l'activité maximale\n\n2. **Type probable**: {pd_type} basé sur le pattern de décharge\n\n3. **Facteurs aggravants**: {aggravating_factors}",
                    "L'analyse SHAP indique les contributions suivantes aux PD:\n- Intensité totale: {intensity_contrib:.0f}%\n- Asymétrie inter-canaux: {asymmetry_contrib:.0f}%\n- Tendance temporelle: {trend_contrib:.0f}%"
                ]
            },
            "recommendation": {
                "CRITICAL": [
                    "⚠️ INTERVENTION URGENTE REQUISE\n\n1. **Action immédiate**: {immediate_action}\n\n2. **Vérifications**: {checks}\n\n3. **Équipe requise**: {team}\n\n4. **Délai**: Intervention sous 24h maximum\n\n5. **Préparation**: {preparation}",
                ],
                "HIGH": [
                    "🔶 INTERVENTION RAPIDE RECOMMANDÉE\n\n1. **Planifier intervention**: Sous 72 heures\n\n2. **Actions préliminaires**:\n   - {action1}\n   - {action2}\n\n3. **Équipements nécessaires**: {equipment}\n\n4. **Documentation**: Préparer historique et tendances",
                ],
                "MEDIUM": [
                    "📋 MAINTENANCE PLANIFIÉE\n\n1. **Inspection recommandée**: Lors du prochain arrêt programmé\n\n2. **Surveillance renforcée**: {monitoring}\n\n3. **Actions préventives**:\n   - {preventive1}\n   - {preventive2}\n\n4. **Suivi**: Réévaluer dans {review_days} jours",
                ]
            },
            "prevention": {
                "THERMAL": "Pour prévenir les futures occurrences:\n1. Optimiser le planning de charge en période de forte chaleur\n2. Augmenter la fréquence de nettoyage des filtres (mensuel → bi-hebdomadaire en été)\n3. Vérifier le bon fonctionnement des alarmes de température\n4. Planifier une maintenance préventive du système de refroidissement",
                "COOLING": "Mesures préventives recommandées:\n1. Mettre en place un programme de nettoyage préventif des échangeurs\n2. Installer des capteurs de débit sur le circuit de refroidissement\n3. Prévoir un stock de filtres de rechange\n4. Former l'équipe aux procédures de maintenance du système",
                "ELECTRICAL": "Actions préventives:\n1. Programmer des contrôles thermographiques trimestriels des connexions\n2. Vérifier le serrage des connexions HT à chaque arrêt\n3. Mettre à jour les procédures de surveillance réseau\n4. Installer des systèmes de surveillance continue de la qualité électrique",
                "PD": "Prévention dégradation isolation:\n1. Contrôler l'humidité dans l'alternateur (maintenir < 40%)\n2. Éviter les surtensions par amélioration des protections\n3. Programme de mesure PD en ligne continue\n4. Planifier remplacement préventif des barres à risque (> 20 ans)"
            }
        }
    
    def generate_description(self, module: str, metrics: Dict, ml_analysis: Dict) -> str:
        """Générer une description intelligente avec Ollama LLM"""
        
        # Préparer le contexte pour l'LLM
        severity_score = metrics.get("severity_score", 50)
        temperature = metrics.get("temperature", 85)
        confidence = ml_analysis.get("confidence", 95)
        priority = ml_analysis.get("priority", "MEDIUM")
        
        # Déterminer le status
        if severity_score > 70:
            status = "CRITICAL"
        elif severity_score > 40:
            status = "WARNING"
        else:
            status = "NORMAL"
        
        prompt = f"""You are a power plant maintenance expert. Analyze this anomaly ticket and provide a professional technical description.

ANOMALY DATA:
- Module: {module}
- Severity Score: {severity_score}/100
- Status: {status}
- Temperature: {temperature}°C
- ML Confidence: {confidence}%
- Priority: {priority}

Write a concise professional problem description (100 words max) that includes:
1. What was detected
2. Current severity level
3. Affected component/system

Be technical and specific. Use professional maintenance language."""

        # Appeler Ollama
        llm_response = self._call_ollama(prompt)
        
        if llm_response and len(llm_response.strip()) > 30:
            return f"## Problem Analysis\n\n{llm_response.strip()}"
        
        # Fallback to template
        template = self.templates["description"].get(module, self.templates["description"]["THERMAL"])[0]
        vars = {
            "module": module,
            "temp": temperature,
            "threshold": 80,
            "severity": "haute" if severity_score > 70 else "modérée",
            "confidence": confidence,
            "priority_text": "urgente" if priority == "CRITICAL" else "particulière",
            "delta_t": metrics.get("delta_t", 10),
            "efficiency_loss": (1 - metrics.get("efficiency", 0.7)) * 100,
            "efficiency": metrics.get("efficiency", 0.7) * 100,
            "nominal": 85,
            "symptom": self._get_symptom(module, metrics),
            "deviation": metrics.get("deviation", 5),
            "asymmetry": metrics.get("asymmetry", 20),
            "level": "ÉLEVÉ" if severity_score > 70 else "MOYEN",
            "trend": "ascendante" if metrics.get("trend", 0) > 0 else "stable"
        }
        try:
            return template.format(**vars)
        except:
            return f"Anomalie détectée sur module {module}. Score de sévérité: {severity_score:.1f}/100"
    
    def generate_resolution(self, module: str, metrics: Dict, ml_analysis: Dict, procedures: List[str]) -> str:
        """Générer des instructions de résolution avec Ollama LLM"""
        
        severity_score = metrics.get("severity_score", 50)
        priority = ml_analysis.get("priority", "MEDIUM")
        
        # Contexte des procédures RAG
        procedure_context = ""
        if procedures:
            procedure_context = "\n".join([p[:200] for p in procedures[:2]])
        
        prompt = f"""You are a power plant maintenance expert. Provide step-by-step resolution instructions for this issue.

ISSUE CONTEXT:
- Module: {module}
- Severity: {severity_score}/100
- Priority: {priority}

RELEVANT PROCEDURES:
{procedure_context if procedure_context else "Standard maintenance procedures apply"}

Provide a clear resolution guide (150 words max) with:
1. Immediate actions (if critical)
2. Diagnostic steps
3. Resolution procedure
4. Verification steps

Use numbered steps. Be specific and actionable."""

        llm_response = self._call_ollama(prompt)
        
        if llm_response and len(llm_response.strip()) > 30:
            return f"## Resolution Guide\n\n{llm_response.strip()}"
        
        # Fallback
        return f"""## Resolution Guide

### Immediate Actions
- Notify maintenance supervisor
- Document current readings

### Diagnostic Steps
1. Check {module} parameters
2. Review recent trends
3. Inspect related components

### Resolution
Follow standard {module} maintenance procedure

### Verification
Confirm parameters return to normal range"""
    
    def _get_symptom(self, module: str, metrics: Dict) -> str:
        """Obtenir le symptôme principal"""
        if module == "ELECTRICAL":
            if metrics.get("asymmetry", 0) > 10:
                return f"Asymétrie des phases détectée ({metrics.get('asymmetry', 0):.1f}%)"
            if abs(metrics.get("frequency_deviation", 0)) > 0.5:
                return f"Déviation de fréquence ({metrics.get('frequency_deviation', 0):.2f} Hz)"
            return "Instabilité des paramètres électriques"
        return ""
    
    def generate_root_cause(self, module: str, metrics: Dict, shap_features: Dict) -> str:
        """Générer l'analyse de cause racine avec Ollama LLM"""
        
        # Calculer les contributions SHAP
        total_shap = sum(abs(v) for v in shap_features.values()) if shap_features else 1
        
        # Top contributing features
        top_features = []
        if shap_features:
            sorted_features = sorted(shap_features.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
            for feat, value in sorted_features:
                contribution = abs(value) / total_shap * 100
                top_features.append(f"{feat}: {contribution:.1f}%")
        
        prompt = f"""You are a power plant diagnostic expert. Analyze the root cause of this anomaly.

MODULE: {module}
SEVERITY: {metrics.get('severity_score', 50)}/100
TEMPERATURE: {metrics.get('temperature', 85)}°C
LOAD: {metrics.get('load', 100)} MW

ML FEATURE CONTRIBUTIONS (SHAP):
{chr(10).join(top_features) if top_features else 'No feature data available'}

Provide a root cause analysis (120 words max) including:
1. Most likely cause
2. Contributing factors
3. Correlation patterns

Be technical and specific for power plant engineers."""

        llm_response = self._call_ollama(prompt)
        
        if llm_response and len(llm_response.strip()) > 30:
            return f"## Root Cause Analysis\n\n{llm_response.strip()}"
        
        # Fallback to template-based response
        if module == "THERMAL":
            return f"""**Root Cause Analysis (ML + SHAP)**

🔍 **Primary Factors Identified:**

1. **Thermal Load** - Contribution: {abs(shap_features.get('load', 0.3))/total_shap*100:.0f}%
   - Current Power: {metrics.get('load', 100)} MW
   - Ratio vs Nominal: {metrics.get('load', 100)/124*100:.0f}%

2. **Cooling Efficiency** - Contribution: {abs(shap_features.get('cooling', 0.25))/total_shap*100:.0f}%
   - ΔT Measured: {metrics.get('delta_t', 15):.1f}°C
   - Efficiency: {metrics.get('efficiency', 0.8)*100:.0f}%

3. **Ambient Conditions** - Contribution: {abs(shap_features.get('ambient', 0.15))/total_shap*100:.0f}%
   - Ambient Temperature: {metrics.get('ambient_temp', 30)}°C

📊 **Detected Correlations:**
- Temperature ↔ Load: r = 0.85
- Température ↔ ΔT Refroidissement: r = -0.72
"""
        
        elif module == "PD":
            return f"""**Analyse causale décharges partielles (ML + Pattern Recognition)**

🔍 **Localisation:**
- Canaux affectés: CH1, CH3 (asymétrie maximale)
- Zone probable: Barres stator côté excitation

📊 **Type de décharge identifié:**
- Pattern: Décharges internes (cavités)
- Confiance classification: 87%

⚠️ **Facteurs aggravants:**
- Intensité totale: {metrics.get('pd_intensity', 100000):.0f} (élevé)
- Asymétrie inter-canaux: {metrics.get('asymmetry', 30):.1f}%
- Tendance: {'Ascendante ↑' if metrics.get('trend', 0) > 0 else 'Stable →'}

🎯 **Contributions SHAP:**
- CURRENT_TOTAL: {abs(shap_features.get('CURRENT_TOTAL', 0.4))/total_shap*100:.0f}%
- INTENSITY_ASYMMETRY: {abs(shap_features.get('INTENSITY_ASYMMETRY', 0.3))/total_shap*100:.0f}%
- PULSE_TOTAL: {abs(shap_features.get('PULSE_TOTAL', 0.2))/total_shap*100:.0f}%
"""
        
        else:
            return f"Root cause analysis based on {len(shap_features)} features. Module: {module}"
    
    def generate_recommendation(self, priority: str, module: str, metrics: Dict, procedures: List[str]) -> str:
        """Générer les recommandations avec Ollama LLM"""
        
        # Contexte des procédures RAG
        procedure_context = ""
        if procedures:
            procedure_context = "\n".join([p[:150] for p in procedures[:2]])
        
        prompt = f"""You are a power plant maintenance supervisor. Provide actionable recommendations for this issue.

ISSUE DETAILS:
- Module: {module}
- Priority: {priority}
- Severity: {metrics.get('severity_score', 50)}/100

RELEVANT PROCEDURES:
{procedure_context if procedure_context else "Standard procedures apply"}

Based on priority level {priority}, provide recommendations (150 words max) including:
1. Timeline for intervention
2. Required team/resources
3. Key actions to take
4. Safety considerations

Be specific and actionable for maintenance teams."""

        llm_response = self._call_ollama(prompt)
        
        if llm_response and len(llm_response.strip()) > 30:
            # Add priority header
            if priority == "CRITICAL":
                header = "⚠️ **URGENT INTERVENTION REQUIRED**"
            elif priority == "HIGH":
                header = "🔶 **RAPID INTERVENTION RECOMMENDED**"
            else:
                header = "📋 **PLANNED MAINTENANCE**"
            
            return f"{header}\n\n{llm_response.strip()}"
        
        # Fallback to template
        procedure_actions = self._extract_actions(procedures)
        
        default_immediate = "- Follow standard manual procedure"
        default_planned = "- Visual inspection\n- Parameter check"
        default_monitoring = "- Daily readings\n- Trend analysis"
        default_preventive = "- Standard verifications"
        
        immediate_action = procedure_actions.get('immediate') or default_immediate
        planned_action = procedure_actions.get('planned') or default_planned
        monitoring_action = procedure_actions.get('monitoring') or default_monitoring
        preventive_action = procedure_actions.get('preventive') or default_preventive
        doc_text = procedures[0][:200] + '...' if procedures else 'Consult maintenance manual'
        
        if priority == "CRITICAL":
            return f"""⚠️ **INTERVENTION URGENTE - PRIORITÉ MAXIMALE**

📋 **Actions immédiates (< 4 heures):**
1. Notifier le responsable maintenance et le chef de quart
2. Réduire la charge à 80% si possible
3. Préparer l'équipe d'intervention

🔧 **Procédure d'intervention ({module}):**
{immediate_action}

👥 **Équipe requise:**
- {self._get_team(module)}
- Durée estimée: 4-8 heures

📄 **Documentation:**
{doc_text}

⏱️ **RUL estimé:** < 24 heures - Intervention avant dégradation majeure
"""
        
        elif priority == "HIGH":
            return f"""🔶 **INTERVENTION RAPIDE RECOMMANDÉE**

📋 **Planification (24-72 heures):**
1. Programmer intervention lors de la prochaine fenêtre disponible
2. Préparer outillage et pièces de rechange
3. Documenter l'évolution des paramètres

🔧 **Actions recommandées:**
{planned_action}

📊 **Surveillance renforcée:**
- Fréquence relevés: Toutes les 2 heures
- Seuils d'alerte: {self._get_alert_thresholds(module)}

👥 **Ressources:** {self._get_team(module)}
"""
        
        else:  # MEDIUM or LOW
            return f"""📋 **MAINTENANCE PLANIFIÉE**

📅 **Planification:**
- Intervention recommandée: Prochain arrêt programmé
- Délai acceptable: 1-2 semaines

🔍 **Surveillance continue:**
{monitoring_action}

📝 **Actions préventives:**
{preventive_action}

📊 **Indicateurs à suivre:**
{self._get_follow_up_indicators(module)}
"""
    
    def _extract_actions(self, procedures: List[str]) -> Dict[str, str]:
        """Extraire les actions des procédures RAG"""
        actions = {
            "immediate": "",
            "planned": "",
            "monitoring": "",
            "preventive": ""
        }
        
        for proc in procedures:
            if "IMMÉDIAT" in proc.upper() or "URGENT" in proc.upper():
                actions["immediate"] += proc[:300] + "\n"
            if "CONTRÔLE" in proc.upper() or "VÉRIFI" in proc.upper():
                actions["monitoring"] += "- " + proc[:150] + "\n"
            if "PRÉVENT" in proc.upper() or "MAINTENANCE" in proc.upper():
                actions["preventive"] += proc[:200] + "\n"
        
        return actions
    
    def _get_team(self, module: str) -> str:
        teams = {
            "THERMAL": "Équipe Maintenance Mécanique + Électrique",
            "COOLING": "Équipe Maintenance Mécanique",
            "ELECTRICAL": "Équipe Maintenance Électrique HT",
            "PD": "Spécialistes Isolation + Maintenance Électrique"
        }
        return teams.get(module, "Équipe Maintenance Générale")
    
    def _get_alert_thresholds(self, module: str) -> str:
        thresholds = {
            "THERMAL": "Temp > 95°C (alarme), > 100°C (arrêt)",
            "COOLING": "ΔT < 5°C, Efficacité < 70%",
            "ELECTRICAL": "Fréq ±0.5Hz, Asymétrie > 5%",
            "PD": "Sévérité > 80, Intensité > 500K"
        }
        return thresholds.get(module, "Selon procédure")
    
    def _get_follow_up_indicators(self, module: str) -> str:
        indicators = {
            "THERMAL": "- Température moyenne stator\n- Gradient thermique\n- ΔT refroidissement",
            "COOLING": "- Débit eau/air\n- Pression circuit\n- Efficacité échangeur",
            "ELECTRICAL": "- Équilibre phases\n- Fréquence réseau\n- Facteur de puissance",
            "PD": "- Score sévérité PD\n- Asymétrie canaux\n- Tendance 7 jours"
        }
        return indicators.get(module, "- Paramètres standards")
    
    def generate_prevention(self, module: str) -> str:
        """Générer les mesures préventives"""
        return self.templates["prevention"].get(module, self.templates["prevention"]["THERMAL"])


# Backward compatibility alias
LLMGenerator = OllamaLLMGenerator


# =============================================================================
# SMART TICKET ENGINE
# =============================================================================

class SmartTicketEngine:
    """
    Moteur de tickets intelligent avec ML + RAG + LLM
    """
    
    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or Path(__file__).parent / "smart_tickets_db.json"
        self.tickets: List[SmartTicket] = []
        self.knowledge_base = KnowledgeBase()
        self.llm = LLMGenerator()
        self._load_tickets()
    
    def _load_tickets(self):
        """Charger les tickets existants"""
        if self.storage_path.exists():
            try:
                with open(self.storage_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.tickets = [SmartTicket(**t) for t in data]
            except Exception as e:
                print(f"Erreur chargement: {e}")
                self.tickets = []
    
    def _save_tickets(self):
        """Sauvegarder les tickets"""
        with open(self.storage_path, 'w', encoding='utf-8') as f:
            data = [asdict(t) for t in self.tickets]
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    @staticmethod
    def define_priority(severity_score: float) -> Priority:
        if severity_score >= 90:
            return Priority.CRITICAL
        elif severity_score >= 70:
            return Priority.HIGH
        elif severity_score >= 40:
            return Priority.MEDIUM
        return Priority.LOW
    
    @staticmethod
    def estimate_rul(severity_score: float, trend: float = 0) -> str:
        if severity_score >= 90:
            return "Critique: < 24 heures"
        elif severity_score >= 75:
            return "Urgent: 24-72 heures"
        elif severity_score >= 60:
            return "Planifié: 1-2 semaines"
        elif severity_score >= 40:
            return "Préventif: 1-3 mois"
        return "Normal: Surveillance continue"
    
    def generate_smart_ticket(self,
                             module: Module,
                             severity_score: float,
                             metrics: Dict[str, float],
                             shap_features: Optional[Dict[str, float]] = None,
                             ml_confidence: float = 0.95) -> SmartTicket:
        """
        Générer un ticket intelligent avec ML + RAG + LLM
        """
        start_time = datetime.now()
        
        # 1. ML Analysis
        priority = self.define_priority(severity_score)
        anomaly_type = self._detect_anomaly_type(module, metrics)
        shap_features = shap_features or {}
        
        # 2. RAG: Récupérer contexte pertinent
        retrieved_docs = []
        kb_results = self.knowledge_base.search(f"{module.value} {anomaly_type.value}", top_k=3)
        for chunk in kb_results:
            retrieved_docs.append(f"[{chunk.source}] {chunk.content[:300]}...")
        
        relevant_procedures = self.knowledge_base.get_procedures(module.value, anomaly_type.value)
        similar_incidents = self.knowledge_base.get_similar_incidents(anomaly_type.value, module.value)
        
        # 3. LLM: Générer contenu intelligent avec Ollama
        ml_analysis = {
            "priority": priority.value,
            "confidence": ml_confidence * 100,
            "severity_score": severity_score
        }
        
        llm_description = self.llm.generate_description(module.value, metrics, ml_analysis)
        llm_root_cause = self.llm.generate_root_cause(module.value, metrics, shap_features)
        llm_recommendation = self.llm.generate_recommendation(
            priority.value, module.value, metrics, relevant_procedures
        )
        llm_prevention = self.llm.generate_prevention(module.value)
        llm_resolution = self.llm.generate_resolution(module.value, metrics, ml_analysis, relevant_procedures)
        
        # 4. Créer le ticket
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        ticket = SmartTicket(
            ticket_id=f"SMART-{datetime.now().strftime('%Y%m%d')}-{str(uuid.uuid4())[:8].upper()}",
            timestamp=datetime.now().isoformat(),
            module=module.value,
            anomaly_type=anomaly_type.value,
            severity_score=severity_score,
            priority=priority.value,
            status=TicketStatus.OPEN.value,
            
            # ML
            ml_confidence=ml_confidence,
            detected_features=metrics,
            shap_analysis=shap_features,
            
            # RAG
            retrieved_docs=retrieved_docs,
            relevant_procedures=[p[:500] for p in relevant_procedures],
            similar_incidents=[i[:500] for i in similar_incidents],
            
            # LLM (Ollama-powered)
            llm_description=llm_description,
            llm_root_cause=llm_root_cause,
            llm_recommendation=llm_recommendation,
            llm_prevention=llm_prevention,
            llm_resolution=llm_resolution,
            
            # Standard
            assigned_service=self._get_service(module),
            estimated_rul=self.estimate_rul(severity_score),
            auto_close_eligible=severity_score < 70,
            created_by="SMART_AI_SYSTEM",
            updated_at=datetime.now().isoformat(),
            resolution_notes="",
            
            # Metadata
            processing_time_ms=processing_time,
            model_versions={
                "ml_detector": "v2.0",
                "rag_retriever": "v1.0",
                "llm_generator": "ollama_v1.0"
            }
        )
        
        self.tickets.append(ticket)
        self._save_tickets()
        
        return ticket
    
    def _detect_anomaly_type(self, module: Module, metrics: Dict) -> AnomalyType:
        """Détecter le type d'anomalie"""
        if module == Module.THERMAL:
            temp = metrics.get('temperature', 0)
            if temp >= 100:
                return AnomalyType.TEMPERATURE_CRITICAL
            return AnomalyType.TEMPERATURE_HIGH
        elif module == Module.COOLING:
            return AnomalyType.COOLING_INEFFICIENCY
        elif module == Module.ELECTRICAL:
            if metrics.get('asymmetry', 0) > 10:
                return AnomalyType.PHASE_ASYMMETRY
            return AnomalyType.ELECTRICAL_INSTABILITY
        elif module == Module.PD:
            severity = metrics.get('pd_severity', 0)
            if severity >= 75:
                return AnomalyType.PD_CRITICAL
            return AnomalyType.PD_ACTIVITY_HIGH
        return AnomalyType.HEALTH_DEGRADATION
    
    def _get_service(self, module: Module) -> str:
        services = {
            Module.THERMAL: "Maintenance Mécanique",
            Module.COOLING: "Maintenance Mécanique",
            Module.ELECTRICAL: "Maintenance Électrique HT",
            Module.PD: "Spécialistes Isolation & Électrique",
            Module.GLOBAL: "Direction Maintenance"
        }
        return services.get(module, "Maintenance Générale")
    
    def get_statistics(self) -> Dict:
        """Obtenir les statistiques"""
        total = len(self.tickets)
        if total == 0:
            return {"total": 0}
        
        return {
            "total": total,
            "open": len([t for t in self.tickets if t.status == "OPEN"]),
            "by_priority": {
                p.value: len([t for t in self.tickets if t.priority == p.value])
                for p in Priority
            },
            "by_module": {
                m.value: len([t for t in self.tickets if t.module == m.value])
                for m in Module
            },
            "avg_severity": np.mean([t.severity_score for t in self.tickets]),
            "avg_processing_time_ms": np.mean([t.processing_time_ms for t in self.tickets])
        }
    
    def export_to_dataframe(self) -> pd.DataFrame:
        if not self.tickets:
            return pd.DataFrame()
        return pd.DataFrame([asdict(t) for t in self.tickets])


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("🧠 SMART TICKETING ENGINE - ML + RAG + LLM - TEST")
    print("=" * 70)
    
    engine = SmartTicketEngine()
    
    # Test 1: Thermal anomaly
    print("\n📝 Test 1: Anomalie Thermique")
    ticket1 = engine.generate_smart_ticket(
        module=Module.THERMAL,
        severity_score=85,
        metrics={
            "temperature": 95,
            "load": 110,
            "delta_t": 12,
            "efficiency": 0.75,
            "ambient_temp": 35
        },
        shap_features={
            "load": 0.45,
            "cooling": 0.30,
            "ambient": 0.15,
            "other": 0.10
        },
        ml_confidence=0.92
    )
    
    print(f"\n✅ Ticket: {ticket1.ticket_id}")
    print(f"   Priorité: {ticket1.priority}")
    print(f"   Temps traitement: {ticket1.processing_time_ms:.1f} ms")
    print(f"\n📄 Description LLM:\n{ticket1.llm_description}")
    print(f"\n🔍 Root Cause:\n{ticket1.llm_root_cause[:500]}...")
    
    # Test 2: PD anomaly
    print("\n" + "=" * 70)
    print("📝 Test 2: Anomalie PD")
    ticket2 = engine.generate_smart_ticket(
        module=Module.PD,
        severity_score=78,
        metrics={
            "pd_severity": 78,
            "pd_intensity": 350000,
            "asymmetry": 45
        },
        shap_features={
            "CURRENT_TOTAL": 0.35,
            "INTENSITY_ASYMMETRY": 0.30,
            "PULSE_TOTAL": 0.20
        }
    )
    
    print(f"\n✅ Ticket: {ticket2.ticket_id}")
    print(f"\n🛠️ Recommandation:\n{ticket2.llm_recommendation[:600]}...")
    
    # Stats
    print("\n" + "=" * 70)
    print("📊 STATISTIQUES")
    stats = engine.get_statistics()
    print(f"Total: {stats['total']}")
    print(f"Par priorité: {stats['by_priority']}")
    print(f"Temps moyen: {stats['avg_processing_time_ms']:.1f} ms")
