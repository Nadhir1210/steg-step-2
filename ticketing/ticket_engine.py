"""
🎫 MAINTENANCE TICKETING ENGINE
================================
Système automatisé de génération de tickets de maintenance
basé sur la détection d'anomalies ML.

Auteur: Nadhir - Stage STEG 2026
"""

import uuid
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum


class Priority(Enum):
    """Niveaux de priorité des tickets"""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class TicketStatus(Enum):
    """États possibles d'un ticket"""
    OPEN = "OPEN"
    IN_PROGRESS = "IN_PROGRESS"
    PENDING = "PENDING"
    RESOLVED = "RESOLVED"
    CLOSED = "CLOSED"
    AUTO_CLOSED = "AUTO_CLOSED"


class Module(Enum):
    """Modules du système TG1"""
    THERMAL = "THERMAL"
    COOLING = "COOLING"
    ELECTRICAL = "ELECTRICAL"
    PD = "PD"
    GLOBAL = "GLOBAL"


class AnomalyType(Enum):
    """Types d'anomalies détectées"""
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
class MaintenanceTicket:
    """Structure d'un ticket de maintenance"""
    ticket_id: str
    timestamp: str
    module: str
    anomaly_type: str
    severity_score: float
    priority: str
    status: str
    description: str
    root_cause: str
    recommended_action: str
    assigned_service: str
    estimated_rul: Optional[str]
    related_features: List[str]
    threshold_exceeded: Dict[str, float]
    auto_close_eligible: bool
    created_by: str
    updated_at: str
    resolution_notes: str


# =============================================================================
# CONFIGURATION DES SEUILS
# =============================================================================

SEVERITY_THRESHOLDS = {
    "NORMAL": (0, 40),
    "WARNING": (40, 70),
    "HIGH": (70, 90),
    "CRITICAL": (90, 100)
}

# Seuils par module
MODULE_THRESHOLDS = {
    Module.THERMAL: {
        "temp_max": 90,  # °C
        "temp_warning": 80,
        "temp_critical": 100,
        "gradient_warning": 5,  # °C/min
    },
    Module.COOLING: {
        "delta_t_min": 5,  # °C
        "delta_t_warning": 3,
        "efficiency_min": 0.7,
    },
    Module.ELECTRICAL: {
        "frequency_tolerance": 0.5,  # Hz
        "voltage_tolerance": 5,  # %
        "asymmetry_max": 10,  # %
    },
    Module.PD: {
        "intensity_warning": 100000,
        "intensity_critical": 500000,
        "severity_warning": 50,
        "severity_critical": 75,
    }
}

# Services responsables
SERVICE_MAPPING = {
    Module.THERMAL: "Maintenance Mécanique",
    Module.COOLING: "Maintenance Mécanique",
    Module.ELECTRICAL: "Maintenance Électrique",
    Module.PD: "Maintenance Électrique - Isolation",
    Module.GLOBAL: "Direction Maintenance",
}

# Actions recommandées
ACTION_MAPPING = {
    AnomalyType.TEMPERATURE_HIGH: "Vérifier système de refroidissement et isolation thermique",
    AnomalyType.TEMPERATURE_CRITICAL: "ARRÊT IMMÉDIAT - Inspection complète du stator",
    AnomalyType.COOLING_INEFFICIENCY: "Inspecter échangeurs thermiques et ventilateurs",
    AnomalyType.ELECTRICAL_INSTABILITY: "Vérifier connexions et équilibre des phases",
    AnomalyType.PD_ACTIVITY_HIGH: "Planifier inspection isolation stator",
    AnomalyType.PD_CRITICAL: "Intervention urgente - Test isolation complète",
    AnomalyType.PHASE_ASYMMETRY: "Contrôler équilibrage des phases et connexions",
    AnomalyType.DRIFT_DETECTED: "Recalibrer modèles ML et vérifier capteurs",
    AnomalyType.HEALTH_DEGRADATION: "Analyse globale - inspection multi-systèmes",
    AnomalyType.MULTI_FAULT: "Intervention équipe pluridisciplinaire requise",
}


# =============================================================================
# TICKET ENGINE
# =============================================================================

class TicketEngine:
    """Moteur de génération de tickets de maintenance"""
    
    def __init__(self, storage_path: Optional[Path] = None):
        """
        Initialiser le moteur de tickets
        
        Args:
            storage_path: Chemin pour stocker les tickets (JSON)
        """
        self.storage_path = storage_path or Path(__file__).parent / "tickets_db.json"
        self.tickets: List[MaintenanceTicket] = []
        self._load_tickets()
    
    def _load_tickets(self):
        """Charger les tickets existants"""
        if self.storage_path.exists():
            try:
                with open(self.storage_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.tickets = [MaintenanceTicket(**t) for t in data]
            except Exception as e:
                print(f"Erreur chargement tickets: {e}")
                self.tickets = []
    
    def _save_tickets(self):
        """Sauvegarder les tickets"""
        try:
            with open(self.storage_path, 'w', encoding='utf-8') as f:
                data = [asdict(t) for t in self.tickets]
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Erreur sauvegarde tickets: {e}")
    
    @staticmethod
    def define_priority(severity_score: float) -> Priority:
        """
        Définir la priorité basée sur le score de sévérité
        
        Args:
            severity_score: Score entre 0 et 100
            
        Returns:
            Priority enum
        """
        if severity_score >= 90:
            return Priority.CRITICAL
        elif severity_score >= 70:
            return Priority.HIGH
        elif severity_score >= 40:
            return Priority.MEDIUM
        else:
            return Priority.LOW
    
    @staticmethod
    def estimate_rul(severity_score: float, trend: float = 0) -> Optional[str]:
        """
        Estimer la durée de vie restante (RUL)
        
        Args:
            severity_score: Score de sévérité actuel
            trend: Tendance de dégradation (positif = aggravation)
            
        Returns:
            Estimation RUL en texte
        """
        if severity_score >= 90:
            return "< 24 heures"
        elif severity_score >= 75:
            return "24-72 heures"
        elif severity_score >= 60:
            return "1-2 semaines"
        elif severity_score >= 40:
            return "1-3 mois"
        else:
            return "Pas de dégradation significative"
    
    def detect_anomaly_type(self, 
                           module: Module, 
                           metrics: Dict[str, float]) -> Tuple[AnomalyType, str]:
        """
        Détecter le type d'anomalie basé sur les métriques
        
        Args:
            module: Module concerné
            metrics: Dictionnaire des métriques
            
        Returns:
            Tuple (AnomalyType, description)
        """
        thresholds = MODULE_THRESHOLDS.get(module, {})
        
        if module == Module.THERMAL:
            temp = metrics.get('temperature', 0)
            if temp >= thresholds.get('temp_critical', 100):
                return (AnomalyType.TEMPERATURE_CRITICAL, 
                       f"Température critique: {temp:.1f}°C (seuil: {thresholds.get('temp_critical')}°C)")
            elif temp >= thresholds.get('temp_warning', 80):
                return (AnomalyType.TEMPERATURE_HIGH,
                       f"Température élevée: {temp:.1f}°C (seuil: {thresholds.get('temp_warning')}°C)")
        
        elif module == Module.COOLING:
            delta_t = metrics.get('delta_t', 0)
            efficiency = metrics.get('efficiency', 1)
            if delta_t < thresholds.get('delta_t_warning', 3) or efficiency < thresholds.get('efficiency_min', 0.7):
                return (AnomalyType.COOLING_INEFFICIENCY,
                       f"ΔT: {delta_t:.1f}°C, Efficacité: {efficiency*100:.1f}%")
        
        elif module == Module.ELECTRICAL:
            freq_dev = abs(metrics.get('frequency_deviation', 0))
            asymmetry = metrics.get('asymmetry', 0)
            if asymmetry > thresholds.get('asymmetry_max', 10):
                return (AnomalyType.PHASE_ASYMMETRY,
                       f"Asymétrie phases: {asymmetry:.1f}% (max: {thresholds.get('asymmetry_max')}%)")
            elif freq_dev > thresholds.get('frequency_tolerance', 0.5):
                return (AnomalyType.ELECTRICAL_INSTABILITY,
                       f"Déviation fréquence: {freq_dev:.2f} Hz")
        
        elif module == Module.PD:
            intensity = metrics.get('pd_intensity', 0)
            severity = metrics.get('pd_severity', 0)
            if severity >= thresholds.get('severity_critical', 75):
                return (AnomalyType.PD_CRITICAL,
                       f"Score PD critique: {severity:.1f}/100")
            elif severity >= thresholds.get('severity_warning', 50):
                return (AnomalyType.PD_ACTIVITY_HIGH,
                       f"Activité PD élevée: {severity:.1f}/100")
        
        # Par défaut
        return (AnomalyType.HEALTH_DEGRADATION, "Dégradation détectée par modèle ML")
    
    def generate_root_cause(self, 
                           module: Module,
                           metrics: Dict[str, float],
                           shap_features: Optional[List[Tuple[str, float]]] = None) -> str:
        """
        Générer une suggestion de cause racine basée sur SHAP
        
        Args:
            module: Module concerné
            metrics: Métriques actuelles
            shap_features: Top features SHAP [(name, value), ...]
            
        Returns:
            Description de la cause probable
        """
        causes = []
        
        if shap_features:
            top_features = shap_features[:3]
            causes.append("Basé sur analyse SHAP:")
            for feat, val in top_features:
                causes.append(f"  - {feat}: contribution {val:.2f}")
        
        # Analyse contextuelle
        if module == Module.THERMAL:
            if metrics.get('load', 0) > 100:
                causes.append("- Charge élevée détectée")
            if metrics.get('ambient_temp', 0) > 35:
                causes.append("- Température ambiante élevée")
        
        elif module == Module.PD:
            if metrics.get('asymmetry', 0) > 50:
                causes.append("- Forte asymétrie inter-canaux")
            if metrics.get('trend', 0) > 0:
                causes.append("- Tendance à la dégradation")
        
        return "\n".join(causes) if causes else "Analyse en cours"
    
    def generate_ticket(self,
                       module: Module,
                       severity_score: float,
                       metrics: Dict[str, float],
                       shap_features: Optional[List[Tuple[str, float]]] = None,
                       auto_close_eligible: bool = True) -> MaintenanceTicket:
        """
        Générer un ticket de maintenance complet
        
        Args:
            module: Module concerné
            severity_score: Score de sévérité (0-100)
            metrics: Métriques détectées
            shap_features: Features SHAP pour root cause
            auto_close_eligible: Si le ticket peut être fermé automatiquement
            
        Returns:
            MaintenanceTicket créé
        """
        # Détection du type d'anomalie
        anomaly_type, description = self.detect_anomaly_type(module, metrics)
        
        # Définir priorité
        priority = self.define_priority(severity_score)
        
        # Estimer RUL
        rul = self.estimate_rul(severity_score, metrics.get('trend', 0))
        
        # Root cause
        root_cause = self.generate_root_cause(module, metrics, shap_features)
        
        # Action recommandée
        recommended_action = ACTION_MAPPING.get(anomaly_type, "Inspection requise")
        
        # Service responsable
        assigned_service = SERVICE_MAPPING.get(module, "Maintenance Générale")
        
        # Créer le ticket
        ticket = MaintenanceTicket(
            ticket_id=f"TKT-{datetime.now().strftime('%Y%m%d')}-{str(uuid.uuid4())[:8].upper()}",
            timestamp=datetime.now().isoformat(),
            module=module.value,
            anomaly_type=anomaly_type.value,
            severity_score=severity_score,
            priority=priority.value,
            status=TicketStatus.OPEN.value,
            description=description,
            root_cause=root_cause,
            recommended_action=recommended_action,
            assigned_service=assigned_service,
            estimated_rul=rul,
            related_features=list(metrics.keys()),
            threshold_exceeded={k: v for k, v in metrics.items() if v is not None},
            auto_close_eligible=auto_close_eligible,
            created_by="ML_DETECTION_SYSTEM",
            updated_at=datetime.now().isoformat(),
            resolution_notes=""
        )
        
        # Sauvegarder
        self.tickets.append(ticket)
        self._save_tickets()
        
        return ticket
    
    def auto_close_tickets(self, current_metrics: Dict[str, float], threshold_margin: float = 0.9):
        """
        Fermer automatiquement les tickets si l'anomalie disparaît
        
        Args:
            current_metrics: Métriques actuelles
            threshold_margin: Marge pour considérer l'anomalie résolue
        """
        for ticket in self.tickets:
            if ticket.status == TicketStatus.OPEN.value and ticket.auto_close_eligible:
                # Vérifier si les métriques sont revenues à la normale
                severity_key = 'severity_score'
                if current_metrics.get(severity_key, 100) < 40:
                    ticket.status = TicketStatus.AUTO_CLOSED.value
                    ticket.updated_at = datetime.now().isoformat()
                    ticket.resolution_notes = "Fermé automatiquement - Métriques revenues à la normale"
        
        self._save_tickets()
    
    def update_ticket_status(self, ticket_id: str, status: TicketStatus, notes: str = ""):
        """Mettre à jour le statut d'un ticket"""
        for ticket in self.tickets:
            if ticket.ticket_id == ticket_id:
                ticket.status = status.value
                ticket.updated_at = datetime.now().isoformat()
                if notes:
                    ticket.resolution_notes = notes
                break
        self._save_tickets()
    
    def get_open_tickets(self) -> List[MaintenanceTicket]:
        """Obtenir tous les tickets ouverts"""
        return [t for t in self.tickets if t.status in [TicketStatus.OPEN.value, TicketStatus.IN_PROGRESS.value]]
    
    def get_tickets_by_priority(self, priority: Priority) -> List[MaintenanceTicket]:
        """Obtenir les tickets par priorité"""
        return [t for t in self.tickets if t.priority == priority.value]
    
    def get_tickets_by_module(self, module: Module) -> List[MaintenanceTicket]:
        """Obtenir les tickets par module"""
        return [t for t in self.tickets if t.module == module.value]
    
    def get_statistics(self) -> Dict:
        """Obtenir les statistiques des tickets"""
        total = len(self.tickets)
        if total == 0:
            return {"total": 0}
        
        stats = {
            "total": total,
            "open": len([t for t in self.tickets if t.status == TicketStatus.OPEN.value]),
            "in_progress": len([t for t in self.tickets if t.status == TicketStatus.IN_PROGRESS.value]),
            "resolved": len([t for t in self.tickets if t.status == TicketStatus.RESOLVED.value]),
            "closed": len([t for t in self.tickets if t.status in [TicketStatus.CLOSED.value, TicketStatus.AUTO_CLOSED.value]]),
            "by_priority": {
                "CRITICAL": len([t for t in self.tickets if t.priority == Priority.CRITICAL.value]),
                "HIGH": len([t for t in self.tickets if t.priority == Priority.HIGH.value]),
                "MEDIUM": len([t for t in self.tickets if t.priority == Priority.MEDIUM.value]),
                "LOW": len([t for t in self.tickets if t.priority == Priority.LOW.value]),
            },
            "by_module": {
                m.value: len([t for t in self.tickets if t.module == m.value])
                for m in Module
            },
            "avg_severity": np.mean([t.severity_score for t in self.tickets]),
        }
        
        return stats
    
    def export_to_dataframe(self) -> pd.DataFrame:
        """Exporter les tickets en DataFrame"""
        if not self.tickets:
            return pd.DataFrame()
        return pd.DataFrame([asdict(t) for t in self.tickets])


# =============================================================================
# ANOMALY DETECTOR INTEGRATION
# =============================================================================

class AnomalyTicketIntegrator:
    """
    Intégrateur entre la détection d'anomalies et le système de tickets
    """
    
    def __init__(self, ticket_engine: TicketEngine):
        self.engine = ticket_engine
    
    def process_health_data(self, 
                           health_df: pd.DataFrame,
                           threshold: float = 70) -> List[MaintenanceTicket]:
        """
        Traiter les données de santé et générer des tickets
        
        Args:
            health_df: DataFrame avec colonnes de score de santé
            threshold: Seuil pour déclencher un ticket
            
        Returns:
            Liste des tickets générés
        """
        generated_tickets = []
        
        # Vérifier Health Index global
        if 'HEALTH_INDEX' in health_df.columns:
            latest = health_df['HEALTH_INDEX'].iloc[-1] if len(health_df) > 0 else 100
            if latest < threshold:
                severity = 100 - latest  # Inverser pour avoir le score de sévérité
                metrics = {'health_index': latest}
                
                ticket = self.engine.generate_ticket(
                    module=Module.GLOBAL,
                    severity_score=severity,
                    metrics=metrics
                )
                generated_tickets.append(ticket)
        
        # Vérifier scores individuels
        score_columns = {
            'THERMAL_SCORE': Module.THERMAL,
            'COOLING_SCORE': Module.COOLING,
            'ELECTRICAL_SCORE': Module.ELECTRICAL,
            'PD_SCORE': Module.PD
        }
        
        for col, module in score_columns.items():
            if col in health_df.columns:
                latest = health_df[col].iloc[-1] if len(health_df) > 0 else 100
                if latest < threshold:
                    severity = 100 - latest
                    metrics = {col.lower(): latest}
                    
                    ticket = self.engine.generate_ticket(
                        module=module,
                        severity_score=severity,
                        metrics=metrics
                    )
                    generated_tickets.append(ticket)
        
        return generated_tickets
    
    def process_pd_anomalies(self,
                            pd_df: pd.DataFrame,
                            severity_col: str = 'PD_SEVERITY_SCORE') -> List[MaintenanceTicket]:
        """
        Traiter les anomalies PD
        """
        generated_tickets = []
        
        if severity_col in pd_df.columns:
            # Prendre les dernières valeurs critiques
            critical_mask = pd_df[severity_col] >= 70
            if critical_mask.any():
                critical_rows = pd_df[critical_mask].tail(5)
                
                for _, row in critical_rows.iterrows():
                    severity = row[severity_col]
                    metrics = {
                        'pd_severity': severity,
                        'pd_intensity': row.get('PD_INTENSITY_TOTAL', 0),
                        'asymmetry': row.get('INTENSITY_ASYMMETRY', 0)
                    }
                    
                    # SHAP features si disponibles
                    shap_features = None
                    if 'CURRENT_TOTAL' in row.index:
                        shap_features = [
                            ('CURRENT_TOTAL', row.get('CURRENT_TOTAL', 0)),
                            ('PULSE_TOTAL', row.get('PULSE_TOTAL', 0)),
                        ]
                    
                    ticket = self.engine.generate_ticket(
                        module=Module.PD,
                        severity_score=severity,
                        metrics=metrics,
                        shap_features=shap_features
                    )
                    generated_tickets.append(ticket)
        
        return generated_tickets


# =============================================================================
# DEMO / TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("🎫 MAINTENANCE TICKETING ENGINE - TEST")
    print("=" * 60)
    
    # Créer le moteur
    engine = TicketEngine()
    
    # Générer quelques tickets de test
    test_cases = [
        {
            "module": Module.THERMAL,
            "severity_score": 85,
            "metrics": {"temperature": 92, "load": 110, "ambient_temp": 38}
        },
        {
            "module": Module.PD,
            "severity_score": 72,
            "metrics": {"pd_severity": 72, "pd_intensity": 250000, "asymmetry": 45}
        },
        {
            "module": Module.ELECTRICAL,
            "severity_score": 55,
            "metrics": {"frequency_deviation": 0.8, "asymmetry": 12}
        },
        {
            "module": Module.COOLING,
            "severity_score": 45,
            "metrics": {"delta_t": 2.5, "efficiency": 0.65}
        }
    ]
    
    print("\n📝 Génération de tickets de test...")
    for case in test_cases:
        ticket = engine.generate_ticket(**case)
        print(f"\n✅ Ticket créé: {ticket.ticket_id}")
        print(f"   Module: {ticket.module}")
        print(f"   Type: {ticket.anomaly_type}")
        print(f"   Priorité: {ticket.priority}")
        print(f"   Sévérité: {ticket.severity_score}/100")
        print(f"   RUL: {ticket.estimated_rul}")
        print(f"   Service: {ticket.assigned_service}")
    
    # Statistiques
    print("\n📊 STATISTIQUES")
    print("-" * 40)
    stats = engine.get_statistics()
    print(f"Total tickets: {stats['total']}")
    print(f"Ouverts: {stats['open']}")
    print(f"Par priorité: {stats['by_priority']}")
    
    print("\n✅ Tickets sauvegardés dans:", engine.storage_path)
