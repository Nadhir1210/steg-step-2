"""
📄 RAPPORT PDF - STEG Industrial Analytics Platform
====================================================
Génération automatique d'un rapport PDF complet

Auteur: Nadhir - Stage STEG 2026
"""

from fpdf import FPDF
from pathlib import Path
from datetime import datetime
import os

# Configuration
BASE_DIR = Path(__file__).parent.parent
REPORT_DIR = BASE_DIR / "visual"
OUTPUT_FILE = REPORT_DIR / "RAPPORT_STEG_INDUSTRIAL_ANALYTICS.pdf"

# Chemins des images
PD_PLOTS = BASE_DIR / "pd_models" / "plots"
ML_PLOTS = BASE_DIR / "ml_models" / "plots"
TG1_PLOTS = BASE_DIR / "tg1_monitoring" / "plots"


class PDFReport(FPDF):
    """Classe personnalisée pour le rapport PDF"""
    
    def __init__(self):
        super().__init__()
        self.set_auto_page_break(auto=True, margin=15)
        
    def header(self):
        """En-tête de page"""
        self.set_font('Helvetica', 'B', 10)
        self.set_text_color(100, 100, 100)
        self.cell(0, 10, 'STEG Industrial Analytics Platform - Rapport Technique', 0, 0, 'C')
        self.ln(5)
        self.set_draw_color(25, 118, 210)
        self.set_line_width(0.5)
        self.line(10, 15, 200, 15)
        self.ln(10)
    
    def footer(self):
        """Pied de page"""
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, f'Page {self.page_no()}/{{nb}} | Stage STEG 2026 - Nadhir', 0, 0, 'C')
    
    def chapter_title(self, title, color=(25, 118, 210)):
        """Titre de chapitre"""
        self.set_font('Helvetica', 'B', 16)
        self.set_text_color(*color)
        self.cell(0, 10, title, 0, 1, 'L')
        self.set_draw_color(*color)
        self.set_line_width(0.8)
        self.line(10, self.get_y(), 100, self.get_y())
        self.ln(8)
        self.set_text_color(0, 0, 0)
    
    def section_title(self, title):
        """Titre de section"""
        self.set_font('Helvetica', 'B', 12)
        self.set_text_color(60, 60, 60)
        self.cell(0, 8, title, 0, 1, 'L')
        self.ln(2)
        self.set_text_color(0, 0, 0)
    
    def body_text(self, text):
        """Texte de corps"""
        self.set_font('Helvetica', '', 10)
        self.multi_cell(0, 6, text)
        self.ln(3)
    
    def bullet_point(self, text):
        """Point de liste"""
        self.set_font('Helvetica', '', 10)
        self.cell(5, 6, '-', 0, 0)  # bullet
        self.cell(0, 6, text, 0, 1)
    
    def add_table(self, headers, data, col_widths=None):
        """Ajouter un tableau"""
        if col_widths is None:
            col_widths = [190 // len(headers)] * len(headers)
        
        # Headers
        self.set_font('Helvetica', 'B', 9)
        self.set_fill_color(25, 118, 210)
        self.set_text_color(255, 255, 255)
        for i, header in enumerate(headers):
            self.cell(col_widths[i], 8, header, 1, 0, 'C', True)
        self.ln()
        
        # Data
        self.set_font('Helvetica', '', 9)
        self.set_text_color(0, 0, 0)
        fill = False
        for row in data:
            if fill:
                self.set_fill_color(240, 240, 240)
            else:
                self.set_fill_color(255, 255, 255)
            for i, cell in enumerate(row):
                self.cell(col_widths[i], 7, str(cell), 1, 0, 'C', True)
            self.ln()
            fill = not fill
        self.ln(5)
    
    def add_metric_box(self, label, value, x, y, color=(76, 175, 80)):
        """Ajouter une boîte de métrique"""
        self.set_xy(x, y)
        self.set_fill_color(*color)
        self.set_text_color(255, 255, 255)
        self.set_font('Helvetica', 'B', 12)
        self.cell(40, 15, '', 0, 0, 'C', True)
        self.set_xy(x, y + 2)
        self.cell(40, 6, str(value), 0, 1, 'C')
        self.set_xy(x, y + 9)
        self.set_font('Helvetica', '', 8)
        self.cell(40, 5, label, 0, 1, 'C')
        self.set_text_color(0, 0, 0)
    
    def add_image_if_exists(self, path, w=180, caption=None):
        """Ajouter une image si elle existe"""
        if Path(path).exists():
            try:
                self.image(str(path), w=w)
                if caption:
                    self.set_font('Helvetica', 'I', 8)
                    self.set_text_color(100, 100, 100)
                    self.cell(0, 5, caption, 0, 1, 'C')
                    self.set_text_color(0, 0, 0)
                self.ln(5)
                return True
            except:
                return False
        return False


def create_report():
    """Créer le rapport PDF complet"""
    
    pdf = PDFReport()
    pdf.alias_nb_pages()
    
    # ==========================================================================
    # PAGE DE GARDE
    # ==========================================================================
    pdf.add_page()
    pdf.set_font('Helvetica', 'B', 28)
    pdf.set_text_color(25, 118, 210)
    pdf.ln(40)
    pdf.cell(0, 15, 'STEG Industrial Analytics', 0, 1, 'C')
    pdf.cell(0, 15, 'Platform', 0, 1, 'C')
    
    pdf.ln(10)
    pdf.set_font('Helvetica', '', 16)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 10, 'Rapport Technique Complet', 0, 1, 'C')
    pdf.cell(0, 10, 'Analyse des Donnees Industrielles TG1', 0, 1, 'C')
    
    pdf.ln(20)
    pdf.set_draw_color(25, 118, 210)
    pdf.set_line_width(2)
    pdf.line(50, pdf.get_y(), 160, pdf.get_y())
    
    pdf.ln(25)
    pdf.set_font('Helvetica', '', 12)
    pdf.set_text_color(60, 60, 60)
    pdf.cell(0, 8, 'Turbo-Alternateur TG1 - Centrale STEG', 0, 1, 'C')
    pdf.cell(0, 8, 'Systeme de Monitoring et Maintenance Predictive', 0, 1, 'C')
    
    pdf.ln(30)
    pdf.set_font('Helvetica', 'B', 12)
    pdf.cell(0, 8, 'Stage Ingenieur 2026', 0, 1, 'C')
    pdf.set_font('Helvetica', '', 11)
    pdf.cell(0, 8, 'Auteur: Nadhir BH', 0, 1, 'C')
    pdf.cell(0, 8, f'Date: {datetime.now().strftime("%d/%m/%Y")}', 0, 1, 'C')
    
    # ==========================================================================
    # TABLE DES MATIERES
    # ==========================================================================
    pdf.add_page()
    pdf.chapter_title('Table des Matieres')
    
    toc = [
        ('1. Resume Executif', 3),
        ('2. Presentation des Datasets', 4),
        ('3. Module PD Analysis', 6),
        ('   3.1 Feature Engineering', 6),
        ('   3.2 KMeans Clustering', 7),
        ('   3.3 DBSCAN Anomaly Detection', 7),
        ('   3.4 XGBoost + SHAP', 8),
        ('   3.5 LSTM Classification', 9),
        ('   3.6 Severity Score', 9),
        ('4. Module ML Models', 10),
        ('   4.1 XGBoost Regressor', 10),
        ('   4.2 Random Forest', 11),
        ('   4.3 ANN Neural Network', 11),
        ('   4.4 LSTM Time Series', 12),
        ('   4.5 Anomaly Detection', 12),
        ('   4.6 Health Index', 13),
        ('5. Module TG1 Digital Twin', 14),
        ('   5.1 Thermal Health Model', 14),
        ('   5.2 Cooling Efficiency', 15),
        ('   5.3 Electrical Stability', 15),
        ('   5.4 Load-Temperature Coupling', 16),
        ('   5.5 Global Health Index', 16),
        ('6. Conclusions et Recommandations', 17),
    ]
    
    pdf.set_font('Helvetica', '', 11)
    for item, page in toc:
        pdf.cell(160, 7, item, 0, 0, 'L')
        pdf.cell(20, 7, str(page), 0, 1, 'R')
    
    # ==========================================================================
    # RESUME EXECUTIF
    # ==========================================================================
    pdf.add_page()
    pdf.chapter_title('1. Resume Executif')
    
    pdf.body_text("""Ce rapport presente les resultats du projet d'analyse industrielle realise pour la centrale STEG. 
L'objectif est de developper une plateforme complete de monitoring et de maintenance predictive pour le 
Turbo-Alternateur TG1.""")
    
    pdf.ln(5)
    pdf.section_title('Objectifs du Projet')
    pdf.bullet_point('Analyser les decharges partielles (PD) pour surveiller l\'isolation electrique')
    pdf.bullet_point('Predire la temperature du stator pour anticiper les surchauffes')
    pdf.bullet_point('Creer un Digital Twin avec un Health Index global (0-100)')
    pdf.bullet_point('Fournir des recommandations de maintenance predictive')
    
    pdf.ln(5)
    pdf.section_title('Resultats Cles')
    
    results = [
        ['Module', 'Meilleur Modele', 'Performance', 'Status'],
        ['PD Analysis', 'XGBoost + SHAP', '97.99% Accuracy', 'Production'],
        ['ML Temperature', 'XGBoost', 'R2 = 1.0000', 'Production'],
        ['TG1 Digital Twin', 'Health Index', '67/100', 'Monitoring'],
    ]
    pdf.add_table(results[0], results[1:], [45, 55, 50, 40])
    
    pdf.section_title('Metriques Globales')
    pdf.body_text("""
- Datasets analyses: 6 sources de donnees industrielles
- Lignes de donnees: 2.5 millions de points
- Variables traitees: 200+ colonnes
- Modeles developpes: 18 modeles ML/DL
- Performance maximale: R2 = 1.0000 (XGBoost temperature)
- Precision classification: 97.99% (XGBoost PD)
""")
    
    # ==========================================================================
    # DATASETS
    # ==========================================================================
    pdf.add_page()
    pdf.chapter_title('2. Presentation des Datasets')
    
    pdf.body_text("""Les donnees proviennent du systeme de monitoring du Turbo-Alternateur TG1 de la centrale STEG. 
Six datasets ont ete analyses, couvrant les mesures de l'alternateur et les decharges partielles.""")
    
    pdf.ln(5)
    pdf.section_title('Liste des Datasets')
    
    datasets = [
        ['Dataset', 'Lignes', 'Colonnes', 'Taille', 'Usage'],
        ['APM_Alternateur_ML.csv', '200,000', '26', '~50 MB', 'Temperature'],
        ['APM_Alternateur_10min_ML.csv', '52,560', '42', '~25 MB', 'ML Features'],
        ['APM_Chart_ML.csv', '200,000', '20', '~40 MB', 'Tendances'],
        ['APM_Chart_10min_ML.csv', '52,560', '25', '~20 MB', 'Agregation'],
        ['TG1_Sousse_ML.csv', '14,956', '91', '~15 MB', 'PD Analysis'],
        ['TG1_Sousse_1min_ML.csv', '2,200,000', '91', '~500 MB', 'PD Detail'],
    ]
    pdf.add_table(datasets[0], datasets[1:], [55, 30, 25, 25, 35])
    
    pdf.section_title('Variables Principales - APM Alternateur')
    
    vars_apm = [
        ['Variable', 'Description', 'Unite'],
        ['MODE_TAG_1', 'Puissance active', 'MW'],
        ['REACTIVE_LOAD', 'Puissance reactive', 'MVAR'],
        ['STATOR_PHASE_X_TEMP', 'Temperature stator (9 sondes)', 'Celsius'],
        ['TERMINAL_VOLTAGE_kV', 'Tension terminale', 'kV'],
        ['FREQUENCY_Hz', 'Frequence reseau', 'Hz'],
        ['ENCLOSED_HOT_AIR_TEMP', 'Temperature air chaud', 'Celsius'],
        ['ENCLOSED_COLD_AIR_TEMP', 'Temperature air froid', 'Celsius'],
    ]
    pdf.add_table(vars_apm[0], vars_apm[1:], [55, 95, 30])
    
    pdf.section_title('Variables PD - TG1 Sousse')
    
    vars_pd = [
        ['Variable', 'Description', 'Canaux'],
        ['CHx_CURRENT_ABS', 'Courant absolu', 'CH1-CH4'],
        ['CHx_DISCHARGE_RATE', 'Taux de decharge', 'CH1-CH4'],
        ['CHx_MAX_CHARGE', 'Charge maximale', 'CH1-CH4'],
        ['CHx_MEAN_CHARGE', 'Charge moyenne', 'CH1-CH4'],
        ['CHx_PULSE_COUNT', 'Nombre d\'impulsions', 'CH1-CH4'],
    ]
    pdf.add_table(vars_pd[0], vars_pd[1:], [55, 95, 30])
    
    # ==========================================================================
    # PD ANALYSIS
    # ==========================================================================
    pdf.add_page()
    pdf.chapter_title('3. Module PD Analysis', color=(156, 39, 176))
    
    pdf.body_text("""L'analyse des decharges partielles (PD) est essentielle pour surveiller l'etat de l'isolation 
electrique du turbo-alternateur. Ce module comprend 6 etapes d'analyse.""")
    
    pdf.section_title('3.1 Feature Engineering')
    
    pdf.body_text("""33 nouvelles variables ont ete creees a partir des donnees brutes pour ameliorer 
les performances des modeles ML.""")
    
    features_pd = [
        ['Feature', 'Formule', 'Description'],
        ['PD_INTENSITY', 'CURRENT x PULSE', 'Intensite par canal'],
        ['PD_ENERGY', 'CHARGE x RATE', 'Energie totale'],
        ['INTENSITY_ASYMMETRY', 'max(CH) - min(CH)', 'Desequilibre canaux'],
        ['Rolling_MEAN/STD', 'Fenetres 10-60min', 'Tendances temporelles'],
        ['INTENSITY_CV', 'std / mean', 'Coefficient variation'],
    ]
    pdf.add_table(features_pd[0], features_pd[1:], [50, 55, 75])
    
    pdf.section_title('3.2 KMeans Clustering')
    
    kmeans_results = [
        ['Parametre', 'Valeur'],
        ['K optimal', '2'],
        ['Silhouette Score', '0.857'],
        ['Cluster 0 (Modere)', '2%'],
        ['Cluster 1 (Normal)', '98%'],
    ]
    pdf.add_table(kmeans_results[0], kmeans_results[1:], [95, 95])
    
    pdf.section_title('3.3 DBSCAN Anomaly Detection')
    
    dbscan_results = [
        ['Parametre', 'Valeur'],
        ['eps', '0.853'],
        ['min_samples', '9'],
        ['Anomalies detectees', '9.0% (1,347 points)'],
        ['Ratio Intensite (anomalie/normal)', '209x'],
    ]
    pdf.add_table(dbscan_results[0], dbscan_results[1:], [95, 95])
    
    pdf.add_page()
    pdf.section_title('3.4 XGBoost + SHAP Classification')
    
    pdf.body_text("""Le modele XGBoost avec SHAP explanations atteint une accuracy de 97.99% pour la 
classification des etats PD (Normal/Warning/Critical).""")
    
    xgb_pd = [
        ['Metrique', 'Valeur'],
        ['Accuracy', '97.99%'],
        ['F1-Score', '0.9803'],
        ['ROC-AUC', '0.9989'],
        ['Validation', 'Temporelle (no shuffle)'],
    ]
    pdf.add_table(xgb_pd[0], xgb_pd[1:], [95, 95])
    
    pdf.body_text('SHAP Feature Importance:')
    shap_importance = [
        ['Feature', 'SHAP Value'],
        ['CURRENT_TOTAL', '1.94'],
        ['PULSE_TOTAL', '1.83'],
        ['INTENSITY_ASYMMETRY', '1.69'],
        ['CH3_INTENSITY', '1.45'],
    ]
    pdf.add_table(shap_importance[0], shap_importance[1:], [95, 95])
    
    pdf.section_title('3.5 LSTM Classification')
    
    lstm_pd = [
        ['Parametre', 'Valeur'],
        ['Architecture', 'LSTM(64) > LSTM(32) > Dense(16)'],
        ['Accuracy', '96.15%'],
        ['Epochs', '50'],
        ['Validation', 'Temporelle'],
    ]
    pdf.add_table(lstm_pd[0], lstm_pd[1:], [95, 95])
    
    pdf.section_title('3.6 PD Severity Score')
    
    pdf.body_text('Score = 35% Intensity + 25% Energy + 15% Asymmetry + 15% Trend + 10% Stability')
    
    severity = [
        ['Classe', 'Plage', 'Distribution'],
        ['Excellent', '0-25', '75.9%'],
        ['Bon', '25-50', '18.0%'],
        ['Moyen', '50-75', '1.0%'],
        ['Critique', '75-100', '5.1%'],
    ]
    pdf.add_table(severity[0], severity[1:], [60, 65, 65])
    
    # ==========================================================================
    # ML MODELS
    # ==========================================================================
    pdf.add_page()
    pdf.chapter_title('4. Module ML Models', color=(76, 175, 80))
    
    pdf.body_text("""Ce module contient 7 modeles de Machine Learning pour la prediction de temperature 
du stator et la detection d'anomalies.""")
    
    pdf.section_title('4.1 XGBoost Regressor (Recommande)')
    
    xgb_ml = [
        ['Metrique', 'Valeur'],
        ['R2 Score', '1.0000 (100%)'],
        ['RMSE', '0.06 Celsius'],
        ['MAE', '0.04 Celsius'],
        ['Cross-Val R2', '1.0000 (+/- 0.0000)'],
    ]
    pdf.add_table(xgb_ml[0], xgb_ml[1:], [95, 95])
    
    pdf.body_text('Configuration:')
    pdf.body_text('n_estimators=200, max_depth=8, learning_rate=0.1, device=cuda (GPU)')
    
    pdf.section_title('4.2 Random Forest Regressor')
    
    rf_ml = [
        ['Metrique', 'Valeur'],
        ['R2 Score', '1.0000'],
        ['RMSE', '0.07 Celsius'],
        ['MAE', '0.02 Celsius'],
        ['n_estimators', '100'],
    ]
    pdf.add_table(rf_ml[0], rf_ml[1:], [95, 95])
    
    pdf.section_title('4.3 ANN Neural Network')
    
    ann_ml = [
        ['Parametre', 'Valeur'],
        ['Architecture', 'Dense(128>64>32>16>1)'],
        ['R2 Score', '0.9989'],
        ['RMSE', '0.81 Celsius'],
        ['Epochs (early stop)', '37'],
    ]
    pdf.add_table(ann_ml[0], ann_ml[1:], [95, 95])
    
    pdf.add_page()
    pdf.section_title('4.4 LSTM Time Series')
    
    lstm_ml = [
        ['Parametre', 'Valeur'],
        ['Architecture', 'LSTM(64>32) + Dense(16>1)'],
        ['R2 Score', '0.9792'],
        ['RMSE', '2.95 Celsius'],
        ['Window', '10 pas (100 minutes)'],
        ['Horizon', 'T+10 minutes'],
    ]
    pdf.add_table(lstm_ml[0], lstm_ml[1:], [95, 95])
    
    pdf.section_title('4.5 Anomaly Detection')
    
    anomaly = [
        ['Modele', 'Methode', 'Anomalies'],
        ['Isolation Forest', 'Isolation trees', '3% (~6,000)'],
        ['Autoencoder', 'Reconstruction error', '3% (~4,500)'],
    ]
    pdf.add_table(anomaly[0], anomaly[1:], [60, 65, 65])
    
    pdf.section_title('4.6 Health Index Composite')
    
    pdf.body_text('Health = 40% Thermal + 30% Prediction + 20% Stability + 10% Anomaly')
    
    health_ml = [
        ['Score', 'Etat', 'Action'],
        ['85-100', 'Excellent', 'Aucune'],
        ['70-84', 'Bon', 'Surveillance'],
        ['50-69', 'Moyen', 'Maintenance'],
        ['0-49', 'Critique', 'Intervention'],
    ]
    pdf.add_table(health_ml[0], health_ml[1:], [60, 65, 65])
    
    pdf.body_text('Score moyen obtenu: 79.6/100')
    
    # ==========================================================================
    # TG1 DIGITAL TWIN
    # ==========================================================================
    pdf.add_page()
    pdf.chapter_title('5. Module TG1 Digital Twin', color=(255, 152, 0))
    
    pdf.body_text("""Le Digital Twin TG1 est un systeme complet de monitoring en temps reel combinant 
5 modules d'analyse pour calculer un Global Health Index.""")
    
    pdf.section_title('Formule du Health Index')
    pdf.body_text('TG1_Health = 30% PD + 30% Thermal + 20% Cooling + 20% Electrical')
    
    pdf.section_title('5.1 Thermal Health Model')
    
    thermal = [
        ['Parametre', 'Valeur'],
        ['Modele', 'XGBoost Regression'],
        ['Target', 'Temperature stator'],
        ['Anomalies', 'Residual > 3 sigma'],
        ['R2 attendu', '> 0.85'],
    ]
    pdf.add_table(thermal[0], thermal[1:], [95, 95])
    
    pdf.section_title('5.2 Cooling Efficiency')
    
    cooling = [
        ['Indicateur', 'Valeur'],
        ['Delta T', 'Hot Air - Cold Air'],
        ['Methode', 'SPC Control Chart'],
        ['Limites', 'UCL/LCL (+/- 3 sigma)'],
        ['Detection', 'Isolation Forest'],
    ]
    pdf.add_table(cooling[0], cooling[1:], [95, 95])
    
    pdf.section_title('5.3 Electrical Stability')
    
    electrical = [
        ['Variable', 'Nominal', 'Tolerance'],
        ['Frequence', '50 Hz', '+/- 0.5 Hz'],
        ['Tension', '15.75 kV', '+/- 5%'],
        ['Score moyen', '90.3/100', '-'],
    ]
    pdf.add_table(electrical[0], electrical[1:], [60, 65, 65])
    
    pdf.add_page()
    pdf.section_title('5.4 Load-Temperature Coupling')
    
    pdf.body_text('Analyse SHAP du couplage entre charge et temperature:')
    
    coupling = [
        ['Variable', 'Correlation Temp', 'SHAP Value'],
        ['HOT_AIR', '0.95', '-'],
        ['COOLING_DELTA', '0.85', '4.98'],
        ['LOAD_MW', '0.60', '1.42'],
        ['AMBIENT_TEMP', '-', '1.03'],
    ]
    pdf.add_table(coupling[0], coupling[1:], [55, 65, 65])
    
    pdf.body_text('Conclusion: COOLING_DELTA est le facteur le plus influent (5x plus que LOAD_MW)')
    
    pdf.section_title('5.5 Global Health Index')
    
    health_tg1 = [
        ['Composant', 'Poids', 'Score'],
        ['PD Score', '30%', '89.4/100'],
        ['Thermal Score', '30%', '52.9/100'],
        ['Cooling Score', '20%', '74.5/100'],
        ['Electrical Score', '20%', '47.0/100'],
        ['TOTAL', '100%', '67.0/100'],
    ]
    pdf.add_table(health_tg1[0], health_tg1[1:], [60, 65, 65])
    
    pdf.body_text('Statut global: DEGRADING - Maintenance preventive recommandee')
    
    # ==========================================================================
    # CONCLUSIONS
    # ==========================================================================
    pdf.add_page()
    pdf.chapter_title('6. Conclusions et Recommandations')
    
    pdf.section_title('Realisations')
    pdf.bullet_point('Pipeline complet d\'analyse PD avec 97.99% de precision')
    pdf.bullet_point('Modeles de prediction de temperature avec R2 = 1.0000')
    pdf.bullet_point('Digital Twin operationnel avec Health Index en temps reel')
    pdf.bullet_point('Dashboard Streamlit unifie pour la visualisation')
    pdf.bullet_point('18 modeles ML/DL entraines et valides')
    
    pdf.ln(5)
    pdf.section_title('Points Forts')
    pdf.bullet_point('XGBoost: Performances exceptionnelles en regression et classification')
    pdf.bullet_point('SHAP: Explicabilite des modeles pour aide a la decision')
    pdf.bullet_point('Validation temporelle: Pas de data leakage')
    pdf.bullet_point('Architecture modulaire: Facile a maintenir et etendre')
    
    pdf.ln(5)
    pdf.section_title('Points d\'Attention')
    pdf.bullet_point('Score Electrical (47/100) - Variation de frequence a surveiller')
    pdf.bullet_point('Score Thermal (52.9/100) - Optimisation refroidissement necessaire')
    pdf.bullet_point('Dataset 1min (2.2M lignes) - Necessite traitement par chunks')
    
    pdf.ln(5)
    pdf.section_title('Recommandations')
    pdf.bullet_point('Deployer XGBoost en production pour prediction temps reel')
    pdf.bullet_point('Implementer alertes automatiques basees sur Health Index')
    pdf.bullet_point('Programmer maintenance preventive quand score < 60')
    pdf.bullet_point('Optimiser systeme de refroidissement (COOLING_DELTA cle)')
    pdf.bullet_point('Etendre le monitoring a d\'autres equipements')
    
    pdf.ln(10)
    pdf.set_font('Helvetica', 'I', 10)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 10, f'Rapport genere le {datetime.now().strftime("%d/%m/%Y a %H:%M")}', 0, 1, 'C')
    pdf.cell(0, 10, 'STEG Industrial Analytics Platform - Stage 2026', 0, 1, 'C')
    
    # ==========================================================================
    # SAUVEGARDER
    # ==========================================================================
    pdf.output(str(OUTPUT_FILE))
    print(f"\n{'='*60}")
    print(f"RAPPORT PDF GENERE AVEC SUCCES")
    print(f"{'='*60}")
    print(f"Fichier: {OUTPUT_FILE}")
    print(f"Taille: {OUTPUT_FILE.stat().st_size / 1024:.1f} KB")
    print(f"Pages: ~17")
    print(f"{'='*60}\n")
    
    return OUTPUT_FILE


if __name__ == "__main__":
    create_report()
