"""
🚀 TG1 Digital Twin - FastAPI Backend
=====================================
API REST pour l'application React

Auteur: Nadhir - Stage STEG 2026
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import json
import sys
import warnings
warnings.filterwarnings('ignore')

# Paths
BASE_DIR = Path(__file__).parent.parent
ML_MODELS_DIR = BASE_DIR / "ml_models" / "plots"
PD_MODELS_DIR = BASE_DIR / "pd_models" / "plots"
TG1_MODELS_DIR = BASE_DIR / "tg1_monitoring" / "plots"
DATA_DIR = BASE_DIR / "LAST_DATA"
TICKETING_DIR = BASE_DIR / "ticketing"

# Add ticketing module
sys.path.insert(0, str(TICKETING_DIR))

# Try importing smart ticket engine
try:
    from smart_ticket_engine import SmartTicketEngine, Module, Priority
    SMART_TICKETING_AVAILABLE = True
except ImportError:
    SMART_TICKETING_AVAILABLE = False

# Try TensorFlow
try:
    import tensorflow as tf
    from tensorflow import keras
    KERAS_AVAILABLE = True
except ImportError:
    KERAS_AVAILABLE = False

# =============================================================================
# FASTAPI APP
# =============================================================================

app = FastAPI(
    title="TG1 Digital Twin API",
    description="API REST pour le système de monitoring intelligent TG1",
    version="2.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# PYDANTIC MODELS
# =============================================================================

class HealthResponse(BaseModel):
    status: str
    version: str
    timestamp: str
    models_loaded: Dict[str, bool]

class ModelInfo(BaseModel):
    name: str
    type: str
    status: str
    path: Optional[str] = None

class PredictionRequest(BaseModel):
    model_name: str
    features: Dict[str, float]

class PredictionResponse(BaseModel):
    model: str
    prediction: Any
    confidence: Optional[float] = None
    timestamp: str

class TicketRequest(BaseModel):
    module: str = Field(..., description="THERMAL, COOLING, ELECTRICAL, PD, GLOBAL")
    severity_score: float = Field(..., ge=0, le=100)
    metrics: Dict[str, float] = {}
    ml_confidence: float = Field(0.85, ge=0, le=1)

class TicketResponse(BaseModel):
    ticket_id: str
    module: str
    priority: str
    severity_score: float
    status: str
    description: str
    recommendation: str
    root_cause: str
    estimated_rul: str
    timestamp: str

class AnomalyDetectionRequest(BaseModel):
    data: List[Dict[str, float]]
    model: str = "isolation_forest"

class DriftMetrics(BaseModel):
    metric_name: str
    current_value: float
    mean: float
    std: float
    ucl: float
    lcl: float
    is_out_of_control: bool

# =============================================================================
# GLOBAL STATE
# =============================================================================

class ModelRegistry:
    def __init__(self):
        self.ml_models = {}
        self.pd_models = {}
        self.tg1_models = {}
        self.ticket_engine = None
        self._load_models()
    
    def _load_models(self):
        """Load all ML models"""
        # ML Models
        ml_model_files = {
            'xgboost': '01_xgboost_model.pkl',
            'random_forest': '02_random_forest_model.pkl',
            'isolation_forest': '05_isolation_forest_model.pkl',
            'health_rf': '07_health_index_rf_model.pkl',
            'health_isoforest': '07_health_index_isoforest_model.pkl'
        }
        
        for name, filename in ml_model_files.items():
            try:
                self.ml_models[name] = {
                    'model': joblib.load(ML_MODELS_DIR / filename),
                    'status': 'loaded'
                }
            except Exception as e:
                self.ml_models[name] = {'status': 'error', 'error': str(e)}
        
        # Scalers
        try:
            self.ml_models['isolation_forest_scaler'] = joblib.load(ML_MODELS_DIR / '05_isolation_forest_scaler.pkl')
        except:
            pass
        
        # PD Models
        pd_model_files = {
            'xgb_classifier': '04_xgboost_classifier.pkl',
            'kmeans': '02_kmeans_model.pkl',
            'dbscan': '03_dbscan_model.pkl'
        }
        
        for name, filename in pd_model_files.items():
            try:
                self.pd_models[name] = {
                    'model': joblib.load(PD_MODELS_DIR / filename),
                    'status': 'loaded'
                }
            except Exception as e:
                self.pd_models[name] = {'status': 'error', 'error': str(e)}
        
        # TG1 Models
        tg1_model_files = {
            'thermal': '01_thermal_xgb_model.pkl',
            'cooling': '02_cooling_lr_model.pkl',
            'coupling': '04_xgb_coupling_model.pkl'
        }
        
        for name, filename in tg1_model_files.items():
            try:
                self.tg1_models[name] = {
                    'model': joblib.load(TG1_MODELS_DIR / filename),
                    'status': 'loaded'
                }
            except Exception as e:
                self.tg1_models[name] = {'status': 'error', 'error': str(e)}
        
        # Smart Ticketing
        if SMART_TICKETING_AVAILABLE:
            try:
                self.ticket_engine = SmartTicketEngine()
            except Exception as e:
                print(f"Ticket engine error: {e}")

registry = ModelRegistry()

# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.get("/", tags=["Root"])
async def root():
    return {"message": "TG1 Digital Twin API", "docs": "/api/docs"}

@app.get("/api/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Health check endpoint"""
    models_status = {
        'ml_models': len([m for m in registry.ml_models.values() if isinstance(m, dict) and m.get('status') == 'loaded']),
        'pd_models': len([m for m in registry.pd_models.values() if isinstance(m, dict) and m.get('status') == 'loaded']),
        'tg1_models': len([m for m in registry.tg1_models.values() if isinstance(m, dict) and m.get('status') == 'loaded']),
        'ticketing': registry.ticket_engine is not None
    }
    
    return HealthResponse(
        status="healthy",
        version="2.0.0",
        timestamp=datetime.now().isoformat(),
        models_loaded=models_status
    )

@app.get("/api/models", tags=["Models"])
async def list_models():
    """List all available models"""
    models = []
    
    for name, info in registry.ml_models.items():
        if isinstance(info, dict) and 'model' in info:
            models.append(ModelInfo(
                name=name,
                type="ml",
                status=info.get('status', 'unknown'),
                path=f"ml_models/plots/{name}"
            ))
    
    for name, info in registry.pd_models.items():
        if isinstance(info, dict) and 'model' in info:
            models.append(ModelInfo(
                name=name,
                type="pd",
                status=info.get('status', 'unknown'),
                path=f"pd_models/plots/{name}"
            ))
    
    for name, info in registry.tg1_models.items():
        if isinstance(info, dict) and 'model' in info:
            models.append(ModelInfo(
                name=name,
                type="tg1",
                status=info.get('status', 'unknown'),
                path=f"tg1_monitoring/plots/{name}"
            ))
    
    return {"models": models, "total": len(models)}

@app.get("/api/data/datasets", tags=["Data"])
async def list_datasets():
    """List available datasets"""
    datasets = []
    
    for file in DATA_DIR.glob("*.csv"):
        try:
            df = pd.read_csv(file, nrows=5)
            datasets.append({
                "name": file.stem,
                "filename": file.name,
                "columns": len(df.columns),
                "sample_columns": df.columns.tolist()[:10]
            })
        except:
            pass
    
    return {"datasets": datasets}

@app.get("/api/data/{dataset_name}", tags=["Data"])
async def get_dataset(dataset_name: str, limit: int = 1000, offset: int = 0):
    """Get dataset data"""
    file_path = DATA_DIR / f"{dataset_name}.csv"
    
    if not file_path.exists():
        # Try with _ML suffix
        file_path = DATA_DIR / f"{dataset_name}_ML.csv"
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"Dataset {dataset_name} not found")
    
    try:
        df = pd.read_csv(file_path, skiprows=range(1, offset + 1) if offset > 0 else None, nrows=limit)
        
        # Convert to JSON-serializable format
        data = df.replace({np.nan: None}).to_dict(orient='records')
        
        return {
            "dataset": dataset_name,
            "data": data,
            "columns": df.columns.tolist(),
            "count": len(data),
            "offset": offset,
            "limit": limit
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/data/{dataset_name}/stats", tags=["Data"])
async def get_dataset_stats(dataset_name: str):
    """Get dataset statistics"""
    file_path = DATA_DIR / f"{dataset_name}.csv"
    
    if not file_path.exists():
        file_path = DATA_DIR / f"{dataset_name}_ML.csv"
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"Dataset {dataset_name} not found")
    
    try:
        df = pd.read_csv(file_path)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        stats = {}
        for col in numeric_cols[:20]:  # Limit to 20 columns
            stats[col] = {
                "mean": float(df[col].mean()) if not pd.isna(df[col].mean()) else None,
                "std": float(df[col].std()) if not pd.isna(df[col].std()) else None,
                "min": float(df[col].min()) if not pd.isna(df[col].min()) else None,
                "max": float(df[col].max()) if not pd.isna(df[col].max()) else None,
                "median": float(df[col].median()) if not pd.isna(df[col].median()) else None
            }
        
        return {
            "dataset": dataset_name,
            "rows": len(df),
            "columns": len(df.columns),
            "numeric_columns": len(numeric_cols),
            "statistics": stats
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/predict/anomaly", tags=["Predictions"])
async def detect_anomalies(request: AnomalyDetectionRequest):
    """Detect anomalies using Isolation Forest"""
    if 'isolation_forest' not in registry.ml_models:
        raise HTTPException(status_code=404, detail="Isolation Forest model not loaded")
    
    try:
        model = registry.ml_models['isolation_forest']['model']
        scaler = registry.ml_models.get('isolation_forest_scaler')
        
        df = pd.DataFrame(request.data)
        X = df.select_dtypes(include=[np.number]).values
        
        if scaler:
            X = scaler.transform(X)
        
        predictions = model.predict(X)
        scores = model.decision_function(X)
        
        # -1 = anomaly, 1 = normal
        anomaly_indices = np.where(predictions == -1)[0].tolist()
        
        return {
            "total_samples": len(X),
            "anomalies_detected": len(anomaly_indices),
            "anomaly_ratio": len(anomaly_indices) / len(X) if len(X) > 0 else 0,
            "anomaly_indices": anomaly_indices,
            "scores": scores.tolist()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/drift/metrics", tags=["Drift Control"])
async def get_drift_metrics(dataset: str = "TG1_Sousse_ML"):
    """Get drift control metrics"""
    file_path = DATA_DIR / f"{dataset}.csv"
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"Dataset {dataset} not found")
    
    try:
        df = pd.read_csv(file_path)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()[:10]
        
        metrics = []
        for col in numeric_cols:
            values = df[col].dropna()
            if len(values) > 0:
                mean = values.mean()
                std = values.std()
                current = values.iloc[-1] if len(values) > 0 else mean
                ucl = mean + 3 * std
                lcl = mean - 3 * std
                
                metrics.append(DriftMetrics(
                    metric_name=col,
                    current_value=float(current),
                    mean=float(mean),
                    std=float(std),
                    ucl=float(ucl),
                    lcl=float(lcl),
                    is_out_of_control=current > ucl or current < lcl
                ))
        
        return {"metrics": metrics, "dataset": dataset}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# =============================================================================
# TICKETING ENDPOINTS
# =============================================================================

@app.get("/api/tickets", tags=["Ticketing"])
async def list_tickets(
    status: Optional[str] = None,
    priority: Optional[str] = None,
    module: Optional[str] = None,
    limit: int = 50
):
    """List all tickets"""
    if not registry.ticket_engine:
        raise HTTPException(status_code=503, detail="Ticketing engine not available")
    
    df = registry.ticket_engine.export_to_dataframe()
    
    if df.empty:
        return {"tickets": [], "total": 0}
    
    # Apply filters
    if status:
        df = df[df['status'] == status]
    if priority:
        df = df[df['priority'] == priority]
    if module:
        df = df[df['module'] == module]
    
    # Convert to list
    tickets = df.tail(limit).to_dict(orient='records')
    
    return {"tickets": tickets, "total": len(tickets)}

@app.get("/api/tickets/stats", tags=["Ticketing"])
async def get_ticket_stats():
    """Get ticket statistics"""
    if not registry.ticket_engine:
        raise HTTPException(status_code=503, detail="Ticketing engine not available")
    
    stats = registry.ticket_engine.get_statistics()
    return stats

@app.post("/api/tickets", response_model=TicketResponse, tags=["Ticketing"])
async def create_ticket(request: TicketRequest):
    """Create a new smart ticket"""
    if not registry.ticket_engine:
        raise HTTPException(status_code=503, detail="Ticketing engine not available")
    
    try:
        module = Module[request.module.upper()]
    except KeyError:
        raise HTTPException(status_code=400, detail=f"Invalid module: {request.module}")
    
    ticket = registry.ticket_engine.generate_smart_ticket(
        module=module,
        severity_score=request.severity_score,
        metrics=request.metrics,
        ml_confidence=request.ml_confidence
    )
    
    return TicketResponse(
        ticket_id=ticket.ticket_id,
        module=ticket.module,
        priority=ticket.priority,
        severity_score=ticket.severity_score,
        status=ticket.status,
        description=ticket.llm_description[:500] if ticket.llm_description else "",
        recommendation=ticket.llm_recommendation[:500] if ticket.llm_recommendation else "",
        root_cause=ticket.llm_root_cause[:500] if ticket.llm_root_cause else "",
        estimated_rul=ticket.estimated_rul,
        timestamp=ticket.timestamp.isoformat() if hasattr(ticket.timestamp, 'isoformat') else str(ticket.timestamp)
    )

@app.get("/api/tickets/{ticket_id}", tags=["Ticketing"])
async def get_ticket(ticket_id: str):
    """Get a specific ticket"""
    if not registry.ticket_engine:
        raise HTTPException(status_code=503, detail="Ticketing engine not available")
    
    df = registry.ticket_engine.export_to_dataframe()
    ticket_data = df[df['ticket_id'] == ticket_id]
    
    if ticket_data.empty:
        raise HTTPException(status_code=404, detail=f"Ticket {ticket_id} not found")
    
    return ticket_data.iloc[0].to_dict()

@app.patch("/api/tickets/{ticket_id}/status", tags=["Ticketing"])
async def update_ticket_status(ticket_id: str, new_status: str):
    """Update ticket status"""
    if not registry.ticket_engine:
        raise HTTPException(status_code=503, detail="Ticketing engine not available")
    
    success = registry.ticket_engine.update_ticket_status(ticket_id, new_status)
    
    if not success:
        raise HTTPException(status_code=404, detail=f"Ticket {ticket_id} not found")
    
    return {"message": f"Ticket {ticket_id} updated to {new_status}"}

# =============================================================================
# HEALTH INDEX ENDPOINT
# =============================================================================

@app.get("/api/health-index", tags=["Monitoring"])
async def get_health_index():
    """Calculate current system health index"""
    try:
        # Load latest data
        df = pd.read_csv(DATA_DIR / "TG1_Sousse_ML.csv", nrows=100)
        
        health_score = 100
        issues = []
        
        # Check temperature columns
        temp_cols = [c for c in df.columns if 'temp' in c.lower() or 'T_' in c]
        if temp_cols:
            max_temp = df[temp_cols].max().max()
            if max_temp > 90:
                health_score -= 30
                issues.append(f"Critical temperature: {max_temp:.1f}°C")
            elif max_temp > 80:
                health_score -= 15
                issues.append(f"High temperature: {max_temp:.1f}°C")
        
        # Check for anomalies using Isolation Forest
        if 'isolation_forest' in registry.ml_models and registry.ml_models['isolation_forest'].get('status') == 'loaded':
            model = registry.ml_models['isolation_forest']['model']
            numeric_cols = df.select_dtypes(include=[np.number]).columns[:10]
            X = df[numeric_cols].dropna()
            
            if len(X) > 0:
                predictions = model.predict(X.values)
                anomaly_ratio = (predictions == -1).sum() / len(predictions)
                
                if anomaly_ratio > 0.2:
                    health_score -= 25
                    issues.append(f"High anomaly rate: {anomaly_ratio*100:.1f}%")
                elif anomaly_ratio > 0.1:
                    health_score -= 10
                    issues.append(f"Elevated anomalies: {anomaly_ratio*100:.1f}%")
        
        health_score = max(0, health_score)
        
        return {
            "health_index": health_score,
            "status": "CRITICAL" if health_score < 50 else "WARNING" if health_score < 75 else "HEALTHY",
            "issues": issues,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# =============================================================================
# RUN
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
