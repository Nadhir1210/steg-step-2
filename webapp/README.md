# TG1 Digital Twin - React Web Application

Professional React + FastAPI web application for the TG1 Digital Twin Health Monitoring System.

## Architecture

```
webapp/
├── backend/                # FastAPI Backend
│   ├── main.py            # API endpoints
│   └── requirements.txt   # Python dependencies
│
└── frontend/              # React Frontend
    ├── src/
    │   ├── components/    # Reusable UI components
    │   ├── pages/         # Application pages
    │   ├── services/      # API client
    │   └── types/         # TypeScript definitions
    ├── package.json
    └── vite.config.ts
```

## Tech Stack

### Backend
- **FastAPI** - Modern Python web framework
- **Pydantic** - Data validation
- **Uvicorn** - ASGI server

### Frontend
- **React 18** + TypeScript
- **Vite** - Build tool
- **TailwindCSS** - Styling
- **Material-UI** - Components
- **Recharts** - Data visualization
- **React Query** - Data fetching
- **Framer Motion** - Animations

## Quick Start

### 1. Backend

```bash
cd webapp/backend

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Run server
uvicorn main:app --reload --port 8000
```

API available at: http://localhost:8000
Docs at: http://localhost:8000/docs

### 2. Frontend

```bash
cd webapp/frontend

# Install dependencies
npm install

# Run development server
npm run dev
```

App available at: http://localhost:5173

## Features

### Dashboard
- Real-time health indicators
- System metrics overview
- Recent alerts and tickets
- Interactive charts

### Models
- ML models registry
- Model status and metadata
- Category filtering
- Search functionality

### Monitoring
- Live sensor data
- Temperature, pressure, vibration gauges
- Time-series charts
- Alert thresholds visualization

### Ticketing
- Smart ticket generation (ML + RAG)
- Ticket list with filters
- Status workflow (Open → In Progress → Resolved → Closed)
- Priority management

### Drift Control
- Statistical Process Control charts
- Control limits (UCL, LCL, Mean)
- Out-of-control detection
- Drift severity alerts

### Analytics
- Model performance metrics
- Ticket trends
- Health radar charts
- Performance tables

### Settings
- Alert thresholds configuration
- Notification preferences
- Model parameters
- Data retention settings

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | System health check |
| `/api/models` | GET | List all ML models |
| `/api/data/datasets` | GET | Available datasets |
| `/api/data/{name}` | GET | Load dataset |
| `/api/predict/anomaly` | POST | Anomaly detection |
| `/api/drift/metrics` | GET | Drift analysis |
| `/api/tickets` | GET/POST | Ticket management |
| `/api/tickets/{id}/status` | PATCH | Update ticket status |
| `/api/health-index` | GET | System health index |

## Environment Variables

### Frontend (.env)
```
VITE_API_URL=http://localhost:8000/api
```

### Backend (.env)
```
MODEL_PATH=../../ML_MODELS
DATA_PATH=../../LAST_DATA
```

## Development

### Build for Production

```bash
# Frontend
cd webapp/frontend
npm run build

# The build output will be in dist/
```

### Linting

```bash
npm run lint
```

## Integration with Existing System

This web application integrates with the existing TG1 Digital Twin components:

- **ML Models**: Loads trained XGBoost, Keras, and scikit-learn models
- **PD Models**: Partial discharge detection models
- **Smart Ticketing**: ML + RAG + LLM ticket generation
- **Data**: Connects to LAST_DATA CSV files

## License

STEG Internal Use Only
