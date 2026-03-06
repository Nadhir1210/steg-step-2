import axios from 'axios'

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000/api'

const api = axios.create({
  baseURL: API_BASE,
  headers: {
    'Content-Type': 'application/json',
  },
})

// Health & System
export const getHealth = () => api.get('/health')
export const getModels = () => api.get('/models')

// Data
export const getDatasets = () => api.get('/data/datasets')
export const loadDataset = (name: string, limit = 1000, offset = 0) => 
  api.get(`/data/${name}`, { params: { limit, offset } })
export const getDataset = loadDataset
export const getDatasetStats = (name: string) => api.get(`/data/${name}/stats`)

// Predictions
export const detectAnomalies = (data: any[]) => 
  api.post('/predict/anomaly', { data, model: 'isolation_forest' })

// Drift Control
export const getDriftMetrics = (dataset?: string) => 
  api.get('/drift/metrics', { params: { dataset } })

// Ticketing
export const getTickets = (params?: { status?: string; priority?: string; module?: string }) =>
  api.get('/tickets', { params })
export const getTicketStats = () => api.get('/tickets/stats')
export const createTicket = (data: {
  module: string
  severity_score: number
  metrics: Record<string, number>
  ml_confidence?: number
}) => api.post('/tickets', data)
export const getTicket = (id: string) => api.get(`/tickets/${id}`)
export const updateTicketStatus = (id: string, status: string) =>
  api.patch(`/tickets/${id}/status`, null, { params: { new_status: status } })

// Health Index
export const getHealthIndex = () => api.get('/health-index')

export default api
