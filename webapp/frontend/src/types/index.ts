export interface Model {
  name: string
  type: string
  category: 'ml' | 'pd' | 'tg1' | 'scaler'
  loaded: boolean
  path?: string
  status?: 'loaded' | 'error' | 'loading'
}

export interface Dataset {
  name: string
  filename: string
  columns: number
  sample_columns: string[]
}

export interface Ticket {
  ticket_id: string
  module: string
  priority: 'CRITICAL' | 'HIGH' | 'MEDIUM' | 'LOW'
  severity_score: number
  status: 'OPEN' | 'IN_PROGRESS' | 'RESOLVED' | 'CLOSED'
  description?: string
  recommendation?: string
  root_cause?: string
  estimated_rul?: string
  timestamp: string
  anomaly_type?: string
  assigned_service?: string
  ml_confidence?: number
}

export interface TicketStats {
  total: number
  open: number
  by_priority: Record<string, number>
  by_module: Record<string, number>
  avg_severity?: number
}

export interface DriftMetric {
  metric_name: string
  current_value: number
  mean: number
  std: number
  ucl: number
  lcl: number
  is_out_of_control: boolean
}

export interface HealthIndex {
  health_index: number
  status: 'HEALTHY' | 'WARNING' | 'CRITICAL'
  issues: string[]
  timestamp: string
}

export interface DataStats {
  dataset: string
  rows: number
  columns: number
  numeric_columns: number
  statistics: Record<string, {
    mean: number | null
    std: number | null
    min: number | null
    max: number | null
    median: number | null
  }>
}
