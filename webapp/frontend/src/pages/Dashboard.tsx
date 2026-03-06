import { useQuery } from '@tanstack/react-query'
import { motion } from 'framer-motion'
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
  Legend,
} from 'recharts'
import {
  FiActivity,
  FiThermometer,
  FiZap,
  FiAlertTriangle,
  FiCheckCircle,
  FiClock,
  FiTrendingUp,
  FiCpu,
  FiBarChart2,
} from 'react-icons/fi'
import { getHealthIndex, getTicketStats, getModels, getDriftMetrics, getAllModelsResults, getRealtimeMonitoring } from '../services/api'

const COLORS = ['#f5365c', '#fb6340', '#ffd600', '#2dce89']

function StatCard({ 
  title, 
  value, 
  subtitle, 
  icon: Icon, 
  color = 'blue',
  trend 
}: {
  title: string
  value: string | number
  subtitle?: string
  icon: any
  color?: 'blue' | 'green' | 'orange' | 'red' | 'purple'
  trend?: { value: number; label: string }
}) {
  const colorClasses = {
    blue: 'from-blue-500 to-blue-600',
    green: 'from-green-500 to-green-600',
    orange: 'from-orange-500 to-orange-600',
    red: 'from-red-500 to-red-600',
    purple: 'from-purple-500 to-purple-600',
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="card p-6"
    >
      <div className="flex items-start justify-between">
        <div>
          <p className="text-sm font-medium text-slate-500">{title}</p>
          <p className="text-3xl font-bold text-slate-800 mt-2">{value}</p>
          {subtitle && (
            <p className="text-sm text-slate-500 mt-1">{subtitle}</p>
          )}
          {trend && (
            <div className="flex items-center gap-1 mt-2">
              <FiTrendingUp className={trend.value >= 0 ? 'text-green-500' : 'text-red-500'} />
              <span className={`text-sm ${trend.value >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                {trend.value >= 0 ? '+' : ''}{trend.value}%
              </span>
              <span className="text-xs text-slate-400">{trend.label}</span>
            </div>
          )}
        </div>
        <div className={`w-12 h-12 rounded-xl bg-gradient-to-br ${colorClasses[color]} flex items-center justify-center`}>
          <Icon className="text-white" size={24} />
        </div>
      </div>
    </motion.div>
  )
}

function HealthGauge({ value, status }: { value: number; status: string }) {
  const statusColors = {
    HEALTHY: '#2dce89',
    WARNING: '#fb6340',
    CRITICAL: '#f5365c',
  }
  const color = statusColors[status as keyof typeof statusColors] || '#2dce89'

  return (
    <div className="relative w-48 h-48 mx-auto">
      <svg viewBox="0 0 100 100" className="transform -rotate-90">
        <circle
          cx="50"
          cy="50"
          r="45"
          fill="none"
          stroke="#e2e8f0"
          strokeWidth="10"
        />
        <circle
          cx="50"
          cy="50"
          r="45"
          fill="none"
          stroke={color}
          strokeWidth="10"
          strokeLinecap="round"
          strokeDasharray={`${value * 2.83} 283`}
          className="transition-all duration-1000"
        />
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <span className="text-4xl font-bold text-slate-800">{value}</span>
        <span className="text-sm text-slate-500">{status}</span>
      </div>
    </div>
  )
}

export default function Dashboard() {
  const { data: healthIndex } = useQuery({
    queryKey: ['healthIndex'],
    queryFn: () => getHealthIndex().then(r => r.data),
    refetchInterval: 60000,
  })

  const { data: ticketStats } = useQuery({
    queryKey: ['ticketStats'],
    queryFn: () => getTicketStats().then(r => r.data),
  })

  const { data: models } = useQuery({
    queryKey: ['models'],
    queryFn: () => getModels().then(r => r.data),
  })

  const { data: driftMetrics } = useQuery({
    queryKey: ['driftMetrics'],
    queryFn: () => getDriftMetrics().then(r => r.data),
  })

  const { data: modelResults } = useQuery({
    queryKey: ['modelResults'],
    queryFn: () => getAllModelsResults().then(r => r.data),
    refetchInterval: 120000,
  })

  const { data: realtimeMonitoring } = useQuery({
    queryKey: ['realtimeMonitoring'],
    queryFn: () => getRealtimeMonitoring('APM_Alternateur_ML', 40).then(r => r.data),
    refetchInterval: 30000,
  })

  // Use real monitoring data or fallback
  const comparisonData = realtimeMonitoring?.comparison_data || []
  const currentValues = realtimeMonitoring?.current_values || {}
  const modelPredictions = realtimeMonitoring?.model_predictions || {}

  const ticketPieData = ticketStats?.by_priority
    ? Object.entries(ticketStats.by_priority).map(([name, value]) => ({
        name,
        value: value as number,
      }))
    : []

  return (
    <div className="space-y-6">
      {/* Welcome Banner */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="bg-gradient-steg rounded-2xl p-8 text-white"
      >
        <h1 className="text-3xl font-bold mb-2">TG1 Digital Twin</h1>
        <p className="text-white/80 max-w-2xl">
          Système de monitoring intelligent pour la turbine à gaz TG1.
          Surveillance en temps réel, détection d'anomalies et maintenance prédictive.
        </p>
        <div className="flex gap-4 mt-6">
          <div className="bg-white/20 rounded-lg px-4 py-2">
            <span className="text-sm text-white/70">Modèles ML actifs</span>
            <p className="text-2xl font-bold">{models?.total || 0}</p>
          </div>
          <div className="bg-white/20 rounded-lg px-4 py-2">
            <span className="text-sm text-white/70">Tickets ouverts</span>
            <p className="text-2xl font-bold">{ticketStats?.open || 0}</p>
          </div>
          <div className="bg-white/20 rounded-lg px-4 py-2">
            <span className="text-sm text-white/70">Métriques surveillées</span>
            <p className="text-2xl font-bold">{driftMetrics?.metrics?.length || 0}</p>
          </div>
        </div>
      </motion.div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <StatCard
          title="Health Index"
          value={healthIndex?.health_index || 0}
          subtitle={healthIndex?.status || 'Loading...'}
          icon={FiActivity}
          color={healthIndex?.status === 'HEALTHY' ? 'green' : healthIndex?.status === 'WARNING' ? 'orange' : 'red'}
        />
        <StatCard
          title="Tickets Critiques"
          value={ticketStats?.by_priority?.CRITICAL || 0}
          subtitle="Intervention urgente"
          icon={FiAlertTriangle}
          color="red"
        />
        <StatCard
          title="Modèles Actifs"
          value={models?.total || 0}
          subtitle="ML + PD + TG1"
          icon={FiZap}
          color="purple"
        />
        <StatCard
          title="Temps de réponse"
          value="<50ms"
          subtitle="API Backend"
          icon={FiClock}
          color="blue"
        />
      </div>

      {/* Charts Row */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Health Gauge */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="card p-6"
        >
          <h3 className="text-lg font-semibold text-slate-800 mb-4">System Health</h3>
          <HealthGauge 
            value={healthIndex?.health_index || 0} 
            status={healthIndex?.status || 'LOADING'} 
          />
          {healthIndex?.issues && healthIndex.issues.length > 0 && (
            <div className="mt-4 space-y-2">
              {healthIndex.issues.map((issue: string, i: number) => (
                <div key={i} className="flex items-center gap-2 text-sm text-orange-600">
                  <FiAlertTriangle size={14} />
                  <span>{issue}</span>
                </div>
              ))}
            </div>
          )}
        </motion.div>

        {/* Temperature Chart */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="card p-6 lg:col-span-2"
        >
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold text-slate-800">
              <FiThermometer className="inline mr-2" />
              Real Values vs Model Predictions
            </h3>
            <div className="flex items-center gap-4 text-xs">
              <span className="flex items-center gap-1"><span className="w-3 h-3 rounded-full bg-blue-500" />Real</span>
              <span className="flex items-center gap-1"><span className="w-3 h-3 rounded-full bg-green-500" />Predicted</span>
            </div>
          </div>
          {comparisonData.length > 0 ? (
            <ResponsiveContainer width="100%" height={220}>
              <LineChart data={comparisonData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                <XAxis dataKey="index" stroke="#94a3b8" fontSize={11} tickFormatter={(v) => `T${v}`} />
                <YAxis stroke="#94a3b8" fontSize={11} />
                <Tooltip 
                  contentStyle={{ backgroundColor: '#fff', borderRadius: '8px', border: '1px solid #e2e8f0' }}
                  formatter={(value: number, name: string) => [value.toFixed(2), name === 'real' ? 'Real Value' : name === 'predicted' ? 'Predicted' : 'Error']}
                />
                <Legend />
                <Line type="monotone" dataKey="real" stroke="#3b82f6" strokeWidth={2} dot={false} name="Real Value" />
                <Line type="monotone" dataKey="predicted" stroke="#10b981" strokeWidth={2} dot={false} name="Predicted" />
              </LineChart>
            </ResponsiveContainer>
          ) : (
            <div className="flex items-center justify-center h-[220px] text-slate-400">
              <div className="text-center">
                <div className="animate-spin w-8 h-8 border-2 border-steg-accent border-t-transparent rounded-full mx-auto mb-2" />
                Loading monitoring data...
              </div>
            </div>
          )}
          
          {/* Current Sensor Values */}
          {Object.keys(currentValues).length > 0 && (
            <div className="mt-4 pt-4 border-t border-slate-100">
              <p className="text-sm font-medium text-slate-600 mb-2">Current Sensor Readings</p>
              <div className="grid grid-cols-3 md:grid-cols-4 lg:grid-cols-6 gap-2">
                {Object.entries(currentValues).slice(0, 6).map(([key, value]) => (
                  <div key={key} className="bg-slate-50 rounded-lg p-2 text-center">
                    <div className="text-xs text-slate-500 truncate" title={key}>{key.slice(0, 12)}</div>
                    <div className="text-sm font-semibold text-slate-800">
                      {typeof value === 'number' ? value.toFixed(1) : 'N/A'}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
          
          {/* Model Predictions Status */}
          {modelPredictions && Object.keys(modelPredictions).length > 0 && (
            <div className="mt-4 pt-4 border-t border-slate-100">
              <p className="text-sm font-medium text-slate-600 mb-2">Active Model Predictions</p>
              <div className="flex gap-4">
                {Object.entries(modelPredictions).map(([key, pred]: [string, any]) => (
                  <div key={key} className={`flex items-center gap-2 px-3 py-1 rounded-full text-xs ${pred.status === 'active' ? 'bg-green-100 text-green-700' : 'bg-slate-100 text-slate-500'}`}>
                    <span className={`w-2 h-2 rounded-full ${pred.status === 'active' ? 'bg-green-500' : 'bg-slate-400'}`} />
                    {pred.model}: {pred.last_prediction?.toFixed(2) || 'N/A'}
                  </div>
                ))}
              </div>
            </div>
          )}
        </motion.div>
      </div>

      {/* Bottom Row */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Tickets by Priority */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="card p-6"
        >
          <h3 className="text-lg font-semibold text-slate-800 mb-4">Tickets by Priority</h3>
          <div className="flex items-center">
            <ResponsiveContainer width="50%" height={200}>
              <PieChart>
                <Pie
                  data={ticketPieData}
                  cx="50%"
                  cy="50%"
                  innerRadius={50}
                  outerRadius={80}
                  dataKey="value"
                >
                  {ticketPieData.map((_, index) => (
                    <Cell key={index} fill={COLORS[index % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
            <div className="space-y-2">
              {['CRITICAL', 'HIGH', 'MEDIUM', 'LOW'].map((priority, i) => (
                <div key={priority} className="flex items-center gap-2">
                  <div 
                    className="w-3 h-3 rounded-full" 
                    style={{ backgroundColor: COLORS[i] }}
                  />
                  <span className="text-sm text-slate-600">{priority}</span>
                  <span className="text-sm font-semibold text-slate-800">
                    {ticketStats?.by_priority?.[priority] || 0}
                  </span>
                </div>
              ))}
            </div>
          </div>
        </motion.div>

        {/* Drift Alerts */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="card p-6"
        >
          <h3 className="text-lg font-semibold text-slate-800 mb-4">Drift Alerts</h3>
          <div className="space-y-3">
            {driftMetrics?.metrics?.slice(0, 5).map((metric: any) => (
              <div 
                key={metric.metric_name} 
                className={`flex items-center justify-between p-3 rounded-lg ${
                  metric.is_out_of_control ? 'bg-red-50' : 'bg-green-50'
                }`}
              >
                <div className="flex items-center gap-3">
                  {metric.is_out_of_control ? (
                    <FiAlertTriangle className="text-red-500" />
                  ) : (
                    <FiCheckCircle className="text-green-500" />
                  )}
                  <span className="text-sm font-medium text-slate-700">
                    {metric.metric_name}
                  </span>
                </div>
                <span className={`text-sm font-semibold ${
                  metric.is_out_of_control ? 'text-red-600' : 'text-green-600'
                }`}>
                  {metric.current_value?.toFixed(2)}
                </span>
              </div>
            ))}
            {(!driftMetrics?.metrics || driftMetrics.metrics.length === 0) && (
              <p className="text-sm text-slate-500 text-center py-8">
                Loading drift metrics...
              </p>
            )}
          </div>
        </motion.div>
      </div>

      {/* Model Results Section */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        className="card p-6"
      >
        <div className="flex items-center justify-between mb-6">
          <h3 className="text-lg font-semibold text-slate-800">
            <FiCpu className="inline mr-2" />
            Model Predictions Overview
          </h3>
          <span className="text-sm text-slate-500">{modelResults?.results?.length || 0} models analyzed</span>
        </div>
        
        {modelResults?.results && modelResults.results.length > 0 ? (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
            {modelResults.results.map((result: any) => (
              <motion.div
                key={result.name}
                whileHover={{ y: -2 }}
                className={`p-4 rounded-xl border-2 transition-all ${
                  result.status === 'success' 
                    ? 'border-green-200 bg-gradient-to-br from-green-50 to-emerald-50'
                    : 'border-red-200 bg-gradient-to-br from-red-50 to-orange-50'
                }`}
              >
                <div className="flex items-start justify-between mb-3">
                  <div className="flex items-center gap-2">
                    <div className={`w-2 h-2 rounded-full ${result.status === 'success' ? 'bg-green-500' : 'bg-red-500'}`} />
                    <span className="text-xs font-medium uppercase text-slate-500">{result.category}</span>
                  </div>
                  {result.anomaly_ratio !== undefined && result.anomaly_ratio > 0 && (
                    <span className="text-xs px-2 py-0.5 rounded-full bg-red-100 text-red-700">
                      {(result.anomaly_ratio * 100).toFixed(1)}% anomaly
                    </span>
                  )}
                </div>
                <h4 className="font-semibold text-slate-800 text-sm truncate mb-2" title={result.name}>
                  {result.name.replace(/_/g, ' ').slice(0, 25)}{result.name.length > 25 ? '...' : ''}
                </h4>
                <div className="flex items-center justify-between text-xs">
                  <div className="flex items-center gap-1 text-slate-500">
                    <FiBarChart2 size={12} />
                    <span>{result.predictions_count} predictions</span>
                  </div>
                  {result.status === 'success' ? (
                    <FiCheckCircle className="text-green-500" size={16} />
                  ) : (
                    <FiAlertTriangle className="text-red-500" size={16} />
                  )}
                </div>
              </motion.div>
            ))}
          </div>
        ) : (
          <div className="text-center py-8">
            <div className="animate-pulse flex justify-center gap-4">
              {[1,2,3,4].map(i => (
                <div key={i} className="w-48 h-24 bg-slate-100 rounded-xl" />
              ))}
            </div>
            <p className="text-slate-500 mt-4">Loading model predictions...</p>
          </div>
        )}
      </motion.div>
    </div>
  )
}
