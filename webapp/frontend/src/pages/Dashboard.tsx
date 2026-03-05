import { useQuery } from '@tanstack/react-query'
import { motion } from 'framer-motion'
import {
  LineChart,
  Line,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
} from 'recharts'
import {
  FiActivity,
  FiThermometer,
  FiZap,
  FiAlertTriangle,
  FiCheckCircle,
  FiClock,
  FiTrendingUp,
} from 'react-icons/fi'
import { getHealthIndex, getTicketStats, getModels, getDriftMetrics } from '../services/api'

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

  // Mock data for charts
  const temperatureData = Array.from({ length: 24 }, (_, i) => ({
    time: `${i}:00`,
    value: 75 + Math.random() * 15,
  }))

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
              {healthIndex.issues.map((issue, i) => (
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
          <h3 className="text-lg font-semibold text-slate-800 mb-4">
            <FiThermometer className="inline mr-2" />
            Temperature Monitoring
          </h3>
          <ResponsiveContainer width="100%" height={200}>
            <AreaChart data={temperatureData}>
              <defs>
                <linearGradient id="tempGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#667eea" stopOpacity={0.3} />
                  <stop offset="95%" stopColor="#667eea" stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
              <XAxis dataKey="time" stroke="#94a3b8" fontSize={12} />
              <YAxis stroke="#94a3b8" fontSize={12} domain={[70, 100]} />
              <Tooltip />
              <Area
                type="monotone"
                dataKey="value"
                stroke="#667eea"
                strokeWidth={2}
                fill="url(#tempGradient)"
              />
            </AreaChart>
          </ResponsiveContainer>
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
    </div>
  )
}
