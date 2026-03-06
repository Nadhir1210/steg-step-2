import { useQuery } from '@tanstack/react-query'
import { motion } from 'framer-motion'
import { useState } from 'react'
import {
  LineChart,
  Line,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ReferenceLine,
} from 'recharts'
import {
  FiThermometer,
  FiZap,
  FiDroplet,
  FiWind,
  FiActivity,
  FiAlertTriangle,
  FiRefreshCw,
} from 'react-icons/fi'
import { getHealthIndex, getDatasets, loadDataset } from '../services/api'

// Generate simulated real-time data
function generateTimeSeriesData(points: number = 50) {
  const now = Date.now()
  return Array.from({ length: points }, (_, i) => ({
    time: new Date(now - (points - i) * 60000).toLocaleTimeString('fr-FR', { hour: '2-digit', minute: '2-digit' }),
    temperature: 70 + Math.random() * 30 + Math.sin(i / 5) * 10,
    pressure: 1.0 + Math.random() * 0.3,
    vibration: 0.5 + Math.random() * 1.5,
    current: 80 + Math.random() * 40,
    efficiency: 85 + Math.random() * 10,
  }))
}

function MetricGauge({ 
  label, 
  value, 
  unit, 
  min, 
  max, 
  warning, 
  critical,
  icon: Icon 
}: {
  label: string
  value: number
  unit: string
  min: number
  max: number
  warning: number
  critical: number
  icon: React.ElementType
}) {
  const percentage = ((value - min) / (max - min)) * 100
  const status = value >= critical ? 'critical' : value >= warning ? 'warning' : 'normal'
  const colors = {
    normal: 'text-green-500',
    warning: 'text-orange-500',
    critical: 'text-red-500',
  }
  const bgColors = {
    normal: 'from-green-500 to-green-400',
    warning: 'from-orange-500 to-orange-400',
    critical: 'from-red-500 to-red-400',
  }

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      className="card p-6"
    >
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-3">
          <div className={`w-10 h-10 rounded-xl bg-slate-100 flex items-center justify-center ${colors[status]}`}>
            <Icon size={20} />
          </div>
          <div>
            <h3 className="font-medium text-slate-800">{label}</h3>
            <p className="text-xs text-slate-400">{min} - {max} {unit}</p>
          </div>
        </div>
        {status !== 'normal' && (
          <FiAlertTriangle className={colors[status]} size={20} />
        )}
      </div>
      
      <div className="flex items-end gap-2 mb-3">
        <span className={`text-3xl font-bold ${colors[status]}`}>
          {value.toFixed(1)}
        </span>
        <span className="text-slate-400 mb-1">{unit}</span>
      </div>
      
      {/* Progress bar */}
      <div className="h-2 bg-slate-100 rounded-full overflow-hidden">
        <motion.div
          initial={{ width: 0 }}
          animate={{ width: `${Math.min(percentage, 100)}%` }}
          className={`h-full bg-gradient-to-r ${bgColors[status]}`}
        />
      </div>
      
      {/* Thresholds */}
      <div className="flex justify-between mt-2 text-xs text-slate-400">
        <span>Warning: {warning}</span>
        <span>Critical: {critical}</span>
      </div>
    </motion.div>
  )
}

export default function Monitoring() {
  const [timeSeriesData] = useState(() => generateTimeSeriesData(50))
  const [activeDataset, setActiveDataset] = useState('APM_Alternateur')

  const { data: healthData, isLoading: healthLoading, refetch } = useQuery({
    queryKey: ['healthIndex'],
    queryFn: () => getHealthIndex().then(r => r.data),
    refetchInterval: 30000,
  })

  const { data: datasets } = useQuery({
    queryKey: ['datasets'],
    queryFn: () => getDatasets().then(r => r.data),
  })

  // Current values (simulated)
  const currentValues = {
    temperature: 85 + Math.random() * 15,
    pressure: 1.2 + Math.random() * 0.2,
    vibration: 0.8 + Math.random() * 0.4,
    current: 95 + Math.random() * 20,
    efficiency: 88 + Math.random() * 5,
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-slate-800">Real-Time Monitoring</h1>
          <p className="text-slate-500">Live sensor data and system status</p>
        </div>
        <div className="flex gap-3">
          <select
            value={activeDataset}
            onChange={(e) => setActiveDataset(e.target.value)}
            className="px-4 py-2 border border-slate-200 rounded-lg"
          >
            {datasets?.datasets?.map((ds: any) => (
              <option key={ds.name || ds} value={ds.name || ds}>
                {(ds.name || ds).replace(/_/g, ' ')}
              </option>
            ))}
          </select>
          <button
            onClick={() => refetch()}
            className="px-4 py-2 bg-steg-accent text-white rounded-lg hover:bg-steg-light transition-colors flex items-center gap-2"
          >
            <FiRefreshCw size={18} />
            Refresh
          </button>
        </div>
      </div>

      {/* Health Overview */}
      <div className="card p-6">
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-lg font-semibold text-slate-800">System Health Overview</h2>
          <div className="flex items-center gap-2">
            <div className={`w-3 h-3 rounded-full ${
              healthData?.overall_health >= 80 ? 'bg-green-500 animate-pulse' :
              healthData?.overall_health >= 60 ? 'bg-orange-500 animate-pulse' :
              'bg-red-500 animate-pulse'
            }`} />
            <span className="text-sm text-slate-500">Live</span>
          </div>
        </div>
        
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="text-center">
            <div className="text-4xl font-bold text-slate-800">
              {healthData?.overall_health?.toFixed(1) || '--'}%
            </div>
            <div className="text-sm text-slate-500">Overall Health</div>
          </div>
          <div className="text-center">
            <div className="text-4xl font-bold text-blue-600">
              {healthData?.thermal?.toFixed(1) || '--'}%
            </div>
            <div className="text-sm text-slate-500">Thermal</div>
          </div>
          <div className="text-center">
            <div className="text-4xl font-bold text-purple-600">
              {healthData?.electrical?.toFixed(1) || '--'}%
            </div>
            <div className="text-sm text-slate-500">Electrical</div>
          </div>
          <div className="text-center">
            <div className="text-4xl font-bold text-green-600">
              {healthData?.cooling?.toFixed(1) || '--'}%
            </div>
            <div className="text-sm text-slate-500">Cooling</div>
          </div>
        </div>
      </div>

      {/* Current Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-5 gap-4">
        <MetricGauge
          label="Temperature"
          value={currentValues.temperature}
          unit="°C"
          min={50}
          max={120}
          warning={90}
          critical={100}
          icon={FiThermometer}
        />
        <MetricGauge
          label="Pressure"
          value={currentValues.pressure}
          unit="bar"
          min={0.8}
          max={2.0}
          warning={1.5}
          critical={1.8}
          icon={FiDroplet}
        />
        <MetricGauge
          label="Vibration"
          value={currentValues.vibration}
          unit="mm/s"
          min={0}
          max={3}
          warning={1.5}
          critical={2.5}
          icon={FiActivity}
        />
        <MetricGauge
          label="Current"
          value={currentValues.current}
          unit="A"
          min={0}
          max={150}
          warning={110}
          critical={130}
          icon={FiZap}
        />
        <MetricGauge
          label="Efficiency"
          value={currentValues.efficiency}
          unit="%"
          min={70}
          max={100}
          warning={82}
          critical={78}
          icon={FiWind}
        />
      </div>

      {/* Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Temperature Trend */}
        <div className="card p-6">
          <h3 className="text-lg font-semibold text-slate-800 mb-4">Temperature Trend</h3>
          <ResponsiveContainer width="100%" height={250}>
            <LineChart data={timeSeriesData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
              <XAxis dataKey="time" tick={{ fontSize: 11 }} />
              <YAxis domain={[60, 110]} tick={{ fontSize: 11 }} />
              <Tooltip />
              <ReferenceLine y={90} stroke="#f59e0b" strokeDasharray="5 5" label="Warning" />
              <ReferenceLine y={100} stroke="#ef4444" strokeDasharray="5 5" label="Critical" />
              <Line
                type="monotone"
                dataKey="temperature"
                stroke="#3b82f6"
                strokeWidth={2}
                dot={false}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* Vibration Analysis */}
        <div className="card p-6">
          <h3 className="text-lg font-semibold text-slate-800 mb-4">Vibration Analysis</h3>
          <ResponsiveContainer width="100%" height={250}>
            <AreaChart data={timeSeriesData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
              <XAxis dataKey="time" tick={{ fontSize: 11 }} />
              <YAxis domain={[0, 3]} tick={{ fontSize: 11 }} />
              <Tooltip />
              <ReferenceLine y={1.5} stroke="#f59e0b" strokeDasharray="5 5" />
              <Area
                type="monotone"
                dataKey="vibration"
                fill="#8b5cf6"
                fillOpacity={0.3}
                stroke="#8b5cf6"
                strokeWidth={2}
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>

        {/* Current Load */}
        <div className="card p-6">
          <h3 className="text-lg font-semibold text-slate-800 mb-4">Current Load</h3>
          <ResponsiveContainer width="100%" height={250}>
            <AreaChart data={timeSeriesData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
              <XAxis dataKey="time" tick={{ fontSize: 11 }} />
              <YAxis domain={[60, 140]} tick={{ fontSize: 11 }} />
              <Tooltip />
              <Area
                type="monotone"
                dataKey="current"
                fill="#10b981"
                fillOpacity={0.3}
                stroke="#10b981"
                strokeWidth={2}
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>

        {/* Efficiency */}
        <div className="card p-6">
          <h3 className="text-lg font-semibold text-slate-800 mb-4">System Efficiency</h3>
          <ResponsiveContainer width="100%" height={250}>
            <LineChart data={timeSeriesData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
              <XAxis dataKey="time" tick={{ fontSize: 11 }} />
              <YAxis domain={[80, 100]} tick={{ fontSize: 11 }} />
              <Tooltip />
              <Line
                type="monotone"
                dataKey="efficiency"
                stroke="#f59e0b"
                strokeWidth={2}
                dot={false}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  )
}
