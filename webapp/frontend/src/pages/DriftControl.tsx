import { useQuery } from '@tanstack/react-query'
import { motion } from 'framer-motion'
import { useState } from 'react'
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ReferenceLine,
  ScatterChart,
  Scatter,
} from 'recharts'
import {
  FiTrendingUp,
  FiTrendingDown,
  FiAlertTriangle,
  FiCheckCircle,
  FiRefreshCw,
  FiSettings,
  FiActivity,
} from 'react-icons/fi'
import { getDriftMetrics } from '../services/api'
import type { DriftMetric } from '../types'

// Generate control chart data
function generateControlChartData(metric: string, points: number = 30) {
  const mean = 100 + Math.random() * 20
  const sigma = 5 + Math.random() * 3
  
  return Array.from({ length: points }, (_, i) => {
    const value = mean + (Math.random() - 0.5) * sigma * 4
    return {
      sample: i + 1,
      value,
      ucl: mean + 3 * sigma,
      lcl: mean - 3 * sigma,
      mean,
      uwl: mean + 2 * sigma,
      lwl: mean - 2 * sigma,
      outOfControl: value > mean + 3 * sigma || value < mean - 3 * sigma,
    }
  })
}

function ControlChart({ 
  title, 
  data, 
  metric 
}: { 
  title: string
  data: any[]
  metric: string 
}) {
  const outOfControlPoints = data.filter(d => d.outOfControl).length
  const status = outOfControlPoints === 0 ? 'stable' : outOfControlPoints <= 2 ? 'warning' : 'critical'
  
  const colors = {
    stable: 'text-green-600',
    warning: 'text-orange-600',
    critical: 'text-red-600',
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="card p-6"
    >
      <div className="flex items-center justify-between mb-4">
        <div>
          <h3 className="text-lg font-semibold text-slate-800">{title}</h3>
          <p className="text-sm text-slate-500">Control Chart (X-bar)</p>
        </div>
        <div className={`flex items-center gap-2 ${colors[status]}`}>
          {status === 'stable' ? <FiCheckCircle /> : <FiAlertTriangle />}
          <span className="text-sm font-medium capitalize">{status}</span>
        </div>
      </div>
      
      <ResponsiveContainer width="100%" height={250}>
        <LineChart data={data}>
          <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
          <XAxis dataKey="sample" tick={{ fontSize: 11 }} />
          <YAxis tick={{ fontSize: 11 }} domain={['auto', 'auto']} />
          <Tooltip 
            content={({ active, payload }) => {
              if (active && payload?.[0]) {
                const d = payload[0].payload
                return (
                  <div className="bg-white p-3 rounded-lg shadow-lg border">
                    <p className="font-medium">Sample {d.sample}</p>
                    <p>Value: {d.value.toFixed(2)}</p>
                    <p className="text-xs text-slate-500">UCL: {d.ucl.toFixed(2)}</p>
                    <p className="text-xs text-slate-500">LCL: {d.lcl.toFixed(2)}</p>
                  </div>
                )
              }
              return null
            }}
          />
          {/* Control Limits */}
          <ReferenceLine y={data[0]?.ucl} stroke="#ef4444" strokeDasharray="5 5" label="UCL" />
          <ReferenceLine y={data[0]?.lcl} stroke="#ef4444" strokeDasharray="5 5" label="LCL" />
          <ReferenceLine y={data[0]?.mean} stroke="#3b82f6" strokeDasharray="3 3" label="Mean" />
          <ReferenceLine y={data[0]?.uwl} stroke="#f59e0b" strokeDasharray="2 2" />
          <ReferenceLine y={data[0]?.lwl} stroke="#f59e0b" strokeDasharray="2 2" />
          
          {/* Data line */}
          <Line
            type="monotone"
            dataKey="value"
            stroke="#10b981"
            strokeWidth={2}
            dot={({ cx, cy, payload }) => (
              <circle
                cx={cx}
                cy={cy}
                r={payload.outOfControl ? 6 : 3}
                fill={payload.outOfControl ? '#ef4444' : '#10b981'}
                stroke={payload.outOfControl ? '#ef4444' : '#10b981'}
              />
            )}
          />
        </LineChart>
      </ResponsiveContainer>
      
      {/* Stats */}
      <div className="grid grid-cols-4 gap-4 mt-4 pt-4 border-t border-slate-100">
        <div className="text-center">
          <div className="text-lg font-bold text-slate-800">{data[0]?.mean.toFixed(1)}</div>
          <div className="text-xs text-slate-500">Mean</div>
        </div>
        <div className="text-center">
          <div className="text-lg font-bold text-slate-800">{(data[0]?.ucl - data[0]?.mean).toFixed(1)}</div>
          <div className="text-xs text-slate-500">3σ</div>
        </div>
        <div className="text-center">
          <div className="text-lg font-bold text-red-600">{outOfControlPoints}</div>
          <div className="text-xs text-slate-500">Out of Control</div>
        </div>
        <div className="text-center">
          <div className="text-lg font-bold text-green-600">{((1 - outOfControlPoints/data.length) * 100).toFixed(1)}%</div>
          <div className="text-xs text-slate-500">Cpk</div>
        </div>
      </div>
    </motion.div>
  )
}

function DriftAlert({ metric }: { metric: DriftMetric }) {
  return (
    <motion.div
      initial={{ opacity: 0, x: -20 }}
      animate={{ opacity: 1, x: 0 }}
      className={`p-4 rounded-xl border-l-4 ${
        metric.severity === 'CRITICAL' ? 'bg-red-50 border-red-500' :
        metric.severity === 'HIGH' ? 'bg-orange-50 border-orange-500' :
        metric.severity === 'MEDIUM' ? 'bg-yellow-50 border-yellow-500' :
        'bg-green-50 border-green-500'
      }`}
    >
      <div className="flex items-center justify-between">
        <div>
          <h4 className="font-medium text-slate-800">{metric.feature}</h4>
          <p className="text-sm text-slate-600">
            Drift Score: {(metric.drift_score * 100).toFixed(1)}%
          </p>
        </div>
        <div className="text-right">
          <span className={`px-2 py-1 rounded-full text-xs font-medium ${
            metric.severity === 'CRITICAL' ? 'bg-red-100 text-red-700' :
            metric.severity === 'HIGH' ? 'bg-orange-100 text-orange-700' :
            metric.severity === 'MEDIUM' ? 'bg-yellow-100 text-yellow-700' :
            'bg-green-100 text-green-700'
          }`}>
            {metric.severity}
          </span>
          <p className="text-xs text-slate-500 mt-1">{metric.timestamp}</p>
        </div>
      </div>
    </motion.div>
  )
}

export default function DriftControl() {
  const [selectedFeature, setSelectedFeature] = useState('temperature')
  
  // Generate control chart data for different features
  const [chartData] = useState(() => ({
    temperature: generateControlChartData('temperature', 30),
    vibration: generateControlChartData('vibration', 30),
    pressure: generateControlChartData('pressure', 30),
    current: generateControlChartData('current', 30),
  }))

  const { data: driftMetrics, isLoading, refetch } = useQuery({
    queryKey: ['driftMetrics'],
    queryFn: () => getDriftMetrics().then(r => r.data),
  })

  // Summary stats
  const criticalCount = driftMetrics?.metrics?.filter((m: DriftMetric) => m.severity === 'CRITICAL').length || 0
  const highCount = driftMetrics?.metrics?.filter((m: DriftMetric) => m.severity === 'HIGH').length || 0
  const stableCount = driftMetrics?.metrics?.filter((m: DriftMetric) => m.severity === 'LOW').length || 0

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-slate-800">Drift Control Dashboard</h1>
          <p className="text-slate-500">Statistical Process Control & Drift Detection</p>
        </div>
        <div className="flex gap-3">
          <button className="px-4 py-2 border border-slate-200 rounded-lg hover:bg-slate-50 transition-colors flex items-center gap-2">
            <FiSettings size={18} />
            Configure
          </button>
          <button
            onClick={() => refetch()}
            className="px-4 py-2 bg-steg-accent text-white rounded-lg hover:bg-steg-light transition-colors flex items-center gap-2"
          >
            <FiRefreshCw size={18} />
            Refresh
          </button>
        </div>
      </div>

      {/* Summary Cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="card p-6"
        >
          <div className="flex items-center gap-3">
            <div className="w-12 h-12 rounded-xl bg-green-100 flex items-center justify-center">
              <FiCheckCircle className="text-green-600" size={24} />
            </div>
            <div>
              <div className="text-2xl font-bold text-green-600">{stableCount}</div>
              <div className="text-sm text-slate-500">Stable Features</div>
            </div>
          </div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="card p-6"
        >
          <div className="flex items-center gap-3">
            <div className="w-12 h-12 rounded-xl bg-orange-100 flex items-center justify-center">
              <FiTrendingUp className="text-orange-600" size={24} />
            </div>
            <div>
              <div className="text-2xl font-bold text-orange-600">{highCount}</div>
              <div className="text-sm text-slate-500">High Drift</div>
            </div>
          </div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="card p-6"
        >
          <div className="flex items-center gap-3">
            <div className="w-12 h-12 rounded-xl bg-red-100 flex items-center justify-center">
              <FiAlertTriangle className="text-red-600" size={24} />
            </div>
            <div>
              <div className="text-2xl font-bold text-red-600">{criticalCount}</div>
              <div className="text-sm text-slate-500">Critical Drift</div>
            </div>
          </div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
          className="card p-6"
        >
          <div className="flex items-center gap-3">
            <div className="w-12 h-12 rounded-xl bg-blue-100 flex items-center justify-center">
              <FiActivity className="text-blue-600" size={24} />
            </div>
            <div>
              <div className="text-2xl font-bold text-blue-600">
                {driftMetrics?.overall_drift?.toFixed(1) || '--'}%
              </div>
              <div className="text-sm text-slate-500">Overall Drift</div>
            </div>
          </div>
        </motion.div>
      </div>

      {/* Control Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <ControlChart 
          title="Temperature Control" 
          data={chartData.temperature} 
          metric="temperature" 
        />
        <ControlChart 
          title="Vibration Control" 
          data={chartData.vibration} 
          metric="vibration" 
        />
        <ControlChart 
          title="Pressure Control" 
          data={chartData.pressure} 
          metric="pressure" 
        />
        <ControlChart 
          title="Current Control" 
          data={chartData.current} 
          metric="current" 
        />
      </div>

      {/* Drift Alerts */}
      <div className="card p-6">
        <h3 className="text-lg font-semibold text-slate-800 mb-4">Drift Alerts</h3>
        {isLoading ? (
          <div className="text-center py-8">
            <div className="animate-spin w-6 h-6 border-2 border-steg-accent border-t-transparent rounded-full mx-auto" />
          </div>
        ) : driftMetrics?.metrics?.length > 0 ? (
          <div className="space-y-3">
            {driftMetrics.metrics.slice(0, 5).map((metric: DriftMetric, i: number) => (
              <DriftAlert key={i} metric={metric} />
            ))}
          </div>
        ) : (
          <div className="text-center py-8 text-slate-500">
            <FiCheckCircle size={32} className="mx-auto mb-2 text-green-500" />
            <p>No drift alerts - all features are stable</p>
          </div>
        )}
      </div>
    </div>
  )
}
