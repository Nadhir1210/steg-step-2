import { useQuery } from '@tanstack/react-query'
import { motion } from 'framer-motion'
import { useState } from 'react'
import {
  BarChart,
  Bar,
  LineChart,
  Line,
  PieChart,
  Pie,
  Cell,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  RadarChart,
  Radar,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
} from 'recharts'
import {
  FiTrendingUp,
  FiClock,
  FiAlertCircle,
  FiCheckCircle,
  FiDownload,
  FiCalendar,
} from 'react-icons/fi'
import { getTicketStats, getHealthIndex, getDriftMetrics } from '../services/api'

const COLORS = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6']

// Sample analytics data
const modelPerformance = [
  { name: 'XGBoost Temp', accuracy: 94.2, precision: 92.1, recall: 95.8, f1: 93.9 },
  { name: 'RF Cooling', accuracy: 91.5, precision: 89.3, recall: 93.2, f1: 91.2 },
  { name: 'LSTM Seq', accuracy: 88.7, precision: 86.4, recall: 90.1, f1: 88.2 },
  { name: 'AE Anomaly', accuracy: 85.3, precision: 83.2, recall: 87.4, f1: 85.3 },
  { name: 'KNN PD', accuracy: 82.1, precision: 80.5, recall: 83.7, f1: 82.1 },
]

const monthlyTickets = [
  { month: 'Jan', tickets: 45, resolved: 42 },
  { month: 'Feb', tickets: 52, resolved: 48 },
  { month: 'Mar', tickets: 38, resolved: 35 },
  { month: 'Apr', tickets: 61, resolved: 55 },
  { month: 'May', tickets: 49, resolved: 47 },
  { month: 'Jun', tickets: 55, resolved: 52 },
]

const anomalyTypes = [
  { name: 'Thermal', value: 35 },
  { name: 'Vibration', value: 25 },
  { name: 'Electrical', value: 20 },
  { name: 'Pressure', value: 12 },
  { name: 'Other', value: 8 },
]

const healthRadar = [
  { subject: 'Thermal', A: 85 },
  { subject: 'Cooling', A: 92 },
  { subject: 'Electrical', A: 78 },
  { subject: 'Mechanical', A: 88 },
  { subject: 'Control', A: 95 },
  { subject: 'Safety', A: 97 },
]

function StatCard({ 
  title, 
  value, 
  change, 
  icon: Icon,
  color 
}: { 
  title: string
  value: string
  change?: string
  icon: React.ElementType
  color: string
}) {
  const isPositive = change?.startsWith('+')
  
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="card p-6"
    >
      <div className="flex items-center justify-between">
        <div className={`w-12 h-12 rounded-xl bg-${color}-100 flex items-center justify-center`}>
          <Icon className={`text-${color}-600`} size={24} />
        </div>
        {change && (
          <span className={`text-sm font-medium ${isPositive ? 'text-green-600' : 'text-red-600'}`}>
            {change}
          </span>
        )}
      </div>
      <div className="mt-4">
        <div className="text-2xl font-bold text-slate-800">{value}</div>
        <div className="text-sm text-slate-500">{title}</div>
      </div>
    </motion.div>
  )
}

export default function Analytics() {
  const [dateRange, setDateRange] = useState('30d')

  const { data: ticketStats } = useQuery({
    queryKey: ['ticketStats'],
    queryFn: () => getTicketStats().then(r => r.data),
  })

  const { data: healthData } = useQuery({
    queryKey: ['healthIndex'],
    queryFn: () => getHealthIndex().then(r => r.data),
  })

  const { data: driftData } = useQuery({
    queryKey: ['driftMetrics'],
    queryFn: () => getDriftMetrics().then(r => r.data),
  })

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-slate-800">Analytics & Reports</h1>
          <p className="text-slate-500">Performance insights and statistics</p>
        </div>
        <div className="flex gap-3">
          <select
            value={dateRange}
            onChange={(e) => setDateRange(e.target.value)}
            className="px-4 py-2 border border-slate-200 rounded-lg"
          >
            <option value="7d">Last 7 days</option>
            <option value="30d">Last 30 days</option>
            <option value="90d">Last 90 days</option>
            <option value="1y">Last year</option>
          </select>
          <button className="px-4 py-2 bg-steg-accent text-white rounded-lg hover:bg-steg-light transition-colors flex items-center gap-2">
            <FiDownload size={18} />
            Export
          </button>
        </div>
      </div>

      {/* KPI Cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <StatCard
          title="System Uptime"
          value="99.7%"
          change="+0.2%"
          icon={FiTrendingUp}
          color="green"
        />
        <StatCard
          title="Avg Response Time"
          value="2.3h"
          change="-15%"
          icon={FiClock}
          color="blue"
        />
        <StatCard
          title="Total Anomalies"
          value={(ticketStats?.total || 127).toString()}
          change="+8"
          icon={FiAlertCircle}
          color="orange"
        />
        <StatCard
          title="Resolution Rate"
          value="94.2%"
          change="+3.1%"
          icon={FiCheckCircle}
          color="purple"
        />
      </div>

      {/* Charts Row 1 */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Model Performance */}
        <div className="card p-6">
          <h3 className="text-lg font-semibold text-slate-800 mb-4">Model Performance</h3>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={modelPerformance} layout="vertical">
              <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
              <XAxis type="number" domain={[70, 100]} tick={{ fontSize: 11 }} />
              <YAxis type="category" dataKey="name" tick={{ fontSize: 11 }} width={100} />
              <Tooltip />
              <Legend />
              <Bar dataKey="accuracy" fill="#3b82f6" name="Accuracy" radius={[0, 4, 4, 0]} />
              <Bar dataKey="f1" fill="#10b981" name="F1 Score" radius={[0, 4, 4, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Health Radar */}
        <div className="card p-6">
          <h3 className="text-lg font-semibold text-slate-800 mb-4">System Health Radar</h3>
          <ResponsiveContainer width="100%" height={300}>
            <RadarChart data={healthRadar}>
              <PolarGrid />
              <PolarAngleAxis dataKey="subject" tick={{ fontSize: 11 }} />
              <PolarRadiusAxis angle={30} domain={[0, 100]} tick={{ fontSize: 10 }} />
              <Radar
                name="Health Score"
                dataKey="A"
                stroke="#3b82f6"
                fill="#3b82f6"
                fillOpacity={0.5}
              />
              <Tooltip />
            </RadarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Charts Row 2 */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Monthly Tickets */}
        <div className="card p-6 lg:col-span-2">
          <h3 className="text-lg font-semibold text-slate-800 mb-4">Monthly Ticket Trends</h3>
          <ResponsiveContainer width="100%" height={250}>
            <LineChart data={monthlyTickets}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
              <XAxis dataKey="month" tick={{ fontSize: 11 }} />
              <YAxis tick={{ fontSize: 11 }} />
              <Tooltip />
              <Legend />
              <Line
                type="monotone"
                dataKey="tickets"
                stroke="#f59e0b"
                strokeWidth={2}
                dot={{ fill: '#f59e0b' }}
                name="Total Tickets"
              />
              <Line
                type="monotone"
                dataKey="resolved"
                stroke="#10b981"
                strokeWidth={2}
                dot={{ fill: '#10b981' }}
                name="Resolved"
              />
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* Anomaly Distribution */}
        <div className="card p-6">
          <h3 className="text-lg font-semibold text-slate-800 mb-4">Anomaly Distribution</h3>
          <ResponsiveContainer width="100%" height={250}>
            <PieChart>
              <Pie
                data={anomalyTypes}
                cx="50%"
                cy="50%"
                innerRadius={40}
                outerRadius={80}
                paddingAngle={5}
                dataKey="value"
                label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                labelLine={false}
              >
                {anomalyTypes.map((_, index) => (
                  <Cell key={index} fill={COLORS[index % COLORS.length]} />
                ))}
              </Pie>
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Performance Summary Table */}
      <div className="card p-6">
        <h3 className="text-lg font-semibold text-slate-800 mb-4">Model Performance Summary</h3>
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="border-b border-slate-200">
                <th className="text-left py-3 px-4 text-sm font-medium text-slate-600">Model</th>
                <th className="text-center py-3 px-4 text-sm font-medium text-slate-600">Accuracy</th>
                <th className="text-center py-3 px-4 text-sm font-medium text-slate-600">Precision</th>
                <th className="text-center py-3 px-4 text-sm font-medium text-slate-600">Recall</th>
                <th className="text-center py-3 px-4 text-sm font-medium text-slate-600">F1 Score</th>
                <th className="text-center py-3 px-4 text-sm font-medium text-slate-600">Status</th>
              </tr>
            </thead>
            <tbody>
              {modelPerformance.map((model, i) => (
                <tr key={i} className="border-b border-slate-100 hover:bg-slate-50">
                  <td className="py-3 px-4 font-medium text-slate-800">{model.name}</td>
                  <td className="py-3 px-4 text-center">{model.accuracy}%</td>
                  <td className="py-3 px-4 text-center">{model.precision}%</td>
                  <td className="py-3 px-4 text-center">{model.recall}%</td>
                  <td className="py-3 px-4 text-center">{model.f1}%</td>
                  <td className="py-3 px-4 text-center">
                    <span className={`px-2 py-1 rounded-full text-xs font-medium ${
                      model.accuracy >= 90 ? 'bg-green-100 text-green-700' :
                      model.accuracy >= 85 ? 'bg-yellow-100 text-yellow-700' :
                      'bg-red-100 text-red-700'
                    }`}>
                      {model.accuracy >= 90 ? 'Excellent' : model.accuracy >= 85 ? 'Good' : 'Needs Improvement'}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}
