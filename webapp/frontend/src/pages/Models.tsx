import { useQuery } from '@tanstack/react-query'
import { motion, AnimatePresence } from 'framer-motion'
import { useState } from 'react'
import {
  BarChart,
  Bar,
  PieChart,
  Pie,
  Cell,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  LineChart,
  Line,
} from 'recharts'
import {
  FiCpu,
  FiActivity,
  FiCheckCircle,
  FiXCircle,
  FiDatabase,
  FiLayers,
  FiInfo,
  FiRefreshCw,
  FiSearch,
  FiX,
  FiTrendingUp,
  FiAlertTriangle,
  FiMessageSquare,
  FiBarChart2,
} from 'react-icons/fi'
import { getModels, getModelResults } from '../services/api'
import type { Model } from '../types'

const COLORS = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899']

const categoryColors: Record<string, { bg: string; text: string; icon: string }> = {
  ml: { bg: 'bg-blue-100', text: 'text-blue-700', icon: '🤖' },
  pd: { bg: 'bg-purple-100', text: 'text-purple-700', icon: '⚡' },
  tg1: { bg: 'bg-green-100', text: 'text-green-700', icon: '🔧' },
  scaler: { bg: 'bg-orange-100', text: 'text-orange-700', icon: '📊' },
  default: { bg: 'bg-slate-100', text: 'text-slate-700', icon: '📦' },
}

interface ModelResultsModalProps {
  model: Model
  isOpen: boolean
  onClose: () => void
}

function ModelResultsModal({ model, isOpen, onClose }: ModelResultsModalProps) {
  const { data: results, isLoading, error } = useQuery({
    queryKey: ['modelResults', model.name],
    queryFn: () => getModelResults(model.name).then(r => r.data),
    enabled: isOpen,
  })

  if (!isOpen) return null

  const predictionDistData = results?.summary?.prediction_distribution
    ? Object.entries(results.summary.prediction_distribution).map(([key, value]) => ({
        name: key === '-1' ? 'Anomaly' : key === '1' ? 'Normal' : `Class ${key}`,
        value: value as number,
      }))
    : []

  const featureImportanceData = results?.feature_importance?.slice(0, 8) || []
  const anomalyScoresData = results?.anomaly_scores?.slice(0, 50).map((score: number, i: number) => ({
    index: i + 1,
    score: score,
  })) || []

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4"
        onClick={onClose}
      >
        <motion.div
          initial={{ opacity: 0, scale: 0.95, y: 20 }}
          animate={{ opacity: 1, scale: 1, y: 0 }}
          exit={{ opacity: 0, scale: 0.95, y: 20 }}
          className="bg-white rounded-2xl shadow-2xl w-full max-w-5xl max-h-[90vh] overflow-hidden"
          onClick={(e) => e.stopPropagation()}
        >
          <div className="bg-gradient-to-r from-steg-primary to-steg-accent p-6 text-white">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-4">
                <div className="w-14 h-14 rounded-xl bg-white/20 flex items-center justify-center text-3xl">
                  {categoryColors[model.category]?.icon || '📦'}
                </div>
                <div>
                  <h2 className="text-2xl font-bold">{model.name}</h2>
                  <p className="text-white/80">{model.type} • {model.category.toUpperCase()}</p>
                </div>
              </div>
              <button onClick={onClose} className="p-2 hover:bg-white/20 rounded-lg">
                <FiX size={24} />
              </button>
            </div>
          </div>

          <div className="p-6 overflow-y-auto max-h-[calc(90vh-120px)]">
            {isLoading ? (
              <div className="text-center py-12">
                <div className="animate-spin w-10 h-10 border-4 border-steg-accent border-t-transparent rounded-full mx-auto" />
                <p className="text-slate-500 mt-4">Running model predictions...</p>
              </div>
            ) : error ? (
              <div className="text-center py-12">
                <FiAlertTriangle size={48} className="mx-auto text-red-400" />
                <p className="text-red-500 mt-4">Error loading results</p>
              </div>
            ) : results ? (
              <div className="space-y-6">
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <div className="card p-4 text-center">
                    <FiDatabase className="w-6 h-6 mx-auto mb-2 text-blue-600" />
                    <div className="text-2xl font-bold text-slate-800">{results.samples_processed}</div>
                    <div className="text-xs text-slate-500">Samples Processed</div>
                  </div>
                  <div className="card p-4 text-center">
                    <FiBarChart2 className="w-6 h-6 mx-auto mb-2 text-purple-600" />
                    <div className="text-2xl font-bold text-slate-800">{results.summary?.unique_predictions || 0}</div>
                    <div className="text-xs text-slate-500">Unique Predictions</div>
                  </div>
                  {results.summary?.anomaly_ratio !== undefined && (
                    <div className="card p-4 text-center">
                      <FiAlertTriangle className="w-6 h-6 mx-auto mb-2 text-red-600" />
                      <div className="text-2xl font-bold text-red-600">{(results.summary.anomaly_ratio * 100).toFixed(1)}%</div>
                      <div className="text-xs text-slate-500">Anomaly Rate</div>
                    </div>
                  )}
                  {results.summary?.anomalies_detected !== undefined && (
                    <div className="card p-4 text-center">
                      <FiActivity className="w-6 h-6 mx-auto mb-2 text-orange-600" />
                      <div className="text-2xl font-bold text-orange-600">{results.summary.anomalies_detected}</div>
                      <div className="text-xs text-slate-500">Anomalies Detected</div>
                    </div>
                  )}
                </div>

                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                  {predictionDistData.length > 0 && (
                    <div className="card p-6">
                      <h3 className="text-lg font-semibold text-slate-800 mb-4">Prediction Distribution</h3>
                      <ResponsiveContainer width="100%" height={250}>
                        <PieChart>
                          <Pie data={predictionDistData} cx="50%" cy="50%" innerRadius={50} outerRadius={90} paddingAngle={5} dataKey="value"
                            label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}>
                            {predictionDistData.map((_, index) => (<Cell key={index} fill={COLORS[index % COLORS.length]} />))}
                          </Pie>
                          <Tooltip />
                        </PieChart>
                      </ResponsiveContainer>
                    </div>
                  )}
                  {featureImportanceData.length > 0 && (
                    <div className="card p-6">
                      <h3 className="text-lg font-semibold text-slate-800 mb-4">Feature Importance</h3>
                      <ResponsiveContainer width="100%" height={250}>
                        <BarChart data={featureImportanceData} layout="vertical">
                          <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                          <XAxis type="number" tick={{ fontSize: 11 }} />
                          <YAxis type="category" dataKey="feature" tick={{ fontSize: 10 }} width={100} />
                          <Tooltip />
                          <Bar dataKey="importance" fill="#3b82f6" radius={[0, 4, 4, 0]} />
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                  )}
                  {anomalyScoresData.length > 0 && (
                    <div className="card p-6 lg:col-span-2">
                      <h3 className="text-lg font-semibold text-slate-800 mb-4">Anomaly Scores</h3>
                      <ResponsiveContainer width="100%" height={200}>
                        <LineChart data={anomalyScoresData}>
                          <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                          <XAxis dataKey="index" tick={{ fontSize: 11 }} />
                          <YAxis tick={{ fontSize: 11 }} />
                          <Tooltip />
                          <Line type="monotone" dataKey="score" stroke="#8b5cf6" strokeWidth={2} dot={false} />
                        </LineChart>
                      </ResponsiveContainer>
                    </div>
                  )}
                </div>

                {results.predictions && results.predictions.length > 0 && (
                  <div className="card p-6">
                    <h3 className="text-lg font-semibold text-slate-800 mb-4">Sample Predictions (First 20)</h3>
                    <div className="flex flex-wrap gap-2">
                      {results.predictions.slice(0, 20).map((pred: number, i: number) => (
                        <span key={i} className={`px-3 py-1 rounded-full text-sm font-medium ${
                          pred === -1 ? 'bg-red-100 text-red-700' : pred === 1 ? 'bg-green-100 text-green-700' : 'bg-blue-100 text-blue-700'
                        }`}>
                          {pred === -1 ? 'Anomaly' : pred === 1 ? 'Normal' : pred}
                        </span>
                      ))}
                    </div>
                  </div>
                )}

                {/* LLM Situation Description */}
                {results.llm_description && (
                  <div className="card p-6 bg-gradient-to-br from-indigo-50 to-purple-50 border-2 border-indigo-200">
                    <div className="flex items-center gap-3 mb-4">
                      <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center">
                        <FiMessageSquare className="text-white" size={20} />
                      </div>
                      <div>
                        <h3 className="text-lg font-semibold text-slate-800">AI Analysis</h3>
                        <p className="text-xs text-slate-500">Intelligent situation description powered by LLM</p>
                      </div>
                    </div>
                    <div className="prose prose-sm max-w-none text-slate-700">
                      <pre className="whitespace-pre-wrap font-sans bg-white/50 rounded-xl p-4 text-sm leading-relaxed border border-indigo-100">
                        {results.llm_description}
                      </pre>
                    </div>
                  </div>
                )}
              </div>
            ) : null}
          </div>
        </motion.div>
      </motion.div>
    </AnimatePresence>
  )
}

function ModelCard({ model, onClick }: { model: Model; onClick: () => void }) {
  const categoryStyle = categoryColors[model.category] || categoryColors.default

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      whileHover={{ y: -4, scale: 1.02 }}
      whileTap={{ scale: 0.98 }}
      className="card p-6 cursor-pointer hover:shadow-lg transition-shadow"
      onClick={onClick}
    >
      <div className="flex items-start justify-between mb-4">
        <div className="flex items-center gap-3">
          <div className={`w-12 h-12 rounded-xl ${categoryStyle.bg} flex items-center justify-center text-2xl`}>
            {categoryStyle.icon}
          </div>
          <div>
            <h3 className="font-semibold text-slate-800 line-clamp-1" title={model.name}>{model.name}</h3>
            <div className="flex items-center gap-2 mt-1">
              <span className={`px-2 py-0.5 rounded-full text-xs font-medium ${categoryStyle.bg} ${categoryStyle.text}`}>
                {model.category.toUpperCase()}
              </span>
              <span className="text-xs text-slate-400">{model.type}</span>
            </div>
          </div>
        </div>
        <div className={`w-3 h-3 rounded-full ${model.loaded ? 'bg-green-500' : 'bg-slate-300'}`} />
      </div>
      <div className="flex items-center justify-between text-sm">
        <div className="flex items-center gap-1 text-green-600">
          {model.loaded ? <><FiCheckCircle size={14} /><span>Loaded</span></> : <><FiXCircle size={14} className="text-slate-400" /><span className="text-slate-400">Not Loaded</span></>}
        </div>
        <div className="flex items-center gap-1 text-steg-accent">
          <FiTrendingUp size={14} />
          <span className="text-xs font-medium">View Results</span>
        </div>
      </div>
    </motion.div>
  )
}

export default function Models() {
  const [searchTerm, setSearchTerm] = useState('')
  const [filterCategory, setFilterCategory] = useState<string>('')
  const [selectedModel, setSelectedModel] = useState<Model | null>(null)

  const { data: models, isLoading, refetch } = useQuery({
    queryKey: ['models'],
    queryFn: () => getModels().then(r => r.data),
  })

  const filteredModels = models?.filter((model: Model) => {
    const matchesSearch = model.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
                          model.category.toLowerCase().includes(searchTerm.toLowerCase())
    const matchesCategory = !filterCategory || model.category === filterCategory
    return matchesSearch && matchesCategory
  }) || []

  const categories = [...new Set(models?.map((m: Model) => m.category) || [])] as string[]
  
  const stats = {
    total: models?.length || 0,
    loaded: models?.filter((m: Model) => m.loaded).length || 0,
    ml: models?.filter((m: Model) => m.category === 'ml').length || 0,
    pd: models?.filter((m: Model) => m.category === 'pd').length || 0,
    tg1: models?.filter((m: Model) => m.category === 'tg1').length || 0,
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-slate-800">ML Models Registry</h1>
          <p className="text-slate-500">Manage and monitor trained models</p>
        </div>
        <button
          onClick={() => refetch()}
          className="px-4 py-2 bg-steg-accent text-white rounded-lg hover:bg-steg-light transition-colors flex items-center gap-2"
        >
          <FiRefreshCw size={18} />
          Reload Models
        </button>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
        <div className="card p-4 text-center">
          <FiLayers className="w-6 h-6 mx-auto mb-2 text-slate-600" />
          <div className="text-2xl font-bold text-slate-800">{stats.total}</div>
          <div className="text-xs text-slate-500">Total Models</div>
        </div>
        <div className="card p-4 text-center">
          <FiCheckCircle className="w-6 h-6 mx-auto mb-2 text-green-600" />
          <div className="text-2xl font-bold text-green-600">{stats.loaded}</div>
          <div className="text-xs text-slate-500">Loaded</div>
        </div>
        <div className="card p-4 text-center">
          <FiCpu className="w-6 h-6 mx-auto mb-2 text-blue-600" />
          <div className="text-2xl font-bold text-blue-600">{stats.ml}</div>
          <div className="text-xs text-slate-500">ML Models</div>
        </div>
        <div className="card p-4 text-center">
          <FiActivity className="w-6 h-6 mx-auto mb-2 text-purple-600" />
          <div className="text-2xl font-bold text-purple-600">{stats.pd}</div>
          <div className="text-xs text-slate-500">PD Models</div>
        </div>
        <div className="card p-4 text-center">
          <FiDatabase className="w-6 h-6 mx-auto mb-2 text-green-600" />
          <div className="text-2xl font-bold text-green-600">{stats.tg1}</div>
          <div className="text-xs text-slate-500">TG1 Models</div>
        </div>
      </div>

      {/* Filters */}
      <div className="card p-4 flex flex-wrap gap-4">
        <div className="relative flex-1 min-w-[200px]">
          <FiSearch className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" />
          <input
            type="text"
            placeholder="Search models..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="w-full pl-10 pr-4 py-2 border border-slate-200 rounded-lg focus:ring-2 focus:ring-steg-accent focus:border-transparent"
          />
        </div>
        <select
          value={filterCategory}
          onChange={(e) => setFilterCategory(e.target.value)}
          className="px-4 py-2 border border-slate-200 rounded-lg"
        >
          <option value="">All Categories</option>
          {categories.map((cat) => (
            <option key={cat} value={cat}>{cat.toUpperCase()}</option>
          ))}
        </select>
      </div>

      {/* Models Grid */}
      {isLoading ? (
        <div className="text-center py-12">
          <div className="animate-spin w-8 h-8 border-4 border-steg-accent border-t-transparent rounded-full mx-auto" />
          <p className="text-slate-500 mt-4">Loading models...</p>
        </div>
      ) : filteredModels.length > 0 ? (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {filteredModels.map((model: Model) => (
            <ModelCard key={model.name} model={model} onClick={() => setSelectedModel(model)} />
          ))}
        </div>
      ) : (
        <div className="text-center py-12 card">
          <FiInfo size={48} className="mx-auto text-slate-300" />
          <p className="text-slate-500 mt-4">No models found</p>
          <p className="text-sm text-slate-400">Try adjusting your search or filter</p>
        </div>
      )}

      {/* Model Results Modal */}
      {selectedModel && (
        <ModelResultsModal
          model={selectedModel}
          isOpen={!!selectedModel}
          onClose={() => setSelectedModel(null)}
        />
      )}
    </div>
  )
}
