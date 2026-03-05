import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { motion } from 'framer-motion'
import { useState } from 'react'
import toast from 'react-hot-toast'
import {
  FiPlus,
  FiFilter,
  FiSearch,
  FiAlertTriangle,
  FiClock,
  FiUser,
  FiCheckCircle,
  FiXCircle,
  FiRefreshCw,
} from 'react-icons/fi'
import { getTickets, getTicketStats, createTicket, updateTicketStatus } from '../services/api'
import type { Ticket } from '../types'

const priorityColors = {
  CRITICAL: 'bg-red-100 text-red-800 border-red-200',
  HIGH: 'bg-orange-100 text-orange-800 border-orange-200',
  MEDIUM: 'bg-yellow-100 text-yellow-800 border-yellow-200',
  LOW: 'bg-green-100 text-green-800 border-green-200',
}

const statusColors = {
  OPEN: 'bg-blue-100 text-blue-800',
  IN_PROGRESS: 'bg-purple-100 text-purple-800',
  RESOLVED: 'bg-green-100 text-green-800',
  CLOSED: 'bg-slate-100 text-slate-800',
}

const modules = ['THERMAL', 'COOLING', 'ELECTRICAL', 'PD', 'GLOBAL']

function TicketCard({ ticket, onStatusChange }: { ticket: Ticket; onStatusChange: (id: string, status: string) => void }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="card p-6"
    >
      <div className="flex items-start justify-between mb-4">
        <div>
          <div className="flex items-center gap-2">
            <span className={`px-2 py-1 rounded-full text-xs font-medium border ${priorityColors[ticket.priority]}`}>
              {ticket.priority}
            </span>
            <span className={`px-2 py-1 rounded-full text-xs font-medium ${statusColors[ticket.status]}`}>
              {ticket.status}
            </span>
          </div>
          <h3 className="text-lg font-semibold text-slate-800 mt-2">
            {ticket.ticket_id}
          </h3>
          <p className="text-sm text-slate-500">{ticket.module}</p>
        </div>
        <div className="text-right">
          <div className="text-2xl font-bold text-slate-800">
            {ticket.severity_score?.toFixed(0)}
          </div>
          <div className="text-xs text-slate-500">Severity</div>
        </div>
      </div>

      {ticket.anomaly_type && (
        <p className="text-sm text-slate-600 mb-3">
          <strong>Type:</strong> {ticket.anomaly_type}
        </p>
      )}

      {ticket.description && (
        <p className="text-sm text-slate-600 mb-3 line-clamp-2">
          {ticket.description}
        </p>
      )}

      <div className="flex items-center gap-4 text-sm text-slate-500 mb-4">
        <div className="flex items-center gap-1">
          <FiClock size={14} />
          <span>{new Date(ticket.timestamp).toLocaleDateString('fr-FR')}</span>
        </div>
        {ticket.estimated_rul && (
          <div className="flex items-center gap-1">
            <FiAlertTriangle size={14} />
            <span>RUL: {ticket.estimated_rul}</span>
          </div>
        )}
      </div>

      {/* Actions */}
      <div className="flex gap-2 border-t border-slate-100 pt-4">
        {ticket.status === 'OPEN' && (
          <button
            onClick={() => onStatusChange(ticket.ticket_id, 'IN_PROGRESS')}
            className="flex-1 px-3 py-2 bg-purple-500 text-white rounded-lg text-sm hover:bg-purple-600 transition-colors"
          >
            Start Work
          </button>
        )}
        {ticket.status === 'IN_PROGRESS' && (
          <button
            onClick={() => onStatusChange(ticket.ticket_id, 'RESOLVED')}
            className="flex-1 px-3 py-2 bg-green-500 text-white rounded-lg text-sm hover:bg-green-600 transition-colors"
          >
            Resolve
          </button>
        )}
        {ticket.status === 'RESOLVED' && (
          <button
            onClick={() => onStatusChange(ticket.ticket_id, 'CLOSED')}
            className="flex-1 px-3 py-2 bg-slate-500 text-white rounded-lg text-sm hover:bg-slate-600 transition-colors"
          >
            Close
          </button>
        )}
      </div>
    </motion.div>
  )
}

function CreateTicketModal({ isOpen, onClose }: { isOpen: boolean; onClose: () => void }) {
  const queryClient = useQueryClient()
  const [formData, setFormData] = useState({
    module: 'THERMAL',
    severity_score: 70,
    metrics: { temperature: 85, load: 100 },
    ml_confidence: 0.85,
  })

  const mutation = useMutation({
    mutationFn: createTicket,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['tickets'] })
      queryClient.invalidateQueries({ queryKey: ['ticketStats'] })
      toast.success('Ticket created successfully!')
      onClose()
    },
    onError: () => {
      toast.error('Failed to create ticket')
    },
  })

  if (!isOpen) return null

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
      <motion.div
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        className="bg-white rounded-2xl p-6 w-full max-w-md shadow-xl"
      >
        <h2 className="text-xl font-bold text-slate-800 mb-6">Create Smart Ticket</h2>
        
        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-slate-700 mb-1">Module</label>
            <select
              value={formData.module}
              onChange={(e) => setFormData({ ...formData, module: e.target.value })}
              className="w-full px-4 py-2 border border-slate-200 rounded-lg focus:ring-2 focus:ring-steg-accent"
            >
              {modules.map((m) => (
                <option key={m} value={m}>{m}</option>
              ))}
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium text-slate-700 mb-1">
              Severity Score: {formData.severity_score}
            </label>
            <input
              type="range"
              min="0"
              max="100"
              value={formData.severity_score}
              onChange={(e) => setFormData({ ...formData, severity_score: parseInt(e.target.value) })}
              className="w-full"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-slate-700 mb-1">
              ML Confidence: {(formData.ml_confidence * 100).toFixed(0)}%
            </label>
            <input
              type="range"
              min="50"
              max="100"
              value={formData.ml_confidence * 100}
              onChange={(e) => setFormData({ ...formData, ml_confidence: parseInt(e.target.value) / 100 })}
              className="w-full"
            />
          </div>
        </div>

        <div className="flex gap-3 mt-6">
          <button
            onClick={onClose}
            className="flex-1 px-4 py-2 border border-slate-200 rounded-lg hover:bg-slate-50 transition-colors"
          >
            Cancel
          </button>
          <button
            onClick={() => mutation.mutate(formData)}
            disabled={mutation.isPending}
            className="flex-1 px-4 py-2 bg-steg-accent text-white rounded-lg hover:bg-steg-light transition-colors disabled:opacity-50"
          >
            {mutation.isPending ? 'Creating...' : 'Create Ticket'}
          </button>
        </div>
      </motion.div>
    </div>
  )
}

export default function Ticketing() {
  const queryClient = useQueryClient()
  const [showCreate, setShowCreate] = useState(false)
  const [filterPriority, setFilterPriority] = useState<string>('')
  const [filterStatus, setFilterStatus] = useState<string>('')

  const { data: tickets, isLoading, refetch } = useQuery({
    queryKey: ['tickets', filterPriority, filterStatus],
    queryFn: () => getTickets({ 
      priority: filterPriority || undefined, 
      status: filterStatus || undefined 
    }).then(r => r.data),
  })

  const { data: stats } = useQuery({
    queryKey: ['ticketStats'],
    queryFn: () => getTicketStats().then(r => r.data),
  })

  const statusMutation = useMutation({
    mutationFn: ({ id, status }: { id: string; status: string }) => updateTicketStatus(id, status),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['tickets'] })
      queryClient.invalidateQueries({ queryKey: ['ticketStats'] })
      toast.success('Ticket updated!')
    },
    onError: () => {
      toast.error('Failed to update ticket')
    },
  })

  const handleStatusChange = (id: string, status: string) => {
    statusMutation.mutate({ id, status })
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-slate-800">Smart Ticketing</h1>
          <p className="text-slate-500">ML + RAG + LLM Generated Tickets</p>
        </div>
        <div className="flex gap-3">
          <button
            onClick={() => refetch()}
            className="px-4 py-2 border border-slate-200 rounded-lg hover:bg-slate-50 transition-colors flex items-center gap-2"
          >
            <FiRefreshCw size={18} />
            Refresh
          </button>
          <button
            onClick={() => setShowCreate(true)}
            className="px-4 py-2 bg-steg-accent text-white rounded-lg hover:bg-steg-light transition-colors flex items-center gap-2"
          >
            <FiPlus size={18} />
            New Ticket
          </button>
        </div>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="card p-4 text-center">
          <div className="text-3xl font-bold text-slate-800">{stats?.total || 0}</div>
          <div className="text-sm text-slate-500">Total</div>
        </div>
        <div className="card p-4 text-center">
          <div className="text-3xl font-bold text-blue-600">{stats?.open || 0}</div>
          <div className="text-sm text-slate-500">Open</div>
        </div>
        <div className="card p-4 text-center">
          <div className="text-3xl font-bold text-red-600">{stats?.by_priority?.CRITICAL || 0}</div>
          <div className="text-sm text-slate-500">Critical</div>
        </div>
        <div className="card p-4 text-center">
          <div className="text-3xl font-bold text-orange-600">{stats?.by_priority?.HIGH || 0}</div>
          <div className="text-sm text-slate-500">High</div>
        </div>
      </div>

      {/* Filters */}
      <div className="card p-4 flex flex-wrap gap-4">
        <div className="flex items-center gap-2">
          <FiFilter className="text-slate-400" />
          <span className="text-sm font-medium text-slate-600">Filters:</span>
        </div>
        <select
          value={filterPriority}
          onChange={(e) => setFilterPriority(e.target.value)}
          className="px-3 py-1.5 border border-slate-200 rounded-lg text-sm"
        >
          <option value="">All Priorities</option>
          <option value="CRITICAL">Critical</option>
          <option value="HIGH">High</option>
          <option value="MEDIUM">Medium</option>
          <option value="LOW">Low</option>
        </select>
        <select
          value={filterStatus}
          onChange={(e) => setFilterStatus(e.target.value)}
          className="px-3 py-1.5 border border-slate-200 rounded-lg text-sm"
        >
          <option value="">All Status</option>
          <option value="OPEN">Open</option>
          <option value="IN_PROGRESS">In Progress</option>
          <option value="RESOLVED">Resolved</option>
          <option value="CLOSED">Closed</option>
        </select>
      </div>

      {/* Tickets Grid */}
      {isLoading ? (
        <div className="text-center py-12">
          <div className="animate-spin w-8 h-8 border-4 border-steg-accent border-t-transparent rounded-full mx-auto" />
          <p className="text-slate-500 mt-4">Loading tickets...</p>
        </div>
      ) : tickets?.tickets?.length > 0 ? (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {tickets.tickets.map((ticket: Ticket) => (
            <TicketCard 
              key={ticket.ticket_id} 
              ticket={ticket} 
              onStatusChange={handleStatusChange}
            />
          ))}
        </div>
      ) : (
        <div className="text-center py-12 card">
          <FiCheckCircle size={48} className="mx-auto text-slate-300" />
          <p className="text-slate-500 mt-4">No tickets found</p>
          <button
            onClick={() => setShowCreate(true)}
            className="mt-4 px-4 py-2 bg-steg-accent text-white rounded-lg hover:bg-steg-light transition-colors"
          >
            Create First Ticket
          </button>
        </div>
      )}

      {/* Create Modal */}
      <CreateTicketModal isOpen={showCreate} onClose={() => setShowCreate(false)} />
    </div>
  )
}
