import { useState } from 'react'
import { Outlet, NavLink, useLocation } from 'react-router-dom'
import { motion } from 'framer-motion'
import {
  FiHome,
  FiCpu,
  FiActivity,
  FiFileText,
  FiTrendingUp,
  FiBarChart2,
  FiSettings,
  FiMenu,
  FiX,
  FiBell,
  FiUser,
} from 'react-icons/fi'
import { useQuery } from '@tanstack/react-query'
import { getHealth, getTicketStats } from '../services/api'

const navItems = [
  { path: '/', icon: FiHome, label: 'Dashboard' },
  { path: '/models', icon: FiCpu, label: 'ML Models' },
  { path: '/monitoring', icon: FiActivity, label: 'Monitoring' },
  { path: '/ticketing', icon: FiFileText, label: 'Ticketing' },
  { path: '/drift', icon: FiTrendingUp, label: 'Drift Control' },
  { path: '/analytics', icon: FiBarChart2, label: 'Analytics' },
  { path: '/settings', icon: FiSettings, label: 'Settings' },
]

export default function Layout() {
  const [sidebarOpen, setSidebarOpen] = useState(true)
  const location = useLocation()
  
  const { data: healthData } = useQuery({
    queryKey: ['health'],
    queryFn: () => getHealth().then(r => r.data),
    refetchInterval: 30000,
  })
  
  const { data: ticketStats } = useQuery({
    queryKey: ['ticketStats'],
    queryFn: () => getTicketStats().then(r => r.data),
    refetchInterval: 30000,
  })

  const openTickets = ticketStats?.open || 0
  
  return (
    <div className="min-h-screen bg-slate-50">
      {/* Sidebar */}
      <motion.aside
        initial={{ width: sidebarOpen ? 256 : 80 }}
        animate={{ width: sidebarOpen ? 256 : 80 }}
        className="fixed left-0 top-0 h-full bg-gradient-steg text-white z-50 shadow-xl"
      >
        {/* Logo */}
        <div className="h-16 flex items-center justify-between px-4">
          {sidebarOpen && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="flex items-center gap-3"
            >
              <div className="w-10 h-10 bg-white/20 rounded-lg flex items-center justify-center">
                <span className="text-xl font-bold">TG1</span>
              </div>
              <div>
                <h1 className="font-bold text-lg">Digital Twin</h1>
                <p className="text-xs text-white/70">STEG Monitoring</p>
              </div>
            </motion.div>
          )}
          <button
            onClick={() => setSidebarOpen(!sidebarOpen)}
            className="p-2 hover:bg-white/10 rounded-lg transition-colors"
          >
            {sidebarOpen ? <FiX size={20} /> : <FiMenu size={20} />}
          </button>
        </div>

        {/* Navigation */}
        <nav className="mt-8 px-3">
          {navItems.map((item) => {
            const isActive = location.pathname === item.path
            return (
              <NavLink
                key={item.path}
                to={item.path}
                className={`flex items-center gap-3 px-4 py-3 rounded-lg mb-2 transition-all ${
                  isActive
                    ? 'bg-white/20 text-white'
                    : 'text-white/70 hover:bg-white/10 hover:text-white'
                }`}
              >
                <item.icon size={20} />
                {sidebarOpen && (
                  <span className="font-medium">{item.label}</span>
                )}
                {item.path === '/ticketing' && openTickets > 0 && sidebarOpen && (
                  <span className="ml-auto bg-red-500 text-white text-xs px-2 py-0.5 rounded-full">
                    {openTickets}
                  </span>
                )}
              </NavLink>
            )
          })}
        </nav>

        {/* Status */}
        {sidebarOpen && (
          <div className="absolute bottom-4 left-4 right-4">
            <div className="bg-white/10 rounded-lg p-3">
              <div className="flex items-center gap-2">
                <div className={`w-2 h-2 rounded-full ${
                  healthData?.status === 'healthy' ? 'bg-green-400' : 'bg-red-400'
                }`} />
                <span className="text-sm text-white/80">
                  {healthData?.status === 'healthy' ? 'System Healthy' : 'Checking...'}
                </span>
              </div>
              <p className="text-xs text-white/50 mt-1">
                {healthData?.models_loaded?.ml_models || 0} ML models loaded
              </p>
            </div>
          </div>
        )}
      </motion.aside>

      {/* Main Content */}
      <main
        className="transition-all duration-300"
        style={{ marginLeft: sidebarOpen ? 256 : 80 }}
      >
        {/* Header */}
        <header className="h-16 bg-white border-b border-slate-200 flex items-center justify-between px-6 sticky top-0 z-40">
          <div>
            <h2 className="text-lg font-semibold text-slate-800">
              {navItems.find(item => item.path === location.pathname)?.label || 'Dashboard'}
            </h2>
            <p className="text-sm text-slate-500">
              {new Date().toLocaleDateString('fr-FR', { 
                weekday: 'long', 
                year: 'numeric', 
                month: 'long', 
                day: 'numeric' 
              })}
            </p>
          </div>
          
          <div className="flex items-center gap-4">
            {/* Notifications */}
            <button className="relative p-2 hover:bg-slate-100 rounded-lg transition-colors">
              <FiBell size={20} className="text-slate-600" />
              {openTickets > 0 && (
                <span className="absolute top-1 right-1 w-2 h-2 bg-red-500 rounded-full" />
              )}
            </button>
            
            {/* User */}
            <div className="flex items-center gap-3 pl-4 border-l border-slate-200">
              <div className="w-8 h-8 bg-steg-accent rounded-full flex items-center justify-center">
                <FiUser className="text-white" size={16} />
              </div>
              <div>
                <p className="text-sm font-medium text-slate-700">Admin</p>
                <p className="text-xs text-slate-500">STEG</p>
              </div>
            </div>
          </div>
        </header>

        {/* Page Content */}
        <div className="p-6">
          <Outlet />
        </div>
      </main>
    </div>
  )
}
