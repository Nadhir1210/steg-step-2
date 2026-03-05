import { motion } from 'framer-motion'
import { useState } from 'react'
import toast from 'react-hot-toast'
import {
  FiSave,
  FiRefreshCw,
  FiBell,
  FiDatabase,
  FiCpu,
  FiShield,
  FiMail,
  FiSliders,
  FiToggleLeft,
  FiToggleRight,
} from 'react-icons/fi'

interface SettingsSection {
  id: string
  title: string
  icon: React.ElementType
  description: string
}

const sections: SettingsSection[] = [
  { id: 'thresholds', title: 'Alert Thresholds', icon: FiSliders, description: 'Configure anomaly detection thresholds' },
  { id: 'notifications', title: 'Notifications', icon: FiBell, description: 'Manage notification preferences' },
  { id: 'models', title: 'Model Settings', icon: FiCpu, description: 'Configure ML model parameters' },
  { id: 'data', title: 'Data Sources', icon: FiDatabase, description: 'Manage data connections' },
  { id: 'security', title: 'Security', icon: FiShield, description: 'Security and access settings' },
]

function Toggle({ enabled, onChange }: { enabled: boolean; onChange: (v: boolean) => void }) {
  return (
    <button
      onClick={() => onChange(!enabled)}
      className={`relative w-12 h-6 rounded-full transition-colors ${
        enabled ? 'bg-steg-accent' : 'bg-slate-300'
      }`}
    >
      <motion.div
        animate={{ x: enabled ? 24 : 2 }}
        className="absolute top-1 w-4 h-4 bg-white rounded-full shadow"
      />
    </button>
  )
}

export default function Settings() {
  const [activeSection, setActiveSection] = useState('thresholds')
  const [settings, setSettings] = useState({
    // Thresholds
    tempWarning: 90,
    tempCritical: 100,
    vibrationWarning: 1.5,
    vibrationCritical: 2.5,
    pressureWarning: 1.5,
    pressureCritical: 1.8,
    
    // Notifications
    emailEnabled: true,
    smsEnabled: false,
    slackEnabled: true,
    emailAddress: 'admin@steg.com.tn',
    notifyOnCritical: true,
    notifyOnHigh: true,
    notifyOnMedium: false,
    
    // Models
    autoRetrain: true,
    retrainInterval: 7,
    driftThreshold: 0.15,
    confidenceThreshold: 0.85,
    
    // Data
    dataRefreshInterval: 60,
    retentionDays: 90,
    compressionEnabled: true,
  })

  const handleSave = () => {
    // In real app, this would call the API
    toast.success('Settings saved successfully!')
  }

  const handleReset = () => {
    toast.success('Settings reset to defaults')
  }

  const updateSetting = (key: string, value: any) => {
    setSettings(prev => ({ ...prev, [key]: value }))
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-slate-800">Settings</h1>
          <p className="text-slate-500">Configure system parameters and preferences</p>
        </div>
        <div className="flex gap-3">
          <button
            onClick={handleReset}
            className="px-4 py-2 border border-slate-200 rounded-lg hover:bg-slate-50 transition-colors flex items-center gap-2"
          >
            <FiRefreshCw size={18} />
            Reset
          </button>
          <button
            onClick={handleSave}
            className="px-4 py-2 bg-steg-accent text-white rounded-lg hover:bg-steg-light transition-colors flex items-center gap-2"
          >
            <FiSave size={18} />
            Save Changes
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        {/* Sidebar */}
        <div className="card p-4">
          <nav className="space-y-1">
            {sections.map((section) => (
              <button
                key={section.id}
                onClick={() => setActiveSection(section.id)}
                className={`w-full flex items-center gap-3 px-4 py-3 rounded-lg transition-colors text-left ${
                  activeSection === section.id
                    ? 'bg-steg-accent text-white'
                    : 'hover:bg-slate-100 text-slate-700'
                }`}
              >
                <section.icon size={20} />
                <span className="font-medium">{section.title}</span>
              </button>
            ))}
          </nav>
        </div>

        {/* Content */}
        <div className="lg:col-span-3 card p-6">
          {activeSection === 'thresholds' && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="space-y-6"
            >
              <div>
                <h2 className="text-lg font-semibold text-slate-800 mb-2">Alert Thresholds</h2>
                <p className="text-sm text-slate-500">Configure when alerts should be triggered</p>
              </div>

              {/* Temperature */}
              <div className="space-y-4">
                <h3 className="font-medium text-slate-700">Temperature (°C)</h3>
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm text-slate-600 mb-1">Warning Threshold</label>
                    <input
                      type="number"
                      value={settings.tempWarning}
                      onChange={(e) => updateSetting('tempWarning', parseInt(e.target.value))}
                      className="w-full px-4 py-2 border border-slate-200 rounded-lg focus:ring-2 focus:ring-steg-accent"
                    />
                  </div>
                  <div>
                    <label className="block text-sm text-slate-600 mb-1">Critical Threshold</label>
                    <input
                      type="number"
                      value={settings.tempCritical}
                      onChange={(e) => updateSetting('tempCritical', parseInt(e.target.value))}
                      className="w-full px-4 py-2 border border-slate-200 rounded-lg focus:ring-2 focus:ring-steg-accent"
                    />
                  </div>
                </div>
              </div>

              {/* Vibration */}
              <div className="space-y-4">
                <h3 className="font-medium text-slate-700">Vibration (mm/s)</h3>
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm text-slate-600 mb-1">Warning Threshold</label>
                    <input
                      type="number"
                      step="0.1"
                      value={settings.vibrationWarning}
                      onChange={(e) => updateSetting('vibrationWarning', parseFloat(e.target.value))}
                      className="w-full px-4 py-2 border border-slate-200 rounded-lg focus:ring-2 focus:ring-steg-accent"
                    />
                  </div>
                  <div>
                    <label className="block text-sm text-slate-600 mb-1">Critical Threshold</label>
                    <input
                      type="number"
                      step="0.1"
                      value={settings.vibrationCritical}
                      onChange={(e) => updateSetting('vibrationCritical', parseFloat(e.target.value))}
                      className="w-full px-4 py-2 border border-slate-200 rounded-lg focus:ring-2 focus:ring-steg-accent"
                    />
                  </div>
                </div>
              </div>

              {/* Pressure */}
              <div className="space-y-4">
                <h3 className="font-medium text-slate-700">Pressure (bar)</h3>
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm text-slate-600 mb-1">Warning Threshold</label>
                    <input
                      type="number"
                      step="0.1"
                      value={settings.pressureWarning}
                      onChange={(e) => updateSetting('pressureWarning', parseFloat(e.target.value))}
                      className="w-full px-4 py-2 border border-slate-200 rounded-lg focus:ring-2 focus:ring-steg-accent"
                    />
                  </div>
                  <div>
                    <label className="block text-sm text-slate-600 mb-1">Critical Threshold</label>
                    <input
                      type="number"
                      step="0.1"
                      value={settings.pressureCritical}
                      onChange={(e) => updateSetting('pressureCritical', parseFloat(e.target.value))}
                      className="w-full px-4 py-2 border border-slate-200 rounded-lg focus:ring-2 focus:ring-steg-accent"
                    />
                  </div>
                </div>
              </div>
            </motion.div>
          )}

          {activeSection === 'notifications' && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="space-y-6"
            >
              <div>
                <h2 className="text-lg font-semibold text-slate-800 mb-2">Notification Settings</h2>
                <p className="text-sm text-slate-500">Configure how you receive alerts</p>
              </div>

              {/* Channels */}
              <div className="space-y-4">
                <h3 className="font-medium text-slate-700">Notification Channels</h3>
                <div className="space-y-3">
                  <div className="flex items-center justify-between p-4 bg-slate-50 rounded-lg">
                    <div className="flex items-center gap-3">
                      <FiMail className="text-slate-600" size={20} />
                      <div>
                        <p className="font-medium text-slate-800">Email Notifications</p>
                        <p className="text-sm text-slate-500">Receive alerts via email</p>
                      </div>
                    </div>
                    <Toggle
                      enabled={settings.emailEnabled}
                      onChange={(v) => updateSetting('emailEnabled', v)}
                    />
                  </div>

                  <div className="flex items-center justify-between p-4 bg-slate-50 rounded-lg">
                    <div className="flex items-center gap-3">
                      <FiBell className="text-slate-600" size={20} />
                      <div>
                        <p className="font-medium text-slate-800">SMS Notifications</p>
                        <p className="text-sm text-slate-500">Receive alerts via SMS</p>
                      </div>
                    </div>
                    <Toggle
                      enabled={settings.smsEnabled}
                      onChange={(v) => updateSetting('smsEnabled', v)}
                    />
                  </div>

                  <div className="flex items-center justify-between p-4 bg-slate-50 rounded-lg">
                    <div>
                      <p className="font-medium text-slate-800">Slack Integration</p>
                      <p className="text-sm text-slate-500">Post alerts to Slack channel</p>
                    </div>
                    <Toggle
                      enabled={settings.slackEnabled}
                      onChange={(v) => updateSetting('slackEnabled', v)}
                    />
                  </div>
                </div>
              </div>

              {/* Email */}
              {settings.emailEnabled && (
                <div>
                  <label className="block text-sm text-slate-600 mb-1">Email Address</label>
                  <input
                    type="email"
                    value={settings.emailAddress}
                    onChange={(e) => updateSetting('emailAddress', e.target.value)}
                    className="w-full px-4 py-2 border border-slate-200 rounded-lg focus:ring-2 focus:ring-steg-accent"
                  />
                </div>
              )}

              {/* Priority */}
              <div className="space-y-4">
                <h3 className="font-medium text-slate-700">Notify on Priority</h3>
                <div className="flex flex-wrap gap-4">
                  <label className="flex items-center gap-2">
                    <input
                      type="checkbox"
                      checked={settings.notifyOnCritical}
                      onChange={(e) => updateSetting('notifyOnCritical', e.target.checked)}
                      className="w-4 h-4 text-steg-accent"
                    />
                    <span className="text-sm">Critical</span>
                  </label>
                  <label className="flex items-center gap-2">
                    <input
                      type="checkbox"
                      checked={settings.notifyOnHigh}
                      onChange={(e) => updateSetting('notifyOnHigh', e.target.checked)}
                      className="w-4 h-4 text-steg-accent"
                    />
                    <span className="text-sm">High</span>
                  </label>
                  <label className="flex items-center gap-2">
                    <input
                      type="checkbox"
                      checked={settings.notifyOnMedium}
                      onChange={(e) => updateSetting('notifyOnMedium', e.target.checked)}
                      className="w-4 h-4 text-steg-accent"
                    />
                    <span className="text-sm">Medium</span>
                  </label>
                </div>
              </div>
            </motion.div>
          )}

          {activeSection === 'models' && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="space-y-6"
            >
              <div>
                <h2 className="text-lg font-semibold text-slate-800 mb-2">Model Settings</h2>
                <p className="text-sm text-slate-500">Configure ML model behavior</p>
              </div>

              <div className="flex items-center justify-between p-4 bg-slate-50 rounded-lg">
                <div>
                  <p className="font-medium text-slate-800">Auto Retrain</p>
                  <p className="text-sm text-slate-500">Automatically retrain models when drift detected</p>
                </div>
                <Toggle
                  enabled={settings.autoRetrain}
                  onChange={(v) => updateSetting('autoRetrain', v)}
                />
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm text-slate-600 mb-1">Retrain Interval (days)</label>
                  <input
                    type="number"
                    value={settings.retrainInterval}
                    onChange={(e) => updateSetting('retrainInterval', parseInt(e.target.value))}
                    className="w-full px-4 py-2 border border-slate-200 rounded-lg focus:ring-2 focus:ring-steg-accent"
                  />
                </div>
                <div>
                  <label className="block text-sm text-slate-600 mb-1">Drift Threshold</label>
                  <input
                    type="number"
                    step="0.01"
                    value={settings.driftThreshold}
                    onChange={(e) => updateSetting('driftThreshold', parseFloat(e.target.value))}
                    className="w-full px-4 py-2 border border-slate-200 rounded-lg focus:ring-2 focus:ring-steg-accent"
                  />
                </div>
              </div>

              <div>
                <label className="block text-sm text-slate-600 mb-1">
                  Confidence Threshold: {(settings.confidenceThreshold * 100).toFixed(0)}%
                </label>
                <input
                  type="range"
                  min="50"
                  max="99"
                  value={settings.confidenceThreshold * 100}
                  onChange={(e) => updateSetting('confidenceThreshold', parseInt(e.target.value) / 100)}
                  className="w-full"
                />
                <div className="flex justify-between text-xs text-slate-400">
                  <span>50%</span>
                  <span>99%</span>
                </div>
              </div>
            </motion.div>
          )}

          {activeSection === 'data' && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="space-y-6"
            >
              <div>
                <h2 className="text-lg font-semibold text-slate-800 mb-2">Data Settings</h2>
                <p className="text-sm text-slate-500">Configure data sources and retention</p>
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm text-slate-600 mb-1">Refresh Interval (seconds)</label>
                  <input
                    type="number"
                    value={settings.dataRefreshInterval}
                    onChange={(e) => updateSetting('dataRefreshInterval', parseInt(e.target.value))}
                    className="w-full px-4 py-2 border border-slate-200 rounded-lg focus:ring-2 focus:ring-steg-accent"
                  />
                </div>
                <div>
                  <label className="block text-sm text-slate-600 mb-1">Data Retention (days)</label>
                  <input
                    type="number"
                    value={settings.retentionDays}
                    onChange={(e) => updateSetting('retentionDays', parseInt(e.target.value))}
                    className="w-full px-4 py-2 border border-slate-200 rounded-lg focus:ring-2 focus:ring-steg-accent"
                  />
                </div>
              </div>

              <div className="flex items-center justify-between p-4 bg-slate-50 rounded-lg">
                <div>
                  <p className="font-medium text-slate-800">Data Compression</p>
                  <p className="text-sm text-slate-500">Compress historical data to save storage</p>
                </div>
                <Toggle
                  enabled={settings.compressionEnabled}
                  onChange={(v) => updateSetting('compressionEnabled', v)}
                />
              </div>
            </motion.div>
          )}

          {activeSection === 'security' && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="space-y-6"
            >
              <div>
                <h2 className="text-lg font-semibold text-slate-800 mb-2">Security Settings</h2>
                <p className="text-sm text-slate-500">Manage access and security preferences</p>
              </div>

              <div className="p-6 bg-slate-50 rounded-lg text-center">
                <FiShield className="mx-auto text-slate-400" size={48} />
                <p className="mt-4 text-slate-600">Security settings will be available in a future update.</p>
                <p className="text-sm text-slate-400 mt-2">Contact your administrator for access control.</p>
              </div>
            </motion.div>
          )}
        </div>
      </div>
    </div>
  )
}
