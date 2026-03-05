import { Routes, Route } from 'react-router-dom'
import { motion, AnimatePresence } from 'framer-motion'
import Layout from './components/Layout'
import Dashboard from './pages/Dashboard'
import Models from './pages/Models'
import Monitoring from './pages/Monitoring'
import Ticketing from './pages/Ticketing'
import DriftControl from './pages/DriftControl'
import Analytics from './pages/Analytics'
import Settings from './pages/Settings'

function App() {
  return (
    <AnimatePresence mode="wait">
      <Routes>
        <Route path="/" element={<Layout />}>
          <Route index element={<Dashboard />} />
          <Route path="models" element={<Models />} />
          <Route path="monitoring" element={<Monitoring />} />
          <Route path="ticketing" element={<Ticketing />} />
          <Route path="drift" element={<DriftControl />} />
          <Route path="analytics" element={<Analytics />} />
          <Route path="settings" element={<Settings />} />
        </Route>
      </Routes>
    </AnimatePresence>
  )
}

export default App
