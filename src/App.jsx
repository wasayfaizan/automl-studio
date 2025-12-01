import { useState, useEffect } from 'react'
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom'
import Sidebar from './components/Sidebar'
import UploadPage from './pages/UploadPage'
import CleaningPage from './pages/CleaningPage'
import EDAPage from './pages/EDAPage'
import ModelingPage from './pages/ModelingPage'
import ResultsPage from './pages/ResultsPage'
import ExportPage from './pages/ExportPage'

function App() {
  const [darkMode, setDarkMode] = useState(() => {
    const saved = localStorage.getItem('darkMode')
    return saved ? JSON.parse(saved) : false
  })

  useEffect(() => {
    if (darkMode) {
      document.documentElement.classList.add('dark')
    } else {
      document.documentElement.classList.remove('dark')
    }
    localStorage.setItem('darkMode', JSON.stringify(darkMode))
  }, [darkMode])

  return (
    <Router>
      <div className="flex h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-50 dark:from-gray-900 dark:via-slate-900 dark:to-indigo-950 transition-all duration-300">
        <Sidebar darkMode={darkMode} setDarkMode={setDarkMode} />
        <main className="flex-1 overflow-y-auto">
          <div className="min-h-full">
            <Routes>
              <Route path="/" element={<Navigate to="/upload" replace />} />
              <Route path="/upload" element={<UploadPage />} />
              <Route path="/cleaning" element={<CleaningPage />} />
              <Route path="/eda" element={<EDAPage />} />
              <Route path="/modeling" element={<ModelingPage />} />
              <Route path="/results" element={<ResultsPage />} />
              <Route path="/export" element={<ExportPage />} />
            </Routes>
          </div>
        </main>
      </div>
    </Router>
  )
}

export default App

