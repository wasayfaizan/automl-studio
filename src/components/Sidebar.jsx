import { Link, useLocation } from 'react-router-dom'
import { 
  Upload, 
  Sparkles, 
  BarChart3, 
  Brain, 
  TrendingUp, 
  Download,
  Moon,
  Sun
} from 'lucide-react'
import { motion } from 'framer-motion'

const menuItems = [
  { path: '/upload', icon: Upload, label: 'Upload Dataset' },
  { path: '/cleaning', icon: Sparkles, label: 'Data Cleaning' },
  { path: '/eda', icon: BarChart3, label: 'EDA Report' },
  { path: '/modeling', icon: Brain, label: 'Modeling' },
  { path: '/results', icon: TrendingUp, label: 'Results & Metrics' },
  { path: '/export', icon: Download, label: 'Model Export' },
]

export default function Sidebar({ darkMode, setDarkMode }) {
  const location = useLocation()

  return (
    <motion.aside
      initial={{ x: -100 }}
      animate={{ x: 0 }}
      className="w-72 bg-white/80 dark:bg-gray-900/80 backdrop-blur-xl border-r border-gray-200/50 dark:border-gray-700/50 flex flex-col shadow-2xl"
    >
      <div className="p-6 border-b border-gray-200/50 dark:border-gray-700/50 bg-gradient-to-r from-primary-500 to-accent-500">
        <div className="bg-white/10 dark:bg-gray-900/20 backdrop-blur-sm rounded-xl p-4 border border-white/20">
          <h1 className="text-2xl font-bold text-white drop-shadow-lg">
            AutoML Studio
          </h1>
          <p className="text-sm text-white/90 mt-1 font-medium">
            Machine Learning Made Simple
          </p>
        </div>
      </div>

      <nav className="flex-1 p-4 space-y-2 overflow-y-auto">
        {menuItems.map((item, index) => {
          const Icon = item.icon
          const isActive = location.pathname === item.path
          
          return (
            <Link key={item.path} to={item.path}>
              <motion.div
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: index * 0.1 }}
                whileHover={{ x: 4, scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
                className={`flex items-center gap-3 px-4 py-3 rounded-xl transition-all duration-200 ${
                  isActive
                    ? 'bg-gradient-to-r from-primary-500 to-accent-500 text-white shadow-lg shadow-primary-500/50'
                    : 'text-gray-700 dark:text-gray-300 hover:bg-gradient-to-r hover:from-primary-50 hover:to-accent-50 dark:hover:from-primary-900/30 dark:hover:to-accent-900/30 hover:shadow-md'
                }`}
              >
                <Icon size={20} className={isActive ? 'text-white' : ''} />
                <span className={`font-semibold ${isActive ? 'text-white' : ''}`}>{item.label}</span>
              </motion.div>
            </Link>
          )
        })}
      </nav>

      <div className="p-4 border-t border-gray-200/50 dark:border-gray-700/50">
        <motion.button
          whileHover={{ scale: 1.02 }}
          whileTap={{ scale: 0.98 }}
          onClick={() => setDarkMode(!darkMode)}
          className="w-full flex items-center gap-3 px-4 py-3 rounded-xl text-gray-700 dark:text-gray-300 bg-gradient-to-r from-gray-100 to-gray-200 dark:from-gray-800 dark:to-gray-700 hover:from-gray-200 hover:to-gray-300 dark:hover:from-gray-700 dark:hover:to-gray-600 transition-all duration-200 shadow-md hover:shadow-lg"
        >
          {darkMode ? <Sun size={20} className="text-yellow-500" /> : <Moon size={20} className="text-indigo-500" />}
          <span className="font-semibold">{darkMode ? 'Light Mode' : 'Dark Mode'}</span>
        </motion.button>
      </div>
    </motion.aside>
  )
}

