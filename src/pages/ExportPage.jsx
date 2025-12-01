import { useState } from 'react'
import { motion } from 'framer-motion'
import { Download, FileText, Database, Brain, FileDown } from 'lucide-react'
import axios from 'axios'

export default function ExportPage() {
  const [downloading, setDownloading] = useState({
    model: false,
    dataset: false,
    report: false
  })

  const handleDownload = async (type) => {
    setDownloading({ ...downloading, [type]: true })
    try {
      let endpoint = ''
      let filename = ''
      
      switch (type) {
        case 'model':
          endpoint = '/api/download/model'
          filename = 'trained_model.pkl'
          break
        case 'dataset':
          endpoint = '/api/download/cleaned-data'
          filename = 'cleaned_dataset.csv'
          break
        case 'report':
          endpoint = '/api/generate-report'
          filename = 'ml_report.pdf'
          break
      }

      const response = await axios.get(endpoint, {
        responseType: 'blob'
      })
      
      const url = window.URL.createObjectURL(new Blob([response.data]))
      const link = document.createElement('a')
      link.href = url
      link.setAttribute('download', filename)
      document.body.appendChild(link)
      link.click()
      link.remove()
      window.URL.revokeObjectURL(url)
    } catch (error) {
      alert(`Failed to download ${type}: ${error.response?.data?.detail || error.message}`)
    } finally {
      setDownloading({ ...downloading, [type]: false })
    }
  }

  const exportItems = [
    {
      id: 'model',
      title: 'Trained Model',
      description: 'Download your trained model as a .pkl file for deployment',
      icon: Brain,
      color: 'primary'
    },
    {
      id: 'dataset',
      title: 'Cleaned Dataset',
      description: 'Download the preprocessed and cleaned dataset',
      icon: Database,
      color: 'green'
    },
    {
      id: 'report',
      title: 'Full PDF Report',
      description: 'Generate and download a comprehensive PDF report with all insights',
      icon: FileText,
      color: 'purple'
    }
  ]

  return (
    <div className="p-8 max-w-7xl mx-auto">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="mb-8"
      >
        <h1 className="text-3xl font-bold text-gray-900 dark:text-white mb-2">
          Model Export
        </h1>
        <p className="text-gray-600 dark:text-gray-400">
          Download your trained model, cleaned dataset, and comprehensive reports
        </p>
      </motion.div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
        {exportItems.map((item) => {
          const Icon = item.icon
          const colorClasses = {
            primary: 'bg-primary-100 dark:bg-primary-900 text-primary-600 dark:text-primary-400 border-primary-200 dark:border-primary-700',
            green: 'bg-green-100 dark:bg-green-900 text-green-600 dark:text-green-400 border-green-200 dark:border-green-700',
            purple: 'bg-purple-100 dark:bg-purple-900 text-purple-600 dark:text-purple-400 border-purple-200 dark:border-purple-700'
          }

          return (
            <motion.div
              key={item.id}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              whileHover={{ scale: 1.02 }}
              className="bg-white dark:bg-gray-800 rounded-lg shadow border border-gray-200 dark:border-gray-700 p-6"
            >
              <div className={`w-12 h-12 rounded-lg ${colorClasses[item.color]} flex items-center justify-center mb-4`}>
                <Icon className="w-6 h-6" />
              </div>
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-2">
                {item.title}
              </h3>
              <p className="text-sm text-gray-600 dark:text-gray-400 mb-4">
                {item.description}
              </p>
              <button
                onClick={() => handleDownload(item.id)}
                disabled={downloading[item.id]}
                className={`w-full px-4 py-2 ${
                  item.color === 'primary' ? 'bg-primary-600 hover:bg-primary-700' :
                  item.color === 'green' ? 'bg-green-600 hover:bg-green-700' :
                  'bg-purple-600 hover:bg-purple-700'
                } disabled:bg-gray-400 text-white rounded-lg font-medium transition-colors flex items-center justify-center gap-2`}
              >
                {downloading[item.id] ? (
                  <>
                    <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                    Downloading...
                  </>
                ) : (
                  <>
                    <Download className="w-4 h-4" />
                    Download
                  </>
                )}
              </button>
            </motion.div>
          )
        })}
      </div>

      {/* Report Contents */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="bg-white dark:bg-gray-800 rounded-lg shadow border border-gray-200 dark:border-gray-700 p-6"
      >
        <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-4">
          PDF Report Contents
        </h3>
        <div className="space-y-3">
          <div className="flex items-center gap-3 p-3 bg-gray-50 dark:bg-gray-700 rounded-lg">
            <FileDown className="w-5 h-5 text-gray-600 dark:text-gray-400" />
            <div>
              <div className="font-medium text-gray-900 dark:text-white">Dataset Summary</div>
              <div className="text-sm text-gray-600 dark:text-gray-400">
                Overview of your dataset including row count, columns, and data types
              </div>
            </div>
          </div>
          <div className="flex items-center gap-3 p-3 bg-gray-50 dark:bg-gray-700 rounded-lg">
            <FileDown className="w-5 h-5 text-gray-600 dark:text-gray-400" />
            <div>
              <div className="font-medium text-gray-900 dark:text-white">Data Cleaning Steps</div>
              <div className="text-sm text-gray-600 dark:text-gray-400">
                Detailed list of all preprocessing operations performed
              </div>
            </div>
          </div>
          <div className="flex items-center gap-3 p-3 bg-gray-50 dark:bg-gray-700 rounded-lg">
            <FileDown className="w-5 h-5 text-gray-600 dark:text-gray-400" />
            <div>
              <div className="font-medium text-gray-900 dark:text-white">Model Performance Metrics</div>
              <div className="text-sm text-gray-600 dark:text-gray-400">
                Accuracy, Precision, Recall, F1 Score, and AUC metrics
              </div>
            </div>
          </div>
          <div className="flex items-center gap-3 p-3 bg-gray-50 dark:bg-gray-700 rounded-lg">
            <FileDown className="w-5 h-5 text-gray-600 dark:text-gray-400" />
            <div>
              <div className="font-medium text-gray-900 dark:text-white">AI Insights & Recommendations</div>
              <div className="text-sm text-gray-600 dark:text-gray-400">
                AI-generated summary with actionable recommendations
              </div>
            </div>
          </div>
        </div>
      </motion.div>

      {/* Usage Instructions */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="mt-6 bg-blue-50 dark:bg-blue-900 border border-blue-200 dark:border-blue-700 rounded-lg p-6"
      >
        <h3 className="font-semibold text-blue-900 dark:text-blue-200 mb-3">
          How to Use Your Exported Model
        </h3>
        <div className="space-y-2 text-sm text-blue-800 dark:text-blue-300">
          <p>
            <strong>1. Load the model:</strong> Use <code className="bg-blue-100 dark:bg-blue-800 px-1 rounded">joblib.load('trained_model.pkl')</code> in Python
          </p>
          <p>
            <strong>2. Make predictions:</strong> Call <code className="bg-blue-100 dark:bg-blue-800 px-1 rounded">model.predict(X_new)</code> with your new data
          </p>
          <p>
            <strong>3. Deploy:</strong> Use the model in production with Flask, FastAPI, or cloud services
          </p>
        </div>
      </motion.div>
    </div>
  )
}

