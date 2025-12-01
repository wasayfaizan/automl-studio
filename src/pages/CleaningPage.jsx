import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { Sparkles, CheckCircle2, Download, ArrowRight } from 'lucide-react'
import axios from 'axios'
import LoadingSpinner from '../components/LoadingSpinner'

export default function CleaningPage() {
  const [analysis, setAnalysis] = useState(null)
  const [cleaning, setCleaning] = useState(false)
  const [cleanedData, setCleanedData] = useState(null)
  const [targetColumn, setTargetColumn] = useState('')

  useEffect(() => {
    const saved = localStorage.getItem('datasetAnalysis')
    if (saved) {
      const data = JSON.parse(saved)
      setAnalysis(data)
      // Auto-detect target (last column or column with 'target' in name)
      const cols = data.column_names
      const target = cols.find(c => c.toLowerCase().includes('target')) || 
                     cols.find(c => c.toLowerCase().includes('label')) ||
                     cols[cols.length - 1]
      setTargetColumn(target || '')
    }
  }, [])

  const handleClean = async () => {
    if (!analysis) return

    setCleaning(true)
    try {
      const response = await axios.post('/api/clean', {
        target_column: targetColumn || null,
        handle_missing: 'auto',
        encode_categoricals: true,
        normalize: true,
        remove_outliers: true
      })
      setCleanedData(response.data)
      localStorage.setItem('cleanedData', JSON.stringify(response.data))
    } catch (error) {
      console.error('Cleaning failed:', error)
      alert('Failed to clean data: ' + (error.response?.data?.detail || error.message))
    } finally {
      setCleaning(false)
    }
  }

  const handleDownloadCleaned = async () => {
    try {
      const response = await axios.get('/api/download/cleaned-data', {
        responseType: 'blob'
      })
      const url = window.URL.createObjectURL(new Blob([response.data]))
      const link = document.createElement('a')
      link.href = url
      link.setAttribute('download', 'cleaned_dataset.csv')
      document.body.appendChild(link)
      link.click()
      link.remove()
    } catch (error) {
      alert('Failed to download cleaned dataset')
    }
  }

  if (!analysis) {
    return (
      <div className="p-8 max-w-7xl mx-auto">
        <div className="bg-yellow-50 dark:bg-yellow-900 border border-yellow-200 dark:border-yellow-700 rounded-lg p-6">
          <p className="text-yellow-800 dark:text-yellow-200">
            Please upload a dataset first on the Upload page.
          </p>
        </div>
      </div>
    )
  }

  return (
    <div className="p-8 max-w-7xl mx-auto">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="mb-8"
      >
        <h1 className="text-3xl font-bold text-gray-900 dark:text-white mb-2">
          AI Data Cleaning
        </h1>
        <p className="text-gray-600 dark:text-gray-400">
          Automatically clean and preprocess your dataset
        </p>
      </motion.div>

      {!cleanedData ? (
        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-8 border border-gray-200 dark:border-gray-700"
        >
          <div className="mb-6">
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
              Target Column (optional)
            </label>
            <select
              value={targetColumn}
              onChange={(e) => setTargetColumn(e.target.value)}
              className="w-full px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
            >
              <option value="">None (Unsupervised)</option>
              {analysis.column_names.map((col) => (
                <option key={col} value={col}>
                  {col}
                </option>
              ))}
            </select>
          </div>

          <div className="mb-6 p-4 bg-blue-50 dark:bg-blue-900 rounded-lg">
            <h3 className="font-semibold text-gray-900 dark:text-white mb-2">
              Cleaning Operations
            </h3>
            <ul className="space-y-2 text-sm text-gray-700 dark:text-gray-300">
              <li className="flex items-center gap-2">
                <CheckCircle2 className="w-4 h-4 text-green-600" />
                Handle missing values (median for numeric, mode for categorical)
              </li>
              <li className="flex items-center gap-2">
                <CheckCircle2 className="w-4 h-4 text-green-600" />
                Encode categorical variables (One-Hot or Label Encoding)
              </li>
              <li className="flex items-center gap-2">
                <CheckCircle2 className="w-4 h-4 text-green-600" />
                Normalize numerical columns (StandardScaler)
              </li>
              <li className="flex items-center gap-2">
                <CheckCircle2 className="w-4 h-4 text-green-600" />
                Remove outliers using IQR method
              </li>
            </ul>
          </div>

          <button
            onClick={handleClean}
            disabled={cleaning}
            className="w-full px-6 py-3 bg-primary-600 hover:bg-primary-700 disabled:bg-gray-400 text-white rounded-lg font-medium transition-colors flex items-center justify-center gap-2"
          >
            {cleaning ? (
              <>
                <LoadingSpinner size="sm" />
                Cleaning Data...
              </>
            ) : (
              <>
                <Sparkles className="w-5 h-5" />
                Start AI Cleaning
              </>
            )}
          </button>
        </motion.div>
      ) : (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="space-y-6"
        >
          {/* Success Message */}
          <div className="bg-green-50 dark:bg-green-900 border border-green-200 dark:border-green-700 rounded-lg p-6">
            <div className="flex items-center gap-3">
              <CheckCircle2 className="w-6 h-6 text-green-600 dark:text-green-400" />
              <div>
                <h3 className="font-semibold text-green-900 dark:text-green-200">
                  Data Cleaning Complete!
                </h3>
                <p className="text-green-700 dark:text-green-300 text-sm mt-1">
                  Your dataset has been successfully cleaned and preprocessed.
                </p>
              </div>
            </div>
          </div>

          {/* Before/After Comparison */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow border border-gray-200 dark:border-gray-700">
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
                Before Cleaning
              </h3>
              <div className="space-y-3">
                <div className="flex justify-between">
                  <span className="text-gray-600 dark:text-gray-400">Rows:</span>
                  <span className="font-medium text-gray-900 dark:text-white">
                    {cleanedData.comparison.before.rows.toLocaleString()}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600 dark:text-gray-400">Columns:</span>
                  <span className="font-medium text-gray-900 dark:text-white">
                    {cleanedData.comparison.before.columns}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600 dark:text-gray-400">Missing Values:</span>
                  <span className="font-medium text-red-600 dark:text-red-400">
                    {cleanedData.comparison.before.missing_values}
                  </span>
                </div>
              </div>
            </div>

            <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow border border-gray-200 dark:border-gray-700">
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
                After Cleaning
              </h3>
              <div className="space-y-3">
                <div className="flex justify-between">
                  <span className="text-gray-600 dark:text-gray-400">Rows:</span>
                  <span className="font-medium text-green-600 dark:text-green-400">
                    {cleanedData.comparison.after.rows.toLocaleString()}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600 dark:text-gray-400">Columns:</span>
                  <span className="font-medium text-green-600 dark:text-green-400">
                    {cleanedData.comparison.after.columns}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600 dark:text-gray-400">Missing Values:</span>
                  <span className="font-medium text-green-600 dark:text-green-400">
                    {cleanedData.comparison.after.missing_values}
                  </span>
                </div>
              </div>
            </div>
          </div>

          {/* Cleaning Steps */}
          <div className="bg-white dark:bg-gray-800 rounded-lg shadow border border-gray-200 dark:border-gray-700 p-6">
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
              Cleaning Steps Performed
            </h3>
            <div className="space-y-2">
              {cleanedData.comparison.steps.map((step, idx) => (
                <div
                  key={idx}
                  className="flex items-start gap-3 p-3 bg-gray-50 dark:bg-gray-700 rounded-lg"
                >
                  <div className="w-6 h-6 rounded-full bg-primary-600 text-white flex items-center justify-center text-sm font-medium flex-shrink-0">
                    {idx + 1}
                  </div>
                  <p className="text-gray-700 dark:text-gray-300">{step}</p>
                </div>
              ))}
            </div>
          </div>

          {/* Preview */}
          <div className="bg-white dark:bg-gray-800 rounded-lg shadow border border-gray-200 dark:border-gray-700 p-6">
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
              Cleaned Data Preview
            </h3>
            <div className="overflow-x-auto">
              <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
                <thead className="bg-gray-50 dark:bg-gray-700">
                  <tr>
                    {cleanedData.preview[0] && Object.keys(cleanedData.preview[0]).map((col) => (
                      <th
                        key={col}
                        className="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase"
                      >
                        {col}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody className="bg-white dark:bg-gray-800 divide-y divide-gray-200 dark:divide-gray-700">
                  {cleanedData.preview.slice(0, 10).map((row, idx) => (
                    <tr key={idx}>
                      {Object.values(row).map((val, i) => (
                        <td
                          key={i}
                          className="px-4 py-3 text-sm text-gray-700 dark:text-gray-300"
                        >
                          {typeof val === 'number' ? val.toFixed(3) : String(val).substring(0, 20)}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Download Button */}
          <div className="flex justify-end">
            <button
              onClick={handleDownloadCleaned}
              className="px-6 py-3 bg-primary-600 hover:bg-primary-700 text-white rounded-lg font-medium transition-colors flex items-center gap-2"
            >
              <Download className="w-5 h-5" />
              Download Cleaned Dataset
            </button>
          </div>

          {/* Next Step */}
          <div className="bg-blue-50 dark:bg-blue-900 border border-blue-200 dark:border-blue-700 rounded-lg p-6">
            <div className="flex items-center justify-between">
              <div>
                <h3 className="font-semibold text-blue-900 dark:text-blue-200 mb-1">
                  Ready for EDA
                </h3>
                <p className="text-blue-700 dark:text-blue-300 text-sm">
                  Proceed to the EDA Report page to explore your cleaned data.
                </p>
              </div>
              <a
                href="/eda"
                className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg font-medium transition-colors flex items-center gap-2"
              >
                Next: EDA Report
                <ArrowRight className="w-4 h-4" />
              </a>
            </div>
          </div>
        </motion.div>
      )}
    </div>
  )
}

