import { useState } from 'react'
import { motion } from 'framer-motion'
import { Upload, FileText, AlertCircle, CheckCircle2 } from 'lucide-react'
import axios from 'axios'
import LoadingSpinner from '../components/LoadingSpinner'

export default function UploadPage() {
  const [file, setFile] = useState(null)
  const [uploading, setUploading] = useState(false)
  const [analysis, setAnalysis] = useState(null)
  const [error, setError] = useState(null)

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0]
    if (selectedFile && selectedFile.type === 'text/csv') {
      setFile(selectedFile)
      setError(null)
    } else {
      setError('Please upload a valid CSV file')
    }
  }

  const handleUpload = async () => {
    if (!file) {
      setError('Please select a file first')
      return
    }

    setUploading(true)
    setError(null)

    const formData = new FormData()
    formData.append('file', file)

    try {
      const response = await axios.post('/api/upload', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      })
      setAnalysis(response.data)
      localStorage.setItem('datasetAnalysis', JSON.stringify(response.data))
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to upload file')
    } finally {
      setUploading(false)
    }
  }

  return (
    <div className="p-8 max-w-7xl mx-auto">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="mb-8"
      >
        <div className="flex items-center gap-3 mb-2">
          <div className="p-3 bg-gradient-to-br from-primary-500 to-accent-500 rounded-xl shadow-lg">
            <Upload className="w-6 h-6 text-white" />
          </div>
          <div>
            <h1 className="text-4xl font-bold bg-gradient-to-r from-primary-600 to-accent-600 dark:from-primary-400 dark:to-accent-400 bg-clip-text text-transparent">
              Upload Dataset
            </h1>
            <p className="text-gray-600 dark:text-gray-400 mt-1 text-lg">
              Upload your CSV file to begin the AutoML pipeline
            </p>
          </div>
        </div>
      </motion.div>

      {!analysis ? (
        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          className="bg-white/90 dark:bg-gray-800/90 backdrop-blur-xl rounded-2xl shadow-2xl p-8 border border-gray-200/50 dark:border-gray-700/50"
        >
          <div className="flex flex-col items-center justify-center py-12">
            <div className="mb-6 p-6 bg-gradient-to-br from-primary-500 to-accent-500 rounded-2xl shadow-xl transform hover:scale-110 transition-transform duration-300">
              <Upload className="w-12 h-12 text-white" />
            </div>

            <input
              type="file"
              accept=".csv"
              onChange={handleFileChange}
              className="hidden"
              id="file-upload"
            />
            <label
              htmlFor="file-upload"
              className="cursor-pointer mb-4 px-8 py-4 bg-gradient-to-r from-primary-500 to-accent-500 hover:from-primary-600 hover:to-accent-600 text-white rounded-xl font-semibold transition-all duration-200 shadow-lg hover:shadow-xl transform hover:scale-105"
            >
              Select CSV File
            </label>

            {file && (
              <div className="mb-6 flex items-center gap-2 text-gray-700 dark:text-gray-300">
                <FileText size={20} />
                <span>{file.name}</span>
              </div>
            )}

            {error && (
              <div className="mb-4 flex items-center gap-2 text-red-600 dark:text-red-400">
                <AlertCircle size={20} />
                <span>{error}</span>
              </div>
            )}

            <button
              onClick={handleUpload}
              disabled={!file || uploading}
              className="px-8 py-4 bg-gradient-to-r from-primary-500 to-accent-500 hover:from-primary-600 hover:to-accent-600 disabled:from-gray-400 disabled:to-gray-500 text-white rounded-xl font-semibold transition-all duration-200 shadow-lg hover:shadow-xl transform hover:scale-105 disabled:transform-none disabled:cursor-not-allowed flex items-center gap-2"
            >
              {uploading ? (
                <>
                  <LoadingSpinner size="sm" />
                  Uploading...
                </>
              ) : (
                'Upload & Analyze'
              )}
            </button>
          </div>
        </motion.div>
      ) : (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="space-y-6"
        >
          {/* AI Summary */}
          <div className="bg-gradient-to-r from-primary-50 via-accent-50 to-blue-50 dark:from-primary-900/30 dark:via-accent-900/30 dark:to-blue-900/30 rounded-2xl p-6 border border-primary-200/50 dark:border-primary-700/50 backdrop-blur-sm shadow-lg">
            <div className="flex items-start gap-3">
              <CheckCircle2 className="w-6 h-6 text-primary-600 dark:text-primary-400 mt-1" />
              <div>
                <h3 className="font-semibold text-gray-900 dark:text-white mb-2">AI Summary</h3>
                <p className="text-gray-700 dark:text-gray-300">{analysis.ai_summary}</p>
              </div>
            </div>
          </div>

          {/* Dataset Overview */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <motion.div 
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.1 }}
              className="bg-white/90 dark:bg-gray-800/90 backdrop-blur-xl rounded-2xl p-6 shadow-xl border border-gray-200/50 dark:border-gray-700/50 hover:shadow-2xl transform hover:scale-105 transition-all duration-300"
            >
              <div className="text-4xl font-bold bg-gradient-to-r from-primary-600 to-accent-600 dark:from-primary-400 dark:to-accent-400 bg-clip-text text-transparent mb-2">
                {analysis.rows.toLocaleString()}
              </div>
              <div className="text-gray-600 dark:text-gray-400 font-medium">Total Rows</div>
            </motion.div>
            <motion.div 
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.2 }}
              className="bg-white/90 dark:bg-gray-800/90 backdrop-blur-xl rounded-2xl p-6 shadow-xl border border-gray-200/50 dark:border-gray-700/50 hover:shadow-2xl transform hover:scale-105 transition-all duration-300"
            >
              <div className="text-4xl font-bold bg-gradient-to-r from-primary-600 to-accent-600 dark:from-primary-400 dark:to-accent-400 bg-clip-text text-transparent mb-2">
                {analysis.columns}
              </div>
              <div className="text-gray-600 dark:text-gray-400 font-medium">Total Columns</div>
            </motion.div>
            <motion.div 
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.3 }}
              className="bg-white/90 dark:bg-gray-800/90 backdrop-blur-xl rounded-2xl p-6 shadow-xl border border-gray-200/50 dark:border-gray-700/50 hover:shadow-2xl transform hover:scale-105 transition-all duration-300"
            >
              <div className="text-4xl font-bold bg-gradient-to-r from-primary-600 to-accent-600 dark:from-primary-400 dark:to-accent-400 bg-clip-text text-transparent mb-2">
                {Object.values(analysis.missing_values).reduce((a, b) => a + b, 0).toLocaleString()}
              </div>
              <div className="text-gray-600 dark:text-gray-400 font-medium">Missing Values</div>
            </motion.div>
          </div>

          {/* Column Information */}
          <div className="bg-white/90 dark:bg-gray-800/90 backdrop-blur-xl rounded-2xl shadow-xl border border-gray-200/50 dark:border-gray-700/50 p-6">
            <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-4">
              Column Information
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <h4 className="font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Numeric Columns ({analysis.numeric_columns.length})
                </h4>
                <div className="flex flex-wrap gap-2">
                  {analysis.numeric_columns.map((col) => (
                    <span
                      key={col}
                      className="px-3 py-1 bg-blue-100 dark:bg-blue-900 text-blue-800 dark:text-blue-200 rounded-full text-sm"
                    >
                      {col}
                    </span>
                  ))}
                </div>
              </div>
              <div>
                <h4 className="font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Categorical Columns ({analysis.categorical_columns.length})
                </h4>
                <div className="flex flex-wrap gap-2">
                  {analysis.categorical_columns.map((col) => (
                    <span
                      key={col}
                      className="px-3 py-1 bg-green-100 dark:bg-green-900 text-green-800 dark:text-green-200 rounded-full text-sm"
                    >
                      {col}
                    </span>
                  ))}
                </div>
              </div>
            </div>
          </div>

          {/* Missing Values */}
          {Object.keys(analysis.missing_values).some(k => analysis.missing_values[k] > 0) && (
            <div className="bg-white dark:bg-gray-800 rounded-lg shadow border border-gray-200 dark:border-gray-700 p-6">
              <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-4">
                Missing Values
              </h3>
              <div className="space-y-2">
                {Object.entries(analysis.missing_values)
                  .filter(([_, count]) => count > 0)
                  .map(([col, count]) => (
                    <div key={col} className="flex justify-between items-center">
                      <span className="text-gray-700 dark:text-gray-300">{col}</span>
                      <span className="text-red-600 dark:text-red-400 font-medium">
                        {count} ({analysis.missing_percentage[col]}%)
                      </span>
                    </div>
                  ))}
              </div>
            </div>
          )}

          {/* Outliers */}
          {Object.keys(analysis.outliers_detected).length > 0 && (
            <div className="bg-white dark:bg-gray-800 rounded-lg shadow border border-gray-200 dark:border-gray-700 p-6">
              <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-4">
                Outliers Detected
              </h3>
              <div className="space-y-2">
                {Object.entries(analysis.outliers_detected).map(([col, count]) => (
                  <div key={col} className="flex justify-between items-center">
                    <span className="text-gray-700 dark:text-gray-300">{col}</span>
                    <span className="text-orange-600 dark:text-orange-400 font-medium">
                      {count} outliers
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Data Preview */}
          <div className="bg-white dark:bg-gray-800 rounded-lg shadow border border-gray-200 dark:border-gray-700 p-6">
            <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-4">
              Data Preview (First 20 rows)
            </h3>
            <div className="overflow-x-auto">
              <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
                <thead className="bg-gray-50 dark:bg-gray-700">
                  <tr>
                    {analysis.column_names.map((col) => (
                      <th
                        key={col}
                        className="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider"
                      >
                        {col}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody className="bg-white dark:bg-gray-800 divide-y divide-gray-200 dark:divide-gray-700">
                  {analysis.preview.map((row, idx) => (
                    <tr key={idx}>
                      {analysis.column_names.map((col) => (
                        <td
                          key={col}
                          className="px-4 py-3 text-sm text-gray-700 dark:text-gray-300"
                        >
                          {row[col]?.toString().substring(0, 30) || 'N/A'}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </motion.div>
      )}
    </div>
  )
}

