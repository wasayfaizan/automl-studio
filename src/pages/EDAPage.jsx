import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { BarChart3, RefreshCw, Lightbulb } from 'lucide-react'
import axios from 'axios'
import LoadingSpinner from '../components/LoadingSpinner'

export default function EDAPage() {
  const [loading, setLoading] = useState(false)
  const [edaData, setEdaData] = useState(null)
  const [error, setError] = useState(null)

  useEffect(() => {
    loadEDA()
  }, [])

  const loadEDA = async () => {
    setLoading(true)
    setError(null)
    try {
      const response = await axios.get('/api/eda')
      setEdaData(response.data)
      localStorage.setItem('edaData', JSON.stringify(response.data))
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to generate EDA report')
    } finally {
      setLoading(false)
    }
  }

  if (loading) {
    return (
      <div className="p-8 max-w-7xl mx-auto flex items-center justify-center min-h-[60vh]">
        <div className="text-center">
          <LoadingSpinner size="lg" />
          <p className="mt-4 text-gray-600 dark:text-gray-400">
            Generating EDA Report...
          </p>
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="p-8 max-w-7xl mx-auto">
        <div className="bg-red-50 dark:bg-red-900 border border-red-200 dark:border-red-700 rounded-lg p-6">
          <p className="text-red-800 dark:text-red-200">{error}</p>
          <p className="text-red-700 dark:text-red-300 text-sm mt-2">
            Please make sure you've cleaned your data first.
          </p>
        </div>
      </div>
    )
  }

  if (!edaData) {
    return (
      <div className="p-8 max-w-7xl mx-auto">
        <div className="bg-yellow-50 dark:bg-yellow-900 border border-yellow-200 dark:border-yellow-700 rounded-lg p-6">
          <p className="text-yellow-800 dark:text-yellow-200">
            No EDA data available. Please clean your data first.
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
        className="mb-8 flex items-center justify-between"
      >
        <div className="flex items-center gap-3">
          <div className="p-3 bg-gradient-to-br from-primary-500 to-accent-500 rounded-xl shadow-lg">
            <BarChart3 className="w-6 h-6 text-white" />
          </div>
          <div>
            <h1 className="text-4xl font-bold bg-gradient-to-r from-primary-600 to-accent-600 dark:from-primary-400 dark:to-accent-400 bg-clip-text text-transparent">
              EDA Report
            </h1>
            <p className="text-gray-600 dark:text-gray-400 mt-1 text-lg">
              Comprehensive Exploratory Data Analysis with AI-generated insights
            </p>
          </div>
        </div>
        <button
          onClick={loadEDA}
          className="px-6 py-3 bg-gradient-to-r from-primary-500 to-accent-500 hover:from-primary-600 hover:to-accent-600 text-white rounded-xl font-semibold transition-all duration-200 shadow-lg hover:shadow-xl transform hover:scale-105 flex items-center gap-2"
        >
          <RefreshCw className="w-4 h-4" />
          Regenerate
        </button>
      </motion.div>

      {/* AI Summary */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
          className="mb-8 bg-gradient-to-r from-purple-50 via-pink-50 to-indigo-50 dark:from-purple-900/30 dark:via-pink-900/30 dark:to-indigo-900/30 rounded-2xl p-6 border border-purple-200/50 dark:border-purple-700/50 backdrop-blur-sm shadow-lg"
      >
        <div className="flex items-start gap-3">
          <Lightbulb className="w-6 h-6 text-purple-600 dark:text-purple-400 mt-1" />
          <div>
            <h3 className="font-semibold text-gray-900 dark:text-white mb-2">AI EDA Summary</h3>
            <div className="text-gray-700 dark:text-gray-300 whitespace-pre-line">
              {edaData.summary}
            </div>
          </div>
        </div>
      </motion.div>

      {/* Correlation Heatmap */}
      {edaData.charts.correlation && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-6 bg-white/90 dark:bg-gray-800/90 backdrop-blur-xl rounded-2xl shadow-xl border border-gray-200/50 dark:border-gray-700/50 p-6"
        >
          <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-4">
            Correlation Heatmap
          </h3>
          <div className="flex justify-center">
            <img
              src={`data:image/png;base64,${edaData.charts.correlation}`}
              alt="Correlation Heatmap"
              className="max-w-full h-auto rounded-lg"
            />
          </div>
        </motion.div>
      )}

      {/* Histograms */}
      {edaData.charts.histograms && Object.keys(edaData.charts.histograms).length > 0 && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-6 bg-white/90 dark:bg-gray-800/90 backdrop-blur-xl rounded-2xl shadow-xl border border-gray-200/50 dark:border-gray-700/50 p-6"
        >
          <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-4">
            Numeric Column Distributions
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {Object.entries(edaData.charts.histograms).map(([col, img]) => (
              <div key={col} className="flex flex-col items-center">
                <img
                  src={`data:image/png;base64,${img}`}
                  alt={`Histogram of ${col}`}
                  className="max-w-full h-auto rounded-lg"
                />
                <p className="mt-2 text-sm text-gray-600 dark:text-gray-400">{col}</p>
              </div>
            ))}
          </div>
        </motion.div>
      )}

      {/* Box Plots */}
      {edaData.charts.boxplots && Object.keys(edaData.charts.boxplots).length > 0 && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-6 bg-white/90 dark:bg-gray-800/90 backdrop-blur-xl rounded-2xl shadow-xl border border-gray-200/50 dark:border-gray-700/50 p-6"
        >
          <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-4">
            Box Plots (Outlier Detection)
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {Object.entries(edaData.charts.boxplots).map(([col, img]) => (
              <div key={col} className="flex flex-col items-center">
                <img
                  src={`data:image/png;base64,${img}`}
                  alt={`Box plot of ${col}`}
                  className="max-w-full h-auto rounded-lg"
                />
                <p className="mt-2 text-sm text-gray-600 dark:text-gray-400">{col}</p>
              </div>
            ))}
          </div>
        </motion.div>
      )}

      {/* Categorical Charts */}
      {edaData.charts.categorical && Object.keys(edaData.charts.categorical).length > 0 && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-6 bg-white/90 dark:bg-gray-800/90 backdrop-blur-xl rounded-2xl shadow-xl border border-gray-200/50 dark:border-gray-700/50 p-6"
        >
          <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-4">
            Categorical Column Distributions
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {Object.entries(edaData.charts.categorical).map(([col, img]) => (
              <div key={col} className="flex flex-col items-center">
                <img
                  src={`data:image/png;base64,${img}`}
                  alt={`Distribution of ${col}`}
                  className="max-w-full h-auto rounded-lg"
                />
                <p className="mt-2 text-sm text-gray-600 dark:text-gray-400">{col}</p>
              </div>
            ))}
          </div>
        </motion.div>
      )}

      {/* Target Distribution */}
      {edaData.charts.target_distribution && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-6 bg-white/90 dark:bg-gray-800/90 backdrop-blur-xl rounded-2xl shadow-xl border border-gray-200/50 dark:border-gray-700/50 p-6"
        >
          <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-4">
            Target Distribution
          </h3>
          <div className="flex justify-center">
            <img
              src={`data:image/png;base64,${edaData.charts.target_distribution}`}
              alt="Target Distribution"
              className="max-w-full h-auto rounded-lg"
            />
          </div>
        </motion.div>
      )}

      {/* Data Quality Metrics */}
      {edaData.quality_metrics && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-6 bg-white/90 dark:bg-gray-800/90 backdrop-blur-xl rounded-2xl shadow-xl border border-gray-200/50 dark:border-gray-700/50 p-6"
        >
          <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-4">
            Data Quality Metrics
          </h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="p-4 bg-gradient-to-br from-blue-50 to-cyan-50 dark:from-blue-900/30 dark:to-cyan-900/30 rounded-xl">
              <div className="text-2xl font-bold text-blue-600 dark:text-blue-400">
                {edaData.quality_metrics.total_rows?.toLocaleString()}
              </div>
              <div className="text-sm text-gray-600 dark:text-gray-400">Total Rows</div>
            </div>
            <div className="p-4 bg-gradient-to-br from-green-50 to-emerald-50 dark:from-green-900/30 dark:to-emerald-900/30 rounded-xl">
              <div className="text-2xl font-bold text-green-600 dark:text-green-400">
                {edaData.quality_metrics.total_columns}
              </div>
              <div className="text-sm text-gray-600 dark:text-gray-400">Total Columns</div>
            </div>
            <div className="p-4 bg-gradient-to-br from-purple-50 to-pink-50 dark:from-purple-900/30 dark:to-pink-900/30 rounded-xl">
              <div className="text-2xl font-bold text-purple-600 dark:text-purple-400">
                {edaData.quality_metrics.duplicate_rows || 0}
              </div>
              <div className="text-sm text-gray-600 dark:text-gray-400">Duplicate Rows</div>
            </div>
            <div className="p-4 bg-gradient-to-br from-orange-50 to-amber-50 dark:from-orange-900/30 dark:to-amber-900/30 rounded-xl">
              <div className="text-2xl font-bold text-orange-600 dark:text-orange-400">
                {edaData.quality_metrics.memory_usage_mb?.toFixed(2) || '0'} MB
              </div>
              <div className="text-sm text-gray-600 dark:text-gray-400">Memory Usage</div>
            </div>
          </div>
        </motion.div>
      )}

      {/* Extended Statistics Table */}
      {edaData.statistics?.extended_stats && Object.keys(edaData.statistics.extended_stats).length > 0 && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-6 bg-white/90 dark:bg-gray-800/90 backdrop-blur-xl rounded-2xl shadow-xl border border-gray-200/50 dark:border-gray-700/50 p-6"
        >
          <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-4">
            Extended Statistics (Skewness & Kurtosis)
          </h3>
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
              <thead className="bg-gradient-to-r from-primary-500 to-accent-500">
                <tr>
                  <th className="px-4 py-3 text-left text-xs font-semibold text-white uppercase">Column</th>
                  <th className="px-4 py-3 text-left text-xs font-semibold text-white uppercase">Mean</th>
                  <th className="px-4 py-3 text-left text-xs font-semibold text-white uppercase">Median</th>
                  <th className="px-4 py-3 text-left text-xs font-semibold text-white uppercase">Std Dev</th>
                  <th className="px-4 py-3 text-left text-xs font-semibold text-white uppercase">Skewness</th>
                  <th className="px-4 py-3 text-left text-xs font-semibold text-white uppercase">Kurtosis</th>
                  <th className="px-4 py-3 text-left text-xs font-semibold text-white uppercase">IQR</th>
                </tr>
              </thead>
              <tbody className="bg-white dark:bg-gray-800 divide-y divide-gray-200 dark:divide-gray-700">
                {Object.entries(edaData.statistics.extended_stats).slice(0, 10).map(([col, stats]) => (
                  <tr key={col} className="hover:bg-gray-50 dark:hover:bg-gray-700">
                    <td className="px-4 py-3 text-sm font-medium text-gray-900 dark:text-white">{col}</td>
                    <td className="px-4 py-3 text-sm text-gray-600 dark:text-gray-400">{stats.mean?.toFixed(3)}</td>
                    <td className="px-4 py-3 text-sm text-gray-600 dark:text-gray-400">{stats.median?.toFixed(3)}</td>
                    <td className="px-4 py-3 text-sm text-gray-600 dark:text-gray-400">{stats.std?.toFixed(3)}</td>
                    <td className="px-4 py-3 text-sm text-gray-600 dark:text-gray-400">{stats.skewness?.toFixed(3)}</td>
                    <td className="px-4 py-3 text-sm text-gray-600 dark:text-gray-400">{stats.kurtosis?.toFixed(3)}</td>
                    <td className="px-4 py-3 text-sm text-gray-600 dark:text-gray-400">{stats.iqr?.toFixed(3)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </motion.div>
      )}

      {/* Outlier Analysis */}
      {edaData.outlier_analysis && Object.keys(edaData.outlier_analysis).length > 0 && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-6 bg-white/90 dark:bg-gray-800/90 backdrop-blur-xl rounded-2xl shadow-xl border border-gray-200/50 dark:border-gray-700/50 p-6"
        >
          <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-4">
            Outlier Analysis
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {Object.entries(edaData.outlier_analysis).map(([col, analysis]) => (
              <div key={col} className="p-4 bg-gradient-to-br from-red-50 to-orange-50 dark:from-red-900/20 dark:to-orange-900/20 rounded-xl border border-red-200/50 dark:border-red-700/50">
                <div className="font-semibold text-gray-900 dark:text-white mb-2">{col}</div>
                <div className="text-2xl font-bold text-red-600 dark:text-red-400 mb-1">
                  {analysis.count} ({analysis.percentage?.toFixed(1)}%)
                </div>
                <div className="text-xs text-gray-600 dark:text-gray-400">
                  Range: [{analysis.lower_bound?.toFixed(2)}, {analysis.upper_bound?.toFixed(2)}]
                </div>
              </div>
            ))}
          </div>
        </motion.div>
      )}

      {/* Violin Plots */}
      {edaData.charts.violin_plots && Object.keys(edaData.charts.violin_plots).length > 0 && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-6 bg-white/90 dark:bg-gray-800/90 backdrop-blur-xl rounded-2xl shadow-xl border border-gray-200/50 dark:border-gray-700/50 p-6"
        >
          <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-4">
            Violin Plots (Distribution Shape)
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {Object.entries(edaData.charts.violin_plots).map(([col, img]) => (
              <div key={col} className="flex flex-col items-center">
                <img
                  src={`data:image/png;base64,${img}`}
                  alt={`Violin plot of ${col}`}
                  className="max-w-full h-auto rounded-lg"
                />
                <p className="mt-2 text-sm text-gray-600 dark:text-gray-400 font-medium">{col}</p>
              </div>
            ))}
          </div>
        </motion.div>
      )}

      {/* Pair Plot */}
      {edaData.charts.pairplot && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-6 bg-white/90 dark:bg-gray-800/90 backdrop-blur-xl rounded-2xl shadow-xl border border-gray-200/50 dark:border-gray-700/50 p-6"
        >
          <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-4">
            Pair Plot (Feature Relationships)
          </h3>
          <div className="flex justify-center">
            <img
              src={`data:image/png;base64,${edaData.charts.pairplot}`}
              alt="Pair Plot"
              className="max-w-full h-auto rounded-lg"
            />
          </div>
        </motion.div>
      )}

      {/* Scatter Plots */}
      {edaData.charts.scatter_plots && Object.keys(edaData.charts.scatter_plots).length > 0 && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-6 bg-white/90 dark:bg-gray-800/90 backdrop-blur-xl rounded-2xl shadow-xl border border-gray-200/50 dark:border-gray-700/50 p-6"
        >
          <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-4">
            Scatter Plots (Feature Correlations)
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {Object.entries(edaData.charts.scatter_plots).map(([key, img]) => (
              <div key={key} className="flex flex-col items-center">
                <img
                  src={`data:image/png;base64,${img}`}
                  alt={`Scatter plot ${key}`}
                  className="max-w-full h-auto rounded-lg"
                />
                <p className="mt-2 text-sm text-gray-600 dark:text-gray-400 font-medium">{key.replace('_vs_', ' vs ')}</p>
              </div>
            ))}
          </div>
        </motion.div>
      )}

      {/* Distribution Comparison */}
      {edaData.charts.distribution_comparison && Object.keys(edaData.charts.distribution_comparison).length > 0 && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-6 bg-white/90 dark:bg-gray-800/90 backdrop-blur-xl rounded-2xl shadow-xl border border-gray-200/50 dark:border-gray-700/50 p-6"
        >
          <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-4">
            Distribution Comparison by Target
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {Object.entries(edaData.charts.distribution_comparison).map(([col, img]) => (
              <div key={col} className="flex flex-col items-center">
                <img
                  src={`data:image/png;base64,${img}`}
                  alt={`Distribution comparison for ${col}`}
                  className="max-w-full h-auto rounded-lg"
                />
                <p className="mt-2 text-sm text-gray-600 dark:text-gray-400 font-medium">{col}</p>
              </div>
            ))}
          </div>
        </motion.div>
      )}
    </div>
  )
}

