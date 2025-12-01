import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { TrendingUp, Lightbulb, CheckCircle2 } from 'lucide-react'
import axios from 'axios'

export default function ResultsPage() {
  const [results, setResults] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const saved = localStorage.getItem('modelResults')
    if (saved) {
      setResults(JSON.parse(saved))
    }
    setLoading(false)
  }, [])

  if (loading) {
    return (
      <div className="p-8 max-w-7xl mx-auto flex items-center justify-center min-h-[60vh]">
        <p className="text-gray-600 dark:text-gray-400">Loading results...</p>
      </div>
    )
  }

  if (!results) {
    return (
      <div className="p-8 max-w-7xl mx-auto">
        <div className="bg-yellow-50 dark:bg-yellow-900 border border-yellow-200 dark:border-yellow-700 rounded-lg p-6">
          <p className="text-yellow-800 dark:text-yellow-200">
            No model results available. Please train a model first.
          </p>
        </div>
      </div>
    )
  }

  const metrics = results.metrics
  const visualizations = results.visualizations

  return (
    <div className="p-8 max-w-7xl mx-auto">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="mb-8"
      >
        <div className="flex items-center gap-3 mb-2">
          <div className="p-3 bg-gradient-to-br from-primary-500 to-accent-500 rounded-xl shadow-lg">
            <TrendingUp className="w-6 h-6 text-white" />
          </div>
          <div>
            <h1 className="text-4xl font-bold bg-gradient-to-r from-primary-600 to-accent-600 dark:from-primary-400 dark:to-accent-400 bg-clip-text text-transparent">
              Results & Metrics
            </h1>
            <p className="text-gray-600 dark:text-gray-400 mt-1 text-lg">
              Model performance evaluation and insights
            </p>
          </div>
        </div>
      </motion.div>

      {/* AI Insights */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="mb-8 bg-gradient-to-r from-green-50 via-emerald-50 to-teal-50 dark:from-green-900/30 dark:via-emerald-900/30 dark:to-teal-900/30 rounded-2xl p-6 border border-green-200/50 dark:border-green-700/50 backdrop-blur-sm shadow-lg"
      >
        <div className="flex items-start gap-3">
          <Lightbulb className="w-6 h-6 text-green-600 dark:text-green-400 mt-1" />
          <div>
            <h3 className="font-semibold text-gray-900 dark:text-white mb-2">
              AI Model Evaluation
            </h3>
            <div className="text-gray-700 dark:text-gray-300 whitespace-pre-line">
              {results.ai_insights}
            </div>
          </div>
        </div>
      </motion.div>

      {/* Metrics Grid */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="grid grid-cols-2 md:grid-cols-4 gap-6 mb-8"
      >
        <div className="bg-white/90 dark:bg-gray-800/90 backdrop-blur-xl rounded-2xl p-6 shadow-xl border border-gray-200/50 dark:border-gray-700/50 hover:shadow-2xl transform hover:scale-105 transition-all duration-300">
          <div className="text-3xl font-bold text-primary-600 dark:text-primary-400 mb-2">
            {(metrics.accuracy * 100).toFixed(2)}%
          </div>
          <div className="text-gray-600 dark:text-gray-400">Accuracy</div>
        </div>
        <div className="bg-white/90 dark:bg-gray-800/90 backdrop-blur-xl rounded-2xl p-6 shadow-xl border border-gray-200/50 dark:border-gray-700/50 hover:shadow-2xl transform hover:scale-105 transition-all duration-300">
          <div className="text-3xl font-bold text-green-600 dark:text-green-400 mb-2">
            {metrics.precision.toFixed(4)}
          </div>
          <div className="text-gray-600 dark:text-gray-400">Precision</div>
        </div>
        <div className="bg-white/90 dark:bg-gray-800/90 backdrop-blur-xl rounded-2xl p-6 shadow-xl border border-gray-200/50 dark:border-gray-700/50 hover:shadow-2xl transform hover:scale-105 transition-all duration-300">
          <div className="text-3xl font-bold text-blue-600 dark:text-blue-400 mb-2">
            {metrics.recall.toFixed(4)}
          </div>
          <div className="text-gray-600 dark:text-gray-400">Recall</div>
        </div>
        <div className="bg-white/90 dark:bg-gray-800/90 backdrop-blur-xl rounded-2xl p-6 shadow-xl border border-gray-200/50 dark:border-gray-700/50 hover:shadow-2xl transform hover:scale-105 transition-all duration-300">
          <div className="text-3xl font-bold text-purple-600 dark:text-purple-400 mb-2">
            {metrics.f1.toFixed(4)}
          </div>
          <div className="text-gray-600 dark:text-gray-400">F1 Score</div>
        </div>
        {metrics.auc && (
          <div className="bg-white/90 dark:bg-gray-800/90 backdrop-blur-xl rounded-2xl p-6 shadow-xl border border-gray-200/50 dark:border-gray-700/50 hover:shadow-2xl transform hover:scale-105 transition-all duration-300">
            <div className="text-3xl font-bold text-orange-600 dark:text-orange-400 mb-2">
              {metrics.auc.toFixed(4)}
            </div>
            <div className="text-gray-600 dark:text-gray-400">AUC Score</div>
          </div>
        )}
      </motion.div>

      {/* Visualizations */}
      <div className="space-y-6">
        {/* Confusion Matrix */}
        {visualizations.confusion_matrix && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="bg-white dark:bg-gray-800 rounded-lg shadow border border-gray-200 dark:border-gray-700 p-6"
          >
            <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-4">
              Confusion Matrix
            </h3>
            <div className="flex justify-center">
              <img
                src={`data:image/png;base64,${visualizations.confusion_matrix}`}
                alt="Confusion Matrix"
                className="max-w-full h-auto rounded-lg"
              />
            </div>
          </motion.div>
        )}

        {/* ROC Curve */}
        {visualizations.roc_curve && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="bg-white dark:bg-gray-800 rounded-lg shadow border border-gray-200 dark:border-gray-700 p-6"
          >
            <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-4">
              ROC Curve
            </h3>
            <div className="flex justify-center">
              <img
                src={`data:image/png;base64,${visualizations.roc_curve}`}
                alt="ROC Curve"
                className="max-w-full h-auto rounded-lg"
              />
            </div>
          </motion.div>
        )}

        {/* Precision-Recall Curve */}
        {visualizations.pr_curve && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="bg-white dark:bg-gray-800 rounded-lg shadow border border-gray-200 dark:border-gray-700 p-6"
          >
            <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-4">
              Precision-Recall Curve
            </h3>
            <div className="flex justify-center">
              <img
                src={`data:image/png;base64,${visualizations.pr_curve}`}
                alt="Precision-Recall Curve"
                className="max-w-full h-auto rounded-lg"
              />
            </div>
          </motion.div>
        )}

        {/* Feature Importance */}
        {visualizations.feature_importance && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="bg-white dark:bg-gray-800 rounded-lg shadow border border-gray-200 dark:border-gray-700 p-6"
          >
            <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-4">
              Feature Importance
            </h3>
            <div className="flex justify-center">
              <img
                src={`data:image/png;base64,${visualizations.feature_importance}`}
                alt="Feature Importance"
                className="max-w-full h-auto rounded-lg"
              />
            </div>
          </motion.div>
        )}
      </div>

      {/* Next Step */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="mt-8 bg-blue-50 dark:bg-blue-900 border border-blue-200 dark:border-blue-700 rounded-lg p-6"
      >
        <div className="flex items-center gap-3">
          <CheckCircle2 className="w-6 h-6 text-blue-600 dark:text-blue-400" />
          <div className="flex-1">
            <h3 className="font-semibold text-blue-900 dark:text-blue-200 mb-1">
              Model Training Complete!
            </h3>
            <p className="text-blue-700 dark:text-blue-300 text-sm">
              Proceed to the Export page to download your trained model and generate a comprehensive report.
            </p>
          </div>
          <a
            href="/export"
            className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg font-medium transition-colors"
          >
            Export Model
          </a>
        </div>
      </motion.div>
    </div>
  )
}

