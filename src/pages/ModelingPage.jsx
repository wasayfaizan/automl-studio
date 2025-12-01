import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { Brain, Play, Settings } from 'lucide-react'
import axios from 'axios'
import LoadingSpinner from '../components/LoadingSpinner'

export default function ModelingPage() {
  const [analysis, setAnalysis] = useState(null)
  const [modelType, setModelType] = useState('random_forest')
  const [targetColumn, setTargetColumn] = useState('')
  const [testSize, setTestSize] = useState(0.2)
  const [hyperparameterTuning, setHyperparameterTuning] = useState(false)
  const [training, setTraining] = useState(false)
  const [progress, setProgress] = useState(0)

  useEffect(() => {
    const saved = localStorage.getItem('datasetAnalysis')
    if (saved) {
      const data = JSON.parse(saved)
      setAnalysis(data)
      const cols = data.column_names
      const target = cols.find(c => c.toLowerCase().includes('target')) || 
                     cols.find(c => c.toLowerCase().includes('label')) ||
                     cols[cols.length - 1]
      setTargetColumn(target || '')
    }
  }, [])

  const handleTrain = async () => {
    if (!targetColumn) {
      alert('Please select a target column')
      return
    }

    setTraining(true)
    setProgress(0)

    // Simulate progress
    const progressInterval = setInterval(() => {
      setProgress((prev) => {
        if (prev >= 90) {
          clearInterval(progressInterval)
          return 90
        }
        return prev + 10
      })
    }, 200)

    try {
      const response = await axios.post('/api/train', {
        model_type: modelType,
        target_column: targetColumn,
        test_size: testSize,
        hyperparameter_tuning: hyperparameterTuning
      }, {
        headers: {
          'Content-Type': 'application/json'
        }
      })
      
      clearInterval(progressInterval)
      setProgress(100)
      
      localStorage.setItem('modelResults', JSON.stringify(response.data))
      
      // Redirect to results page
      setTimeout(() => {
        window.location.href = '/results'
      }, 500)
    } catch (error) {
      clearInterval(progressInterval)
      alert('Training failed: ' + (error.response?.data?.detail || error.message))
      setTraining(false)
      setProgress(0)
    }
  }

  if (!analysis) {
    return (
      <div className="p-8 max-w-7xl mx-auto">
        <div className="bg-yellow-50 dark:bg-yellow-900 border border-yellow-200 dark:border-yellow-700 rounded-lg p-6">
          <p className="text-yellow-800 dark:text-yellow-200">
            Please upload a dataset first.
          </p>
        </div>
      </div>
    )
  }

  const models = [
    {
      id: 'logistic_regression',
      name: 'Logistic Regression',
      description: 'Fast, interpretable, good for linear relationships'
    },
    {
      id: 'random_forest',
      name: 'Random Forest',
      description: 'Robust, handles non-linear patterns, feature importance'
    },
    {
      id: 'xgboost',
      name: 'XGBoost',
      description: 'High performance, gradient boosting, best for complex patterns'
    }
  ]

  return (
    <div className="p-8 max-w-7xl mx-auto">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="mb-8"
      >
        <div className="flex items-center gap-3 mb-2">
          <div className="p-3 bg-gradient-to-br from-primary-500 to-accent-500 rounded-xl shadow-lg">
            <Brain className="w-6 h-6 text-white" />
          </div>
          <div>
            <h1 className="text-4xl font-bold bg-gradient-to-r from-primary-600 to-accent-600 dark:from-primary-400 dark:to-accent-400 bg-clip-text text-transparent">
              Model Training
            </h1>
            <p className="text-gray-600 dark:text-gray-400 mt-1 text-lg">
              Select a model and configure training parameters
            </p>
          </div>
        </div>
      </motion.div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Model Selection */}
        <div className="lg:col-span-2 space-y-6">
          <div className="bg-white/90 dark:bg-gray-800/90 backdrop-blur-xl rounded-2xl shadow-xl border border-gray-200/50 dark:border-gray-700/50 p-6">
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
              Select Model
            </h3>
            <div className="space-y-3">
              {models.map((model) => (
                <motion.div
                  key={model.id}
                  whileHover={{ scale: 1.02 }}
                  onClick={() => setModelType(model.id)}
                  className={`p-4 rounded-xl border-2 cursor-pointer transition-all duration-200 ${
                    modelType === model.id
                      ? 'border-primary-500 bg-gradient-to-r from-primary-50 to-accent-50 dark:from-primary-900/50 dark:to-accent-900/50 shadow-lg scale-105'
                      : 'border-gray-200 dark:border-gray-700 hover:border-primary-300 dark:hover:border-primary-700 hover:shadow-md hover:scale-[1.02]'
                  }`}
                >
                  <div className="flex items-start justify-between">
                    <div>
                      <h4 className="font-semibold text-gray-900 dark:text-white">
                        {model.name}
                      </h4>
                      <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                        {model.description}
                      </p>
                    </div>
                    <div
                      className={`w-5 h-5 rounded-full border-2 flex items-center justify-center ${
                        modelType === model.id
                          ? 'border-primary-600 bg-primary-600'
                          : 'border-gray-300 dark:border-gray-600'
                      }`}
                    >
                      {modelType === model.id && (
                        <div className="w-3 h-3 rounded-full bg-white" />
                      )}
                    </div>
                  </div>
                </motion.div>
              ))}
            </div>
          </div>

          {/* Configuration */}
          <div className="bg-white/90 dark:bg-gray-800/90 backdrop-blur-xl rounded-2xl shadow-xl border border-gray-200/50 dark:border-gray-700/50 p-6">
            <div className="flex items-center gap-2 mb-4">
              <Settings className="w-5 h-5 text-gray-600 dark:text-gray-400" />
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
                Configuration
              </h3>
            </div>

            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Target Column
                </label>
                <select
                  value={targetColumn}
                  onChange={(e) => setTargetColumn(e.target.value)}
                  className="w-full px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                >
                  <option value="">Select target column</option>
                  {analysis.column_names.map((col) => (
                    <option key={col} value={col}>
                      {col}
                    </option>
                  ))}
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Test Size: {(testSize * 100).toFixed(0)}%
                </label>
                <input
                  type="range"
                  min="0.1"
                  max="0.5"
                  step="0.05"
                  value={testSize}
                  onChange={(e) => setTestSize(parseFloat(e.target.value))}
                  className="w-full"
                />
                <div className="flex justify-between text-xs text-gray-500 dark:text-gray-400 mt-1">
                  <span>10%</span>
                  <span>50%</span>
                </div>
              </div>

              <div className="flex items-center gap-3">
                <input
                  type="checkbox"
                  id="hyperparameter-tuning"
                  checked={hyperparameterTuning}
                  onChange={(e) => setHyperparameterTuning(e.target.checked)}
                  className="w-4 h-4 text-primary-600 rounded"
                />
                <label
                  htmlFor="hyperparameter-tuning"
                  className="text-sm text-gray-700 dark:text-gray-300"
                >
                  Enable Hyperparameter Tuning (slower but better results)
                </label>
              </div>
            </div>
          </div>
        </div>

        {/* Training Panel */}
        <div className="lg:col-span-1">
          <div className="bg-white dark:bg-gray-800 rounded-lg shadow border border-gray-200 dark:border-gray-700 p-6 sticky top-8">
            <div className="flex items-center gap-2 mb-4">
              <Brain className="w-5 h-5 text-primary-600 dark:text-primary-400" />
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
                Train Model
              </h3>
            </div>

            {training ? (
              <div className="space-y-4">
                <div className="text-center">
                  <LoadingSpinner size="lg" />
                  <p className="mt-4 text-gray-600 dark:text-gray-400">
                    Training in progress...
                  </p>
                </div>
                <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                  <motion.div
                    initial={{ width: 0 }}
                    animate={{ width: `${progress}%` }}
                    className="bg-primary-600 h-2 rounded-full transition-all"
                  />
                </div>
                <p className="text-center text-sm text-gray-600 dark:text-gray-400">
                  {progress}% Complete
                </p>
              </div>
            ) : (
              <button
                onClick={handleTrain}
                disabled={!targetColumn}
                className="w-full px-6 py-4 bg-gradient-to-r from-primary-500 to-accent-500 hover:from-primary-600 hover:to-accent-600 disabled:from-gray-400 disabled:to-gray-500 text-white rounded-xl font-semibold transition-all duration-200 shadow-lg hover:shadow-xl transform hover:scale-105 disabled:transform-none disabled:cursor-not-allowed flex items-center justify-center gap-2"
              >
                <Play className="w-5 h-5" />
                Train Model
              </button>
            )}

            <div className="mt-6 p-4 bg-blue-50 dark:bg-blue-900 rounded-lg">
              <p className="text-sm text-blue-800 dark:text-blue-200">
                <strong>Tip:</strong> Start with Random Forest for balanced performance, or XGBoost for maximum accuracy.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

