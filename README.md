# 🤖 AutoML Studio

> A modern, full-stack AutoML platform that makes machine learning accessible to everyone. Upload your data, let AI handle the preprocessing, and train production-ready models with just a few clicks.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![React](https://img.shields.io/badge/React-18.2-blue.svg)](https://reactjs.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104-green.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)


---

## ✨ Features

### 🎯 Core Capabilities

- **📊 Intelligent Data Processing**: Automated data cleaning, preprocessing, and feature engineering
- **📈 Comprehensive EDA**: Interactive exploratory data analysis with AI-generated insights
- **🧠 Multiple ML Models**: Support for Logistic Regression, Random Forest, and XGBoost
- **🎨 Beautiful Visualizations**: Confusion matrices, ROC curves, feature importance, and more
- **📥 Model Export**: Download trained models, cleaned datasets, and comprehensive PDF reports
- **🤖 AI-Powered Insights**: Automated recommendations and explanations throughout the pipeline

### 🎨 User Experience

- **🌓 Dark Mode**: Beautiful dark/light theme support
- **📱 Responsive Design**: Works seamlessly on desktop, tablet, and mobile
- **⚡ Real-time Updates**: Live progress tracking and instant feedback
- **🎭 Modern UI**: Glassmorphism design with smooth animations
- **🔍 Interactive Charts**: Explore your data with intuitive visualizations

---

## 🚀 Quick Start

### Prerequisites

- **Node.js** 18+ and npm
- **Python** 3.8+
- **pip** package manager

### Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/wasayfaizan/automl-studio.git
   cd automl-studio
   ```

2. **Install frontend dependencies**

   ```bash
   npm install
   ```

3. **Install backend dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Running the Application

**Terminal 1 - Start Backend:**

```bash
cd backend
python main.py
```

**Terminal 2 - Start Frontend:**

```bash
npm run dev
```

Visit `http://localhost:3000` to access the application! 🎉

---

## 📖 Usage Guide

### 1. **Upload Dataset** 📤

- Upload your CSV file
- View automatic data profiling and analysis
- Get AI-generated dataset summary

### 2. **Data Cleaning** 🧹

- Automated missing value handling
- Categorical encoding (One-Hot, Label)
- Outlier detection and removal
- Feature normalization

### 3. **EDA Report** 📊

- Comprehensive statistical analysis
- Interactive visualizations
- Correlation analysis
- Outlier detection
- AI-generated insights

### 4. **Model Training** 🎯

- Choose from Logistic Regression, Random Forest, or XGBoost
- Automatic classification/regression detection
- Configurable train/test split
- Optional hyperparameter tuning

### 5. **Results & Metrics** 📈

- Performance metrics (Accuracy, Precision, Recall, F1, AUC)
- Visualizations (Confusion Matrix, ROC Curve, Feature Importance)
- AI-powered model evaluation
- Actionable recommendations

### 6. **Model Export** 💾

- Download trained model (.pkl)
- Export cleaned dataset
- Generate comprehensive PDF report

---

## 🛠️ Tech Stack

### Frontend

- **React 18** - UI framework
- **Tailwind CSS** - Styling
- **Framer Motion** - Animations
- **React Router** - Navigation
- **Axios** - API client
- **Lucide React** - Icons

### Backend

- **FastAPI** - Web framework
- **scikit-learn** - Machine learning
- **XGBoost** - Gradient boosting
- **pandas** - Data processing
- **matplotlib/seaborn** - Visualizations
- **joblib** - Model serialization
- **reportlab** - PDF generation

---

## 📁 Project Structure

```
automl-studio/
├── backend/
│   ├── main.py              # FastAPI application
│   ├── data_processor.py    # Data cleaning & preprocessing
│   ├── ml_engine.py         # Model training & evaluation
│   ├── ai_insights.py       # AI summary generation
│   └── report_generator.py  # PDF report generation
├── src/
│   ├── components/          # React components
│   ├── pages/               # Page components
│   ├── App.jsx              # Main app component
│   └── index.css            # Global styles
├── package.json             # Frontend dependencies
├── requirements.txt         # Backend dependencies
└── README.md               # This file
```

---

## 🎯 API Endpoints

| Endpoint                     | Method | Description                 |
| ---------------------------- | ------ | --------------------------- |
| `/api/upload`                | POST   | Upload and analyze CSV file |
| `/api/clean`                 | POST   | Clean and preprocess data   |
| `/api/eda`                   | GET    | Generate EDA report         |
| `/api/train`                 | POST   | Train ML model              |
| `/api/download/model`        | GET    | Download trained model      |
| `/api/download/cleaned-data` | GET    | Download cleaned dataset    |
| `/api/generate-report`       | POST   | Generate PDF report         |

Full API documentation available at `http://localhost:8000/docs` when backend is running.

---

## 🎨 Screenshots

### Upload & Analysis

<img width="1916" height="854" alt="Screenshot 2025-12-01 at 6 34 52 PM" src="https://github.com/user-attachments/assets/b37e29bb-74f9-49a6-bbc0-c18f32acf894" />
<img width="1914" height="854" alt="Screenshot 2025-12-01 at 6 35 06 PM" src="https://github.com/user-attachments/assets/1f58785e-a7ff-4656-adc5-68db5fc9717c" />


### EDA Report

<img width="1913" height="858" alt="Screenshot 2025-12-01 at 6 35 21 PM" src="https://github.com/user-attachments/assets/86533d96-670d-42f3-8387-f564b508873f" />


### Model Training

<img width="1917" height="530" alt="Screenshot 2025-12-01 at 6 35 37 PM" src="https://github.com/user-attachments/assets/2882ce99-79ef-4e29-967c-3fe7bbc7730d" />


### Results Dashboard

<img width="1917" height="852" alt="Screenshot 2025-12-01 at 6 35 47 PM" src="https://github.com/user-attachments/assets/3902a5de-e943-405b-9f65-9ed2b38de284" />


---

## 🔧 Configuration

### Environment Variables

Create a `.env` file in the root directory:

```env
# Backend
BACKEND_PORT=8000
UPLOAD_DIR=backend/uploads
MODELS_DIR=backend/models
REPORTS_DIR=backend/reports

# Frontend
VITE_API_URL=http://localhost:8000
```

---




