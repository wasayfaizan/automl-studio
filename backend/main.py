from fastapi import FastAPI, UploadFile, File, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
import pandas as pd
import numpy as np
import json
import os
import joblib
from pathlib import Path
from typing import Optional, Dict, Any
import base64
from io import BytesIO

from ml_engine import MLEngine
from data_processor import DataProcessor
from ai_insights import AIGenerator
from report_generator import generate_pdf_report

app = FastAPI(title="AutoML Studio API")

def convert_numpy_types(obj):
    """Convert numpy types to native Python types for JSON serialization"""
    if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {str(key): convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_types(item) for item in obj]
    elif pd.isna(obj):
        return None
    elif hasattr(obj, 'item'):  # Handle numpy scalars that have .item() method
        try:
            return obj.item()
        except (ValueError, AttributeError):
            return obj
    return obj

# Pydantic models for request bodies
class CleanConfig(BaseModel):
    target_column: Optional[str] = None
    handle_missing: str = "auto"
    encode_categoricals: bool = True
    normalize: bool = True
    remove_outliers: bool = True

class TrainConfig(BaseModel):
    model_type: str = "random_forest"
    target_column: str
    test_size: float = 0.2
    hyperparameter_tuning: bool = False
    
    class Config:
        # Allow extra fields to be ignored
        extra = "forbid"

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create necessary directories
UPLOAD_DIR = Path("uploads")
MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")
for dir_path in [UPLOAD_DIR, MODELS_DIR, REPORTS_DIR]:
    dir_path.mkdir(exist_ok=True)

# Global state (in production, use Redis or database)
app.state.current_dataset = None
app.state.cleaned_dataset = None
app.state.model = None
app.state.preprocessor = None
app.state.metrics = None
app.state.target_column = None

ml_engine = MLEngine()
data_processor = DataProcessor()
ai_generator = AIGenerator()

@app.get("/")
def read_root():
    return {"message": "AutoML Studio API"}

@app.post("/api/upload")
async def upload_dataset(file: UploadFile = File(...)):
    """Upload and analyze CSV dataset"""
    try:
        # Save uploaded file
        file_path = UPLOAD_DIR / file.filename
        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)
        
        # Load and analyze
        df = pd.read_csv(file_path)
        app.state.current_dataset = df
        
        # Basic analysis
        missing_vals = df.isnull().sum().to_dict()
        missing_pct = (df.isnull().sum() / len(df) * 100).round(2).to_dict()
        
        analysis = {
            "rows": int(len(df)),
            "columns": int(len(df.columns)),
            "column_names": [str(col) for col in df.columns.tolist()],
            "missing_values": convert_numpy_types(missing_vals),
            "missing_percentage": convert_numpy_types(missing_pct),
            "dtypes": {str(k): str(v) for k, v in df.dtypes.astype(str).to_dict().items()},
            "numeric_columns": [str(col) for col in df.select_dtypes(include=[np.number]).columns.tolist()],
            "categorical_columns": [str(col) for col in df.select_dtypes(exclude=[np.number]).columns.tolist()],
            "preview": convert_numpy_types(df.head(20).to_dict(orient="records")),
            "outliers_detected": convert_numpy_types(_detect_outliers(df)),
            "file_path": str(file_path)
        }
        
        # AI summary
        analysis["ai_summary"] = ai_generator.generate_upload_summary(analysis)
        
        return analysis
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

def _detect_outliers(df):
    """Simple outlier detection using IQR"""
    outliers = {}
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outlier_count = len(df[(df[col] < lower_bound) | (df[col] > upper_bound)])
        if outlier_count > 0:
            outliers[str(col)] = int(outlier_count)
    return outliers

@app.post("/api/clean")
async def clean_data(config: CleanConfig):
    """Clean and preprocess the dataset"""
    try:
        if app.state.current_dataset is None:
            raise HTTPException(status_code=400, detail="No dataset uploaded")
        
        df = app.state.current_dataset.copy()
        
        # Perform cleaning
        cleaned_df, preprocessing_steps, preprocessor = data_processor.clean_data(
            df, 
            target_column=config.target_column,
            handle_missing=config.handle_missing,
            encode_categoricals=config.encode_categoricals,
            normalize=config.normalize,
            remove_outliers=config.remove_outliers
        )
        
        app.state.cleaned_dataset = cleaned_df
        app.state.preprocessor = preprocessor
        app.state.target_column = config.target_column
        
        # Generate comparison
        comparison = {
            "before": {
                "rows": int(len(df)),
                "columns": int(len(df.columns)),
                "missing_values": int(df.isnull().sum().sum()),
                "dtypes": {str(k): str(v) for k, v in df.dtypes.astype(str).to_dict().items()}
            },
            "after": {
                "rows": int(len(cleaned_df)),
                "columns": int(len(cleaned_df.columns)),
                "missing_values": int(cleaned_df.isnull().sum().sum()),
                "dtypes": {str(k): str(v) for k, v in cleaned_df.dtypes.astype(str).to_dict().items()}
            },
            "steps": preprocessing_steps
        }
        
        # Save cleaned dataset
        cleaned_path = UPLOAD_DIR / "cleaned_dataset.csv"
        cleaned_df.to_csv(cleaned_path, index=False)
        
        # Convert preview to JSON-serializable format
        preview_data = cleaned_df.head(20).to_dict(orient="records")
        preview_data = convert_numpy_types(preview_data)
        
        return {
            "comparison": comparison,
            "cleaned_data_path": str(cleaned_path),
            "preview": preview_data
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/eda")
async def generate_eda():
    """Generate EDA report with charts"""
    try:
        if app.state.cleaned_dataset is None:
            raise HTTPException(status_code=400, detail="No cleaned dataset available")
        
        df = app.state.cleaned_dataset
        
        # Generate charts
        charts = {}
        
        # Histograms for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) > 0:
            charts["histograms"] = _generate_histograms(df, numeric_cols[:8])  # Increased limit
        
        # Correlation heatmap
        if len(numeric_cols) > 1:
            charts["correlation"] = _generate_correlation_heatmap(df[numeric_cols])
        
        # Box plots
        if len(numeric_cols) > 0:
            charts["boxplots"] = _generate_boxplots(df, numeric_cols[:8])
        
        # Violin plots
        if len(numeric_cols) > 0:
            charts["violin_plots"] = _generate_violin_plots(df, numeric_cols[:6])
        
        # Pair plot (sample if too many columns)
        if len(numeric_cols) > 1:
            charts["pairplot"] = _generate_pairplot(df, numeric_cols[:5])
        
        # Categorical bar charts
        categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
        if len(categorical_cols) > 0:
            charts["categorical"] = _generate_categorical_charts(df, categorical_cols[:8])
        
        # Target distribution if target exists
        if app.state.target_column and app.state.target_column in df.columns:
            charts["target_distribution"] = _generate_target_distribution(df, app.state.target_column)
        
        # Scatter plots for top correlated pairs
        if len(numeric_cols) > 1:
            charts["scatter_plots"] = _generate_scatter_plots(df, numeric_cols[:6])
        
        # Distribution comparison (if target exists)
        if app.state.target_column and app.state.target_column in df.columns and len(numeric_cols) > 0:
            charts["distribution_comparison"] = _generate_distribution_comparison(df, numeric_cols[:4], app.state.target_column)
        
        # AI EDA Summary
        eda_summary = ai_generator.generate_eda_summary(df, charts, app.state.target_column)
        
        # Convert statistics to JSON-serializable format
        numeric_stats = {}
        if numeric_cols:
            numeric_stats_raw = df[numeric_cols].describe().to_dict()
            numeric_stats = convert_numpy_types(numeric_stats_raw)
        
        # Extended statistics (skewness, kurtosis)
        extended_stats = {}
        if numeric_cols:
            for col in numeric_cols:
                extended_stats[col] = {
                    "mean": float(df[col].mean()),
                    "median": float(df[col].median()),
                    "std": float(df[col].std()),
                    "skewness": float(df[col].skew()) if len(df) > 2 else 0,
                    "kurtosis": float(df[col].kurtosis()) if len(df) > 2 else 0,
                    "min": float(df[col].min()),
                    "max": float(df[col].max()),
                    "q25": float(df[col].quantile(0.25)),
                    "q75": float(df[col].quantile(0.75)),
                    "iqr": float(df[col].quantile(0.75) - df[col].quantile(0.25))
                }
        
        categorical_stats = {}
        if categorical_cols:
            categorical_stats_raw = {col: df[col].value_counts().to_dict() for col in categorical_cols[:8]}
            categorical_stats = convert_numpy_types(categorical_stats_raw)
        
        # Data quality metrics
        quality_metrics = {
            "total_rows": int(len(df)),
            "total_columns": int(len(df.columns)),
            "numeric_columns": len(numeric_cols),
            "categorical_columns": len(categorical_cols),
            "missing_values": int(df.isnull().sum().sum()),
            "duplicate_rows": int(df.duplicated().sum()),
            "memory_usage_mb": float(df.memory_usage(deep=True).sum() / 1024 / 1024)
        }
        
        # Outlier analysis
        outlier_analysis = {}
        if numeric_cols:
            for col in numeric_cols[:10]:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                if IQR > 0:
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
                    outlier_analysis[col] = {
                        "count": int(len(outliers)),
                        "percentage": float(len(outliers) / len(df) * 100),
                        "lower_bound": float(lower_bound),
                        "upper_bound": float(upper_bound)
                    }
        
        return {
            "charts": charts,
            "summary": str(eda_summary),
            "statistics": {
                "numeric_stats": numeric_stats,
                "extended_stats": extended_stats,
                "categorical_stats": categorical_stats
            },
            "quality_metrics": quality_metrics,
            "outlier_analysis": outlier_analysis
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

def _generate_histograms(df, columns):
    """Generate histogram data for numeric columns"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    histograms = {}
    for col in columns:
        fig, ax = plt.subplots(figsize=(8, 6))
        df[col].hist(bins=30, ax=ax, edgecolor='black')
        ax.set_title(f'Distribution of {col}')
        ax.set_xlabel(col)
        ax.set_ylabel('Frequency')
        
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        histograms[col] = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
    
    return histograms

def _generate_correlation_heatmap(df):
    """Generate correlation heatmap"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    fig, ax = plt.subplots(figsize=(10, 8))
    corr = df.corr()
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=ax, square=True)
    ax.set_title('Correlation Heatmap')
    
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    result = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    
    return result

def _generate_boxplots(df, columns):
    """Generate boxplot data"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    boxplots = {}
    for col in columns:
        fig, ax = plt.subplots(figsize=(8, 6))
        df.boxplot(column=col, ax=ax)
        ax.set_title(f'Box Plot: {col}')
        ax.set_ylabel(col)
        
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        boxplots[col] = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
    
    return boxplots

def _generate_categorical_charts(df, columns):
    """Generate bar charts for categorical columns"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    charts = {}
    for col in columns:
        value_counts = df[col].value_counts().head(10)  # Top 10
        fig, ax = plt.subplots(figsize=(10, 6))
        value_counts.plot(kind='bar', ax=ax, color='steelblue')
        ax.set_title(f'Distribution of {col}')
        ax.set_xlabel(col)
        ax.set_ylabel('Count')
        plt.xticks(rotation=45, ha='right')
        
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        charts[col] = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
    
    return charts

def _generate_target_distribution(df, target_col):
    """Generate target distribution chart"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(8, 6))
    if df[target_col].dtype in ['int64', 'float64']:
        df[target_col].hist(bins=30, ax=ax, edgecolor='black')
    else:
        value_counts = df[target_col].value_counts()
        value_counts.plot(kind='bar', ax=ax, color='steelblue')
    ax.set_title(f'Target Distribution: {target_col}')
    ax.set_xlabel(target_col)
    ax.set_ylabel('Frequency')
    
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    result = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    
    return result

def _generate_violin_plots(df, columns):
    """Generate violin plots for numeric columns"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    plots = {}
    for col in columns:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.violinplot(y=df[col], ax=ax, color='steelblue')
        ax.set_title(f'Violin Plot: {col}')
        ax.set_ylabel(col)
        
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        plots[col] = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
    
    return plots

def _generate_pairplot(df, columns):
    """Generate pair plot for numeric columns"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    if len(columns) < 2:
        return None
    
    # Limit to 5 columns for performance
    plot_cols = columns[:5]
    fig, axes = plt.subplots(len(plot_cols), len(plot_cols), figsize=(15, 15))
    
    if len(plot_cols) == 1:
        axes = [[axes]]
    elif len(plot_cols) == 2:
        axes = [[axes[0], axes[1]]]
    
    for i, col1 in enumerate(plot_cols):
        for j, col2 in enumerate(plot_cols):
            if i == j:
                axes[i][j].hist(df[col1], bins=20, color='steelblue', edgecolor='black')
                axes[i][j].set_title(col1)
            else:
                axes[i][j].scatter(df[col2], df[col1], alpha=0.5, s=10)
                axes[i][j].set_xlabel(col2)
                axes[i][j].set_ylabel(col1)
    
    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    result = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    
    return result

def _generate_scatter_plots(df, columns):
    """Generate scatter plots for top correlated pairs"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    if len(columns) < 2:
        return {}
    
    plots = {}
    # Generate scatter plots for first few pairs
    for i in range(min(3, len(columns) - 1)):
        col1 = columns[i]
        col2 = columns[i + 1]
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(df[col1], df[col2], alpha=0.5, s=20)
        ax.set_xlabel(col1)
        ax.set_ylabel(col2)
        ax.set_title(f'Scatter Plot: {col1} vs {col2}')
        ax.grid(True, alpha=0.3)
        
        # Add correlation coefficient
        corr = df[col1].corr(df[col2])
        ax.text(0.05, 0.95, f'Correlation: {corr:.2f}', 
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        plots[f"{col1}_vs_{col2}"] = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
    
    return plots

def _generate_distribution_comparison(df, numeric_cols, target_col):
    """Generate distribution comparison by target"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    if len(numeric_cols) == 0:
        return {}
    
    plots = {}
    df_work = df.copy()  # Work on a copy to avoid modifying original
    
    for col in numeric_cols[:4]:  # Limit to 4
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Check if target is categorical
        if df_work[target_col].dtype == 'object' or df_work[target_col].nunique() < 10:
            # Group by target categories
            for category in df_work[target_col].unique()[:5]:  # Limit to 5 categories
                subset = df_work[df_work[target_col] == category][col]
                if len(subset) > 0:
                    ax.hist(subset, alpha=0.6, label=str(category), bins=20)
            ax.legend()
            ax.set_title(f'Distribution of {col} by {target_col}')
        else:
            # For continuous target, create bins
            try:
                df_work['target_bin'] = pd.qcut(df_work[target_col], q=3, duplicates='drop')
                for bin_val in df_work['target_bin'].unique():
                    subset = df_work[df_work['target_bin'] == bin_val][col]
                    if len(subset) > 0:
                        ax.hist(subset, alpha=0.6, label=str(bin_val), bins=20)
                ax.legend()
                ax.set_title(f'Distribution of {col} by {target_col} (binned)')
            except:
                # Fallback if binning fails
                ax.hist(df_work[col], bins=20, color='steelblue', edgecolor='black')
                ax.set_title(f'Distribution of {col}')
        
        ax.set_xlabel(col)
        ax.set_ylabel('Frequency')
        
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        plots[col] = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
    
    return plots

@app.post("/api/train")
async def train_model(config: TrainConfig):
    """Train ML model"""
    try:
        # Debug logging
        print(f"Training request: model_type={config.model_type}, target_column={config.target_column}, test_size={config.test_size}")
        print(f"Cleaned dataset shape: {app.state.cleaned_dataset.shape if app.state.cleaned_dataset is not None else 'None'}")
        if app.state.cleaned_dataset is not None:
            print(f"Available columns: {list(app.state.cleaned_dataset.columns)}")
        if app.state.cleaned_dataset is None:
            raise HTTPException(status_code=400, detail="No cleaned dataset available. Please clean your data first.")
        
        # Validate target column
        if not config.target_column or config.target_column.strip() == "":
            raise HTTPException(status_code=400, detail="Target column is required and cannot be empty.")
        
        # Check if target column exists in cleaned dataset
        if config.target_column not in app.state.cleaned_dataset.columns:
            # Try to find the target column (might have been encoded)
            possible_targets = [col for col in app.state.cleaned_dataset.columns 
                              if config.target_column.lower() in col.lower()]
            if not possible_targets:
                # List available columns for better error message
                available_cols = list(app.state.cleaned_dataset.columns)[:10]  # First 10 columns
                raise HTTPException(
                    status_code=400, 
                    detail=f"Target column '{config.target_column}' not found in cleaned dataset. Available columns: {', '.join(available_cols)}"
                )
            # Use the original target column name from state if available
            if app.state.target_column and app.state.target_column in app.state.cleaned_dataset.columns:
                target_column = app.state.target_column
            else:
                target_column = possible_targets[0]
        else:
            target_column = config.target_column
        
        # Validate test size
        if config.test_size <= 0 or config.test_size >= 1:
            raise HTTPException(status_code=400, detail="Test size must be between 0 and 1 (exclusive).")
        
        # Check if we have enough data
        if len(app.state.cleaned_dataset) < 10:
            raise HTTPException(status_code=400, detail="Dataset is too small. Need at least 10 rows for training.")
        
        # Train model
        model, metrics, visualizations = ml_engine.train_model(
            app.state.cleaned_dataset,
            target_column,
            config.model_type,
            config.test_size,
            config.hyperparameter_tuning
        )
        
        app.state.model = model
        app.state.metrics = metrics
        
        # Save model
        model_path = MODELS_DIR / f"model_{config.model_type}.pkl"
        joblib.dump(model, model_path)
        
        # Generate AI insights
        ai_insights = ai_generator.generate_model_insights(metrics, config.model_type, model)
        
        # Convert all data to JSON-serializable format
        metrics_serializable = convert_numpy_types(metrics)
        visualizations_serializable = convert_numpy_types(visualizations)
        
        return {
            "metrics": metrics_serializable,
            "visualizations": visualizations_serializable,
            "model_path": str(model_path),
            "ai_insights": str(ai_insights)  # Ensure string type
        }
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except ValueError as e:
        # Handle value errors from ml_engine
        raise HTTPException(status_code=400, detail=f"Validation error: {str(e)}")
    except Exception as e:
        # Log the full error for debugging
        import traceback
        error_details = traceback.format_exc()
        print(f"Training error: {error_details}")  # Log to console
        raise HTTPException(status_code=400, detail=f"Training failed: {str(e)}")

@app.get("/api/download/model")
async def download_model():
    """Download trained model"""
    model_path = MODELS_DIR / "model_random_forest.pkl"
    if not model_path.exists():
        # Try other model types
        for model_type in ["logistic_regression", "xgboost"]:
            model_path = MODELS_DIR / f"model_{model_type}.pkl"
            if model_path.exists():
                break
        else:
            raise HTTPException(status_code=404, detail="No model found")
    
    return FileResponse(
        model_path,
        media_type="application/octet-stream",
        filename=model_path.name
    )

@app.get("/api/download/cleaned-data")
async def download_cleaned_data():
    """Download cleaned dataset"""
    cleaned_path = UPLOAD_DIR / "cleaned_dataset.csv"
    if not cleaned_path.exists():
        raise HTTPException(status_code=404, detail="No cleaned dataset found")
    
    return FileResponse(
        cleaned_path,
        media_type="text/csv",
        filename="cleaned_dataset.csv"
    )

@app.post("/api/generate-report")
async def generate_report():
    """Generate PDF report"""
    try:
        report_path = generate_pdf_report(
            app.state.current_dataset,
            app.state.cleaned_dataset,
            app.state.metrics,
            app.state.target_column,
            REPORTS_DIR
        )
        
        return FileResponse(
            report_path,
            media_type="application/pdf",
            filename="ml_report.pdf"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

