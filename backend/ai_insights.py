import pandas as pd
import numpy as np

class AIGenerator:
    """
    Generate AI-powered insights and summaries
    This is a simplified version - in production, you'd use GPT-4 or similar
    """
    
    def generate_upload_summary(self, analysis):
        """Generate AI summary for uploaded dataset"""
        rows = analysis["rows"]
        cols = analysis["columns"]
        missing_pct = sum(analysis["missing_percentage"].values()) / cols if cols > 0 else 0
        numeric_count = len(analysis["numeric_columns"])
        categorical_count = len(analysis["categorical_columns"])
        outliers = analysis["outliers_detected"]
        
        summary = f"This dataset contains {rows:,} rows and {cols} features. "
        
        if missing_pct > 0:
            summary += f"Missing values account for {missing_pct:.1f}% of the data. "
        else:
            summary += "The dataset has no missing values. "
        
        summary += f"There are {numeric_count} numeric and {categorical_count} categorical columns. "
        
        if outliers:
            total_outliers = sum(outliers.values())
            summary += f"Outliers were detected in {len(outliers)} column(s) ({total_outliers} total outliers). "
        
        summary += "The dataset is ready for preprocessing and analysis."
        
        return summary
    
    def generate_eda_summary(self, df, charts, target_column=None):
        """Generate EDA insights"""
        summary_parts = []
        
        # Basic stats
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            corr_matrix = df[numeric_cols].corr()
            
            # Find strong correlations
            strong_corrs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_val = corr_matrix.iloc[i, j]
                    if abs(corr_val) > 0.6:
                        strong_corrs.append((
                            corr_matrix.columns[i],
                            corr_matrix.columns[j],
                            corr_val
                        ))
            
            if strong_corrs:
                summary_parts.append("Strong correlations detected:")
                for col1, col2, corr in strong_corrs[:3]:
                    summary_parts.append(f"  • {col1} and {col2}: {corr:.2f}")
        
        # Target analysis
        if target_column and target_column in df.columns:
            target = df[target_column]
            if target.dtype in ['int64', 'float64']:
                summary_parts.append(f"\nTarget '{target_column}' is numeric with mean {target.mean():.2f} and std {target.std():.2f}.")
            else:
                class_dist = target.value_counts()
                summary_parts.append(f"\nTarget '{target_column}' has {len(class_dist)} classes. Distribution: {dict(class_dist.head(5))}")
        
        # Outlier detection
        if len(numeric_cols) > 0:
            outlier_cols = []
            for col in numeric_cols[:5]:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers = len(df[(df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)])
                if outliers > 0:
                    outlier_cols.append(f"{col} ({outliers} outliers)")
            
            if outlier_cols:
                summary_parts.append(f"\nOutliers detected in: {', '.join(outlier_cols)}")
        
        # Feature insights
        if len(numeric_cols) > 0:
            summary_parts.append(f"\nNumeric features show varying distributions. Consider normalization for better model performance.")
            
            # Skewness analysis
            skewed_cols = []
            for col in numeric_cols[:5]:
                skew = df[col].skew()
                if abs(skew) > 1:
                    skewed_cols.append(f"{col} (skew={skew:.2f})")
            if skewed_cols:
                summary_parts.append(f"\nHighly skewed features detected: {', '.join(skewed_cols)}. Consider log transformation.")
        
        # Data quality insights
        duplicate_count = df.duplicated().sum()
        if duplicate_count > 0:
            summary_parts.append(f"\n⚠️ Warning: {duplicate_count} duplicate rows found. Consider removing duplicates.")
        
        missing_count = df.isnull().sum().sum()
        if missing_count == 0:
            summary_parts.append(f"\n✅ No missing values detected. Dataset is clean.")
        
        # Feature count insights
        if len(numeric_cols) > 10:
            summary_parts.append(f"\nDataset has {len(numeric_cols)} numeric features. Consider feature selection to avoid overfitting.")
        
        if not summary_parts:
            summary_parts.append("Dataset appears well-structured. Ready for modeling.")
        
        return "\n".join(summary_parts)
    
    def generate_model_insights(self, metrics, model_type, model):
        """Generate model evaluation insights"""
        insights = []
        
        accuracy = metrics.get("accuracy", 0)
        f1 = metrics.get("f1", 0)
        auc = metrics.get("auc")
        
        # Performance assessment
        if accuracy >= 0.9:
            insights.append(f"Excellent performance! {model_type.replace('_', ' ').title()} achieved {accuracy:.1%} accuracy.")
        elif accuracy >= 0.7:
            insights.append(f"Good performance. {model_type.replace('_', ' ').title()} achieved {accuracy:.1%} accuracy.")
        else:
            insights.append(f"Model performance is moderate ({accuracy:.1%} accuracy). Consider feature engineering or trying different models.")
        
        # Metric analysis
        if f1 < 0.5:
            insights.append("Low F1 score suggests class imbalance or poor model fit. Consider using SMOTE or class weights.")
        
        if auc:
            if auc >= 0.9:
                insights.append(f"Strong discriminative ability (AUC = {auc:.2f}).")
            elif auc < 0.7:
                insights.append(f"Low AUC ({auc:.2f}) suggests the model struggles to distinguish classes.")
        
        # Model-specific insights
        if model_type == "random_forest":
            insights.append("Random Forest provides good feature importance insights and handles non-linear relationships well.")
        elif model_type == "xgboost":
            insights.append("XGBoost is powerful for complex patterns but may overfit with small datasets.")
        elif model_type == "logistic_regression":
            insights.append("Logistic Regression is interpretable and fast, but may struggle with non-linear patterns.")
        
        # Recommendations
        insights.append("\nRecommendations:")
        if accuracy < 0.8:
            insights.append("  • Try hyperparameter tuning")
            insights.append("  • Collect more training data")
            insights.append("  • Feature engineering: create interaction terms")
        
        if metrics.get("precision", 0) < metrics.get("recall", 0):
            insights.append("  • Model has high recall but low precision - consider threshold tuning")
        elif metrics.get("recall", 0) < metrics.get("precision", 0):
            insights.append("  • Model has high precision but low recall - consider threshold tuning")
        
        return "\n".join(insights)

