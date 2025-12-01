import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve,
    mean_squared_error, mean_absolute_error, r2_score
)
import xgboost as xgb
import base64
from io import BytesIO
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

class MLEngine:
    def __init__(self):
        # Classification models
        self.classification_models = {
            "logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
            "random_forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            "xgboost": xgb.XGBClassifier(random_state=42, eval_metric='logloss')
        }
        # Regression models
        self.regression_models = {
            "logistic_regression": LinearRegression(),  # Use LinearRegression for regression
            "random_forest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            "xgboost": xgb.XGBRegressor(random_state=42)
        }
        self.models = self.classification_models  # Default to classification
    
    def train_model(self, df, target_column, model_type="random_forest", 
                   test_size=0.2, hyperparameter_tuning=False):
        """
        Train a machine learning model
        """
        # Prepare data
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in dataset")
        
        # Check for NaN values before training
        nan_count = df.isnull().sum().sum()
        if nan_count > 0:
            raise ValueError(f"Dataset contains {nan_count} NaN values. Please ensure data cleaning removes all missing values.")
        
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Check for NaN in features
        X_nan_count = X.isnull().sum().sum()
        if X_nan_count > 0:
            raise ValueError(f"Features contain {X_nan_count} NaN values. Cannot train model with missing values.")
        
        # Check for NaN in target
        y_nan_count = y.isnull().sum()
        if y_nan_count > 0:
            raise ValueError(f"Target column contains {y_nan_count} NaN values. Cannot train model with missing target values.")
        
        # Check if we have enough data
        if len(X) < 2:
            raise ValueError("Dataset is too small. Need at least 2 rows for training.")
        
        # Detect if this is classification or regression
        is_classification = self._detect_task_type(y)
        
        # Handle target encoding if needed (for classification)
        if y.dtype == 'object':
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            y = le.fit_transform(y)
            is_classification = True  # Categorical targets are always classification
        
        # Select appropriate models
        if is_classification:
            self.models = self.classification_models
        else:
            self.models = self.regression_models
        
        # Check if we can use stratified split (only for classification)
        can_stratify = False
        if is_classification:
            unique_classes, class_counts = np.unique(y, return_counts=True)
            can_stratify = len(unique_classes) > 1 and np.all(class_counts >= 2)
            # Additional validation for small datasets
            if len(unique_classes) == 1:
                raise ValueError("Target column has only one unique value. Classification requires at least 2 classes.")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=42, 
            stratify=y if can_stratify else None
        )
        
        # Get model
        if model_type not in self.models:
            raise ValueError(f"Unknown model type: {model_type}")
        
        model = self.models[model_type]
        
        # Hyperparameter tuning (simple grid search)
        if hyperparameter_tuning:
            from sklearn.model_selection import GridSearchCV
            if is_classification:
                scoring = 'f1'
            else:
                scoring = 'neg_mean_squared_error'
            
            if model_type == "random_forest":
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [10, 20, None]
                }
                model = GridSearchCV(model, param_grid, cv=3, scoring=scoring, n_jobs=-1)
            elif model_type == "xgboost":
                param_grid = {
                    'n_estimators': [50, 100],
                    'max_depth': [3, 5, 7]
                }
                model = GridSearchCV(model, param_grid, cv=3, scoring=scoring, n_jobs=-1)
        
        # Train
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_test)
        
        # Calculate metrics based on task type
        if is_classification:
            # Classification metrics
            try:
                y_pred_proba = model.predict_proba(X_test)[:, 1] if len(np.unique(y)) == 2 else None
            except AttributeError:
                y_pred_proba = None
            
            metrics = {
                "task_type": "classification",
                "accuracy": float(accuracy_score(y_test, y_pred)),
                "precision": float(precision_score(y_test, y_pred, average='weighted', zero_division=0)),
                "recall": float(recall_score(y_test, y_pred, average='weighted', zero_division=0)),
                "f1": float(f1_score(y_test, y_pred, average='weighted', zero_division=0)),
            }
            
            # AUC (only for binary classification)
            if y_pred_proba is not None and len(np.unique(y)) == 2:
                metrics["auc"] = float(roc_auc_score(y_test, y_pred_proba))
            else:
                metrics["auc"] = None
        else:
            # Regression metrics
            metrics = {
                "task_type": "regression",
                "r2_score": float(r2_score(y_test, y_pred)),
                "mse": float(mean_squared_error(y_test, y_pred)),
                "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
                "mae": float(mean_absolute_error(y_test, y_pred)),
            }
            y_pred_proba = None  # No probability for regression
        
        # Generate visualizations
        visualizations = {}
        
        if is_classification:
            # Classification visualizations
            cm = confusion_matrix(y_test, y_pred)
            visualizations["confusion_matrix"] = self._plot_confusion_matrix(cm)
            
            # ROC Curve (binary only)
            if y_pred_proba is not None and len(np.unique(y)) == 2:
                visualizations["roc_curve"] = self._plot_roc_curve(y_test, y_pred_proba)
                visualizations["pr_curve"] = self._plot_pr_curve(y_test, y_pred_proba)
        else:
            # Regression visualizations
            visualizations["prediction_scatter"] = self._plot_prediction_scatter(y_test, y_pred)
            visualizations["residuals"] = self._plot_residuals(y_test, y_pred)
        
        # Feature Importance (for tree-based models)
        if model_type in ["random_forest", "xgboost"]:
            if hasattr(model, 'best_estimator_'):
                feature_importance = model.best_estimator_.feature_importances_
            else:
                feature_importance = model.feature_importances_
            
            visualizations["feature_importance"] = self._plot_feature_importance(
                X.columns.tolist(), feature_importance
            )
        
        return model, metrics, visualizations
    
    def _detect_task_type(self, y):
        """
        Detect if the task is classification or regression
        """
        # If target is object/string type, it's classification
        if y.dtype == 'object':
            return True
        
        # If target is integer with few unique values, likely classification
        if y.dtype in ['int64', 'int32', 'int16', 'int8']:
            unique_count = len(np.unique(y))
            total_count = len(y)
            # If less than 20 unique values or less than 10% unique values, likely classification
            if unique_count < 20 or (unique_count / total_count) < 0.1:
                return True
        
        # If target is float with many unique values, likely regression
        if y.dtype in ['float64', 'float32']:
            unique_count = len(np.unique(y))
            total_count = len(y)
            # If more than 20 unique values and more than 10% unique values, likely regression
            if unique_count > 20 and (unique_count / total_count) > 0.1:
                return False
            # Otherwise, check if values look continuous
            if unique_count > total_count * 0.5:
                return False
        
        # Default: assume regression for continuous-looking data
        return False
    
    def _plot_confusion_matrix(self, cm):
        """Generate confusion matrix plot"""
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title('Confusion Matrix')
        
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        result = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        return result
    
    def _plot_roc_curve(self, y_test, y_pred_proba):
        """Generate ROC curve"""
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        auc_score = roc_auc_score(y_test, y_pred_proba)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.2f})')
        ax.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        result = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        return result
    
    def _plot_pr_curve(self, y_test, y_pred_proba):
        """Generate Precision-Recall curve"""
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(recall, precision, label='Precision-Recall Curve')
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curve')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        result = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        return result
    
    def _plot_feature_importance(self, feature_names, importance):
        """Generate feature importance plot"""
        # Get top 20 features
        indices = np.argsort(importance)[::-1][:20]
        top_features = [feature_names[i] for i in indices]
        top_importance = importance[indices]
        
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.barh(range(len(top_features)), top_importance)
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features)
        ax.set_xlabel('Importance')
        ax.set_title('Top 20 Feature Importance')
        ax.invert_yaxis()
        
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        result = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        return result

