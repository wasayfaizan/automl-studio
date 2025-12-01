import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

class DataProcessor:
    def __init__(self):
        self.preprocessor = None
    
    def clean_data(self, df, target_column=None, handle_missing="auto", 
                   encode_categoricals=True, normalize=True, remove_outliers=True):
        """
        Clean and preprocess the dataset
        """
        preprocessing_steps = []
        df_cleaned = df.copy()
        
        # Separate target if provided
        if target_column and target_column in df_cleaned.columns:
            y = df_cleaned[target_column].copy()
            X = df_cleaned.drop(columns=[target_column])
            target_name = target_column
        else:
            X = df_cleaned
            y = None
            target_name = None
        
        # 1. Handle missing values
        if handle_missing == "auto":
            numeric_cols = X.select_dtypes(include=[np.number]).columns
            categorical_cols = X.select_dtypes(exclude=[np.number]).columns
            
            # Numeric: impute with median
            if len(numeric_cols) > 0:
                imputer_numeric = SimpleImputer(strategy='median')
                X[numeric_cols] = imputer_numeric.fit_transform(X[numeric_cols])
                preprocessing_steps.append(f"Imputed missing values in {len(numeric_cols)} numeric columns using median")
            
            # Categorical: impute with mode
            if len(categorical_cols) > 0:
                for col in categorical_cols:
                    if X[col].isnull().sum() > 0:
                        mode_value = X[col].mode()[0] if len(X[col].mode()) > 0 else "Unknown"
                        X[col].fillna(mode_value, inplace=True)
                preprocessing_steps.append(f"Imputed missing values in {len(categorical_cols)} categorical columns using mode")
        
        # 2. Remove outliers (only for numeric columns)
        if remove_outliers:
            numeric_cols = X.select_dtypes(include=[np.number]).columns
            initial_rows = len(X)
            # Create a mask for all rows (start with all True)
            mask = pd.Series([True] * len(X), index=X.index)
            
            for col in numeric_cols:
                # Skip if column has NaN values (will be handled by imputation)
                if X[col].isnull().sum() > 0:
                    continue
                try:
                    Q1 = X[col].quantile(0.25)
                    Q3 = X[col].quantile(0.75)
                    IQR = Q3 - Q1
                    if IQR > 0:  # Only filter if IQR is valid
                        lower_bound = Q1 - 1.5 * IQR
                        upper_bound = Q3 + 1.5 * IQR
                        # Update mask to exclude outliers
                        mask = mask & ((X[col] >= lower_bound) & (X[col] <= upper_bound))
                except Exception:
                    # Skip this column if quantile calculation fails
                    continue
            
            # Apply mask to remove outliers (reset index to ensure alignment)
            X = X[mask].reset_index(drop=True)
            if target_name and y is not None:
                y = y[mask].reset_index(drop=True)
            
            if initial_rows > len(X):
                preprocessing_steps.append(f"Removed {initial_rows - len(X)} outlier rows using IQR method")
        
        # 3. Encode categorical variables
        if encode_categoricals:
            categorical_cols = X.select_dtypes(exclude=[np.number]).columns
            
            # One-hot encode if few unique values, otherwise label encode
            for col in categorical_cols:
                unique_count = X[col].nunique()
                if unique_count <= 10:  # One-hot encode
                    dummies = pd.get_dummies(X[col], prefix=col, drop_first=True)
                    X = pd.concat([X.drop(columns=[col]), dummies], axis=1)
                    preprocessing_steps.append(f"One-hot encoded '{col}' ({unique_count} categories)")
                else:  # Label encode
                    le = LabelEncoder()
                    X[col] = le.fit_transform(X[col].astype(str))
                    preprocessing_steps.append(f"Label encoded '{col}' ({unique_count} categories)")
        
        # 4. Normalize numeric columns
        scaler = None
        if normalize:
            numeric_cols = X.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                scaler = StandardScaler()
                X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
                preprocessing_steps.append(f"Normalized {len(numeric_cols)} numeric columns using StandardScaler")
        
        # Rejoin target if it exists
        if target_name and y is not None:
            # Reset index for both X and y to ensure alignment
            X = X.reset_index(drop=True)
            y = y.reset_index(drop=True)
            # Create a DataFrame with the target column
            y_df = pd.DataFrame({target_name: y})
            # Concatenate X and target
            df_cleaned = pd.concat([X, y_df], axis=1)
        else:
            df_cleaned = X
        
        # Final check: Remove any remaining NaN values
        # This can happen if outlier removal creates misalignment
        initial_rows = len(df_cleaned)
        df_cleaned = df_cleaned.dropna()
        if len(df_cleaned) < initial_rows:
            preprocessing_steps.append(f"Removed {initial_rows - len(df_cleaned)} rows with remaining NaN values after preprocessing")
        
        # Ensure no NaN values remain
        if df_cleaned.isnull().sum().sum() > 0:
            # Fill any remaining NaN with forward fill then backward fill
            df_cleaned = df_cleaned.fillna(method='ffill').fillna(method='bfill')
            # If still NaN, fill with 0 for numeric and 'Unknown' for categorical
            numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns
            categorical_cols = df_cleaned.select_dtypes(exclude=[np.number]).columns
            df_cleaned[numeric_cols] = df_cleaned[numeric_cols].fillna(0)
            df_cleaned[categorical_cols] = df_cleaned[categorical_cols].fillna('Unknown')
            preprocessing_steps.append("Final cleanup: Removed all remaining NaN values")
        
        # Verify no NaN values remain
        if df_cleaned.isnull().sum().sum() > 0:
            raise ValueError(f"Unable to remove all NaN values. Remaining NaN count: {df_cleaned.isnull().sum().sum()}")
        
        # Create preprocessor pipeline for future use
        self.preprocessor = {
            "scaler": scaler,
            "encoders": {}
        }
        
        return df_cleaned, preprocessing_steps, self.preprocessor

