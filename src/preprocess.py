"""Data preprocessing module."""

import sys
import os
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import logger

class DataPreprocessor:
    """Preprocess data for ML models."""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def preprocess(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Apply preprocessing to dataframe."""
        logger.info("Starting data preprocessing")
        
        df_processed = df.copy()
        
        # Separate features and target
        target_col = 'churn'
        if target_col in df_processed.columns:
            y = df_processed[target_col]
            X = df_processed.drop(columns=[target_col])
        else:
            X = df_processed
            y = None
        
        # Handle outliers in numerical features
        numerical_cols = ['age', 'tenure_months', 'monthly_spend', 
                         'support_tickets', 'last_login_days', 'satisfaction_score']
        
        for col in numerical_cols:
            if col in X.columns:
                # Cap outliers at 99th percentile
                if fit:
                    upper_limit = X[col].quantile(0.99)
                    lower_limit = X[col].quantile(0.01)
                    self.outlier_limits = getattr(self, 'outlier_limits', {})
                    self.outlier_limits[col] = {'upper': upper_limit, 'lower': lower_limit}
                else:
                    upper_limit = self.outlier_limits.get(col, {}).get('upper', X[col].quantile(0.99))
                    lower_limit = self.outlier_limits.get(col, {}).get('lower', X[col].quantile(0.01))
                
                X[col] = X[col].clip(lower_limit, upper_limit)
                logger.info(f"Capped outliers in {col} to [{lower_limit:.2f}, {upper_limit:.2f}]")
        
        # Encode categorical variables
        categorical_cols = ['gender', 'contract_type']
        
        for col in categorical_cols:
            if col in X.columns:
                if fit:
                    le = LabelEncoder()
                    X[col] = le.fit_transform(X[col].astype(str))
                    self.label_encoders[col] = le
                    logger.info(f"Encoded {col} with classes: {le.classes_.tolist()}")
                else:
                    le = self.label_encoders.get(col)
                    if le:
                        # Handle unknown categories
                        X[col] = X[col].astype(str)
                        X[col] = X[col].apply(lambda x: x if x in le.classes_ else le.classes_[0])
                        X[col] = le.transform(X[col])
                    else:
                        logger.warning(f"No encoder found for {col}, skipping")
        
        # Scale numerical features
        if fit:
            X_scaled = self.scaler.fit_transform(X[numerical_cols])
            logger.info(f"Fitted scaler with means: {self.scaler.mean_}")
        else:
            X_scaled = self.scaler.transform(X[numerical_cols])
        
        # Replace original numerical columns with scaled versions
        X[numerical_cols] = X_scaled
        
        # Log preprocessing info
        logger.info(f"Preprocessed data shape: {X.shape}")
        logger.info(f"Feature columns: {X.columns.tolist()}")
        
        # Reattach target if exists
        if y is not None:
            X[target_col] = y.values
        
        return X
    
    def save_preprocessor(self, path: str = "artifacts/models/preprocessor.pkl"):
        """Save preprocessor objects."""
        preprocessor_data = {
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'outlier_limits': getattr(self, 'outlier_limits', {})
        }
        joblib.dump(preprocessor_data, path)
        logger.info(f"Preprocessor saved to {path}")
    
    def load_preprocessor(self, path: str = "artifacts/models/preprocessor.pkl"):
        """Load preprocessor objects."""
        preprocessor_data = joblib.load(path)
        self.scaler = preprocessor_data['scaler']
        self.label_encoders = preprocessor_data['label_encoders']
        self.outlier_limits = preprocessor_data.get('outlier_limits', {})
        logger.info(f"Preprocessor loaded from {path}")

def main():
    """Main preprocessing entry point."""
    parser = argparse.ArgumentParser(description="Preprocess data")
    parser.add_argument("--retraining", action="store_true", help="Running in retraining mode")
    
    args = parser.parse_args()
    
    # Load validated data
    data_path = Path("artifacts/data/validated_data.csv")
    if not data_path.exists():
        logger.error(f"Validated data not found at {data_path}")
        sys.exit(1)
    
    df = pd.read_csv(data_path)
    logger.info(f"Loaded {len(df)} rows for preprocessing")
    
    # Preprocess
    preprocessor = DataPreprocessor()
    
    if args.retraining and Path("artifacts/models/preprocessor.pkl").exists():
        # Load existing preprocessor for consistency
        preprocessor.load_preprocessor()
        df_processed = preprocessor.preprocess(df, fit=False)
    else:
        df_processed = preprocessor.preprocess(df, fit=True)
        preprocessor.save_preprocessor()
    
    # Save preprocessed data
    output_path = Path("artifacts/data/preprocessed_data.csv")
    df_processed.to_csv(output_path, index=False)
    logger.info(f"Preprocessed data saved to {output_path}")
    
    # Save feature info
    import json
    feature_info = {
        "features": [col for col in df_processed.columns if col != 'churn'],
        "target": "churn",
        "n_features": len([col for col in df_processed.columns if col != 'churn']),
        "n_samples": len(df_processed)
    }
    
    info_path = Path("artifacts/reports/feature_info.json")
    info_path.parent.mkdir(parents=True, exist_ok=True)
    with open(info_path, 'w') as f:
        json.dump(feature_info, f, indent=2)
    
    logger.info("Preprocessing completed successfully")

if __name__ == "__main__":
    main()
