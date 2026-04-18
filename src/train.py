"""Model training module."""

import sys
import os
import argparse
import json
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.impute import SimpleImputer
import xgboost as xgb
import joblib
import logging
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """Train multiple ML models."""
    
    def __init__(self):
        self.models = {}
        self.best_params = {}
        self.imputer = SimpleImputer(strategy='median')
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean data by handling NaN and infinite values."""
        logger.info("Cleaning data...")
        
        # Check for NaN values
        nan_counts = df.isnull().sum()
        if nan_counts.any():
            logger.warning(f"NaN values found:\n{nan_counts[nan_counts > 0]}")
        
        # Check for infinite values
        inf_counts = np.isinf(df.select_dtypes(include=[np.number])).sum()
        if inf_counts.any():
            logger.warning(f"Infinite values found:\n{inf_counts[inf_counts > 0]}")
        
        # Replace infinite values with NaN
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Fill NaN values with median for numerical columns
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if df[col].isnull().any():
                median_val = df[col].median()
                df[col].fillna(median_val, inplace=True)
                logger.info(f"Filled {df[col].isnull().sum()} NaN values in {col} with median: {median_val}")
        
        # Fill NaN values with mode for categorical columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].isnull().any():
                mode_val = df[col].mode()[0] if len(df[col].mode()) > 0 else "Unknown"
                df[col].fillna(mode_val, inplace=True)
                logger.info(f"Filled NaN values in {col} with mode: {mode_val}")
        
        # Verify no NaN values remain
        if df.isnull().any().any():
            logger.error("NaN values still present after cleaning!")
            logger.error(f"Remaining NaN counts:\n{df.isnull().sum()}")
            # Drop rows with any NaN as last resort
            initial_rows = len(df)
            df = df.dropna()
            logger.warning(f"Dropped {initial_rows - len(df)} rows with NaN values")
        
        logger.info(f"Data cleaning complete. Final shape: {df.shape}")
        return df
    
    def load_data(self) -> tuple:
        """Load and split data with proper cleaning."""
        data_path = Path("artifacts/data/feature_data.csv")
        if not data_path.exists():
            logger.error(f"Feature data not found at {data_path}")
            sys.exit(1)
        
        df = pd.read_csv(data_path)
        logger.info(f"Loaded data shape: {df.shape}")
        
        # Clean the data
        df = self.clean_data(df)
        
        # Separate features and target
        X = df.drop('churn', axis=1)
        y = df['churn']
        
        # Check for any remaining issues
        if X.isnull().any().any():
            logger.error("X still contains NaN values!")
            # Drop columns with NaN
            cols_with_nan = X.columns[X.isnull().any()].tolist()
            logger.warning(f"Dropping columns with NaN: {cols_with_nan}")
            X = X.drop(columns=cols_with_nan)
        
        # Split data with stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        logger.info(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
        logger.info(f"Target distribution - Train: {y_train.value_counts().to_dict()}")
        logger.info(f"Target distribution - Test: {y_test.value_counts().to_dict()}")
        
        # Save test data for later evaluation
        test_df = X_test.copy()
        test_df['churn'] = y_test
        test_df.to_csv("artifacts/data/test_data.csv", index=False)
        logger.info("Test data saved to artifacts/data/test_data.csv")
        
        return X_train, X_test, y_train, y_test
    
    def train_logistic_regression(self, X_train, y_train) -> LogisticRegression:
        """Train logistic regression with proper error handling."""
        logger.info("Training Logistic Regression...")
        
        # Ensure no NaN values
        if X_train.isnull().any().any():
            logger.error("NaN values found in training data!")
            logger.info("Applying imputation...")
            X_train = pd.DataFrame(self.imputer.fit_transform(X_train), columns=X_train.columns)
        
        lr = LogisticRegression(
            C=1.0,
            max_iter=1000,
            random_state=42,
            class_weight='balanced',
            solver='lbfgs'
        )
        
        try:
            lr.fit(X_train, y_train)
            logger.info(f"Logistic Regression training complete")
            logger.info(f"Model coefficients: {lr.coef_[0][:5]}...")  # Show first 5 coefficients
        except Exception as e:
            logger.error(f"Error training Logistic Regression: {str(e)}")
            raise
        
        return lr
    
    def train_random_forest(self, X_train, y_train) -> RandomForestClassifier:
        """Train random forest with proper error handling."""
        logger.info("Training Random Forest...")
        
        # Random Forest can handle NaN better, but still clean
        if X_train.isnull().any().any():
            logger.warning("NaN values found in training data, applying imputation...")
            X_train = pd.DataFrame(self.imputer.fit_transform(X_train), columns=X_train.columns)
        
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=42,
            class_weight='balanced',
            n_jobs=-1
        )
        
        rf.fit(X_train, y_train)
        logger.info(f"Random Forest training complete")
        logger.info(f"Feature importance sum: {rf.feature_importances_.sum():.4f}")
        
        return rf
    
    def train_xgboost(self, X_train, y_train) -> xgb.XGBClassifier:
        """Train XGBoost with proper error handling."""
        logger.info("Training XGBoost...")
        
        # XGBoost can handle NaN natively, but we'll clean anyway
        if X_train.isnull().any().any():
            logger.warning("NaN values found in training data, XGBoost can handle them")
        
        # Calculate scale_pos_weight for imbalanced data
        scale_pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()
        logger.info(f"Scale pos weight: {scale_pos_weight:.4f}")
        
        xgb_model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            scale_pos_weight=scale_pos_weight,
            use_label_encoder=False,
            eval_metric='logloss',
            missing=np.nan  # Explicitly handle NaN
        )
        
        xgb_model.fit(X_train, y_train)
        logger.info(f"XGBoost training complete")
        
        return xgb_model
    
    def save_model(self, model, model_name: str):
        """Save trained model."""
        model_path = Path(f"artifacts/models/{model_name}.pkl")
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_path)
        logger.info(f"Model saved to {model_path}")


def main():
    """Main training entry point."""
    parser = argparse.ArgumentParser(description="Train ML models")
    parser.add_argument("--model", choices=['logistic_regression', 'random_forest', 'xgboost'], 
                       required=True, help="Model to train")
    parser.add_argument("--retraining", action="store_true", help="Running in retraining mode")
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = ModelTrainer()
    
    try:
        # Load and clean data
        X_train, X_test, y_train, y_test = trainer.load_data()
        
        # Verify data quality before training
        logger.info("Verifying data quality...")
        logger.info(f"X_train - NaN count: {X_train.isnull().sum().sum()}")
        logger.info(f"X_train - Inf count: {np.isinf(X_train).sum().sum()}")
        
        if X_train.isnull().sum().sum() > 0:
            logger.warning("NaN values still present in training data. Applying final imputation...")
            X_train = pd.DataFrame(trainer.imputer.fit_transform(X_train), columns=X_train.columns)
            X_test = pd.DataFrame(trainer.imputer.transform(X_test), columns=X_test.columns)
        
        # Train specified model
        if args.model == 'logistic_regression':
            model = trainer.train_logistic_regression(X_train, y_train)
        elif args.model == 'random_forest':
            model = trainer.train_random_forest(X_train, y_train)
        elif args.model == 'xgboost':
            model = trainer.train_xgboost(X_train, y_train)
        else:
            logger.error(f"Unknown model: {args.model}")
            sys.exit(1)
        
        # Save model
        trainer.save_model(model, args.model)
        
        # Calculate and save metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
        
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1_score": float(f1_score(y_test, y_pred, zero_division=0))
        }
        
        # Add confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        metrics["confusion_matrix"] = cm.tolist()
        
        # Save metrics
        metrics_path = Path(f"artifacts/metrics/{args.model}_metrics.json")
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        logger.info(f"Model {args.model} metrics: {metrics}")
        logger.info(f"Training completed successfully for {args.model}")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
