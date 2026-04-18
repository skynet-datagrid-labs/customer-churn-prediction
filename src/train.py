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
    
    def load_data(self) -> tuple:
        """Load and split data."""
        data_path = Path("artifacts/data/feature_data.csv")
        if not data_path.exists():
            logger.error(f"Feature data not found at {data_path}")
            sys.exit(1)
        
        df = pd.read_csv(data_path)
        
        # Separate features and target
        X = df.drop('churn', axis=1)
        y = df['churn']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        logger.info(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
        logger.info(f"Target distribution - Train: {y_train.value_counts().to_dict()}")
        
        # Save test data for later evaluation
        test_df = X_test.copy()
        test_df['churn'] = y_test
        test_df.to_csv("artifacts/data/test_data.csv", index=False)
        
        return X_train, X_test, y_train, y_test
    
    def train_logistic_regression(self, X_train, y_train) -> LogisticRegression:
        """Train logistic regression."""
        logger.info("Training Logistic Regression...")
        
        lr = LogisticRegression(
            C=1.0,
            max_iter=1000,
            random_state=42,
            class_weight='balanced'
        )
        lr.fit(X_train, y_train)
        
        logger.info(f"Logistic Regression training complete")
        return lr
    
    def train_random_forest(self, X_train, y_train) -> RandomForestClassifier:
        """Train random forest."""
        logger.info("Training Random Forest...")
        
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
        return rf
    
    def train_xgboost(self, X_train, y_train) -> xgb.XGBClassifier:
        """Train XGBoost."""
        logger.info("Training XGBoost...")
        
        # Calculate scale_pos_weight for imbalanced data
        scale_pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()
        
        xgb_model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            scale_pos_weight=scale_pos_weight,
            use_label_encoder=False,
            eval_metric='logloss'
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
    
    # Load data
    X_train, X_test, y_train, y_test = trainer.load_data()
    
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
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_test, y_pred, zero_division=0))
    }
    
    # Save metrics
    metrics_path = Path(f"artifacts/metrics/{args.model}_metrics.json")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    logger.info(f"Model {args.model} metrics: {metrics}")
    logger.info(f"Training completed for {args.model}")


if __name__ == "__main__":
    main()
