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
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import logger

class ModelTrainer:
    """Train multiple ML models."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = self._load_config(config_path)
        self.models = {}
        self.best_params = {}
        
    def _load_config(self, config_path: str) -> dict:
        """Load configuration."""
        try:
            import yaml
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except:
            logger.warning("Could not load config, using defaults")
            return {}
    
    def load_data(self) -> tuple:
        """Load and split data."""
        data_path = Path("artifacts/data/feature_data.csv")
        df = pd.read_csv(data_path)
        
        # Separate features and target
        X = df.drop('churn', axis=1)
        y = df['churn']
        
        # Split data
        test_size = self.config.get('training', {}).get('test_size', 0.2)
        random_state = self.config.get('training', {}).get('random_state', 42)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        logger.info(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
        logger.info(f"Target distribution - Train: {y_train.value_counts().to_dict()}")
        logger.info(f"Target distribution - Test: {y_test.value_counts().to_dict()}")
        
        return X_train, X_test, y_train, y_test
    
    def train_logistic_regression(self, X_train, y_train) -> LogisticRegression:
        """Train logistic regression with hyperparameter tuning."""
        logger.info("Training Logistic Regression...")
        
        params = self.config.get('models', {}).get('logistic_regression', {}).get('params', {})
        C_values = params.get('C', [0.1, 1.0, 10.0])
        max_iter = params.get('max_iter', 1000)
        
        param_grid = {'C': C_values}
        
        lr = LogisticRegression(max_iter=max_iter, random_state=42, class_weight='balanced')
        
        # Use grid search with cross-validation
        grid_search = GridSearchCV(lr, param_grid, cv=5, scoring='f1', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        self.best_params['logistic_regression'] = grid_search.best_params_
        
        logger.info(f"Best Logistic Regression params: {grid_search.best_params_}")
        logger.info(f"Best CV score: {grid_search.best_score_:.4f}")
        
        return best_model
    
    def train_random_forest(self, X_train, y_train) -> RandomForestClassifier:
        """Train random forest with hyperparameter tuning."""
        logger.info("Training Random Forest...")
        
        params = self.config.get('models', {}).get('random_forest', {}).get('params', {})
        n_estimators = params.get('n_estimators', [100, 200])
        max_depth = params.get('max_depth', [10, 20, None])
        min_samples_split = params.get('min_samples_split', [2, 5, 10])
        
        param_grid = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split
        }
        
        rf = RandomForestClassifier(random_state=42, class_weight='balanced', n_jobs=-1)
        
        # Use randomized search for efficiency
        from sklearn.model_selection import RandomizedSearchCV
        random_search = RandomizedSearchCV(
            rf, param_grid, n_iter=10, cv=5, scoring='f1', 
            random_state=42, n_jobs=-1
        )
        random_search.fit(X_train, y_train)
        
        best_model = random_search.best_estimator_
        self.best_params['random_forest'] = random_search.best_params_
        
        logger.info(f"Best Random Forest params: {random_search.best_params_}")
        logger.info(f"Best CV score: {random_search.best_score_:.4f}")
        
        return best_model
    
    def train_xgboost(self, X_train, y_train) -> xgb.XGBClassifier:
        """Train XGBoost with hyperparameter tuning."""
        logger.info("Training XGBoost...")
        
        params = self.config.get('models', {}).get('xgboost', {}).get('params', {})
        n_estimators = params.get('n_estimators', [100, 200])
        max_depth = params.get('max_depth', [3, 6, 9])
        learning_rate = params.get('learning_rate', [0.01, 0.1, 0.3])
        
        param_grid = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'learning_rate': learning_rate
        }
        
        xgb_model = xgb.XGBClassifier(
            random_state=42,
            scale_pos_weight=(len(y_train) - y_train.sum()) / y_train.sum(),
            use_label_encoder=False,
            eval_metric='logloss'
        )
        
        # Use randomized search
        from sklearn.model_selection import RandomizedSearchCV
        random_search = RandomizedSearchCV(
            xgb_model, param_grid, n_iter=10, cv=5, scoring='f1',
            random_state=42, n_jobs=-1
        )
        random_search.fit(X_train, y_train)
        
        best_model = random_search.best_estimator_
        self.best_params['xgboost'] = random_search.best_params_
        
        logger.info(f"Best XGBoost params: {random_search.best_params_}")
        logger.info(f"Best CV score: {random_search.best_score_:.4f}")
        
        return best_model
    
    def save_model(self, model, model_name: str):
        """Save trained model."""
        model_path = Path(f"artifacts/models/{model_name}.pkl")
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_path)
        logger.info(f"Model saved to {model_path}")
        
        # Save training parameters
        params_path = Path(f"artifacts/metrics/{model_name}_params.json")
        with open(params_path, 'w') as f:
            json.dump(self.best_params.get(model_name, {}), f, indent=2)

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
    
    # Save test data for evaluation
    test_data = pd.DataFrame(X_test)
    test_data['churn'] = y_test.values
    test_data.to_csv(f"artifacts/data/test_data_{args.model}.csv", index=False)
    logger.info(f"Test data saved for {args.model}")
    
    # Calculate basic metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "f1_score": float(f1_score(y_test, y_pred))
    }
    
    # Save metrics
    metrics_path = Path(f"artifacts/metrics/{args.model}_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    logger.info(f"Model {args.model} metrics: {metrics}")
    logger.info(f"Training completed for {args.model}")

if __name__ == "__main__":
    main()
