"""Model evaluation module."""

import sys
import os
import argparse
import json
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, roc_auc_score, confusion_matrix)
import joblib
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Evaluate and compare multiple models."""
    
    def __init__(self):
        self.results = {}
    
    def evaluate_model(self, model, X_test, y_test, model_name: str) -> dict:
        """Evaluate a single model."""
        logger.info(f"Evaluating {model_name}...")
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Calculate metrics
        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1_score": float(f1_score(y_test, y_pred, zero_division=0))
        }
        
        if y_pred_proba is not None:
            metrics["roc_auc"] = float(roc_auc_score(y_test, y_pred_proba))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        metrics["confusion_matrix"] = cm.tolist()
        
        logger.info(f"{model_name}: F1={metrics['f1_score']:.4f}, AUC={metrics.get('roc_auc', 0):.4f}")
        
        return metrics
    
    def compare_models(self) -> dict:
        """Compare all trained models."""
        logger.info("Comparing all models...")
        
        # Load test data
        test_data_path = Path("artifacts/data/test_data.csv")
        if not test_data_path.exists():
            logger.error("Test data not found")
            return {"comparison": {}, "best_model": None, "best_model_metrics": None}
        
        test_df = pd.read_csv(test_data_path)
        X_test = test_df.drop('churn', axis=1)
        y_test = test_df['churn']
        
        # Load and evaluate each model
        models = ['logistic_regression', 'random_forest', 'xgboost']
        
        for model_name in models:
            model_path = Path(f"artifacts/models/{model_name}.pkl")
            if model_path.exists():
                model = joblib.load(model_path)
                metrics = self.evaluate_model(model, X_test, y_test, model_name)
                self.results[model_name] = metrics
            else:
                logger.warning(f"Model not found: {model_name}")
        
        if not self.results:
            logger.error("No models were successfully evaluated")
            return {"comparison": {}, "best_model": None, "best_model_metrics": None}
        
        # Find best model based on F1-score
        best_model_name = max(self.results.items(), key=lambda x: x[1]['f1_score'])[0]
        best_metrics = self.results[best_model_name]
        
        # Create comparison table
        comparison_df = pd.DataFrame(self.results).T
        logger.info("\n" + "="*60)
        logger.info("MODEL COMPARISON RESULTS")
        logger.info("="*60)
        logger.info(f"\n{comparison_df[['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']].to_string()}")
        logger.info(f"\nBest Model: {best_model_name}")
        logger.info(f"Best F1-Score: {best_metrics['f1_score']:.4f}")
        
        return {
            "comparison": self.results,
            "best_model": best_model_name,
            "best_model_metrics": best_metrics
        }


def main():
    """Main evaluation entry point."""
    parser = argparse.ArgumentParser(description="Evaluate models")
    parser.add_argument("--retraining", action="store_true", help="Running in retraining mode")
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = ModelEvaluator()
    
    # Compare all models
    comparison_results = evaluator.compare_models()
    
    if not comparison_results['comparison']:
        logger.error("No models were successfully evaluated")
        sys.exit(1)
    
    # Save evaluation report
    report_path = Path("artifacts/reports/evaluation_report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(report_path, 'w') as f:
        json.dump(comparison_results, f, indent=2)
    
    logger.info(f"Evaluation completed. Best model: {comparison_results['best_model']}")


if __name__ == "__main__":
    main()
