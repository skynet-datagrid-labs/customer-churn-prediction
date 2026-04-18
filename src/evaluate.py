"""Model evaluation module."""

import sys
import os
import argparse
import json
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, roc_auc_score, confusion_matrix,
                            classification_report)
import joblib

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import logger

class ModelEvaluator:
    """Evaluate and compare multiple models."""
    
    def __init__(self):
        self.results = {}
        
    def evaluate_model(self, model_path: str, test_data_path: str, model_name: str) -> dict:
        """Evaluate a single model."""
        logger.info(f"Evaluating {model_name}...")
        
        # Load model and test data
        model = joblib.load(model_path)
        test_df = pd.read_csv(test_data_path)
        
        # Separate features and target
        X_test = test_df.drop('churn', axis=1)
        y_test = test_df['churn']
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Calculate metrics
        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred)),
            "recall": float(recall_score(y_test, y_pred)),
            "f1_score": float(f1_score(y_test, y_pred))
        }
        
        if y_pred_proba is not None:
            metrics["roc_auc"] = float(roc_auc_score(y_test, y_pred_proba))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        metrics["confusion_matrix"] = cm.tolist()
        
        # Classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        metrics["classification_report"] = report
        
        logger.info(f"{model_name} metrics: {metrics}")
        
        return metrics
    
    def compare_models(self) -> dict:
        """Compare all trained models."""
        logger.info("Comparing all models...")
        
        models = ['logistic_regression', 'random_forest', 'xgboost']
        
        for model_name in models:
            model_path = Path(f"artifacts/models/{model_name}.pkl")
            test_data_path = Path(f"artifacts/data/test_data_{model_name}.csv")
            
            if model_path.exists() and test_data_path.exists():
                metrics = self.evaluate_model(model_path, test_data_path, model_name)
                self.results[model_name] = metrics
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame({
            model: self.results[model] for model in self.results
        }).T
        
        logger.info("\nModel Comparison:")
        logger.info(f"\n{comparison_df[['accuracy', 'precision', 'recall', 'f1_score']]}")
        
        # Find best model
        best_model = max(self.results.items(), key=lambda x: x[1]['f1_score'])
        logger.info(f"\nBest model based on F1-score: {best_model[0]} with F1={best_model[1]['f1_score']:.4f}")
        
        return {
            "comparison": {k: v for k, v in self.results.items()},
            "best_model": best_model[0],
            "best_model_metrics": best_model[1]
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
    
    # Save evaluation report
    report_path = Path("artifacts/reports/evaluation_report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(report_path, 'w') as f:
        json.dump(comparison_results, f, indent=2)
    
    # Create summary visualization
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Prepare data for plotting
    models = list(comparison_results['comparison'].keys())
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for idx, metric in enumerate(metrics):
        values = [comparison_results['comparison'][model][metric] for model in models]
        axes[idx].bar(models, values, color=['blue', 'green', 'orange'])
        axes[idx].set_title(f'{metric.capitalize()} Comparison')
        axes[idx].set_ylabel(metric.capitalize())
        axes[idx].set_ylim([0, 1])
        axes[idx].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, v in enumerate(values):
            axes[idx].text(i, v + 0.02, f'{v:.3f}', ha='center')
    
    plt.tight_layout()
    plt.savefig('artifacts/plots/model_comparison.png', dpi=100)
    logger.info("Model comparison plot saved")
    
    # Save best model info
    best_model_info = {
        "best_model": comparison_results['best_model'],
        "metrics": comparison_results['best_model_metrics'],
        "timestamp": str(pd.Timestamp.now())
    }
    
    best_model_path = Path("artifacts/reports/best_model_info.json")
    with open(best_model_path, 'w') as f:
        json.dump(best_model_info, f, indent=2)
    
    logger.info(f"Evaluation completed. Best model: {comparison_results['best_model']}")

if __name__ == "__main__":
    main()
