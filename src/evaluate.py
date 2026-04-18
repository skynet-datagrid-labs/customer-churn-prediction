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
        
    def evaluate_model(self, model_path: str, test_data_path: str, model_name: str) -> dict:
        """Evaluate a single model."""
        logger.info(f"Evaluating {model_name}...")
        
        try:
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
                "precision": float(precision_score(y_test, y_pred, zero_division=0)),
                "recall": float(recall_score(y_test, y_pred, zero_division=0)),
                "f1_score": float(f1_score(y_test, y_pred, zero_division=0))
            }
            
            if y_pred_proba is not None:
                metrics["roc_auc"] = float(roc_auc_score(y_test, y_pred_proba))
            
            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            metrics["confusion_matrix"] = cm.tolist()
            
            # Classification report
            report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
            metrics["classification_report"] = report
            
            logger.info(f"{model_name} metrics: accuracy={metrics['accuracy']:.4f}, f1={metrics['f1_score']:.4f}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating {model_name}: {str(e)}")
            return None
    
    def compare_models(self) -> dict:
        """Compare all trained models."""
        logger.info("Comparing all models...")
        
        models = ['logistic_regression', 'random_forest', 'xgboost']
        
        for model_name in models:
            model_path = Path(f"artifacts/models/{model_name}.pkl")
            test_data_path = Path(f"artifacts/data/test_data_{model_name}.csv")
            
            if model_path.exists() and test_data_path.exists():
                metrics = self.evaluate_model(model_path, test_data_path, model_name)
                if metrics is not None:
                    self.results[model_name] = metrics
            else:
                logger.warning(f"Model or test data not found for {model_name}")
        
        if not self.results:
            logger.error("No models were successfully evaluated")
            return {
                "comparison": {},
                "best_model": None,
                "best_model_metrics": None
            }
        
        # Create comparison DataFrame
        comparison_data = {}
        for model_name, metrics in self.results.items():
            # Extract only scalar metrics for comparison
            comparison_data[model_name] = {
                "accuracy": metrics.get("accuracy", 0),
                "precision": metrics.get("precision", 0),
                "recall": metrics.get("recall", 0),
                "f1_score": metrics.get("f1_score", 0),
                "roc_auc": metrics.get("roc_auc", 0)
            }
        
        comparison_df = pd.DataFrame(comparison_data).T
        
        logger.info("\n" + "="*50)
        logger.info("MODEL COMPARISON")
        logger.info("="*50)
        logger.info(f"\n{comparison_df[['accuracy', 'precision', 'recall', 'f1_score']].to_string()}")
        
        # Find best model based on F1-score
        best_model_name = max(self.results.items(), key=lambda x: x[1].get('f1_score', 0))
        best_model = best_model_name[0]
        best_metrics = best_model_name[1]
        
        logger.info(f"\n{'='*50}")
        logger.info(f"Best model based on F1-score: {best_model}")
        logger.info(f"F1-Score: {best_metrics['f1_score']:.4f}")
        logger.info(f"Accuracy: {best_metrics['accuracy']:.4f}")
        logger.info(f"Precision: {best_metrics['precision']:.4f}")
        logger.info(f"Recall: {best_metrics['recall']:.4f}")
        logger.info("="*50)
        
        return {
            "comparison": {k: v for k, v in self.results.items()},
            "best_model": best_model,
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
        logger.error("No models were successfully evaluated. Exiting.")
        sys.exit(1)
    
    # Save evaluation report
    report_path = Path("artifacts/reports/evaluation_report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(report_path, 'w') as f:
        json.dump(comparison_results, f, indent=2)
    
    # Create summary visualization
    try:
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Prepare data for plotting
        models = list(comparison_results['comparison'].keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        for idx, metric in enumerate(metrics):
            values = [comparison_results['comparison'][model].get(metric, 0) for model in models]
            bars = axes[idx].bar(models, values, color=['#1f77b4', '#2ca02c', '#ff7f0e'])
            axes[idx].set_title(f'{metric.capitalize()} Comparison', fontsize=14, fontweight='bold')
            axes[idx].set_ylabel(metric.capitalize(), fontsize=12)
            axes[idx].set_ylim([0, 1])
            axes[idx].grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for i, (bar, v) in enumerate(zip(bars, values)):
                axes[idx].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                             f'{v:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Ensure plots directory exists
        plots_dir = Path("artifacts/plots")
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        plt.savefig(plots_dir / 'model_comparison.png', dpi=100, bbox_inches='tight')
        logger.info("Model comparison plot saved to artifacts/plots/model_comparison.png")
        
    except Exception as e:
        logger.warning(f"Could not create visualization: {str(e)}")
    
    # Save best model info
    best_model_info = {
        "best_model": comparison_results['best_model'],
        "metrics": comparison_results['best_model_metrics'],
        "timestamp": str(pd.Timestamp.now()),
        "all_models_evaluated": list(comparison_results['comparison'].keys())
    }
    
    best_model_path = Path("artifacts/reports/best_model_info.json")
    with open(best_model_path, 'w') as f:
        json.dump(best_model_info, f, indent=2)
    
    logger.info(f"Evaluation completed successfully. Best model: {comparison_results['best_model']}")
    
    # Create a summary for GitHub Actions
    summary_file = Path("artifacts/reports/evaluation_summary.txt")
    with open(summary_file, 'w') as f:
        f.write("="*60 + "\n")
        f.write("MODEL EVALUATION SUMMARY\n")
        f.write("="*60 + "\n\n")
        f.write(f"Best Model: {comparison_results['best_model']}\n")
        f.write(f"F1-Score: {comparison_results['best_model_metrics']['f1_score']:.4f}\n")
        f.write(f"Accuracy: {comparison_results['best_model_metrics']['accuracy']:.4f}\n")
        f.write(f"Precision: {comparison_results['best_model_metrics']['precision']:.4f}\n")
        f.write(f"Recall: {comparison_results['best_model_metrics']['recall']:.4f}\n\n")
        f.write("All Models Performance:\n")
        f.write("-"*40 + "\n")
        for model, metrics in comparison_results['comparison'].items():
            f.write(f"\n{model}:\n")
            f.write(f"  Accuracy:  {metrics.get('accuracy', 0):.4f}\n")
            f.write(f"  Precision: {metrics.get('precision', 0):.4f}\n")
            f.write(f"  Recall:    {metrics.get('recall', 0):.4f}\n")
            f.write(f"  F1-Score:  {metrics.get('f1_score', 0):.4f}\n")
    
    logger.info(f"Evaluation summary saved to {summary_file}")

if __name__ == "__main__":
    main()
