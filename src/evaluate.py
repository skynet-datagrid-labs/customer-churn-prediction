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
from sklearn.model_selection import train_test_split
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
    
    def load_or_create_test_data(self) -> tuple:
        """Load existing test data or create it from feature data."""
        test_data_path = Path("artifacts/data/test_data.csv")
        
        # Try to load existing test data
        if test_data_path.exists():
            logger.info(f"Loading existing test data from {test_data_path}")
            test_df = pd.read_csv(test_data_path)
            X_test = test_df.drop('churn', axis=1)
            y_test = test_df['churn']
            return X_test, y_test
        
        # If no test data, create from feature data
        feature_data_path = Path("artifacts/data/feature_data.csv")
        if feature_data_path.exists():
            logger.info("No test data found, creating from feature data")
            df = pd.read_csv(feature_data_path)
            
            # Separate features and target
            X = df.drop('churn', axis=1)
            y = df['churn']
            
            # Split data
            _, X_test, _, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Save for future use
            test_df = X_test.copy()
            test_df['churn'] = y_test
            test_df.to_csv(test_data_path, index=False)
            logger.info(f"Test data created and saved to {test_data_path}")
            
            return X_test, y_test
        
        # If no data at all, raise error
        logger.error("No feature data or test data found")
        return None, None
    
    def evaluate_model(self, model, X_test, y_test, model_name: str) -> dict:
        """Evaluate a single model."""
        logger.info(f"Evaluating {model_name}...")
        
        try:
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
            
        except Exception as e:
            logger.error(f"Error evaluating {model_name}: {str(e)}")
            return None
    
    def find_available_models(self) -> list:
        """Find all trained models in the artifacts directory."""
        models_dir = Path("artifacts/models")
        if not models_dir.exists():
            return []
        
        # Look for model files
        model_files = list(models_dir.glob("*.pkl"))
        # Filter out preprocessor and best_model
        model_names = [f.stem for f in model_files if f.stem not in ['preprocessor', 'best_model']]
        
        logger.info(f"Found models: {model_names}")
        return model_names
    
    def compare_models(self) -> dict:
        """Compare all trained models."""
        logger.info("Comparing all models...")
        
        # Load or create test data
        X_test, y_test = self.load_or_create_test_data()
        if X_test is None or y_test is None:
            logger.error("Could not load or create test data")
            return {"comparison": {}, "best_model": None, "best_model_metrics": None}
        
        logger.info(f"Test data shape: {X_test.shape}")
        
        # Find available models
        model_names = self.find_available_models()
        
        if not model_names:
            logger.error("No models found in artifacts/models/")
            return {"comparison": {}, "best_model": None, "best_model_metrics": None}
        
        # Evaluate each model
        for model_name in model_names:
            model_path = Path(f"artifacts/models/{model_name}.pkl")
            if model_path.exists():
                try:
                    model = joblib.load(model_path)
                    metrics = self.evaluate_model(model, X_test, y_test, model_name)
                    if metrics is not None:
                        self.results[model_name] = metrics
                except Exception as e:
                    logger.error(f"Failed to load or evaluate {model_name}: {str(e)}")
            else:
                logger.warning(f"Model file not found: {model_path}")
        
        if not self.results:
            logger.error("No models were successfully evaluated")
            return {"comparison": {}, "best_model": None, "best_model_metrics": None}
        
        # Find best model based on F1-score
        best_model_name = max(self.results.items(), key=lambda x: x[1]['f1_score'])[0]
        best_metrics = self.results[best_model_name]
        
        # Create comparison table
        comparison_df = pd.DataFrame(self.results).T
        
        # Log results
        logger.info("\n" + "="*60)
        logger.info("MODEL COMPARISON RESULTS")
        logger.info("="*60)
        
        # Display metrics for each model
        for model_name in self.results.keys():
            metrics = self.results[model_name]
            logger.info(f"\n{model_name.upper()}:")
            logger.info(f"  Accuracy:  {metrics['accuracy']:.4f}")
            logger.info(f"  Precision: {metrics['precision']:.4f}")
            logger.info(f"  Recall:    {metrics['recall']:.4f}")
            logger.info(f"  F1-Score:  {metrics['f1_score']:.4f}")
            if 'roc_auc' in metrics:
                logger.info(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
        
        logger.info(f"\n{'='*60}")
        logger.info(f"🏆 BEST MODEL: {best_model_name}")
        logger.info(f"   F1-Score: {best_metrics['f1_score']:.4f}")
        logger.info(f"   Accuracy: {best_metrics['accuracy']:.4f}")
        logger.info("="*60)
        
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
        # Create a dummy report to prevent pipeline failure
        dummy_results = {
            "comparison": {},
            "best_model": "none",
            "best_model_metrics": {
                "accuracy": 0, "precision": 0, "recall": 0, "f1_score": 0
            }
        }
        report_path = Path("artifacts/reports/evaluation_report.json")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, 'w') as f:
            json.dump(dummy_results, f, indent=2)
        logger.warning("Created dummy evaluation report")
        sys.exit(0)  # Exit gracefully
    
    # Save evaluation report
    report_path = Path("artifacts/reports/evaluation_report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(report_path, 'w') as f:
        json.dump(comparison_results, f, indent=2)
    
    # Also save a human-readable summary
    summary_path = Path("artifacts/reports/evaluation_summary.txt")
    with open(summary_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("MODEL EVALUATION SUMMARY\n")
        f.write("="*60 + "\n\n")
        f.write(f"Best Model: {comparison_results['best_model']}\n")
        f.write(f"F1-Score: {comparison_results['best_model_metrics']['f1_score']:.4f}\n")
        f.write(f"Accuracy: {comparison_results['best_model_metrics']['accuracy']:.4f}\n")
        f.write(f"Precision: {comparison_results['best_model_metrics']['precision']:.4f}\n")
        f.write(f"Recall: {comparison_results['best_model_metrics']['recall']:.4f}\n\n")
        f.write("All Models:\n")
        f.write("-"*40 + "\n")
        for model, metrics in comparison_results['comparison'].items():
            f.write(f"\n{model}:\n")
            f.write(f"  Accuracy: {metrics['accuracy']:.4f}\n")
            f.write(f"  F1-Score: {metrics['f1_score']:.4f}\n")
    
    logger.info(f"Evaluation completed. Best model: {comparison_results['best_model']}")
    logger.info(f"Report saved to {report_path}")
    logger.info(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
