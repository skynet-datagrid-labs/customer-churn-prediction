"""Model selection module."""

import sys
import os
import argparse
import json
import shutil
from pathlib import Path
import pandas as pd
import joblib

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import logger

class ModelSelector:
    """Select and promote the best model."""
    
    def __init__(self):
        self.evaluation_report_path = Path("artifacts/reports/evaluation_report.json")
        self.models_dir = Path("artifacts/models")
        
    def select_best_model(self) -> dict:
        """Select the best model based on evaluation metrics."""
        logger.info("Selecting best model...")
        
        # Load evaluation report
        if not self.evaluation_report_path.exists():
            logger.error("Evaluation report not found")
            return None
        
        with open(self.evaluation_report_path, 'r') as f:
            eval_report = json.load(f)
        
        best_model_name = eval_report['best_model']
        best_model_metrics = eval_report['best_model_metrics']
        
        logger.info(f"Selected best model: {best_model_name}")
        logger.info(f"Metrics: {best_model_metrics}")
        
        # Copy best model to standard location
        source_path = self.models_dir / f"{best_model_name}.pkl"
        dest_path = self.models_dir / "best_model.pkl"
        
        if source_path.exists():
            shutil.copy2(source_path, dest_path)
            logger.info(f"Copied {best_model_name} to best_model.pkl")
        else:
            logger.error(f"Best model file not found: {source_path}")
        
        # Save selection info
        selection_info = {
            "best_model": best_model_name,
            "metrics": best_model_metrics,
            "selection_criteria": "f1_score",
            "timestamp": str(pd.Timestamp.now()),
            "model_path": str(dest_path)
        }
        
        selection_path = Path("artifacts/reports/model_selection.json")
        with open(selection_path, 'w') as f:
            json.dump(selection_info, f, indent=2)
        
        return selection_info
    
    def compare_with_production(self) -> dict:
        """Compare new model with production model."""
        logger.info("Comparing with production model...")
        
        prod_model_path = self.models_dir / "best_model.pkl"
        new_model_path = self.models_dir / "best_model_new.pkl"
        
        if not prod_model_path.exists():
            logger.info("No production model found, will deploy new model")
            return {"new_model_better": True, "improvement": 1.0}
        
        # Load metrics for both models
        selection_info_path = Path("artifacts/reports/model_selection.json")
        if selection_info_path.exists():
            with open(selection_info_path, 'r') as f:
                new_model_info = json.load(f)
            new_f1 = new_model_info['metrics']['f1_score']
        else:
            # Evaluate new model
            from src.evaluate import ModelEvaluator
            evaluator = ModelEvaluator()
            test_data_path = Path("artifacts/data/test_data.csv")
            
            if not test_data_path.exists():
                # Create test data from feature data
                df = pd.read_csv("artifacts/data/feature_data.csv")
                from sklearn.model_selection import train_test_split
                X = df.drop('churn', axis=1)
                y = df['churn']
                _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                test_df = X_test.copy()
                test_df['churn'] = y_test
                test_df.to_csv(test_data_path, index=False)
            
            new_model = joblib.load(new_model_path)
            test_df = pd.read_csv(test_data_path)
            X_test = test_df.drop('churn', axis=1)
            y_test = test_df['churn']
            from sklearn.metrics import f1_score
            y_pred = new_model.predict(X_test)
            new_f1 = f1_score(y_test, y_pred)
        
        # Load production metrics
        prod_metrics_path = Path("artifacts/reports/best_model_info.json")
        if prod_metrics_path.exists():
            with open(prod_metrics_path, 'r') as f:
                prod_info = json.load(f)
            prod_f1 = prod_info['metrics']['f1_score']
        else:
            prod_model = joblib.load(prod_model_path)
            test_df = pd.read_csv("artifacts/data/test_data.csv")
            X_test = test_df.drop('churn', axis=1)
            y_test = test_df['churn']
            y_pred = prod_model.predict(X_test)
            from sklearn.metrics import f1_score
            prod_f1 = f1_score(y_test, y_pred)
        
        improvement = (new_f1 - prod_f1) / prod_f1
        new_model_better = new_f1 > prod_f1
        
        logger.info(f"Production F1: {prod_f1:.4f}")
        logger.info(f"New Model F1: {new_f1:.4f}")
        logger.info(f"Improvement: {improvement:.2%}")
        
        comparison = {
            "new_model_better": new_model_better,
            "improvement": improvement,
            "prod_f1": prod_f1,
            "new_f1": new_f1
        }
        
        # Save comparison
        comparison_path = Path("artifacts/reports/model_comparison.json")
        with open(comparison_path, 'w') as f:
            json.dump(comparison, f, indent=2)
        
        return comparison

def main():
    """Main model selection entry point."""
    parser = argparse.ArgumentParser(description="Select best model")
    parser.add_argument("--retraining", action="store_true", help="Running in retraining mode")
    parser.add_argument("--compare", action="store_true", help="Compare with production model")
    
    args = parser.parse_args()
    
    selector = ModelSelector()
    
    if args.retraining and args.compare:
        # Compare new model with production
        comparison = selector.compare_with_production()
        
        if comparison['new_model_better']:
            # Promote new model
            shutil.copy2("artifacts/models/best_model_new.pkl", "artifacts/models/best_model.pkl")
            logger.info("New model promoted to production")
            
            # Create deploy flag
            with open("artifacts/deploy_flag", 'w') as f:
                f.write("deploy")
        else:
            logger.info("New model not better than production, keeping current model")
    else:
        # Regular selection
        selection_info = selector.select_best_model()
        
        if selection_info:
            logger.info(f"Best model selected: {selection_info['best_model']}")
        else:
            logger.error("Model selection failed")
            sys.exit(1)
    
    logger.info("Model selection completed")

if __name__ == "__main__":
    main()
