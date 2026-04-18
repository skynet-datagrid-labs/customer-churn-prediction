"""Model selection module."""

import sys
import os
import argparse
import json
import shutil
from pathlib import Path
import pandas as pd
import joblib
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelSelector:
    """Select and promote the best model."""
    
    def __init__(self):
        self.evaluation_report_path = Path("artifacts/reports/evaluation_report.json")
        self.models_dir = Path("artifacts/models")
        self.models_dir.mkdir(parents=True, exist_ok=True)
    
    def select_best_model(self) -> dict:
        """Select the best model based on evaluation metrics."""
        logger.info("Selecting best model...")
        
        if not self.evaluation_report_path.exists():
            logger.error("Evaluation report not found")
            return None
        
        with open(self.evaluation_report_path, 'r') as f:
            eval_report = json.load(f)
        
        best_model_name = eval_report.get('best_model')
        best_model_metrics = eval_report.get('best_model_metrics', {})
        
        if not best_model_name:
            logger.error("No best model identified")
            return None
        
        logger.info(f"Selected best model: {best_model_name}")
        logger.info(f"F1-Score: {best_model_metrics.get('f1_score', 0):.4f}")
        
        # Copy best model to standard location
        source_path = self.models_dir / f"{best_model_name}.pkl"
        dest_path = self.models_dir / "best_model.pkl"
        
        if source_path.exists():
            shutil.copy2(source_path, dest_path)
            logger.info(f"Copied {best_model_name} to best_model.pkl")
        else:
            logger.error(f"Best model file not found: {source_path}")
            return None
        
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


def main():
    """Main model selection entry point."""
    parser = argparse.ArgumentParser(description="Select best model")
    parser.add_argument("--retraining", action="store_true", help="Running in retraining mode")
    parser.add_argument("--compare", action="store_true", help="Compare with production model")
    
    args = parser.parse_args()
    
    selector = ModelSelector()
    selection_info = selector.select_best_model()
    
    if selection_info:
        logger.info(f"Best model selected: {selection_info['best_model']}")
    else:
        logger.error("Model selection failed")
        sys.exit(1)
    
    logger.info("Model selection completed")


if __name__ == "__main__":
    main()
