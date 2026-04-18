"""Model saving and artifact management module."""

import sys
import os
import argparse
import json
import hashlib
from pathlib import Path
import pandas as pd
import joblib
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ArtifactManager:
    """Manage and save all model artifacts."""
    
    def __init__(self):
        self.artifacts_dir = Path("artifacts")
        self.models_dir = self.artifacts_dir / "models"
        self.metrics_dir = self.artifacts_dir / "metrics"
        self.reports_dir = self.artifacts_dir / "reports"
        
        for dir_path in [self.models_dir, self.metrics_dir, self.reports_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def save_model_metadata(self) -> dict:
        """Save metadata about the model."""
        model_path = self.models_dir / "best_model.pkl"
        
        if not model_path.exists():
            logger.error(f"Model file not found: {model_path}")
            return None
        
        # Calculate model hash
        with open(model_path, 'rb') as f:
            model_bytes = f.read()
            model_hash = hashlib.sha256(model_bytes).hexdigest()
        
        model_size = model_path.stat().st_size / (1024 * 1024)
        
        metadata = {
            "model_name": "best_model",
            "model_path": str(model_path),
            "model_hash": model_hash,
            "model_size_mb": model_size,
            "created_at": str(datetime.now()),
            "python_version": sys.version
        }
        
        # Load model info if available
        info_path = self.reports_dir / "evaluation_report.json"
        if info_path.exists():
            with open(info_path, 'r') as f:
                model_info = json.load(f)
            metadata["metrics"] = model_info.get("best_model_metrics", {})
        
        metadata_path = self.reports_dir / "best_model_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Model metadata saved")
        return metadata


def main():
    """Main artifact saving entry point."""
    parser = argparse.ArgumentParser(description="Save model artifacts")
    
    args = parser.parse_args()
    
    manager = ArtifactManager()
    
    # Save model metadata
    metadata = manager.save_model_metadata()
    
    if metadata:
        logger.info("All artifacts saved successfully")
    else:
        logger.error("Failed to save artifacts")
        sys.exit(1)


if __name__ == "__main__":
    main()
