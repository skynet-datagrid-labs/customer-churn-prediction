"""Model saving and artifact management module."""

import sys
import os
import argparse
import json
import shutil
from pathlib import Path
import pandas as pd
import joblib
import hashlib
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import logger

class ArtifactManager:
    """Manage and save all model artifacts."""
    
    def __init__(self):
        self.artifacts_dir = Path("artifacts")
        self.models_dir = self.artifacts_dir / "models"
        self.metrics_dir = self.artifacts_dir / "metrics"
        self.reports_dir = self.artifacts_dir / "reports"
        
        # Create directories
        for dir_path in [self.models_dir, self.metrics_dir, self.reports_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def save_model_metadata(self, model_name: str = "best_model") -> dict:
        """Save metadata about the model."""
        model_path = self.models_dir / f"{model_name}.pkl"
        
        if not model_path.exists():
            logger.error(f"Model file not found: {model_path}")
            return None
        
        # Calculate model hash
        with open(model_path, 'rb') as f:
            model_bytes = f.read()
            model_hash = hashlib.sha256(model_bytes).hexdigest()
        
        # Get model file size
        model_size = model_path.stat().st_size / (1024 * 1024)  # MB
        
        metadata = {
            "model_name": model_name,
            "model_path": str(model_path),
            "model_hash": model_hash,
            "model_size_mb": model_size,
            "created_at": str(datetime.now()),
            "python_version": sys.version,
            "dependencies": self._get_dependencies()
        }
        
        # Load model info if available
        info_path = self.reports_dir / "best_model_info.json"
        if info_path.exists():
            with open(info_path, 'r') as f:
                model_info = json.load(f)
            metadata["metrics"] = model_info.get("metrics", {})
        
        # Save metadata
        metadata_path = self.reports_dir / f"{model_name}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Model metadata saved to {metadata_path}")
        return metadata
    
    def _get_dependencies(self) -> dict:
        """Get package dependencies."""
        try:
            import pkg_resources
            packages = {}
            for dist in pkg_resources.working_set:
                packages[dist.key] = dist.version
            return packages
        except:
            return {}
    
    def save_feature_importance(self, model_name: str = "best_model"):
        """Save feature importance if available."""
        model_path = self.models_dir / f"{model_name}.pkl"
        
        if not model_path.exists():
            logger.warning(f"Model not found for feature importance: {model_path}")
            return
        
        model = joblib.load(model_path)
        
        # Load feature names
        feature_info_path = self.reports_dir / "feature_info_enhanced.json"
        if feature_info_path.exists():
            with open(feature_info_path, 'r') as f:
                feature_info = json.load(f)
            feature_names = feature_info.get('feature_names', [])
        else:
            # Try to get from training data
            data_path = Path("artifacts/data/feature_data.csv")
            if data_path.exists():
                df = pd.read_csv(data_path)
                feature_names = [col for col in df.columns if col != 'churn']
            else:
                feature_names = [f"feature_{i}" for i in range(10)]
        
        # Get feature importance
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            importance_df = pd.DataFrame({
                'feature': feature_names[:len(importances)],
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            # Save importance
            importance_path = self.reports_dir / "feature_importance.json"
            importance_df.to_json(importance_path, orient='records', indent=2)
            
            # Save top features plot
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            plt.figure(figsize=(10, 6))
            top_features = importance_df.head(15)
            sns.barplot(data=top_features, x='importance', y='feature')
            plt.title('Top 15 Feature Importances')
            plt.xlabel('Importance')
            plt.tight_layout()
            plt.savefig(self.artifacts_dir / "plots" / "feature_importance.png", dpi=100)
            
            logger.info(f"Feature importance saved for {len(importances)} features")
        elif hasattr(model, 'coef_'):
            # For linear models
            coefficients = model.coef_[0] if len(model.coef_.shape) > 1 else model.coef_
            coef_df = pd.DataFrame({
                'feature': feature_names[:len(coefficients)],
                'coefficient': coefficients
            }).sort_values('coefficient', key=abs, ascending=False)
            
            coef_path = self.reports_dir / "feature_coefficients.json"
            coef_df.to_json(coef_path, orient='records', indent=2)
            logger.info(f"Feature coefficients saved")
    
    def create_model_card(self, model_name: str = "best_model"):
        """Create a model card with all relevant information."""
        metadata_path = self.reports_dir / f"{model_name}_metadata.json"
        
        if not metadata_path.exists():
            logger.warning("Metadata not found, cannot create model card")
            return
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        model_card = f"""
# Model Card: {model_name}

## Model Details
- **Model Name**: {metadata.get('model_name')}
- **Created**: {metadata.get('created_at')}
- **Model Size**: {metadata.get('model_size_mb', 0):.2f} MB
- **Model Hash**: {metadata.get('model_hash', 'N/A')[:16]}...

## Performance Metrics
"""
        
        if 'metrics' in metadata:
            for metric, value in metadata['metrics'].items():
                if metric != 'confusion_matrix' and metric != 'classification_report':
                    model_card += f"- **{metric}**: {value:.4f}\n"
        
        model_card += f"""
## Usage
- **Input Features**: {metadata.get('feature_count', 'N/A')} features
- **Output**: Binary classification (churn: 0 or 1)

## Training Details
- **Python Version**: {metadata.get('python_version', 'N/A')}
- **Dependencies**: {len(metadata.get('dependencies', {}))} packages

## Ethical Considerations
This model predicts customer churn based on historical data. 
It should be used as a decision support tool, not as the sole 
determinant for customer retention actions.
"""
        
        # Save model card
        card_path = self.reports_dir / f"{model_name}_card.md"
        with open(card_path, 'w') as f:
            f.write(model_card)
        
        logger.info(f"Model card saved to {card_path}")

def main():
    """Main artifact saving entry point."""
    parser = argparse.ArgumentParser(description="Save model artifacts")
    parser.add_argument("--model", default="best_model", help="Model name to save")
    
    args = parser.parse_args()
    
    manager = ArtifactManager()
    
    # Save model metadata
    metadata = manager.save_model_metadata(args.model)
    
    # Save feature importance
    manager.save_feature_importance(args.model)
    
    # Create model card
    manager.create_model_card(args.model)
    
    # Create final manifest
    manifest = {
        "artifacts": {
            "models": [str(p) for p in manager.models_dir.glob("*.pkl")],
            "metrics": [str(p) for p in manager.metrics_dir.glob("*.json")],
            "reports": [str(p) for p in manager.reports_dir.glob("*.json")],
            "plots": [str(p) for p in (manager.artifacts_dir / "plots").glob("*")] if (manager.artifacts_dir / "plots").exists() else []
        },
        "timestamp": str(datetime.now()),
        "total_artifacts": 0
    }
    
    for category in manifest['artifacts']:
        manifest['total_artifacts'] += len(manifest['artifacts'][category])
    
    manifest_path = manager.artifacts_dir / "manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    logger.info(f"Artifacts manifest saved with {manifest['total_artifacts']} items")
    logger.info("All artifacts saved successfully")

if __name__ == "__main__":
    main()
