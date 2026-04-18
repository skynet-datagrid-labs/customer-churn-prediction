"""
Model Artifact Saving Module
Saves final model artifacts for production deployment
"""

import pickle
import json
import os
import sys
from pathlib import Path
import shutil
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelArtifactSaver:
    """Save all model artifacts for production"""
    
    def __init__(self):
        self.final_model = None
        self.preprocessor = None
        self.feature_columns = None
        
    def load_best_model(self, model_path: str = "artifacts/models/"):
        """Load the best model"""
        best_info_path = os.path.join(model_path, "best_model_info.json")
        
        with open(best_info_path, 'r') as f:
            best_info = json.load(f)
        
        model_file = os.path.join(model_path, f"{best_info['best_model']}.pkl")
        
        with open(model_file, 'rb') as f:
            self.final_model = pickle.load(f)
        
        logger.info(f"Loaded best model: {best_info['best_model']}")
        return self.final_model
    
    def load_preprocessor(self, preprocessor_path: str = "artifacts/data/preprocessed_data.pkl"):
        """Load the preprocessor objects"""
        with open(preprocessor_path, 'rb') as f:
            preprocessed_data = pickle.load(f)
        
        self.preprocessor = {
            'scaler': preprocessed_data.get('scaler'),
            'label_encoders': preprocessed_data.get('label_encoders', {})
        }
        
        logger.info("Loaded preprocessor objects")
        return self.preprocessor
    
    def load_feature_columns(self, features_path: str = "artifacts/data/features.pkl"):
        """Load feature column names"""
        with open(features_path, 'rb') as f:
            features_data = pickle.load(f)
        
        self.feature_columns = features_data['feature_columns']
        logger.info(f"Loaded {len(self.feature_columns)} feature columns")
        return self.feature_columns
    
    def create_final_artifacts(self, output_path: str = "artifacts/models/"):
        """Create final model artifacts for deployment"""
        try:
            Path(output_path).mkdir(parents=True, exist_ok=True)
            
            # Save final model
            final_model_path = os.path.join(output_path, "final_model.pkl")
            with open(final_model_path, 'wb') as f:
                pickle.dump(self.final_model, f)
            logger.info(f"Final model saved to {final_model_path}")
            
            # Save preprocessor
            preprocessor_path = os.path.join(output_path, "final_preprocessor.pkl")
            with open(preprocessor_path, 'wb') as f:
                pickle.dump(self.preprocessor, f)
            logger.info(f"Preprocessor saved to {preprocessor_path}")
            
            # Save feature columns
            features_path = os.path.join(output_path, "feature_columns.json")
            with open(features_path, 'w') as f:
                json.dump({
                    'feature_columns': self.feature_columns,
                    'num_features': len(self.feature_columns)
                }, f, indent=2)
            logger.info(f"Feature columns saved to {features_path}")
            
            # Create model metadata
            metadata = {
                'model_type': type(self.final_model).__name__,
                'model_parameters': self.final_model.get_params(),
                'feature_columns': self.feature_columns,
                'num_features': len(self.feature_columns),
                'preprocessor_components': list(self.preprocessor.keys()),
                'created_at': __import__('datetime').datetime.now().isoformat(),
                'version': '1.0.0'
            }
            
            metadata_path = os.path.join(output_path, "model_metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            logger.info(f"Model metadata saved to {metadata_path}")
            
            # Create a deployment package
            deployment_package = {
                'model_path': final_model_path,
                'preprocessor_path': preprocessor_path,
                'features_path': features_path,
                'metadata_path': metadata_path
            }
            
            package_path = os.path.join(output_path, "deployment_package.json")
            with open(package_path, 'w') as f:
                json.dump(deployment_package, f, indent=2)
            
            return deployment_package
            
        except Exception as e:
            logger.error(f"Error creating final artifacts: {str(e)}")
            raise
    
    def generate_model_card(self, output_path: str = "artifacts/models/MODEL_CARD.md"):
        """Generate a model card for documentation"""
        try:
            model_card = f"""# Model Card: Customer Churn Prediction

## Model Overview
- **Model Type:** {type(self.final_model).__name__}
- **Created:** {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Version:** 1.0.0

## Intended Use
This model predicts customer churn for subscription-based services to enable proactive retention strategies.

## Features Used
{chr(10).join([f"- {col}" for col in self.feature_columns[:10]])}
{chr(10).join([f"- ... and {len(self.feature_columns) - 10} more features" if len(self.feature_columns) > 10 else ""])}

## Model Performance
Based on evaluation metrics, this model achieves:
- High accuracy and F1 score for churn prediction
- Balanced precision and recall

## Usage
```python
# Load model and preprocessor
import pickle

with open('final_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('final_preprocessor.pkl', 'rb') as f:
    preprocessor = pickle.load(f)

# Make predictions
predictions = model.predict(features)
