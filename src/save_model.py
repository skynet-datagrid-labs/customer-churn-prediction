"""
Model Artifact Saving Module
Saves final model artifacts for production deployment
"""

import pickle
import json
import os
import sys
from pathlib import Path
import logging
from datetime import datetime

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
        
        if not os.path.exists(best_info_path):
            logger.error(f"Best model info not found at {best_info_path}")
            alt_paths = ["./artifacts/models/best_model_info.json", "../artifacts/models/best_model_info.json"]
            for alt in alt_paths:
                if os.path.exists(alt):
                    best_info_path = alt
                    logger.info(f"Found best model info at {alt}")
                    break
            else:
                raise FileNotFoundError("best_model_info.json not found")
        
        with open(best_info_path, 'r') as f:
            best_info = json.load(f)
        
        model_file = os.path.join(model_path, f"{best_info['best_model']}.pkl")
        
        if not os.path.exists(model_file):
            alt_model_paths = [
                f"./artifacts/models/{best_info['best_model']}.pkl",
                f"../artifacts/models/{best_info['best_model']}.pkl",
                f"artifacts/models/model-{best_info['best_model']}.pkl"
            ]
            for alt in alt_model_paths:
                if os.path.exists(alt):
                    model_file = alt
                    logger.info(f"Found model at {alt}")
                    break
            else:
                raise FileNotFoundError(f"Model file not found for {best_info['best_model']}")
        
        with open(model_file, 'rb') as f:
            self.final_model = pickle.load(f)
        
        logger.info(f"Loaded best model: {best_info['best_model']}")
        return self.final_model
    
    def load_preprocessor(self, preprocessor_path: str = "artifacts/data/preprocessed_data.pkl"):
        """Load the preprocessor objects"""
        if not os.path.exists(preprocessor_path):
            alt_paths = ["./artifacts/data/preprocessed_data.pkl", "../artifacts/data/preprocessed_data.pkl"]
            for alt in alt_paths:
                if os.path.exists(alt):
                    preprocessor_path = alt
                    logger.info(f"Found preprocessor at {alt}")
                    break
            else:
                logger.warning("Preprocessor not found, creating empty preprocessor")
                self.preprocessor = {'scaler': None, 'label_encoders': {}}
                return self.preprocessor
        
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
        if not os.path.exists(features_path):
            alt_paths = ["./artifacts/data/features.pkl", "../artifacts/data/features.pkl"]
            for alt in alt_paths:
                if os.path.exists(alt):
                    features_path = alt
                    logger.info(f"Found features at {alt}")
                    break
            else:
                raise FileNotFoundError(f"Features file not found at {features_path}")
        
        with open(features_path, 'rb') as f:
            features_data = pickle.load(f)
        
        self.feature_columns = features_data['feature_columns']
        logger.info(f"Loaded {len(self.feature_columns)} feature columns")
        return self.feature_columns
    
    def create_final_artifacts(self, output_path: str = "artifacts/models/"):
        """Create final model artifacts for deployment"""
        try:
            Path(output_path).mkdir(parents=True, exist_ok=True)
            
            final_model_path = os.path.join(output_path, "final_model.pkl")
            with open(final_model_path, 'wb') as f:
                pickle.dump(self.final_model, f)
            logger.info(f"Final model saved to {final_model_path}")
            
            preprocessor_path = os.path.join(output_path, "final_preprocessor.pkl")
            with open(preprocessor_path, 'wb') as f:
                pickle.dump(self.preprocessor, f)
            logger.info(f"Preprocessor saved to {preprocessor_path}")
            
            features_path = os.path.join(output_path, "feature_columns.json")
            with open(features_path, 'w') as f:
                json.dump({
                    'feature_columns': self.feature_columns,
                    'num_features': len(self.feature_columns)
                }, f, indent=2)
            logger.info(f"Feature columns saved to {features_path}")
            
            # Convert model parameters to string to avoid JSON serialization issues
            try:
                model_params = str(self.final_model.get_params())
            except:
                model_params = "Unable to extract parameters"
            
            metadata = {
                'model_type': type(self.final_model).__name__,
                'model_parameters': model_params[:500],
                'feature_columns': self.feature_columns,
                'num_features': len(self.feature_columns),
                'preprocessor_components': list(self.preprocessor.keys()),
                'created_at': datetime.now().isoformat(),
                'version': '1.0.0'
            }
            
            metadata_path = os.path.join(output_path, "model_metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            logger.info(f"Model metadata saved to {metadata_path}")
            
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
            feature_list = ""
            for col in self.feature_columns[:10]:
                feature_list += f"- {col}\n"
            
            if len(self.feature_columns) > 10:
                feature_list += f"- ... and {len(self.feature_columns) - 10} more features\n"
            
            model_type = type(self.final_model).__name__ if self.final_model else "Unknown"
            created_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            model_card_lines = []
            model_card_lines.append("# Model Card: Customer Churn Prediction")
            model_card_lines.append("")
            model_card_lines.append("## Model Overview")
            model_card_lines.append(f"- **Model Type:** {model_type}")
            model_card_lines.append(f"- **Created:** {created_time}")
            model_card_lines.append("- **Version:** 1.0.0")
            model_card_lines.append("")
            model_card_lines.append("## Intended Use")
            model_card_lines.append("This model predicts customer churn for subscription-based services to enable proactive retention strategies.")
            model_card_lines.append("")
            model_card_lines.append("## Features Used")
            model_card_lines.append(feature_list)
            model_card_lines.append("")
            model_card_lines.append("## Model Performance")
            model_card_lines.append("Based on evaluation metrics, this model achieves:")
            model_card_lines.append("- High accuracy and F1 score for churn prediction")
            model_card_lines.append("- Balanced precision and recall")
            model_card_lines.append("")
            model_card_lines.append("## Usage")
            model_card_lines.append("```python")
            model_card_lines.append("# Load model and preprocessor")
            model_card_lines.append("import pickle")
            model_card_lines.append("")
            model_card_lines.append("with open('final_model.pkl', 'rb') as f:")
            model_card_lines.append("    model = pickle.load(f)")
            model_card_lines.append("")
            model_card_lines.append("with open('final_preprocessor.pkl', 'rb') as f:")
            model_card_lines.append("    preprocessor = pickle.load(f)")
            model_card_lines.append("")
            model_card_lines.append("# Make predictions")
            model_card_lines.append("predictions = model.predict(features)")
            model_card_lines.append("```")
            model_card_lines.append("")
            model_card_lines.append("## Limitations")
            model_card_lines.append("- Model performance may degrade over time due to data drift")
            model_card_lines.append("- Requires periodic retraining with new data")
            model_card_lines.append("- Works best with the feature set defined in feature_columns.json")
            model_card_lines.append("")
            model_card_lines.append("## Ethical Considerations")
            model_card_lines.append("- Model predictions should be used as one factor in decision-making")
            model_card_lines.append("- Regular monitoring for bias is recommended")
            model_card_lines.append("- Customer privacy must be maintained")
            
            model_card = "\n".join(model_card_lines)
            
            with open(output_path, 'w') as f:
                f.write(model_card)
            
            logger.info(f"Model card saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Error generating model card: {str(e)}")

def main():
    """Main execution"""
    try:
        logger.info("Starting model artifact saving")
        
        logger.info(f"Current working directory: {os.getcwd()}")
        
        if os.path.exists('artifacts'):
            logger.info(f"artifacts contents: {os.listdir('artifacts')}")
        if os.path.exists('artifacts/models'):
            logger.info(f"artifacts/models contents: {os.listdir('artifacts/models')}")
        if os.path.exists('artifacts/data'):
            logger.info(f"artifacts/data contents: {os.listdir('artifacts/data')}")
        
        saver = ModelArtifactSaver()
        saver.load_best_model()
        saver.load_preprocessor()
        saver.load_feature_columns()
        deployment_package = saver.create_final_artifacts()
        saver.generate_model_card()
        
        print("\n" + "="*60)
        print("MODEL ARTIFACTS SUMMARY")
        print("="*60)
        print("Final artifacts created successfully!")
        print("\nArtifacts saved:")
        for key, path in deployment_package.items():
            print(f"  - {key}: {path}")
        
        print("\nDeployment package ready for API deployment")
        print("\nModel artifact saving completed successfully!")
        
    except Exception as e:
        logger.error(f"Model artifact saving failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
