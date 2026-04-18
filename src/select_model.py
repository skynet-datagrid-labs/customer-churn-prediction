"""
Model Selection Module
Selects the best performing model based on multiple criteria
"""

import json
import os
import sys
import shutil
from pathlib import Path
import pickle
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelSelector:
    """Select best model based on evaluation metrics"""
    
    def __init__(self):
        self.best_model = None
        self.best_model_name = None
        self.best_metrics = None
        
    def load_evaluation_report(self, report_path: str = "artifacts/reports/evaluation_report.json"):
        """Load evaluation results"""
        logger.info(f"Loading evaluation report from {report_path}")
        
        if not os.path.exists(report_path):
            raise FileNotFoundError(f"Evaluation report not found at {report_path}")
        
        with open(report_path, 'r') as f:
            self.report = json.load(f)
        
        self.comparison = self.report['comparison']
        logger.info(f"Loaded evaluation for {len(self.report['metrics'])} models")
        return self.report
    
    def select_best_model(self, criteria='f1_score'):
        """Select best model based on specified criteria"""
        
        best_score = 0
        best_model = None
        
        for model_name, metrics in self.report['metrics'].items():
            score = metrics.get(criteria, 0)
            
            if score > best_score:
                best_score = score
                best_model = model_name
        
        self.best_model_name = best_model
        self.best_metrics = self.report['metrics'][best_model]
        
        logger.info(f"Selected best model: {best_model} with {criteria}={best_score:.4f}")
        
        # Create selection report
        selection_report = {
            'best_model': best_model,
            'selection_criteria': criteria,
            'best_score': best_score,
            'all_scores': {
                model: metrics.get(criteria, 0) 
                for model, metrics in self.report['metrics'].items()
            },
            'metrics_for_best': self.best_metrics,
            'selection_rationale': f"Selected based on highest {criteria} score",
            'runner_up': self.comparison['model_ranking'][1] if len(self.comparison['model_ranking']) > 1 else None
        }
        
        return selection_report
    
    def load_best_model(self, models_path: str = "artifacts/models/"):
        """Load the best model from disk with better error handling"""
        
        # Debug: List what's in the models directory
        logger.info(f"Looking for models in: {models_path}")
        
        if not os.path.exists(models_path):
            logger.error(f"Models directory does not exist: {models_path}")
            # Try to find models in current directory
            alt_paths = ["./artifacts/models/", "../artifacts/models/", "artifacts/models/"]
            for alt in alt_paths:
                if os.path.exists(alt):
                    models_path = alt
                    logger.info(f"Found models at alternative path: {models_path}")
                    break
            else:
                raise FileNotFoundError(f"Models directory not found. Tried: {models_path} and alternatives")
        
        # List all files in the directory
        try:
            files = os.listdir(models_path)
            logger.info(f"Files in {models_path}: {files}")
        except Exception as e:
            logger.error(f"Could not list directory: {e}")
            files = []
        
        # Try different possible model file names
        possible_names = [
            f"{self.best_model_name}.pkl",
            f"{self.best_model_name}.pkl",
            f"model-{self.best_model_name}.pkl",
            f"{self.best_model_name}_model.pkl",
            "final_model.pkl"  # Fallback
        ]
        
        model_file = None
        for name in possible_names:
            test_path = os.path.join(models_path, name)
            if os.path.exists(test_path):
                model_file = test_path
                logger.info(f"Found model file: {model_file}")
                break
        
        if not model_file:
            # Try to find any .pkl file
            pkl_files = [f for f in files if f.endswith('.pkl')]
            if pkl_files:
                model_file = os.path.join(models_path, pkl_files[0])
                logger.info(f"Using first available .pkl file: {model_file}")
            else:
                raise FileNotFoundError(
                    f"No model file found for {self.best_model_name}. "
                    f"Directory contents: {files}"
                )
        
        try:
            with open(model_file, 'rb') as f:
                self.best_model = pickle.load(f)
            logger.info(f"Successfully loaded best model from {model_file}")
            return self.best_model
        except Exception as e:
            logger.error(f"Error loading model from {model_file}: {str(e)}")
            raise
    
    def save_best_model_info(self, output_path: str = "artifacts/models/best_model_info.json"):
        """Save information about the best model"""
        try:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Custom JSON encoder for numpy types
            class NumpyEncoder(json.JSONEncoder):
                def default(self, obj):
                    if hasattr(obj, 'tolist'):
                        return obj.tolist()
                    return super(NumpyEncoder, self).default(obj)
            
            best_info = {
                'best_model': self.best_model_name,
                'metrics': self.best_metrics,
                'selection_time': __import__('datetime').datetime.now().isoformat(),
                'model_type': type(self.best_model).__name__ if self.best_model else None,
                'model_parameters': str(self.best_model.get_params())[:500] if self.best_model else None
            }
            
            with open(output_path, 'w') as f:
                json.dump(best_info, f, indent=2, cls=NumpyEncoder)
            
            logger.info(f"Best model info saved to {output_path}")
            
            # Also create a simple text file
            txt_path = output_path.replace('.json', '.txt')
            with open(txt_path, 'w') as f:
                f.write(f"Best Model: {self.best_model_name}\n")
                f.write(f"F1 Score: {self.best_metrics.get('f1_score', 0):.4f}\n")
                f.write(f"Accuracy: {self.best_metrics.get('accuracy', 0):.4f}\n")
                f.write(f"Precision: {self.best_metrics.get('precision', 0):.4f}\n")
                f.write(f"Recall: {self.best_metrics.get('recall', 0):.4f}\n")
            
            return output_path
            
        except Exception as e:
            logger.error(f"Error saving best model info: {str(e)}")
            raise

def main():
    """Main execution"""
    try:
        logger.info("Starting model selection")
        
        # Debug current directory structure
        logger.info(f"Current working directory: {os.getcwd()}")
        logger.info(f"Directory contents: {os.listdir('.')}")
        
        if os.path.exists('artifacts'):
            logger.info(f"artifacts contents: {os.listdir('artifacts')}")
        if os.path.exists('artifacts/models'):
            logger.info(f"artifacts/models contents: {os.listdir('artifacts/models')}")
        if os.path.exists('artifacts/reports'):
            logger.info(f"artifacts/reports contents: {os.listdir('artifacts/reports')}")
        
        selector = ModelSelector()
        selector.load_evaluation_report()
        selection_report = selector.select_best_model(criteria='f1_score')
        selector.load_best_model()
        selector.save_best_model_info()
        
        # Print summary
        print("\n" + "="*60)
        print("MODEL SELECTION RESULTS")
        print("="*60)
        print(f"✅ Best Model Selected: {selection_report['best_model'].upper()}")
        print(f"Selection Criteria: {selection_report['selection_criteria']}")
        print(f"Best Score: {selection_report['best_score']:.4f}")
        print("\nBest Model Metrics:")
        print(f"  Accuracy:  {selection_report['metrics_for_best'].get('accuracy', 0):.4f}")
        print(f"  Precision: {selection_report['metrics_for_best'].get('precision', 0):.4f}")
        print(f"  Recall:    {selection_report['metrics_for_best'].get('recall', 0):.4f}")
        print(f"  F1-Score:  {selection_report['metrics_for_best'].get('f1_score', 0):.4f}")
        
        if selection_report.get('runner_up'):
            print(f"\nRunner-up: {selection_report['runner_up']['model']} "
                  f"(F1={selection_report['runner_up']['f1_score']:.4f})")
        
        print("\n✅ Model selection completed successfully!")
        
    except Exception as e:
        logger.error(f"Model selection failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
