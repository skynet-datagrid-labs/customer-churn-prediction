"""
Model Evaluation Module
Evaluates all trained models with comprehensive metrics
"""

import pickle
import json
import os
import sys
from pathlib import Path
import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, confusion_matrix,
                           classification_report)
from sklearn.model_selection import train_test_split
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelEvaluator:
    """Evaluate all trained models"""
    
    def __init__(self):
        self.models = {}
        self.metrics = {}
        self.X_test = None
        self.y_test = None
        
    def load_features(self, features_path: str = "artifacts/data/features.pkl"):
        """Load test features with error handling"""
        logger.info(f"Looking for features at: {features_path}")
        
        # Check if file exists
        if not os.path.exists(features_path):
            # Try alternative paths
            alt_paths = [
                "artifacts/data/features.pkl",
                "../artifacts/data/features.pkl",
                "./artifacts/data/features.pkl",
                "features.pkl"
            ]
            
            for alt_path in alt_paths:
                if os.path.exists(alt_path):
                    features_path = alt_path
                    logger.info(f"Found features at alternative path: {features_path}")
                    break
            else:
                # List directory contents for debugging
                logger.error("features.pkl not found! Listing directory contents:")
                if os.path.exists("artifacts/data/"):
                    logger.error(f"Contents of artifacts/data/: {os.listdir('artifacts/data/')}")
                else:
                    logger.error("artifacts/data/ directory doesn't exist!")
                
                raise FileNotFoundError(f"Features file not found at {features_path}")
        
        try:
            with open(features_path, 'rb') as f:
                features_data = pickle.load(f)
            
            logger.info(f"Successfully loaded features data with keys: {features_data.keys()}")
            
            X = features_data['X']
            y = features_data['y']
            
            logger.info(f"Original data shape: X={X.shape}, y={y.shape}")
            
            # Split data
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            logger.info(f"Train/Test split: Train={len(self.X_train)}, Test={len(self.X_test)}")
            return self.X_test, self.y_test
            
        except Exception as e:
            logger.error(f"Error loading features: {str(e)}")
            raise
    
    def load_models(self, models_path: str = "artifacts/models/"):
        """Load all trained models with error handling"""
        logger.info(f"Looking for models at: {models_path}")
        
        # Check if directory exists
        if not os.path.exists(models_path):
            logger.error(f"Models directory not found: {models_path}")
            # Try alternative path
            alt_path = "./artifacts/models/"
            if os.path.exists(alt_path):
                models_path = alt_path
                logger.info(f"Using alternative models path: {models_path}")
            else:
                raise FileNotFoundError(f"Models directory not found at {models_path} or {alt_path}")
        
        model_files = ['logistic_regression.pkl', 'random_forest.pkl', 'xgboost.pkl']
        
        logger.info(f"Scanning {models_path} for model files...")
        logger.info(f"Files in directory: {os.listdir(models_path)}")
        
        loaded_count = 0
        for model_file in model_files:
            model_path = os.path.join(models_path, model_file)
            if os.path.exists(model_path):
                try:
                    with open(model_path, 'rb') as f:
                        model_name = model_file.replace('.pkl', '')
                        self.models[model_name] = pickle.load(f)
                        logger.info(f"✅ Loaded {model_name} model from {model_path}")
                        loaded_count += 1
                except Exception as e:
                    logger.error(f"Failed to load {model_file}: {str(e)}")
            else:
                logger.warning(f"Model file not found: {model_path}")
        
        if loaded_count == 0:
            raise FileNotFoundError(f"No models found in {models_path}. Expected: {model_files}")
        
        logger.info(f"Successfully loaded {loaded_count}/{len(model_files)} models")
        return self.models
    
    def evaluate_model(self, model_name, model):
        """Evaluate a single model"""
        logger.info(f"Evaluating {model_name}")
        
        # Make predictions
        y_pred = model.predict(self.X_test)
        y_pred_proba = model.predict_proba(self.X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Calculate metrics
        metrics = {
            'accuracy': float(accuracy_score(self.y_test, y_pred)),
            'precision': float(precision_score(self.y_test, y_pred, average='binary', zero_division=0)),
            'recall': float(recall_score(self.y_test, y_pred, average='binary', zero_division=0)),
            'f1_score': float(f1_score(self.y_test, y_pred, average='binary', zero_division=0)),
        }
        
        if y_pred_proba is not None:
            metrics['roc_auc'] = float(roc_auc_score(self.y_test, y_pred_proba))
        
        # Confusion matrix
        cm = confusion_matrix(self.y_test, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        metrics['true_negatives'] = int(cm[0, 0])
        metrics['false_positives'] = int(cm[0, 1])
        metrics['false_negatives'] = int(cm[1, 0])
        metrics['true_positives'] = int(cm[1, 1])
        
        # Classification report
        report = classification_report(self.y_test, y_pred, output_dict=True, zero_division=0)
        metrics['classification_report'] = report
        
        logger.info(f"{model_name} - Accuracy: {metrics['accuracy']:.4f}, "
                   f"F1: {metrics['f1_score']:.4f}, AUC: {metrics.get('roc_auc', 0):.4f}")
        
        return metrics
    
    def evaluate_all_models(self):
        """Evaluate all loaded models"""
        for model_name, model in self.models.items():
            try:
                self.metrics[model_name] = self.evaluate_model(model_name, model)
            except Exception as e:
                logger.error(f"Failed to evaluate {model_name}: {str(e)}")
                self.metrics[model_name] = {
                    'accuracy': 0.0,
                    'precision': 0.0,
                    'recall': 0.0,
                    'f1_score': 0.0,
                    'error': str(e)
                }
        
        return self.metrics
    
    def compare_models(self):
        """Compare all models and generate comparison report"""
        comparison = {
            'best_model': None,
            'best_f1_score': 0,
            'model_ranking': []
        }
        
        for model_name, metrics in self.metrics.items():
            if 'error' not in metrics:
                comparison['model_ranking'].append({
                    'model': model_name,
                    'f1_score': metrics['f1_score'],
                    'accuracy': metrics['accuracy'],
                    'precision': metrics['precision'],
                    'recall': metrics['recall']
                })
                
                if metrics['f1_score'] > comparison['best_f1_score']:
                    comparison['best_f1_score'] = metrics['f1_score']
                    comparison['best_model'] = model_name
        
        # Sort by F1 score
        comparison['model_ranking'] = sorted(comparison['model_ranking'], 
                                            key=lambda x: x['f1_score'], 
                                            reverse=True)
        
        if comparison['best_model']:
            logger.info(f"Best model: {comparison['best_model']} with F1={comparison['best_f1_score']:.4f}")
        else:
            logger.warning("No valid models found for comparison")
        
        return comparison
    
    def save_evaluation_results(self, output_path: str = "artifacts/reports/evaluation_report.json"):
        """Save evaluation results"""
        try:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Custom JSON encoder for numpy types
            class NumpyEncoder(json.JSONEncoder):
                def default(self, obj):
                    if isinstance(obj, np.integer):
                        return int(obj)
                    if isinstance(obj, np.floating):
                        return float(obj)
                    if isinstance(obj, np.bool_):
                        return bool(obj)
                    if isinstance(obj, np.ndarray):
                        return obj.tolist()
                    return super(NumpyEncoder, self).default(obj)
            
            report = {
                'metrics': self.metrics,
                'comparison': self.compare_models(),
                'test_samples': len(self.X_test) if self.X_test is not None else 0,
                'class_distribution': self.y_test.value_counts().to_dict() if self.y_test is not None else {}
            }
            
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2, cls=NumpyEncoder)
            
            logger.info(f"Evaluation report saved to {output_path}")
            
            # Generate readable summary
            summary_path = output_path.replace('.json', '_summary.txt')
            with open(summary_path, 'w') as f:
                f.write("="*70 + "\n")
                f.write("MODEL EVALUATION REPORT\n")
                f.write("="*70 + "\n\n")
                
                if self.X_test is not None:
                    f.write(f"Test Samples: {len(self.X_test)}\n")
                    f.write(f"Class Distribution: {report['class_distribution']}\n\n")
                
                f.write("MODEL PERFORMANCE:\n")
                f.write("-"*70 + "\n")
                
                for model_name, metrics in self.metrics.items():
                    if 'error' in metrics:
                        f.write(f"\n{model_name.upper()}: FAILED - {metrics['error']}\n")
                    else:
                        f.write(f"\n{model_name.upper()}:\n")
                        f.write(f"  Accuracy:  {metrics['accuracy']:.4f}\n")
                        f.write(f"  Precision: {metrics['precision']:.4f}\n")
                        f.write(f"  Recall:    {metrics['recall']:.4f}\n")
                        f.write(f"  F1-Score:  {metrics['f1_score']:.4f}\n")
                        if 'roc_auc' in metrics:
                            f.write(f"  ROC-AUC:   {metrics['roc_auc']:.4f}\n")
                        f.write(f"\n  Confusion Matrix:\n")
                        f.write(f"    TN: {metrics['true_negatives']:4d}  FP: {metrics['false_positives']:4d}\n")
                        f.write(f"    FN: {metrics['false_negatives']:4d}  TP: {metrics['true_positives']:4d}\n")
                
                if report['comparison']['best_model']:
                    f.write("\n" + "="*70 + "\n")
                    f.write(f"BEST MODEL: {report['comparison']['best_model'].upper()}\n")
                    f.write(f"BEST F1-SCORE: {report['comparison']['best_f1_score']:.4f}\n")
                    f.write("\nMODEL RANKING (by F1-Score):\n")
                    for i, ranking in enumerate(report['comparison']['model_ranking'], 1):
                        f.write(f"  {i}. {ranking['model']}: {ranking['f1_score']:.4f}\n")
            
            return output_path
            
        except Exception as e:
            logger.error(f"Error saving evaluation results: {str(e)}")
            raise

def main():
    """Main execution"""
    try:
        logger.info("Starting model evaluation")
        
        # Debug: List directories
        logger.info("Current working directory: " + os.getcwd())
        logger.info("Directory contents: " + str(os.listdir('.')))
        
        if os.path.exists('artifacts/'):
            logger.info("artifacts/ contents: " + str(os.listdir('artifacts/')))
        if os.path.exists('artifacts/data/'):
            logger.info("artifacts/data/ contents: " + str(os.listdir('artifacts/data/')))
        if os.path.exists('artifacts/models/'):
            logger.info("artifacts/models/ contents: " + str(os.listdir('artifacts/models/')))
        
        evaluator = ModelEvaluator()
        evaluator.load_features()
        evaluator.load_models()
        evaluator.evaluate_all_models()
        evaluator.save_evaluation_results()
        
        # Print summary
        comparison = evaluator.compare_models()
        
        print("\n" + "="*60)
        print("MODEL EVALUATION SUMMARY")
        print("="*60)
        if comparison['best_model']:
            print(f"Best Model: {comparison['best_model'].upper()}")
            print(f"Best F1-Score: {comparison['best_f1_score']:.4f}")
            print("\nModel Rankings:")
            for i, ranking in enumerate(comparison['model_ranking'], 1):
                print(f"  {i}. {ranking['model']}: F1={ranking['f1_score']:.4f}, "
                      f"Acc={ranking['accuracy']:.4f}")
        else:
            print("No models were successfully evaluated!")
        
        print("\n✅ Model evaluation completed successfully!")
        
    except Exception as e:
        logger.error(f"Model evaluation failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
