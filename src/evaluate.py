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
        """Load test features"""
        logger.info(f"Loading features from {features_path}")
        
        with open(features_path, 'rb') as f:
            features_data = pickle.load(f)
        
        from sklearn.model_selection import train_test_split
        X = features_data['X']
        y = features_data['y']
        
        _, self.X_test, _, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        logger.info(f"Loaded {len(self.X_test)} test samples")
        return self.X_test, self.y_test
    
    def load_models(self, models_path: str = "artifacts/models/"):
        """Load all trained models"""
        model_files = ['logistic_regression.pkl', 'random_forest.pkl', 'xgboost.pkl']
        
        for model_file in model_files:
            model_path = os.path.join(models_path, model_file)
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    model_name = model_file.replace('.pkl', '')
                    self.models[model_name] = pickle.load(f)
                    logger.info(f"Loaded {model_name} model")
        
        logger.info(f"Loaded {len(self.models)} models")
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
            'precision': float(precision_score(self.y_test, y_pred, average='binary')),
            'recall': float(recall_score(self.y_test, y_pred, average='binary')),
            'f1_score': float(f1_score(self.y_test, y_pred, average='binary')),
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
        report = classification_report(self.y_test, y_pred, output_dict=True)
        metrics['classification_report'] = report
        
        logger.info(f"{model_name} - Accuracy: {metrics['accuracy']:.4f}, "
                   f"F1: {metrics['f1_score']:.4f}, AUC: {metrics.get('roc_auc', 0):.4f}")
        
        return metrics
    
    def evaluate_all_models(self):
        """Evaluate all loaded models"""
        for model_name, model in self.models.items():
            self.metrics[model_name] = self.evaluate_model(model_name, model)
        
        return self.metrics
    
    def compare_models(self):
        """Compare all models and generate comparison report"""
        comparison = {
            'best_model': None,
            'best_f1_score': 0,
            'model_ranking': []
        }
        
        for model_name, metrics in self.metrics.items():
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
        
        logger.info(f"Best model: {comparison['best_model']} with F1={comparison['best_f1_score']:.4f}")
        
        return comparison
    
    def save_evaluation_results(self, output_path: str = "artifacts/reports/evaluation_report.json"):
        """Save evaluation results"""
        try:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            report = {
                'metrics': self.metrics,
                'comparison': self.compare_models(),
                'test_samples': len(self.X_test),
                'class_distribution': self.y_test.value_counts().to_dict()
            }
            
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            logger.info(f"Evaluation report saved to {output_path}")
            
            # Generate readable summary
            summary_path = output_path.replace('.json', '_summary.txt')
            with open(summary_path, 'w') as f:
                f.write("="*70 + "\n")
                f.write("MODEL EVALUATION REPORT\n")
                f.write("="*70 + "\n\n")
                
                f.write(f"Test Samples: {len(self.X_test)}\n")
                f.write(f"Class Distribution: {report['class_distribution']}\n\n")
                
                f.write("MODEL PERFORMANCE:\n")
                f.write("-"*70 + "\n")
                
                for model_name, metrics in self.metrics.items():
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
        print(f"Best Model: {comparison['best_model'].upper()}")
        print(f"Best F1-Score: {comparison['best_f1_score']:.4f}")
        print("\nModel Rankings:")
        for i, ranking in enumerate(comparison['model_ranking'], 1):
            print(f"  {i}. {ranking['model']}: F1={ranking['f1_score']:.4f}, "
                  f"Acc={ranking['accuracy']:.4f}")
        
        print("\n✅ Model evaluation completed successfully!")
        
    except Exception as e:
        logger.error(f"Model evaluation failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
