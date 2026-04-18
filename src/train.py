"""
Model Training Module
Trains multiple ML models with hyperparameter tuning
"""

import pandas as pd
import numpy as np
import pickle
import json
import os
import sys
import argparse
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, cross_val_score
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelTrainer:
    """Train individual ML models"""
    
    def __init__(self, model_name):
        self.model_name = model_name
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def load_features(self, features_path: str = "artifacts/data/features.pkl"):
        """Load engineered features"""
        logger.info(f"Loading features from {features_path}")
        
        with open(features_path, 'rb') as f:
            features_data = pickle.load(f)
        
        X = features_data['X']
        y = features_data['y']
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        logger.info(f"Data split: Train={len(self.X_train)}, Test={len(self.X_test)}")
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def get_model(self):
        """Get model instance based on model name"""
        if self.model_name == 'logistic_regression':
            self.model = LogisticRegression(
                C=1.0,
                max_iter=1000,
                random_state=42,
                class_weight='balanced'
            )
        elif self.model_name == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                class_weight='balanced',
                n_jobs=-1
            )
        elif self.model_name == 'xgboost':
            self.model = XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                scale_pos_weight=1,
                use_label_encoder=False,
                eval_metric='logloss'
            )
        else:
            raise ValueError(f"Unknown model: {self.model_name}")
        
        logger.info(f"Initialized {self.model_name} model")
        return self.model
    
    def train(self):
        """Train the model"""
        logger.info(f"Training {self.model_name}")
        
        self.model.fit(self.X_train, self.y_train)
        
        # Calculate training metrics
        train_score = self.model.score(self.X_train, self.y_train)
        test_score = self.model.score(self.X_test, self.y_test)
        
        logger.info(f"Training accuracy: {train_score:.4f}")
        logger.info(f"Testing accuracy: {test_score:.4f}")
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, self.X_train, self.y_train, cv=5)
        logger.info(f"Cross-validation scores: {cv_scores}")
        logger.info(f"CV mean: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        return {
            'train_accuracy': train_score,
            'test_accuracy': test_score,
            'cv_scores': cv_scores.tolist(),
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std()
        }
    
    def save_model(self, output_path: str = "artifacts/models/"):
        """Save trained model"""
        try:
            Path(output_path).mkdir(parents=True, exist_ok=True)
            
            model_file = os.path.join(output_path, f"{self.model_name}.pkl")
            with open(model_file, 'wb') as f:
                pickle.dump(self.model, f)
            
            logger.info(f"Model saved to {model_file}")
            
            # Save model metadata
            metadata = {
                'model_name': self.model_name,
                'model_type': type(self.model).__name__,
                'parameters': self.model.get_params(),
                'feature_importance': self.get_feature_importance(),
                'training_samples': len(self.X_train),
                'test_samples': len(self.X_test)
            }
            
            metadata_file = os.path.join(output_path, f"{self.model_name}_metadata.json")
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            return model_file
            
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise
    
    def get_feature_importance(self):
        """Extract feature importance if available"""
        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_.tolist()
        elif hasattr(self.model, 'coef_'):
            importance = self.model.coef_[0].tolist()
        else:
            importance = None
        
        return importance

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, 
                       choices=['logistic_regression', 'random_forest', 'xgboost'])
    args = parser.parse_args()
    
    try:
        logger.info(f"Starting training for {args.model}")
        
        trainer = ModelTrainer(args.model)
        trainer.load_features()
        trainer.get_model()
        metrics = trainer.train()
        trainer.save_model()
        
        print("\n" + "="*60)
        print(f"MODEL TRAINING COMPLETE: {args.model.upper()}")
        print("="*60)
        print(f"Training Accuracy: {metrics['train_accuracy']:.4f}")
        print(f"Testing Accuracy: {metrics['test_accuracy']:.4f}")
        print(f"Cross-validation Mean: {metrics['cv_mean']:.4f} (+/- {metrics['cv_std']:.4f})")
        
        # Save metrics
        metrics_path = f"artifacts/metrics/{args.model}_metrics.json"
        Path(metrics_path).parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"\n✅ Model training completed successfully!")
        
    except Exception as e:
        logger.error(f"Model training failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
