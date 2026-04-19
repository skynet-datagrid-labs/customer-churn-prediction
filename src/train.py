#!/usr/bin/env python3
import joblib
import sys
import time
import os
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import numpy as np

def train_model():
    try:
        if len(sys.argv) < 2 or sys.argv[1] not in ['logistic_regression', 'random_forest', 'xgboost']:
            print("Usage: python src/train.py [logistic_regression|random_forest|xgboost]")
            sys.exit(1)
        
        model_name = sys.argv[1]
        X_train = joblib.load('artifacts/X_train_fe.pkl')
        y_train = joblib.load('artifacts/y_train.pkl')
        
        # Compute scale_pos_weight for XGBoost
        neg_count = (y_train == 0).sum()
        pos_count = (y_train == 1).sum()
        scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
        
        models = {
            "logistic_regression": LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced"),
            "random_forest": RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, class_weight="balanced", n_jobs=-1),
            "xgboost": XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.05, random_state=42, eval_metric="logloss", scale_pos_weight=scale_pos_weight)
        }
        
        model = models[model_name]
        start_time = time.time()
        model.fit(X_train, y_train)
        end_time = time.time()
        
        # Verify model has required methods
        if not hasattr(model, 'predict') or not hasattr(model, 'predict_proba'):
            raise RuntimeError(f"Model {model_name} missing predict/predict_proba methods")
        
        joblib.dump(model, f'artifacts/model_{model_name}.pkl')
        
        file_size = os.path.getsize(f'artifacts/model_{model_name}.pkl')
        print(f"Model: {model_name}")
        print(f"Training duration: {end_time - start_time:.2f} seconds")
        print(f"File size: {file_size} bytes")
        
    except Exception as e:
        print(f"ERROR training {model_name if 'model_name' in locals() else 'unknown'}: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    train_model()
