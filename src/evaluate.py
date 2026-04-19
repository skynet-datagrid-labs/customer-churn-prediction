#!/usr/bin/env python3
import joblib
import json
import sys
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from datetime import datetime

def evaluate_model():
    try:
        if len(sys.argv) < 2:
            print("Usage: python src/evaluate.py [model_name]")
            sys.exit(1)
        
        model_name = sys.argv[1]
        model = joblib.load(f'artifacts/model_{model_name}.pkl')
        X_test = joblib.load('artifacts/X_test_fe.pkl')
        y_test = joblib.load('artifacts/y_test.pkl')
        
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        metrics = {
            "model_name": model_name,
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1": float(f1_score(y_test, y_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, y_pred_proba)),
            "evaluated_at": datetime.now().isoformat()
        }
        
        with open(f'artifacts/metrics_{model_name}.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Print formatted table
        print(f"{'Metric':<15} {'Value':<10}")
        print("-" * 25)
        for key, value in metrics.items():
            if key not in ['model_name', 'evaluated_at']:
                print(f"{key:<15} {value:<10.4f}")
        
    except Exception as e:
        print(f"ERROR evaluating {model_name if 'model_name' in locals() else 'unknown'}: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    evaluate_model()
