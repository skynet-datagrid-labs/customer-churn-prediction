#!/usr/bin/env python3
import joblib
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from pathlib import Path

def save_artifacts():
    try:
        with open('artifacts/best_model.json', 'r') as f:
            best_info = json.load(f)
        
        best_model_name = best_info['best_model_name']
        best_model_path = best_info['best_model_path']
        
        # Copy best model and metrics
        import shutil
        shutil.copy(best_model_path, 'artifacts/model.pkl')
        shutil.copy(f'artifacts/metrics_{best_model_name}.json', 'artifacts/metrics.json')
        
        # Load model and test data
        model = joblib.load('artifacts/model.pkl')
        X_test = joblib.load('artifacts/X_test_fe.pkl')
        y_test = joblib.load('artifacts/y_test.pkl')
        
        # Generate predictions and plots
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['No Churn', 'Churn'],
                    yticklabels=['No Churn', 'Churn'])
        plt.title(f'Confusion Matrix - {best_model_name}')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        
        Path('reports').mkdir(exist_ok=True)
        plt.savefig('reports/confusion_matrix.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {best_model_name}')
        plt.legend(loc="lower right")
        plt.savefig('reports/roc_curve.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # Verify all files exist and have size > 0
        files_to_check = [
            'artifacts/model.pkl',
            'artifacts/metrics.json',
            'reports/confusion_matrix.png',
            'reports/roc_curve.png'
        ]
        
        for file_path in files_to_check:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Missing file: {file_path}")
            file_size = os.path.getsize(file_path)
            if file_size == 0:
                raise ValueError(f"Empty file: {file_path}")
            print(f"{file_path}: {file_size} bytes")
        
        print("ARTIFACTS SAVED SUCCESSFULLY")
        
    except Exception as e:
        print(f"ERROR saving artifacts: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    save_artifacts()
