#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path

def select_best_model():
    try:
        models = ['logistic_regression', 'random_forest', 'xgboost']
        all_scores = {}
        
        for model in models:
            with open(f'artifacts/metrics_{model}.json', 'r') as f:
                metrics = json.load(f)
                all_scores[model] = {
                    'f1': metrics['f1'],
                    'roc_auc': metrics['roc_auc']
                }
        
        # Sort by F1 descending, then ROC-AUC
        sorted_models = sorted(all_scores.items(), key=lambda x: (x[1]['f1'], x[1]['roc_auc']), reverse=True)
        
        print("Ranked Models:")
        print(f"{'Rank':<5} {'Model':<20} {'F1':<10} {'ROC-AUC':<10}")
        print("-" * 45)
        for rank, (model, scores) in enumerate(sorted_models, 1):
            print(f"{rank:<5} {model:<20} {scores['f1']:<10.4f} {scores['roc_auc']:<10.4f}")
        
        best_model_name = sorted_models[0][0]
        best_model_path = f'artifacts/model_{best_model_name}.pkl'
        
        best_model_info = {
            "best_model_name": best_model_name,
            "best_model_path": best_model_path,
            "selection_criterion": "f1_then_roc_auc",
            "all_scores": all_scores
        }
        
        with open('artifacts/best_model.json', 'w') as f:
            json.dump(best_model_info, f, indent=2)
        
        # GitHub Actions output
        github_output = os.environ.get("GITHUB_OUTPUT")
        if github_output:
            with open(github_output, 'a') as f:
                f.write(f"best_model={best_model_name}\n")
        else:
            print(f"best_model={best_model_name}")
        
    except Exception as e:
        print(f"ERROR selecting best model: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    select_best_model()
