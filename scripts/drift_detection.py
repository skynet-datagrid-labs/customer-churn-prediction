#!/usr/bin/env python3
import joblib
import numpy as np
import pandas as pd
import json
import sys
from pathlib import Path

def calculate_psi(expected, actual, bins=10):
    """Calculate Population Stability Index"""
    # Create bins based on expected distribution
    expected_percentiles = np.percentile(expected, np.linspace(0, 100, bins+1))
    expected_percentiles[0] = -np.inf
    expected_percentiles[-1] = np.inf
    
    # Count observations in each bin
    expected_counts = np.histogram(expected, bins=expected_percentiles)[0]
    actual_counts = np.histogram(actual, bins=expected_percentiles)[0]
    
    # Calculate proportions
    expected_props = expected_counts / len(expected)
    actual_props = actual_counts / len(actual)
    
    # Clip to avoid log(0)
    expected_props = np.clip(expected_props, 1e-10, 1)
    actual_props = np.clip(actual_props, 1e-10, 1)
    
    # Calculate PSI
    psi = np.sum((actual_props - expected_props) * np.log(actual_props / expected_props))
    return psi

def detect_drift():
    try:
        # Load reference data
        reference_df = joblib.load('artifacts/raw_data.pkl')
        
        # Simulate production data (with drift)
        np.random.seed(42)
        production_df = reference_df.copy()
        numeric_cols = ['age', 'tenure_months', 'monthly_spend', 'support_tickets', 
                       'last_login_days', 'satisfaction_score']
        
        for col in numeric_cols:
            noise = np.random.normal(0, 2, len(production_df))
            production_df[col] = production_df[col] + noise
            if col in ['age']:
                production_df[col] = production_df[col].clip(18, 100)
            elif col in ['satisfaction_score']:
                production_df[col] = production_df[col].clip(1, 5)
            elif col in ['monthly_spend']:
                production_df[col] = production_df[col].clip(0.01, None)
        
        # Calculate PSI for each numeric feature
        drift_results = {}
        for col in numeric_cols:
            psi = calculate_psi(reference_df[col].values, production_df[col].values)
            
            if psi < 0.1:
                status = "stable"
            elif psi < 0.25:
                status = "warning"
            else:
                status = "critical"
            
            drift_results[col] = {
                "psi": psi,
                "status": status
            }
        
        # Print table
        print(f"{'Feature':<20} {'PSI':<10} {'Status':<10}")
        print("-" * 40)
        for feature, results in drift_results.items():
            print(f"{feature:<20} {results['psi']:<10.4f} {results['status']:<10}")
        
        # Save report
        Path('reports').mkdir(exist_ok=True)
        report = {
            "psi_scores": drift_results,
            "overall_status": "critical" if any(r['status'] == 'critical' for r in drift_results.values()) 
                              else "warning" if any(r['status'] == 'warning' for r in drift_results.values())
                              else "stable",
            "reference_shape": reference_df.shape,
            "production_shape": production_df.shape
        }
        
        with open('reports/drift_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        # Exit with appropriate code
        if any(r['status'] == 'critical' for r in drift_results.values()):
            print("\nDRIFT DETECTED — RETRAINING REQUIRED")
            sys.exit(1)
        else:
            print("\nNO DRIFT DETECTED")
            sys.exit(0)
            
    except Exception as e:
        print(f"ERROR in drift detection: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    detect_drift()
