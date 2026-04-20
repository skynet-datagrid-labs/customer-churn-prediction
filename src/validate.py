#!/usr/bin/env python3
import joblib
import pandas as pd
import numpy as np
import json
import sys
from datetime import datetime
from pathlib import Path

def validate_data():
    try:
        df = joblib.load('artifacts/raw_data.pkl')
        
        # Expected schema
        expected_columns = ['customer_id', 'age', 'gender', 'tenure_months', 'monthly_spend', 
                           'contract_type', 'support_tickets', 'last_login_days', 'satisfaction_score', 'churn']
        
        # 1. All columns present
        missing_cols = [col for col in expected_columns if col not in df.columns]
        if missing_cols:
            print(f"FAIL: Missing columns: {missing_cols}")
            sys.exit(1)
        print("PASS: All columns present")
        
        # 2. Column dtypes match expected
        dtype_map = {'customer_id': 'object', 'age': 'int64', 'gender': 'object', 'tenure_months': 'int64',
                    'monthly_spend': 'float64', 'contract_type': 'object', 'support_tickets': 'int64',
                    'last_login_days': 'int64', 'satisfaction_score': 'float64', 'churn': 'int64'}
        for col, expected_dtype in dtype_map.items():
            actual_dtype = str(df[col].dtype)
            if expected_dtype not in actual_dtype:
                print(f"FAIL: Column {col} - expected {expected_dtype}, found {actual_dtype}")
                sys.exit(1)
        print("PASS: Column dtypes match")
        
        # 3. No nulls
        null_counts = df.isnull().sum()
        if null_counts.any():
            print(f"FAIL: Null values found:\n{null_counts[null_counts > 0]}")
            sys.exit(1)
        print("PASS: No nulls")
        
        # 4. customer_id unique
        duplicate_ids = df['customer_id'].duplicated().sum()
        if duplicate_ids > 0:
            print(f"FAIL: {duplicate_ids} duplicate customer_ids found")
            sys.exit(1)
        print("PASS: customer_id unique")
        
        # 5-13. Range checks
        if not df['age'].between(18, 100).all():
            invalid = df[~df['age'].between(18, 100)][['age']]
            print(f"FAIL: age out of range 18-100:\n{invalid}")
            sys.exit(1)
        print("PASS: age in range 18-100")
        
        valid_genders = ['Male', 'Female', 'Other']
        invalid_genders = df[~df['gender'].isin(valid_genders)]['gender'].unique()
        if len(invalid_genders) > 0:
            print(f"FAIL: Invalid gender values detected (count={len(invalid_genders)})")
            sys.exit(1)
        print("PASS: gender valid")
        
        if (df['tenure_months'] < 0).any():
            print(f"FAIL: Negative tenure_months:\n{df[df['tenure_months'] < 0][['tenure_months']]}")
            sys.exit(1)
        print("PASS: tenure_months >= 0")
        
        if (df['monthly_spend'] <= 0).any():
            print(f"FAIL: Non-positive monthly_spend:\n{df[df['monthly_spend'] <= 0][['monthly_spend']]}")
            sys.exit(1)
        print("PASS: monthly_spend > 0")
        
        valid_contracts = ['Monthly', 'Annual', 'Two-Year']
        invalid_contracts = df[~df['contract_type'].isin(valid_contracts)]['contract_type'].unique()
        if len(invalid_contracts) > 0:
            print(f"FAIL: Invalid contract types: {invalid_contracts}")
            sys.exit(1)
        print("PASS: contract_type valid")
        
        if (df['support_tickets'] < 0).any():
            print(f"FAIL: Negative support_tickets:\n{df[df['support_tickets'] < 0][['support_tickets']]}")
            sys.exit(1)
        print("PASS: support_tickets >= 0")
        
        if (df['last_login_days'] < 0).any():
            print(f"FAIL: Negative last_login_days:\n{df[df['last_login_days'] < 0][['last_login_days']]}")
            sys.exit(1)
        print("PASS: last_login_days >= 0")
        
        if not df['satisfaction_score'].between(1.0, 5.0).all():
            invalid = df[~df['satisfaction_score'].between(1.0, 5.0)][['satisfaction_score']]
            print(f"FAIL: satisfaction_score out of range 1-5:\n{invalid}")
            sys.exit(1)
        print("PASS: satisfaction_score in 1.0-5.0")
        
        if not df['churn'].isin([0, 1]).all():
            invalid = df[~df['churn'].isin([0, 1])]['churn'].unique()
            print(f"FAIL: Invalid churn values: {invalid}")
            sys.exit(1)
        print("PASS: churn binary 0/1")
        
        # Save validation report
        report = {
            "timestamp": datetime.now().isoformat(),
            "row_count": len(df),
            "column_count": len(df.columns),
            "null_counts": df.isnull().sum().to_dict(),
            "churn_distribution": df['churn'].value_counts().to_dict(),
            "numeric_summary": df.select_dtypes(include=[np.number]).describe().to_dict()
        }
        
        Path('artifacts').mkdir(exist_ok=True)
        with open('artifacts/validation_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        print("\nVALIDATION PASSED")
        
    except Exception as e:
        print(f"ERROR in validation: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    validate_data()
