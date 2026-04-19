#!/usr/bin/env python3
import numpy as np
import pandas as pd
import os
import sys
from pathlib import Path

def generate_data():
    try:
        np.random.seed(42)
        n_samples = 5000
        
        # Generate realistic distributions
        age = np.random.normal(45, 15, n_samples).clip(18, 100).astype(int)
        gender = np.random.choice(['Male', 'Female', 'Other'], n_samples, p=[0.48, 0.48, 0.04])
        tenure_months = np.random.exponential(24, n_samples).clip(0, 600).astype(int)
        monthly_spend = np.random.lognormal(3.5, 0.8, n_samples).clip(10, 5000)
        contract_type = np.random.choice(['Monthly', 'Annual', 'Two-Year'], n_samples, p=[0.6, 0.3, 0.1])
        
        # Churn correlated with satisfaction and support tickets
        satisfaction_score = np.random.normal(3.5, 1.0, n_samples).clip(1, 5)
        support_tickets = np.random.poisson(1, n_samples)
        churn_prob = 1 / (1 + np.exp(-(-3 + 0.5 * support_tickets - 0.4 * satisfaction_score + 0.1 * (tenure_months/12))))
        churn = (np.random.random(n_samples) < churn_prob).astype(int)
        
        last_login_days = np.random.exponential(14, n_samples).clip(0, 365).astype(int)
        
        customer_id = [f"CUST_{i:05d}" for i in range(1, n_samples + 1)]
        
        df = pd.DataFrame({
            'customer_id': customer_id,
            'age': age,
            'gender': gender,
            'tenure_months': tenure_months,
            'monthly_spend': monthly_spend.round(2),
            'contract_type': contract_type,
            'support_tickets': support_tickets,
            'last_login_days': last_login_days,
            'satisfaction_score': satisfaction_score.round(2),
            'churn': churn
        })
        
        # Ensure all constraints
        df['monthly_spend'] = df['monthly_spend'].clip(lower=0.01)
        df['support_tickets'] = df['support_tickets'].clip(lower=0)
        df['last_login_days'] = df['last_login_days'].clip(lower=0)
        
        Path('data').mkdir(exist_ok=True)
        output_path = 'data/fictitious_company_data.xlsx'
        df.to_excel(output_path, index=False, engine='openpyxl')
        
        print(f"Saved: {output_path}")
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print(f"Churn rate: {df['churn'].mean():.2%}")
        
    except Exception as e:
        print(f"ERROR generating data: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    generate_data()
