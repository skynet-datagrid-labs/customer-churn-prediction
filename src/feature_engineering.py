#!/usr/bin/env python3
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import sys
from pathlib import Path

def engineer_features():
    try:
        X_train = joblib.load('artifacts/X_train.pkl')
        X_test = joblib.load('artifacts/X_test.pkl')
        y_train = joblib.load('artifacts/y_train.pkl')
        y_test = joblib.load('artifacts/y_test.pkl')
        encoders = joblib.load('artifacts/encoders.pkl')
        
        # Derive new features
        for df in [X_train, X_test]:
            df['spend_per_tenure'] = df['monthly_spend'] / (df['tenure_months'] + 1)
            df['ticket_rate'] = df['support_tickets'] / (df['tenure_months'] + 1)
            df['recency_spend_ratio'] = df['last_login_days'] / (df['monthly_spend'] + 1)
            df['satisfaction_risk'] = (5.0 - df['satisfaction_score']) * df['support_tickets']
        
        # High value customer based on train 75th percentile
        spend_75th = X_train['monthly_spend'].quantile(0.75)
        X_train['high_value_customer'] = (X_train['monthly_spend'] > spend_75th).astype(int)
        X_test['high_value_customer'] = (X_test['monthly_spend'] > spend_75th).astype(int)
        
        # Scale all features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Convert back to DataFrame with column names
        X_train_fe = pd.DataFrame(X_train_scaled, columns=X_train.columns)
        X_test_fe = pd.DataFrame(X_test_scaled, columns=X_test.columns)
        
        # Save artifacts
        joblib.dump(X_train_fe, 'artifacts/X_train_fe.pkl')
        joblib.dump(X_test_fe, 'artifacts/X_test_fe.pkl')
        joblib.dump(scaler, 'artifacts/scaler.pkl')
        joblib.dump(list(X_train_fe.columns), 'artifacts/feature_names.pkl')
        joblib.dump({'spend_75th': spend_75th}, 'artifacts/fe_params.pkl')
        
        print(f"Feature list: {list(X_train_fe.columns)}")
        print(f"Train shape: {X_train_fe.shape}")
        print(f"Test shape: {X_test_fe.shape}")
        
    except Exception as e:
        print(f"ERROR in feature engineering: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    engineer_features()
