#!/usr/bin/env python3
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
import sys
from pathlib import Path

def preprocess_data():
    try:
        df = joblib.load('artifacts/raw_data.pkl')
        
        # Drop customer_id
        X = df.drop('churn', axis=1)
        y = df['churn']
        X = X.drop('customer_id', axis=1)
        
        # Stratified split 80/20
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Encode categorical features
        gender_encoder = LabelEncoder()
        X_train['gender'] = gender_encoder.fit_transform(X_train['gender'])
        X_test['gender'] = gender_encoder.transform(X_test['gender'])
        
        contract_encoder = OrdinalEncoder(categories=[['Monthly', 'Annual', 'Two-Year']])
        X_train['contract_type'] = contract_encoder.fit_transform(X_train[['contract_type']])
        X_test['contract_type'] = contract_encoder.transform(X_test[['contract_type']])
        
        # Fill nulls with median/mode from train
        numeric_cols = X_train.select_dtypes(include=[np.number]).columns
        categorical_cols = X_train.select_dtypes(include=['object']).columns
        
        for col in numeric_cols:
            median_val = X_train[col].median()
            X_train[col] = X_train[col].fillna(median_val)
            X_test[col] = X_test[col].fillna(median_val)
        
        for col in categorical_cols:
            mode_val = X_train[col].mode()[0] if len(X_train[col].mode()) > 0 else 'Unknown'
            X_train[col] = X_train[col].fillna(mode_val)
            X_test[col] = X_test[col].fillna(mode_val)
        
        # Save artifacts
        Path('artifacts').mkdir(exist_ok=True)
        joblib.dump(X_train, 'artifacts/X_train.pkl')
        joblib.dump(X_test, 'artifacts/X_test.pkl')
        joblib.dump(y_train, 'artifacts/y_train.pkl')
        joblib.dump(y_test, 'artifacts/y_test.pkl')
        
        encoders = {
            'gender_encoder': gender_encoder,
            'contract_encoder': contract_encoder
        }
        joblib.dump(encoders, 'artifacts/encoders.pkl')
        
        print(f"X_train shape: {X_train.shape}")
        print(f"X_test shape: {X_test.shape}")
        print(f"y_train shape: {y_train.shape}")
        print(f"y_test shape: {y_test.shape}")
        
    except Exception as e:
        print(f"ERROR in preprocessing: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    preprocess_data()
