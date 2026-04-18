from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import pickle
import os
from typing import List, Dict, Any
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="ML Company Churn Prediction API", version="1.0.0")

# Load models and artifacts
MODEL_PATH = "artifacts/models/final_model.pkl"
PREPROCESSOR_PATH = "artifacts/models/final_preprocessor.pkl"
FEATURES_PATH = "artifacts/models/feature_columns.json"

class PredictionRequest(BaseModel):
    customer_id: int
    age: int
    gender: str
    tenure_months: int
    monthly_spend: float
    contract_type: str
    support_tickets: int
    last_login_days: int
    satisfaction_score: int

class PredictionResponse(BaseModel):
    customer_id: int
    churn_prediction: int
    churn_probability: float
    prediction_timestamp: str
    model_version: str

class BatchPredictionRequest(BaseModel):
    customers: List[PredictionRequest]

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    timestamp: str

# Global variables
model = None
preprocessor = None
feature_columns = None

@app.on_event("startup")
async def load_artifacts():
    global model, preprocessor, feature_columns
    try:
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        
        with open(PREPROCESSOR_PATH, 'rb') as f:
            preprocessor = pickle.load(f)
        
        with open(FEATURES_PATH, 'r') as f:
            import json
            feature_columns = json.load(f)['feature_columns']
        
        logger.info("Models and artifacts loaded successfully")
    except Exception as e:
        logger.error(f"Error loading artifacts: {str(e)}")

def preprocess_input(request: PredictionRequest) -> np.ndarray:
    """Preprocess input data for prediction"""
    # Convert to DataFrame
    data = {
        'age': request.age,
        'tenure_months': request.tenure_months,
        'monthly_spend': request.monthly_spend,
        'support_tickets': request.support_tickets,
        'last_login_days': request.last_login_days,
        'satisfaction_score': request.satisfaction_score,
        'gender': 1 if request.gender == 'Male' else 0,
        'contract_type': 1 if request.contract_type == 'Yearly' else 0
    }
    
    # Add engineered features
    data['engagement_score'] = (data['tenure_months'] / 72) * 0.6 + (1 - data['last_login_days'] / 365) * 0.4
    data['tickets_per_month'] = data['support_tickets'] / (data['tenure_months'] + 1)
    data['spend_per_tenure'] = data['monthly_spend'] / (data['tenure_months'] + 1)
    data['risk_score'] = ((data['support_tickets'] > 5) + (data['satisfaction_score'] < 4) + (data['last_login_days'] > 30)) / 3
    data['tenure_satisfaction'] = data['tenure_months'] * data['satisfaction_score']
    
    # Create DataFrame with all features
    df = pd.DataFrame([data])
    
    # Ensure all feature columns are present
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0
    
    # Select and order features
    X = df[feature_columns].values
    
    # Apply scaler if available
    if preprocessor and 'scaler' in preprocessor and preprocessor['scaler']:
        X = preprocessor['scaler'].transform(X)
    
    return X

@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None,
        timestamp=datetime.now().isoformat()
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    try:
        # Preprocess input
        features = preprocess_input(request)
        
        # Make prediction
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0][1]
        
        return PredictionResponse(
            customer_id=request.customer_id,
            churn_prediction=int(prediction),
            churn_probability=float(probability),
            prediction_timestamp=datetime.now().isoformat(),
            model_version="1.0.0"
        )
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch", response_model=List[PredictionResponse])
async def predict_batch(request: BatchPredictionRequest):
    try:
        responses = []
        for customer in request.customers:
            features = preprocess_input(customer)
            prediction = model.predict(features)[0]
            probability = model.predict_proba(features)[0][1]
            
            responses.append(PredictionResponse(
                customer_id=customer.customer_id,
                churn_prediction=int(prediction),
                churn_probability=float(probability),
                prediction_timestamp=datetime.now().isoformat(),
                model_version="1.0.0"
            ))
        
        return responses
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model/info")
async def model_info():
    return {
        "model_type": type(model).__name__ if model else None,
        "feature_count": len(feature_columns) if feature_columns else 0,
        "features": feature_columns,
        "preprocessor_components": list(preprocessor.keys()) if preprocessor else []
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
