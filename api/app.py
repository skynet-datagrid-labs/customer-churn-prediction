"""FastAPI application for model serving."""

import sys
import os
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import logging

from api.schemas import (
    PredictionRequest, PredictionResponse, 
    HealthResponse, BatchPredictionRequest
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Customer Churn Prediction API",
    description="ML API for predicting customer churn probability",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
model = None
preprocessor = None
feature_columns = None


def load_model():
    """Load the trained model and preprocessor."""
    global model, preprocessor, feature_columns
    
    model_path = Path("artifacts/models/best_model.pkl")
    preprocessor_path = Path("artifacts/models/preprocessor.pkl")
    
    if not model_path.exists():
        logger.warning(f"Model not found at {model_path}")
        return False
    
    try:
        model = joblib.load(model_path)
        logger.info(f"Model loaded from {model_path}")
        
        if preprocessor_path.exists():
            preprocessor_data = joblib.load(preprocessor_path)
            preprocessor = preprocessor_data
            logger.info("Preprocessor loaded")
        
        # Load feature columns from preprocessor or create default
        global feature_columns
        feature_columns = ['age', 'gender', 'tenure_months', 'monthly_spend', 
                          'contract_type', 'support_tickets', 'last_login_days', 
                          'satisfaction_score']
        
        return True
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return False


@app.on_event("startup")
async def startup_event():
    """Load model when API starts."""
    load_model()


@app.get("/", response_model=dict)
async def root():
    """Root endpoint."""
    return {
        "message": "Customer Churn Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if model is not None else "degraded",
        model_loaded=model is not None,
        timestamp=datetime.now().isoformat()
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Predict churn for a single customer."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert request to feature vector
        input_data = request.dict()
        
        # Create DataFrame
        df = pd.DataFrame([input_data])
        
        # Simple preprocessing
        df['gender'] = df['gender'].map({'Male': 0, 'Female': 1})
        df['contract_type'] = df['contract_type'].map({'Monthly': 0, 'Yearly': 1})
        
        # Ensure correct feature order
        if feature_columns:
            df = df[feature_columns]
        
        # Make prediction
        prediction = model.predict(df)[0]
        prediction_proba = model.predict_proba(df)[0]
        
        return PredictionResponse(
            churn_prediction=int(prediction),
            churn_probability=float(prediction_proba[1]),
            prediction_timestamp=datetime.now().isoformat(),
            model_version="latest"
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/predict/batch", response_model=list)
async def predict_batch(request: BatchPredictionRequest):
    """Predict churn for multiple customers."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        predictions = []
        for customer in request.customers:
            pred_response = await predict(customer)
            predictions.append(pred_response)
        return predictions
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
