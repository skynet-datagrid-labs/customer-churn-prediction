"""FastAPI application for model serving."""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import pandas as pd
import numpy as np
import joblib
from typing import List, Dict, Any
from datetime import datetime
import logging

from api.schemas import (
    PredictionRequest, PredictionResponse, 
    HealthResponse, BatchPredictionRequest,
    ModelInfoResponse
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

# Global variables for model and preprocessor
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
        
        # Load feature columns
        feature_info_path = Path("artifacts/reports/feature_info_enhanced.json")
        if feature_info_path.exists():
            import json
            with open(feature_info_path, 'r') as f:
                feature_info = json.load(f)
            feature_columns = feature_info.get('feature_names', [])
        
        return True
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return False

# Load model on startup
@app.on_event("startup")
async def startup_event():
    """Load model when API starts."""
    success = load_model()
    if not success:
        logger.warning("Model not loaded - predictions will not work")
    else:
        logger.info("API ready for predictions")

@app.get("/", response_model=Dict[str, str])
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

@app.get("/model/info", response_model=ModelInfoResponse)
async def model_info():
    """Get model information."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Try to load model metadata
    metadata_path = Path("artifacts/reports/best_model_metadata.json")
    metadata = {}
    
    if metadata_path.exists():
        import json
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    
    return ModelInfoResponse(
        model_type=type(model).__name__,
        features=feature_columns or [],
        n_features=len(feature_columns) if feature_columns else 0,
        metadata=metadata
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
        
        # Preprocess if preprocessor is available
        if preprocessor:
            from src.preprocess import DataPreprocessor
            prep = DataPreprocessor()
            prep.load_preprocessor()
            df_processed = prep.preprocess(df, fit=False)
            
            # Get features (exclude target if exists)
            if 'churn' in df_processed.columns:
                df_processed = df_processed.drop('churn', axis=1)
        else:
            # Basic preprocessing
            df_processed = df.copy()
            
            # Encode categoricals
            df_processed['gender'] = df_processed['gender'].map({'Male': 0, 'Female': 1})
            df_processed['contract_type'] = df_processed['contract_type'].map({'Monthly': 0, 'Yearly': 1})
        
        # Ensure correct feature order
        if feature_columns:
            for col in feature_columns:
                if col not in df_processed.columns:
                    df_processed[col] = 0
            df_processed = df_processed[feature_columns]
        
        # Make prediction
        prediction = model.predict(df_processed)[0]
        prediction_proba = model.predict_proba(df_processed)[0]
        
        return PredictionResponse(
            customer_id=request.customer_id,
            churn_prediction=int(prediction),
            churn_probability=float(prediction_proba[1]),
            prediction_timestamp=datetime.now().isoformat(),
            model_version="latest"
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict/batch", response_model=List[PredictionResponse])
async def predict_batch(request: BatchPredictionRequest):
    """Predict churn for multiple customers."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        predictions = []
        
        for customer in request.customers:
            # Create single prediction request
            pred_request = PredictionRequest(**customer.dict())
            pred_response = await predict(pred_request)
            predictions.append(pred_response)
        
        return predictions
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/metrics")
async def get_metrics():
    """Get model performance metrics."""
    metrics_path = Path("artifacts/reports/best_model_info.json")
    
    if not metrics_path.exists():
        raise HTTPException(status_code=404, detail="Metrics not found")
    
    import json
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    
    return JSONResponse(content=metrics)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
