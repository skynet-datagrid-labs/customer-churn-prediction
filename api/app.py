from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd
import numpy as np
import time
import logging
from contextlib import asynccontextmanager
from api.schemas import PredictRequest, PredictResponse, HealthResponse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for loaded artifacts
model = None
scaler = None
encoders = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, scaler, encoders
    try:
        model = joblib.load('artifacts/model.pkl')
        scaler = joblib.load('artifacts/scaler.pkl')
        encoders = joblib.load('artifacts/encoders.pkl')
        logger.info("Artifacts loaded successfully")
        yield
    except Exception as e:
        logger.error(f"Failed to load artifacts: {str(e)}")
        raise RuntimeError(f"Artifact loading failed: {str(e)}")

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Middleware for request logging
@app.middleware("http")
async def log_requests(request, call_next):
    start_time = time.time()
    response = await call_next(request)
    duration = (time.time() - start_time) * 1000
    logger.info(f"{request.method} {request.url.path} - {response.status_code} - {duration:.2f}ms")
    return response

@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="ok",
        model_loaded=model is not None
    )

@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    try:
        # Convert request to DataFrame
        input_data = pd.DataFrame([request.dict()])
        
        # Apply feature engineering
        input_data['spend_per_tenure'] = input_data['monthly_spend'] / (input_data['tenure_months'] + 1)
        input_data['ticket_rate'] = input_data['support_tickets'] / (input_data['tenure_months'] + 1)
        input_data['recency_spend_ratio'] = input_data['last_login_days'] / (input_data['monthly_spend'] + 1)
        input_data['satisfaction_risk'] = (5.0 - input_data['satisfaction_score']) * input_data['support_tickets']
        
        # Apply encoders
        input_data['gender'] = encoders['gender_encoder'].transform(input_data['gender'])
        input_data['contract_type'] = encoders['contract_encoder'].transform(input_data[['contract_type']])
        
        # Apply scaler
        input_scaled = scaler.transform(input_data)
        
        # Predict
        prediction = int(model.predict(input_scaled)[0])
        probability = float(model.predict_proba(input_scaled)[0][1])
        
        # Load best model name from metrics
        import json
        with open('artifacts/metrics.json', 'r') as f:
            metrics = json.load(f)
        
        return PredictResponse(
            churn_prediction=prediction,
            churn_probability=probability,
            model_used=metrics['model_name']
        )
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
