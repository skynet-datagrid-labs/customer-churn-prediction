from pydantic import BaseModel, Field, validator
from typing import Optional, List
from datetime import datetime

class CustomerBase(BaseModel):
    customer_id: int = Field(..., gt=0, description="Unique customer identifier")
    age: int = Field(..., ge=18, le=100, description="Customer age in years")
    gender: str = Field(..., regex="^(Male|Female)$", description="Customer gender")
    tenure_months: int = Field(..., ge=0, le=120, description="Months as customer")
    monthly_spend: float = Field(..., ge=0, le=10000, description="Average monthly spend")
    contract_type: str = Field(..., regex="^(Monthly|Yearly)$", description="Contract type")
    support_tickets: int = Field(..., ge=0, le=50, description="Number of support tickets")
    last_login_days: int = Field(..., ge=0, le=365, description="Days since last login")
    satisfaction_score: int = Field(..., ge=1, le=10, description="Customer satisfaction score")

class PredictionRequest(CustomerBase):
    pass

class PredictionResponse(BaseModel):
    customer_id: int
    churn_prediction: int = Field(..., description="0=No churn, 1=Churn predicted")
    churn_probability: float = Field(..., ge=0, le=1, description="Probability of churn")
    prediction_timestamp: str
    model_version: str

class BatchPredictionRequest(BaseModel):
    customers: List[PredictionRequest]

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    timestamp: str

class MetricsResponse(BaseModel):
    total_predictions: int
    average_confidence: float
    model_version: str
    last_updated: str
