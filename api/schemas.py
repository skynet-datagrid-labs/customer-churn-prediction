"""Pydantic schemas for API request/response validation."""

from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any
from datetime import datetime

class PredictionRequest(BaseModel):
    """Single prediction request schema."""
    
    customer_id: Optional[int] = Field(None, description="Customer identifier")
    age: int = Field(..., ge=18, le=100, description="Customer age")
    gender: str = Field(..., regex="^(Male|Female)$", description="Customer gender")
    tenure_months: int = Field(..., ge=0, le=120, description="Months as customer")
    monthly_spend: float = Field(..., ge=0, le=1000, description="Average monthly spend")
    contract_type: str = Field(..., regex="^(Monthly|Yearly)$", description="Contract type")
    support_tickets: int = Field(..., ge=0, le=50, description="Number of support tickets")
    last_login_days: int = Field(..., ge=0, le=365, description="Days since last login")
    satisfaction_score: int = Field(..., ge=1, le=10, description="Customer satisfaction score")
    
    @validator('gender')
    def validate_gender(cls, v):
        if v not in ['Male', 'Female']:
            raise ValueError('Gender must be Male or Female')
        return v
    
    @validator('contract_type')
    def validate_contract(cls, v):
        if v not in ['Monthly', 'Yearly']:
            raise ValueError('Contract type must be Monthly or Yearly')
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "customer_id": 1001,
                "age": 35,
                "gender": "Male",
                "tenure_months": 12,
                "monthly_spend": 250.50,
                "contract_type": "Monthly",
                "support_tickets": 2,
                "last_login_days": 15,
                "satisfaction_score": 7
            }
        }

class PredictionResponse(BaseModel):
    """Prediction response schema."""
    
    customer_id: Optional[int] = None
    churn_prediction: int = Field(..., description="Predicted churn (0=No, 1=Yes)")
    churn_probability: float = Field(..., ge=0, le=1, description="Probability of churn")
    prediction_timestamp: str
    model_version: str = "latest"
    
    class Config:
        schema_extra = {
            "example": {
                "customer_id": 1001,
                "churn_prediction": 0,
                "churn_probability": 0.23,
                "prediction_timestamp": "2024-01-15T10:30:00",
                "model_version": "latest"
            }
        }

class BatchPredictionRequest(BaseModel):
    """Batch prediction request schema."""
    
    customers: List[PredictionRequest] = Field(..., min_items=1, max_items=100)
    
    class Config:
        schema_extra = {
            "example": {
                "customers": [
                    {
                        "age": 35,
                        "gender": "Male",
                        "tenure_months": 12,
                        "monthly_spend": 250.50,
                        "contract_type": "Monthly",
                        "support_tickets": 2,
                        "last_login_days": 15,
                        "satisfaction_score": 7
                    }
                ]
            }
        }

class HealthResponse(BaseModel):
    """Health check response schema."""
    
    status: str = Field(..., description="API status (healthy/degraded)")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    timestamp: str = Field(..., description="Current timestamp")

class ModelInfoResponse(BaseModel):
    """Model information response schema."""
    
    model_type: str = Field(..., description="Model class name")
    features: List[str] = Field(..., description="Feature names")
    n_features: int = Field(..., description="Number of features")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
