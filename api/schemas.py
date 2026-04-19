from pydantic import BaseModel, Field
from typing import Literal

class PredictRequest(BaseModel):
    age: int = Field(..., ge=18, le=100)
    gender: Literal['Male', 'Female', 'Other'] = Field(...)
    tenure_months: int = Field(..., ge=0)
    monthly_spend: float = Field(..., gt=0)
    contract_type: Literal['Monthly', 'Annual', 'Two-Year'] = Field(...)
    support_tickets: int = Field(..., ge=0)
    last_login_days: int = Field(..., ge=0)
    satisfaction_score: float = Field(..., ge=1.0, le=5.0)

class PredictResponse(BaseModel):
    churn_prediction: int
    churn_probability: float
    model_used: str

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
