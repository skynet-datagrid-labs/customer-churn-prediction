"""Unit tests for API endpoints."""

import pytest
from fastapi.testclient import TestClient
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.app import app

client = TestClient(app)

def test_root_endpoint():
    """Test root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "Customer Churn Prediction API" in data["message"]
    assert "version" in data

def test_health_check():
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data
    assert "timestamp" in data

def test_model_info():
    """Test model info endpoint."""
    response = client.get("/model/info")
    # Model may or may not be loaded in test environment
    assert response.status_code in [200, 503]
    
    if response.status_code == 200:
        data = response.json()
        assert "model_type" in data
        assert "features" in data
        assert "n_features" in data

def test_predict_valid():
    """Test prediction with valid input."""
    valid_request = {
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
    
    response = client.post("/predict", json=valid_request)
    
    # Model may not be loaded in test environment
    if response.status_code == 503:
        assert response.json()["detail"] == "Model not loaded"
    else:
        assert response.status_code == 200
        data = response.json()
        assert "churn_prediction" in data
        assert "churn_probability" in data
        assert "prediction_timestamp" in data
        assert 0 <= data["churn_probability"] <= 1

def test_predict_invalid_age():
    """Test prediction with invalid age."""
    invalid_request = {
        "customer_id": 1001,
        "age": 150,  # Invalid age
        "gender": "Male",
        "tenure_months": 12,
        "monthly_spend": 250.50,
        "contract_type": "Monthly",
        "support_tickets": 2,
        "last_login_days": 15,
        "satisfaction_score": 7
    }
    
    response = client.post("/predict", json=invalid_request)
    assert response.status_code == 422  # Validation error

def test_predict_invalid_gender():
    """Test prediction with invalid gender."""
    invalid_request = {
        "customer_id": 1001,
        "age": 35,
        "gender": "Other",  # Invalid gender
        "tenure_months": 12,
        "monthly_spend": 250.50,
        "contract_type": "Monthly",
        "support_tickets": 2,
        "last_login_days": 15,
        "satisfaction_score": 7
    }
    
    response = client.post("/predict", json=invalid_request)
    assert response.status_code == 422

def test_predict_missing_field():
    """Test prediction with missing required field."""
    invalid_request = {
        "customer_id": 1001,
        "age": 35,
        "gender": "Male",
        "tenure_months": 12,
        # Missing monthly_spend
        "contract_type": "Monthly",
        "support_tickets": 2,
        "last_login_days": 15,
        "satisfaction_score": 7
    }
    
    response = client.post("/predict", json=invalid_request)
    assert response.status_code == 422

def test_batch_predict():
    """Test batch prediction endpoint."""
    batch_request = {
        "customers": [
            {
                "customer_id": 1,
                "age": 35,
                "gender": "Male",
                "tenure_months": 12,
                "monthly_spend": 250.50,
                "contract_type": "Monthly",
                "support_tickets": 2,
                "last_login_days": 15,
                "satisfaction_score": 7
            },
            {
                "customer_id": 2,
                "age": 45,
                "gender": "Female",
                "tenure_months": 24,
                "monthly_spend": 300.75,
                "contract_type": "Yearly",
                "support_tickets": 1,
                "last_login_days": 30,
                "satisfaction_score": 9
            }
        ]
    }
    
    response = client.post("/predict/batch", json=batch_request)
    
    if response.status_code == 503:
        assert response.json()["detail"] == "Model not loaded"
    else:
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 2
        for pred in data:
            assert "churn_prediction" in pred
            assert "churn_probability" in pred

def test_metrics_endpoint():
    """Test metrics endpoint."""
    response = client.get("/metrics")
    # Metrics may not exist in test environment
    assert response.status_code in [200, 404]
    
    if response.status_code == 200:
        data = response.json()
        assert isinstance(data, dict)

def test_docs_endpoints():
    """Test documentation endpoints."""
    response = client.get("/docs")
    assert response.status_code in [200, 307]
    
    response = client.get("/openapi.json")
    assert response.status_code == 200
    data = response.json()
    assert "openapi" in data
    assert "info" in data

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
