import pytest
import httpx
import asyncio
from typing import Dict, Any
import os

BASE_URL = os.getenv("API_URL", "http://localhost:8000")

@pytest.mark.asyncio
async def test_health_endpoint():
    """Test health check endpoint"""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{BASE_URL}/health")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"
            assert "model_loaded" in data
    except httpx.ConnectError:
        pytest.skip(f"API server not running at {BASE_URL}")

@pytest.mark.asyncio
async def test_prediction_endpoint():
    """Test single prediction endpoint"""
    test_data = {
        "customer_id": 9999,
        "age": 35,
        "gender": "Male",
        "tenure_months": 24,
        "monthly_spend": 250.50,
        "contract_type": "Monthly",
        "support_tickets": 2,
        "last_login_days": 15,
        "satisfaction_score": 7
    }
    
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.post(f"{BASE_URL}/predict", json=test_data)
            assert response.status_code == 200
            data = response.json()
            assert data["customer_id"] == test_data["customer_id"]
            assert "churn_prediction" in data
            assert "churn_probability" in data
            assert 0 <= data["churn_probability"] <= 1
    except httpx.ConnectError:
        pytest.skip(f"API server not running at {BASE_URL}")

@pytest.mark.asyncio
async def test_batch_prediction():
    """Test batch prediction endpoint"""
    test_batch = {
        "customers": [
            {
                "customer_id": 1001,
                "age": 45,
                "gender": "Female",
                "tenure_months": 36,
                "monthly_spend": 300.00,
                "contract_type": "Yearly",
                "support_tickets": 1,
                "last_login_days": 5,
                "satisfaction_score": 9
            },
            {
                "customer_id": 1002,
                "age": 28,
                "gender": "Male",
                "tenure_months": 12,
                "monthly_spend": 150.75,
                "contract_type": "Monthly",
                "support_tickets": 5,
                "last_login_days": 45,
                "satisfaction_score": 3
            }
        ]
    }
    
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.post(f"{BASE_URL}/predict/batch", json=test_batch)
            assert response.status_code == 200
            data = response.json()
            assert len(data) == 2
            assert data[0]["customer_id"] == 1001
            assert data[1]["customer_id"] == 1002
    except httpx.ConnectError:
        pytest.skip(f"API server not running at {BASE_URL}")

@pytest.mark.asyncio
async def test_model_info():
    """Test model info endpoint"""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{BASE_URL}/model/info")
            assert response.status_code == 200
            data = response.json()
            assert "model_type" in data
            assert "feature_count" in data
    except httpx.ConnectError:
        pytest.skip(f"API server not running at {BASE_URL}")

@pytest.mark.asyncio
async def test_invalid_prediction():
    """Test prediction with invalid data"""
    invalid_data = {
        "customer_id": -1,  # Invalid negative ID
        "age": 150,  # Invalid age
        "gender": "Invalid",  # Invalid gender
        "tenure_months": -5,  # Invalid tenure
        "monthly_spend": -100,  # Invalid spend
        "contract_type": "Invalid",  # Invalid contract
        "support_tickets": -1,  # Invalid tickets
        "last_login_days": -10,  # Invalid login days
        "satisfaction_score": 20  # Invalid score
    }
    
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.post(f"{BASE_URL}/predict", json=invalid_data)
            # Should return validation error
            assert response.status_code == 422
    except httpx.ConnectError:
        pytest.skip(f"API server not running at {BASE_URL}")

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
