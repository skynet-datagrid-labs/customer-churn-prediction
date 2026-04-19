import pytest
from httpx import AsyncClient
from api.app import app

@pytest.fixture
def valid_payload():
    return {
        "age": 35,
        "gender": "Male",
        "tenure_months": 24,
        "monthly_spend": 150.50,
        "contract_type": "Annual",
        "support_tickets": 2,
        "last_login_days": 5,
        "satisfaction_score": 4.2
    }

@pytest.mark.asyncio
async def test_health_check():
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"
        assert response.json()["model_loaded"] == True

@pytest.mark.asyncio
async def test_valid_prediction(valid_payload):
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.post("/predict", json=valid_payload)
        assert response.status_code == 200
        data = response.json()
        assert data["churn_prediction"] in [0, 1]
        assert 0.0 <= data["churn_probability"] <= 1.0
        assert isinstance(data["model_used"], str)

@pytest.mark.asyncio
async def test_invalid_age_negative(valid_payload):
    valid_payload["age"] = -5
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.post("/predict", json=valid_payload)
        assert response.status_code == 422

@pytest.mark.asyncio
async def test_invalid_age_over_100(valid_payload):
    valid_payload["age"] = 150
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.post("/predict", json=valid_payload)
        assert response.status_code == 422

@pytest.mark.asyncio
async def test_invalid_gender(valid_payload):
    valid_payload["gender"] = "Invalid"
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.post("/predict", json=valid_payload)
        assert response.status_code == 422

@pytest.mark.asyncio
async def test_invalid_contract_type(valid_payload):
    valid_payload["contract_type"] = "Invalid"
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.post("/predict", json=valid_payload)
        assert response.status_code == 422

@pytest.mark.asyncio
async def test_satisfaction_out_of_range(valid_payload):
    valid_payload["satisfaction_score"] = 6.0
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.post("/predict", json=valid_payload)
        assert response.status_code == 422

@pytest.mark.asyncio
async def test_missing_required_field(valid_payload):
    del valid_payload["age"]
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.post("/predict", json=valid_payload)
        assert response.status_code == 422

@pytest.mark.asyncio
async def test_negative_monthly_spend(valid_payload):
    valid_payload["monthly_spend"] = -10.0
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.post("/predict", json=valid_payload)
        assert response.status_code == 422
