import pytest
from httpx import AsyncClient, ASGITransport
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
        "satisfaction_score": 4.2,
    }


# ─── Health endpoint ──────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_health_check():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"
        assert response.json()["model_loaded"] is True


@pytest.mark.asyncio
async def test_health_returns_json():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/health")
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data


# ─── Valid prediction ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_valid_prediction(valid_payload):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/predict", json=valid_payload)
        assert response.status_code == 200
        data = response.json()
        assert data["churn_prediction"] in [0, 1]
        assert 0.0 <= data["churn_probability"] <= 1.0
        assert isinstance(data["model_used"], str)


@pytest.mark.asyncio
async def test_predict_returns_probability_between_0_and_1(valid_payload):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/predict", json=valid_payload)
        assert response.status_code == 200
        prob = response.json()["churn_probability"]
        assert 0.0 <= prob <= 1.0


@pytest.mark.asyncio
async def test_predict_churn_prediction_is_binary(valid_payload):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/predict", json=valid_payload)
        assert response.status_code == 200
        assert response.json()["churn_prediction"] in [0, 1]


@pytest.mark.asyncio
async def test_predict_model_used_is_string(valid_payload):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/predict", json=valid_payload)
        assert response.status_code == 200
        assert len(response.json()["model_used"]) > 0


# ─── Input validation: age ────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_invalid_age_negative(valid_payload):
    valid_payload["age"] = -5
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/predict", json=valid_payload)
        assert response.status_code == 422


@pytest.mark.asyncio
async def test_invalid_age_over_100(valid_payload):
    valid_payload["age"] = 150
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/predict", json=valid_payload)
        assert response.status_code == 422


# ─── Input validation: gender ─────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_invalid_gender(valid_payload):
    valid_payload["gender"] = "Invalid"
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/predict", json=valid_payload)
        assert response.status_code == 422


# ─── Input validation: contract_type ─────────────────────────────────────────

@pytest.mark.asyncio
async def test_invalid_contract_type(valid_payload):
    valid_payload["contract_type"] = "Invalid"
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/predict", json=valid_payload)
        assert response.status_code == 422


# ─── Input validation: satisfaction_score ────────────────────────────────────

@pytest.mark.asyncio
async def test_satisfaction_out_of_range_high(valid_payload):
    valid_payload["satisfaction_score"] = 6.0
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/predict", json=valid_payload)
        assert response.status_code == 422


@pytest.mark.asyncio
async def test_satisfaction_out_of_range_low(valid_payload):
    valid_payload["satisfaction_score"] = 0.5
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/predict", json=valid_payload)
        assert response.status_code == 422


# ─── Input validation: missing / numeric fields ───────────────────────────────

@pytest.mark.asyncio
async def test_missing_required_field(valid_payload):
    del valid_payload["age"]
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/predict", json=valid_payload)
        assert response.status_code == 422


@pytest.mark.asyncio
async def test_negative_monthly_spend(valid_payload):
    valid_payload["monthly_spend"] = -10.0
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/predict", json=valid_payload)
        assert response.status_code == 422


@pytest.mark.asyncio
async def test_negative_support_tickets(valid_payload):
    valid_payload["support_tickets"] = -1
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/predict", json=valid_payload)
        assert response.status_code == 422


@pytest.mark.asyncio
async def test_negative_last_login_days(valid_payload):
    valid_payload["last_login_days"] = -3
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/predict", json=valid_payload)
        assert response.status_code == 422


@pytest.mark.asyncio
async def test_negative_tenure_months(valid_payload):
    valid_payload["tenure_months"] = -1
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/predict", json=valid_payload)
        assert response.status_code == 422


# ─── Boundary / edge values ───────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_predict_min_boundary_values():
    payload = {
        "age": 18,
        "gender": "Other",
        "tenure_months": 0,
        "monthly_spend": 0.01,
        "contract_type": "Monthly",
        "support_tickets": 0,
        "last_login_days": 0,
        "satisfaction_score": 1.0,
    }
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/predict", json=payload)
        assert response.status_code == 200


@pytest.mark.asyncio
async def test_predict_max_boundary_values():
    payload = {
        "age": 100,
        "gender": "Female",
        "tenure_months": 600,
        "monthly_spend": 5000.0,
        "contract_type": "Two-Year",
        "support_tickets": 50,
        "last_login_days": 365,
        "satisfaction_score": 5.0,
    }
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/predict", json=payload)
        assert response.status_code == 200

