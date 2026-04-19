import sys
from pathlib import Path

# Ensure the project root is on sys.path so that `from api.app import app`
# and similar imports work when pytest is invoked from the repo root.
sys.path.insert(0, str(Path(__file__).parent))

import pytest
import joblib


@pytest.fixture(autouse=True, scope="session")
def load_test_artifacts():
    """Pre-load ML artifacts into the API app globals.

    httpx.ASGITransport does not trigger FastAPI's lifespan, so we populate
    the module-level globals directly before any test runs.
    """
    import api.app as app_module

    app_module.model = joblib.load("artifacts/model.pkl")
    app_module.scaler = joblib.load("artifacts/scaler.pkl")
    app_module.encoders = joblib.load("artifacts/encoders.pkl")
    app_module.fe_params = joblib.load("artifacts/fe_params.pkl")
    app_module.feature_names = joblib.load("artifacts/feature_names.pkl")
