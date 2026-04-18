FROM python:3.9-slim

WORKDIR /app

# Install system dependencies including curl for healthcheck
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all necessary files
COPY api/ ./api/
COPY src/ ./src/
COPY config/ ./config/

# Create necessary directories
RUN mkdir -p artifacts/models artifacts/reports artifacts/metrics artifacts/plots

# Create a dummy model if none exists (for testing)
RUN python -c "
import pickle
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import os

os.makedirs('artifacts/models', exist_ok=True)

# Create a dummy model for testing if real model doesn't exist
try:
    with open('artifacts/models/final_model.pkl', 'rb') as f:
        pass
except:
    print('Creating dummy model for testing...')
    dummy_model = RandomForestClassifier(n_estimators=10, random_state=42)
    X_dummy = np.random.rand(100, 15)
    y_dummy = np.random.randint(0, 2, 100)
    dummy_model.fit(X_dummy, y_dummy)
    
    with open('artifacts/models/final_model.pkl', 'wb') as f:
        pickle.dump(dummy_model, f)
    
    dummy_scaler = {'scaler': None, 'label_encoders': {}}
    with open('artifacts/models/final_preprocessor.pkl', 'wb') as f:
        pickle.dump(dummy_scaler, f)
    
    feature_cols = {
        'feature_columns': ['age', 'tenure_months', 'monthly_spend', 'support_tickets', 
                           'last_login_days', 'satisfaction_score', 'gender', 'contract_type']
    }
    with open('artifacts/models/feature_columns.json', 'w') as f:
        import json
        json.dump(feature_cols, f)
"

# Expose port
EXPOSE 8000

# Run the API server
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "info"]
