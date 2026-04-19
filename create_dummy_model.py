# create_dummy_model.py
import pickle
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import json
import os

os.makedirs('artifacts/models', exist_ok=True)

print('Creating dummy model for testing...')

# Create dummy model
dummy_model = RandomForestClassifier(n_estimators=10, random_state=42)
X_dummy = np.random.rand(100, 15)
y_dummy = np.random.randint(0, 2, 100)
dummy_model.fit(X_dummy, y_dummy)

# Save model
with open('artifacts/models/final_model.pkl', 'wb') as f:
    pickle.dump(dummy_model, f)

# Save preprocessor
dummy_scaler = {'scaler': None, 'label_encoders': {}}
with open('artifacts/models/final_preprocessor.pkl', 'wb') as f:
    pickle.dump(dummy_scaler, f)

# Save feature columns
feature_cols = {
    'feature_columns': ['age', 'tenure_months', 'monthly_spend', 'support_tickets', 
                       'last_login_days', 'satisfaction_score', 'gender', 'contract_type']
}
with open('artifacts/models/feature_columns.json', 'w') as f:
    json.dump(feature_cols, f)

print('Dummy model created successfully!')
