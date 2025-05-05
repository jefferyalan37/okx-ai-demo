import numpy as np
import os
import joblib
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

os.makedirs("models", exist_ok=True)

# Dummy regression data
X = np.array([[i] for i in range(10)])
y = np.array([i * 2.5 for i in range(10)])  # Regression-style y

# Save Scaler
scaler = StandardScaler()
scaler.fit(X)
joblib.dump(scaler, "models/scaler.pkl")

# Save XGBoost Regressor
xgb_model = xgb.XGBRegressor()
xgb_model.fit(X, y)
xgb_model.save_model("models/xgb.json")

# Save Stacking Regressor
stack_model = RandomForestRegressor()
stack_model.fit(X, y)
joblib.dump(stack_model, "models/stack.pkl")

# Save Dummy TFT PyTorch model
class DummyTFT(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

model = DummyTFT()
torch.save(model.state_dict(), "models/tft_state_dict.pt")

print("✅ All models saved cleanly.")
