import numpy as np
import os
import joblib
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

# Ensure models directory exists
os.makedirs("models", exist_ok=True)

# Dummy regression data (replace with real processed features later)
X = np.array([[i] for i in range(10)])
y = np.array([i * 2.5 for i in range(10)])  # Regression target

# ── 1. Save StandardScaler ──
scaler = StandardScaler()
scaler.fit(X)
joblib.dump(scaler, "models/scaler.pkl")

# ── 2. Save XGBoost Regressor ──
xgb_model = xgb.XGBRegressor()
xgb_model.fit(X, y)
xgb_model.save_model("models/xgb.json")

# ── 3. Save Stacking Regressor ──
stack_model = RandomForestRegressor()
stack_model.fit(X, y)
joblib.dump(stack_model, "models/stack.pkl")

# ── 4. Save dummy TFT PyTorch model state dict ──
class DummyTFT(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

model = DummyTFT()
torch.save(model.state_dict(), "models/tft_state_dict.pt")

print("✅ Clean regression models saved to /models/")
