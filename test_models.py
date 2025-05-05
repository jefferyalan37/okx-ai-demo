import numpy as np, torch
from joblib import load
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor

# Define dummy TFT
class DummyTFT(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 1)

# Load all models
scaler = load("models/scaler.pkl")
stack = load("models/stack.pkl")
xgb    = XGBRegressor()
xgb.load_model("models/xgb.json")

tft = DummyTFT()
tft.load_state_dict(torch.load("models/tft_state_dict.pt"))
tft.eval()

# Dummy input
X_raw    = np.array([[6.0]])
X_scaled = scaler.transform(X_raw)
tensor   = torch.tensor(X_scaled, dtype=torch.float32).unsqueeze(1)

with torch.no_grad():
    tft_pred = tft(tensor).squeeze().numpy()

xgb_pred  = xgb.predict(X_scaled)
meta_input = np.column_stack([X_scaled, tft_pred])
stack_pred = stack.predict(meta_input)

print("✅ All model predictions succeeded.")
print("TFT: ", tft_pred)
print("XGB: ", xgb_pred)
print("Stacked: ", stack_pred)
