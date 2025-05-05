import os
import joblib
import pickle
import xgboost as xgb
import torch
import torch.nn as nn

# Ensure models directory exists
os.makedirs("models", exist_ok=True)

# ----- 1. Save dummy stack.pkl -----
stack_model = {"models": ["xgb", "tft", "baseline"], "meta": "placeholder stack"}
with open("models/stack.pkl", "wb") as f:
    pickle.dump(stack_model, f)

# ----- 2. Save dummy xgb.json -----
xgb = xgb.XGBRegressor() 
xgb_model.fit([[0, 0], [1, 1]], [0, 1])
xgb_model.save_model("models/xgb.json")

# ----- 3. Save dummy tft_state_dict.pt -----
class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 1)

model = DummyModel()
torch.save(model.state_dict(), "models/tft_state_dict.pt")

print("✅ All model files generated in models/")

