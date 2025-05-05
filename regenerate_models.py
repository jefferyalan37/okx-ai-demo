import numpy as np, os, joblib, torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

df = pd.read_csv("data/BTCUSD_d.csv")  # path must match
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)

# Normalize, create features, etc.


# Create models directory
os.makedirs("models", exist_ok=True)

X = np.array([[i] for i in range(10)])
y = np.array([i * 2.5 for i in range(10)])
df = pd.read_csv("OKXDemoProject-5/final_bitcoin_daily_processed.csv")

# Your feature engineering logic here
X = df[["open", "high", "low", "close", "volume"]]  # adjust as needed


# Save scaler
scaler = StandardScaler().fit(X)
joblib.dump(scaler, "models/scaler.pkl")

# Save XGBoost model
xgb_model = xgb.XGBRegressor().fit(X, y)
xgb_model.save_model("models/xgb.json")

# Save stacking model
stack = RandomForestRegressor().fit(X, y)
joblib.dump(stack, "models/stack.pkl")

# Save dummy TFT
model = DummyTFT()
torch.save(model.state_dict(), "models/tft_state_dict.pt")
st.write("Preview BTC Dataset:", df.head())
st.write("Shape:", df.shape)
st.write("Columns:", df.columns)



print("✅ All model files generated in models/")
