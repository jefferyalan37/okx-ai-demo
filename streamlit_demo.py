# placeholder for actual streamlit_demo.py content
import streamlit as st
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from joblib import load
from xgboost import XGBRegressor
import matplotlib.pyplot as plt

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = Path('.')
MODEL_DIR  = BASE_DIR / 'models'
DATA_DIR   = BASE_DIR / 'data'

SCALER_PATH   = MODEL_DIR / 'scaler.pkl'
TFT_STATE     = MODEL_DIR / 'tft_state_dict.pt'
XGB_PATH      = MODEL_DIR / 'xgb.json'
STACK_PATH    = MODEL_DIR / 'stack.pkl'
PROCESSED_CSV = DATA_DIR / 'final_bitcoin_hourly_processed.csv'

# ── Load Ensemble ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_ensemble():
    scaler = load(SCALER_PATH)
    stack  = load(STACK_PATH)
    xgb    = XGBRegressor()
    xgb.load_model(str(XGB_PATH))
    # TFT Definition
    class TemporalFusionTransformer(torch.nn.Module):
        def __init__(self, input_size, hidden_dim=64, num_heads=4):
            super().__init__()
            self.inp  = torch.nn.Linear(input_size, hidden_dim)
            self.attn = torch.nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
            self.hid  = torch.nn.Linear(hidden_dim, hidden_dim)
            self.out  = torch.nn.Linear(hidden_dim, 1)
            self.act  = torch.nn.ReLU()
        def forward(self, x):
            x = self.act(self.inp(x))
            x, _ = self.attn(x, x, x)
            x = self.act(self.hid(x))
            return self.out(x)
    # Instantiate and load
    df         = pd.read_csv(PROCESSED_CSV).dropna()
    input_size = df.drop(columns=['close','timestamp']).shape[1]
    tft        = TemporalFusionTransformer(input_size)
    device     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tft.load_state_dict(torch.load(TFT_STATE, map_location=device))
    tft.to(device).eval()
    return scaler, tft, xgb, stack, device

scaler, tft, xgb, stack, device = load_ensemble()

# ── Ensemble Prediction ────────────────────────────────────────────────────────
def predict_ensemble(X_raw: np.ndarray) -> np.ndarray:
    X_scaled = scaler.transform(X_raw)
    tensor   = torch.tensor(X_scaled, dtype=torch.float32, device=device).unsqueeze(1)
    with torch.no_grad():
        tft_pred = tft(tensor).squeeze().cpu().numpy()
    xgb_pred = xgb.predict(X_scaled)
    meta_in  = np.column_stack([X_scaled, tft_pred])
    return stack.predict(meta_in)

# ── Simulation & Optimization Utilities ─────────────────────────────────────────
def simulate_execution(volume: float, horizon: int):
    df       = pd.read_csv(PROCESSED_CSV).dropna()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    prices   = df['close'].values
    hist     = prices[-(horizon+1):]
    returns  = np.diff(hist) / hist[:-1]
    base     = volume / horizon
    weights  = np.where(returns < 0, 2.0, 0.5)
    weights  = weights / weights.sum() * volume
    exec_prices = hist[1:]
    ai_price  = (exec_prices * weights).sum() / volume
    vw_price  = (exec_prices * base).sum()   / volume
    times     = df['timestamp'].iloc[-horizon:].to_list()
    return times, exec_prices, weights, ai_price, vw_price

def optimize_portfolio(assets: list):
    mus, sigs = [], []
    for asset in assets:
        path = DATA_DIR / f'final_{asset.lower()}_hourly_processed.csv'
        df   = pd.read_csv(path).dropna()
        close = df['close']
        X_raw = df.drop(columns=['close','timestamp']).values[-1:].reshape(1, -1)
        pred  = predict_ensemble(X_raw)[0]
        mu    = (pred - close.values[-1]) / close.values[-1]
        sigma = close.pct_change().std()
        mus.append(mu)
        sigs.append(sigma)
    ratios = np.array(mus) / np.array(sigs)
    weights = ratios / ratios.sum()
    return dict(zip(assets, weights)), mus, sigs

# ── Streamlit App ──────────────────────────────────────────────────────────────
st.set_page_config(layout='wide')
st.title('OKX AI Demo Suite')
mode = st.sidebar.radio('Choose a Demo:', [
    'Liquidity Optimization',
    'Risk Surveillance',
    'Compliance & AML',
    'AIOps Intelligence',
    'Trading Signals',
    'Conversational Analytics',
    'Execution Simulator',
    'Portfolio Optimizer'
])

# (Sections omitted for brevity; see full script for details)
# regenerate_models.py
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import os

os.makedirs("models", exist_ok=True)

# Dummy data
X = np.array([[i] for i in range(10)])
y = np.array([i * 2 for i in range(10)])

# Scaler
scaler = StandardScaler()
scaler.fit(X)
joblib.dump(scaler, "models/scaler.pkl")

# XGBoost model
xgb_model = xgb.XGBRegressor()
xgb_model.fit(X, y)
xgb_model.save_model("models/xgb.json")

# Stacking model
stack_model = RandomForestRegressor()
stack_model.fit(X, y)
joblib.dump(stack_model, "models/stack.pkl")

# Fake TFT state dict
tft_state_dict = {"weights": [1, 2, 3]}
joblib.dump(tft_state_dict, "models/tft_state_dict.pt")

print("✅ Model artifacts successfully regenerated.")
