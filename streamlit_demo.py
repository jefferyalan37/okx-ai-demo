
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Inject OKX institutional CSS styling
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif !important;
    background: linear-gradient(to bottom, #000814, #001F54) !important;
    color: white !important;
}

h1, h2, h3, h4, h5, h6 {
    font-weight: 700 !important;
    color: white !important;
}

.stButton>button {
    background-color: #1F51FF !important;
    color: white !important;
    padding: 0.75rem 1.5rem;
    font-weight: 700;
    border-radius: 10px;
    border: none;
}

.stTextInput>div>div>input {
    background-color: #001F54 !important;
    color: white !important;
    border-radius: 8px;
    border: 1px solid #1F51FF !important;
}

.stSelectbox>div>div>div>div {
    background-color: #001F54 !important;
    color: white !important;
}

.block-container {
    padding-top: 3rem;
    padding-bottom: 3rem;
    padding-left: 2rem;
    padding-right: 2rem;
}

.css-1d391kg, .css-1kyxreq {
    background-color: transparent !important;
}

.stTabs [data-baseweb="tab-list"] {
    background-color: #000814;
    border-bottom: 1px solid #1F51FF;
}

.stTabs [data-baseweb="tab"] {
    color: white;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

# Hero section
st.markdown("<h1 style='text-align: center;'>Empowering Institutions for the Digital Asset Era</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>A platform for AI-driven execution, trading, and strategy insights</p>", unsafe_allow_html=True)

if st.button("Contact Us"):
    st.success("✅ Thank you — we’ll be in touch soon.")

# Metrics section
col1, col2, col3 = st.columns(3)
col1.metric("Total Traded", "$37T+", "+1.2T MoM")
col2.metric("Assets on Platform", "$25B+", "+5% QoQ")
col3.metric("Trading Instruments", "900+", "+20 added")

# Tabs for navigation
tab1, tab2, tab3 = st.tabs(["Products", "Programs", "Services"])

with tab1:
    st.subheader("Live Exchange Order Book")
    df = pd.DataFrame({
        "Asset": ["BTC", "ETH", "OKB", "SOL", "TON", "DOGE", "XRP"],
        "Price (USD)": [100533.1, 3908.65, 56.05, 230.23, 6.45, 0.414, 2.43],
        "Change": ["+0.09%", "+3.12%", "+1.71%", "+0.72%", "+2.71%", "+0.32%", "-1.02%"]
    })
    st.dataframe(df, use_container_width=True)

with tab2:
    st.write("📈 Custom strategy programs launching soon.")

with tab3:
    st.write("💼 Advanced reporting & white-glove support available.")

# Sample AI forecast chart
st.subheader("AI Forecast (Sample Data)")
data = np.cumsum(np.random.randn(60))  # Simulated trend
st.line_chart(data, use_container_width=True)
