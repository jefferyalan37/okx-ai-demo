
import streamlit as st
import numpy as np
import pandas as pd

# Strong global CSS override targeting html/body directly (not scoped)
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;700&display=swap');

html, body {
    font-family: 'Space Grotesk', sans-serif !important;
    background: linear-gradient(to bottom, #000814, #001F54) !important;
    color: white !important;
}

h1, h2, h3, h4, h5, h6, p, div {
    color: white !important;
    font-weight: 700 !important;
}

button {
    background-color: #1F51FF !important;
    color: white !important;
    border-radius: 10px !important;
    padding: 0.75rem 1.5rem !important;
    font-weight: 700 !important;
    border: none !important;
}

input, select, textarea {
    background-color: #001F54 !important;
    color: white !important;
    border-radius: 8px !important;
    border: 1px solid #1F51FF !important;
}

section.main {
    padding-top: 3rem;
    padding-bottom: 3rem;
    padding-left: 2rem;
    padding-right: 2rem;
}
</style>
""", unsafe_allow_html=True)

# Hero UI
st.markdown("<h1 style='text-align: center;'>Empowering Institutions for the Digital Asset Era</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>AI-powered forecasting and execution for institutional crypto strategy</p>", unsafe_allow_html=True)

if st.button("Contact Us"):
    st.success("📨 Message received – we’ll contact you shortly.")

# Key Stats
col1, col2, col3 = st.columns(3)
col1.metric("Total Traded", "$37T+", "+1.2T MoM")
col2.metric("Assets on Platform", "$25B+", "+5% QoQ")
col3.metric("Trading Instruments", "900+", "+20 added")

# Tabs
tab1, tab2, tab3 = st.tabs(["Products", "Programs", "Services"])

with tab1:
    st.subheader("Exchange Order Book")
    df = pd.DataFrame({
        "Asset": ["BTC", "ETH", "OKB", "SOL", "TON", "DOGE", "XRP"],
        "Price (USD)": [100533.1, 3908.65, 56.05, 230.23, 6.45, 0.414, 2.43],
        "Change": ["+0.09%", "+3.12%", "+1.71%", "+0.72%", "+2.71%", "+0.32%", "-1.02%"]
    })
    st.dataframe(df, use_container_width=True)

with tab2:
    st.markdown("📊 Custom forecasting & alpha programs")

with tab3:
    st.markdown("🤝 Onboarding, analytics & support packages")

# AI Forecast
st.subheader("AI Forecast (Sample Output)")
data = np.cumsum(np.random.randn(60))
st.line_chart(data, use_container_width=True)
