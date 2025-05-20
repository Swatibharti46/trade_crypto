# streamlit_app.py

import streamlit as st
import asyncio
import websockets
import json
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.linear_model import LinearRegression




# ----- Setup -----
st.set_page_config(layout="wide", page_title="Crypto Trading Simulator")

# ----- Global State -----
if "spread_data" not in st.session_state:
    st.session_state.spread_data = pd.DataFrame(columns=["timestamp", "spread"])
if "last_tick_time" not in st.session_state:
    st.session_state.last_tick_time = None
if "latency" not in st.session_state:
    st.session_state.latency = 0.0

# ----- Slippage Regression Model -----
X_train = np.array([[50, 1], [100, 2], [150, 3], [200, 5]])
y_train = np.array([0.1, 0.3, 0.5, 0.7])
slippage_model = LinearRegression().fit(X_train, y_train)

def estimate_slippage(quantity, volatility):
    return round(slippage_model.predict([[quantity, volatility]])[0], 4)

# ----- Almgren-Chriss Model (Simplified) -----
def almgren_chriss_impact(quantity, volatility):
    gamma = 0.01  # market impact coefficient
    eta = 0.002   # temporary impact coefficient
    perm_impact = gamma * quantity
    temp_impact = eta * quantity * volatility
    return round(perm_impact + temp_impact, 4)

# ----- Async WebSocket Client -----
async def process_orderbook(uri):
    retry = 0
    while retry < 5:
        try:
            async with websockets.connect(uri) as ws:
                st.toast("✅ Connected to market feed")
                retry = 0
                while True:
                    raw = await ws.recv()
                    data = json.loads(raw)

                    # Latency
                    now = datetime.utcnow()
                    if st.session_state.last_tick_time:
                        delta = (now - st.session_state.last_tick_time).total_seconds() * 1000
                        st.session_state.latency = round(delta, 2)
                    st.session_state.last_tick_time = now

                    # Top-of-book
                    best_ask = float(data["asks"][0][0])
                    best_bid = float(data["bids"][0][0])
                    spread = best_ask - best_bid

                    # Save to chart data
                    st.session_state.spread_data.loc[len(st.session_state.spread_data)] = [now, spread]
                    st.session_state.spread_data = st.session_state.spread_data.tail(100)

                    # Input values
                    quantity = st.session_state.quantity
                    volatility = st.session_state.volatility

                    slippage = estimate_slippage(quantity, volatility)
                    fees = round(0.001 * quantity, 2)
                    market_impact = almgren_chriss_impact(quantity, volatility)
                    net_cost = round(slippage / 100 * quantity + fees + market_impact, 2)
                    maker_taker_ratio = round(np.random.uniform(0.3, 0.7), 2)

                    # Update UI
                    slippage_placeholder.metric("📉 Slippage (%)", f"{slippage}%")
                    fee_placeholder.metric("💸 Fees ($)", f"${fees}")
                    impact_placeholder.metric("📊 Market Impact", f"${market_impact}")
                    net_cost_placeholder.metric("🧾 Net Cost", f"${net_cost}")
                    maker_taker_placeholder.metric("⚖️ Maker/Taker Ratio", f"{maker_taker_ratio}")
                    latency_placeholder.metric("⏱️ Latency (ms)", f"{st.session_state.latency}")

                    spread_chart.line_chart(st.session_state.spread_data.set_index("timestamp"))

                    await asyncio.sleep(0.5)
        except Exception as e:
            retry += 1
            st.warning(f"🔁 Reconnecting... Attempt {retry}/5")
            await asyncio.sleep(2 ** retry)

# ----- UI Layout -----
left, right = st.columns(2)

with left:
    st.header("⚙️ Input Parameters")
    st.selectbox("Exchange", ["OKX"], key="exchange")
    st.text_input("Spot Asset", "BTC-USDT-SWAP", key="asset")
    st.selectbox("Order Type", ["Market"], key="order_type")
    st.number_input("Quantity (USD)", min_value=1.0, value=100.0, key="quantity")
    st.slider("Volatility (%)", 0.0, 10.0, 2.0, key="volatility")
    st.selectbox("Fee Tier", ["Tier 1", "VIP 1", "VIP 2"], key="fee_tier")

    if st.button("🚀 Start Real-Time Simulation"):
        st.session_state.run = True

with right:
    st.header("📊 Output Metrics")
    slippage_placeholder = st.empty()
    fee_placeholder = st.empty()
    impact_placeholder = st.empty()
    net_cost_placeholder = st.empty()
    maker_taker_placeholder = st.empty()
    latency_placeholder = st.empty()
    spread_chart = st.empty()

# ----- Launch Simulation -----
if st.session_state.get("run"):
    uri = "wss://ws.gomarket-cpp.goquant.io/ws/l2-orderbook/okx/BTC-USDT-SWAP"
    asyncio.run(process_orderbook(uri))


import streamlit as st
import plotly.graph_objects as go
from utils import fetch_price_and_volatility  # import your function
import pandas as pd
from almgren_chriss import almgren_chriss_optimal_execution  # ✅ import the function

st.title("Volatility Estimation & Price Chart")

instId = st.text_input("Instrument ID", value="BTC-USDT")
minutes = st.slider("Minutes of data", min_value=10, max_value=100, value=60)

if st.button("Fetch Data"):
    vol, prices, log_returns = fetch_price_and_volatility(instId, minutes)
    if vol is None:
        st.error("Failed to fetch data or compute volatility.")
    else:
        st.success(f"Annualized Volatility: {vol} %")

        # Price line chart
        fig_price = go.Figure()
        fig_price.add_trace(go.Scatter(y=prices, mode='lines', name='Close Price'))
        fig_price.update_layout(title="Close Prices (1-min intervals)", xaxis_title="Time", yaxis_title="Price")
        st.plotly_chart(fig_price, use_container_width=True)

        # Log returns histogram
        fig_returns = go.Figure()
        fig_returns.add_trace(go.Histogram(x=log_returns, nbinsx=30))
        fig_returns.update_layout(title="Log Returns Distribution", xaxis_title="Log Return", yaxis_title="Frequency")
        st.plotly_chart(fig_returns, use_container_width=True)

