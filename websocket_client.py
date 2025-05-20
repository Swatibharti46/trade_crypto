import streamlit as st
import asyncio
import websockets
import json
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go
from threading import Thread
import time

# ========== Regression Model ========== #
X_train = np.array([[50, 1], [100, 2], [150, 3], [200, 5]])
y_train = np.array([0.1, 0.3, 0.5, 0.7])
slippage_model = LinearRegression().fit(X_train, y_train)

def estimate_slippage(quantity, volatility):
    return round(slippage_model.predict([[quantity, volatility]])[0], 4)

def almgren_chriss_impact(quantity, volatility):
    gamma = 0.01  # market impact coefficient
    eta = 0.002   # volatility scaling
    return round(gamma * quantity + eta * quantity * volatility, 4)

# ========== Global State ========== #
spread_data = pd.DataFrame(columns=["timestamp", "spread"])
imbalance_data = pd.DataFrame(columns=["timestamp", "imbalance"])
orderbook_ready = False

# ========== WebSocket Background Thread ========== #
def start_websocket():
    async def listen():
        uri = "wss://ws.okx.com:8443/ws/v5/public"
        subscribe_msg = {
            "op": "subscribe",
            "args": [{"channel": "books5", "instId": "BTC-USDT-SWAP"}]
        }

        async with websockets.connect(uri) as ws:
            await ws.send(json.dumps(subscribe_msg))
            while True:
                try:
                    msg = await ws.recv()
                    data = json.loads(msg)

                    if "data" not in data or not data["data"]:
                        continue

                    snapshot = data["data"][0]
                    asks = sorted([[float(p), float(q)] for p, q in snapshot["asks"]], key=lambda x: x[0])[:20]
                    bids = sorted([[float(p), float(q)] for p, q in snapshot["bids"]], key=lambda x: -x[0])[:20]

                    if not asks or not bids:
                        continue

                    best_ask = asks[0][0]
                    best_bid = bids[0][0]
                    spread = best_ask - best_bid
                    imbalance = round(
                        (sum(q for _, q in bids) - sum(q for _, q in asks)) / (sum(q for _, q in bids + asks) + 1e-8), 4
                    )
                    now = datetime.utcnow()

                    spread_data.loc[len(spread_data)] = [now, spread]
                    imbalance_data.loc[len(imbalance_data)] = [now, imbalance]
                    global orderbook_ready
                    orderbook_ready = True

                    time.sleep(0.2)
                except Exception as e:
                    print("WebSocket error:", e)
                    await asyncio.sleep(1)

    asyncio.run(listen())

# Launch background thread once
if "ws_thread_started" not in st.session_state:
    Thread(target=start_websocket, daemon=True).start()
    st.session_state["ws_thread_started"] = True

# ========== Streamlit UI ========== #
st.title("📈 Real-time Crypto Trading Simulator (BTC-USDT)")

with st.sidebar:
    st.subheader("📊 Trade Input")
    quantity = st.slider("Trade Quantity (BTC)", 10, 200, 50, 10)
    volatility = st.slider("Estimated Volatility", 1.0, 5.0, 2.0, 0.1)

    if st.button("Simulate Trade"):
        slippage = estimate_slippage(quantity, volatility)
        market_impact = almgren_chriss_impact(quantity, volatility)
        st.success(f"📉 Slippage Estimate: {slippage}")
        st.warning(f"💥 Market Impact (Almgren-Chriss): {market_impact}")

st.markdown("### 📡 Live Order Book Analytics")

if orderbook_ready and len(spread_data) > 1:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=spread_data["timestamp"], y=spread_data["spread"], name="Spread"))
    fig.add_trace(go.Scatter(x=imbalance_data["timestamp"], y=imbalance_data["imbalance"], name="Imbalance"))
    fig.update_layout(
        xaxis_title="Timestamp",
        yaxis_title="Value",
        legend_title="Metrics",
        title="Spread and Order Book Imbalance Over Time"
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("⏳ Waiting for live data... Please wait 5–10 seconds.")
