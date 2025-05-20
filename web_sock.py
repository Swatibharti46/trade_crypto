import streamlit as st
import asyncio
import websockets
import json
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go
import random

# ----- Setup -----]st.title("📈 Real-time Crypto Trading Simulator (BTC-USDT)")
st.set_page_config(layout="wide", page_title="Crypto Trading Simulator")

# ----- Session State Init -----
if "spread_data" not in st.session_state:
    st.session_state.spread_data = pd.DataFrame(columns=["timestamp", "spread"])
if "last_tick_time" not in st.session_state:
    st.session_state.last_tick_time = None
if "latency" not in st.session_state:
    st.session_state.latency = 0.0
if "orderbook_snapshot" not in st.session_state:
    st.session_state.orderbook_snapshot = {"asks": [], "bids": []}
if "imbalance_data" not in st.session_state:
    st.session_state.imbalance_data = pd.DataFrame(columns=["timestamp", "imbalance"])
if "cumulative_depth" not in st.session_state:
    st.session_state.cumulative_depth = {"asks": [], "bids": []}
if "trade_tape" not in st.session_state:
    st.session_state.trade_tape = []

# ----- Slippage Regression Model (dummy) -----
X_train = np.array([[50, 1], [100, 2], [150, 3], [200, 5]])
y_train = np.array([0.1, 0.3, 0.5, 0.7])
slippage_model = LinearRegression().fit(X_train, y_train)

def estimate_slippage(quantity, volatility):
    return round(slippage_model.predict([[quantity, volatility]])[0], 4)

# ----- Almgren-Chriss Model (Simplified) -----
def almgren_chriss_impact(quantity, volatility):
    gamma = 0.01  # permanent impact coefficient
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
                    message = json.loads(raw)

                    now = datetime.utcnow()

                    # Latency calc
                    if st.session_state.last_tick_time:
                        delta = (now - st.session_state.last_tick_time).total_seconds() * 1000
                        st.session_state.latency = round(delta, 2)
                    st.session_state.last_tick_time = now

                    # Process top 20 asks and bids sorted properly
                    asks = sorted([[float(p), float(q)] for p, q in message["asks"]], key=lambda x: x[0])[:20]
                    bids = sorted([[float(p), float(q)] for p, q in message["bids"]], key=lambda x: -x[0])[:20]

                    best_ask = asks[0][0] if asks else 0
                    best_bid = bids[0][0] if bids else 0
                    spread = best_ask - best_bid

                    # Store spread
                    st.session_state.spread_data.loc[len(st.session_state.spread_data)] = [now, spread]
                    st.session_state.spread_data = st.session_state.spread_data.tail(100)

                    # Store orderbook snapshot
                    st.session_state.orderbook_snapshot = {"asks": asks, "bids": bids}

                    # Order Book Imbalance
                    ask_vol = sum(q for _, q in asks)
                    bid_vol = sum(q for _, q in bids)
                    imbalance = round((bid_vol - ask_vol) / (bid_vol + ask_vol + 1e-8), 4)
                    st.session_state.imbalance_data.loc[len(st.session_state.imbalance_data)] = [now, imbalance]
                    st.session_state.imbalance_data = st.session_state.imbalance_data.tail(100)

                    # Cumulative Depth Calculation
                    cum_asks = []
                    cum_bids = []
                    total = 0
                    for p, q in asks:
                        total += q
                        cum_asks.append((p, total))
                    total = 0
                    for p, q in bids:
                        total += q
                        cum_bids.append((p, total))
                    st.session_state.cumulative_depth = {"asks": cum_asks, "bids": cum_bids}

                    # Simulate trades (trade tape)
                    # Randomly generate 0-3 trades per tick
                    trades_this_tick = []
                    for _ in range(random.randint(0,3)):
                        price = round(random.uniform(best_bid, best_ask), 2)
                        quantity = round(random.uniform(0.1, 2), 2)
                        side = random.choice(["buy", "sell"])
                        timestamp = now.strftime("%H:%M:%S.%f")[:-3]
                        trades_this_tick.append({"price": price, "quantity": quantity, "side": side, "time": timestamp})
                    st.session_state.trade_tape = (st.session_state.trade_tape + trades_this_tick)[-50:]  # keep last 50 trades

                    # Fetch input parameters
                    quantity = st.session_state.quantity
                    volatility = st.session_state.volatility

                    # Compute metrics
                    slippage = estimate_slippage(quantity, volatility)
                    fees = round(0.001 * quantity, 2)  # example fee model
                    market_impact = almgren_chriss_impact(quantity, volatility)
                    net_cost = round(slippage / 100 * quantity + fees + market_impact, 2)
                    maker_taker_ratio = round(np.random.uniform(0.3, 0.7), 2)

                    # Update UI placeholders
                    slippage_placeholder.metric("📉 Slippage (%)", f"{slippage}%")
                    fee_placeholder.metric("💸 Fees ($)", f"${fees}")
                    impact_placeholder.metric("📊 Market Impact ($)", f"${market_impact}")
                    net_cost_placeholder.metric("🧾 Net Cost ($)", f"${net_cost}")
                    maker_taker_placeholder.metric("⚖️ Maker/Taker Ratio", f"{maker_taker_ratio}")
                    latency_placeholder.metric("⏱️ Latency (ms)", f"{st.session_state.latency}")

                    # Spread line chart
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

    st.divider()

    # Order Book Depth Heatmap (Bar chart)
    st.subheader("📶 Order Book Depth (Top 20 levels)")
    depth_fig = go.Figure()
    asks = st.session_state.orderbook_snapshot.get("asks", [])
    bids = st.session_state.orderbook_snapshot.get("bids", [])

    if asks and bids:
        depth_fig.add_trace(go.Bar(
            x=[p for p, _ in bids],
            y=[q for _, q in bids],
            name="Bids",
            marker_color="green"
        ))

        depth_fig.add_trace(go.Bar(
            x=[p for p, _ in asks],
            y=[q for _, q in asks],
            name="Asks",
            marker_color="red"
        ))

    depth_fig.update_layout(
        title="Order Book Depth (Top 20 Levels)",
        xaxis_title="Price",
        yaxis_title="Quantity",
        barmode="overlay",
        height=400
    )
    st.plotly_chart(depth_fig, use_container_width=True)

    # Order Book Imbalance Trend
    st.subheader("📈 Order Book Imbalance Trend")
    imb_fig = go.Figure()
    if not st.session_state.imbalance_data.empty:
        imb_fig.add_trace(go.Scatter(
            x=st.session_state.imbalance_data["timestamp"],
            y=st.session_state.imbalance_data["imbalance"],
            mode="lines+markers",
            name="Imbalance"
        ))
    imb_fig.update_layout(xaxis_title="Time", yaxis_title="Imbalance", height=300)
    st.plotly_chart(imb_fig, use_container_width=True)

    # Cumulative Depth Curve
    st.subheader("📐 Cumulative Depth Curve")
    cum_fig = go.Figure()
    cum_asks = st.session_state.cumulative_depth.get("asks", [])
    cum_bids = st.session_state.cumulative_depth.get("bids", [])

    if cum_asks:
        cum_fig.add_trace(go.Scatter(
            x=[p for p, _ in cum_asks],
            y=[q for _, q in cum_asks],
            mode="lines+markers",
            name="Cumulative Asks",
            line=dict(color="red")
        ))

    if cum_bids:
        cum_fig.add_trace(go.Scatter(
            x=[p for p, _ in cum_bids],
            y=[q for _, q in cum_bids],
            mode="lines+markers",
            name="Cumulative Bids",
            line=dict(color="green")
        ))

    cum_fig.update_layout(
        xaxis_title="Price",
        yaxis_title="Cumulative Quantity",
        height=350
    )
    st.plotly_chart(cum_fig, use_container_width=True)

    # Real-Time Trade Tape
    st.subheader("📝 Real-Time Trade Tape (Last 50 Trades)")
    if st.session_state.trade_tape:
        trade_df = pd.DataFrame(st.session_state.trade_tape)
        # Color buy green, sell red
        def color_side(val):
            color = 'green' if val == 'buy' else 'red'
            return f'color: {color}'

        st.dataframe(trade_df.style.applymap(color_side, subset=["side"]), height=300)
    else:
        st.write("Waiting for trades...")

# ----- Run WebSocket Client -----
if st.session_state.get("run"):
    exchange = st.session_state.exchange.lower()
    symbol = st.session_state.asset
    if exchange == "okx":
        # OKX public L2 snapshot WebSocket topic format
        ws_url = "wss://ws.okx.com:8443/ws/v5/public"
        # Subscribe message for L2 orderbook of the symbol
        subscribe_msg = {
            "op": "subscribe",
            "args": [{"channel": "books5", "instId": symbol}]
        }

        async def run_ws():
            async with websockets.connect(ws_url) as ws:
                await ws.send(json.dumps(subscribe_msg))
                while True:
                    raw = await ws.recv()
                    data = json.loads(raw)

                    # OKX returns some heartbeat/ping messages, ignore
                    if "event" in data:
                        continue
                    if "data" not in data or not data["data"]:
                        continue

                    msg = data["data"][0]
                    # Structure to emulate sample format:
                    # { "asks": [...], "bids": [...] }
                    # But OKX orderbook may differ; extract asks/bids with price and size
                    ob_asks = [[item[0], item[1]] for item in msg.get("asks", [])]
                    ob_bids = [[item[0], item[1]] for item in msg.get("bids", [])]

                    # Prepare a normalized message dict for our processing
                    message = {"asks": ob_asks, "bids": ob_bids}

                    # Call our processing (mimic main logic)
                    now = datetime.utcnow()

                    # Latency calc
                    if st.session_state.last_tick_time:
                        delta = (now - st.session_state.last_tick_time).total_seconds() * 1000
                        st.session_state.latency = round(delta, 2)
                    st.session_state.last_tick_time = now

                    # Process top 20 asks and bids sorted properly
                    asks = sorted([[float(p), float(q)] for p, q in message["asks"]], key=lambda x: x[0])[:20]
                    bids = sorted([[float(p), float(q)] for p, q in message["bids"]], key=lambda x: -x[0])[:20]

                    best_ask = asks[0][0] if asks else 0
                    best_bid = bids[0][0] if bids else 0
                    spread = best_ask - best_bid

                    # Store spread
                    st.session_state.spread_data.loc[len(st.session_state.spread_data)] = [now, spread]
                    st.session_state.spread_data = st.session_state.spread_data.tail(100)

                    # Store orderbook snapshot
                    st.session_state.orderbook_snapshot = {"asks": asks, "bids": bids}

                    # Order Book Imbalance
                    ask_vol = sum(q for _, q in asks)
                    bid_vol = sum(q for _, q in bids)
                    imbalance = round((bid_vol - ask_vol) / (bid_vol + ask_vol + 1e-8), 4)
                    st.session_state.imbalance_data.loc[len(st.session_state.imbalance_data)] = [now, imbalance]
                    st.session_state.imbalance_data = st.session_state.imbalance_data.tail(100)

                    # Cumulative Depth Calculation
                    cum_asks = []
                    cum_bids = []
                    total = 0
                    for p, q in asks:
                        total += q
                        cum_asks.append((p, total))
                    total = 0
                    for p, q in bids:
                        total += q
                        cum_bids.append((p, total))
                    st.session_state.cumulative_depth = {"asks": cum_asks, "bids": cum_bids}

                    # Simulate trades (trade tape)
                    trades_this_tick = []
                    for _ in range(random.randint(0,3)):
                        price = round(random.uniform(best_bid, best_ask), 2)
                        quantity = round(random.uniform(0.1, 2), 2)
                        side = random.choice(["buy", "sell"])
                        timestamp = now.strftime("%H:%M:%S.%f")[:-3]
                        trades_this_tick.append({"price": price, "quantity": quantity, "side": side, "time": timestamp})
                    st.session_state.trade_tape = (st.session_state.trade_tape + trades_this_tick)[-50:]

                    # Fetch input parameters
                    quantity = st.session_state.quantity
                    volatility = st.session_state.volatility

                    # Compute metrics
                    slippage = estimate_slippage(quantity, volatility)
                    fees = round(0.001 * quantity, 2)  # example fee model
                    market_impact = almgren_chriss_impact(quantity, volatility)
                    net_cost = round(slippage / 100 * quantity + fees + market_impact, 2)
                    maker_taker_ratio = round(np.random.uniform(0.3, 0.7), 2)

                    # Update UI placeholders
                    slippage_placeholder.metric("📉 Slippage (%)", f"{slippage}%")
                    fee_placeholder.metric("💸 Fees ($)", f"${fees}")
                    impact_placeholder.metric("📊 Market Impact ($)", f"${market_impact}")
                    net_cost_placeholder.metric("🧾 Net Cost ($)", f"${net_cost}")
                    maker_taker_placeholder.metric("⚖️ Maker/Taker Ratio", f"{maker_taker_ratio}")
                    latency_placeholder.metric("⏱️ Latency (ms)", f"{st.session_state.latency}")

                    # Update spread chart
                    spread_chart.line_chart(st.session_state.spread_data.set_index("timestamp"))

                    await asyncio.sleep(0.5)

        asyncio.run(run_ws())
