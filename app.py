import streamlit as st
from websocket_client import start_ws_thread, orderbook_data
from qua.okx_api import get_spot_assets, fetch_and_compute_volatility
from models import predict_slippage, calculate_almgren_chriss, predict_maker_taker
import time


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


# Start WebSocket thread once
if "ws_started" not in st.session_state:
    start_ws_thread()
    st.session_state.ws_started = True

# Sidebar Inputs
with st.sidebar:
    st.title("Input Parameters")
    exchange = st.selectbox("Exchange", ["OKX"], index=0)
    asset = st.selectbox("Spot Asset", get_spot_assets())
    order_type = st.selectbox("Order Type", ["market"], index=0)
    quantity_usd = st.number_input("Quantity (USD)", min_value=10.0, value=100.0, step=10.0)
    
    if st.button("Estimate Volatility"):
        vol = fetch_and_compute_volatility(asset)
        if vol:
            st.session_state.volatility = vol
            st.success(f"Estimated Volatility: {vol}%")
        else:
            st.error("Volatility estimation failed.")
    volatility = st.number_input("Volatility (%)", value=st.session_state.get("volatility", 2.0))

    fee_tier = st.selectbox("Fee Tier", ["Regular", "VIP1", "VIP2"])  # Example tiers

# Main Output Panel
st.title("Processed Output Values")

# Show some live orderbook info
st.subheader("Real-Time Orderbook Data")
col1, col2 = st.columns(2)
col1.metric("Top Bid", orderbook_data["bids"][0][0] if orderbook_data["bids"] else "N/A")
col2.metric("Top Ask", orderbook_data["asks"][0][0] if orderbook_data["asks"] else "N/A")
st.metric("Internal Latency (ms)", orderbook_data.get("latency_ms", 0))
st.text(f"Last Update: {orderbook_data.get('timestamp', 'N/A')}")

# Compute models (use simplified placeholder functions)
slippage = predict_slippage(orderbook_data["bids"], orderbook_data["asks"], quantity_usd, volatility)
market_impact = calculate_almgren_chriss(quantity_usd, volatility)
maker_taker = predict_maker_taker(orderbook_data["bids"], orderbook_data["asks"])

fees = quantity_usd * 0.001  # Example fee 0.1%

net_cost = slippage + fees + market_impact

st.subheader("Estimated Costs")
st.write(f"Expected Slippage: {slippage:.4f} USD")
st.write(f"Expected Fees: {fees:.4f} USD")
st.write(f"Expected Market Impact: {market_impact:.4f} USD")
st.write(f"Net Cost: {net_cost:.4f} USD")
st.write(f"Maker/Taker Proportion (logistic regression): {maker_taker:.4f}")
