# GoQuant Real-Time Crypto Trade Simulator
#  find link of deployed WebApp :https://tradecryptogit-quant-assign.streamlit.app/

## 📈 Objective

This project is a high-performance trade simulator designed for GoQuant's recruitment assignment. It connects to a real-time WebSocket stream from the OKX exchange to process Level 2 (L2) order book data and estimate transaction costs, slippage, fees, and market impact using advanced financial models.

---

## 🛠️ Features

- Live WebSocket connection to OKX SPOT exchange
- Real-time L2 order book data processing
- User Interface with:
  - 📥 **Input Panel:** Order parameters and configurations
  - 📤 **Output Panel:** Processed results and predictions
- Predictive modeling for:
  - Slippage (Linear/Quantile Regression)
  - Market Impact (Almgren-Chriss model)
  - Fees (Tier-based)
  - Maker/Taker Proportion (Logistic Regression)
- Performance profiling and latency benchmarking
- Clean codebase with modular architecture and logging

---

## 🚀 Setup Instructions

 1. Clone the Repository

```bash
git clone https://github.com/Swatibharti46/trade_crypto.git
  ```
```bash
cd trade_crypto
  ```


2. Install Dependencies
bash```
   pip install -r requirements.txt ```

💡 If using Streamlit Cloud, all dependencies will be installed automatically from requirements.txt.


3. Run the App
bash```
   streamlit run streamlit_app.py```
4.  WebSocket Integration
Endpoint:
wss://ws.gomarket-cpp.goquant.io/ws/l2-orderbook/okx/BTC-USDT-SWAP

{
  "timestamp": "2025-05-04T10:39:13Z",
  "exchange": "OKX",
  "symbol": "BTC-USDT-SWAP",
  "asks": [["95445.5", "9.06"], ["95448", "2.05"]],
  "bids": [["95445.4", "1104.23"], ["95445.3", "0.02"]]
}
