import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from almgren_chriss import almgren_chriss_optimal_execution

st.title("📉 Market Impact Model - Almgren-Chriss")

st.markdown("Model the market impact of large trades using Almgren-Chriss optimal execution framework.")

with st.form("ac_form"):
    S0 = st.number_input("Initial price (S₀)", value=100.0)
    X = st.number_input("Total shares to sell (X)", value=1_000.0)
    T = st.number_input("Time to execute (T)", value=1.0)
    N = st.number_input("Number of periods (N)", value=10, step=1)
    sigma = st.number_input("Volatility (σ)", value=0.01)
    eta = st.number_input("Temporary impact (η)", value=0.1)
    gamma = st.number_input("Permanent impact (γ)", value=0.1)
    lam = st.number_input("Risk aversion (λ)", value=1e-6)
    submitted = st.form_submit_button("Run Model")

if submitted:
    t, x_star, cost, var = almgren_chriss_optimal_execution(S0, X, T, int(N), sigma, eta, gamma, lam)

    fig, ax = plt.subplots()
    ax.plot(t, x_star, marker='o')
    ax.set_xlabel("Time")
    ax.set_ylabel("Remaining Inventory")
    ax.set_title("Optimal Execution Trajectory")
    st.pyplot(fig)

    st.success(f"✅ Expected Cost: {cost:.2f} | Variance: {var:.4f}")
