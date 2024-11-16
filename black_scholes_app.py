import streamlit as st
import numpy as np
import pandas as pd
from scipy.stats import norm

# Function to initialize default values
def initialize_default_values():
    return {
        "S": 100.0,
        "K": 100.0,
        "T": 1.0,
        "r": 0.05,
        "sigma": 0.2,
        "q": 0.0
    }

# Initialize session state
if "values" not in st.session_state:
    st.session_state["values"] = initialize_default_values()

# Reset values directly
def reset_to_defaults():
    defaults = initialize_default_values()
    st.session_state.S = defaults["S"]
    st.session_state.K = defaults["K"]
    st.session_state.T = defaults["T"]
    st.session_state.r = defaults["r"]
    st.session_state.sigma = defaults["sigma"]
    st.session_state.q = defaults["q"]

# Main app
def run_app():
    st.title("Expanded Black-Scholes Option Pricing Model with Sensitivities")
    st.markdown("For more details on the Black-Scholes model, check out our [documentation page](https://your-quarto-site.com/black_scholes).")

    # Input fields
    S = st.number_input("Stock Price (S)", key="S", step=1.0)
    K = st.number_input("Strike Price (K)", key="K", step=1.0)
    T = st.number_input("Time to Maturity (T, in years)", key="T", step=0.1)
    r = st.number_input("Risk-Free Rate (r, enter as a decimal)", key="r", step=0.01)
    sigma = st.number_input("Volatility (σ, enter as a decimal)", key="sigma", step=0.01)
    q = st.number_input("Dividend Yield (q, enter as a decimal, e.g., 0.02 for 2%)", key="q", step=0.01)

    # Buttons layout
    col1, col2 = st.columns([1, 1])

    with col1:
        reset = st.button("Reset")

    with col2:
        calculate = st.button("Calculate")

    # Reset values to defaults
    if reset:
        reset_to_defaults()

    # Validation
    inputs_valid = True
    if r < 0 or r > 1:
        st.warning("Please enter the risk-free rate (r) as a decimal between 0 and 1 (e.g., 0.05 for 5%).")
        inputs_valid = False
    if sigma < 0 or sigma > 1:
        st.warning("Please enter the volatility (σ) as a decimal between 0 and 1 (e.g., 0.2 for 20%).")
        inputs_valid = False
    if q < 0 or q > 1:
        st.warning("Please enter the dividend yield (q) as a decimal between 0 and 1 (e.g., 0.02 for 2%).")
        inputs_valid = False

    # Black-Scholes function
    def black_scholes(S, K, T, r, sigma, q):
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        call_price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
        return call_price, put_price, d1, d2

    # Greeks calculation
    def calculate_greeks(S, K, T, r, sigma, q, d1, d2):
        call_delta = np.exp(-q * T) * norm.cdf(d1)
        put_delta = np.exp(-q * T) * (norm.cdf(d1) - 1)
        vega = S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T) / 100  # Vega per 1% change in vol
        gamma = np.exp(-q * T) * norm.pdf(d1) / (S * sigma * np.sqrt(T))
        call_theta = (-S * norm.pdf(d1) * sigma * np.exp(-q * T) / (2 * np.sqrt(T))
                      - r * K * np.exp(-r * T) * norm.cdf(d2)
                      + q * S * np.exp(-q * T) * norm.cdf(d1)) / 365  # Theta per day
        put_theta = (-S * norm.pdf(d1) * sigma * np.exp(-q * T) / (2 * np.sqrt(T))
                     + r * K * np.exp(-r * T) * norm.cdf(-d2)
                     - q * S * np.exp(-q * T) * norm.cdf(-d1)) / 365  # Theta per day
        call_rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100  # Rho per 1% change in rate
        put_rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100  # Rho per 1% change in rate
        phi_call = -call_delta
        phi_put = -put_delta
        vanna = d1 * vega / S  # Simplified Vanna formula
        return call_delta, put_delta, call_theta, put_theta, call_rho, put_rho, phi_call, phi_put, gamma, vega, vanna

    # Calculate values
    if calculate:
        if inputs_valid:
            with st.spinner("Calculating..."):
                call_premium, put_premium, d1, d2 = black_scholes(S, K, T, r, sigma, q)
                call_delta, put_delta, call_theta, put_theta, call_rho, put_rho, phi_call, phi_put, gamma, vega, vanna = calculate_greeks(
                    S, K, T, r, sigma, q, d1, d2)

            st.success("Calculation completed! Check the table below for results.")

            data = {
                "": ["Premium", "Delta", "Theta", "Rho", "Phi", "Charm", "Vanna", "Gamma", "Vega"],
                "Call": [call_premium, call_delta, call_theta, call_rho, phi_call, None, vanna, gamma, vega],
                "Put": [put_premium, put_delta, put_theta, put_rho, phi_put, None, vanna, gamma, vega],
            }
            df = pd.DataFrame(data)
            st.dataframe(df)
        else:
            st.warning("Fix the highlighted issues above to calculate.")

# Run the app
run_app()
