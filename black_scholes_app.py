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


# Reset session state directly
def reset_to_defaults():
    defaults = initialize_default_values()
    for key in defaults:
        if key in st.session_state:
            del st.session_state[key]


#  display input fields
def display_inputs(defaults):
    S = st.number_input("Stock Price (S)", value=defaults["S"], key="S", step=1.0)
    K = st.number_input("Strike Price (K)", value=defaults["K"], key="K", step=1.0)
    T = st.number_input("Time to Maturity (T, in years)", value=defaults["T"], key="T", step=0.1)
    r = st.number_input("Risk-Free Rate (r, enter as a decimal)", value=defaults["r"], key="r", step=0.01)
    sigma = st.number_input("Volatility (σ, enter as a decimal)", value=defaults["sigma"], key="sigma", step=0.01)
    q = st.number_input("Dividend Yield (q, enter as a decimal, e.g., 0.02 for 2%)", value=defaults["q"], key="q", step=0.01)
    return S, K, T, r, sigma, q


# validate inputs
def validate_inputs(r, sigma, q):
    valid = True
    if r < 0 or r > 1:
        st.warning("Risk-free rate (r) must be between 0 and 1.")
        valid = False
    if sigma < 0 or sigma > 1:
        st.warning("Volatility (σ) must be between 0 and 1.")
        valid = False
    if q < 0 or q > 1:
        st.warning("Dividend yield (q) must be between 0 and 1.")
        valid = False
    return valid


# Black-Scholes 
def black_scholes(S, K, T, r, sigma, q):
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
    return call_price, put_price, d1, d2


# greeks
def calculate_greeks(S, K, T, r, sigma, q, d1, d2):
    call_delta = np.exp(-q * T) * norm.cdf(d1)
    put_delta = np.exp(-q * T) * (norm.cdf(d1) - 1)
    vega = S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T) / 100
    gamma = np.exp(-q * T) * norm.pdf(d1) / (S * sigma * np.sqrt(T))
    call_theta = (-S * norm.pdf(d1) * sigma * np.exp(-q * T) / (2 * np.sqrt(T))
                  - r * K * np.exp(-r * T) * norm.cdf(d2)
                  + q * S * np.exp(-q * T) * norm.cdf(d1)) / 365
    put_theta = (-S * norm.pdf(d1) * sigma * np.exp(-q * T) / (2 * np.sqrt(T))
                 + r * K * np.exp(-r * T) * norm.cdf(-d2)
                 - q * S * np.exp(-q * T) * norm.cdf(-d1)) / 365
    call_rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
    put_rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100
    return call_delta, put_delta, call_theta, put_theta, call_rho, put_rho, gamma, vega


# Main app logic
def run_app():
    st.title("Black-Scholes Option Pricing Model with Sensitivities")
    
    # Get default values
    defaults = initialize_default_values()

    # Display input fields
    S, K, T, r, sigma, q = display_inputs(defaults)

    # Buttons layout
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Reset"):
            reset_to_defaults()
    with col2:
        calculate = st.button("Calculate")

    if calculate:
        if validate_inputs(r, sigma, q):
            with st.spinner("Calculating..."):
                call_price, put_price, d1, d2 = black_scholes(S, K, T, r, sigma, q)
                call_delta, put_delta, call_theta, put_theta, call_rho, put_rho, gamma, vega = calculate_greeks(
                    S, K, T, r, sigma, q, d1, d2)

            st.success("Calculation completed! Check the table below for results.")
            data = {
                "": ["Premium", "Delta", "Theta", "Rho", "Gamma", "Vega"],
                "Call": [call_price, call_delta, call_theta, call_rho, gamma, vega],
                "Put": [put_price, put_delta, put_theta, put_rho, gamma, vega],
            }
            df = pd.DataFrame(data)
            st.dataframe(df)


# Run the app
run_app()
