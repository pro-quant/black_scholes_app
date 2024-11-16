import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

st.title("Expanded Black-Scholes Option Pricing Model with Sensitivities")
st.markdown("For more details on the Black-Scholes model, check out our [documentation page](https://your-quarto-site.com/black_scholes).")

# Input fields for the Black-Scholes model with dividends
S = st.number_input("Stock Price (S)", value=100.0)
K = st.number_input("Strike Price (K)", value=100.0)
T = st.number_input("Time to Maturity (T, in years)", value=1.0)
r = st.number_input("Risk-Free Rate (r, enter as a decimal)", value=0.05)
sigma = st.number_input("Volatility (σ, enter as a decimal)", value=0.2)
q = st.number_input("Dividend Yield (q, enter as a decimal, e.g., 0.02 for 2%)", value=0.0)

# Check if dividend yield is input correctly as a decimal
if q > 1:
    st.warning("Please enter the dividend yield (q) as a decimal, e.g., 0.02 for 2%.")

# Black-Scholes function for call and put options with dividends
def black_scholes(S, K, T, r, sigma, q):
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
    return call_price, put_price, d1, d2

# Greeks: delta, theta, vega, rho, phi, charm, vanna, gamma
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

# Calculate current premiums and Greeks
call_premium, put_premium, d1, d2 = black_scholes(S, K, T, r, sigma, q)
call_delta, put_delta, call_theta, put_theta, call_rho, put_rho, phi_call, phi_put, gamma, vega, vanna = calculate_greeks(S, K, T, r, sigma, q, d1, d2)

# Table data
data = {
    "": ["Premium", "Delta", "Theta", "Rho", "Phi", "Charm", "Vanna", "Gamma", "Vega"],
    "Call": [call_premium, call_delta, call_theta, call_rho, phi_call, None, vanna, gamma, vega],
    "Put": [put_premium, put_delta, put_theta, put_rho, phi_put, None, vanna, gamma, vega]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Style function to color negative values in orange
def highlight_negative(val):
    color = 'orange' if val < 0 else 'black'
    return f'color: {color}'

# Display table with styling
st.subheader("Option Values and Greeks Summary")
st.dataframe(df.style.applymap(highlight_negative, subset=['Call', 'Put']).format(precision=3))
