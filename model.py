
# IMPORTS

from math import log, sqrt, exp
from scipy.stats import norm
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



# BLACK–SCHOLES CORE

def d1(S, K, T, r, sigma):
    return (log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))

def d2(S, K, T, r, sigma):
    return d1(S, K, T, r, sigma) - sigma * sqrt(T)

def bs_call(S, K, T, r, sigma):
    return S * norm.cdf(d1(S, K, T, r, sigma)) - K * exp(-r * T) * norm.cdf(d2(S, K, T, r, sigma))

def bs_put(S, K, T, r, sigma):
    return K * exp(-r * T) - S + bs_call(S, K, T, r, sigma)



# GREEKS

def call_delta(S, K, T, r, sigma):
    return norm.cdf(d1(S, K, T, r, sigma))

def call_gamma(S, K, T, r, sigma):
    return norm.pdf(d1(S, K, T, r, sigma)) / (S * sigma * sqrt(T))

def call_vega(S, K, T, r, sigma):
    return 0.01 * (S * norm.pdf(d1(S, K, T, r, sigma)) * sqrt(T))

def call_theta(S, K, T, r, sigma):
    return 0.01 * (-(S * norm.pdf(d1(S, K, T, r, sigma)) * sigma) / (2 * sqrt(T))
                   - r * K * exp(-r * T) * norm.cdf(d2(S, K, T, r, sigma)))

def call_rho(S, K, T, r, sigma):
    return 0.01 * (K * T * exp(-r * T) * norm.cdf(d2(S, K, T, r, sigma)))

def put_delta(S, K, T, r, sigma):
    return call_delta(S, K, T, r, sigma) - 1

def put_gamma(S, K, T, r, sigma):
    return call_gamma(S, K, T, r, sigma)

def put_vega(S, K, T, r, sigma):
    return call_vega(S, K, T, r, sigma)

def put_theta(S, K, T, r, sigma):
    return 0.01 * (-(S * norm.pdf(d1(S, K, T, r, sigma)) * sigma) / (2 * sqrt(T))
                   + r * K * exp(-r * T) * norm.cdf(-d2(S, K, T, r, sigma)))

def put_rho(S, K, T, r, sigma):
    return -0.01 * (K * T * exp(-r * T) * norm.cdf(-d2(S, K, T, r, sigma)))



# IMPLIED VOL (PRESERVED BRUTE FORCE)

def implied_volatility(price, S, K, T, r, option_type='C'):
    price = float(price)
    sigma = 0.001

    while sigma < 1:
        if option_type == 'C':
            price_model = bs_call(S, K, T, r, sigma)
        else:
            price_model = bs_put(S, K, T, r, sigma)

        if abs(price - price_model) < 0.001:
            return sigma

        sigma += 0.001

    return None



# INPUT DATA (PRESERVED STRUCTURE)

input_frame = pd.DataFrame(
    {
        "Symbol": ["S", "K", "T", "r", "sigma"],
        "Input": [100.0, 90.0, 1.0, 5.0, 20.0]
    },
    index=["Underlying price", "Strike price", "Time to maturity",
           "Risk-free interest rate", "Volatility"]
)

print(input_frame)

S = input_frame.loc["Underlying price", "Input"]
K = input_frame.loc["Strike price", "Input"]
T = input_frame.loc["Time to maturity", "Input"]
r = input_frame.loc["Risk-free interest rate", "Input"] / 100
sigma = input_frame.loc["Volatility", "Input"] / 100



# PRICE + GREEKS TABLE

price_and_greeks = pd.DataFrame(
    {
        "Call": [
            bs_call(S, K, T, r, sigma),
            call_delta(S, K, T, r, sigma),
            call_gamma(S, K, T, r, sigma),
            call_vega(S, K, T, r, sigma),
            call_rho(S, K, T, r, sigma),
            call_theta(S, K, T, r, sigma),
        ],
        "Put": [
            bs_put(S, K, T, r, sigma),
            put_delta(S, K, T, r, sigma),
            put_gamma(S, K, T, r, sigma),
            put_vega(S, K, T, r, sigma),
            put_rho(S, K, T, r, sigma),
            put_theta(S, K, T, r, sigma),
        ],
    },
    index=["Price", "Delta", "Gamma", "Vega", "Rho", "Theta"],
)

print("\nOption Prices and Greeks:")
print(price_and_greeks)



# VECTORIZED VOLATILITY SURFACE

strikes = np.linspace(80, 120, 30)
maturities = np.linspace(0.1, 2.0, 25)

K_grid, T_grid = np.meshgrid(strikes, maturities)

def vol_surface(K, T, S):
    base = 0.20
    smile = 0.30 * (np.log(K / S))**2
    term = 0.05 * np.sqrt(T)
    return base + smile + term

sigma_surface = vol_surface(K_grid, T_grid, S)



# IMPLIED VOL SURFACE (NEWTON METHOD)

def implied_vol_surface(C, S, K, T, r):
    sigma = np.full_like(C, 0.2)

    for _ in range(50):
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        vega = S * norm.pdf(d1) * np.sqrt(T)

        sigma -= (price - C) / (vega + 1e-8)

    return sigma


call_price_surface = S * norm.cdf(
    (np.log(S / K_grid) + (r + 0.5 * sigma_surface**2) * T_grid)
    / (sigma_surface * np.sqrt(T_grid))
) - K_grid * np.exp(-r * T_grid) * norm.cdf(
    (np.log(S / K_grid) + (r - 0.5 * sigma_surface**2) * T_grid)
    / (sigma_surface * np.sqrt(T_grid))
)

implied_vol_grid = implied_vol_surface(
    call_price_surface, S, K_grid, T_grid, r
)



# 3D VOLATILITY SURFACE PLOT

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection="3d")

surf = ax.plot_surface(
    K_grid,
    T_grid,
    implied_vol_grid,
    cmap="viridis",
    edgecolor="none",
    alpha=0.9
)

ax.set_xlabel("Strike")
ax.set_ylabel("Maturity")
ax.set_zlabel("Implied Volatility")
ax.set_title("3D Implied Volatility Surface")

fig.colorbar(surf, shrink=0.5, aspect=12)
plt.show()
