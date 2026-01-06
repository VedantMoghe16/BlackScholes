# Black–Scholes Option Pricing & Volatility Surface

A complete implementation of the Black–Scholes option pricing model with Greeks calculation, implied volatility solving, and 3D volatility surface visualization.

## Overview

This project provides a professional-grade Black–Scholes pricing engine suitable for derivatives coursework, quantitative finance projects, interview demonstrations, and risk and volatility analysis.

## Key Features

- European Call & Put Pricing
- Complete Greeks Suite (Delta, Gamma, Vega, Theta, Rho)
- Implied Volatility Solver
- Vectorized Volatility Surface Generation
- 3D Surface Visualization
- **Interactive Volatility Surface Explorer** (Streamlit Web App)

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/VedantMoghe16/BlackScholes.git
cd BlackScholes

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```bash
# Run the complete pricing and visualization pipeline
python main.py

# Launch the Interactive Volatility Surface Explorer
streamlit run vol_surface.py
```

## Mathematical Background

### Black–Scholes Formula

**Call Option:**

```
C(S,K,T,r,σ) = S·N(d₁) - K·e^(-rT)·N(d₂)
```

**Put Option:**

```
P(S,K,T,r,σ) = K·e^(-rT)·N(-d₂) - S·N(-d₁)
```

**Where:**

```
d₁ = [ln(S/K) + (r + σ²/2)T] / (σ√T)
d₂ = d₁ - σ√T
N(·) = Standard normal cumulative distribution function
```

### Greeks Formulas

| Greek | Symbol | Formula | Units |
|-------|--------|---------|-------|
| Delta | Δ | ∂V/∂S | per $1 |
| Gamma | Γ | ∂²V/∂S² | per $1² |
| Vega | ν | ∂V/∂σ | per 1% |
| Theta | Θ | ∂V/∂t | per day |
| Rho | ρ | ∂V/∂r | per 1% |

## Interactive Volatility Surface Explorer

The `vol_surface.py` module provides a comprehensive web-based interface for exploring volatility surfaces and advanced option pricing models.

### Features

**🎛️ Interactive Web Dashboard**
- Real-time parameter adjustment using intuitive sliders
- Dynamic 3D volatility surface visualization
- Side-by-side comparison of different models

**📊 Advanced Modeling Capabilities**
- **Heston Stochastic Volatility Model:** Captures the reality of non-constant volatility
- **Volatility Smile Generation:** Models market-observed implied volatility patterns
- **Parameter Calibration:** Fits model parameters to market prices

**🔬 Heston Model Implementation**

The Heston model extends Black–Scholes by treating volatility as a stochastic process:

**Asset Price Dynamics:**
```
dS_t = μS_t dt + √v_t S_t dW_t^S
```

**Variance Dynamics:**
```
dv_t = κ(θ - v_t)dt + σ_v √v_t dW_t^v
```

**Parameters:**
- **κ (kappa):** Mean reversion speed - how quickly volatility returns to long-term average
- **θ (theta):** Long-run average variance - the equilibrium level of volatility
- **σ_v (sigma_v):** Volatility of volatility - controls smile curvature
- **ρ (rho):** Correlation between asset and volatility - controls skew (negative ρ creates left skew)

### Launching the Dashboard

```bash
streamlit run vol_surface.py
```

The dashboard opens in your browser at `http://localhost:8501` and provides:
1. **Parameter Controls:** Adjust Heston parameters (κ, θ, σ_v, ρ) in real-time
2. **3D Surface Plot:** Interactive Plotly visualization of implied volatility across strikes and maturities
3. **Model Comparison:** Compare Black–Scholes flat volatility vs. Heston stochastic volatility
4. **Calibration Tools:** Fit model parameters to match market-observed option prices

### Use Cases

- **Educational:** Understand how volatility smile and skew emerge from stochastic volatility
- **Research:** Test different parameter configurations and their impact on option prices
- **Calibration:** Fit Heston parameters to real market data
- **Risk Analysis:** Visualize volatility surface characteristics across strikes and tenors

## Model Assumptions

The Black–Scholes model operates under these assumptions:

- Log-normal asset price dynamics
- Constant volatility
- No arbitrage opportunities
- Frictionless markets (no transaction costs)
- European-style options
- No dividends
- Constant risk-free rate

## Project Structure

```
.
├── main.py                 # Main execution script
├── vol_surface.py          # Interactive Streamlit volatility explorer
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Dependencies

```
numpy>=1.21.0
scipy>=1.7.0
pandas>=1.3.0
matplotlib>=3.4.0
streamlit>=1.28.0
plotly>=5.17.0
```

Install all dependencies:

```bash
pip install -r requirements.txt
```

## Usage Examples

### 1. Price a Single Option

```python
from src.pricing import black_scholes_call, black_scholes_put

# Parameters
S = 100    # Spot price
K = 105    # Strike price
T = 0.5    # 6 months to maturity
r = 0.05   # 5% risk-free rate
sigma = 0.25  # 25% volatility

call_price = black_scholes_call(S, K, T, r, sigma)
put_price = black_scholes_put(S, K, T, r, sigma)

print(f"Call: ${call_price:.4f}")
print(f"Put: ${put_price:.4f}")
```

### 2. Calculate Greeks

```python
from src.greeks import calculate_greeks

greeks = calculate_greeks(S=100, K=100, T=1.0, r=0.05, sigma=0.20)

print(f"Call Delta: {greeks['call_delta']:.4f}")
print(f"Gamma: {greeks['gamma']:.4f}")
print(f"Vega: {greeks['vega']:.4f}")
```

### 3. Find Implied Volatility

```python
from src.implied_vol import implied_volatility

market_price = 10.45
iv = implied_volatility(
    option_price=market_price,
    S=100, K=100, T=1.0, r=0.05,
    option_type='call'
)

print(f"Implied Volatility: {iv*100:.2f}%")
```

### 4. Generate Volatility Surface

```python
from src.surface import generate_volatility_surface
import matplotlib.pyplot as plt

strikes = np.linspace(70, 130, 25)
maturities = np.linspace(0.1, 2.0, 20)

surface = generate_volatility_surface(
    S=100, strikes=strikes, maturities=maturities,
    r=0.05, atm_vol=0.20
)
```

## Output Example

The program displays option prices and Greeks in tabular format, generates a smooth volatility smile across strikes with upward-sloping term structure in maturity, and renders an interactive 3D volatility surface visualization.

## Volatility Surface

The project generates a realistic implied volatility surface using:

### Surface Model

```
σ_impl(K, T) = σ_ATM(T) × [1 + a × m² + b × m³]
```

**Where:**

- `m = ln(K/S)` — Log-moneyness
- `σ_ATM(T) = σ₀ + β√T` — ATM volatility term structure
- `a, b` — Smile curvature parameters

### Surface Parameters

- Spot Price: $100
- Strike Range: $70 – $130
- Maturity Range: 0.1 – 2.0 years
- ATM Volatility: 20%
- Term Structure Slope: 5% √T
- Volatility Smile: Symmetric (a=0.3, b=-0.1)

## License

MIT License

## Contact

For questions or feedback, please open an issue on GitHub.

---

## About

A comprehensive Python-based implementation of the Black-Scholes Option Pricing Model with analytical tools and visualizations for European options.
