import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm
from scipy.optimize import minimize
from scipy.integrate import quad

# ==========================================
# 1. Black-Scholes & Implied Volatility Logic
# ==========================================

def bs_price(S, K, T, r, sigma, option_type='call'):
    """Calculates Black-Scholes option price."""
    if T <= 0 or sigma <= 0:
        return max(0, S - K) if option_type == 'call' else max(0, K - S)
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == 'call':
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return price

def implied_volatility(price, S, K, T, r, option_type='call'):
    """
    Back-calculates Implied Volatility from a Market Price.
    Uses simple minimization since Newton-Raphson can be unstable with extreme inputs.
    """
    def objective(sigma):
        return (bs_price(S, K, T, r, sigma, option_type) - price) ** 2
    
    res = minimize(objective, 0.2, bounds=[(0.001, 3.0)], method='L-BFGS-B')
    return res.x[0]

# ==========================================
# 2. Heston Model Logic (The "Smile" Generator)
# ==========================================

def heston_char_func(u, S0, K, T, r, kappa, theta, sigma_v, rho, v0, type_j):
    """
    Heston Characteristic Function.
    This is the complex math that allows Heston to handle stochastic volatility.
    """
    i = 1j
    b_j = kappa - rho * sigma_v if type_j == 1 else kappa
    u_j = 0.5 if type_j == 1 else -0.5
    
    d = np.sqrt((rho * sigma_v * u * i - b_j)**2 - sigma_v**2 * (2 * u_j * u * i - u**2))
    g = (b_j - rho * sigma_v * u * i + d) / (b_j - rho * sigma_v * u * i - d)
    
    C = (r * u * i * T + (kappa * theta / sigma_v**2) * ((b_j - rho * sigma_v * u * i + d) * T - 2 * np.log((1 - g * np.exp(d * T))/(1 - g))))
    
    D = ((b_j - rho * sigma_v * u * i + d) / sigma_v**2) * ((1 - np.exp(d * T)) / (1 - g * np.exp(d * T)))
    
    return np.exp(C + D * v0 + i * u * np.log(S0))

def heston_price(S0, K, T, r, kappa, theta, sigma_v, rho, v0):
    """
    Calculates Call Price using Heston Model via integration.
    """
    # Integration limits
    a, b = 0, 100 
    
    def integrand(phi, type_j):
        func = heston_char_func(phi, S0, K, T, r, kappa, theta, sigma_v, rho, v0, type_j)
        val = np.real(np.exp(-1j * phi * np.log(K)) * func / (1j * phi))
        return val

    # P1 and P2 integrals
    int1, _ = quad(lambda phi: integrand(phi, 1), a, b)
    int2, _ = quad(lambda phi: integrand(phi, 2), a, b)
    
    P1 = 0.5 + (1/np.pi) * int1
    P2 = 0.5 + (1/np.pi) * int2
    
    return S0 * P1 - K * np.exp(-r * T) * P2

# ==========================================
# 3. Streamlit App Interface
# ==========================================

st.set_page_config(layout="wide", page_title="Volatility Surface Explorer")

st.title("⚡ Volatility Surface Explorer")
st.markdown("""
This tool visualizes the difference between **Black-Scholes (constant volatility)** and 
**Heston (stochastic volatility)**. 
- **Black-Scholes** assumes a flat surface.
- **Heston** generates the famous "Volatility Smile" observed in real markets.
""")

# --- Sidebar Inputs ---
st.sidebar.header("Market Parameters")
S0 = st.sidebar.number_input("Spot Price ($)", value=100.0)
r = st.sidebar.number_input("Risk-Free Rate", value=0.03)

st.sidebar.header("Heston Parameters")
st.sidebar.markdown("*These control the shape of the surface/smile*")
kappa = st.sidebar.slider("Mean Reversion (Kappa)", 0.0, 5.0, 2.0, help="How fast vol returns to average")
theta = st.sidebar.slider("Long-run Vol (Theta)", 0.01, 0.5, 0.04, help="The average variance")
sigma_v = st.sidebar.slider("Vol of Vol (Sigma V)", 0.0, 1.0, 0.3, help="Creates the 'Smile' curvature")
rho = st.sidebar.slider("Correlation (Rho)", -1.0, 1.0, -0.7, help="Creates the 'Skew' (tilt)")
v0 = st.sidebar.slider("Initial Vol (v0)", 0.01, 0.5, 0.04)

# --- Main Logic ---

# Generate Grid
strikes = np.linspace(S0 * 0.7, S0 * 1.3, 15)
maturities = np.linspace(0.1, 2.0, 15)
X, Y = np.meshgrid(strikes, maturities)

# Calculate Surface Data
Z_iv = np.zeros_like(X)

with st.spinner('Calculating Heston Surface (Integrating)...'):
    for i in range(len(maturities)):
        for j in range(len(strikes)):
            T_val = maturities[i]
            K_val = strikes[j]
            
            # 1. Get Heston Price
            h_price = heston_price(S0, K_val, T_val, r, kappa, theta, sigma_v, rho, v0)
            
            # 2. Convert Heston Price -> Black-Scholes Implied Volatility
            # This is the "Inverted" step to visualize the surface
            try:
                iv = implied_volatility(h_price, S0, K_val, T_val, r)
            except:
                iv = 0
            Z_iv[i, j] = iv

# --- Plotting ---
fig = go.Figure(data=[go.Surface(
    z=Z_iv, 
    x=strikes, 
    y=maturities, 
    colorscale='Viridis',
    opacity=0.9
)])

fig.update_layout(
    title='Implied Volatility Surface (Heston Model)',
    scene = dict(
        xaxis_title='Strike Price (K)',
        yaxis_title='Time to Maturity (T)',
        zaxis_title='Implied Volatility (IV)',
    ),
    width=900,
    height=600,
    margin=dict(l=65, r=50, b=65, t=90)
)

col1, col2 = st.columns([3, 1])

with col1:
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Calibration Insight")
    st.info(f"""
    **Current Surface Properties:**
    - **Skew:** Driven by Rho ({rho}). Negative rho makes puts expensive (IV higher at low strikes).
    - **Smile:** Driven by Vol-of-Vol ({sigma_v}). Higher values curve the edges up.
    """)
    
    st.markdown("### How to use:")
    st.markdown("1. Change **Rho** to see the surface tilt.")
    st.markdown("2. Increase **Sigma V** to deepen the 'U' shape.")
    st.markdown("3. This surface is what traders *actually* quote, not the raw dollar prices.")

# --- Simple Calibration Demo ---
st.divider()
st.header("Calibration Demo")
st.markdown("Fit Heston parameters to a single mock market price.")

if st.button("Run Calibration Test"):
    # Mock "Market" Price for a specific option
    target_price = 12.50
    test_K = 100
    test_T = 1.0
    
    st.write(f"Target Market Price: ${target_price} (Strike: {test_K}, T: {test_T})")
    
    def error_function(params):
        # params = [kappa, theta, sigma_v, rho, v0]
        # Constraints would be needed for a robust solver, this is a simple demo
        k, th, s_v, rh, v_0 = params
        # Simple bounds enforcement
        if s_v < 0 or v_0 < 0 or th < 0: return 1e6
        if abs(rh) > 1: return 1e6
        
        model_price = heston_price(S0, test_K, test_T, r, k, th, s_v, rh, v_0)
        return (model_price - target_price)**2

    # Initial guess
    x0 = [2.0, 0.04, 0.3, -0.7, 0.04]
    
    with st.spinner("Calibrating..."):
        res = minimize(error_function, x0, method='Nelder-Mead', tol=1e-3)
        
    st.success(f"Calibrated Price: ${heston_price(S0, test_K, test_T, r, *res.x):.4f}")
    st.json({
        "Optimized Kappa": res.x[0],
        "Optimized Theta": res.x[1],
        "Optimized Vol-Vol": res.x[2],
        "Optimized Rho": res.x[3]
    })