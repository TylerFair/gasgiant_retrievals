import jax.numpy as jnp
from .core import compute_transit_model

def spot_crossing(t, amp, mu, sigma):
    return amp * jnp.exp(-0.5 * (t - mu) **2 / sigma **2)

def compute_lc_none(params, t):
    return compute_transit_model(params, t) + 1.0

def compute_lc_linear(params, t):
    lc_transit = compute_transit_model(params, t)
    trend = params["c"] + params["v"] * (t - jnp.min(t))
    return lc_transit + trend

def compute_lc_quadratic(params, t):
    lc_transit = compute_transit_model(params, t)
    t_norm = t - jnp.min(t)
    trend = params["c"] + params["v"] * t_norm + params["v2"] * t_norm**2
    return lc_transit + trend

def compute_lc_cubic(params, t):
    lc_transit = compute_transit_model(params, t)
    t_norm = t - jnp.min(t)
    trend = params["c"] + params["v"] * t_norm + params["v2"] * t_norm**2 + params["v3"] * t_norm**3
    return lc_transit + trend

def compute_lc_quartic(params, t):
    lc_transit = compute_transit_model(params, t)
    t_norm = t - jnp.min(t)
    trend = params["c"] + params["v"] * t_norm + params["v2"] * t_norm**2 + params["v3"] * t_norm**3 + params["v4"] * t_norm**4
    return lc_transit + trend

def compute_lc_linear_discontinuity(params, t):
    lc_transit = compute_transit_model(params, t)
    jump = jnp.where(t > params["t_jump"], params["jump"], 0.0)
    trend = params["c"] + params["v"] * (t - jnp.min(t)) + jump
    return lc_transit + trend

def compute_lc_explinear(params, t):
    lc_transit = compute_transit_model(params, t)
    trend = params["c"] + params["v"] * (t - jnp.min(t)) + params['A'] * jnp.exp(-(t-jnp.min(t)) / params['tau'])
    return lc_transit + trend

def compute_lc_spot(params, t):
    lc_transit = compute_transit_model(params, t)
    spot = spot_crossing(t, params["spot_amp"], params["spot_mu"], params["spot_sigma"])
    trend = params["c"] + params["v"] * (t - jnp.min(t))
    return lc_transit + trend + spot

# Spectroscopic models that use a fixed trend template
def compute_lc_spot_spectroscopic(params, t, spot_trend):
    lc_transit = compute_transit_model(params, t)
    return lc_transit + params["c"] + params["A_spot"] * spot_trend

def compute_lc_linear_discontinuity_spectroscopic(params, t, jump_trend):
    lc_transit = compute_transit_model(params, t)
    return lc_transit + params["c"] + params["A_jump"] * jump_trend
