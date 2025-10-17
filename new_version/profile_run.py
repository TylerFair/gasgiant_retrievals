import numpy as np
import pickle

from platon.fit_info import FitInfo
from platon.combined_retriever import CombinedRetriever
from platon.constants import R_sun, R_jup, M_jup
from platon.transit_depth_calculator import TransitDepthCalculator
import pandas as pd

# Dummy profiler definition so kernprof doesn't complain
try:
    import builtins
except ImportError:
    import __builtin__ as builtins

if 'profile' not in builtins.__dict__:
    builtins.__dict__['profile'] = lambda f: f

# Load data
df1 = pd.read_csv('hatp18b_niriss_R100.csv')
waves = df1['CENTRALWAVELNG'].values
hw = df1['BANDWIDTH'].values /2
depths = df1['PL_TRANDEP'].values / 100
errors = df1['PL_TRANDEPERR1'].values / 100
bins = 1e-6 * np.array([waves - hw, waves + hw]).T

# Set parameters
Rs = 0.749 * R_sun
Mp = 0.197 * M_jup
Rp = 0.995 * R_jup
T_eq = 852
T_star = 4803

# Get fit_info
retriever = CombinedRetriever()
fit_info = retriever.get_default_fit_info(
    Rs=Rs, Mp=Mp, Rp=Rp,
    logZ=None, CO_ratio=None, fit_vmr=True,
    log_cloudtop_P=3, log_scatt_factor=0., scatt_slope=4,
    T_star=T_star, T_spot=4300, spot_cov_frac=0.05, cloud_cov_frac=0.8,
    profile_type='parametric',
    T0=T_eq, log_P1=2, alpha1=0.9, alpha2=0.8, log_P2=4, log_P3=6)

fit_info.add_uniform_fit_param('Rp', 0.85 * Rp, 1.15 * Rp)
fit_info.add_uniform_fit_param('T0', 300, 1500)
fit_info.add_uniform_fit_param("log_cloudtop_P", 0, 8)
fit_info.add_uniform_fit_param("log_scatt_factor", -4, 8)
fit_info.add_uniform_fit_param("cloud_cov_frac", 0, 1)
all_gases = ["H2O", "CO2", "Na", "K", "CO",  "CH4", "HCN", "NH3", "H2-He"]
fit_info.add_gases_vmr(all_gases, 1e-12, 1e-1)

# Get a random parameter vector
params = fit_info.get_random_params()

# Initialize calculators
transit_calc = TransitDepthCalculator()
transit_calc.change_wavelength_bins(bins)

# Call the likelihood function once for profiling
retriever._ln_like(params, transit_calc, None, fit_info, depths, errors, None, None)

print("Profiling run complete. Check kernprof output for results.")