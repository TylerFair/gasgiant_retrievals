import numpy as np
import matplotlib.pyplot as plt
import pickle

from platon.fit_info import FitInfo
from platon.combined_retriever import CombinedRetriever
from platon.constants import R_sun, R_earth, M_earth, R_jup, M_jup
from platon.plotter import Plotter
import pandas as pd 
from platon.transit_depth_calculator import TransitDepthCalculator
from platon.TP_profile import Profile


df1 = pd.read_csv('hatp18b_niriss_R100.csv')
waves = df1['CENTRALWAVELNG'].values 
hw = df1['BANDWIDTH'].values /2
depths = df1['PL_TRANDEP'].values / 100  
errors = df1['PL_TRANDEPERR1'].values/100

bins = 1e-6 * np.array([waves - hw, waves + hw]).T

Rs = 0.749 * R_sun
Mp = 0.197 * M_jup
Rp = 0.995 * R_jup
T_eq = 852  
T_star = 4803

profile_type = 'parametric'  # 'isothermal', 'parametric'

retriever = CombinedRetriever()

if profile_type == 'isothermal':
    fit_info = retriever.get_default_fit_info(
        Rs=Rs, Mp=Mp, Rp=Rp,
        logZ=None, CO_ratio=None, fit_vmr=True,
        log_cloudtop_P=3, log_scatt_factor=0., scatt_slope=4, scattering_ref_wavelength=1e-6,  # cloudtop pressure , haze factor, haze slope, haze ref wave
        n=None, log_k=-np.inf, # no mie scattering
        T_star=T_star, T_spot=4300, spot_cov_frac=0.05,
        profile_type=profile_type,
        T=T_eq

    )
elif profile_type == 'parametric':
    fit_info = retriever.get_default_fit_info(
        Rs=Rs, Mp=Mp, Rp=Rp, 
        logZ=None, CO_ratio=None, fit_vmr=True,
        log_cloudtop_P=3, log_scatt_factor=0., scatt_slope=4, scattering_ref_wavelength=1e-6,  # cloudtop pressure , haze factor, haze slope, haze ref wave
        n=None, log_k=-np.inf, # no mie scattering
        T_star=T_star, T_spot=4300, spot_cov_frac=0.05,
        profile_type=profile_type, 
        T0=T_eq, 
        log_P1=2,           
        alpha1=0.9, 
        alpha2=0.8, 
        log_P2=4,
        log_P3=6,           
    )


fit_info.add_gaussian_fit_param('Mp', 0.013 * M_jup) #0.197 * M_jup, 0.013 * M_jup) # can use this 
fit_info.add_uniform_fit_param('Rp', 0.85 * Rp, 1.15 * Rp)
if profile_type == 'isothermal':
    fit_info.add_uniform_fit_param('T', 300, 1000)
elif profile_type == 'parametric':
    fit_info.add_uniform_fit_param('T0', 300, 1000)
    fit_info.add_uniform_fit_param('log_P1', -3, 7)
    fit_info.add_uniform_fit_param('log_P2', -3, 7)
    fit_info.add_uniform_fit_param('log_P3', 3, 7)
    fit_info.add_uniform_fit_param('alpha1', 0.02, 2)
    fit_info.add_uniform_fit_param('alpha2', 0.02, 2)

fit_info.add_uniform_fit_param("log_cloudtop_P", 0, 8) 
fit_info.add_uniform_fit_param("log_scatt_factor", -2, 5) # investigate -3, 3; -4, 10

fit_info.add_uniform_fit_param("scatt_slope", 0, 12) # investigate 2, 5; -2, 20
fit_info.add_uniform_fit_param("T_spot", 0.5 * T_star, 1.5 * T_star)  
fit_info.add_uniform_fit_param("spot_cov_frac", 0, 0.2)
fit_info.add_gaussian_fit_param("T_star", 80) #4803, 80)

fit_info.add_gases_vmr(["Na", "K", "H2O", "CO", "CO2", "CH4", "HCN", "NH3", "H2-He"], 1e-12, 1e-1)

result = retriever.run_multinest(bins, depths, errors,
                                 None, None, None,
                                 fit_info,
                                 #sample="rwalk",
                                 rad_method="xsec",
                                 nlive=250
                                 )
with open("retrieval_result_hatp18b_2.pkl", "wb") as f:
    pickle.dump(result, f)

plotter = Plotter()
plotter.plot_retrieval_transit_spectrum(result, prefix="best_fit")
plotter.plot_retrieval_corner(result, filename="dynesty_corner.png")
