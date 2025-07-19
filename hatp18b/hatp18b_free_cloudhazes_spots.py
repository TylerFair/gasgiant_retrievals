import numpy as np
import matplotlib.pyplot as plt
import pickle

from platon.fit_info import FitInfo
from platon.combined_retriever import CombinedRetriever
from platon.constants import R_sun, R_earth, M_earth, R_jup, M_jup
from platon.plotter import Plotter
import pandas as pd 
from platon.transit_depth_calculator import TransitDepthCalculator


df1 = pd.read_csv('hatp18b_niriss_R100.csv')
waves = df1['CENTRALWAVELNG'].values 
hw = df1['BANDWIDTH'].values 
depths = df1['PL_TRANDEP'].values / 100  
errors = np.sqrt((df1['PL_TRANDEPERR1'].values/100)**2 + (df1['PL_TRANDEPERR2'].values/100)**2) 

bins = 1e-6 * np.array([waves - hw, waves + hw]).T
'''plt.errorbar(waves[0:1010], depths[0:1010], yerr=errors[0:1010], fmt='.')
plt.errorbar(waves[1010:2397], depths[1010:2397], yerr=errors[1010:2397], fmt='.')
plt.errorbar(waves[2397:4411], depths[2397:4411], yerr=errors[2397:4411], fmt='.')
plt.errorbar(waves[4411:], depths[4411:], yerr=errors[4411:], fmt='.')
plt.show()'''

#create a Retriever object
retriever = CombinedRetriever()

#create a FitInfo object and set best guess parameters
fit_info = retriever.get_default_fit_info(
    Rs=0.749 * R_sun,  # stellar params
    Mp=0.197 * M_jup, Rp=0.995 * R_jup, T=852, # planet params
    logZ=None, CO_ratio=None, fit_vmr=True, # composition
    #add_gas_absorption=True, add_H_minus_absorption=False, add_scattering=True, add_collisional_absorption=True, # all default params
    log_cloudtop_P=0., log_scatt_factor=0., scatt_slope=4, scattering_ref_wavelength=1e-6,  # cloudtop pressure , haze factor, haze slope, haze ref wave
    n=None, log_k=-np.inf, # no mie scattering
    error_multiple=1, 
    T_star=4803, offset_transit=0, T_spot=4300, spot_cov_frac=0.05)


#fit_info.add_gaussian_fit_param('Mp', 1.35*M_earth)
fit_info.add_uniform_fit_param('Rp', 0.85 * R_jup, 1.15 * R_jup)
fit_info.add_uniform_fit_param('T', 300, 1000)
fit_info.add_uniform_fit_param("log_cloudtop_P", -1, 5)
fit_info.add_uniform_fit_param("log_scatt_factor", -2, 5)
fit_info.add_uniform_fit_param("scatt_slope", 0, 12)
#fit_info.add_uniform_fit_param("offset_niriss", -200e-6, 200e-6)
#fit_info.add_uniform_fit_param("offset_nrs1", -200e-6, 200e-6)
#fit_info.add_uniform_fit_param("offset_miri", -200e-6, 200e-6)
fit_info.add_uniform_fit_param("T_star", 4803, 80)
fit_info.add_uniform_fit_param("T_spot", 3500, 1.2 * 4803)
fit_info.add_uniform_fit_param("spot_cov_frac", 0, 0.5)
fit_info.add_uniform_fit_param("offset_transit", -200e-6, 200e-6)
fit_info.add_uniform_fit_param("error_multiple", 0.5, 5)

fit_info.add_gases_vmr(["Na", "K", "H2O", "CO", "CO2", "CH4", "HCN", "NH3", "H2-He"], 1e-12, 1e-1)

#Use Nested Sampling to do the fitting
result = retriever.run_multinest(bins, depths, errors,
                                 None, None, None,
                                 fit_info,
                                 #sample="rwalk",
                                 rad_method="xsec",
                                 nlive=250
                                 )
with open("retrieval_result_hatp18b.pkl", "wb") as f:
    pickle.dump(result, f)

plotter = Plotter()
plotter.plot_retrieval_transit_spectrum(result, prefix="best_fit")
plotter.plot_retrieval_corner(result, filename="dynesty_corner.png")
