from jwstdata import SpectroData, process_spectroscopy_data
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 

data_exoted = SpectroData.load('TOI1130b/nirspec_exoted_nrs1/TOI1130b_NIRSPEC_G395H_nrs1_spectroscopy_data.pkl')
plt.scatter(data_exoted.wl_time, data_exoted.wl_flux, label='ExoTED NRS1', s=2, zorder=3)
data_eureka = SpectroData.load('TOI1130b_package/NIRSPEC_NRS1_LINEAR_FIXEDLD/TOI1130b_NIRSPEC_G395H_nrs1_spectroscopy_data.pkl')
plt.scatter(data_eureka.wl_time, data_eureka.wl_flux,s=2,  label='Eureka NRS1')
data_saugata = np.loadtxt('TOI-1130b_NRS1_whitelc.txt')
#plt.scatter(data_saugata[:, 0], data_saugata[:, 1]/np.nanmedian(data_saugata[:,1])-0.00025,  s=2, label='Saugata NRS1')
plt.legend()
sigma1 = np.nanmedian(np.abs(np.diff(data_exoted.wl_flux)))
sigma2 = np.nanmedian(np.abs(np.diff(data_eureka.wl_flux)))
sigma3 = np.nanmedian(np.abs(np.diff(data_saugata[:, 1]/np.nanmedian(data_saugata[:,1]))))
plt.title(f'Tyler Exoted Sigma: {round(sigma1*1e6)}, Tyler Eureka Sigma: {round(sigma2*1e6)}, Saugata Eureka Sigma: {round(sigma3*1e6)} [ppm]')
plt.show()
