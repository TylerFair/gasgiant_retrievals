import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from platon.TP_profile import Profile
import pickle
from platon.constants import R_sun, R_earth, M_earth, R_jup, M_jup
from platon.transit_depth_calculator import TransitDepthCalculator


main_dir = './'
with open(main_dir+'retrieval_result_hatp18b_2.pkl', 'rb') as f:
    retrieval_result = pickle.load(f)

original_samples = retrieval_result['equal_samples'][:]
original_labels = list(retrieval_result['labels'])
print("Original Labels:", original_labels)

Rs =  0.749 * R_sun  
Mp = original_samples[:,0]
Rp = original_samples[:,1]
T0 = original_samples[:,2]
log_P1 = original_samples[:,3]
log_P2 = original_samples[:,4]
log_P3 = original_samples[:,5]
alpha1 = original_samples[:,6]
alpha2 = original_samples[:,7]
log_cloudtop_P = original_samples[:,8]
log_scattering_factor = original_samples[:,9]
scattering_slope = original_samples[:,10]
T_spot = original_samples[:,11]
spot_cov_frac = original_samples[:,12]
T_star = original_samples[:,13]
cloud_cov_frac = original_samples[:,14]
log_H2O = original_samples[:,15]
log_CO2 = original_samples[:,16]
log_Na = original_samples[:,17]
log_K = original_samples[:,18]
log_CO = original_samples[:,19]
log_CH4 = original_samples[:,20]
log_HCN = original_samples[:,21]
log_NH3 = original_samples[:,22]




p = Profile()
p.set_from_params_dict("parametric", {
    'T0': np.nanmedian(T0), 
    'alpha1': np.nanmedian(alpha1),
    'alpha2': np.nanmedian(alpha2),
    'log_P1': np.nanmedian(log_P1),
    'log_P2': np.nanmedian(log_P2),
    'log_P3': np.nanmedian(log_P3),
})
calculator = TransitDepthCalculator()


#Rp = 1.40 * R_jup
parametric_median = {'T0': np.nanmedian(T0), 
                     'alpha1': np.nanmedian(alpha1) ,
                     'alpha2': np.nanmedian(alpha2) ,
                     'log_P1': np.nanmedian(log_P1),
                     'log_P2': np.nanmedian(log_P2),
                     'log_P3': np.nanmedian(log_P3),
                     }
p = Profile()
p.set_from_params_dict("parametric",parametric_median)
P_profile = p.get_pressures() * 1e-5

temperature_myfit = [] 
for i in range(100):
    j = np.random.randint(0, len(alpha1))
    param_dict = {'T0': T0[j], 
                  'alpha1': alpha1[j],
                  'alpha2': alpha2[j],
                  'log_P1': log_P1[j],
                  'log_P2': log_P2[j],
                  'log_P3': log_P3[j],
                  }
    p.set_from_params_dict("parametric", param_dict)
    temperature_myfit.append(p.get_temperatures())

my_temps = np.nanmedian(temperature_myfit, axis=0)
my_temps_lower, my_temps_upper = np.percentile(temperature_myfit, [16, 84], axis=0)


plt.figure(figsize=(10, 6))
plt.plot(my_temps, P_profile, c='b', label='Tyler (PLATON)')
plt.fill_betweenx(P_profile, my_temps_lower, my_temps_upper, color='b', alpha=0.3)
plt.xlim(400, 1600)
plt.ylim(1e2, 1e-8)
plt.yscale('log')
plt.ylabel('Pressure (bar)')
plt.xlabel('Temperature (K)')
plt.title('Temperature-Pressure Profile')
plt.legend()
plt.savefig('TP_profile_comparison.png')
plt.show()


METRES_TO_UM = 1e6
plt.errorbar(METRES_TO_UM * retrieval_result.transit_wavelengths,
                retrieval_result.transit_depths,
                yerr = retrieval_result.transit_errors,
                fmt='o', mfc='k', mec='k', ecolor='k', ms=4,capsize=5, label="Observed", zorder=5)


lower_spectrum = np.percentile(retrieval_result.random_binned_transit_depths, 5, axis=0)
upper_spectrum = np.percentile(retrieval_result.random_binned_transit_depths, 95, axis=0)
plt.fill_between(METRES_TO_UM * retrieval_result.transit_wavelengths,
                lower_spectrum,
                upper_spectrum,
                color="r",alpha=0.6,label='Tyler 2$\\sigma$', zorder=2)   

plt.plot(METRES_TO_UM * retrieval_result.transit_wavelengths,
            retrieval_result.best_fit_transit_depths,
            color='r', label="Tyler PLATON", zorder=4)     
           

plt.xlabel("Wavelength ($\mu m$)")
plt.ylabel("Transit depth")
plt.xscale('log')
plt.ylim(1.84/100, 2.08/100)
plt.tight_layout()
plt.legend()
plt.savefig('spectrum_comparison_2sigma.png')
plt.show()



# abundance hists 



my_dict = {'log_Na': log_Na, 'log_K': log_K, 'log_H2O': log_H2O,
            'log_CH4': log_CH4, 'log_NH3': log_NH3, 'log_HCN': log_HCN, 
            'log_CO': log_CO, 'log_CO2': log_CO2, 'T0': T0, 
            'alpha1': alpha1, 'alpha2': alpha2, 'log_P1':log_P1-5, 
            'log_P2':  log_P2-5, 'log_P3':  log_P3-5, 'log_Pref': None,
            'log_a': log_scattering_factor, 'gamma': -1*scattering_slope,
            'log_Pcloud': log_cloudtop_P - 5, 'cloud_cov_frac': cloud_cov_frac,
            'spot_cov_frac': spot_cov_frac, 'T_spot': T_spot,
            'T_star': T_star}



keys_to_plot = [
    'log_Na', 'log_K', 'log_H2O', 'log_CH4', 'log_NH3', 'log_HCN', 
    'log_CO', 'log_CO2', 'T0', 'alpha1', 'alpha2', 'log_P1', 
    'log_P2', 'log_P3', 'log_a', 'gamma', 'log_Pcloud',
    'spot_cov_frac', 'T_spot', 'T_star', 'cloud_cov_frac'
]


fig, axes = plt.subplots(6, 4, figsize=(55, 20))
ax = axes.flatten()

plot_index = 0

for key in keys_to_plot:
    if my_dict.get(key) is not None:
        ax[plot_index].hist(my_dict[key], bins=50, alpha=0.5, density=True, label='Tyler (PLATON)', color='blue')
        ax[plot_index].set_title(key)
        
        plot_index += 1

for i in range(plot_index, len(ax)):
    ax[i].axis('off')

plt.tight_layout()
plt.savefig('plot_hists_abundances.png')
plt.show()

def load_posteriors(h2_he_ratio=0.17):
    """
    Parameters
    ----------
    filename : str
       Path + file name to load.
    Returns
    -------
    M_to_H : np.array
       Metallicity array.
    C_O : np.array
       C/O array.
    """

    H0= 0.910734595152
    He0= 0.0882603796846
    C0= 0.000252825549774
    N0= 7.45466155105e-05
    O0= 0.000552066916753
    Na0= 2.02893382781e-06
    K0= 1.32214752038e-07
    S0= 1.48038326085e-05
    HetoH=He0/H0
    posteriors = original_samples

    h2o, co2, na,k,co, ch4, hcn, nh3 = 10**posteriors[:,15:23].T
    #h2o, co2, so2, na, k, co, h2s, ch4, c2h2, hcn, nh3 = 10**posteriors[:,4:15].T
    #h2o, ch4, nh3, hcn, co, co2, so2, h2s = 10**posteriors[:,0:8].T
    # Convert to metallicity and C/O
    N_C = ch4 + co + co2 + hcn
    #N_C = ch4 + co + co2 + hcn
    N_O= h2o +co+(2*co2)
    N_Na= na
    N_K= k
    #N_S= so2 + h2s
    N_N= nh3+hcn
    sum_molecules=h2o+ na+ k +ch4 +nh3+hcn+co+ co2#+ c2h2+so2+h2s
    #sum_molecules=h2o + ch4 + nh3 + hcn + co + co2 + so2 + h2s
    N_H2=(1-sum_molecules)/(1+(h2_he_ratio))

    N_H=2*N_H2+2*h2o+4*ch4+3*nh3+hcn#+2*h2s
    #C to O ratio
    C_O = N_C / N_O
    #print("-----PLANET--------")
    #l2, l1, med, h1, h2= get_median_and_sigmas(N_Na)
    #print('Na',"{:.2e}".format(med),"{:.2e}".format(med-l1),"{:.2e}".format(h1-med))
    #l2, l1, med, h1, h2= get_median_and_sigmas(N_K)
    #print('K',"{:.2e}".format(med),"{:.2e}".format(med-l1),"{:.2e}".format(h1-med))
    #l2, l1, med, h1, h2= get_median_and_sigmas(N_C)
    #print('C',"{:.2e}".format(med),"{:.2e}".format(med-l1),"{:.2e}".format(h1-med))
    #l2, l1, med, h1, h2= get_median_and_sigmas(N_O)
    #print('O',"{:.2e}".format(med),"{:.2e}".format(med-l1),"{:.2e}".format(h1-med))
    #l2, l1, med, h1, h2= get_median_and_sigmas(N_S)
    #print('S',"{:.2e}".format(med),"{:.2e}".format(med-l1),"{:.2e}".format(h1-med))

    #Metallicity
    sum_all=N_C+N_O+N_N # + N_S

    # reference solar values
    H0= 0.910734595152
    He0= 0.0882603796846
    C0= 0.000252825549774
    N0= 7.45466155105e-05
    O0= 0.000552066916753
    Na0= 2.02893382781e-06
    K0= 1.32214752038e-07
    S0= 1.48038326085e-05
    Met0=C0+O0+N0+S0#+Na0+K0
    HHe0=1.-Met0  #adjusting H-He abundances such that everything sums to 1
    HetoH=He0/H0
    H0=HHe0/(HetoH+1)
    He0=HetoH*H0
    M0=Met0/H0

    metallicity=sum_all/N_H
    M_to_H=np.log10(metallicity/M0)

    return np.c_[M_to_H, np.log10(C_O)]

post = load_posteriors(h2_he_ratio=0.17)

metallicity_log10 = post[:, 0]  
c_to_o_log10      = post[:, 1] 

metallicity_xsolar = 10**metallicity_log10     # M/H in ×Solar
c_to_o_linear      = 10**c_to_o_log10          # C/O (linear)

# 16/50/84 percentiles
met_pers  = np.percentile(metallicity_xsolar, [16, 50, 84])
co_pers   = np.percentile(c_to_o_linear,      [16, 50, 84])

# pretty print
print(f"Metallicity (× Solar): {met_pers[1]:.2f} "
      f"+{met_pers[2]-met_pers[1]:.2f} -{met_pers[1]-met_pers[0]:.2f}")

print(f"C/O (linear): {co_pers[1]:.3f} "
      f"+{co_pers[2]-co_pers[1]:.3f} -{co_pers[1]-co_pers[0]:.3f} ")