import matplotlib.pyplot as plt
from . import _cupy_numpy as xp
expn = xp.scipy.special.expn

from pkg_resources import resource_filename

from .constants import h, c, k_B, AMU, G
from .params import NUM_LAYERS, MIN_P, MAX_P

class Profile:
    # Class-level cache for commonly used arrays
    _pressure_cache = {}
    _log_pressure_cache = {}
    
    def __init__(self):
        # Cache the pressure array since it's constant
        cache_key = (NUM_LAYERS, MIN_P, MAX_P)
        if cache_key not in Profile._pressure_cache:
            Profile._pressure_cache[cache_key] = xp.logspace(
                xp.log10(MIN_P),
                xp.log10(MAX_P),
                NUM_LAYERS)
            Profile._log_pressure_cache[cache_key] = xp.log10(Profile._pressure_cache[cache_key])
        
        self.pressures = Profile._pressure_cache[cache_key]
        self._log_pressures = Profile._log_pressure_cache[cache_key]
        self._num_layers = NUM_LAYERS

    def get_temperatures(self):
        return xp.cpu(self.temperatures)

    def get_pressures(self):
        return xp.cpu(self.pressures)
        
    def set_from_params_dict(self, profile_type, params_dict):
        if profile_type == "isothermal":
            self.set_isothermal(params_dict["T"])
        elif profile_type == "parametric":
            self.set_parametric(
                params_dict["T0"], 10**params_dict["log_P1"],
                params_dict["alpha1"], params_dict["alpha2"],
                10**params_dict["log_P2"], 10**params_dict["log_P3"])
        elif profile_type == 'radiative_solution':
            radiative_keys = [
            'T_star', 'Rs', 'a', 'Mp', 'Rp', 'beta', 'log_k_th',
            'log_gamma', 'log_gamma2', 'alpha', 'T_int'
            ]

            tp_params = {
            key: params_dict[key]
            for key in radiative_keys
            if key in params_dict
            }

            self.set_from_radiative_solution(**tp_params)
        elif profile_type == 'twopoint':
            self.set_twopoint(params_dict["T_top"], params_dict["T_bottom"])
        else:
            assert(False)

    def set_from_arrays(self, P_profile, T_profile):
        """Optimized to use cached log pressures"""
        P_profile = xp.array(P_profile)
        T_profile = xp.array(T_profile)
        log_P_profile = xp.log10(P_profile)
        self.temperatures = xp.interp(self._log_pressures, log_P_profile, T_profile)

    def set_isothermal(self, T_day):
        """Optimized to use pre-allocated array"""
        # Use full instead of ones * value for efficiency
        self.temperatures = xp.full(self._num_layers, T_day, dtype=xp.float64)

    def set_parametric(self, T0, P1, alpha1, alpha2, P2, P3):
        """Vectorized parametric model - MAJOR SPEEDUP"""
        # Cache log values to avoid repeated computation
        log_P = self._log_pressures
        log_P0 = xp.log10(xp.amin(self.pressures))
        log_P1 = xp.log10(P1)
        log_P2 = xp.log10(P2)
        log_P3 = xp.log10(P3)
        
        # Pre-compute temperature boundaries
        T1 = T0 + ((log_P1 - log_P0) / alpha1) ** 2
        T2 = T1 - ((log_P1 - log_P2) / alpha2) ** 2
        T3 = T2 + ((log_P3 - log_P2) / alpha2) ** 2
        
        # Vectorized temperature calculation using where/select
        # This replaces the loop entirely
        mask_region1 = log_P < log_P1
        mask_region2 = (log_P >= log_P1) & (log_P < log_P3)
        mask_region3 = log_P >= log_P3
        
        # Pre-allocate result array
        self.temperatures = xp.empty(self._num_layers, dtype=xp.float64)
        
        # Vectorized computation for each region
        self.temperatures[mask_region1] = T0 + ((log_P[mask_region1] - log_P0) / alpha1) ** 2
        self.temperatures[mask_region2] = T2 + ((log_P[mask_region2] - log_P2) / alpha2) ** 2
        self.temperatures[mask_region3] = T3
        
        return T2, T3
    def set_twopoint(self, T_top, T_bottom):
        log_P = self._log_pressures
        log_P_top = xp.log10(xp.amin(self.pressures))
        log_P_bottom = xp.log10(xp.amax(self.pressures))
        self.temperatures = xp.interp(log_P, xp.array([log_P_top, log_P_bottom]), xp.array([T_top, T_bottom]))
    def set_from_opacity(self, T_irr, info_dict, visible_cutoff=0.8e-6, T_int=100):
        """Optimized opacity-based temperature profile"""
        # Convert to arrays once
        wavelengths = xp.array(info_dict["unbinned_wavelengths"])
        stellar_spectrum = xp.array(info_dict["stellar_spectrum"])
        planet_spectrum = xp.array(info_dict["planet_spectrum"])
        absorption_coeffs = xp.array(info_dict["absorption_coeff_atm"])
        radii = xp.array(info_dict["radii"])
        P_profile = xp.array(info_dict["P_profile"])
        T_profile = xp.array(info_dict["T_profile"])
        
        # Compute wavelength differences efficiently
        d_lambda = xp.diff(wavelengths)
        d_lambda = xp.concatenate([d_lambda[:1], d_lambda])  # Faster than append
        
        # Convert spectra (vectorized operations)
        stellar_spectrum_energy = stellar_spectrum * h * c / wavelengths
        planet_spectrum_energy = planet_spectrum * d_lambda
        
        # Create boolean masks once
        visible_mask = wavelengths < visible_cutoff
        thermal_mask = ~visible_mask
        
        # Number density calculation
        n = P_profile / (k_B * T_profile)
        intermediate_n = 0.5 * (n[:-1] + n[1:])
        
        # Compute cross-sections
        sigmas = absorption_coeffs / n[:, xp.newaxis]
        
        # Weighted averages with pre-computed masks
        sigma_v_weighted = sigmas[:, visible_mask] * stellar_spectrum_energy[visible_mask]
        sigma_th_weighted = sigmas[:, thermal_mask] * planet_spectrum_energy[thermal_mask]
        
        sigma_v = xp.median(sigma_v_weighted.sum(axis=1) / stellar_spectrum_energy[visible_mask].sum())
        sigma_th = xp.median(sigma_th_weighted.sum(axis=1) / planet_spectrum_energy[thermal_mask].sum())
        
        gamma = sigma_v / sigma_th
        
        # Optical depth calculation
        dr = -xp.diff(radii)
        d_taus = sigma_th * intermediate_n * dr
        taus = xp.cumsum(d_taus)
        
        # Temperature calculation (vectorized)
        gamma_taus = gamma * taus
        exp_term = xp.exp(-gamma_taus)
        e2 = expn(2, gamma_taus)
        
        T4 = (0.75 * T_int**4 * (2.0/3 + taus) + 
              0.75 * T_irr**4 * (2.0/3 + 
                                 2.0/(3*gamma) * (1 + (gamma_taus/2 - 1) * exp_term) + 
                                 2*gamma/3 * (1 - taus**2/2) * e2))
        
        T = T4 ** 0.25
        self.temperatures = xp.concatenate([T[:1], T])

    def set_from_radiative_solution(self, T_star, Rs, a, Mp, Rp, beta, log_k_th, 
                                    log_gamma, log_gamma2=None, alpha=0, T_int=100):
        """Optimized radiative solution - vectorized calculations"""
        # Convert logs once
        k_th = 10.0 ** log_k_th
        gamma = 10.0 ** log_gamma
        gamma2 = 10.0 ** log_gamma2 if log_gamma2 is not None else None
        
        # Pre-compute constants
        g = G * Mp / Rp**2
        T_eq = beta * xp.sqrt(Rs / (2*a)) * T_star
        taus = k_th * self.pressures / g
        
        # Pre-compute T_eq^4 and T_int^4
        T_eq4 = T_eq ** 4
        T_int4 = T_int ** 4
        
        # Vectorized incoming stream calculation
        def incoming_stream_contribution(gamma_val):
            gamma_taus = gamma_val * taus
            exp_term = xp.exp(-gamma_taus)
            expn_term = expn(2, gamma_taus)
            
            # Fully vectorized computation
            term1 = 2.0/3
            term2 = 2.0/(3*gamma_val) * (1 + (gamma_taus/2 - 1) * exp_term)
            term3 = 2*gamma_val/3 * (1 - taus**2/2) * expn_term
            
            return 0.75 * T_eq4 * (term1 + term2 + term3)
        
        # Main temperature calculation
        e1 = incoming_stream_contribution(gamma)
        T4 = 0.75 * T_int4 * (2.0/3 + taus) + (1 - alpha) * e1
        
        # Add second gamma contribution if present
        if gamma2 is not None:
            e2 = incoming_stream_contribution(gamma2)
            T4 += alpha * e2
            
        self.temperatures = T4 ** 0.25
