import os
import numpy as np
import pandas as pd
import jax.numpy as jnp
import pickle
from exotedrf.stage4 import bin_at_resolution
import new_unpack
import jax
import matplotlib.pyplot as plt 
jax.config.update('jax_enable_x64', True)


class SpectroData:
    """Simple container for dot notation access."""
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def save(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
    
    @staticmethod
    def load(filepath):
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    
    def save_whitelight_csv(self, output_path):
        df = pd.DataFrame({'time': self.wl_time, 'flux': self.wl_flux, 'flux_err': self.wl_flux_err})
        df.to_csv(output_path, index=False)

def normalize_flux(flux, flux_err, norm_range=slice(0, 50)):
    """Normalize flux arrays by median of first 50 points."""
    flux = np.array(flux)
    flux_err = np.array(flux_err)
    flux_norm = flux * 1.0
    flux_err_norm = flux_err * 1.0
    
    for i in range(flux.shape[0]):
        norm = np.nanmedian(flux[i, norm_range])
        flux_norm[i, :] /= norm
        flux_err_norm[i, :] /= norm
        
    return flux_norm, flux_err_norm


def bin_spectroscopy_data(wavelengths, flux_unbinned, low_res_bins, high_res_bins):
    # Work in numpy here to avoid jax/jnp <-> numpy mixing issues with external libraries.
    # flux_unbinned is (T, L)
    wavelengths = np.array(wavelengths)
    flux_unbinned = np.array(flux_unbinned)

    T, L = flux_unbinned.shape

    # ---  per-wavelength normalization (sum over time) ---
    sum_per_lambda = np.nansum(flux_unbinned, axis=0, keepdims=True)            
    # avoid division by zero
    sum_per_lambda[sum_per_lambda == 0] = np.nan
    flux_norm = flux_unbinned / sum_per_lambda                                     # (T, L)

    if T >= 3:
        resid = 0.5 * (flux_norm[:-2, :] + flux_norm[2:, :]) - flux_norm[1:-1, :] 
        err_per_lambda = np.nanmedian(np.abs(resid), axis=0)                    
    else:
        err_per_lambda = np.full((L,), np.nan)

    # Shapes expected by bin_at_resolution: (n_wavelength, n_time)
    flux_transposed = np.array(flux_unbinned.T)                                    # (L, T)
    flux_err_T = np.tile(err_per_lambda[:, None], (1, T))                          # (L, T)

    # --- Low resolution binning ---
    wl_lr, wl_err_lr, flux_lr, flux_err_lr = bin_at_resolution(
        wavelengths, flux_transposed, flux_err_T, low_res_bins, method='average'
    )

    # --- High resolution binning ---
    if high_res_bins == 'native':
        bin_edges = np.zeros(len(wavelengths) + 1)
        bin_edges[1:-1] = (wavelengths[:-1] + wavelengths[1:]) / 2
        bin_edges[0] = wavelengths[0] - (wavelengths[1] - wavelengths[0]) / 2
        bin_edges[-1] = wavelengths[-1] + (wavelengths[-1] - wavelengths[-2]) / 2
        wl_err_hr = (bin_edges[1:] - bin_edges[:-1]) / 2

        wl_hr, flux_hr, flux_err_hr = wavelengths, flux_transposed, flux_err_T
    else:
        wl_hr, wl_err_hr, flux_hr, flux_err_hr = bin_at_resolution(
            wavelengths, flux_transposed, flux_err_T, high_res_bins, method='average'
        )

    # normalize_flux expects numpy arrays
    flux_lr, flux_err_lr = normalize_flux(flux_lr, flux_err_lr)
    flux_hr, flux_err_hr = normalize_flux(flux_hr, flux_err_hr)

    return {
        'wavelengths_lr': wl_lr, 'wavelengths_err_lr': wl_err_lr, 
        'flux_lr': flux_lr, 'flux_err_lr': flux_err_lr,
        'wavelengths_hr': wl_hr, 'wavelengths_err_hr': wl_err_hr, 
        'flux_hr': flux_hr, 'flux_err_hr': flux_err_hr
    }


def process_spectroscopy_data(instrument, input_dir, output_dir, planet_str, cfg, fits_file, mask_start=None, mask_end=None):
    """Main function to process spectroscopy data."""
    # Unpack data based on instrument
    if instrument == 'NIRSPEC/G395H' or instrument == 'NIRSPEC/G395M':
        nrs = cfg['nrs']
        planet_cfg = cfg['planet']
        prior_duration = planet_cfg['duration']
        prior_t0 = planet_cfg['t0']
        wavelengths, time, flux_unbinned = new_unpack.unpack_nirspec_exoted(fits_file)
        mini_instrument = nrs
    elif instrument == 'NIRISS/SOSS':
        order = cfg['order']
        wavelengths, time, flux_unbinned = new_unpack.unpack_niriss_exoted(fits_file, order)
        mini_instrument = order
    elif instrument == 'MIRI/LRS':
        wavelengths, time, flux_unbinned = new_unpack.unpack_miri_exoted(fits_file)
        mini_instrument = '' 
    else:
        raise NotImplementedError(f'Instrument {instrument} not implemented yet')
    
    wavelengths = np.array(wavelengths)
    time = np.array(time)
    flux_unbinned = np.array(flux_unbinned)  # Shape: (n_time, n_wavelength)
    
    # Remove NaN columns
    nanmask = np.all(np.isnan(flux_unbinned), axis=0)
    wavelengths = wavelengths[~nanmask]
    flux_unbinned = flux_unbinned[:, ~nanmask]

    # Apply time masking criteria (useful for spot-crossings)
    if mask_end:
        if mask_start == False:
            raise print('Time mask for end time supplied but missing start time! Please give mask_start')
    if mask_start:
        if mask_end == False:
            raise print('Time mask for start time supplied but missing end time! Please give mask_end')
        if len(mask_start) > 1:
            timemask = np.zeros_like(time, dtype=bool)
            for start, end in zip(mask_start, mask_end):
                timemask |= (time >= start) & (time <= end)
            time = time[~timemask]
            flux_unbinned = flux_unbinned[~timemask, :]
        else: 
            timemask = (time >= mask_start) & (time <= mask_end)
            time = time[~timemask]
            flux_unbinned = flux_unbinned[~timemask, :]
    
    # Do all the binning
    binned_data = bin_spectroscopy_data(
        wavelengths, flux_unbinned, cfg['resolution'].get('low'), cfg['resolution'].get('high')
    )
    
    # --- White light via nansum/median(nansum) + robust error (3-point residual) ---
    wl_raw = np.nansum(flux_unbinned, axis=1)                  
    wl_norm = wl_raw / np.nanmedian(wl_raw)                    

    if wl_norm.shape[0] >= 3:
        wl_resid = 0.5 * (wl_norm[:-2] + wl_norm[2:]) - wl_norm[1:-1]
        wl_flux_err = np.nanmedian(np.abs(wl_resid))
    else:
        wl_flux_err = np.nan

    wl_flux = wl_norm  

    return SpectroData(
        time=jnp.array(time),
        wavelengths_unbinned=jnp.array(wavelengths),
        flux_unbinned=jnp.array(flux_unbinned),
        wl_time=jnp.array(time),
        wl_flux=wl_flux,
        wl_flux_err=wl_flux_err,
        wavelengths_lr=jnp.array(binned_data['wavelengths_lr']),
        wavelengths_err_lr=jnp.array(binned_data['wavelengths_err_lr']),
        flux_lr=jnp.array(binned_data['flux_lr']),
        flux_err_lr=jnp.array(binned_data['flux_err_lr']),
        wavelengths_hr=jnp.array(binned_data['wavelengths_hr']),
        wavelengths_err_hr=jnp.array(binned_data['wavelengths_err_hr']),
        flux_hr=jnp.array(binned_data['flux_hr']),
        flux_err_hr=jnp.array(binned_data['flux_err_hr']),
        instrument=instrument,
        planet=planet_str,
        mini_instrument=mini_instrument
    )
