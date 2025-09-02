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

def normalize_flux(flux, flux_err, norm_range):
    """Normalize flux arrays by median of first 50 points."""
    flux = np.array(flux)
    flux_err = np.array(flux_err)
    flux_norm = flux * 1.0
    flux_err_norm = flux_err * 1.0
    
    for i in range(flux.shape[0]):
        if norm_range is None or np.sum(norm_range) == 0:
            norm_range_fallback = slice(0,150)
            norm = np.nanmedian(flux[i, norm_range_fallback])
        else:
            norm = np.nanmedian(flux[i, norm_range])
        flux_norm[i, :] /= norm
        flux_err_norm[i, :] /= norm
        
    return flux_norm, flux_err_norm


def bin_spectroscopy_data(wavelengths, wavelengths_err, flux_unbinned, flux_err_unbinned, low_res_bins, high_res_bins, oot_mask):
    """Handle all the binning logic in one place."""
    flux_unbinned_copy = flux_unbinned * 1.0
    flux_err_unbinned_copy = flux_err_unbinned * 1.0
    
    # Transpose flux for binning: (n_time, n_wavelength) -> (n_wavelength, n_time)
    flux_transposed = jnp.array(flux_unbinned_copy.T)
    flux_err_transposed = jnp.array(flux_err_unbinned_copy.T)
    
    # Low resolution binning
    wl_lr, wl_err_lr, flux_lr, flux_err_lr = bin_at_resolution(
        wavelengths, flux_transposed, flux_err_transposed, low_res_bins, method='average'
    )
 
    n_lr = min(len(wl_lr), flux_lr.shape[0], flux_err_lr.shape[0], len(wl_err_lr))
    wl_lr, wl_err_lr = wl_lr[:n_lr], wl_err_lr[:n_lr]
    flux_lr, flux_err_lr = flux_lr[:n_lr, :], flux_err_lr[:n_lr, :]

    # High resolution binning
    if high_res_bins == 'native':
        wl_hr, wl_err_hr, flux_hr, flux_err_hr = wavelengths, wavelengths_err, flux_transposed, flux_err_transposed
    else:
        wl_hr, wl_err_hr, flux_hr, flux_err_hr = bin_at_resolution(
            wavelengths, flux_transposed, flux_err_transposed, high_res_bins, method='average'
        )
        
    n_hr = min(len(wl_hr), flux_hr.shape[0], flux_err_hr.shape[0], len(wl_err_hr))
    wl_hr, wl_err_hr = wl_hr[:n_hr], wl_err_hr[:n_hr]
    flux_hr, flux_err_hr = flux_hr[:n_hr, :], flux_err_hr[:n_hr, :]

    flux_lr, flux_err_lr = normalize_flux(flux_lr, flux_err_lr, norm_range=oot_mask)
    flux_hr, flux_err_hr = normalize_flux(flux_hr, flux_err_hr, norm_range=oot_mask)

    '''
    if flux_err_lr.ndim == 2:
        nanmask_lr = np.any(np.isnan(flux_err_lr), axis=1)
    else:
        nanmask_lr = np.isnan(flux_err_lr)
    if flux_err_hr.ndim == 2:
        nanmask_hr = np.any(np.isnan(flux_err_hr), axis=1)
    else:
        nanmask_hr = np.isnan(flux_err_hr)

    wl_lr, wl_err_lr = wl_lr[~nanmask_lr], wl_err_lr[~nanmask_lr]
    flux_lr, flux_err_lr = flux_lr[~nanmask_lr], flux_err_lr[~nanmask_lr]

    wl_hr, wl_err_hr = wl_hr[~nanmask_hr], wl_err_hr[~nanmask_hr]
    flux_hr, flux_err_hr = flux_hr[~nanmask_hr], flux_err_hr[~nanmask_hr]
    '''
    keep_wl_lr = np.isfinite(flux_lr).all(axis=1) & np.isfinite(flux_err_lr).all(axis=1)
    wl_lr, wl_err_lr = wl_lr[keep_wl_lr], wl_err_lr[keep_wl_lr]
    flux_lr, flux_err_lr = flux_lr[keep_wl_lr, :], flux_err_lr[keep_wl_lr, :]

    keep_wl_hr = np.isfinite(flux_hr).all(axis=1) & np.isfinite(flux_err_hr).all(axis=1)
    wl_hr, wl_err_hr = wl_hr[keep_wl_hr], wl_err_hr[keep_wl_hr]
    flux_hr, flux_err_hr = flux_hr[keep_wl_hr, :], flux_err_hr[keep_wl_hr, :]

    keep_t_lr = np.isfinite(flux_lr).all(axis=0) & np.isfinite(flux_err_lr).all(axis=0)
    keep_t_hr = np.isfinite(flux_hr).all(axis=0) & np.isfinite(flux_err_hr).all(axis=0)
    keep_t_post = keep_t_lr & keep_t_hr

    flux_lr, flux_err_lr = flux_lr[:, keep_t_post], flux_err_lr[:, keep_t_post]
    flux_hr, flux_err_hr = flux_hr[:, keep_t_post], flux_err_hr[:, keep_t_post]

    assert wl_lr.shape[0] == flux_lr.shape[0] == flux_err_lr.shape[0] == wl_err_lr.shape[0], "LR channels misaligned"
    assert wl_hr.shape[0] == flux_hr.shape[0] == flux_err_hr.shape[0] == wl_err_hr.shape[0], "HR channels misaligned"


    return {
        'wavelengths_lr': wl_lr, 'wavelengths_err_lr': wl_err_lr, 
        'flux_lr': flux_lr, 'flux_err_lr': flux_err_lr,
        'wavelengths_hr': wl_hr, 'wavelengths_err_hr': wl_err_hr, 
        'flux_hr': flux_hr, 'flux_err_hr': flux_err_hr
    }


def process_spectroscopy_data(instrument, input_dir, output_dir, planet_str, cfg, fits_file, mask_start=None, mask_end=None):
    """Main function to process spectroscopy data."""
    # Unpack data based on instrument
    if instrument == 'NIRSPEC/G395H' or instrument == 'NIRSPEC/G395M' or instrument == 'NIRSPEC/PRISM':
        nrs = cfg['nrs']
        planet_cfg = cfg['planet']
        prior_duration = planet_cfg['duration']
        prior_t0 = planet_cfg['t0']
        wavelengths, wavelengths_err, time, flux_unbinned, flux_err_unbinned = new_unpack.unpack_nirspec_exoted(fits_file)
        mini_instrument = nrs
    elif instrument == 'NIRISS/SOSS':
        order = cfg['order']
        wavelengths, wavelengths_err, time, flux_unbinned, flux_err_unbinned = new_unpack.unpack_niriss_exoted(fits_file, order)
        mini_instrument = order
    elif instrument == 'MIRI/LRS':
        wavelengths, wavelengths_err, time, flux_unbinned, flux_err_unbinned = new_unpack.unpack_miri_exoted(fits_file)
        mini_instrument = '' 
    else:
        raise NotImplementedError(f'Instrument {instrument} not implemented yet')
    
    wavelengths = np.array(wavelengths)
    wavelengths_err = np.array(wavelengths_err)
    time = np.array(time)
    flux_unbinned = np.array(flux_unbinned)  # Shape: (n_time, n_wavelength)
    flux_err_unbinned = np.array(flux_err_unbinned) 
    
    # Remove NaN columns
    nanmask = np.all(np.isnan(flux_unbinned), axis=0)
    wavelengths, wavelengths_err = wavelengths[~nanmask], wavelengths_err[~nanmask]
    flux_unbinned, flux_err_unbinned = flux_unbinned[:, ~nanmask], flux_err_unbinned[:, ~nanmask]

    # Apply time masking criteria (useful for spot-crossings)
    if mask_end:
        if mask_start == False:
            raise print('Time mask for end time supplied but missing start time! Please give mask_start')
    if mask_start:
        if mask_end == False:
            raise print('Time mask for start time supplied but missing end time! Please give mask_end')
        if hasattr(mask_start, '__len__'):
            timemask = np.zeros_like(time, dtype=bool)
            for start, end in zip(mask_start, mask_end):
                timemask |= (time >= start) & (time <= end)
            time = time[~timemask]
            flux_unbinned = flux_unbinned[~timemask, :]
            flux_err_unbinned = flux_err_unbinned[~timemask, :]
        else: 
            timemask = (time >= mask_start) & (time <= mask_end)
            time = time[~timemask]
            flux_unbinned = flux_unbinned[~timemask, :]
            flux_err_unbinned = flux_err_unbinned[~timemask, :]
            
    planet_cfg = cfg['planet']
    prior_t0 = planet_cfg['t0']
    prior_duration = planet_cfg['duration']
    oot_mask = (time < prior_t0 - 0.6 * prior_duration) | (time > prior_t0 + 0.6 * prior_duration)

    # Do all the binning
    binned_data = bin_spectroscopy_data(
        wavelengths, wavelengths_err, flux_unbinned, flux_err_unbinned, cfg['resolution'].get('low'), cfg['resolution'].get('high'), oot_mask
    )
    
    wlc = np.nansum(flux_unbinned, axis=1)
    wl_flux = wlc/np.nanmedian(wlc[oot_mask], axis=0)
    wl_flux_err = np.nanmedian(np.abs(0.5*(wl_flux[0:-2] + wl_flux[2:]) - wl_flux[1:-1]))

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
