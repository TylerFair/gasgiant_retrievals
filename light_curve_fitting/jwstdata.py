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
    """Handle all the binning logic in one place."""
    # Transpose flux for binning: (n_time, n_wavelength) -> (n_wavelength, n_time)
    flux_transposed = jnp.array(flux_unbinned.T)
    # 1) compute the median spectrum (one value per wavelength)
    median_indiv = jnp.nanmedian(flux_transposed, axis=1, keepdims=True)  
    # 2) compute the MAD per wavelength
    #    subtract that median from each time, take abs, median again along time
    err_indiv = 1.4826 * jnp.nanmedian(jnp.abs(flux_transposed - median_indiv),axis=1,)          
    flux_err = jnp.tile(err_indiv[:, None], (1, flux_transposed.shape[1]))

    # Low resolution binning
    wl_lr, wl_err_lr, flux_lr, flux_err_lr = bin_at_resolution(
        wavelengths, flux_transposed, flux_err, low_res_bins, method='average'
    )
 
    
    # High resolution binning
    if high_res_bins == 1:
        # Use unbinned data - create bin edges
        bin_edges = np.zeros(len(wavelengths) + 1)
        bin_edges[1:-1] = (wavelengths[:-1] + wavelengths[1:]) / 2
        bin_edges[0] = wavelengths[0] - (wavelengths[1] - wavelengths[0]) / 2
        bin_edges[-1] = wavelengths[-1] + (wavelengths[-1] - wavelengths[-2]) / 2
        wl_err_hr = (bin_edges[1:] - bin_edges[:-1]) / 2
        
        wl_hr, flux_hr, flux_err_hr = wavelengths, flux_transposed, flux_err
    else:
        wl_hr, wl_err_hr, flux_hr, flux_err_hr = bin_at_resolution(
            wavelengths, flux_transposed, flux_err, high_res_bins, method='average'
        )
    
    # Normalize both
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
        wavelengths, time, flux_unbinned, *_ = new_unpack.unpack_nirspec_exoted(fits_file)
        mini_instrument = nrs
    elif instrument == 'NIRISS/SOSS':
        order = cfg['order']
        wavelengths, time, flux_unbinned, *_ = new_unpack.unpack_niriss_exoted(
            fits_file, order, planet_str, output_dir, 1
        )
        mini_instrument = order
    elif instrument == 'MIRI/LRS':
        wavelengths, time, flux_unbinned, *_ = new_unpack.unpack_miri_exoted(
                fits_file)
        mini_instrument = '' 
    else:
        raise NotImplementedError(f'Instrument {instrument} not implemented yet')
    
    # Clean up data
    wavelengths = np.array(wavelengths)
    time = np.array(time)
    flux_unbinned = np.array(flux_unbinned)  # Shape: (n_time, n_wavelength)
    
    # Remove NaN columns
    nanmask = np.all(np.isnan(flux_unbinned), axis=0)
    wavelengths = wavelengths[~nanmask]
    flux_unbinned = flux_unbinned[:, ~nanmask]

    if mask_end is not None:
        if mask_start is None:
            raise print('Time mask for end time supplied but missing start time! Please give mask_start')
    if mask_start is not None:
        if mask_end is None:
            raise print('Time mask for start time supplied but missing end time! Please give mask_end')
        if isinstance(mask_start, (list, tuple)) and len(mask_start) > 1:
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
        wavelengths, flux_unbinned, cfg['resolution_bins'].get('low'), cfg['resolution_bins'].get('high')
    )
    
    # Create white light curve
    if instrument == 'MIRI/LRS':
        wl_flux = jnp.nanmedian(flux_unbinned, axis=1)
    else:
        wl_flux = jnp.nanmean(flux_unbinned, axis=1)  
    wl_flux_err = 1.4826 * jnp.nanmedian(jnp.abs(wl_flux - jnp.nanmedian(wl_flux)))
    # Create SpectroData object with clean attribute names
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
