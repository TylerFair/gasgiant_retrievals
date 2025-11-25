import matplotlib.pyplot as plt
import numpy as np
from jaxoplanet.orbits.transit import TransitOrbit
from jaxoplanet.light_curves import limb_dark_light_curve

def spot_crossing(t, amp, mu, sigma):
    """A simple Gaussian spot model."""
    return amp * np.exp(-0.5 * (t - mu)**2 / sigma**2)

def plot_map_fits(t, f, yerr, map_params_mcmc, transit_params, file_path):
    """Stub for plotting MAP fits."""
    print(f"INFO: Plotting MAP fits to {file_path}")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_title("MAP Fits (Stub)")
    ax.errorbar(t, f, yerr=yerr, fmt='.', label='Data')
    ax.legend()
    fig.savefig(file_path)
    plt.close(fig)

def plot_map_residuals(t, f, yerr, map_params_mcmc, transit_params, file_path):
    """Stub for plotting MAP residuals."""
    print(f"INFO: Plotting MAP residuals to {file_path}")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_title("MAP Residuals (Stub)")
    ax.errorbar(t, np.zeros_like(t), yerr=yerr, fmt='.', label='Residuals')
    ax.legend()
    fig.savefig(file_path)
    plt.close(fig)

def plot_transmission_spectrum(wavelengths, rors_samples, file_path_prefix):
    """Stub for plotting the transmission spectrum."""
    print(f"INFO: Plotting transmission spectrum to {file_path_prefix}_.png")
    depth_samples = rors_samples**2
    median_depth = np.median(depth_samples, axis=0)
    
    # Handle single vs multi-planet case by checking dimensions
    if median_depth.ndim > 1:
        num_planets = median_depth.shape[1]
    else:
        num_planets = 1
        median_depth = median_depth[:, np.newaxis] # Ensure it's 2D for consistent indexing

    for i in range(num_planets):
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.set_title(f"Transmission Spectrum - Planet {i} (Stub)")
        ax.plot(wavelengths, median_depth[:, i] * 1e6, 'o-')
        ax.set_xlabel("Wavelength (microns)")
        ax.set_ylabel("Transit Depth (ppm)")
        fig.savefig(f"{file_path_prefix}_planet_{i}.png")
        plt.close(fig)

def plot_wavelength_offset_summary(time, flux_data, flux_err, wavelengths, map_params, transit_params, save_path, detrend_type='linear', gp_trend=None, spot_trend=None, jump_trend=None):
    """
    Plots a summary of light curve fits with wavelength-dependent offsets.
    Handles various detrending models.
    """
    n_lcs = flux_data.shape[0]
    ncols = int(np.ceil(np.sqrt(n_lcs)))
    nrows = int(np.ceil(n_lcs / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows), sharex=True, sharey=True)
    axes = np.atleast_1d(axes).flatten()

    t_shift = time - np.min(time)
    num_planets = len(np.atleast_1d(transit_params['period']))

    for i in range(n_lcs):
        ax = axes[i]
        
        # --- 1. Reconstruct Transit Model ---
        total_transit_model = np.zeros_like(time)
        rors_i = np.atleast_1d(map_params['rors'][i])
        u_i = np.asarray(map_params['u'][i])

        for p_idx in range(num_planets):
            orbit = TransitOrbit(
                period=np.atleast_1d(transit_params["period"])[p_idx],
                duration=np.atleast_1d(map_params["duration"])[p_idx],
                impact_param=np.atleast_1d(map_params["b"])[p_idx],
                time_transit=np.atleast_1d(map_params["t0"])[p_idx],
                radius_ratio=rors_i[p_idx],
            )
            planet_model = limb_dark_light_curve(orbit, u_i)(time)
            total_transit_model += planet_model

        # --- 2. Reconstruct Trend Model ---
        trend_model = np.zeros_like(time)
        
        if detrend_type != 'none':
            c_i = map_params['c'][i]
            
            # Spectroscopic models (using templates)
            if 'gp_spectroscopic' in detrend_type:
                trend_parametric = c_i
                if 'linear' in detrend_type: trend_parametric += map_params['v'][i] * t_shift
                if 'quadratic' in detrend_type: trend_parametric += map_params['v'][i] * t_shift + map_params['v2'][i] * t_shift**2
                if 'explinear' in detrend_type: trend_parametric += map_params['v'][i] * t_shift + map_params['A'][i] * np.exp(-t_shift / map_params['tau'][i])
                trend_model = trend_parametric + map_params['A_gp'][i] * gp_trend
            
            elif detrend_type == 'spot_spectroscopic':
                trend_model = c_i + map_params['A_spot'][i] * spot_trend
                
            elif detrend_type == 'linear_discontinuity_spectroscopic':
                trend_model = c_i + map_params['A_jump'][i] * jump_trend
                
            # Purely parametric models
            else:
                v_i = map_params.get('v', [0.0]*n_lcs)[i]
                if detrend_type == 'linear':
                    trend_model = c_i + v_i * t_shift
                elif detrend_type == 'quadratic':
                    trend_model = c_i + v_i * t_shift + map_params['v2'][i] * t_shift**2
                elif detrend_type == 'cubic':
                    trend_model = c_i + v_i * t_shift + map_params['v2'][i] * t_shift**2 + map_params['v3'][i] * t_shift**3
                elif detrend_type == 'quartic':
                    trend_model = c_i + v_i * t_shift + map_params['v2'][i] * t_shift**2 + map_params['v3'][i] * t_shift**3 + map_params['v4'][i] * t_shift**4
                elif detrend_type == 'explinear':
                    trend_model = c_i + v_i * t_shift + map_params['A'][i] * np.exp(-t_shift / map_params['tau'][i])
                elif detrend_type == 'linear_discontinuity':
                    jump_term = np.where(time > map_params['t_jump'][i], map_params['jump'][i], 0.0)
                    trend_model = c_i + v_i * t_shift + jump_term
                elif detrend_type == 'spot':
                    spot_term = spot_crossing(time, map_params['spot_amp'][i], map_params['spot_mu'][i], map_params['spot_sigma'][i])
                    trend_model = c_i + v_i * t_shift + spot_term
                else:
                     # Fallback for simple models if not specified above
                    trend_model = c_i + v_i * t_shift

        # --- 3. Plotting ---
        full_model = total_transit_model + trend_model
        residuals = flux_data[i] - full_model
        
        offset = i * 0.01  # Add offset for clarity
        ax.errorbar(time, flux_data[i] - offset, yerr=flux_err, fmt='.', color='k', alpha=0.3)
        ax.plot(time, full_model - offset, color='r', lw=1.5)
        ax.text(0.05, 0.95, f'{wavelengths[i]:.2f} $\mu$m', transform=ax.transAxes, va='top', ha='left', fontsize=8)

    for i in range(n_lcs, len(axes)):
        axes[i].set_visible(False)

    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"Saved wavelength offset summary plot to {save_path}")
