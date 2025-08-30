import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from jaxoplanet.light_curves import limb_dark_light_curve
from jaxoplanet.orbits.transit import TransitOrbit
import jax.numpy as jnp

def _spot_sum_for_plot(t, B_row, mu_spot, sigma_spot):
    """Sum_k B_k * exp(-0.5 * (( (t - tmin) - (mu_k - tmin) ) / sigma_k)^2)."""
    if B_row is None or mu_spot is None or sigma_spot is None:
        return 0.0
    B_row = jnp.atleast_1d(jnp.asarray(B_row))
    mu_spot = jnp.atleast_1d(jnp.asarray(mu_spot))
    sigma_spot = jnp.atleast_1d(jnp.asarray(sigma_spot))
    if B_row.size == 0 or mu_spot.size == 0:
        return 0.0
    tmin = jnp.min(t)
    dt = t - tmin
    mu_dt = (mu_spot - tmin)[:, None]           # (K,1)
    z = (dt[None, :] - mu_dt) / sigma_spot[:, None]  # (K,T)
    return (B_row[:, None] * jnp.exp(-0.5 * z**2)).sum(axis=0)  # (T,)

def plot_map_fits(t, indiv_y, yerr, wavelengths, map_params, transit_params, filename, ncols=3, detrend_type='linear'):
    nrows = int(np.ceil(len(wavelengths) / ncols))
    fig = plt.figure(figsize=(15, 3 * nrows))
    gs = gridspec.GridSpec(nrows, ncols, hspace=0, wspace=0)
    norm = plt.Normalize(wavelengths.min(), wavelengths.max())
    colors = plt.cm.winter(norm(wavelengths))

    has_spots = ('B' in map_params) and ('mu_spot' in map_params) and ('sigma_spot' in map_params)
    tmin = jnp.min(t)
    dt = t - tmin

    for i in range(len(wavelengths)):
        ax = plt.subplot(gs[i // ncols, i % ncols])
        rors_i = map_params['rors'][i]
        u_i    = map_params['u'][i]
        c_i    = map_params['c'][i]
        v_i    = map_params['v'][i]

        orbit = TransitOrbit(
            period=transit_params["period"],
            duration=map_params["duration"],
            impact_param=map_params["b"],
            time_transit=map_params["t0"],
            radius_ratio=rors_i,
        )
        model_tr = limb_dark_light_curve(orbit, u_i)(t)

        if detrend_type == 'linear':
            trend = c_i + v_i * dt
        elif detrend_type == 'explinear':
            A_i   = map_params['A'][i]
            tau_i = map_params['tau'][i]
            trend = c_i + v_i * dt + A_i * jnp.exp(-dt / tau_i)
        else:
            raise ValueError(f"Unknown detrend_type: {detrend_type}")

        if has_spots:
            B_row = map_params['B'][i]                  # (K,)
            mu    = map_params['mu_spot']               # (K,)
            sig   = map_params['sigma_spot']            # (K,)
            trend = trend + _spot_sum_for_plot(t, B_row, mu, sig)

        model = (1.0 + model_tr) * (1.0 + trend)

        ax.errorbar(t, indiv_y[i], yerr=yerr[i], fmt='.', alpha=0.3, color=colors[i], ms=1, zorder=2)
        ax.plot(t, model, c='k', lw=2.8, zorder=3)
        ax.text(0.05, 0.95, f'λ = {wavelengths[i]:.2f} μm', transform=ax.transAxes, fontsize=10)
        if i == 0:
            ax.legend(['MAP Model','Data'], fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
    plt.savefig(filename, dpi=200)
    return fig

def plot_map_residuals(t, indiv_y, yerr, wavelengths, map_params, transit_params, filename, ncols=3, detrend_type='linear'):
    nrows = int(np.ceil(len(wavelengths) / ncols))
    fig = plt.figure(figsize=(15, 3 * nrows))
    gs = gridspec.GridSpec(nrows, ncols, hspace=0, wspace=0)
    norm = plt.Normalize(wavelengths.min(), wavelengths.max())
    colors = plt.cm.winter(norm(wavelengths))

    has_spots = ('B' in map_params) and ('mu_spot' in map_params) and ('sigma_spot' in map_params)
    tmin = jnp.min(t)
    dt = t - tmin

    for i in range(len(wavelengths)):
        ax = plt.subplot(gs[i // ncols, i % ncols])
        rors_i = map_params['rors'][i]
        u_i    = map_params['u'][i]
        c_i    = map_params['c'][i]
        v_i    = map_params['v'][i]

        orbit = TransitOrbit(
            period=transit_params["period"],
            duration=map_params["duration"],
            impact_param=map_params["b"],
            time_transit=map_params["t0"],
            radius_ratio=rors_i,
        )
        model_tr = limb_dark_light_curve(orbit, u_i)(t)

        if detrend_type == 'linear':
            trend = c_i + v_i * dt
        elif detrend_type == 'explinear':
            A_i   = map_params['A'][i]
            tau_i = map_params['tau'][i]
            trend = c_i + v_i * dt + A_i * jnp.exp(-dt / tau_i)
        else:
            raise ValueError(f"Unknown detrend_type: {detrend_type}")

        if has_spots:
            B_row = map_params['B'][i]
            mu    = map_params['mu_spot']
            sig   = map_params['sigma_spot']
            trend = trend + _spot_sum_for_plot(t, B_row, mu, sig)

        model = (1.0 + model_tr) * (1.0 + trend)
        residuals = indiv_y[i] - model

        ax.errorbar(t, residuals, yerr=yerr[i], fmt='.', alpha=0.3, ms=1, color=colors[i])
        ax.axhline(y=0, color='k', lw=2.0, zorder=3)
        ax.text(0.05, 0.95, f'λ = {wavelengths[i]:.3f} μm', transform=ax.transAxes, fontsize=10)
        rms = np.nanmedian(np.abs(np.diff(residuals))) * 1e6
        ax.text(0.05, 0.85, f'Noise = {round(rms)}', transform=ax.transAxes, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
    plt.savefig(filename, dpi=200)
    return fig

def plot_transmission_spectrum(wavelengths, rors_posterior, filename):
    """
    Plot the transmission spectrum from MCMC results.
    
    Parameters:
    -----------
    wavelengths : array-like
        The wavelengths at which the spectrum is measured
    rors_posterior : array-like
        The posterior samples for rors (radius ratio)
    """

    depth_chain = rors_posterior**2
    depth_median = np.percentile(depth_chain, [50], axis=0)
    depth_lower, depth_upper = depth_median - np.percentile(depth_chain, [16], axis=0), np.percentile(depth_chain, [84], axis=0) - depth_median
    fig = plt.figure(figsize=(10, 8))
    print(jnp.shape(depth_median))
    depth_median, depth_lower, depth_upper = depth_median[0,:], depth_lower[0,:], depth_upper[0,:]
    print(jnp.shape(depth_median))
    # Just plot the transmission spectrum
    plt.errorbar(wavelengths, depth_median*1e6, 
                    yerr=[depth_lower*1e6, depth_upper*1e6],
                    fmt='o', mfc='white', mec='r', ecolor='r')
    plt.xlabel("Wavelength (µm)")
    plt.ylabel("Depth (ppm)")
    plt.savefig(filename, dpi=200)

    return fig
