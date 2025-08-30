import os
import sys
import glob
from functools import partial
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import numpyro_ext.distributions as distx
import numpyro_ext.optim as optimx
import matplotlib.pyplot as plt
import pandas as pd
from jaxoplanet.light_curves import limb_dark_light_curve
from jaxoplanet.orbits.transit import TransitOrbit
from exotic_ld import StellarLimbDarkening
from plotting_lineartrend import plot_map_fits, plot_map_residuals, plot_transmission_spectrum
import new_unpack
import argparse
import yaml
import arviz as az 
from exotedrf.stage4 import bin_at_resolution
from jwstdata import SpectroData, process_spectroscopy_data
#pd.set_option('display.max_columns', None)
#pd.set_option('display.max_rows', None)
import tinygp 
# ---------------------
# Model Functions
# ---------------------
DTYPE = jnp.float64

def _to_f64(x):
    if isinstance(x, (np.ndarray, jnp.ndarray)) and jnp.issubdtype(jnp.asarray(x).dtype, jnp.floating):
        return jnp.asarray(x, DTYPE)
    return x

def _tree_to_f64(tree):
    return jax.tree_util.tree_map(_to_f64, tree)
    
def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def _compute_transit_model(params, t):
    """Transit Model."""
    orbit = TransitOrbit(
        period=params["period"],
        duration=params["duration"],
        impact_param=params["b"],
        time_transit=params["t0"],
        radius_ratio=params["rors"],
    )
    return limb_dark_light_curve(orbit, params["u"])(t)

def compute_lc_linear(params, t):
    """Computes transit + linear trend."""
    lc_transit = _compute_transit_model(params, t)
    trend = params["c"] + params["v"] * (t - jnp.min(t))
    return (1.0 + lc_transit) * (1.0 + trend)

def compute_lc_explinear(params, t):
    """Computes transit + exponential-linear trend."""
    lc_transit = _compute_transit_model(params, t)
    trend = params["c"] + params["v"] * (t - jnp.min(t)) + params['A'] * jnp.exp(-t / params['tau'])
    return (1.0 + lc_transit) * (1.0 + trend)

def compute_lc_gp_mean(params, t):
    """The mean function for the GP model is just the transit."""
    return (_compute_transit_model(params, t) + 1.0) * (1.0 + params["c"])

def create_whitelight_model(detrend_type='linear'):
    """
    Building a static whitelight model so jax doesn't retrace. 
    """
    print(f"Building whitelight model with: detrend_type='{detrend_type}'")
    def _whitelight_model_static(t, yerr, y=None, prior_params=None):
        logD = numpyro.sample("logD", dist.Normal(jnp.log(prior_params['duration']), 1e-2))
        duration = numpyro.deterministic("duration", jnp.exp(logD))
        t0 = numpyro.sample("t0", dist.Normal(prior_params['t0'], 1e-1))
        _b = numpyro.sample("_b", dist.Uniform(-2.0, 2.0))
        b = numpyro.deterministic('b', jnp.abs(_b))
        u = numpyro.sample('u', dist.Uniform(-3.0, 3.0).expand([2]))
        depths = numpyro.sample('depths', dist.TruncatedNormal(prior_params['rors']**2, prior_params['rors']**2 * 0.2, low=0.0, high=1.0))
        rors = numpyro.deterministic("rors", jnp.sqrt(depths))

        params = {
            "period": prior_params['period'], "duration": duration, "t0": t0, "b": b,
            "rors": rors, "u": u,
        }

        # The returned model will only contain ONE of these blocks.
        if detrend_type == 'linear':
            params['c'] = numpyro.sample('c', dist.Normal(0.0, 0.1))
            params['v'] = numpyro.sample('v', dist.Normal(0.0, 0.1))
            lc_model = compute_lc_linear(params, t)
            numpyro.sample('obs', dist.Normal(lc_model, yerr), obs=y)

        elif detrend_type == 'explinear':
            params['c'] = numpyro.sample('c', dist.Normal(0.0, 0.1))
            params['v'] = numpyro.sample('v', dist.Normal(0.0, 0.1))
            params['A'] = numpyro.sample('A', dist.Normal(0.0, 0.1))
            params['tau'] = numpyro.sample('tau', dist.Normal(1.0, 0.5))
            lc_model = compute_lc_explinear(params, t)
            numpyro.sample('obs', dist.Normal(lc_model, yerr), obs=y)

        elif detrend_type == 'gp':
            params['c'] = numpyro.sample('c', dist.Normal(0.0, 0.1))
            params['v'] = 0.0 

            logs2 = numpyro.sample('logs2', dist.Uniform(2*jnp.log(1e-6), 2*jnp.log(1.0)))
            GP_log_sigma = numpyro.sample('GP_log_sigma', dist.Uniform(jnp.log(1e-6), jnp.log(1.0)))
            GP_log_rho = numpyro.sample('GP_log_rho', dist.Uniform(jnp.log(1e-3), jnp.log(1e3)))

            mean_func = partial(compute_lc_gp_mean, params)
            kernel = tinygp.kernels.quasisep.Matern32(
                scale=jnp.exp(GP_log_rho),
                sigma=jnp.exp(GP_log_sigma),
            )
            gp = tinygp.GaussianProcess(kernel, t, diag=jnp.exp(logs2), mean=mean_func)
            numpyro.sample('obs', gp.numpyro_dist(), obs=y)
        else:
            raise ValueError(f"Unknown detrend_type: {detrend_type}")

    return _whitelight_model_static
    
def create_vectorized_model(detrend_type='linear', ld_mode='free', trend_mode='free'):
    """
    Build a static vectorized model for individual res calls. 
    It must be built in this way or else we deal with numerous if statement problems. 
    """
    print(f"Building vectorized model with: detrend='{detrend_type}', ld='{ld_mode}', trend='{trend_mode}'")

    if detrend_type == 'linear':
        compute_lc_kernel = compute_lc_linear
    elif detrend_type == 'explinear':
        compute_lc_kernel = compute_lc_explinear
    elif detrend_type == 'gp':
        compute_lc_kernel = compute_lc_gp_mean
    else:
        raise ValueError(f"Unsupported detrend_type for vectorized model: {detrend_type}")

    def _vectorized_model_static(t, yerr, y=None, mu_duration=None, mu_t0=None,
                               mu_depths=None, PERIOD=None, trend_fixed=None,
                               ld_interpolated=None, ld_fixed=None):

        num_lcs = jnp.atleast_2d(yerr).shape[0]

        logD = numpyro.sample("logD", dist.Normal(jnp.log(mu_duration), 0.001))
        duration = numpyro.deterministic("duration", jnp.exp(logD))
        t0 = numpyro.sample("t0", dist.Normal(mu_t0, 1e-1))
        _b = numpyro.sample("_b", dist.Uniform(-2.0, 2.0))
        b = numpyro.deterministic('b', jnp.abs(_b))
        depths = numpyro.sample('depths', dist.TruncatedNormal(mu_depths, 0.2 * jnp.ones_like(mu_depths), low=0.0, high=1.0).expand([num_lcs]))
        rors = numpyro.deterministic("rors", jnp.sqrt(depths))

        if ld_mode == 'free':
            u = numpyro.sample('u', dist.Uniform(-3.0, 3.0).expand([num_lcs, 2]))
        elif ld_mode == 'fixed':
            u = numpyro.deterministic("u", ld_fixed)
        elif ld_mode == 'interpolated':
            u = numpyro.deterministic("u", ld_interpolated)
        else:
            raise ValueError(f"Unknown ld_mode: {ld_mode}")

        params = {
            "period": PERIOD, "duration": duration, "t0": t0, "b": b, "rors": rors, "u": u,
        }

        in_axes = {"period": None, "duration": None, "t0": None, "b": None, "rors": 0, "u": 0}

        if trend_mode == 'free':
            params['c'] = numpyro.sample('c', dist.Normal(0.0, 0.1).expand([num_lcs]))
            params['v'] = numpyro.sample('v', dist.Normal(0.0, 0.1).expand([num_lcs]))
            in_axes.update({'c': 0, 'v': 0})
            if detrend_type == 'explinear':
                params['A'] = numpyro.sample('A', dist.Normal(0.0, 0.1).expand([num_lcs]))
                params['tau'] = numpyro.sample('tau', dist.Normal(1.0, 0.5).expand([num_lcs]))
                in_axes.update({'A': 0, 'tau': 0})
        elif trend_mode == 'fixed':
            trend_temp = numpyro.deterministic('trend_temp', trend_fixed)
            params['c'] = numpyro.deterministic('c', trend_temp[:, 0])
            params['v'] = numpyro.deterministic('v', trend_temp[:, 1])
            in_axes.update({'c': 0, 'v': 0})
            if detrend_type == 'explinear':
                params['A'] = numpyro.deterministic('A', trend_temp[:, 2])
                params['tau'] = numpyro.deterministic('tau', trend_temp[:, 3])
                in_axes.update({'A': 0, 'tau': 0})
        else:
            raise ValueError(f"Unknown trend_mode: {trend_mode}")

        y_model = jax.vmap(compute_lc_kernel, in_axes=(in_axes, None))(params, t)
        numpyro.sample('obs', dist.Normal(y_model, yerr), obs=y)

    return _vectorized_model_static
    
def get_samples(model, key, t, yerr, indiv_y, init_params, **model_kwargs):
    """
    Runs MCMC using NUTS to get posterior samples.
    Accepts additional keyword arguments to pass directly to the model.
    """
    t = _to_f64(t)
    yerr = _to_f64(yerr)
    indiv_y = _to_f64(indiv_y)
    init_params = _tree_to_f64(init_params)
    
    model_kwargs = _tree_to_f64(model_kwargs)
        
    mcmc = numpyro.infer.MCMC(
        numpyro.infer.NUTS(
            model,
            regularize_mass_matrix=False,
            init_strategy=numpyro.infer.init_to_value(values=init_params),
            target_accept_prob=0.9
        ),
        num_warmup=1000,
        num_samples=1000,
        progress_bar=True,
        jit_model_args=True
    )
    
    #  **model_kwargs unpacks the dictionary into kwargs
    # e.g., if model_kwargs is {'ld_fixed': arr},mcmc.run(..., ld_fixed=arr)
    mcmc.run(key, t, yerr, y=indiv_y, **model_kwargs)

    return mcmc.get_samples()

def compute_aic(n, residuals, k):
    """Computes an approximate AIC value."""
    rss = np.sum(np.square(residuals))
    rss = rss if rss > 1e-10 else 1e-10
    aic = 2*k + n * np.log(rss/n)
    return aic

def get_limb_darkening(sld, wavelengths, wavelength_err, instrument, order=None):
    """Gets limb darkening coefficients using boundary ranges for out-of-range wavelengths."""
    if instrument == 'NIRSPEC/G395H':
        mode = "JWST_NIRSpec_G395H"
        wl_min, wl_max = 28700.0, 51700.0 #29000, 52000  # Angstroms
    if instrument == 'NIRSPEC/G395M':
        mode = "JWST_NIRSpec_G395M"
        wl_min, wl_max = 28700.0, 51700.0 #29000, 52000  # Angstroms
    elif instrument == 'NIRISS/SOSS':
        mode = f"JWST_NIRISS_SOSSo{order}"
        wl_min, wl_max = 8300.0, 28100.0  # Angstroms
    elif instrument == 'MIRI/LRS':
        mode = f'JWST_MIRI_LRS'
        wl_min, wl_max = 50000.0, 120000.0

    wavelengths = np.array(wavelengths)
    wavelength_err = np.array(wavelength_err) if hasattr(wavelength_err, '__len__') else wavelength_err

    U_mu = []

    if hasattr(wavelength_err, '__len__'):  # Array of errors
        for i in range(len(wavelengths)):
            wl_angstrom = wavelengths[i] * 1e4
            err_angstrom = wavelength_err[i] * 1e4

            # Calculate the intended range
            intended_min = wl_angstrom - err_angstrom
            intended_max = wl_angstrom + err_angstrom

            # Check if range exceeds boundaries
            if intended_max > wl_max:
                # Use boundary range
                if err_angstrom > 0:
                    range_min = max(wl_max - err_angstrom * 2, wl_min)
                    range_max = wl_max
                print(f"Using boundary range for {wavelengths[i]:.4f} μm: [{range_min:.1f}, {range_max:.1f}] Å")
            elif intended_min < wl_min:
                #  logic for lower boundary
                if err_angstrom <= 0:
                    print('Issue here, err_angstrom is', err_angstrom)
                if err_angstrom > 0:
                    range_min = wl_min
                    range_max = min(wl_min + err_angstrom * 2, wl_max)
                print(f"Using boundary range for {wavelengths[i]:.4f} μm: [{range_min:.1f}, {range_max:.1f}] Å")
            else:
                # Normal case - within bounds
                range_min = intended_min
                range_max = intended_max

            U_mu.append(sld.compute_quadratic_ld_coeffs(
                wavelength_range=[range_min, range_max],
                mode=mode,
                return_sigmas=False
            ))

        U_mu = jnp.array(U_mu)
    else:
        # Single error case - clip the overall range
        wl_range_clipped = [max(min(wavelengths)*1e4, wl_min),
                           min(max(wavelengths)*1e4, wl_max)]
        U_mu = sld.compute_quadratic_ld_coeffs(
            wavelength_range=wl_range_clipped,
            mode=mode,
            return_sigmas=False
        )
        U_mu = jnp.array(U_mu)

    return U_mu




def fit_model_map(t, yerr, indiv_y, init_params,
                 mu_duration, mu_t0, mu_depths, PERIOD,
                 key=None, trend_fixed=None, ld_interpolated=None, ld_fixed=None):
    """Fits the model using MAP optimization."""
    if key is None:
        key = jax.random.PRNGKey(111)

    vmodel = partial(vectorized_model,
                     mu_duration=mu_duration, mu_t0=mu_t0, mu_depths=mu_depths,
                     PERIOD=PERIOD, trend_fixed=trend_fixed, ld_interpolated=ld_interpolated, ld_fixed=ld_fixed)

    soln = optimx.optimize(vmodel, start=init_params)(key, t, yerr, y=indiv_y)

    if trend_fixed is not None:
        soln["c"] = np.array(trend_fixed["c"])
        soln["v"] = np.array(trend_fixed["v"])
    if ld_interpolated is not None:
        soln["u"] = np.array(ld_interpolated)
    if ld_fixed is not None:
        soln["u"] = np.array(ld_fixed)

    return soln

def fit_polynomial(x, y, poly_orders):
    """Fits polynomials and selects the best order using AIC."""
    best_order = None
    best_aic = np.inf
    best_coeffs = None

    for deg in poly_orders:
        y_med = np.median(y, axis=0)
        coeffs = np.polyfit(x, y_med, deg)
        pred = np.polyval(coeffs, x)
        aic = compute_aic(len(x), y_med - pred, k=deg+1)

        if aic < best_aic:
            best_aic = aic
            best_order = deg
            best_coeffs = coeffs

    return best_coeffs, best_order, best_aic

def plot_poly_fit(x, y, coeffs, order, xlabel, ylabel, title, save_path):
    """Plots data and its polynomial fit, then saves it."""
    x_fit = np.linspace(x.min(), x.max(), 200)
    y_fit = np.polyval(coeffs, x_fit)

    y_med = np.median(y, axis=0)
    y_err = np.std(y, axis=0)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(x, y_med, yerr=y_err, fmt='o', ls='', ecolor='k', mfc='k', mec='k')
    ax.plot(x_fit, y_fit, '-', label=f'Poly fit (deg {order})')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"Saved polynomial fit plot to {save_path}")

def save_results(wavelengths, samples, csv_filename):
    """Computes and saves transmission spectrum results to CSV."""
    depth_chain = samples['rors']**2
    depth_median = np.nanmedian(depth_chain, axis=0)
    depth_err = np.std(depth_chain, axis=0)

    output_data = np.column_stack((wavelengths, depth_median, depth_err))
    np.savetxt(
        csv_filename, output_data, delimiter=",",
        header="wavelength,depth,depth_err", comments=""
    )
    print(f"Transmission spectroscopy data saved to {csv_filename}")


# ---------------------
# Main Analysis
# ---------------------
def main():
    parser = argparse.ArgumentParser(description="Run transit analysis with YAML config.")
    parser.add_argument(
        "-c", "--config",
        required=True,
        help="Path to YAML configuration file"
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
   #validate_config(cfg)


    instrument = cfg['instrument']
    if instrument == 'NIRSPEC/G395H' or 'NIRSPEC/G395M':
        nrs = cfg['nrs']
    elif instrument == 'NIRISS/SOSS':
        order = cfg['order']
    elif instrument == 'MIRI/LRS':
        pass
    else:
        raise NotImplementedError('Instrument not implemented yet')
    planet_cfg = cfg['planet']
    stellar_cfg = cfg['stellar']


    flags = cfg.get('flags', {})
    bins = cfg.get('resolution', {})
    outlier_clip = cfg.get('outlier_clip', {})
    planet_str = planet_cfg['name']

    # Paths
    base_path = cfg.get('path', '.')
    input_dir = os.path.join(base_path, cfg.get('input_dir', planet_str + '_NIRSPEC'))
    if not os.path.exists(input_dir):
        print(f"Error: No Planet Folder detected. Need directory: {input_dir}")
        sys.exit(1)
    output_dir = os.path.join(base_path, cfg.get('output_dir', planet_str + '_RESULTS'))
    fits_file = os.path.join(input_dir, cfg.get('fits_file'))
    if fits_file is None:
        print("Error: `fits_file` not specified in the config file.")
        sys.exit(1)
    if not os.path.exists(fits_file):
        print(f"Error: FITS file not found at: {fits_file}")
        sys.exit(1)
    os.makedirs(output_dir, exist_ok=True)

    # device (please use gpu!!)
    host_device = cfg.get('host_device', 'gpu').lower()
    numpyro.set_platform(host_device)
    key_master = jax.random.PRNGKey(555)

    # flags
    detrending_type = flags.get('detrending_type', 'linear')
    interpolate_trend = flags.get('interpolate_trend', False)
    interpolate_ld = flags.get('interpolate_ld', False)
    fix_ld = flags.get('fix_ld', False)
    need_lowres = flags.get('need_lowres', True)
    mask_start = flags.get('mask_start', False)
    mask_end = flags.get('mask_end', False)
    save_trace = flags.get('save_whitelight_trace', False)
    # binning nm seperation
    high_resolution_bins = bins.get('high', 1)
    low_resolution_bins = bins.get('low', 100)

    # outlier clipping
    whitelight_sigma = outlier_clip.get('whitelight_sigma', 4)
    spectroscopic_sigma = outlier_clip.get('spectroscopic_sigma', 4)

    # plajet priors
    PERIOD_FIXED = planet_cfg['period']
    PRIOR_DUR = planet_cfg['duration']
    PRIOR_T0 = planet_cfg['t0']
    PRIOR_B = planet_cfg['b']
    PRIOR_RPRS = planet_cfg['rprs']
    PRIOR_DEPTH = PRIOR_RPRS ** 2

    # stellar parameters
    stellar_feh = stellar_cfg['feh']
    stellar_teff = stellar_cfg['teff']
    stellar_logg = stellar_cfg['logg']
    ld_model = stellar_cfg.get('ld_model', 'mps1')
    ld_data_path = stellar_cfg.get('ld_data_path', '../exotic_ld_data')

    sld = StellarLimbDarkening(
        M_H=stellar_feh, Teff=stellar_teff, logg=stellar_logg, ld_model=ld_model,
        ld_data_path=ld_data_path
    )
    mini_instrument = 'order'+str(order) if instrument == 'NIRISS/SOSS' else 'nrs'+str(nrs) if instrument == 'NIRSPEC/G395H' or instrument == 'NIRESPEC/G395M' else ''
    instrument_full_str = f"{planet_str}_{instrument.replace('/', '_')}_{mini_instrument}"
    spectro_data_file = output_dir + f'/{instrument_full_str}_spectroscopy_data_{low_resolution_bins}LR_{high_resolution_bins}HR.pkl'

    if not os.path.exists(spectro_data_file):
        data = process_spectroscopy_data(instrument, input_dir, output_dir, planet_str, cfg, fits_file, mask_start, mask_end)
        data.save(spectro_data_file)
        print("Shapes:")
        print(f"Time: {data.time.shape}")
        print(f"White light: {data.wl_time.shape}, {data.wl_flux.shape}")
        print(f"Low res: {data.wavelengths_lr.shape}, {data.flux_lr.shape}")
        print(f"High res: {data.wavelengths_hr.shape}, {data.flux_hr.shape}")
    else:
        data = SpectroData.load(spectro_data_file)
    print("Shapes:")
    print(f"Time: {data.time.shape}")
    print(f"White light: {data.wl_time.shape}, {data.wl_flux.shape}")
    print(f"Low res: {data.wavelengths_lr.shape}, {data.flux_lr.shape}")
    print(f"High res: {data.wavelengths_hr.shape}, {data.flux_hr.shape}, {data.flux_err_hr.shape}")


    COMPUTE_KERNELS = {
    'linear': compute_lc_linear,
    'explinear': compute_lc_explinear,
     'gp': compute_lc_gp_mean }

    stringcheck = os.path.exists(f'{output_dir}/{instrument_full_str}_whitelight_outlier_mask.npy')

    if instrument == 'NIRSPEC/G395H' or instrument == 'NIRSPEC/G395M' or instrument == 'MIRI/LRS':
        U_mu_wl = get_limb_darkening(sld, data.wavelengths_unbinned,0.0, instrument)
    elif instrument == 'NIRISS/SOSS':
        U_mu_wl = get_limb_darkening(sld, data.wavelengths_unbinned, 0.0, instrument, order=order)

    if not stringcheck or (detrending_type == 'gp'):
        if not os.path.exists(f'{output_dir}/{instrument_full_str}_whitelight_GP_database.csv'):
            plt.scatter(data.wl_time, data.wl_flux)
            plt.savefig(f'{output_dir}/00_{instrument_full_str}_whitelight_precheck.png')
            #keep_going = input('Whitelight precheck with guess T0 has been created, would you like to continue? (Enter to continue/N to exit)')
            #plt.close()
            #if keep_going.lower == 'n':
            #    exit()
            print('Fitting whitelight for outliers and bestfit parameters')
            prior_params_wl = {
                    "duration": PRIOR_DUR, "t0": PRIOR_T0,
                    "rors": jnp.sqrt(PRIOR_DEPTH), 'period': PERIOD_FIXED, '_b': PRIOR_B,
                    'u': U_mu_wl,
                 'logD': jnp.log(PRIOR_DUR), 'b': PRIOR_B, 'depths': PRIOR_DEPTH,
                 'c': 0.0, 'v': 0.0,
                }
    
            if detrending_type == 'explinear':
                prior_params_wl['A'] = 0.0
                prior_params_wl['tau'] = 0.5
            elif detrending_type == 'gp':
                prior_params_wl['logs2'] = jnp.log(2*jnp.nanmedian(data.wl_flux_err))
                prior_params_wl['GP_log_sigma'] = jnp.log(jnp.nanmedian(data.wl_flux_err))
                prior_params_wl['GP_log_rho'] = jnp.log(0.1)
    
            if detrending_type == 'gp':
                print("Setting platform to 'cpu' for GP whitelight fit.")
                numpyro.set_platform('cpu')
    
            whitelight_model_for_run = create_whitelight_model(detrend_type=detrending_type)
            #soln = optimx.optimize(whitelight_model, start=prior_params_wl)(key_master, data.wl_time, data.wl_flux_err, y=data.wl_flux, prior_params=prior_params_wl, detrend_type=detrending_type)
            soln =  optimx.optimize(whitelight_model_for_run, start=prior_params_wl)(key_master, data.wl_time, data.wl_flux_err, y=data.wl_flux, prior_params=prior_params_wl)
            
            mcmc = numpyro.infer.MCMC(
                numpyro.infer.NUTS(
                    whitelight_model_for_run,
                 #   partial(whitelight_model, detrend_type=detrending_type),
                    regularize_mass_matrix=False,
                    init_strategy=numpyro.infer.init_to_value(values=soln),
                    target_accept_prob=0.9,
                ),
                num_warmup=1000,
                num_samples=1000,
                progress_bar=True,
                jit_model_args=True
            )
            mcmc.run(key_master, data.wl_time, data.wl_flux_err, y=data.wl_flux, prior_params=prior_params_wl)
            inf_data = az.from_numpyro(mcmc)
            if save_trace:
                az.to_netcdf(inf_data, 'whitelight_trace.nc')
            wl_samples = mcmc.get_samples()
    
            if detrending_type == 'gp':
                print(f"Setting platform back to '{host_device}' for multi-wavelength fit.")
                numpyro.set_platform(host_device)
            print(az.summary(inf_data, var_names=None, round_to=7))
    
            bestfit_params_wl = {'duration': jnp.nanmedian(wl_samples['duration']), 't0': jnp.nanmedian(wl_samples['t0']),
                                'b': jnp.nanmedian(wl_samples['b']), 'rors': jnp.nanmedian(wl_samples['rors']),
                                'period': PERIOD_FIXED, 'u': jnp.nanmedian(wl_samples['u'], axis=0),
                                'c': jnp.nanmedian(wl_samples['c']),
                                 'v': jnp.nanmedian(wl_samples['v']) if detrending_type != 'gp' else 0.0,
                                }
            if detrending_type == 'explinear':
                bestfit_params_wl['A'] = jnp.nanmedian(wl_samples['A'])
                bestfit_params_wl['tau'] = jnp.nanmedian(wl_samples['tau'])
            elif detrending_type == 'gp':
                bestfit_params_wl['logs2'] = jnp.nanmedian(wl_samples['logs2'])
                bestfit_params_wl['GP_log_sigma'] = jnp.nanmedian(wl_samples['GP_log_sigma'])
                bestfit_params_wl['GP_log_rho'] = jnp.nanmedian(wl_samples['GP_log_rho'])
    
    
            #wl_transit_model = compute_lc_from_params(bestfit_params_wl, data.wl_time, detrending_type)
            if detrending_type == 'linear':
                wl_transit_model = compute_lc_linear(bestfit_params_wl, data.wl_time)
            if detrending_type == 'explinear':
                wl_transit_model = compute_lc_explinear(bestfit_params_wl, data.wl_time)
            if detrending_type == 'gp':
                wl_kernel = tinygp.kernels.quasisep.Matern32(
                    scale=jnp.exp(bestfit_params_wl['GP_log_rho']),
                    sigma=jnp.exp(bestfit_params_wl['GP_log_sigma']),
                )
                wl_gp = tinygp.GaussianProcess(
                    wl_kernel,
                    data.wl_time,
                    diag=jnp.exp(bestfit_params_wl['logs2']),
                   #mean=partial(compute_lc_from_params, bestfit_params_wl, detrend_type='gp'),
                    mean=partial(compute_lc_gp_mean, bestfit_params_wl),
                )
                cond_gp = wl_gp.condition(data.wl_flux, data.wl_time).gp
                mu, var = cond_gp.loc, cond_gp.variance
                wl_transit_model = mu
                #wl_residual = data.wl_flux - mu
            #else:
             #   wl_residual = data.wl_flux - wl_transit_model
            wl_residual = data.wl_flux - wl_transit_model
    
    
            wl_sigma = 1.4826 * jnp.nanmedian(np.abs(wl_residual - jnp.nanmedian(wl_residual)))
    
    
            wl_mad_mask = jnp.abs(wl_residual - jnp.nanmedian(wl_residual)) > whitelight_sigma * wl_sigma
    
            wl_sigma_post_clip = 1.4826 * jnp.nanmedian(jnp.abs(wl_residual[~wl_mad_mask] - jnp.nanmedian(wl_residual[~wl_mad_mask])))
    
    
            plt.plot(data.wl_time, wl_transit_model, color="r", lw=2)
            plt.scatter(data.wl_time, data.wl_flux, c='k', s=1)
    
           # plt.title('WL GP fit')
            plt.savefig(f"{output_dir}/11_{instrument_full_str}_whitelightmodel.png")
            plt.show()
            plt.close()
    
            plt.scatter(data.wl_time, wl_residual, c='k', s=2)
            plt.axhline(0, c='r', lw=2)
            plt.axhline(3*wl_sigma, c='b', lw=2, ls='--')
            plt.axhline(-3*wl_sigma, c='b', lw=2, ls='--')
            plt.axhline(4*wl_sigma, c='r', lw=2, ls='--')
            plt.axhline(-4*wl_sigma, c='r', lw=2, ls='--')
            plt.title('WL Pre-outlier rejection residual. Blue, Red Lines are +/- 3, 4 sigma')
            plt.savefig(f"{output_dir}/12_{instrument_full_str}_whitelightresidual.png")
            plt.show()
            plt.close()
    
    
    
            #plt.plot(data.wl_time, wl_transit_model, color="C0", lw=2)
            plt.scatter(data.wl_time, data.wl_flux, c='r', s=6)
            plt.scatter(data.wl_time[~wl_mad_mask], data.wl_flux[~wl_mad_mask], c='k', s=6)
            plt.title(f'Whitelight Outlier Rejection Plot (Removed = Red)')
            plt.savefig(f'{output_dir}/13_{instrument_full_str}_whitelightpostrejection.png')
            plt.show()
            plt.close()
    
            if detrending_type == 'linear':
                detrended_flux = data.wl_flux[~wl_mad_mask] / (1.0 + (bestfit_params_wl["c"] + bestfit_params_wl["v"] * (data.wl_time[~wl_mad_mask] - jnp.min(data.wl_time[~wl_mad_mask]))))
            if detrending_type == 'explinear': 
                detrended_flux = data.wl_flux[~wl_mad_mask] / (1.0 + (bestfit_params_wl["c"] + bestfit_params_wl["v"] * (data.wl_time[~wl_mad_mask] - jnp.min(data.wl_time[~wl_mad_mask])) 
                                                                + bestfit_params_wl['A'] * jnp.exp(-data.wl_time[~wl_mad_mask]/bestfit_params_wl['tau'])) )
            if detrending_type == 'gp':
                wl_kernel = tinygp.kernels.quasisep.Matern32(
                    scale=jnp.exp(bestfit_params_wl['GP_log_rho']),
                    sigma=jnp.exp(bestfit_params_wl['GP_log_sigma']),
                )
                wl_gp = tinygp.GaussianProcess(
                    wl_kernel,
                   data.wl_time[~wl_mad_mask],
                    diag=jnp.exp(bestfit_params_wl['logs2']),
                   # mean=partial(compute_lc_from_params, bestfit_params_wl, detrend_type='gp'),
                    mean=partial(compute_lc_gp_mean, bestfit_params_wl),
                )
                cond_gp = wl_gp.condition(data.wl_flux[~wl_mad_mask], data.wl_time[~wl_mad_mask]).gp
                mu, var = cond_gp.loc, cond_gp.variance
               # trend_flux = mu - compute_lc_from_params(bestfit_params_wl, data.wl_time[~wl_mad_mask], 'gp')
                trend_flux = mu - compute_lc_gp_mean(bestfit_params_wl, data.wl_time[~wl_mad_mask])
                detrended_flux = data.wl_flux[~wl_mad_mask] / (trend_flux + 1.0)
                planet_model_only = compute_lc_gp_mean(bestfit_params_wl, data.wl_time[~wl_mad_mask])
    
            plt.scatter(data.wl_time[~wl_mad_mask], detrended_flux, c='k', s=6)
            plt.title(f'Detrended WLC: Sigma {round(wl_sigma_post_clip*1e6)} PPM')
            plt.savefig(f'{output_dir}/14_{instrument_full_str}_whitelightdetrended.png')
            plt.show()
            plt.close()
            
            detrended_data = pd.DataFrame({'time': data.wl_time[~wl_mad_mask], 'flux': detrended_flux})
            detrended_data.to_csv(f'{output_dir}/{instrument_full_str}_whitelight_detrended.csv', index=False)
            np.save(f'{output_dir}/{instrument_full_str}_whitelight_outlier_mask.npy', arr=wl_mad_mask)
            if detrending_type == 'gp':
                df = pd.DataFrame({'wl_flux': data.wl_flux[~wl_mad_mask], 'gp_flux': mu, 'gp_err': jnp.sqrt(var), 'transit_model_flux': planet_model_only})
                df.to_csv(f'{output_dir}/{instrument_full_str}_whitelight_GP_database.csv')
            #create new save params wl variable with the bestift_params and their uncertainties
            bestfit_params_wl['duration_err'] = jnp.std(wl_samples['duration'], axis=0)
            bestfit_params_wl['t0_err'] = jnp.std(wl_samples['t0'], axis=0)
            bestfit_params_wl['b_err'] = jnp.std(wl_samples['b'], axis=0)
            bestfit_params_wl['rors_err'] = jnp.std(wl_samples['rors'], axis=0)
            bestfit_params_wl['depths_err'] = jnp.std(wl_samples['rors']**2, axis=0)
            #bestfit_params_wl['u_err'] = jnp.std(wl_samples['u'], axis=0)
            #bestfit_params_wl['c_err'] = jnp.std(wl_samples['c'], axis=0)
            #bestfit_params_wl['v_err'] = jnp.std(wl_samples['v'], axis=0)
            #if detrending_type == 'explinear':
            #    bestfit_params_wl['A_err'] = jnp.std(wl_samples['A'], axis=0)
            #    bestfit_params_wl['tau_err'] = jnp.std(wl_samples['tau'], axis=0)
            #elif detrending_type == 'gp':
            #    bestfit_params_wl['logs2_err'] = jnp.std(wl_samples['logs2'], axis=0)
            #    bestfit_params_wl['GP_log_sigma_err'] = jnp.std(wl_samples['GP_log_sigma'], axis=0)
            #    bestfit_params_wl['GP_log_rho_err'] = jnp.std(wl_samples['GP_log_rho'], axis=0)
    
    
            df = pd.DataFrame.from_dict(bestfit_params_wl, orient='index')
            df = df.transpose()
            df.to_csv(f'{output_dir}/{instrument_full_str}_whitelight_bestfit_params.csv')
            print(f'Saved whitelight parameters to {output_dir}/{instrument_full_str}_whitelight_bestfit_params.csv')
            bestfit_params_wl = pd.read_csv(f'{output_dir}/{instrument_full_str}_whitelight_bestfit_params.csv')
    
    
            DURATION_BASE = jnp.array(bestfit_params_wl['duration'][0])
            T0_BASE = jnp.array(bestfit_params_wl['t0'][0])
            B_BASE = jnp.array(bestfit_params_wl['b'][0])
            RORS_BASE = jnp.array(bestfit_params_wl['rors'][0])
            DEPTH_BASE = RORS_BASE**2
        else:
            print(f'GP trends already exist... If you want to refit GP on whitelight please remove {output_dir}/{instrument_full_str}_whitelight_GP_database.csv')
    else:
        print(f'Whitelight outliers and bestfit parameters already exist, skipping whitelight fit. If you want to fit whitelight please delete {output_dir}/{instrument_full_str}_whitelight_outlier_mask.npy')
        wl_mad_mask = np.load(f'{output_dir}/{instrument_full_str}_whitelight_outlier_mask.npy')
        bestfit_params_wl = pd.read_csv(f'{output_dir}/{instrument_full_str}_whitelight_bestfit_params.csv')
        DURATION_BASE = jnp.array(bestfit_params_wl['duration'][0])
        T0_BASE = jnp.array(bestfit_params_wl['t0'][0])
        B_BASE = jnp.array(bestfit_params_wl['b'][0])
        RORS_BASE = jnp.array(bestfit_params_wl['rors'][0])
        DEPTH_BASE = RORS_BASE**2

    key_lr, key_hr, key_map_lr, key_mcmc_lr, key_map_hr, key_mcmc_hr, key_prior_pred = jax.random.split(key_master, 7)

    need_lowres_analysis = interpolate_trend or interpolate_ld or need_lowres

    # init vars that might be set in low-res analysis
    trend_fixed_hr = None
    ld_fixed_hr = None
    ld_grid_hr = None
    best_poly_coeffs_c = None
    best_poly_coeffs_v = None
    best_poly_coeffs_u1 = None
    best_poly_coeffs_u2 = None
    best_poly_coeffs_A = None
    best_poly_coeffs_tau = None




    # --- Low-resolution Analysis ---
    if need_lowres_analysis:
        print(f"\n--- Running Low-Resolution Analysis (Binned to R{low_resolution_bins}) ---")

        ##### APPLY OUTLIER MASK HERE ####
        time_lr = jnp.array(data.time[~wl_mad_mask])
        flux_lr = jnp.array(data.flux_lr[:, ~wl_mad_mask])
        flux_err_lr = jnp.array(data.flux_err_lr[:, ~wl_mad_mask])
        num_lcs_lr = jnp.array(data.flux_err_lr.shape[0])

        if detrending_type == 'gp':
            gp_df = pd.read_csv(f'{output_dir}/{instrument_full_str}_whitelight_GP_database.csv')
            trend_flux = gp_df['gp_flux'].values - gp_df['transit_model_flux'].values 
            flux_lr = jnp.array(data.flux_lr[:,~wl_mad_mask]) / jnp.array(trend_flux + 1.0)
            temp_err = jnp.nanmedian(jnp.abs(0.5 * (flux_lr[:, :-2] + flux_lr[:, 2:]) - flux_lr[:, 1:-1]), axis=1, keepdims=True)
            flux_err_lr = jnp.repeat(temp_err, flux_lr.shape[1], axis=1)
            assert flux_lr.shape[1] == time_lr.shape[0]
            assert flux_err_lr.shape == flux_lr.shape
            detrend_type_multiwave = 'linear'
        else:
            detrend_type_multiwave = detrending_type


        assert time_lr.dtype == 'float64', "time_lr should be float64"
        assert flux_lr.dtype == 'float64', "indiv_y_lr should be float64"
        assert flux_err_lr.dtype == 'float64', "yerr_lr should be float64"
        assert len(time_lr.shape) == 1, "t_lr should be 1D array"
        assert len(flux_lr.shape) == 2, "indiv_y_lr should be 2D array (channels, time)"
        assert len(flux_err_lr.shape) == 2, "yerr_lr should be 2D array (channels, time)"
        assert time_lr.shape[0] == flux_lr.shape[1], "t_lr and indiv_y_lr should have same number of time points"
        assert time_lr.shape[0] == flux_err_lr.shape[1], "t_lr and yerr_lr should have same number of time points"

        print(f"Low-res: {num_lcs_lr} light curves.")

        DEPTHS_BASE_LR = jnp.full(num_lcs_lr, DEPTH_BASE)

        if instrument == 'NIRSPEC/G395H' or instrument == 'NIRSPEC/G395M' or instrument == 'MIRI/LRS':
            U_mu_lr = get_limb_darkening(sld, data.wavelengths_lr, data.wavelengths_err_lr , instrument)
        elif instrument == 'NIRISS/SOSS':
            U_mu_lr = get_limb_darkening(sld, data.wavelengths_lr, data.wavelengths_err_lr, instrument, order=order)



        if fix_ld is True:
            ld_fixed_lr = U_mu_lr
            print("Using fixed limb darkening coefficients for low-resolution analysis.")
        else:
            ld_fixed_lr = None

        init_params_lr = {
            "logD": jnp.log(bestfit_params_wl['duration'][0]), "t0": bestfit_params_wl['t0'][0], "_b": bestfit_params_wl['b'][0],
            "u": U_mu_lr, "depths": jnp.full(num_lcs_lr, bestfit_params_wl['rors'][0]**2),
            'c': jnp.full(num_lcs_lr, bestfit_params_wl['c'][0]), 'v': jnp.full(num_lcs_lr, bestfit_params_wl['v'][0]),
            }
        if detrend_type_multiwave == 'explinear':
            init_params_lr['A'] = jnp.full(num_lcs_lr, bestfit_params_wl['A'][0])
            init_params_lr['tau'] = jnp.full(num_lcs_lr, bestfit_params_wl['tau'][0])

        print("Sampling low-res model using MCMC to find median coefficients...")
        #samples_lr = get_samples(
        #    partial(vectorized_model, mu_duration=DURATION_BASE, mu_t0=T0_BASE, mu_depths=DEPTHS_BASE_LR, PERIOD=PERIOD_FIXED, detrend_type=detrend_type_multiwave),
        #    key_mcmc_lr, time_lr, flux_err_lr, flux_lr, init_params_lr,)

        lr_trend_mode = 'free' 
        
        lr_ld_mode = 'free'
        if flags.get('fix_ld', False):
            lr_ld_mode = 'fixed'
    
        
        lr_model_for_run = create_vectorized_model(
            detrend_type=detrend_type_multiwave, 
            ld_mode=lr_ld_mode,
            trend_mode=lr_trend_mode
        )
        
        
        model_run_args_lr = {
        'mu_duration': DURATION_BASE,
        'mu_t0': T0_BASE,
        'mu_depths': DEPTHS_BASE_LR,
        'PERIOD': PERIOD_FIXED,
        }
        if lr_ld_mode == 'fixed':
            model_run_args_lr['ld_fixed'] = U_mu_lr
            
        
        samples_lr = get_samples(
            model=lr_model_for_run,
            key=key_mcmc_lr,       
            t=time_lr,                 
            yerr=flux_err_lr,          
            indiv_y=flux_lr,          
            init_params=init_params_lr,
            **model_run_args_lr
        )      
        
        ld_u_lr = np.array(samples_lr["u"])
        trend_c_lr = np.array(samples_lr["c"])
        trend_v_lr = np.array(samples_lr["v"])
        if detrend_type_multiwave == 'explinear':
            trend_A_lr = np.array(samples_lr["A"])
            trend_tau_lr = np.array(samples_lr["tau"])

        map_params_lr = {
            "duration": jnp.nanmedian(samples_lr["duration"]),
            "t0": jnp.nanmedian(samples_lr["t0"]),
            "b": jnp.nanmedian(samples_lr["b"]),
            "rors": jnp.nanmedian(samples_lr["rors"], axis=0),
            "u": jnp.nanmedian(ld_u_lr, axis=0),  "period": PERIOD_FIXED,
            'c': jnp.nanmedian(samples_lr["c"], axis=0),
            'v': jnp.nanmedian(samples_lr["v"], axis=0),
        }
        if detrend_type_multiwave == 'explinear':
            map_params_lr['A'] = jnp.nanmedian(samples_lr['A'], axis=0)
            map_params_lr['tau'] = jnp.nanmedian(samples_lr['tau'], axis=0)


        
        try:
            selected_kernel = COMPUTE_KERNELS[detrend_type_multiwave]
        except KeyError:
            raise ValueError(f"Unknown detrend_type: {detrend_type_multiwave}")
        
        in_axes_map = {
            'rors': 0, 
            'u': 0, 
            'c': 0, 
            'v': 0
        }
        if detrend_type_multiwave == 'explinear':
            in_axes_map.update({'A': 0, 'tau': 0})
        
        final_in_axes = {k: in_axes_map.get(k, None) for k in map_params_lr.keys()}
            
        model_all = jax.vmap(selected_kernel, in_axes=(final_in_axes, None))(map_params_lr, time_lr)
        
        residuals = flux_lr - model_all

        medians = np.nanmedian(residuals, axis=1, keepdims=True)
        sigmas    = 1.4826 * np.nanmedian(np.abs(residuals - medians), axis=1, keepdims=True)

        point_mask = np.abs(residuals - medians) > spectroscopic_sigma * sigmas

        time_mask = np.any(point_mask, axis=0)

        n_channels, n_times = flux_lr.shape

        if n_channels > 0:
            nrows = int(np.ceil(np.sqrt(n_channels)))
            ncols = int(np.ceil(n_channels / nrows))
        else:
            nrows, ncols = 1, 1

        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.5, nrows * 3), squeeze=False)

        axes = axes.flatten()

        plot_count = 0

        for ch in range(n_channels):
            outlier_mask = point_mask[ch]

            if outlier_mask.any():
                ax = axes[plot_count]
                ax.scatter(time_lr, flux_lr[ch, :], c="k", label="data", s=10)

                ax.scatter(time_lr[outlier_mask], flux_lr[ch, outlier_mask],
                            c="r", label="outliers", s=10)

                plot_count += 1 #
        for i in range(plot_count, nrows * ncols):
            axes[i].axis('off')

        if plot_count > 0:
            plt.tight_layout()
            plt.savefig(f"{output_dir}/21_{instrument_full_str}_lowresoutliers.png")
        else:
            print("No channels with outliers found to plot.")

        plt.close(fig)

        valid = ~time_mask
        time_lr       = time_lr[valid]
        flux_lr = flux_lr[:, valid]
        flux_err_lr    = flux_err_lr[:, valid]

        print("Plotting low-resolution fits and residuals...")
        plot_map_fits(time_lr, flux_lr, flux_err_lr, data.wavelengths_lr, map_params_lr,
                    {"period": PERIOD_FIXED},
                    f"{output_dir}/22_{instrument_full_str}_R{low_resolution_bins}_bestfit.png", ncols=5, detrend_type=detrend_type_multiwave)
        plot_map_residuals(time_lr, flux_lr, flux_err_lr, data.wavelengths_lr, map_params_lr,
                        {"period": PERIOD_FIXED},
                        f"{output_dir}/23_{instrument_full_str}_R{low_resolution_bins}_residual.png", ncols=5, detrend_type=detrend_type_multiwave)

        # Polynomial Fitting for Interpolation
        poly_orders = [1, 2, 3, 4]
        wl_lr = np.array(data.wavelengths_lr)

        print("Fitting polynomials to trend coefficients...")
        best_poly_coeffs_c, best_order_c, _ = fit_polynomial(wl_lr, trend_c_lr, poly_orders)
        best_poly_coeffs_v, best_order_v, _ = fit_polynomial(wl_lr, trend_v_lr, poly_orders)
        print(f"Selected polynomial degrees: c={best_order_c}, v={best_order_v}")
        if detrend_type_multiwave == 'explinear':
            best_poly_coeffs_A, best_order_A, _ = fit_polynomial(wl_lr, trend_A_lr, poly_orders)
            best_poly_coeffs_tau, best_order_tau, _ = fit_polynomial(wl_lr, trend_tau_lr, poly_orders)
            print(f"Selected polynomial degrees: A={best_order_A}, tau={best_order_tau}")

        plot_poly_fit(wl_lr, trend_c_lr, best_poly_coeffs_c, best_order_c,
                        "Wavelength (μm)", "Trend coefficient c", "Trend Offset (c) Polynomial Fit",
                        f"{output_dir}/2optional1_{instrument_full_str}_R{low_resolution_bins}_cinterp.png")
        plot_poly_fit(wl_lr, trend_v_lr, best_poly_coeffs_v, best_order_v,
                        "Wavelength (μm)", "Trend coefficient v", "Trend Slope (v) Polynomial Fit",
                        f"{output_dir}/2optional2_{instrument_full_str}_R{low_resolution_bins}_vinterp.png")

        if interpolate_ld:
            print("Fitting polynomials to limb darkening coefficients...")
            best_poly_coeffs_u1, best_order_u1, _ = fit_polynomial(wl_lr, ld_u_lr[:, :, 0], poly_orders)
            best_poly_coeffs_u2, best_order_u2, _ = fit_polynomial(wl_lr, ld_u_lr[:, :, 1], poly_orders)
            print(f"Selected polynomial degrees: u1={best_order_u1}, u2={best_order_u2}")
            plot_poly_fit(wl_lr, ld_u_lr[:,:, 0], best_poly_coeffs_u1, best_order_u1,
                        "Wavelength (μm)", "LD coefficient u1", "Limb Darkening u1 Polynomial Fit",
                        f"{output_dir}/2optional3_{instrument_full_str}_R{low_resolution_bins}_u1interp.png")
            plot_poly_fit(wl_lr, ld_u_lr[:,:, 1], best_poly_coeffs_u2, best_order_u2,
                        "Wavelength (μm)", "LD coefficient u2", "Limb Darkening u2 Polynomial Fit",
                        f"{output_dir}/2optional4_{instrument_full_str}_R{low_resolution_bins}_u2interp.png")


        print("Plotting and saving lowres transmission spectrum...")
        plot_transmission_spectrum(wl_lr, samples_lr["rors"],
                            f"{output_dir}/24_{instrument_full_str}_R{low_resolution_bins}_spectrum.png")
        save_results(wl_lr, samples_lr, f"{output_dir}/{instrument_full_str}_R{low_resolution_bins}.csv")

        oot_mask_lr = (time_lr < T0_BASE - 0.6 * DURATION_BASE) | (time_lr > T0_BASE + 0.6 * DURATION_BASE)

        def calc_rms(y_bin):
            baseline = y_bin[oot_mask_lr]
            return jnp.nanmedian(jnp.abs(baseline - jnp.nanmedian(baseline))) * 1.4826

        rms_vals = jax.vmap(calc_rms)(flux_lr)

        plt.figure(figsize=(8,5))
        plt.scatter(data.wavelengths_lr, rms_vals, c='k')
        plt.xlabel("Wavelength (μm)")
        plt.ylabel("Per Wavelength Noise (ppm)")
       # plt.title("Out‑of‑Transit RMS vs Wavelength")
        plt.tight_layout()
        plt.savefig(f'{output_dir}/24_{instrument_full_str}_R{low_resolution_bins}_rms.png')
        plt.close()
    # --- High-resolution Analysis ---
    print(f"\n--- Running High-Resolution Analysis (Binned to R{high_resolution_bins}) ---")

    ##### APPLY OUTLIER MASK HERE ####
    time_hr = jnp.array(data.time[~wl_mad_mask])
    flux_hr = jnp.array(data.flux_hr[:, ~wl_mad_mask])
    flux_err_hr = jnp.array(data.flux_err_hr[:, ~wl_mad_mask])

    if detrending_type == 'gp':
        gp_df = pd.read_csv(f'{output_dir}/{instrument_full_str}_whitelight_GP_database.csv')
        trend_flux = gp_df['gp_flux'].values - gp_df['transit_model_flux'].values
        flux_hr = jnp.array(data.flux_hr[:,~wl_mad_mask]) / jnp.array(trend_flux + 1.0)
        temp_err = jnp.nanmedian(jnp.abs(0.5 * (flux_hr[:, :-2] + flux_hr[:, 2:]) - flux_hr[:, 1:-1]), axis=1, keepdims=True)
        flux_err_hr = jnp.repeat(temp_err, flux_hr.shape[1], axis=1)
        assert flux_hr.shape[1] == time_hr.shape[0]
        assert flux_err_hr.shape == flux_hr.shape
    else:
        detrend_type_multiwave = detrending_type


    if need_lowres_analysis:
        valid = ~time_mask
        time_hr       = time_hr[valid]
        flux_hr = flux_hr[:, valid]
        flux_err_hr    = flux_err_hr[:, valid]

    assert time_hr.dtype == 'float64', "t_hr should be float64"
    assert flux_hr.dtype == 'float64', "indiv_y_hr should be float64"
    assert flux_err_hr.dtype == 'float64', "yerr_hr should be float64"
    assert len(time_hr.shape) == 1, "t_hr should be 1D array"
    assert len(flux_hr.shape) == 2, "indiv_y_hr should be 2D array (channels, time)"
    assert len(flux_err_hr.shape) == 2, "yerr_hr should be 2D array (channels, time)"
    assert time_hr.shape[0] == flux_hr.shape[1], "t_hr and indiv_y_hr should have same number of time points"
    assert time_hr.shape[0] == flux_err_hr.shape[1], "t_hr and yerr_hr should have same number of time points"

    num_lcs_hr = flux_err_hr.shape[0]
    print(f"High-res: {num_lcs_hr} light curves.")

    DEPTHS_BASE_HR = jnp.full(num_lcs_hr, DEPTH_BASE)
    hr_ld_mode = 'free'
    if flags.get('interpolate_ld', False):
        hr_ld_mode = 'interpolated'
    elif flags.get('fix_ld', False):
        hr_ld_mode = 'fixed'
    
    hr_trend_mode = 'free'
    if flags.get('interpolate_trend', False):
        hr_trend_mode = 'fixed'
    
    model_run_args_hr = {}
    wl_hr = np.array(data.wavelengths_hr)
    
    if hr_ld_mode == 'interpolated':
        u1_interp_hr = np.polyval(best_poly_coeffs_u1, wl_hr)
        u2_interp_hr = np.polyval(best_poly_coeffs_u2, wl_hr)
        ld_interpolated_hr = jnp.array(np.column_stack((u1_interp_hr, u2_interp_hr)))
        model_run_args_hr['ld_interpolated'] = ld_interpolated_hr
        U_mu_hr_init = ld_interpolated_hr 
        print("HR Run Config: Using INTERPOLATED limb darkening.")
    
    elif hr_ld_mode == 'fixed':
        if instrument == 'NIRSPEC/G395H' or instrument == 'NIRSPEC/G395M' or instrument == 'MIRI/LRS':
            U_mu_hr = get_limb_darkening(sld, wl_hr, data.wavelengths_err_hr, instrument)
        elif instrument == 'NIRISS/SOSS':
            U_mu_hr = get_limb_darkening(sld, wl_hr, data.wavelengths_err_hr, instrument, order=order)
        model_run_args_hr['ld_fixed'] = U_mu_hr
        U_mu_hr_init = U_mu_hr 
        print("HR Run Config: Using FIXED limb darkening.")
    
    else: # hr_ld_mode == 'free'
        if instrument == 'NIRSPEC/G395H' or instrument == 'NIRSPEC/G395M' or instrument == 'MIRI/LRS':
            U_mu_hr_init = get_limb_darkening(sld, wl_hr, data.wavelengths_err_hr, instrument)
        elif instrument == 'NIRISS/SOSS':
            U_mu_hr_init = get_limb_darkening(sld, wl_hr, data.wavelengths_err_hr, instrument, order=order)
        print("HR Run Config: FITTING for limb darkening (free).")
    
    if hr_trend_mode == 'fixed':
        c_interp_hr = np.polyval(best_poly_coeffs_c, wl_hr)
        v_interp_hr = np.polyval(best_poly_coeffs_v, wl_hr)
        trend_fixed_hr = np.column_stack((c_interp_hr, v_interp_hr))
        if detrend_type_multiwave == 'explinear':
            A_interp_hr = np.polyval(best_poly_coeffs_A, wl_hr)
            tau_interp_hr = np.polyval(best_poly_coeffs_tau, wl_hr)
            trend_fixed_hr = np.column_stack((c_interp_hr, v_interp_hr, A_interp_hr, tau_interp_hr))
        
        model_run_args_hr['trend_fixed'] = jnp.array(trend_fixed_hr)
        print("HR Run Config: Using FIXED (interpolated) trend.")
    else: # hr_trend_mode == 'free'
        print("HR Run Config: FITTING for trend (free).")
    
    model_run_args_hr['mu_duration'] = DURATION_BASE
    model_run_args_hr['mu_t0'] = T0_BASE
    model_run_args_hr['mu_depths'] = DEPTHS_BASE_HR
    model_run_args_hr['PERIOD'] = PERIOD_FIXED
    
    init_params_hr = {
        "logD": jnp.log(DURATION_BASE), "t0": T0_BASE, "_b": B_BASE,
        "depths": DEPTHS_BASE_HR,
        "u": U_mu_hr_init,
    }
    if hr_trend_mode == 'free':
        init_params_hr["c"] = np.polyval(best_poly_coeffs_c, wl_hr)
        init_params_hr["v"] = np.polyval(best_poly_coeffs_v, wl_hr)
        if detrend_type_multiwave == 'explinear':
            init_params_hr["A"] = np.polyval(best_poly_coeffs_A, wl_hr)
            init_params_hr["tau"] = np.polyval(best_poly_coeffs_tau, wl_hr)
        
    hr_model_for_run = create_vectorized_model(
        detrend_type=detrend_type_multiwave,
        ld_mode=hr_ld_mode,
        trend_mode=hr_trend_mode
    )
    
    samples_hr = get_samples(
        model=hr_model_for_run,
        key=key_mcmc_hr,
        t=time_hr,
        yerr=flux_err_hr,
        indiv_y=flux_hr,
        init_params=init_params_hr,
        **model_run_args_hr 
    )

    print("Plotting and saving final transmission spectrum...")
    plot_transmission_spectrum(wl_hr, samples_hr["rors"],
                            f"{output_dir}/31_{instrument_full_str}_R{high_resolution_bins}_spectrum.png")
    save_results(wl_hr, samples_hr,  f"{output_dir}/{instrument_full_str}_R{high_resolution_bins}.csv")

    if "u" in samples_hr:
        u1_16, u1_median, u1_84 = np.percentile(np.array(samples_hr["u"][:,:,0]), [16, 50, 84], axis=0) # Shape: (n_samples, n_lcs, 2)
        u1_err_low = u1_median - u1_16
        u1_err_high = u1_84 - u1_median
        u1_yerr = np.array([u1_err_low, u1_err_high])

        u2_16, u2_median, u2_84 = np.percentile(np.array(samples_hr["u"][:,:,1]), [16, 50, 84], axis=0)
        u2_err_low = u2_median - u2_16
        u2_err_high = u2_84 - u2_median
        u2_yerr = np.array([u2_err_low, u2_err_high])

        plt.figure(figsize=(10, 6))
        plt.errorbar(wl_hr, u1_median, yerr=u1_yerr, fmt='o', markersize=4,
                    capsize=3, elinewidth=1, markeredgewidth=1, mfc='w',mec='k', label='u1 Median ± 1σ')
        plt.xlabel('Wavelength (micron)')
        plt.ylabel('u1')
        plt.tight_layout()
        u1_save_path = f'{output_dir}/3optional1_{instrument_full_str}_R{high_resolution_bins}_u1.png'
        plt.savefig(u1_save_path)
        print(f"Saved u1 plot with uncertainties to {u1_save_path}")
        plt.close()

        plt.figure(figsize=(10, 6))
        plt.errorbar(wl_hr, u2_median, yerr=u2_yerr, fmt='o', markersize=4,
                    capsize=3, elinewidth=1, markeredgewidth=1, mfc='w',mec='k', label='u2 Median ± 1σ')
        plt.xlabel('Wavelength (micron)')
        plt.ylabel('u2')
        plt.tight_layout()
        u2_save_path = f'{output_dir}/3optional2_{instrument_full_str}_R{high_resolution_bins}_u2.png'
        plt.savefig(u2_save_path)
        print(f"Saved u2 plot with uncertainties to {u2_save_path}")
        plt.close()

    else:
        print("LD coefficients were fixed—skipping u₁–u₂ plots.")

    oot_mask = (time_hr < T0_BASE - 0.6 * DURATION_BASE) | (time_hr > T0_BASE + 0.6 * DURATION_BASE)

    def calc_rms(y_bin):
        baseline = y_bin[oot_mask]
        return jnp.nanmedian(jnp.abs(baseline - jnp.nanmedian(baseline))) * 1.4826

    rms_vals = jax.vmap(calc_rms)(flux_hr)

    plt.figure(figsize=(8,5))
    plt.scatter(wl_hr, rms_vals*1e6, c='k')
    plt.xlabel("Wavelength (μm)")
    plt.ylabel("Per-Wavelength Noise (ppm)")
   # plt.title("Out‑of‑Transit RMS vs Wavelength")
    plt.tight_layout()
    plt.savefig(f'{output_dir}/32_{instrument_full_str}_R{high_resolution_bins}_rms.png')
    plt.close()

    #depth_hr = np.nanmedian(samples_hr["depths"], axis=0) 
    #depth_chain = np.array(samples_hr["depths"])       
    #cov_hr = np.cov(depth_chain, rowvar=False)     
    #np.savez(
    #f"{output_dir}/spectrum_native_cov.npz",
    #wavelength=wl_hr,
    #depth=depth_hr,
    #covariance=cov_hr
    #)  


    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
