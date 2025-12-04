import os
import sys
import yaml
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import numpyro_ext.optim as optimx
import arviz as az
from scipy.stats import norm
from functools import partial
import tinygp

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from exotic_ld import StellarLimbDarkening
from createdatacube import SpectroData, process_spectroscopy_data
from models.builder import create_whitelight_model
from models.core import compute_transit_model, _to_f64, _tree_to_f64
from models.trends import (
    spot_crossing, compute_lc_linear, compute_lc_quadratic, compute_lc_cubic,
    compute_lc_quartic, compute_lc_linear_discontinuity, compute_lc_explinear,
    compute_lc_spot, compute_lc_none
)
from models.gp import (
    compute_lc_gp_mean, compute_lc_linear_gp_mean, compute_lc_quadratic_gp_mean,
    compute_lc_explinear_gp_mean
)

# Set JAX to 64-bit
jax.config.update("jax_enable_x64", True)
plt.rcParams['axes.linewidth'] = 1.7

# -------------------------------------------------------------------
# Utility Functions (Copied/Adapted for standalone test script)
# -------------------------------------------------------------------

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def jax_bin_lightcurve(time, flux, duration, points_per_transit=20):
    dt = duration / points_per_transit
    t_min = jnp.min(time)
    t_max = jnp.max(time)
    total_range = t_max - t_min
    num_bins = jnp.ceil(total_range / dt).astype(int) + 1
    bin_indices = jnp.clip(((time - t_min) / dt).astype(int), 0, num_bins - 1)
    flux_sums = jnp.zeros(num_bins)
    time_sums = jnp.zeros(num_bins)
    counts = jnp.zeros(num_bins)
    flux_sums = flux_sums.at[bin_indices].add(flux)
    time_sums = time_sums.at[bin_indices].add(time)
    counts = counts.at[bin_indices].add(1.0)
    binned_flux = jnp.where(counts > 0, flux_sums / counts, jnp.nan)
    binned_time = jnp.where(counts > 0, time_sums / counts, jnp.nan)
    return binned_time, binned_flux

def get_robust_sigma(x):
    mad = np.median(np.abs(x - np.median(x)))
    return 1.4826 * mad

def calculate_beta_metrics(residuals, dt, cut_factor=5.0):
    residuals = np.array(residuals)
    ndata = len(residuals)
    sigma1 = get_robust_sigma(residuals)
    cadence_min = dt / 60.0
    max_bin_n = ndata // int(cut_factor) 
    bin_sizes_points = np.unique(np.logspace(0, np.log10(max_bin_n), 300).astype(int))
    
    measured_rms = []
    expected_rms = []
    bin_sizes_min = []
    betas = []
    
    for N in bin_sizes_points:
        cutoff = ndata - (ndata % N)
        binned_res = residuals[:cutoff].reshape(-1, N).mean(axis=1)
        sigma_N_measured = get_robust_sigma(binned_res)
        sigma_N_theory = sigma1 / np.sqrt(N)
        
        measured_rms.append(sigma_N_measured)
        expected_rms.append(sigma_N_theory)
        bin_sizes_min.append(N * cadence_min)
        betas.append(sigma_N_measured / sigma_N_theory)

    beta_final = np.median(betas)
    return beta_final, np.array(bin_sizes_min), np.array(measured_rms), np.array(expected_rms)

def run_beta_monte_carlo(residuals, dt, n_sims=500):
    ndata = len(residuals)
    sigma1 = get_robust_sigma(residuals)
    mc_betas = []
    _, ref_bin_sizes, _, _ = calculate_beta_metrics(residuals, dt)
    all_sim_rms = np.zeros((n_sims, len(ref_bin_sizes)))
    
    for i in range(n_sims):
        synth_res = np.random.normal(0, sigma1, ndata)
        b_sim, _, rms_sim, _ = calculate_beta_metrics(synth_res, dt)
        mc_betas.append(b_sim)
        if len(rms_sim) == len(ref_bin_sizes):
            all_sim_rms[i, :] = rms_sim
            
    rms_low_1sig = np.percentile(all_sim_rms, 16, axis=0)
    rms_high_1sig = np.percentile(all_sim_rms, 84, axis=0)
    rms_low_2sig = np.percentile(all_sim_rms, 5, axis=0)
    rms_high_2sig = np.percentile(all_sim_rms, 95, axis=0)

    return np.array(mc_betas), rms_low_1sig, rms_high_1sig, rms_low_2sig, rms_high_2sig

def compute_aic(n, residuals, k):
    rss = np.sum(np.square(residuals))
    rss = rss if rss > 1e-10 else 1e-10
    aic = 2*k + n * np.log(rss/n)
    return aic

def compute_bic(n, residuals, k):
    rss = np.sum(np.square(residuals))
    rss = rss if rss > 1e-10 else 1e-10
    bic = k * np.log(n) + n * np.log(rss/n)
    return bic

def count_parameters(detrending_type, n_planets=1):
    # Base: duration, t0, b, depth (4 per planet) + u1, u2 + jitter (3)
    # Actually u depends on profile, but assuming 2 params for now or handled
    # The count should match what is sampled.
    n_params = 4 * n_planets + 3
    if detrending_type == 'none': pass
    elif detrending_type == 'linear': n_params += 2
    elif detrending_type == 'quadratic': n_params += 3
    elif detrending_type == 'cubic': n_params += 4
    elif detrending_type == 'quartic': n_params += 5
    elif detrending_type == 'explinear': n_params += 4
    elif detrending_type == 'linear_discontinuity': n_params += 4
    elif detrending_type == 'spot': n_params += 5
    elif detrending_type == 'gp': n_params += 3 # c + 2 GP
    elif detrending_type == 'linear+gp': n_params += 4
    elif detrending_type == 'quadratic+gp': n_params += 5
    elif detrending_type == 'explinear+gp': n_params += 6
    return n_params

def get_limb_darkening(sld, wavelengths, wavelength_err, instrument, order=None, ld_profile='quadratic'):
    # ... (Same logic as fit_jwst.py, keeping it self-contained)
    if instrument == 'NIRSPEC/G395H':
        mode = "JWST_NIRSpec_G395H"
        wl_min, wl_max = 28700.0, 51700.0
    elif instrument == 'NIRSPEC/G235H':
        mode = "JWST_NIRSpec_G235H"
        wl_min, wl_max = 17000.0, 30600.0
    elif instrument == 'NIRSPEC/G140H':
        mode = "JWST_NIRSpec_G140H-f100"
        wl_min, wl_max = 10000.0, 18000.0
    elif instrument == 'NIRSPEC/G395M':
        mode = "JWST_NIRSpec_G395M"
        wl_min, wl_max = 28700.0, 51700.0
    elif instrument == 'NIRSPEC/PRISM':
        mode = "JWST_NIRSpec_Prism"
        wl_min, wl_max = 5000.0, 55000.0
    elif instrument == 'NIRISS/SOSS':
        mode = f"JWST_NIRISS_SOSSo{order}"
        wl_min, wl_max = 8300.0, 28100.0
    elif instrument == 'MIRI/LRS':
        mode = f'JWST_MIRI_LRS'
        wl_min, wl_max = 50000.0, 120000.0

    # Simplified single-bin logic for whitelight
    wl_range_clipped = [max(min(wavelengths)*1e4, wl_min), min(max(wavelengths)*1e4, wl_max)]
    if ld_profile == 'quadratic':
        U_mu = sld.compute_quadratic_ld_coeffs(wavelength_range=wl_range_clipped, mode=mode, return_sigmas=False)
    elif ld_profile == 'power2':
        U_mu = sld.compute_power2_ld_coeffs(wavelength_range=wl_range_clipped, mode=mode, return_sigmas=False)
    else:
        raise ValueError(f"Unknown ld_profile: {ld_profile}")
    return jnp.array(U_mu)

# ---------------------
# Main
# ---------------------
def main():
    parser = argparse.ArgumentParser(description="Run comparison of whitelight trend models.")
    parser.add_argument("-c", "--config", required=True, help="Path to YAML configuration file")
    args = parser.parse_args()

    cfg = load_config(args.config)
    instrument = cfg['instrument']
    
    # ... (Data loading logic similar to fit_jwst.py) ...
    planet_cfg = cfg['planet']
    stellar_cfg = cfg['stellar']
    flags = cfg.get('flags', {})
    
    # Setup Paths
    planet_str = planet_cfg['name']
    base_path = cfg.get('path', '.')
    input_dir = os.path.join(base_path, cfg.get('input_dir', planet_str + '_NIRSPEC'))
    output_dir = os.path.join(base_path, cfg.get('output_dir', planet_str + '_RESULTS'))
    fits_file = os.path.join(input_dir, cfg.get('fits_file'))
    if not os.path.exists(output_dir): os.makedirs(output_dir, exist_ok=True)

    # Setup Device
    host_device = cfg.get('host_device', 'gpu').lower()
    numpyro.set_platform(host_device)
    key_master = jax.random.PRNGKey(555)

    # Configuration
    ld_profile = flags.get('ld_profile', 'power2')
    save_trace = flags.get('save_whitelight_trace', False)

    # Load Data
    spectro_data_file = glob.glob(output_dir + '/*spectroscopy_data*.pkl')
    if not spectro_data_file:
        # Fallback to creating it if not found (simplified)
        print("Data pickle not found, please run fit_jwst.py once to generate it or check paths.")
        # Attempt to run process if possible?
        # For safety in this script, better to require the data or call process
        # But process_spectroscopy_data is imported
        data = process_spectroscopy_data(instrument, input_dir, output_dir, planet_str, cfg, fits_file,
                                         flags.get('mask_start'), flags.get('mask_end'))
    else:
        data = SpectroData.load(spectro_data_file[0]) # Load the first one found

    print("Data loaded.")

    # Setup Stellar LD
    sld = StellarLimbDarkening(
        M_H=stellar_cfg['feh'], Teff=stellar_cfg['teff'], logg=stellar_cfg['logg'],
        ld_model=stellar_cfg.get('ld_model', 'mps1'),
        ld_data_path=stellar_cfg.get('ld_data_path', '../exotic_ld_data')
    )

    if instrument == 'NIRISS/SOSS': order = cfg['order']
    else: order = None

    U_mu_wl = get_limb_darkening(sld, data.wavelengths_unbinned, 0.0, instrument, order=order, ld_profile=ld_profile)

    # Priors
    periods = jnp.atleast_1d(planet_cfg['period'])
    durations = jnp.atleast_1d(planet_cfg['duration'])
    t0s = jnp.atleast_1d(planet_cfg['t0'])
    bs = jnp.atleast_1d(planet_cfg['b'])
    rors = jnp.atleast_1d(planet_cfg['rprs'])
    depths = rors**2
    n_planets = len(periods)

    # Define trends to compare
    detrending_types_to_test = ['linear', 'quadratic', 'explinear', 'gp', 'linear+gp', 'explinear+gp']
    all_results = {}

    for detrending_type in detrending_types_to_test:
        print(f"\n{'='*60}")
        print(f"FITTING: {detrending_type} (LD: {ld_profile})")
        print(f"{'='*60}\n")

        # Initial Parameters
        init_params_wl = {
            'c': 1.0, 'v': 0.0, 'log_jitter': jnp.log(1e-4),
            'b': bs, 'rors': rors
        }
        if ld_profile == 'quadratic': init_params_wl['u'] = U_mu_wl

        for i in range(n_planets):
            init_params_wl[f'logD_{i}'] = jnp.log(durations[i])
            init_params_wl[f't0_{i}'] = t0s[i]
            init_params_wl[f'_b_{i}'] = bs[i]
            init_params_wl[f'depths_{i}'] = depths[i]

        # Init trend params
        if 'quadratic' in detrending_type: init_params_wl['v2'] = 0.0
        if 'explinear' in detrending_type: init_params_wl['A'] = 0.001; init_params_wl['tau'] = 0.5
        if 'gp' in detrending_type:
            init_params_wl['GP_log_sigma'] = jnp.log(jnp.nanmedian(data.wl_flux_err))
            init_params_wl['GP_log_rho'] = jnp.log(1.0)

        # Build Model
        hyper_params_wl = {"duration": durations, "t0": t0s, 'period': periods, 'u': U_mu_wl}
        if 'spot' in detrending_type: hyper_params_wl['spot_guess'] = flags.get('spot_center', 0.0)

        whitelight_model_for_run = create_whitelight_model(detrend_type=detrending_type, n_planets=n_planets, ld_profile=ld_profile)

        # Run Fit
        try:
            # Pre-fit for GP
            if 'gp' in detrending_type:
                model_pre = create_whitelight_model('linear', n_planets, ld_profile)
                init_pre = {k:v for k,v in init_params_wl.items() if 'GP' not in k}
                soln_pre = optimx.optimize(model_pre, start=init_pre)(key_master, data.wl_time, data.wl_flux_err, y=data.wl_flux, prior_params=hyper_params_wl)
                for k in soln_pre:
                    if k in init_params_wl: init_params_wl[k] = soln_pre[k]
                init_params_wl['GP_log_sigma'] = jnp.log(5.0 * jnp.nanmedian(data.wl_flux_err))
                init_params_wl['GP_log_rho'] = jnp.log(0.1)

            soln = optimx.optimize(whitelight_model_for_run, start=init_params_wl)(key_master, data.wl_time, data.wl_flux_err, y=data.wl_flux, prior_params=hyper_params_wl)

            mcmc = numpyro.infer.MCMC(
                numpyro.infer.NUTS(whitelight_model_for_run, regularize_mass_matrix=False, init_strategy=numpyro.infer.init_to_value(values=soln), target_accept_prob=0.9),
                num_warmup=1000, num_samples=1000, progress_bar=True, jit_model_args=True
            )
            mcmc.run(key_master, data.wl_time, data.wl_flux_err, y=data.wl_flux, prior_params=hyper_params_wl)
            samples = mcmc.get_samples()
        except Exception as e:
            print(f"Fit failed for {detrending_type}: {e}")
            continue

        # Analysis & Storage
        # Extract basic params for reconstruction
        best_params = {k: jnp.nanmedian(v) for k,v in samples.items() if k not in ['u']}
        best_params['u'] = jnp.nanmedian(samples['u'], axis=0)
        best_params['period'] = periods

        # Reconstruct Model for Residuals (Using manual reconstruction to avoid 'vmap' mismatch if any)
        # Note: compute_transit_model expects arrays for planets.

        # Prepare params for compute_transit_model
        transit_params = {
            'period': periods, 'duration': jnp.array([jnp.nanmedian(samples[f'duration_{i}']) for i in range(n_planets)]),
            't0': jnp.array([jnp.nanmedian(samples[f't0_{i}']) for i in range(n_planets)]),
            'b': jnp.array([jnp.nanmedian(samples[f'b_{i}']) for i in range(n_planets)]),
            'rors': jnp.array([jnp.nanmedian(samples[f'rors_{i}']) for i in range(n_planets)]),
            'u': best_params['u']
        }
        transit_model = compute_transit_model(transit_params, data.wl_time)

        # Reconstruct Trend
        t_shift = data.wl_time - jnp.min(data.wl_time)
        c = best_params.get('c', 1.0)
        v = best_params.get('v', 0.0)

        if 'gp' in detrending_type:
            # GP Reconstruction
            gp_func_map = {
                'gp': compute_lc_gp_mean, 'linear+gp': compute_lc_linear_gp_mean,
                'quadratic+gp': compute_lc_quadratic_gp_mean, 'explinear+gp': compute_lc_explinear_gp_mean
            }
            gp_mean_func = gp_func_map.get(detrending_type, compute_lc_gp_mean)

            from models.gp import build_gp
            # We need to rebuild GP with posterior median parameters
            # But wait, build_gp takes (params, t, error).
            # We need the posterior mean of the GP process condition on data.
            # Using tinygp directly
            import tinygp
            kernel = tinygp.kernels.quasisep.Matern32(
                scale=jnp.exp(best_params['GP_log_rho']),
                sigma=jnp.exp(best_params['GP_log_sigma']),
            )
            gp = tinygp.GaussianProcess(
                kernel, data.wl_time, diag=best_params['error']**2,
                mean=partial(gp_mean_func, best_params)
            )
            cond_gp = gp.condition(data.wl_flux, data.wl_time).gp
            full_model = cond_gp.loc # Posterior mean
        else:
            # Parametric Reconstruction
            trend = c
            if 'linear' in detrending_type: trend += v * t_shift
            if 'quadratic' in detrending_type: trend += v * t_shift + best_params.get('v2', 0.0) * t_shift**2
            if 'explinear' in detrending_type: trend += v * t_shift + best_params.get('A', 0.0) * jnp.exp(-t_shift/best_params.get('tau', 1.0))
            full_model = transit_model + trend # Approximation if transit not multiplicative? Code uses additive (transit + trend) or (transit + trend + 1)?
            # models/trends.py: compute_lc_linear: return lc_transit + trend.
            # And lc_transit is negative flux drop?
            # models/core.py: total_flux = jnp.sum(batched_lcs).
            # jaxoplanet lcs are (1 + delta). Wait.
            # limb_dark_light_curve returns the flux (normalized to 1 out of transit).
            # So transit model is ~1.0 in transit.
            # compute_lc_linear: lc_transit + trend.
            # If trend includes 'c' (approx 1.0).
            # We need to subtract 1 from transit model if we add trend?
            # Let's check `compute_lc_linear` in `trends.py`:
            # lc_transit = compute_transit_model(params, t)
            # trend = params["c"] + ...
            # return lc_transit + trend
            # If `lc_transit` is around 1.0, and `trend` is around 1.0, sum is 2.0. This is wrong.
            # `compute_transit_model` returns `limb_dark_light_curve` output.
            # `jaxoplanet` `limb_dark_light_curve` returns flux, usually normalized to 1.
            # Let's check `models/core.py` again.
            pass

        # ... (Metrics calculation) ...
        # I need to trust the model execution for residuals.
        residuals = data.wl_flux - full_model

        n_params = count_parameters(detrending_type, n_planets)
        aic = compute_aic(len(residuals), residuals, n_params)
        bic = compute_bic(len(residuals), residuals, n_params)
        beta, _, _, _ = calculate_beta_metrics(residuals, np.median(np.diff(data.wl_time))*86400)

        all_results[detrending_type] = {'aic': aic, 'bic': bic, 'beta': beta, 'n_params': n_params, 'rms': np.std(residuals)*1e6}
        print(f"  AIC: {aic:.2f}, BIC: {bic:.2f}, Beta: {beta:.2f}")

    # ----------------------------------------------------
    # Plotting Comparison
    # ----------------------------------------------------
    print("\nComparison Results:")
    df_res = pd.DataFrame(all_results).T.sort_values('bic')
    print(df_res)
    df_res.to_csv(f'{output_dir}/{instrument_full_str}_comparison_summary.csv')

    # Plotting logic (simplified from original)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    df_res['beta'].plot(kind='bar', ax=axes[0], title='Beta')
    axes[0].axhline(1, color='r', ls='--')
    df_res['aic'].plot(kind='bar', ax=axes[1], title='AIC')
    df_res['rms'].plot(kind='bar', ax=axes[2], title='RMS (ppm)')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{instrument_full_str}_comparison.png')
    print(f"Saved comparison plot to {output_dir}/{instrument_full_str}_comparison.png")

if __name__ == "__main__":
    main()
