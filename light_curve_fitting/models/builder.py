import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import numpyro_ext.distributions as distx
from jaxoplanet.experimental import calc_poly_coeffs

from .core import get_I_power2
from .trends import (
    compute_lc_linear, compute_lc_quadratic, compute_lc_cubic, compute_lc_quartic,
    compute_lc_linear_discontinuity, compute_lc_explinear, compute_lc_spot,
    compute_lc_none, compute_lc_spot_spectroscopic, compute_lc_linear_discontinuity_spectroscopic
)
from .gp import (
    compute_lc_gp_mean, compute_lc_linear_gp_mean, compute_lc_quadratic_gp_mean,
    compute_lc_explinear_gp_mean, compute_lc_gp_spectroscopic, compute_lc_linear_gp_spectroscopic,
    compute_lc_quadratic_gp_spectroscopic, compute_lc_explinear_gp_spectroscopic,
    build_gp, build_gp_linear, build_gp_quadratic, build_gp_explinear
)

COMPUTE_KERNELS = {
    'linear': compute_lc_linear,
    'quadratic': compute_lc_quadratic,
    'cubic': compute_lc_cubic,
    'quartic': compute_lc_quartic,
    'linear_discontinuity': compute_lc_linear_discontinuity,
    'explinear': compute_lc_explinear,
    'spot': compute_lc_spot,
    'gp': compute_lc_gp_mean,
    'none': compute_lc_none,
    'gp_spectroscopic': compute_lc_gp_spectroscopic,
    'linear+gp_spectroscopic': compute_lc_linear_gp_spectroscopic,
    'quadratic+gp_spectroscopic': compute_lc_quadratic_gp_spectroscopic,
    'explinear+gp_spectroscopic': compute_lc_explinear_gp_spectroscopic,
    'spot_spectroscopic': compute_lc_spot_spectroscopic,
    'linear_discontinuity_spectroscopic': compute_lc_linear_discontinuity_spectroscopic,
}
    
def create_whitelight_model(detrend_type='linear', n_planets=1, ld_profile='quadratic'):
    print(f"Building whitelight model with: detrend_type='{detrend_type}' for {n_planets} planets")

    # Pre-compute components set for faster lookup
    detrend_components = set(detrend_type.split('+'))

    def _whitelight_model_static(t, yerr, y=None, prior_params=None):
        durations, t0s, bs, rorss = [], [], [], []

        for i in range(n_planets):
            logD = numpyro.sample(f"logD_{i}", dist.Normal(jnp.log(prior_params['duration'][i]), 3e-2))
            durations.append(numpyro.deterministic(f"duration_{i}", jnp.exp(logD)))
            t0s.append(numpyro.sample(f"t0_{i}", dist.Normal(prior_params['t0'][i], 3e-2)))
            _b = numpyro.sample(f"_b_{i}", dist.Uniform(-2.0, 2.0))
            bs.append(numpyro.deterministic(f'b_{i}', jnp.abs(_b)))
            depths = numpyro.sample(f'depths_{i}', dist.Uniform(1e-5, 0.5))
            rorss.append(numpyro.deterministic(f"rors_{i}", jnp.sqrt(depths)))

        if ld_profile == 'quadratic':
            u = numpyro.sample("u", distx.QuadLDParams())
        elif ld_profile == 'power2':
            POLY_DEGREE = 12
            MUS = jnp.linspace(0.0, 1.00, 300, endpoint=True)
            u_theory = prior_params['u']
            c1_mu, c2_mu = u_theory[0], u_theory[1]
            c1 = numpyro.sample('c1', dist.Uniform(0.0, 1.0)) #dist.TruncatedNormal(c1_mu, 0.1, low=0.0, high=1.0))
            c2 = numpyro.sample('c2', dist.Uniform(0.0, 1.0)) #dist.TruncatedNormal(c2_mu, 0.05, low=0.001, high=1.0))
            power2_profile = get_I_power2(c1, c2, MUS)
            u_poly = calc_poly_coeffs(MUS, power2_profile, poly_degree=POLY_DEGREE)
            u = numpyro.deterministic('u', u_poly)
        else:
            raise ValueError(f"Unknown ld_profile: {ld_profile}")

        log_jitter = numpyro.sample('log_jitter', dist.Uniform(jnp.log(1e-5), jnp.log(1e-2)))
        error = numpyro.deterministic('error', jnp.sqrt(jnp.exp(log_jitter)**2 + yerr**2))

        params = {
            "period": prior_params['period'], "duration": jnp.array(durations), "t0": jnp.array(t0s),
            "b": jnp.array(bs), "rors": jnp.array(rorss), "u": u,
        }

        # --- PARAMETER SAMPLING BASED ON COMPONENTS ---
        # Using set intersection for cleaner checks
        has_linear_term = not detrend_components.isdisjoint({'linear', 'quadratic', 'cubic', 'quartic', 'linear_discontinuity', 'explinear', 'spot', 'gp'})
        if has_linear_term:
            params['c'] = numpyro.sample('c', dist.Normal(1.0, 0.1))

        has_velocity = not detrend_components.isdisjoint({'linear', 'quadratic', 'cubic', 'quartic', 'linear_discontinuity', 'explinear', 'spot'})
        if has_velocity:
            params['v'] = numpyro.sample('v', dist.Uniform(-0.1, 0.1))

        if not detrend_components.isdisjoint({'quadratic', 'cubic', 'quartic'}):
            params['v2'] = numpyro.sample('v2', dist.Uniform(-0.1, 0.1))

        if not detrend_components.isdisjoint({'cubic', 'quartic'}):
            params['v3'] = numpyro.sample('v3', dist.Uniform(-0.1, 0.1))

        if 'quartic' in detrend_components:
            params['v4'] = numpyro.sample('v4', dist.Uniform(-0.1, 0.1))

        if 'linear_discontinuity' in detrend_components:
            params['t_jump'] = numpyro.sample('t_jump', dist.Normal(59791.12, 1e-2))
            params['jump'] = numpyro.sample('jump', dist.Normal(0.0, 0.1))

        if 'explinear' in detrend_components:
            params['A'] = numpyro.sample('A', dist.Uniform(-0.1, 0.1))
            log_tau = numpyro.sample('log_tau', dist.Uniform(jnp.log(1e-5), jnp.log(1.0)))
            params['tau'] = numpyro.deterministic('tau', jnp.exp(log_tau))

        if 'spot' in detrend_components:
            params['spot_amp'] = numpyro.sample('spot_amp', dist.Normal(0.0, 0.01))
            params['spot_mu'] = numpyro.sample('spot_mu', dist.Normal(prior_params['spot_guess'], 0.01))
            params['spot_sigma'] = numpyro.sample('spot_sigma', dist.Normal(0.0, 0.01))

        if 'gp' in detrend_components:
            params['GP_log_sigma'] = numpyro.sample('GP_log_sigma', dist.Uniform(jnp.log(1e-5), jnp.log(1e3)))
            params['GP_log_rho'] = numpyro.sample('GP_log_rho', dist.Uniform(jnp.log(0.007), jnp.log(1)))

        # --- MODEL DISPATCH ---
        # Direct dispatch via dictionary to avoid massive if-else chain
        # Note: GP models have a different signature (params, t, error), while others have (params, t)
        # We handle this by checking if it's a GP type.

        if 'gp' in detrend_type: # This covers 'gp', 'linear+gp', etc.
            if detrend_type == 'gp': gp_builder = build_gp
            elif detrend_type == 'linear+gp': gp_builder = build_gp_linear
            elif detrend_type == 'quadratic+gp': gp_builder = build_gp_quadratic
            elif detrend_type == 'explinear+gp': gp_builder = build_gp_explinear
            else: raise ValueError(f"Unknown GP detrend_type: {detrend_type}")

            gp = gp_builder(params, t, error)
            numpyro.sample('obs', gp.numpyro_dist(), obs=y)
        else:
            if detrend_type in COMPUTE_KERNELS:
                lc_model = COMPUTE_KERNELS[detrend_type](params, t)
                numpyro.sample('obs', dist.Normal(lc_model, error), obs=y)
            else:
                 raise ValueError(f"Unknown detrend_type: {detrend_type}")

    return _whitelight_model_static

def create_vectorized_model(detrend_type='linear', ld_mode='free', trend_mode='free', n_planets=1, ld_profile='quadratic'):
    print(f"Building vectorized model with: detrend='{detrend_type}', ld='{ld_mode}', trend='{trend_mode}' for {n_planets} planets")

    if detrend_type not in COMPUTE_KERNELS:
        raise ValueError(f"Unsupported detrend_type for vectorized model: {detrend_type}")

    compute_lc_kernel = COMPUTE_KERNELS[detrend_type]

    def _vectorized_model_static(t, yerr, y=None, mu_duration=None, mu_t0=None, mu_b=None,
                               mu_depths=None, PERIOD=None, trend_fixed=None,
                               ld_interpolated=None, ld_fixed=None,
                               mu_spot_amp=None, mu_spot_mu=None, mu_spot_sigma=None,
                               mu_u_ld=None, gp_trend=None, spot_trend=None, jump_trend=None):

        num_lcs = jnp.atleast_2d(yerr).shape[0]
        durations = mu_duration
        t0s = mu_t0
        bs = mu_b

        depths = numpyro.sample('depths', dist.Uniform(1e-5, 0.5).expand([num_lcs, n_planets]))
        rors = numpyro.deterministic("rors", jnp.sqrt(depths))

        yerr_per_lc = jnp.nanmedian(yerr, axis=1)
        log_jitter = numpyro.sample('log_jitter', dist.Uniform(jnp.log(1e-6), jnp.log(1)).expand([num_lcs]))
        jitter = jnp.exp(log_jitter)
        total_error = numpyro.deterministic('total_error', jnp.sqrt(jitter**2 + yerr_per_lc**2))
        error_broadcast = total_error[:, None] * jnp.ones_like(t)

        if ld_mode == 'free':
            if ld_profile == 'quadratic':
                u = numpyro.sample('u', dist.TruncatedNormal(loc=mu_u_ld, scale=0.2, low=-1.0, high=1.0).to_event(1))
            elif ld_profile == 'power2':
                POLY_DEGREE = 12
                MUS = jnp.linspace(0.0, 1.00, 300, endpoint=True)
                c1_mu = mu_u_ld[:, 0]
                c2_mu = mu_u_ld[:, 1]
                c1 = numpyro.sample('c1', dist.Uniform(0.0, 1.0).expand([num_lcs])) #dist.TruncatedNormal(c1_mu, 0.1, low=0.0, high=1.0))
                c2 = numpyro.sample('c2', dist.Uniform(0.0, 1.0).expand([num_lcs])) #dist.TruncatedNormal(c2_mu, 0.05, low=0.001, high=1.0))

                def compute_one_lc_u(c1_val, c2_val):
                    power2_profile = get_I_power2(c1_val, c2_val, MUS)
                    return calc_poly_coeffs(MUS, power2_profile, poly_degree=POLY_DEGREE)
                u_poly_all = jax.vmap(compute_one_lc_u)(c1, c2)
                u = numpyro.deterministic('u', u_poly_all)
            else:
                raise ValueError(f"Unknown ld_profile: {ld_profile}")
        elif ld_mode == 'fixed':
            if ld_profile == 'quadratic':
                u = numpyro.deterministic('u', ld_fixed)
            elif ld_profile == 'power2':
                POLY_DEGREE = 12
                MUS = jnp.linspace(0.0, 1.00, 300, endpoint=True)
                c1_mu = numpyro.deterministic('c1', ld_fixed[:, 0])
                c2_mu = numpyro.deterministic('c2', ld_fixed[:, 1])
                def compute_one_lc_u(c1_val, c2_val):
                    power2_profile = get_I_power2(c1_val, c2_val, MUS)
                    return calc_poly_coeffs(MUS, power2_profile, poly_degree=POLY_DEGREE)
                u_poly_all = jax.vmap(compute_one_lc_u)(c1_mu, c2_mu)
                u = numpyro.deterministic('u', u_poly_all)
        else:
            raise ValueError(f"Unknown ld_mode: {ld_mode}")

        params = {
            "period": PERIOD, "duration": durations, "t0": t0s, "b": bs, "rors": rors, "u": u,
        }

        in_axes = {"period": None, "duration": None, "t0": None, "b": None, "rors": 0, "u": 0}

        # --- STANDARD DETRENDING (SAMPLES v IF NOT 'none') ---
        if detrend_type != 'none':
            if trend_mode == 'free':
                params['c'] = numpyro.sample('c', dist.Normal(1.0, 0.1).expand([num_lcs]))
                params['v'] = numpyro.sample('v', dist.Normal(0.0, 0.1).expand([num_lcs]))
                in_axes.update({'c': 0, 'v': 0})

                if detrend_type == 'quadratic':
                    params['v2'] = numpyro.sample('v2', dist.Normal(0.0, 0.1).expand([num_lcs]))
                    in_axes.update({'v2': 0})
                elif detrend_type == 'cubic':
                    params['v2'] = numpyro.sample('v2', dist.Normal(0.0, 0.1).expand([num_lcs]))
                    params['v3'] = numpyro.sample('v3', dist.Normal(0.0, 0.1).expand([num_lcs]))
                    in_axes.update({'v2': 0, 'v3': 0})
                elif detrend_type == 'quartic':
                    params['v2'] = numpyro.sample('v2', dist.Normal(0.0, 0.1).expand([num_lcs]))
                    params['v3'] = numpyro.sample('v3', dist.Normal(0.0, 0.1).expand([num_lcs]))
                    params['v4'] = numpyro.sample('v4', dist.Normal(0.0, 0.1).expand([num_lcs]))
                    in_axes.update({'v2': 0, 'v3': 0, 'v4': 0})

                if detrend_type == 'linear_discontinuity':
                    params['t_jump'] = numpyro.sample('t_jump', dist.Normal(59791.12, 0.1).expand([num_lcs]))
                    params['jump'] = numpyro.sample('jump', dist.Normal(0.0, 0.1).expand([num_lcs]))
                    in_axes.update({'t_jump': 0, 'jump': 0})

                if detrend_type == 'explinear':
                    params['A'] = numpyro.sample('A', dist.Normal(0.0, 0.1).expand([num_lcs]))
                    params['tau'] = numpyro.sample('tau', dist.Normal(0.0, 0.1).expand([num_lcs]))
                    in_axes.update({'A': 0, 'tau': 0})
                if detrend_type == 'spot':
                    params['spot_amp'] = numpyro.sample('spot_amp', dist.Normal(mu_spot_amp, 0.01).expand([num_lcs]))
                    params['spot_mu'] = numpyro.sample('spot_mu', dist.Normal(mu_spot_mu, 0.01).expand([num_lcs]))
                    params['spot_sigma'] = numpyro.sample('spot_sigma', dist.Normal(mu_spot_sigma, 0.01).expand([num_lcs]))
                    in_axes.update({'spot_amp': 0, 'spot_mu': 0, 'spot_sigma': 0})

            elif trend_mode == 'fixed':
                trend_temp = numpyro.deterministic('trend_temp', trend_fixed)
                params['c'] = numpyro.deterministic('c', trend_temp[:, 0])
                params['v'] = numpyro.deterministic('v', trend_temp[:, 1])
                in_axes.update({'c': 0, 'v': 0})

                if detrend_type == 'quadratic':
                    params['v2'] = numpyro.deterministic('v2', trend_temp[:, 2])
                    in_axes.update({'v2': 0})
                elif detrend_type == 'cubic':
                    params['v2'] = numpyro.deterministic('v2', trend_temp[:, 2])
                    params['v3'] = numpyro.deterministic('v3', trend_temp[:, 3])
                    in_axes.update({'v2': 0, 'v3': 0})
                elif detrend_type == 'quartic':
                    params['v2'] = numpyro.deterministic('v2', trend_temp[:, 2])
                    params['v3'] = numpyro.deterministic('v3', trend_temp[:, 3])
                    params['v4'] = numpyro.deterministic('v4', trend_temp[:, 4])
                    in_axes.update({'v2': 0, 'v3': 0, 'v4': 0})
                elif detrend_type == 'linear_discontinuity':
                    params['t_jump'] = numpyro.deterministic('t_jump', trend_temp[:, 2])
                    params['jump'] = numpyro.deterministic('jump', trend_temp[:, 3])
                    in_axes.update({'t_jump': 0, 'jump': 0})
                elif detrend_type == 'explinear':
                    params['A'] = numpyro.deterministic('A', trend_temp[:, 2])
                    params['tau'] = numpyro.deterministic('tau', trend_temp[:, 3])
                    in_axes.update({'A': 0, 'tau': 0})
                elif detrend_type == 'spot':
                    params['spot_amp'] = numpyro.deterministic('spot_amp', trend_temp[:, 2])
                    params['spot_mu'] = numpyro.deterministic('spot_mu', trend_temp[:, 3])
                    params['spot_sigma'] = numpyro.deterministic('spot_sigma', trend_temp[:, 4])
                    in_axes.update({'spot_amp': 0, 'spot_mu': 0, 'spot_sigma': 0})
            else:
                raise ValueError(f"Unknown trend_mode: {trend_mode}")

        # Adjust vmap for multi-planet
        if 'gp_spectroscopic' in detrend_type:
            params['A_gp'] = numpyro.sample('A_gp', dist.Uniform(0.5, 2).expand([num_lcs]))
            in_axes['A_gp'] = 0

            # --- FIX: Check if parameters already exist to avoid Duplication Error ---
            if 'linear' in detrend_type:
                if 'v' not in params:
                    params['v'] = numpyro.sample('v', dist.Normal(0.0, 0.1).expand([num_lcs]))
                    in_axes['v'] = 0
            if 'quadratic' in detrend_type:
                if 'v2' not in params:
                    params['v2'] = numpyro.sample('v2', dist.Normal(0.0, 0.1).expand([num_lcs]))
                    in_axes['v2'] = 0
            if 'explinear' in detrend_type:
                if 'A' not in params:
                    params['A'] = numpyro.sample('A', dist.Normal(0.0, 0.1).expand([num_lcs]))
                    params['tau'] = numpyro.sample('tau', dist.HalfNormal(0.1).expand([num_lcs]))
                    in_axes.update({'A': 0, 'tau': 0})

            y_model = jax.vmap(compute_lc_kernel, in_axes=(in_axes, None, None))(params, t, gp_trend)

        elif detrend_type == 'spot_spectroscopic':
            params['A_spot'] = numpyro.sample('A_spot', dist.Uniform(0.5, 2).expand([num_lcs]))
            in_axes['A_spot'] = 0
            y_model = jax.vmap(compute_lc_kernel, in_axes=(in_axes, None, None))(params, t, spot_trend)
        elif detrend_type == 'linear_discontinuity_spectroscopic':
            params['A_jump'] = numpyro.sample('A_jump', dist.Uniform(1e-1, 1e1).expand([num_lcs]))
            in_axes['A_jump'] = 0
            y_model = jax.vmap(compute_lc_kernel, in_axes=(in_axes, None, None))(params, t, jump_trend)
        else:
            y_model = jax.vmap(compute_lc_kernel, in_axes=(in_axes, None))(params, t)

        numpyro.sample('obs', dist.Normal(y_model, error_broadcast), obs=y)

    return _vectorized_model_static

