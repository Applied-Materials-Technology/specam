import warnings
import itertools
from tqdm import tqdm
from typing import Literal, Union

import numpy as np
from numpy.polynomial import polynomial

from lmfit import create_params
from lmfit.models import Model

from scipy.optimize import curve_fit, leastsq, minimize
from scipy.signal import savgol_filter

# from numba import njit

from specam.constants import c1, c2, abs0
from specam.data import SpectralData, SpectralDataFitted


__all__ = [
    'fit_data',
    'fit_data_lmfit', 
    'fit_data_scipy', 
    'fit_data_ratio', 
    'fit_data_pymc',
]

# @njit
def planck_eqn(lam, T):
    return c1 / (lam**5 * (np.exp(c2 / (lam * T)) - 1))


# @njit
def planck_eqn_log(lam, T):
    return np.log(c1) - 5 * np.log(lam) - np.log(np.exp(c2 / (lam * T)) - 1)


# @njit
def planck_wein_eqn(lam, T):
    return c1 / (lam**5 * np.exp(c2 / (lam * T)))


# @njit
def planck_wein_eqn_log(lam, T):
    return np.log(c1) - 5 * np.log(lam) - c2 / (lam * T)


# @njit
def linear_emissivity(lam, C, D, lam_0, lam_inf):
    return D - C * (lam - lam_0) / (lam_inf - lam_0)


# @njit
def intensity_func(lam, T, C, D, lam_0, lam_inf):
    return linear_emissivity(lam, C, D, lam_0, lam_inf) * planck_eqn(lam, T)


# @njit
def intensity_func_log(lam, T, C, D, lam_0, lam_inf):
    return np.log(linear_emissivity(lam, C, D, lam_0, lam_inf)) + planck_eqn_log(lam, T)

def truncate_data(lam, I, I_ratio):
    """Truncate data below ratio of max intensity from left and right sides

    Parameters
    ----------
    lam : np.ndarray
        Wavelength
    I : np.ndarray
        Intensity
    I_ratio : float
        Ratio of maximum intensity to filter below. Set to 0 to remove vales < 0

    Returns
    -------
    lam : np.ndarray
        Truncated wavelength
    I : np.ndarray
        Truncated intensity

    """
    i_max = I.argmax()

    I_cond = I <= I[i_max] * I_ratio
    pos_cond = np.arange(len(I)) < i_max
    # left side
    idxs = np.where(I_cond & pos_cond)[0]
    cutoff_left = idxs.max() + 1 if len(idxs) > 0 else None

    # right side
    idxs = np.where(I_cond & np.logical_not(pos_cond))[0]
    cutoff_right = idxs.min() if len(idxs) > 0 else None

    # if cutoff_left is not None or cutoff_right is not None:
    #     lt = f'{lam[cutoff_left]}' if cutoff_left is not None else 'start'
    #     rt = f'{lam[cutoff_right]}' if cutoff_right is not None else 'end'
    #     print(f'lam range: {lt} - {rt}') 
    return lam[cutoff_left:cutoff_right], I[cutoff_left:cutoff_right]

def fit_data(model_name, spectral_data, **kwargs):
    models = {
        "lmfit": fit_data_lmfit,
        "scipy": fit_data_scipy,
        "pymc": fit_data_pymc,
        "ratio": fit_data_ratio,
    }
    model = models.get(model_name)
    if model is None:
        raise ValueError(f"Unknown model_name `{model_name}`")
    return model(spectral_data, **kwargs)

def fit_data_lmfit(
    spectral_data : type[SpectralData],
    lam_0 : float = None,
    lam_inf : float = None,
) -> SpectralDataFitted:
    """_summary_

    Parameters
    ----------
    spectral_data : type[SpectralData]
        _description_
    lam_0 : float, optional
        _description_, by default None
    lam_inf : float, optional
        _description_, by default None

    Returns
    -------
    SpectralDataFitted
        _description_

    """
    lam = spectral_data.lam
    intensity = spectral_data.intensity

    if lam_0 is None:
        lam_0 = min(lam)
    if lam_inf is None:
        lam_inf = max(lam)

    mod_black = Model(planck_eqn, independent_vars=["lam"])
    params_black = create_params(
        T={"value": 1500 + abs0, "min": 300, "max": 5300},
    )

    mod_linear = Model(intensity_func, independent_vars=["lam"])
    params_linear = create_params(
        lam_0={"value": lam_0, "vary": False},
        lam_inf={"value": lam_inf, "vary": False},
        T={"value": 1500 + abs0, "min": 300, "max": 5300},
        C={"value": 0.5, "min": 0, "max": 1},
        D={"value": 0.5, "min": 0, "max": 1},
        # CD_constraint={'value': 0, 'min': 0},
        # C={'value': 0.5, 'min': 0},
        # D={'expr': 'C+CD_constraint', 'max': 1},
    )

    n_batch = intensity.shape[0]

    fit_vars = ["T", "C", "D"]
    results = {var: np.zeros(n_batch) for var in fit_vars}
    results.update({
        "covar": np.zeros((n_batch, 3, 3)),
        "result": [],
    })

    for i in tqdm(range(n_batch)):
        lam_i, intensity_i = lam, intensity[i]
        lam_i, intensity_i = truncate_data(lam_i, intensity_i, 0.05)

        result_black = mod_black.fit(intensity_i, params_black, lam=lam_i)

        params_linear['T'].value = result_black.best_values['T']
        result = mod_linear.fit(intensity_i, params_linear, lam=lam_i)
        for var in fit_vars:
            results[var][i] = result.best_values[var]
        results["covar"][i] = result.covar
        results["result"].append(result)

    return SpectralDataFitted(
        fitted_data=spectral_data,
        intensity_func=intensity_func,
        intensity_func_log=intensity_func_log,
        intensity_params=("C", "D"),
        intensity_props={"lam_0": lam_0, "lam_inf": lam_inf},
        scalar_data=results,
    )


# @njit
def planck_dT(lam, T):
    return c1 * c2 * np.exp(c2/(lam*T)) / (lam**6 * T**2 * (np.exp(c2/(lam*T)) - 1)**2)


# @njit
def intensity_jac(lam, T, C, D, lam_0, lam_inf):
    dT = linear_emissivity(lam, C, D, lam_0, lam_inf) * planck_dT(lam, T)
    dD = planck_eqn(lam, T)
    dC = dD * (lam_0 - lam) / (lam_0 - lam_inf)
    return np.transpose([dT, dC, dD])


# @njit
def planck_ls(lam, intensity, T):
    return planck_eqn(lam, T) - intensity


# @njit
def intensity_ls(lam, intensity, T, C, D, lam_0, lam_inf):
    return intensity_func(lam, T, C, D, lam_0, lam_inf) - intensity


def lam_range_closure(func, lam_0=0, lam_inf=1):
    def func_wrapped(lam, intensity, T, C, D):
        return func(lam, intensity, T, C, D, lam_0, lam_inf)
        
    return func_wrapped

def ls_closure(func, xdata, ydata):
    def func_wrapped(params):
        return func(xdata, ydata, *params)
    
    return func_wrapped

def fit_data_scipy(
    spectral_data : type[SpectralData],
    lam_0 : float = None,
    lam_inf : float = None,
    init_vals : tuple[float, float, float] = None,
) -> SpectralDataFitted:
    """_summary_

    Parameters
    ----------
    spectral_data : type[SpectralData]
        _description_
    lam_0 : float, optional
        _description_, by default None
    lam_inf : float, optional
        _description_, by default None
    init_vals : tuple[float, float, float], optional
        _description_, by default None

    Returns
    -------
    SpectralDataFitted
        _description_

    """
    lam = spectral_data.lam
    intensity = spectral_data.intensity

    if lam_0 is None:
        lam_0 = min(lam)
    if lam_inf is None:
        lam_inf = max(lam)
    if init_vals is None:
        init_vals = (1500 + abs0, 0.5, 0.5)

    n_batch = intensity.shape[0]

    fit_vars = ["T", "C", "D"]
    results = {var: np.zeros(n_batch) for var in fit_vars}
    results.update({
        "covar": np.zeros((n_batch, 3, 3)),
        "nfev": np.zeros(n_batch),
    })

    intensity_ls_lam = lam_range_closure(intensity_ls, lam_0=lam_0, lam_inf=lam_inf)
    init_vals = np.broadcast_to(init_vals, (n_batch, len(fit_vars)))

    for i, init_val in tqdm(enumerate(init_vals), total=n_batch):
        lam_i, intensity_i = lam, intensity[i]
        lam_i, intensity_i = truncate_data(lam_i, intensity_i, 0.05)

        fit_vals, fit_cov = curve_fit_min(
            planck_ls, 
            lam_i, intensity_i, 
            [init_val[0]],
            # jac=planck_dT,
        )

        fit_vals, fit_cov, infodict, _, _ = curve_fit_min(
            intensity_ls_lam, 
            lam_i, intensity_i, 
            [fit_vals[0], init_val[1], init_val[2]],
            # jac=partial(intensity_jac, lam_0=lam_0, lam_inf=lam_inf),
            full_output=True
        )

        for j, var in enumerate(fit_vars):
            results[var][i] = fit_vals[j]
        results["covar"][i] = fit_cov
        results["nfev"][i] = infodict['nfev']

    return SpectralDataFitted(
        fitted_data=spectral_data,
        intensity_func=intensity_func,
        intensity_func_log=intensity_func_log,
        intensity_params=("C", "D"),
        intensity_props={"lam_0": lam_0, "lam_inf": lam_inf},
        scalar_data=results,
    )

def curve_fit_min(
    f, xdata, ydata, p0=None, jac=None, *, full_output=False, **kwargs
):
    """

    Notes
    -----
    This is a minimal version of `curve_fit` from SciPy with some of the 
    expensive checks removed for efficiency when used for many small datasets.
    Copyright (c) 2001-2002 Enthought, Inc. 2003-2024, SciPy Developers. All 
    rights reserved.

    """
    p0 = np.atleast_1d(p0)

    ydata = np.asarray_chkfinite(ydata, float)
    xdata = np.asarray_chkfinite(xdata, float)

    func_wrapped = ls_closure(f, xdata, ydata)

    res = leastsq(func_wrapped, p0, Dfun=jac, full_output=1, **kwargs)
    popt, pcov, infodict, errmsg, ier = res
    ysize = len(infodict['fvec'])
    cost = np.sum(infodict['fvec'] ** 2)
    if ier not in [1, 2, 3, 4]:
        raise RuntimeError("Optimal parameters not found: " + errmsg)

    warn_cov = False
    if pcov is None or np.isnan(pcov).any():
        # indeterminate covariance
        pcov = np.zeros((len(popt), len(popt)), dtype=float)
        pcov.fill(np.inf)
        warn_cov = True
    elif ysize > p0.size:
        s_sq = cost / (ysize - p0.size)
        pcov = pcov * s_sq
    else:
        pcov.fill(np.inf)
        warn_cov = True

    if warn_cov:
        warnings.warn('Covariance of the parameters could not be estimated')
                    #   category=OptimizeWarning)

    if full_output:
        return popt, pcov, infodict, errmsg, ier
    else:
        return popt, pcov

def fit_data_ratio(
    spectral_data : type[SpectralData],
    lam_0 : float = None,
    lam_inf : float = None,
    polyorder : int = 0, 
    filter_params : dict = None,
    combinations : Union[Literal['all', 'adjacent'], int] = 'all',
    spectral_samples : int = None,
) -> SpectralDataFitted:
    """

    Parameters
    ----------
    spectral_data : type[SpectralData]
        _description_
    polyorder : int, optional
        Order of polynomial fitted to emissivity, by default 2
    filter_params : dict, optional
        Parameters to pass to `savgol_filter`, by default None

    References
    ----------
    .. [1] P.-Y. C. R. Taunay, E. Y. Choueiri, Multi-wavelength Pyrometry Based 
           on Robust Statistics and Cross-Validation of Emissivity Model, 
           Review of Scientific Instruments, 91, 114902, 2020.

    """
    filter_params_default = {
        "window_length": 100,
        "polyorder": 2,
    }
    filter_params_default.update(filter_params or {})
    filter_params = filter_params_default

    lam = spectral_data.lam
    intensity = spectral_data.intensity

    n_batch = intensity.shape[0]

    def func(lam, T, poly_coef, lam_0, lam_inf):
        I_calc = planck_wein_eqn(lam, T)
        if polyorder == 0:
            eps_vec = poly_coef
        else:
            eps_vec = polynomial.polyval(lam * 1e9, poly_coef)
        return planck_wein_eqn(lam, T) * eps_vec

    fit_vars = ["T", "T_std"]
    results = {var: np.zeros(n_batch) for var in fit_vars}
    results.update({
        "poly_coef": np.zeros((n_batch, polyorder + 1)),
        "T_calc": [],
    })

    # # save model fitted intensity curves
    # results["I"] = np.zeros_like(intensity)
    # results["epsilon"] = np.zeros_like(intensity)

    intensity = savgol_filter(intensity, axis=1, **filter_params)

    lam = np.copy(lam)
    for i in tqdm(range(n_batch)):
        lam_i, intensity_i = lam, intensity[i]
        lam_i, intensity_i = truncate_data(lam_i, intensity_i, 0.)

        # all spectral datapoints
        n_lam = len(lam_i)
        idxs = np.arange(n_lam)
        if spectral_samples is not None and spectral_samples < n_lam:
            # downsample
            idxs = np.linspace(0, n_lam-1, spectral_samples + 1)[:-1]
            idxs += n_lam / spectral_samples / 2
            idxs = np.round(idxs).astype(int)

        if combinations == 'all':
            pair_idxs = np.array(list(itertools.combinations(idxs, 2))).T
        elif combinations == 'adjacent':
            pair_idxs = np.array([idxs[:-1], idxs[1:]])
        elif isinstance(combinations, int):
            # combinations up to dist away
            dist = combinations
            assert len(idxs) > 2*(dist+1)
            pair_idxs = [[], []]
            for j in range(1, dist+2):
                pair_idxs[0] += idxs[:-j].tolist()
                pair_idxs[1] += idxs[j:].tolist()
            pair_idxs = np.array(pair_idxs)
        else:
            raise ValueError(f'Unknown combinations `{combinations}`')

        # Ratio of intensities, remove any log < 0
        # ri = intensity_i[pair_idxs[0]] / intensity_i[pair_idxs[1]]
        # keep_cmb = np.where(ri > 0)[0]
        # pair_idxs = pair_idxs[:, keep_cmb]
        # log_ri = np.log(ri[keep_cmb])

        # Ratio of wavelengths and emissivities
        log_ri = np.log(intensity_i[pair_idxs[0]] / intensity_i[pair_idxs[1]])
        log_rl = np.log(lam_i[pair_idxs[1]] / lam_i[pair_idxs[0]])
        log_re = 0.
        T_part = c2 * (1/lam_i[pair_idxs[1]] - 1/lam_i[pair_idxs[0]])

        if polyorder > 0:
            # Determine `log_re` by fitting polynomial to emissivity and
            # finding coefficients that give smallest scatter in results
            min_options = {
                'xatol': 1e-15,
                'fatol': 1e-15, 
                'maxfev': 20000
            }

            def min_func(pc):
                big = 1e16
                eps_vec = polynomial.polyval(lam_i * 1e9, pc)
                if np.any((eps_vec < 0.) | (eps_vec > 1.)):
                    return big
                with np.errstate(invalid='raise'):
                    try:
                        log_re = np.log(eps_vec[pair_idxs[0]] / eps_vec[pair_idxs[1]])
                    except FloatingPointError:
                        return big
                T = T_part / (log_ri - 5 * log_rl - log_re)
                T_qua = np.percentile(T, [25, 75])
                return (T_qua[1] - T_qua[0]) / (T_qua[1] + T_qua[0])
            pc0 = [0.5] + [0.] * polyorder
            sol = minimize(min_func, pc0, method='Nelder-Mead', options=min_options)

            # Calculate temperature from solution
            eps_vec = polynomial.polyval(lam_i * 1e9, sol.x)
            log_re = np.log(eps_vec[pair_idxs[0]] / eps_vec[pair_idxs[1]])

        T_calc = T_part / (log_ri - 5 * log_rl - log_re)
        T_mean = np.mean(T_calc)
        I_calc = planck_wein_eqn(lam_i, T_mean)
        poly_coef = (intensity_i / I_calc).mean() if polyorder == 0 else sol.x

        results["T"][i] = T_mean
        results["T_std"][i] = np.std(T_calc)
        results["poly_coef"][i, :] = poly_coef

        results["T_calc"].append(T_calc)

        # Calculate and save intensity profile
        I_calc = planck_wein_eqn(lam_i, T_mean)
        if polyorder == 0:
            eps_vec = np.full_like(lam_i, (intensity_i / I_calc).mean())
        I_calc *= eps_vec

        # results["I"][i] = I_calc
        # results["epsilon"] [i] = eps_vec

    return SpectralDataFitted(
        fitted_data=spectral_data,
        intensity_func=func,
        intensity_func_log=None,
        intensity_params=("poly_coef", ),
        intensity_props={"lam_0": 0, "lam_inf": 1},
        scalar_data=results,
    )

def fit_data_pymc(
    spectral_data : type[SpectralData],
    lam_0 : float = None,
    lam_inf : float = None,
    model=None,
) -> SpectralDataFitted:
    import pymc as pm
    from sklearn.preprocessing import MinMaxScaler

    lam = spectral_data.lam
    intensity = spectral_data.intensity

    if lam_0 is None:
        lam_0 = min(lam)
    if lam_inf is None:
        lam_inf = max(lam)

    n_batch = intensity.shape[0]

    def default_model(lam_data, intensity_data, lam_0, lam_inf):
        with pm.Model() as model:
            lam_data = pm.Data("lam_data", lam_data)
            scaler = MinMaxScaler()
            intensity_data = scaler.fit_transform(intensity_data[:, None])[:, 0]
            intensity_data = pm.Data("intensity_data", intensity_data)
            T = pm.Uniform("T", 300, 5300)
            # C = pm.Uniform("C", 0, 1)
            # D = pm.Uniform("D", 0, 1)
            # https://discourse.pymc.io/t/how-to-uniformly-sample-two-variables-in-a-lower-triangular-area-y-x/13382/19
            C = pm.Uniform("C", 0, 1)
            D = pm.Uniform("D", C, 1)
            # Penalize high C, so that joint of C,D is uniform
            pm.Potential("pot", pm.math.log(1-C)) 
            sigma = pm.HalfNormal("sigma")

            # likelihood = pm.Normal(
            #     "intensity",
            #     mu=intensity_func(lam_data, T, C, D, lam_0, lam_inf) * scaler.scale_[0] + scaler.min_[0],
            #     sigma=sigma,
            #     observed=intensity_data,
            # )
            normal_dist = pm.Normal.dist(
                mu=intensity_func(lam_data, T, C, D, lam_0, lam_inf) * scaler.scale_[0] + scaler.min_[0], 
                sigma=sigma
            )
            likelihood = pm.Truncated(
                "intensity", normal_dist, lower=0., observed=intensity_data,
            )
            idata = pm.sample(3000)

        return idata, model
    
    def default_log_model(lam_data, intensity_data, lam_0, lam_inf):
        with pm.Model() as model:
            lam_data = pm.Data("lam_data", lam_data)
            intensity_data = pm.Data("intensity_data", np.log(intensity_data))
            T = pm.Uniform("T", 300, 5300)
            C = pm.Uniform("C", 0, 1)
            D = pm.Uniform("D", 0, 1)
            sigma = pm.HalfNormal("sigma")

            likelihood = pm.Normal(
                "intensity",
                mu=intensity_func_log(lam_data, T, C, D, lam_0, lam_inf),
                sigma=sigma,
                observed=intensity_data,
            )
            idata = pm.sample(3000)

        return idata, model

    if model is None:
        model = default_model

    fit_vars = ["T", "C", "D", "sigma"]
    results = {var: np.zeros(n_batch) for var in fit_vars}
    results.update({
        "idata": [],
    })

    for i in range(n_batch):
        lam_i, intensity_i = lam, intensity[i]
        lam_i, intensity_i = truncate_data(lam_i, intensity_i, 0.05)

        idata, _ = model(lam_i, intensity_i, lam_0, lam_inf)
        post_mean = idata.posterior.mean()
        for var in fit_vars:
            results[var][i] = post_mean[var].item()
        results["idata"].append(idata)

    return SpectralDataFitted(
        fitted_data=spectral_data,
        intensity_func=intensity_func,
        intensity_func_log=intensity_func_log,
        intensity_params=("C", "D"),
        intensity_props={"lam_0": lam_0, "lam_inf": lam_inf},
        scalar_data=results,
    )
