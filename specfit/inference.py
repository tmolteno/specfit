#
# Inference Routines for the SpecFIT package.
#
# (c) 2023-2024 Tim Molteno tim@elec.ac.nz
#
import pymc as pm
import numpy as np
import arviz as az

import logging
import os
import traceback

from .posterior_helper import get_stats, chain_covariance, run_or_load
from .posterior_helper import Tflux as flux
from .posterior_helper import flux as num_flux

logger = logging.getLogger()




def get_model(name, freq, mu, sigma, order, nu0):

    mumax = np.max(mu)
    print(f"Mu max: {mumax}")
    print(f"Mu max log: {np.log(mumax)}")

    means = [0.0 for i in range(0, order)]
    sigmas = [2.5 for i in range(0, order)]
    means[0] = np.log(mumax)
    sigmas[0] = 2.5

    with pm.Model() as _model:
        _a = [pm.Normal(f"a[{i}]", mu=means[i], sigma=sigmas[i]) for i in range(order)]

        _x = pm.MutableData('frequencies', freq)  # a data container, mutable
        _brightness = flux(_x, _a, nu0)

        # Use a StudentT likelihood to deal with outliers 
        _likelihood = pm.StudentT("likelihood", nu=5, mu=_brightness,
                                  sigma=np.array(sigma), shape=_x.shape, observed=np.array(mu))
    return _model


def marginal_likelihood(name, freq,
                        mu, sigma, nu0,
                        o_start=2, o_stop=6):
    """
    Estimate the marginal likelihood using Sequential Monte Carlo (SMC)
    sampling.

    The Marginal Likelihood represents the probability of the model itself,
    and can be used to choose the best order for a polynomial model.

    https://www.ma.imperial.ac.uk/~nkantas/sysid09_final_normal_format.pdf

    Parameters
    ----------
    name : str
        A unique key to identify this model. This is used to cache results
        of the inference.
    freq : array-like
        The frequencies at which  the measurements are made (Hz)
    mu : array-like
        The intensities (in Jansky) for each measurement
    sigma : array-like
        The estimate of the standard deviation of each measurement
    nu0 : float
        The normalizing frequency for the polynomial (typically 1 GHz)
    o_start : int
        The lowest order of polynomial fit.
    o_stop : int
        The hightes order of polynomial fit.

    Returns
    -------
    list
        a list of [order ml] where ml is the marginal likelihood
        (or Beyesian Evidence). This is scaled so that the maximum
        marginal likelihood is 1.0 (since we only know this up to
        a scaling)
    """

    ret = []
    rng = np.random.default_rng(123)

    for _ord in range(o_start, o_stop+1):
        try:
            _model = get_model(name, freq, mu, sigma, _ord, nu0=nu0)
            with _model:

                trace = pm.sample_smc(draws=1500*_ord, kernel=pm.smc.kernels.IMH,
                                        chains=6, threshold=0.6,
                                        correlation_threshold=0.01,
                                        random_seed=rng, return_inferencedata=False, progressbar=False)

                lml = trace.report.log_marginal_likelihood
                _evidence = np.mean([chain[-1] for chain in lml])

                print(f"Log Marginal Likelihood: {_ord}:  {_evidence}")

                ret.append([_ord, _evidence])
        except Exception as e:
            print(f"Exception {e}")
            print(_idata.sample_stats)
            print(traceback.format_exc())
            ret.append([_ord, float("nan")])

    ret = np.array(ret)
    bmax = np.max(ret[:,1])
    ret[:,1] = np.exp(ret[:,1] - bmax)

    return ret


def posterior_predictive_sampling(idata, model, num_pp_samples):

    with model:
        pm.sample_posterior_predictive(idata, extend_inferencedata=True)

    # reduced_samples = az.extract(idata.posterior, num_samples=num_pp_samples)
    # post = reduced_samples.posterior
    post = az.extract(idata.posterior, num_samples=num_pp_samples)
    var_names = list(post.data_vars)

    ret = {}
    for v in var_names:
        ret[v] = post[v]

    return ret
    
    # az.plot_ppc(idata, num_pp_samples=100)


def data_inference(name, freq, mu, sigma, order, nu0,
                   n_samples=10000):
    """Infer a spectral polynomial from original measurements

    Use Bayesian inference to infer the coefficients of a polynomial model for
    the spectral data. 

    Parameters
    ----------
    name : str
        A unique key to identify this model. This is used to cache results
        of the inference.
    freq : array-like
        The frequencies at which  the measurements are made (Hz)
    mu : array-like
        The intensities (in Jansky) for each measurement
    sigma : array-like
        The estimate of the standard deviation of each measurement
    order : int
        The order of the polynomial fit.

    Returns
    -------
    list
        a list of strings representing the header columns
    """
    _model = get_model(name, freq, mu, sigma, order, nu0=nu0)
    _idata = run_or_load(_model, fname=f"idata_{name}.nc",
                         n_samples=n_samples, n_tune=n_samples)

    a_cov, a_corr, names = chain_covariance(_idata.posterior)
    stats, names = get_stats(_idata.posterior)

    return names, stats, a_cov, a_corr, _idata, _model


def datafree_inference(name, freq_min, freq_max, nfreq, sigma, a, nu0):
    """Infer a spectral polynomial covariance using data-free inference

    This is the further elaboration of the docstring. Within this section,
    you can elaborate further on details as appropriate for the situation.
    Notice that the summary and the elaboration is separated by a blank new
    line.

    Parameters
    ----------
    name : str
        A unique key to identify this model. This is used to cache results
        of the inference.
    freq : array-like
        The range of frequencies over which this inference should be made (Hz)
    freq : array-like
        The range of frequencies over which this inference should be made (Hz)
    """
    f = np.geomspace(freq_min, freq_max, nfreq)
    flux_data = (num_flux(f, np.array(a), nu0))
    rng = np.random.default_rng(12345)
    noise = rng.normal(loc=np.zeros_like(f), scale=sigma)

    ## Create fake data for the data-free inference step

    order = np.array(a).shape[0]
    # Now do the bayesian inference of the polynomial parameters.
    with pm.Model() as _model:
        fake_data = pm.ConstantData("y", flux_data, dims="obs_id")
        _a = [pm.Normal(f"a[{i}]", mu=a[i], sigma=2.5) for i in  range(order)]
        _brightness = flux(f, _a, nu0)
        _likelihood = pm.Normal("likelihood", mu=_brightness,
                                sigma=np.ones_like(f)*sigma,
                                observed=fake_data)
    _idata = run_or_load(_model, fname=f"idata_{name}.nc",
                         n_samples=5000, n_tune=order*3000)

    a_cov, a_corr, names = chain_covariance(_idata.posterior)
    stats, names = get_stats(_idata.posterior)

    return names, stats, a_cov, a_corr, f, flux_data
