import pymc3 as pm
import numpy as np

from .posterior_helper import run_or_load, get_stats, chain_covariance, flux



def data_inference(name, freq, mu, sigma, order, nu0):
    """Infer a spectral polynomial from original measurements

    Use Bayesian inference to infer the coefficients of a polynomial model for
    the spectral data. 
    
    Parameters
    ----------
    name : str
        A unique key to identify this model. This is used to cache results of the inference.
    freq : array-like
        The frequencies that the measurements are made
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
    with pm.Model() as _model:
        _a  = [ pm.Normal(f"a[{i}]", mu=0, sigma=2.5, testval=0.1) for i in  range(order) ]
        _brightness = flux(np.array(freq), _a, nu0)
        _likelihood = pm.Normal("likelihood", mu=_brightness, sigma=np.array(sigma), observed=np.array(mu))
    _idata = run_or_load(_model, fname = f"idata_{name}.nc", n_samples=5000, n_tune=order*3000)

    a_cov, a_corr, names = chain_covariance(_idata)
    stats, names = get_stats(_idata)

    return names, stats, a_cov, a_corr, _idata

def datafree_inference(name, freq_min, freq_max, nfreq, sigma, a, nu0):
    """Infer a spectral polynomial covariance using data-free inference

    This is the further elaboration of the docstring. Within this section,
    you can elaborate further on details as appropriate for the situation.
    Notice that the summary and the elaboration is separated by a blank new
    line.

    Parameters
    ----------
    name : str
        A unique key to identify this model. This is used to cache results of the inference.
    freq : array-like
        The range of frequencies over which this inference should be made - in other words
        
    """
    f = np.geomspace(freq_min, freq_max, nfreq)
    
    rng = np.random.default_rng(12345)

    ## Create fake data for the data-free inference step
    
    fake_data = flux(f, a, nu0) + rng.normal(loc=np.zeros_like(f), scale=sigma)
    
    # Now do the bayesian inference of the polynomial parameters.
    names, stats, a_cov, a_corr = data_inference(name, f, fake_data, sigma=np.ones_like(f)*sigma, order=a.shape[0], nu0=nu0)

    return names, stats, a_cov, a_corr, f, fake_data
