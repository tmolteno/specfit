import pymc as pm
import numpy as np

from .posterior_helper import get_stats, chain_covariance
from .posterior_helper import Tflux as flux
from .posterior_helper import flux as num_flux


def run_or_load(mcmc_model, fname, n_samples = 5000, n_tune=5000, n_chains=4, cache=False):
    if cache is True and os.path.exists(fname):
        ret = az.from_netcdf(fname)
    else:
        with mcmc_model:
            #approximation = pm.fit(n=n_samples, method='fullrank_advi') # Reutrns 
            #ret = approximation.sample(n_samples)
            # start = pm.find_MAP()
            # ret = pm.sample(n_samples, init='advi+adapt_diag', tune=n_tune, chains=n_chains, start=start, return_inferencedata=True, discard_tuned_samples=True)
            ret = pm.sample(n_samples, init='advi+adapt_diag', tune=n_tune, chains=n_chains, return_inferencedata=True, discard_tuned_samples=True)
        if cache:
            ret.to_netcdf(fname);
    return ret


def get_model(name, freq, mu, sigma, order, nu0):
    with pm.Model() as _model:
        _a  = [ pm.Normal(f"a[{i}]", mu=0, sigma=2.5, initval=0.1) for i in  range(order) ]
        
        _x = pm.MutableData('frequencies', freq) # a data container, can be changed
        _brightness = flux(_x, _a, nu0)
        ''' Use a StudentT likelihood to deal with outliers 
        '''
        _likelihood = pm.StudentT("likelihood", nu=5, mu=_brightness, sigma=np.array(sigma), observed=np.array(mu))
    return _model

def marginal_likelihood(name, freq, mu, sigma, nu0):

    ret = []
    for _ord in range(2,7):
        _model = get_model(name, freq, mu, sigma, _ord, nu0=nu0)
        with _model:
            _chain = pm.sample_smc(5000, chains=4)
            _bic = np.mean(_chain.report.log_marginal_likelihood)
        print(f"Log Marginal Likelihood: {_ord}:  {_bic}")
        
        ret.append([_ord, _bic])
    
    return np.array(ret)

def data_inference(name, freq, mu, sigma, order, nu0):
    """Infer a spectral polynomial from original measurements

    Use Bayesian inference to infer the coefficients of a polynomial model for
    the spectral data. 
    
    Parameters
    ----------
    name : str
        A unique key to identify this model. This is used to cache results of the inference.
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
    _idata = run_or_load(_model, fname = f"idata_{name}.nc", n_samples=5000, n_tune=order*3000)

    a_cov, a_corr, names = chain_covariance(_idata)
    stats, names = get_stats(_idata)
    
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
        A unique key to identify this model. This is used to cache results of the inference.
    freq : array-like
        The range of frequencies over which this inference should be made (Hz) - in other words
    freq : array-like
        The range of frequencies over which this inference should be made (Hz) - in other words
        
    """
    f = np.geomspace(freq_min, freq_max, nfreq)
    flux_data = (num_flux(f, np.array(a), nu0))
    rng = np.random.default_rng(12345)
    noise = rng.normal(loc=np.zeros_like(f), scale=sigma)
    ## Create fake data for the data-free inference step
    
    order = np.array(a).shape[0]
    # Now do the bayesian inference of the polynomial parameters.
    with pm.Model() as _model:
        fake_data = pm.ConstantData("y", flux_data , dims="obs_id")
        _a  = [ pm.Normal(f"a[{i}]", mu=a[i], sigma=2.5) for i in  range(order) ]
        _brightness = flux(f, _a, nu0)
        _likelihood = pm.Normal("likelihood", mu=_brightness, sigma=np.ones_like(f)*sigma,  observed=fake_data)
    _idata = run_or_load(_model, fname = f"idata_{name}.nc", n_samples=5000, n_tune=order*3000)

    a_cov, a_corr, names = chain_covariance(_idata)
    stats, names = get_stats(_idata)

    #return names, stats, a_cov, a_corr, _idata
    
    #names, stats, a_cov, a_corr, idata = data_inference(name, f, fake_data, sigma=np.ones_like(f)*sigma, order=order, nu0=nu0)

    return names, stats, a_cov, a_corr, f, flux_data
