from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
#import kmeans1d

# from patsy import dmatrix

from .posterior_helper import get_stats, chain_covariance,  spline_run_or_load
from .posterior_helper import Tflux as flux

# Need patsy and python3-graphviz 

RANDOM_SEED = 8927


def get_knots(x_values):

#    clusters, centroids = kmeans1d.cluster(x_values, 2)
#    print(clusters)
#    print(centroids)

    knot_list = np.linspace(x_values[0], x_values[-1], 1, endpoint=False)

    # knot_list = centroids
    return knot_list


def get_spline_design(x_values, y_values, knot_list):

    print(f"Knot List: {knot_list}")
    degree = 3
    spline_design = f"bs(freq, knots=knots, degree={degree}) - max"
    #spline_design = f"cr(freq, knots=knots) - max"
    return dmatrix(
        spline_design,
        {"freq": x_values,
         "knots": knot_list,
         "max": np.max(y_values)},
    )


def get_spline_model(name, freq, mu, sigma, nu0):
    # mumax = np.max(mu)
    # print(f"Mu max: {mumax}")
    # print(f"Mu max log: {np.log(mumax)}")
    # 
    # means = [0.0 for i in range(0, order)]
    # sigmas = [2.5 for i in range(0, order)]
    # means[0] = np.log(mumax)
    # sigmas[0] = 2.5
    # 
    # with pm.Model() as _model:
    #     _a = [pm.Normal(f"a[{i}]", mu=means[i], sigma=sigmas[i]) for i in range(order)]
    # 
    #     _x = pm.MutableData('frequencies', freq)  # a data container, mutable
    #     _brightness = flux(_x, _a, nu0)
    # 
    #     # Use a StudentT likelihood to deal with outliers 
    #     _likelihood = pm.StudentT("likelihood", nu=5, mu=_brightness,
    #                               sigma=np.array(sigma), shape=_x.shape, observed=np.array(mu))
    # 
    x_data = np.log(freq/nu0)
    y_data = np.log(mu)
    y_err = np.array(sigma)

    knot_list = get_knots(x_data)

    B = get_spline_design(x_data, y_data, knot_list)

    COORDS = {
                "splines": np.arange(B.shape[1]),
                "obs": range(len(y_data))
            }
    with pm.Model(coords=COORDS) as _model:
        # a = pm.Normal("a", 0, 3)
        w = pm.Normal("w", mu=0, sigma=2, size=B.shape[1], dims="splines")
        # mu = pm.Deterministic("mu", a + pm.math.dot(np.asarray(B, order="F"), w.T))
        # sigma = pm.Exponential("sigma", 1)
        mu = pm.Deterministic("mu",  pm.math.dot(np.asarray(B, order="F"), w.T))
        D = pm.Normal("likelihood", mu=mu, sigma=y_err, observed=y_data, dims="obs")

    return _model, knot_list


def spline_inference(name, freq, mu, sigma, nu0,
                   n_samples=1000):
    """Infer a spectral spline from original measurements

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
    _model, knot_list = get_spline_model(name, freq, mu, sigma, nu0=nu0)
    _idata = spline_run_or_load(_model, fname=f"idata_{name}_{len(knot_list)}.nc",
                                n_samples=n_samples, n_tune=n_samples,
                                cache=False)

    print(_idata.posterior.keys())

    chain = 0
    a = _idata.posterior.get('w')[chain, :]
    a = np.array(a)
    a_cov, a_corr = np.cov(a), np.corrcoef(a)

    stats, names = get_stats(_idata.posterior)

    return names, stats, a_cov, a_corr, _idata, _model, knot_list


# x_data = np.log(freq/nu0)
# y_data = np.log(S)
# y_err = 0.01
# 
# 
# num_knots = 3
# 
# clusters, centroids = kmeans1d.cluster(x_data, num_knots)
# print(clusters)
# print(centroids)
# 
# knot_list = centroids
# # knot_list = np.linspace(x_data[0], x_data[-1], num_knots, endpoint=True)
# print(f"Knot List: {knot_list}")
# 

def plot_spline_design(freq, mu, nu0, knot_list):
    x_data = np.log(freq/nu0)
    y_data = np.log(mu)
    
    test_freq = np.linspace(x_data[0], x_data[-1], 100)
    B = get_spline_design(test_freq, y_data, knot_list)
    spline_df = (
        pd.DataFrame(B)
        .assign(freq=test_freq)
        .melt("freq", var_name="spline_i", value_name="value")
    )

    color = plt.cm.magma(np.linspace(0, 0.80, len(spline_df.spline_i.unique())))

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    for i, c in enumerate(color):
        subset = spline_df.query(f"spline_i == {i}")
        subset.plot("freq", "value", c=c, ax=ax, label=i)
    ax.legend(title="Spline Index", loc="upper center", fontsize=8, ncol=6);
    plt.show()


def plot_spline(idata, freq, mu, nu0, knot_list):

    x_data = np.log(freq/nu0)
    y_data = np.log(mu)

    test_freq = np.linspace(x_data[0], x_data[-1], 100)

    wp = idata.posterior["w"].mean(("chain", "draw")).values

    B = get_spline_design(test_freq, y_data, knot_list)

    spline_df = (
        pd.DataFrame(B * wp.T)
        .assign(freq=test_freq)
        .melt("freq", var_name="spline_i", value_name="value")
    )

    spline_df_merged = (
        pd.DataFrame(np.dot(B, wp.T))
        .assign(freq=test_freq)
        .melt("freq", var_name="spline_i", value_name="value")
    )

    color = plt.cm.rainbow(np.linspace(0, 1, len(spline_df.spline_i.unique())))

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    for i, c in enumerate(color):
        subset = spline_df.query(f"spline_i == {i}")
        subset.plot("freq", "value", c=c, ax=ax, label=i)
    spline_df_merged.plot("freq", "value", c="black", lw=2, ax=ax)
    ax.legend(title="Spline Index", loc="lower center", fontsize=8, ncol=6)

    ax.plot(x_data, y_data, 'o')
    for knot in knot_list:
        ax.axvline(knot, color="grey", alpha=0.4);

    plt.show()

# B = get_spline_design(x_data)
# 
# COORDS = {
#             "splines": np.arange(B.shape[1]),
#             "obs": range(len(y_data))
#          }
# with pm.Model(coords=COORDS) as spline_model:
#     # a = pm.Normal("a", 0, 3)
#     w = pm.Normal("w", mu=0, sigma=2, size=B.shape[1], dims="splines")
#     # mu = pm.Deterministic("mu", a + pm.math.dot(np.asarray(B, order="F"), w.T))
#     # sigma = pm.Exponential("sigma", 1)
#     mu = pm.Deterministic("mu",  pm.math.dot(np.asarray(B, order="F"), w.T))
#     D = pm.Normal("D", mu=mu, sigma=y_err, observed=y_data, dims="obs")


# pm.model_to_graphviz(spline_model)
# plt.show()
#
# 
# 
# original_data = np.array(
#     [[0.408,  6.24, 0.312 ],
#      [0.843, 13.65, 0.6825],
#      [1.38 , 14.96, 0.748 ],
#      [1.413, 14.87, 0.7435],
#      [1.612, 14.47, 0.7235],
#      [1.66 , 14.06, 0.703 ],
#      [1.665, 14.21, 0.7105],
#      [2.295, 11.95, 0.5975],
#      [2.378, 11.75, 0.5875],
#      [4.8  ,  5.81, 0.2905],
#      [4.8  ,  5.76, 0.288 ],
#      [4.835,  5.72, 0.286 ],
#      [4.85 ,  5.74, 0.287 ],
#      [8.415,  2.99, 0.1495],
#      [8.42 ,  2.97, 0.1485],
#      [8.64 ,  2.81, 0.1405]])
# 
# freq_ghz, S, delta_S = original_data.T
# freq = freq_ghz*1e9
# nu0=1.0e9
# 
# order = 3
# 
# name = "j1939-6342"
# mu = S
# sigma = delta_S
# 
# spline_model = get_spline_model(name, freq, mu, sigma, order, nu0=nu0)
    # _idata = run_or_load(_model, fname=f"idata_{name}.nc",
    #                      n_samples=n_samples, n_tune=n_samples)

# with spline_model:
#     idata = pm.sample_prior_predictive()
#     idata.extend(pm.sample(draws=1000, tune=3000, random_seed=RANDOM_SEED, chains=4))
#     pm.sample_posterior_predictive(idata, extend_inferencedata=True)
# 
# print(idata.keys())
# print(az.summary(idata, var_names=[ "w"]))
# 
# az.plot_trace(idata, var_names=[ "w"]);
# plt.show()
# 
# ## Pair Plots
# 
# az.plot_pair(
#         idata,
#         var_names=['w'],
#         kind="hexbin",
#         filter_vars="like",
#         marginals=True,
#         figsize=(12, 12),
#     )
# plt.tight_layout()
# #plt.savefig(f"posterior_pairs.pdf")
# plt.show()
# 
# #  Do some posterior sampling
# 
# wp = idata.posterior["w"].mean(("chain", "draw")).values
# 
# B = get_spline_design(test_freq)
# 
# spline_df = (
#     pd.DataFrame(B * wp.T)
#     .assign(freq=test_freq)
#     .melt("freq", var_name="spline_i", value_name="value")
# )
# 
# spline_df_merged = (
#     pd.DataFrame(np.dot(B, wp.T))
#     .assign(freq=test_freq)
#     .melt("freq", var_name="spline_i", value_name="value")
# )
# 
# 
# color = plt.cm.rainbow(np.linspace(0, 1, len(spline_df.spline_i.unique())))
# fig = plt.figure()
# for i, c in enumerate(color):
#     subset = spline_df.query(f"spline_i == {i}")
#     subset.plot("freq", "value", c=c, ax=plt.gca(), label=i)
# spline_df_merged.plot("freq", "value", c="black", lw=2, ax=plt.gca())
# plt.legend(title="Spline Index", loc="lower center", fontsize=8, ncol=6)
# 
# plt.plot(x_data, y_data, 'o')
# for knot in knot_list:
#     plt.gca().axvline(knot, color="grey", alpha=0.4);
# 
# plt.show()
