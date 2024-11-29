import json
import traceback
import concurrent.futures

import numpy as np
import pymc as pm
import pytensor.tensor as tt

import matplotlib.pyplot as plt
import arviz as az
import specfit as sf


# helper function
def cp_design_mat(x, cps, module=tt):
    return (0.5 * (1.0 + module.sign(module.tile(x[:, None], (1, cps.shape[0])) - cps)))


def get_spline_model(name, freq, S, sigma, nu0, order=1):

    norm_f = freq/nu0
    x_data = tt.log(norm_f)

    y_err = np.array(sigma)

    if (order == 0):
        with pm.Model() as _model:

            k = pm.Normal("k", mu=0, sigma=1)  # initial slope
            m = pm.Normal("m", mu=0, sigma=1)  # offset

            logS = k*x_data + m
            D = pm.StudentT("likelihood", nu=5, mu=tt.exp(logS),
                            sigma=y_err, observed=S)
        _var_names = ["k", "m"]
    else:
        with pm.Model() as _model:

            cps = pm.Uniform("cps",lower=norm_f[0], upper=norm_f[-1], size=order)
            logcps = tt.log(cps)
            A = cp_design_mat(x_data, logcps)

            # the following are unknown parameters that you can estimate by giving 
            # them priors and then sampling with NUTS
            k = pm.Normal("k", mu=0, sigma=1)  # initial slope
            m = pm.Normal("m", mu=0, sigma=1)  # offset
            delta = pm.Normal("delta", mu=0, sigma=2, size=order) # slope parameters

            # generate the function at x values
            # y = (k + tt.dot(A, delta))*x + (m + tt.dot(A, -cps * delta))

            # logS = pm.Deterministic("logS", (k + pm.math.dot(A, delta))*x_data +
            #                       (m + pm.math.dot(A, -logcps * delta)))
            logS = (k + pm.math.dot(A, delta))*x_data + (m + pm.math.dot(A, -logcps * delta))
            D = pm.StudentT("likelihood", nu=5, mu=tt.exp(logS), sigma=y_err,
                            observed=S)
        _var_names = ["cps", "k", "m", "delta"]

    return _model, _var_names


def evaluate_spline(f_values, cps, k, m, delta):
    logcps = np.log(cps)

    A = cp_design_mat(f_values, logcps, module=np)
    logS = (k + np.dot(A, delta))*f_values + (m + np.dot(A, -logcps * delta))

    return logS


def plot_spline(name, freq, S, yerr, nu0, idata, n_samples=300):
    # x axis

    x = np.log(freq/nu0)
    x_curve = np.linspace(x[0], x[-1], 1000)
    y_scale = 1000

    with plt.rc_context({"axes.grid": True,
                         "axes.grid.which": "both",
                         "axes.formatter.min_exponent": 2}):

        fig, ax = plt.subplots(figsize=(6, 4), layout='constrained')
        ax.set_xscale("log", nonpositive='clip')
        ax.set_yscale("log", nonpositive='clip')

        ax.set_xlabel("Frequency (GHz)")
        ax.set_ylabel("Flux (mJy)")

        for i in range(n_samples):
            sample = sf.get_random_sample(idata)

            try:
                cps = sample['cps']
                k = sample['k']
                m = sample['m']
                delta = sample['delta']

                logS = evaluate_spline(x_curve, cps, k, m, delta)
            except:
                k = sample['k']
                m = sample['m']

                logS = k*x_curve + m

            ax.plot(np.exp(x_curve), np.exp(logS)*y_scale, alpha=10/n_samples, color='k', linewidth=3)
        ax.errorbar(freq/nu0, S*y_scale, yerr=yerr*y_scale, fmt='o', label="data")
        ax.set_title(name)
        # ax.legend()
        plt.savefig(f"{name}_{n_samples}.pdf")
        # plt.show()


def treetype(x):
    if isinstance(x, list):
        return f"{type(x)} -> {treetype(x[0])}"
    else:
        return f"{type(x)}"


def inner_piecewise_linear(name, freq, S, sigma, nu0,
                           order=1, n_samples=3000):
    print(f"piecewise_linear(name=\"{name}\")")
    spline_model, var_names = get_spline_model(name, freq, S, sigma, nu0=nu0, order=order)

    n_tune = n_samples
    RANDOM_SEED = 123
    n_chains = 4

    with spline_model:
        idata = pm.sample_prior_predictive()
        idata.extend(pm.sample(draws=n_samples, tune=n_tune,
                            random_seed=RANDOM_SEED,
                            chains=n_chains))
        # pm.sample_posterior_predictive(idata, extend_inferencedata=True)

    lml = sf.process_log_marginal_likelihood(spline_model, n_samples)

    # Compute the LOO model comparison 
    # https://www.pymc.io/projects/docs/en/latest/learn/core_notebooks/model_comparison.html
    with spline_model:
        pm.compute_log_likelihood(idata)
        loo = az.loo(idata)
        print(loo)

    summ = az.summary(idata, var_names=var_names)
    print(summ)

    plot_spline(name, freq=freq, S=S, yerr=sigma, nu0=nu0, idata=idata, n_samples=500)
    plot_spline(name, freq=freq, S=S, yerr=sigma, nu0=nu0, idata=idata, n_samples=0)

    ret = get_posterior_samples(idata, 1000)

    ret['summary'] = str(summ)
    ret['name'] = name
    ret['order'] = order
    ret['log_marginal_likelihood'] = float(np.mean(lml))

    for k in ret.keys():
        print(k, treetype(ret[k]))

    with open(f"{name}.json", 'w') as json_file:
        json.dump(ret, json_file, indent=4, sort_keys=True)

    with plt.rc_context({"axes.grid": True, "figure.constrained_layout.use": True}):
        az.plot_trace(idata, var_names=var_names, filter_vars="like")
        plt.savefig(f"{name}_trace.pdf")

        az.plot_pair(
                idata.posterior,
                var_names=var_names,
                kind="hexbin",
                filter_vars="like",
                marginals=True,
                figsize=(12, 12),
            );
        plt.savefig(f"{name}_posterior_pairs.pdf")

    return idata, ret


def piecewise_linear(*args, **kwargs):
    with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
        future = executor.submit(inner_piecewise_linear, *args, **kwargs)
        return future.result()


def cps_2_slope(k, cps, delta):
    n = len(cps)
    slopes = []
    slope = k
    for i in range(n):
        ds = delta[i]
        c = cps[i]
        # print(f"slope = {slope}, x < {c}")
        slopes.append(slope)
        slope = slope + ds

    # print(f"slope = {slope}, x > {c}")
    slopes.append(slope)
    return slopes


def get_posterior_samples(idata, n_samples):
    '''
        @param n_samples: The number of samples to return
        @param idata: The inference data object

        Return  samples from the posterior distribution of the model fit.
        @return a dict containing the samples, as well as some statistics of these
    '''
    ret = {}
    try:
        slope_array = []
        cps_array = []
        k_array = []
        m_array = []
        delta_array = []

        for i in range(n_samples):
            sample = sf.get_random_sample(idata)

            cps = sample['cps']
            k = sample['k']
            m = sample['m']
            delta = sample['delta']

            slopes = cps_2_slope(k, cps, delta)

            m_array.append(float(m))
            k_array.append(float(k))
            delta_array.append(delta)
            cps_array.append(cps)
            slope_array.append(slopes)

        slope_array = np.array(slope_array)
        cps_array = np.array(cps_array)

        ret['slope'] = slope_array.tolist()
        ret['cps'] = cps_array.tolist()
        ret['k'] = np.array(k_array).tolist()
        ret['m'] = np.array(m_array).tolist()
        ret['delta'] = np.array(delta_array).tolist()
        ret['slopes'] = np.mean(slope_array, axis=0).tolist()
        ret['slopes_sigma'] = np.std(slope_array, axis=0).tolist()
        ret['change_point'] = np.mean(cps_array, axis=0).tolist()
        ret['change_point_sigma'] = np.std(cps_array, axis=0).tolist()
        ret['n_samples'] = len(k_array)
    except Exception as e:
        print(f"Exception {e}")
        print(traceback.format_exc())
        pass
    return ret


if __name__ == "__main__":

    if False:
        # x axis
        x = np.linspace(0, 10, 100)

        # locations on axis of the changepoints
        cps = np.array([2, 4, 7])

        A = cp_design_mat(x, cps, module=np)

        # the following are unknown parameters that you can estimate by giving 
        # them priors and then sampling with NUTS
        k = 0.0  # initial slope
        m = 0.3 # offset
        delta = np.array([1, -1, -1]) # slope parameters

        cps_2_slope(k, cps, delta)
        # generate the function at x values
        y = (k + np.dot(A, delta))*x + (m + np.dot(A, -cps * delta))
        # 
        # print(x)
        # print(y)
        
        data_diff = (np.dot(A, delta))
        print((data_diff))

        plt.plot(x, y)
        plt.show()
    else:
        original_data = np.array(
            [[0.408,  6.24, 0.312 ],
            [0.843, 13.65, 0.6825],
            [1.38 , 14.96, 0.748 ],
            [1.413, 14.87, 0.7435],
            [1.612, 14.47, 0.7235],
            [1.66 , 14.06, 0.703 ],
            [1.665, 14.21, 0.7105],
            [2.295, 11.95, 0.5975],
            [2.378, 11.75, 0.5875],
            [4.8  ,  5.81, 0.2905],
            [4.8  ,  5.76, 0.288 ],
            [4.835,  5.72, 0.286 ],
            [4.85 ,  5.74, 0.287 ],
            [8.415,  2.99, 0.1495],
            [8.42 ,  2.97, 0.1485],
            [8.64 ,  2.81, 0.1405]])

        freq_ghz, S, delta_S = original_data.T
        freq = freq_ghz*1e9
        nu0=1.0e9

        name = "j1939-6342"
        sigma = delta_S
        mu=S

        idata, ret = piecewise_linear(name, freq, mu, sigma, nu0, order=2, n_samples=3000)
        print(ret)

