import matplotlib.pyplot as plt
import json
import numpy as np
import pymc as pm

from piecewise_linear import get_spline_model, get_posterior_samples, evaluate_spline


if __name__ == "__main__":

    name = "data_Source_RA:4d30m20.69878782s_DEC:-25d40m58.29839961s.json"
    print(f"Loading data from (name=\"{name}\")")

    with open(name, 'r') as file:
        data = json.load(file)

    freq = np.array(data['nu'])
    S = np.array(data['S'])
    sigma = np.array(data['sigma'])
    order = 2
    nu0 = data['nu0']

    spline_model, var_names = get_spline_model(name, freq, S, sigma, nu0=nu0,
                                               order=order)

    RANDOM_SEED = 123

    with spline_model:
        idata = pm.sample_prior_predictive()
        idata.extend(pm.sample(draws=3000, tune=3000,
                               random_seed=RANDOM_SEED,
                               chains=4))

    samples = get_posterior_samples(idata, 1000)
    x = np.log(freq/nu0)
    x_curve = np.linspace(x[0], x[-1], 1000)
    y_scale = 1000

    n_samples = samples['n_samples']

    with plt.rc_context({"axes.grid": True,
                         "axes.grid.which": "both",
                         "axes.formatter.min_exponent": 2}):

        fig, ax = plt.subplots(figsize=(6, 4), layout='constrained')
        ax.set_xscale("log", nonpositive='clip')
        ax.set_yscale("log", nonpositive='clip')

        ax.set_xlabel("Frequency (GHz)")
        ax.set_ylabel("Flux (mJy)")

        for i in range(n_samples):

            try:
                k = samples['k']
                m = samples['m']
                cps = samples['cps']
                delta = samples['delta']

                logS = evaluate_spline(x_curve, cps, k, m, delta)
            except Exception as e:
                print(f"Exception {e.message}")
                logS = k*x_curve + m

            ax.plot(np.exp(x_curve), np.exp(logS)*y_scale,
                    alpha=10/n_samples, color='k', linewidth=3)
        ax.errorbar(freq/nu0, S*y_scale, yerr=sigma*y_scale,
                    fmt='o', label="data")
        ax.set_title(name)
        # ax.legend()
        plt.savefig(f"{name}_{n_samples}.pdf")
        plt.show()
