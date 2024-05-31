
import pymc as pm
import numpy as np
import arviz as az

import specfit as sf
import matplotlib.pyplot as plt
import sympy as sp

az.style.use("arviz-darkgrid")



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

order = 3

name = "j1939-6342"
sigma = delta_S

names, stats, a_cov, a_corr, idata, model = \
        sf.spline_inference(name, freq, S, sigma, order, nu0)

# spline_model = sf.get_spline_model(name, freq, mu, sigma, order, nu0=nu0)
# 
# with spline_model:
#     idata = pm.sample_prior_predictive()
#     idata.extend(pm.sample(draws=1000, tune=3000, random_seed=RANDOM_SEED, chains=4))
#     pm.sample_posterior_predictive(idata, extend_inferencedata=True)

print(idata.keys())
print(az.summary(idata, var_names=[ "w"]))

az.plot_trace(idata, var_names=[ "w"]);
plt.show()
plt.savefig(f"spline_trace.pdf")


## Pair Plots

az.plot_pair(
        idata,
        var_names=['w'],
        kind="hexbin",
        filter_vars="like",
        marginals=True,
        figsize=(12, 12),
    )
plt.tight_layout()
plt.savefig(f"spline_posterior_pairs.pdf")
plt.show()

#  Do some posterior sampling
sf.plot_spline_design(freq, order)

sf.plot_spline(idata, freq, S, nu0, order)


