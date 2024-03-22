from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm

from patsy import dmatrix

RANDOM_SEED = 8927
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

x_data = np.log(freq/nu0)
y_data = np.log(S)
y_err = 0.01

num_knots = 1
knot_list = [x_data[0], x_data[np.argmax(S)], x_data[-1]]
# knot_list = np.linspace(x_data[0], x_data[-1], num_knots, endpoint=True)
print(f"Knot List: {knot_list}")

spline_design = "bs(freq, df=4, degree=2, include_intercept=True) - 1"
test_freq = np.linspace(x_data[0], x_data[-1], 100)
B = dmatrix(
    spline_design,
    {"freq": test_freq, "knots": knot_list[1:-1]},
)

# Check out the splines

spline_df = (
    pd.DataFrame(B)
    .assign(freq=test_freq)
    .melt("freq", var_name="spline_i", value_name="value")
)

color = plt.cm.magma(np.linspace(0, 0.80, len(spline_df.spline_i.unique())))

fig = plt.figure()
for i, c in enumerate(color):
    subset = spline_df.query(f"spline_i == {i}")
    subset.plot("freq", "value", c=c, ax=plt.gca(), label=i)
plt.legend(title="Spline Index", loc="upper center", fontsize=8, ncol=6);
plt.show()

B = dmatrix(
    spline_design,
    {"freq": x_data, "knots": knot_list[1:-1]},
)

COORDS = {"splines": np.arange(B.shape[1]),
          "obs": range(len(y_data))
          }
with pm.Model(coords=COORDS) as spline_model:
    # a = pm.Normal("a", 0, 3)
    w = pm.Normal("w", mu=0, sigma=2, size=B.shape[1], dims="splines")
    # mu = pm.Deterministic("mu", a + pm.math.dot(np.asarray(B, order="F"), w.T))
    # sigma = pm.Exponential("sigma", 1)
    mu = pm.Deterministic("mu",  pm.math.dot(np.asarray(B, order="F"), w.T))
    D = pm.Normal("D", mu=mu, sigma=y_err, observed=y_data, dims="obs")


pm.model_to_graphviz(spline_model)
plt.show()
# 
with spline_model:
    idata = pm.sample_prior_predictive()
    idata.extend(pm.sample(draws=1000, tune=3000, random_seed=RANDOM_SEED, chains=4))
    pm.sample_posterior_predictive(idata, extend_inferencedata=True)
    
print(idata.keys())
print(az.summary(idata, var_names=[ "w"]))

az.plot_trace(idata, var_names=[ "w"]);
plt.show()


wp = idata.posterior["w"].mean(("chain", "draw")).values

spline_df = (
    pd.DataFrame(B * wp.T)
    .assign(freq=x_data)
    .melt("freq", var_name="spline_i", value_name="value")
)

spline_df_merged = (
    pd.DataFrame(np.dot(B, wp.T))
    .assign(freq=x_data)
    .melt("freq", var_name="spline_i", value_name="value")
)


color = plt.cm.rainbow(np.linspace(0, 1, len(spline_df.spline_i.unique())))
fig = plt.figure()
for i, c in enumerate(color):
    subset = spline_df.query(f"spline_i == {i}")
    subset.plot("freq", "value", c=c, ax=plt.gca(), label=i)
spline_df_merged.plot("freq", "value", c="black", lw=2, ax=plt.gca())
plt.legend(title="Spline Index", loc="lower center", fontsize=8, ncol=6)

plt.plot(x_data, y_data, 'o')
for knot in knot_list:
    plt.gca().axvline(knot, color="grey", alpha=0.4);

plt.show()
