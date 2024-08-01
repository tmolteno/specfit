#!/usr/bin/env python
# coding: utf-8

# # Calculating Priors to be used for Calibration Source Spectra
# 
# Tim Molteno tim@elec.ac.nz
# 
# A mini project on calculating the uncertainties in the source spectral parameters. The Perley and Butler catalogs are typically 0.5 - 1.5% accurate (Oleg, Personal Communication). What I'd like to do for inference is decide on a prior for the spectral parameters $I_0$ and $\{a_i\}$.
# 
# 
# ## Part 2: Inference from original data
# 
# This is tricky. The original data is referred to in [CASA Docs https://casadocs.readthedocs.io/en/stable/notebooks/memo-series.html#Flux-Calibrator-Models]. However exactly how this is done for each source is a bit opaque.
# 
# In this part, we use the actual measurement data and estimates of likelihood to infer a source model with quantified uncertainty. We'll use the source J1939-6342, also known as 1934-638 in the B1950. The parameters of this source are determined (apparently) be references 1,3,4,5,6,8.
# 
# Standards are: 
# * (1) Perley-Butler 2010, 
# * (2) Perley-Butler 2013, 
# * (3) Perley-Taylor 99,   
# * (4) Perley-Taylor 95
# * (5) Perley 90, 
# * (6) Baars,
# * (7) Scaife-Heald 2012, 
# * (8) Stevens-Reynolds 2016 [https://ui.adsabs.harvard.edu/link_gateway/2016ApJ...821...61P/doi:10.3847/0004-637X/821/1/61]
# 
# The real source of data for J1939-6342 appears to be Reynolds [Reynolds, J. E. A revised flux scale for the AT Compact Array. AT Memo 39.3/040, 1994. https://archive-gw-1.kat.ac.za/public/meerkat/A-Revised-Flux-Scale-for-the-AT-Compact-Array.pdf] in the range 408MHz - 8.6GHz. And from [Partridge, B., et al. "Absolute calibration of the radio astronomy flux density scale at 22 to 43 GHz using Planck." The Astrophysical Journal 821.1 (2016): 61.] in the higher frequency ranges.
# 

# In[3]:


import numpy as np
import matplotlib.pyplot as plt
import json
import os
import arviz as az
from specfit import inference
from specfit import posterior_helper
import h5py
import specfit


# The following coefficients are from J.E. Reynolds.

# In[2]:


src = { 'name': 'J1939-6342',
        'nu0': 1e6,
        'a': [-30.7667, 26.4908, -7.0977, 0.605334]}
name = src['name']
nu0 = src['nu0']
a_array = src['a']


# In[3]:


def log_flux(w, logw, a):
    logS = a[0]
    for i in range(1,len(a)):
        logS += a[i]*logw**i
        
    return logS

def flux(nu, a, nu0):
    w = nu/nu0
    logw = np.log(w)
    
    logS = log_flux(w, logw, a)
    return np.exp(logS)


# In[4]:


fig, ax0 = plt.subplots(nrows=1, sharex=True)
nu = np.linspace(0.5e9, 10e9, 100)

f = flux(nu, src['a'],src['nu0'])
ax0.loglog(nu/1e9, f, '-', label=src['name'])
ax0.grid(True)
# ax0.yaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
#ax0.set_xscale("log", nonpositive='clip')
#ax0.set_yscale("log", nonpositive='clip')
ax0.set_xlabel("Frequency (GHz)")
ax0.set_ylabel("Flux (Jy)")
plt.title(f"Polynomial Model Source Spectrum")
ax0.legend();
plt.tight_layout()
plt.savefig('source_original_spectrum.pdf')


# ## Original Data
# 
# We'll reconstruct the fit from the original data of Reynolds (1994). A discussion of this data is avaialable [here](https://www.narrabri.atnf.csiro.au/observing/users_guide/html_old_20090512/Flux_Density_Measurements.html)

# In[5]:


import pymc as pm
import pytensor.tensor as tt


# In[6]:


default_percent_uncertainty = 5.0

# AT Compact Array using the calibrator 3C286 as the reference. 
# The total flux was estimated by using only the shortest interferometer spacings or
# by extrapolating the flux density to zero spacing (eg McConnell & McKay 1994). 
d_atca = default_percent_uncertainty / 100

# Single dish using parkes and Hydra A (occasionally Virgo A as reference)
d_parkes = default_percent_uncertainty / 100 
d_jer = default_percent_uncertainty / 100 ## J.E. Reynolds 1994 Tidbinbilla single dish using parkes and Hydra A (occasionally Virgo A as reference)

# The Molonglo Reference Catalogue (MRC) at 408MHz is on the scale of Wyllie (1969) 
d_mrc = default_percent_uncertainty / 100 

# MOST (Campbell-Wilson & Hunstead 1994) The MOST flux scale at 843MHz is essentially an 
# interpolation between the Wyllie 408MHz scale and the scale of the Parkes Catalogue.
# See Hunstead (1991) for further details.
d_most = default_percent_uncertainty / 100 

orig_measured_data = np.array(
                [[0.408, 6.24, d_mrc],
                [0.843, 13.65, d_most],
                [1.380, 14.96, d_atca],
                [1.413, 14.87, d_parkes],
                [1.612, 14.47, d_parkes],
                [1.660, 14.06, d_parkes],
                [1.665, 14.21, d_jer],
                [2.295, 11.95, d_jer],
                [2.378, 11.75, d_atca],
                [4.800, 5.81, d_atca],
                [4.800, 5.76, d_atca],
                [4.835, 5.72, d_atca],
                [4.850, 5.74, d_parkes],
                [8.415, 2.99, d_atca],
                [8.420, 2.97, d_jer],
                [8.640, 2.81, d_atca],
                [8.640, 2.81, d_atca]])

orig_measured_data = orig_measured_data.T


# In[46]:


nu = orig_measured_data[0]*1e9
measured_data = orig_measured_data[1]
percent = orig_measured_data[2]
sigma = measured_data*percent

new_d = np.array([nu/1e9, measured_data, sigma]).T
#print(f"Original Data:\n{np.array2string(new_d, separator=', ', precision=4)}")

min_freq = nu[0]
max_freq = nu[-1]

#nu, measured_data


# In[8]:


fig, ax0 = plt.subplots(nrows=1, sharex=True, figsize=(8,6))
ax0.set_xscale("log", nonpositive='clip')
ax0.set_yscale("log", nonpositive='clip')
ax0.errorbar(nu/1e9, measured_data, yerr=sigma, fmt='.')
ax0.grid(True)
ax0.set_title('Flux Measurements J1939-6342')
ax0.set_xlabel("Frequency (GHz)")
ax0.set_ylabel("Flux (Jy)")
fig.tight_layout()
plt.savefig('j1939_measurements.pdf')


# ## Choosing priors
# 
# These need to be chosen carefully. I0 is the value of the flux at $\nu = \nu_0$, and in this case we choose $\nu_0$ to be one of the data points close to the usual choice.

# In[9]:


nu0 = 1.0e9 # nu[3];


order=4
with pm.Model() as model:
    
    # Choose priors
    a_1  = [ pm.Normal(f"a[{i}]", mu=0, sigma=2.5) for i in range(order) ]
    print(a_1)
    brightness_1 = flux(nu, a_1, nu0)

    # likelihood_1 = pm.Normal("likelihood", mu=brightness_1, sigma=sigma, observed=measured_data)
    likelihood_1 = pm.StudentT("likelihood", nu=5, mu=brightness_1, sigma=sigma, observed=measured_data)


# In[10]:


with model:
    prior_checks = pm.sample_prior_predictive(samples=50, return_inferencedata=False)

print("test")
print(prior_checks.keys())
_, ax = plt.subplots()

ax.set_xscale("log", nonpositive='clip')
ax.set_yscale("log", nonpositive='clip')

for a0, a1, a2, a3 in zip( prior_checks["a[0]"],
                   prior_checks["a[1]"],
                   prior_checks["a[2]"],
                   prior_checks["a[3]"]
                             ):
    b = flux(nu, [a0, a1, a2, a3], nu0)
    ax.plot(nu/1e9, b, c="k", alpha=0.4)
ax.plot(nu/1e9, measured_data, c="b", alpha=1, label="J1939-6342")

ax.set_xlabel("Frequency (GHz)")
ax.set_ylabel("Flux (Jy)")
ax.set_title("Prior predictive checks");
ax.grid(True)
ax.legend()
plt.tight_layout()
plt.savefig("original_prior_predictive.pdf")


# Now sample from the model, producing a chain...

# In[11]:


idata_j1939 = inference.run_or_load(model, fname = "idata_j1939.nc",
                         n_samples = 5000, n_tune=3000, n_chains=4)


# ## Results
# 
# 

# In[12]:


stats = pm.summary(idata_j1939)
print(stats.to_string())


# In[13]:


az.plot_pair(
        idata_j1939,
        var_names=['a['],
        kind="hexbin",
        filter_vars="like",
        marginals=True,
        figsize=(12, 12),
    );
plt.savefig('posterior_pairs_J1939.pdf')


# The covariance of the parameters is clearly significant. We can estimate this numerically.

# In[14]:


a_cov, a_corr, names = posterior_helper.chain_covariance(idata_j1939.posterior)
np.set_printoptions(precision=4, suppress=False)
a_cov
stats, names = posterior_helper.get_stats(idata_j1939.posterior)

with h5py.File('calibrator_catalogue.hdf5', 'a') as f:
    grp = f.require_group("J1939-6342")
    ds_mean = grp.create_dataset("mean", data=stats[0])
    ds_sdev = grp.create_dataset("sdev", data=stats[1])
    ds_cov = grp.create_dataset("cov", data=a_cov)
    ds_cor = grp.create_dataset("corr", data=a_corr)

# In[15]:

ofile = open("j1939.tex", 'w')

posterior_helper.full_column(outfile=ofile, all_names=["J1939-6342"], idata=idata_j1939, freq=nu)  # outfile, all_names, idata, freq


# ### Posterior Predictive Sampling
# 
# Sanity check to see that we are producing reasonable spectra.

# In[18]:


RANDOM_SEED=123
with model:
    ppc = pm.sample_posterior_predictive(
        idata_j1939, random_seed=RANDOM_SEED, return_inferencedata=False
    )


# In[19]:


#ppc
#az.plot_ppc(az.from_pymc3(posterior_predictive=ppc, model=model));


# In[20]:


_, ax = plt.subplots()
ax.plot(nu, measured_data, "o", ms=4, alpha=0.4, label="Data")
az.plot_hdi(
    nu,
    ppc["likelihood"],
    ax=ax,
    fill_kwargs={"alpha": 0.8, "color": "#a1dab4", "label": "Outcome 94% HDI"},
)

ax.set_xlabel("Frequency (GHz)")
ax.set_ylabel("Flux (Jy)")
ax.set_title("Posterior predictive checks")
ax.legend(ncol=2, fontsize=10);
plt.savefig('posterior_predictive_j1939.pdf')


# ## Validation against usual fit
# 
# Now have a look at the new spectral fit, taken by using the mean of the posterior parameters, against the polynomial of Reynolds (1994).

# In[21]:


a = [ idata_j1939.posterior.mean().get(f"a[{i}]").values.tolist() for i in range(order)]


# In[22]:


print(f"The mean value nu0 = {nu0} and a={a}, i0={10**a[0]}")


# Now form the spectrum from the mean posterior values and compare it to the Reynolds model (using the Perley butler source spectral characteristics)

# In[23]:


nu_new = np.linspace(0.4e9, 10e9, 1000)
b_new = flux(nu_new, a, nu0)
b_pb = flux(nu_new, src['a'],src['nu0'])


# In[24]:


_, ax = plt.subplots()

ax.set_xscale("log", nonpositive='clip')
ax.set_yscale("log", nonpositive='clip')
ax.plot(nu_new, b_pb, c="k", alpha=0.4, label="Reynolds (1994)")
ax.plot(nu_new, b_new, '-.', c="g", alpha=1, label="This work")
ax.plot(nu_new, np.abs(b_new-b_pb), c="r", alpha=1, label='difference $\\Delta S$')
ax.plot(nu, measured_data, 'o', c="b", alpha=1, label="Measurements")

ax.legend()
ax.set_xlabel("Frequency (GHz)")
ax.set_ylabel("Flux (Jy)")
ax.set_ylim((0.001, 20))
ax.grid(True)
ax.set_title("Comparison of Models");
plt.savefig('model_comparison.pdf')


# In[25]:


ds =(np.abs(b_new - b_pb))
np.percentile(ds, [0.05, 0.5, 0.95, 0.99, 1])


# In[26]:


import matplotlib as mpl

def mean_val(key):
    return idata_j1939.posterior[key].mean().values.tolist()

a_mean = [mean_val(f"a[{i}]") for i in range(order)]


fig, ax = plt.subplots()


ax.set_xscale("log", nonpositive='clip')
ax.set_yscale("log")
for n in np.geomspace(min_freq, max_freq, 50):
    for i in range(100):
        a = posterior_helper.get_random_sample(idata_j1939)
        b_test = flux(n, a, nu0)

        if False:
            b_ref = flux(n, a_mean, nu0)
            ax.plot(n/1e9, 1000*(b_ref-b_test), '.', c="k", alpha=0.05)
        else:
            ax.plot(n/1e9, b_test, '.', c="k", alpha=0.05)

ax.set_xlabel("Frequency (GHz)")
ax.set_ylabel("Flux (Jy))")
ax.set_title("Posterior Spectrum PDF (J1939-6342)")
ax.grid(True);
fig.tight_layout()
plt.savefig('posterior_spectrum_j1939.pdf')


# In[27]:


## Now sample from a multivariate gaussian of the polynomial model
order=4
mean = [ idata_j1939.posterior.mean().get(f"a[{i}]").values.tolist() for i in range(order)]
a_cov, a_corr, names = posterior_helper.chain_covariance(idata_j1939.posterior)
print(mean)
print(a_cov)
samples_cov = np.random.multivariate_normal(mean, a_cov, 100)

diag_cov = np.zeros_like(a_cov)
diag_cov[np.diag_indices(a_cov.shape[0])] = np.diag(a_cov)
print(diag_cov)
samples_diag = np.random.multivariate_normal(mean, diag_cov, 100)
fig, ax = plt.subplots(2, sharex=True, sharey=True, figsize=(7,7))
ax0, ax1 = ax

ax0.set_xscale("log", nonpositive='clip')
ax0.set_yscale("log")

fifth = []
fiftieth = []
ninetyfifth = []

nu = np.geomspace(min_freq, max_freq, 50)

for n in nu:
    fluxes = np.array([flux(n, a, nu0) for a in samples_cov])
    for s in fluxes:
        ax0.plot(n/1e9, s, '.', c="k", alpha=0.05)
ax0.set_title("Posterior with covariance")
ax0.grid(True);
ax0.set_ylabel("Flux (Jy))")

for n in nu:
    fluxes = np.array([flux(n, a, nu0) for a in samples_diag])
    for s in fluxes:
        ax1.plot(n/1e9, s, '.', c="k", alpha=0.05)
ax1.set_title("Posterior without covariance")
ax1.grid(True);
ax1.set_xlabel("Frequency (GHz)")
ax1.set_ylabel("Flux (Jy))")
        
#    b_test = flux(n, a, nu0)

#        if False:
#            b_ref = flux(n, a_mean, nu0)
#            ax.plot(n/1e9, 1000*(b_ref-b_test), '.', c="k", alpha=0.05)
#        else:
#            ax.plot(n/1e9, b_test, '.', c="k", alpha=0.1)

#fig.set_title("Posterior Spectrum PDF (J1939-6342)")
fig.tight_layout()
plt.savefig('posterior_spectrum_model_j1939.pdf')


# ## Discussion
# 
# The new polynomial fit is fairly accurate, within 50 $\mu$Jy. Of more importance is that the uncertainty in the polynomial coefficients can be quantified, and mapped through to uncertainty in the spectrum. This allows the parameters of the model to be used as priors for a Bayesian calibration.

# ## Northern Sky Calibrators
# 
# Table 9
# Adopted Spectral Flux Densities of Steady Sources
# 
# Notes. The derived flux density values, based on the Mars emission model for frequencies between 1 and 50 GHz. The quoted errors are derived from the dispersion of the values over the various sessions. The values for 3C123, 3C286, and 3C295 at 327.5 MHz are derived from their ratios to 3C196, whose flux density is taken to be 46.8 Jy (Scaife & Heald 2012).
# 

# In[28]:


# Freq., 3C123, , 3C196, , 3C286, , 3C295, , N_obs
# (GHz), S(Jy), sigma(Jy), S(Jy), sigma(Jy), S(Jy), sigma(Jy), S(Jy), sigma(Jy)

csv_data = '''0.3275, 145.0, 4.3, 46.8, 1.4, 26.1, 0.8, 60.8, 1.8, 14
1.015, 66.2, 4.3, 20.1, 4.8, 18.4, 4.3, 30.8, 7.3, 1
1.275, 46.6, 3.2, 13.3, 2.0, 13.8, 2.0, 21.5, 3.0, 1
1.465, 47.8, 0.5, 14.1, 0.2, 15.0, 0.2, 22.2, 0.5, 4
1.865, 38.7, 0.6, 11.3, 0.2, 13.2, 0.2, 17.9, 0.3, 2
2.565, 28.9, 0.3, 8.16, 0.1, 10.9, 0.2, 12.8, 0.2, 2
3.565, 21.4, 0.8, 6.22, 0.2, 9.5, 0.1, 9.62, 0.2, 3
4.535, 16.9, 0.2, 4.55, 0.06, 7.68, 0.1, 6.96, 0.09, 7
4.835, 16.0, 0.2, 4.22, 0.1, 7.33, 0.2, 6.45, 0.15, 1
4.885, 15.88, 0.1, 4.189, 0.025, 7.297, 0.046, 6.37, 0.04, 11
6.135, 12.81, 0.15, 3.318, 0.05, 6.49, 0.15, 4.99, 0.05, 2
6.885, 11.20, 0.14, 2.85, 0.05, 5.75, 0.05, 4.21, 0.05, 3
7.465, 11.01, 0.2, 2.79, 0.05, 5.70, 0.10, 4.13, 0.07, 1
8.435, 9.20, 0.04, 2.294, 0.010, 5.059, 0.021, 3.319, 0.014, 11
8.485, 9.10, 0.15, 2.275, 0.03, 5.045, 0.07, 3.295, 0.05, 1
8.735, 8.86, 0.05, 2.202, 0.011, 4.930, 0.024, 3.173, 0.016, 10
11.06, 6.73, 0.15, 1.64, 0.03, 4.053, 0.08, 2.204, 0.05, 1
12.890, , , 1.388, 0.025, 3.662, 0.070, 1.904, 0.04, 1
14.635, 5.34, 0.05, 1.255, 0.020, 3.509, 0.040, 1.694, 0.04, 1
14.715, 5.02, 0.05, 1.206, 0.020, 3.375, 0.040, 1.630, 0.03, 1
14.915, 5.132, 0.025, 1.207, 0.004, 3.399, 0.016, 1.626, 0.008, 7
14.965, 5.092, 0.028, 1.198, 0.007, 3.387, 0.015, 1.617, 0.007, 11
17.422, 4.272, 0.07, 0.988, 0.02, 2.980, 0.04, 1.311, 0.025, 1
18.230, , , 0.932, 0.020, 2.860, 0.045, 1.222, 0.05, 1
18.485, 4.090, 0.055, 0.947, 0.015, 2.925, 0.045, 1.256, 0.020, 1
18.585, 3.934, 0.055, 0.926, 0.015, 2.880, 0.04, 1.221, 0.015, 1
20.485, 3.586, 0.055, 0.820, 0.010, 2.731, 0.05, 1.089, 0.015, 1
22.460, 3.297, 0.022, 0.745, 0.003, 2.505, 0.016, 0.952, 0.005, 13
22.835, 3.334, 0.06, 0.760, 0.010, 2.562, 0.05, 0.967, 0.015, 1
24.450, 2.867, 0.03, 0.657, 0.017, 2.387, 0.03, 0.861, 0.020, 2
25.836, 2.697, 0.06, 0.620, 0.017, 2.181, 0.06, 0.770, 0.02, 1
26.485, 2.716, 0.05, 0.607, 0.017, 2.247, 0.05, 0.779, 0.020, 1
28.450, 2.436, 0.06, 0.568, 0.015, 2.079, 0.05, 0.689, 0.020, 2
29.735, 2.453, 0.05, 0.529, 0.015, 2.011, 0.05, 0.653, 0.020, 1
36.435, 1.841, 0.17, 0.408, 0.005, 1.684, 0.02, 0.484, 0.015, 3
43.065, , , 0.367, 0.015, 1.658, 0.08, 0.442, 0.020, 1
43.340, 1.421, 0.055, 0.342, 0.005, 1.543, 0.024, 0.398, 0.006, 13
48.350, 1.269, 0.12, 0.289, 0.005, 1.449, 0.04, 0.359, 0.013, 4
48.565, , , 0.272, 0.015, 1.465, 0.1, 0.325, 0.025, 1
'''

'''
# In[29]:


import csv
import io
import numpy as np
buff = io.StringIO(csv_data)

def parse(x):
    try:
        return float(x)
    except:
        return None
    
reader = csv.reader(buff)
lines = []
for row in reader:
    
    line = [parse(x) for x in row]
    lines.append(line)


def cleanup(line, sigma, freq):
    good = np.where(np.isnan(line) == False)
    return line[good], sigma[good], freq[good]

lines = np.array(lines, dtype=np.float64).T
frequency = lines[0]*1e9
S_3C123 = lines[1]
sigma_3C123 = lines[2]

S_3C196 = lines[3]
sigma_3C196 = lines[4]

S_3C286 = lines[5]
sigma_3C286 = lines[6]

S_3C295 = lines[7]
sigma_3C295 = lines[8]

min_freq = frequency[0]
max_freq = frequency[-1]

S_3C123, sigma_3C123, f_3C123 = cleanup(S_3C123, sigma_3C123, frequency)


# In[30]:



def dataplot(name, freq, mu, sigma):
    fig, ax = plt.subplots()

    ax.set_xscale("log", nonpositive='clip')
    ax.set_yscale("log", nonpositive='clip')
    ax.errorbar(freq/1e9, mu, yerr=sigma, fmt='.', label="Perley & Butler")

    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel("Flux (Jy)")
    ax.grid(True)
    ax.set_title(f"Flux Measurements: {name}");
    fig.tight_layout()
    plt.savefig(f"source_{name}_spectrum.pdf")
    
def spectral_inference(name, freq, mu, sigma, order, nu0):
    dataplot(name, freq, mu, sigma)
    with pm.Model() as _model:
        _a  = [ pm.Normal(f"a[{i}]", mu=0, sigma=2.5) for i in  range(order) ]
        _brightness = flux(freq, _a, nu0)
        _likelihood = pm.Normal("likelihood", mu=_brightness, sigma=sigma, observed=mu)
    _idata = posterior_helper.run_or_load(_model, fname = f"idata_{name}.nc")
    
    posterior_helper.full_column(name, _idata, freq)

    
S_3C286, f_3C123


# In[31]:


nu0 = 1e9
spectral_inference("3C123", f_3C123, S_3C123, sigma_3C123, order=4, nu0=nu0)


# In[32]:


spectral_inference("3C196", frequency, S_3C196, sigma_3C196, order=4, nu0=nu0)


# In[33]:


import matplotlib.pyplot as plt

fig, ax = plt.subplots()

ax.set_xscale("log", nonpositive='clip')
ax.set_yscale("log", nonpositive='clip')
ax.errorbar(frequency/1e9, S_3C286, yerr=sigma_3C286, fmt='.', label="Perley & Butler")

#ax.legend()
ax.set_xlabel("Frequency (GHz)")
ax.set_ylabel("Flux (Jy)")
ax.grid(True)
ax.set_title("Flux Measurements 3C286");
fig.tight_layout()
plt.savefig('source_3c286_spectrum.pdf')


# In[34]:


nu0 = 1e9
order=4

    
with pm.Model() as model_3c286:
    
    a_1  = [ pm.Normal(f"a[{i}]", mu=0, sigma=2.5) for i in  range(order) ]

    brightness_1 = flux(frequency, a_1, nu0)

    likelihood_1 = pm.Normal("likelihood", mu=brightness_1, sigma=sigma_3C286, observed=S_3C286)
    
with pm.Model() as model_3c295:
    
    a_1  = [ pm.Normal(f"a[{i}]", mu=0, sigma=2.5) for i in  range(order) ]

    brightness_1 = flux(frequency, a_1, nu0)

    likelihood_1 = pm.Normal("likelihood", mu=brightness_1, sigma=sigma_3C295, observed=S_3C295)


# In[35]:


idata_3c286 = posterior_helper.run_or_load(model_3c286, fname = "idata_3c286.nc")
idata_3c295 = posterior_helper.run_or_load(model_3c295, fname = "idata_3c295.nc")


# In[36]:


stats = pm.summary(idata_3c286)
print(stats.to_string())


# In[37]:


stats = pm.summary(idata_3c295)
print(stats.to_string())


# In[38]:


az.plot_pair(
        idata_3c286,
        var_names=['a['],
        kind="hexbin",
        filter_vars="like",
        marginals=True,
        figsize=(12, 12),
    );
plt.savefig('posterior_pairs_3c286.pdf')


# In[39]:


posterior_helper.idata_2_latex(idata_3c286)


# This should be compared to the values provided by Perley & Butler (2017) of
# 
# 1.2515 ± 0.0048 	−0.4605 ± 0.0163 	−0.1715 ± 0.0208 	0.0336 ± 0.0082

# In[40]:


a_cov, a_corr, names = posterior_helper.chain_covariance(idata_3c286)
np.set_printoptions(precision=4, suppress=False)
print("Covariance")
print(a_cov)
print("Correlation")
print(a_corr)


# In[41]:

ofile = open("3C286.tex", 'w')

posterior_helper.full_column(ofile, "3C286", idata_3c286, frequency)


# In[42]:


posterior_helper.full_column(ofile, "3C295", idata_3c295, frequency)


# In[43]:


a_cov, a_corr, names = posterior_helper.chain_covariance(idata_3c286)


# In[44]:


def mean_val(key):
    return idata_3c286.posterior[key].mean().values.tolist()

a_mean = [mean_val(f"a[{i}]") for i in range(order)]


_, ax = plt.subplots()


ax.set_xscale("log", nonpositive='clip')
ax.set_yscale("log")
for n in np.geomspace(min_freq, max_freq, 50):
    for i in range(100):
        a = posterior_helper.get_random_sample(idata_3c286)
        b_test = flux(n, a, nu0)

        if False:
            b_ref = flux(n, a_mean, nu0)
            ax.plot(n/1e9, 1000*(b_ref-b_test), '.', c="k", alpha=0.05)
        else:
            ax.plot(n/1e9, b_test, '.', c="k", alpha=0.1)

ax.set_xlabel("Frequency (GHz)")
ax.set_ylabel("Flux (Jy))")
ax.set_title("Posterior Spectrum PDF (3C286)")
ax.grid(True);
plt.savefig('posterior_spectrum_3c286.pdf')


# ## 

'''
