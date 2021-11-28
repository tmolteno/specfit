import numpy as np
import specfit as sf
import matplotlib.pyplot as plt

# Data from J.E. Reynolds for J1939-6342
freq = [4.080e+08,8.430e+08,1.380e+09,1.413e+09,1.612e+09,1.660e+09,1.665e+09,
        2.295e+09,2.378e+09,4.800e+09,4.800e+09,4.835e+09,4.850e+09,8.415e+09,
        8.420e+09,8.640e+09,8.640e+09]
data = [6.24,13.65,14.96,14.87,14.47,14.06,14.21,11.95,11.75, 5.81,
        5.76, 5.72, 5.74, 2.99, 2.97, 2.81, 2.81]
sigma = [0.312 ,0.6825,0.748 ,0.7435,0.7235,0.703 ,0.7105,0.5975,0.5875,0.2905,
         0.288 ,0.286 ,0.287 ,0.1495,0.1485,0.1405,0.1405]
nu0 = 1.4e9

fig, ax = sf.dataplot(plt, "J1939-6342", freq=freq, mu=data, sigma=sigma)

names, stats, a_cov, a_corr = sf.data_inference("J1939-6342", freq=freq, mu=data, sigma=sigma, order=5, nu0=nu0)

a = stats[0] # Means

nu = np.linspace(freq[0], freq[-1], 100)
S = sf.flux(nu, a, nu0=nu0)
ax.plot(nu/1e9, S, label="polynomial fit")
ax.legend()
fig.tight_layout()
plt.show()

print(f"Variables: {names}")
print(f"Means: {stats[0]}")
print(f"SDev: {stats[1]}")
print(f"Covariance Matrix:\n{np.array2string(a_cov, separator=',', precision=4)}")
print(f"Correlation Matrix:\n{np.array2string(a_corr, separator=',', precision=4)}")


## Now do polynomial inference from the data again.

names2, stats2, a_cov2, a_corr2, f, fake_data = sf.datafree_inference('J1939-6342-poly', freq_min=freq[0], freq_max=freq[-1], nfreq=20, sigma=0.5, a=stats[0], nu0=nu0)

fig, ax = plt.subplots()
a = stats2[0] # Means
ax.set_xscale("log", nonpositive='clip')
ax.set_yscale("log", nonpositive='clip')
ax.errorbar(f/1e9, fake_data, yerr=0.5, fmt='.', label="Fake Data")

ax.set_xlabel("Frequency (GHz)")
ax.set_ylabel("Flux (Jy)")
ax.grid(True)

nu = np.linspace(f[0], f[-1], 100)
S = sf.flux(nu, a, nu0=nu0)
ax.plot(nu/1e9, S, label="data-free polynomial fit")
ax.legend()
fig.tight_layout()
plt.show()

print(f"Variables: {names2}")
print(f"Means: {stats2[0]}")
print(f"SDev: {stats2[1]}")
print(f"Covariance Matrix:\n{np.array2string(a_cov2, separator=',', precision=4)}")
print(f"Correlation Matrix:\n{np.array2string(a_corr2, separator=',', precision=4)}")
