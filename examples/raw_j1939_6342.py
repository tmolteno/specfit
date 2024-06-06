import numpy as np
import specfit as sf
import matplotlib.pyplot as plt
import sympy as sp

# Data from J.E. Reynolds for J1939-6342
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
     [8.64 ,  2.81, 0.1405],
     [8.64 ,  2.81, 0.1405]])

freq_ghz, mu, sigma = original_data.T
freq = freq_ghz*1e9
nu0=1.0e9

if True:
    names, stats, a_cov, a_corr, idata, model = \
        sf.data_inference("J1939-6342",
            freq=freq, mu=mu, sigma=sigma,
            order=4, nu0=nu0, n_samples=5000)

    a = stats[0] # Means

    fig, ax = sf.dataplot(plt, "J1939-6342", freq=freq, mu=mu, sigma=sigma)
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

    post = idata.posterior
    print(post.keys())

    pp = sf.posterior_predictive_sampling(idata, model, 1000)
    print(pp.keys())

    x = sp.Symbol('x', real=True)
    polys = None
    for k in pp.keys():
        n = int(k[2])
        a = pp[k].values

        print(a.shape)
        n_samples = len(a)
        if polys is None:
            polys = [0]*n_samples

        for i in range(n_samples):
            polys[i] += a[i]*x**n

    peaks = []
    for p in polys:
        eqn = sp.Eq(sp.diff(p, x), 0)
        low, high = sp.solveset(eqn, x, domain=sp.S.Reals)

        f = nu0*sp.exp(low)
        print(f"{eqn} --> {low} --> {f}")
        peaks.append(sp.N(f))
        # logw = log(nu/nu0) = soln
        # --> nu = nu0*exp(soln)

    peaks = np.array(peaks, dtype=float)/1e9
    print(peaks.shape)
    plt.hist(peaks, bins='fd')
    plt.title("PDF of spectral peaks J1939-6342")
    plt.xlabel("Frequency (GHz)")
    plt.grid(True)
    plt.savefig("j1939_peaks.pdf")
    plt.show()
## Now do polynomial inference from the data again.

names2, stats2, a_cov2, a_corr2, f, fake_data = sf.datafree_inference('J1939-6342-poly', freq_min=freq[0], freq_max=freq[-1], nfreq=20, sigma=0.5, a=[ 2.69445071,  0.24791307, -0.71448904,  0.11324043], nu0=nu0)

print(fake_data)

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



