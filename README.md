# Specfit 

Infer the polynomial coefficients and their covariance structure for fitting radio-astronometric callibrator spectrum.

Author: Tim Molteno. tim@elec.ac.nz

## Install

    sudo pip3 install specfit

## Examples

Here is an example. This code is in the examples directory.

    import numpy as np
    import specfit as sf
    import matplotlib.pyplot as plt

    # Data from J.E. Reynolds for J1939-6342
    freq = [4.080e+08,8.430e+08,1.380e+09,1.413e+09,1.612e+09,1.660e+09,1.665e+09,
            2.295e+09,2.378e+09,4.800e+09,4.800e+09,4.835e+09,4.850e+09,8.415e+09,
            8.420e+09,8.640e+09,8.640e+09]
    data = [ 6.24,13.65,14.96,14.87,14.47,14.06,14.21,11.95,11.75, 5.81,
            5.76, 5.72, 5.74, 2.99, 2.97, 2.81, 2.81]
    sigma = [0.312 ,0.6825,0.748 ,0.7435,0.7235,0.703 ,0.7105,0.5975,0.5875,0.2905,
            0.288 ,0.286 ,0.287 ,0.1495,0.1485,0.1405,0.1405]

    names, stats, a_cov, a_corr = sf.spectral_inference("J1939-6342", freq=nu, mu=data, sigma=sigma, order=4, nu0=1.4e9)

Now we can plot the data and show the results.

    fig, ax = sf.dataplot(plt, "J1939-6342", freq=freq, mu=data, sigma=sigma)

    a = stats[0] # Means

    nu = np.linspace(min_freq, max_freq, 100)
    S = sf.flux(nu, a, nu0=1.4e9)
    ax.plot(nu/1e9, S, label="polynomial fit")
    ax.legend()
    fig.tight_layout()
    plt.show()

    print(names, stats)
    print(a_cov)


## TODO

- Incorporate some ideas on using variances of parameters and constraints on flux uncertainties in place of requiring an explicit assumption of the sigma (in the case of data-free inference)

## Changelog

- 0.1.0b3 First functioning release.
