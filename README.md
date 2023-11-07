# Specfit 

Infer polynomial coefficients and their covariance structure for fitting radio-astronometric callibrator spectra.

Author: Tim Molteno. tim@elec.ac.nz

## Install

    sudo pip3 install specfit

develop

    pip3 install -e .

## Examples

Here is an example. This code is in the examples directory.

    import numpy as np
    import specfit as sf
    import matplotlib.pyplot as plt

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

    names, stats, a_cov, a_corr, idata = \
        sf.spectral_inference("J1939-6342", 
            freq=nu, mu=data, sigma=sigma, order=4, nu0=1.4e9)

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
- Use smoothness as a prior (rather than model-order).

## Changelog

- 0.3.0b2 Use pymc and upgrade to newer versions.
- 0.2.0b4 Include a separate function (marginal_likelihood) for estimating the marginal likelihood using SMC
          Change the likelihood to use a Student's t distribution for robustness.
- 0.2.0b3 Fix examples, move to github automation for release information.
- 0.1.0b3 First functioning release.
- 0.1.0b4 [In progress] Add the frequency range to the full_column output.
            Return the inference data to allow further processing
            Improved plotting and postprocessing.
            Added posterior PDF helper plotting function (slow)
            Use different tuning depending on polynomial order
            Output to a file, including lists of alternate names
