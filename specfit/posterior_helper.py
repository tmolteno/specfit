import os
import re

import numpy as np
import arviz as az
import pymc as pm


import pytensor.tensor as T

import matplotlib.pyplot as plt

def log_flux(w, logw, a):
    logS = a[0]
    for i in range(1,len(a)):
        logS += a[i]*logw**i

    return logS

def flux(nu, a, nu0):
    """Calculate flux from a polynomial model

    Flux is modeled as a polynomial in log(nu/nu0).
    
    Parameters
    ----------
    nu : array or float
        The frequency(ies) at which to calculate the flux
    a : array-like
        The coefficients of the polynomial model
    nu0 : float
        The normalizing frequency
        
    Returns
    -------
    array-like or float
        The flux (in Jansky) at each of the supplied frequencies
    """
    w = nu/nu0
    logw = np.log10(w)

    logS = np.maximum(log_flux(w, logw, a), -500.0)
    
    return np.power(10, logS)

def Tflux(nu, a, nu0):
    """Calculate flux from a polynomial model

    Flux is modeled as a polynomial in log(nu/nu0).
    
    Parameters
    ----------
    nu : array or float
        The frequency(ies) at which to calculate the flux
    a : array-like
        The coefficients of the polynomial model
    nu0 : float
        The normalizing frequency
        
    Returns
    -------
    array-like or float
        The flux (in Jansky) at each of the supplied frequencies
    """
    w = nu/nu0
    logw = np.log10(w)

    logS = T.maximum(log_flux(w, logw, a), -500.0)
    
    return np.power(10, logS)


def row_2_table(outfile, row):
    print("    ", end='', file=outfile)
    cols = len(row)
    for j in range(cols):
        data = row[j]
        if isinstance(data, str):
            print(f"{data} ", end='', file=outfile)
        else:
            print("{:6.4g} ".format(data), end='', file=outfile)
        if j < cols-1:
            print("& ", end='', file=outfile)
    print("\\\\", file=outfile)

def clean_up_names(names):
    ret = []
    for n in names:
        # Replace x[x] with a_{x}
        newname = re.sub(r"a\[([0-9])\]", "a_{\\1}", n)
        ret.append(f"${newname}$")
    return ret

def matrix_2_latex(outfile, a, names):
    clean_names = clean_up_names(names)
    a = np.array(a)
    cols = a.shape[0]
    columns = ''
    for i in range(cols):
        columns = columns+'r'

    print(f"\\begin{{tabular}}{{{columns}}}", file=outfile)
    print("    \\hline", file=outfile)
    row_2_table(outfile, clean_names)
    print("    \\hline", file=outfile)
    for i in range(a.shape[1]):
        row_2_table(outfile, a[:,i])
    print("    \\hline", file=outfile)
    print("\\end{tabular}", file=outfile)

def vector_2_latex(outfile, a, names):
    '''
        print a vector as a vertival table
    '''
    clean_names = clean_up_names(names)
    a = np.array(a)
    cols = a.shape[0]
    columns = ''
    for i in range(cols):
        columns = columns+'r'

    print(f"\\begin{{tabular}}{{rrr}}", file=outfile)
    print(f"    $x$   & $\mu_x$ & $\sigma_x$   \\\\", file=outfile)
    print(f"    \\hline", file=outfile)
    for n, x in zip(clean_names, a):
        print(f"    {n} & {x[0] :6.4} & {x[1] :6.4}    \\\\", file=outfile)
    print("\\end{tabular}", file=outfile)


def get_stats(idata):
    names = [x for x in idata.posterior.mean()]
    ret = [ np.array([idata.posterior.get(x).mean().values.tolist(), idata.posterior.get(x).std().values.tolist()]) for x in names ]
    return np.array(ret).T, names

def idata_2_latex(idata):
    names = [x for x in idata.posterior.mean()]
    data = [ [idata.posterior.get(x).mean().values.tolist(), idata.posterior.get(x).std().values.tolist()] for x in idata.posterior.mean() ]
    matrix_2_latex(data, names)

def mean_2_latex(outfile, idata):
    names = [x for x in idata.posterior.mean()]
    data = [ [idata.posterior.get(x).mean().values.tolist(), idata.posterior.get(x).std().values.tolist()] for x in idata.posterior.mean() ]
    vector_2_latex(outfile, data, names)

def get_random_sample(idata, chain=0):
    names = [x for x in idata.posterior.mean()]
    n_samples = idata.posterior.get(names[0]).values.shape[1]
    sample = np.random.randint(0,n_samples)

    return [idata.posterior.get(n).values[chain, sample] for n in names]



def full_column(outfile, all_names, idata, freq):
    a_cov, a_corr, names = chain_covariance(idata)
    
    name_list = "\\begin{tabular}{l} " + all_names[0]
    for n in all_names[1:]:
        name_list = name_list + f" \\\\ {n} "
    name_list = name_list + " \\end{tabular}"
    
    print(f"{name_list} & {freq[0]/1e9 :4.2f}-{freq[-1]/1e9 :4.2f} & ", file=outfile)
    mean_2_latex(outfile, idata)
    print(f" & ", file=outfile)
    matrix_2_latex(outfile, a_cov, names)
    print(f" & ", file=outfile)
    matrix_2_latex(outfile, a_corr, names)
    print("  \\\\  \\hline", file=outfile)

def chain_covariance(idata):
    names = [x for x in idata.posterior.mean()]
    chain = 0
    a = [idata.posterior.get(n)[chain, :] for n in names]
    a = np.array(a)
    return np.cov(a), np.corrcoef(a), names




def get_gain(idata, i, chain, sample, use_phase=False):
    if i == 0:
        return 0.75;  # The set value of the ant0 gain and phase

    if use_phase:
        g_i = idata.posterior[f"gain_{i}"][chain, sample]
        p_i = idata.posterior[f"phase_{i}"][chain, sample]
        return g_i * np.exp(1j*p_i)
    else:
        a = idata.posterior[f"re_{i}"][chain, sample]
        b = idata.posterior[f"im_{i}"][chain, sample]
        return a + 1j*b


def correction_matrix(idata, chain=0, sample=700):
    N = 57
    gains = np.array([get_gain(idata, i, chain, sample) for i in range(0, N+1)])

    G = np.zeros((N+1,N+1), dtype=np.complex64)
    for i in range(0, N+1):
        gi = gains[i]
        for j in range(0, N+1):
            gj = gains[j]
            G[j,i] = 1.0/(gi * np.conj(gj))  # Invert the gains

    return G


def dataplot(plt, name, freq, mu, sigma):
    fig, ax = plt.subplots()

    ax.set_xscale("log", nonpositive='clip')
    ax.set_yscale("log", nonpositive='clip')
    ax.errorbar(np.array(freq)/1e9, np.array(mu), yerr=np.array(sigma), fmt='.', label="Original Data")

    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel("Flux (Jy)")
    ax.grid(True)
    ax.set_title(f"Flux Measurements: {name}");
    return fig, ax
    


def posterior_plot(plt, name, freq, idata, nu0):
    fig, ax = plt.subplots()

    min_freq = freq[0]
    max_freq = freq[-1]

    
    dataset = []
    
    num_freqs = 30
    freq = np.geomspace(min_freq, max_freq, num_freqs)
    bar_widths = np.geomspace(min_freq, max_freq, num_freqs+1)[0:-1] - np.geomspace(min_freq, max_freq, num_freqs+1)[1:]
    for n in freq:
        samples = np.array([flux(n, get_random_sample(idata), nu0) for i in range(200)])
        dataset.append(samples)
    #dataset = np.array(dataset)
    
    #ax.plot(freq, dataset, '.', c="k", alpha=0.05)
        
    vp = ax.violinplot(dataset=dataset, positions=freq, widths = bar_widths,
                      showmeans=False, showmedians=False, showextrema=False)
                      #vert=True)
    # styling:
    for body in vp['bodies']:
        body.set_alpha(0.9)

    ax.set_xscale("log", nonpositive='clip')
    ax.set_yscale("log", nonpositive='clip')

    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel("Flux (Jy))")
    ax.set_title(f"Posterior Spectrum PDF ({name})")
    ax.grid(True);
    return fig, ax

if __name__=="__main__":
    a = np.array([[ 1.982e-04, -7.580e-06, -2.200e-05,  1.612e-05, -1.894e-06],
       [-7.580e-06,  6.800e-06, -1.363e-06, -8.556e-06,  4.099e-06],
       [-2.200e-05, -1.363e-06,  6.535e-06, -1.065e-06, -1.217e-06],
       [ 1.612e-05, -8.556e-06, -1.065e-06,  1.570e-05, -7.116e-06],
       [-1.894e-06,  4.099e-06, -1.217e-06, -7.116e-06,  3.798e-06]])
    matrix_2_latex(a, names=["$I_0$", "$a[0]$", "$a_1$", "$a_2$", "$a_3$"])

    idata = az.from_netcdf("source_j1939.nc")

    idata_2_latex(idata)
    a_cov, a_cor, names = chain_covariance(idata)
    print(a_cov)
