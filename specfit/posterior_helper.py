import os
import re

import numpy as np
import arviz as az
import pymc3 as pm

import matplotlib.pyplot as plt



def row_2_table(row):
    print("    ", end='')
    cols = len(row)
    for j in range(cols):
        data = row[j]
        if isinstance(data, str):
            print(f"{data} ", end='')
        else:
            print("{:6.4g} ".format(data), end='')
        if j < cols-1:
            print("& ", end='')
    print("\\\\")

def clean_up_names(names):
    ret = []
    for n in names:
        # Replace x[x] with a_{x}
        newname = re.sub(r"a\[([0-9])\]", "a_{\\1}", n)
        ret.append(f"${newname}$")
    return ret

def matrix_2_latex(a, names):
    clean_names = clean_up_names(names)
    a = np.array(a)
    cols = a.shape[0]
    columns = ''
    for i in range(cols):
        columns = columns+'r'

    print(f"\\begin{{tabular}}{{{columns}}}")
    print("    \\hline")
    row_2_table(clean_names)
    print("    \\hline")
    for i in range(a.shape[1]):
        row_2_table(a[:,i])
    print("    \\hline")
    print("\\end{tabular}")

def vector_2_latex(a, names):
    '''
        print a vector as a vertival table
    '''
    clean_names = clean_up_names(names)
    a = np.array(a)
    cols = a.shape[0]
    columns = ''
    for i in range(cols):
        columns = columns+'r'

    print(f"\\begin{{tabular}}{{rrr}}")
    print(f"    $x$   & $\mu_x$    \\\\")
    print(f"    \\hline")
    for n, x in zip(clean_names, a):
        print(f"    {n} & {x[0] :6.4} & {x[1] :6.4}    \\\\")
    print("\\end{tabular}")


def get_stats(idata):
    names = [x for x in idata.posterior.mean()]
    ret = [ np.array([idata.posterior.get(x).mean().values.tolist(), idata.posterior.get(x).std().values.tolist()]) for x in names ]
    return np.array(ret).T, names

def idata_2_latex(idata):
    names = [x for x in idata.posterior.mean()]
    data = [ [idata.posterior.get(x).mean().values.tolist(), idata.posterior.get(x).std().values.tolist()] for x in idata.posterior.mean() ]
    matrix_2_latex(data, names)

def mean_2_latex(idata):
    names = [x for x in idata.posterior.mean()]
    data = [ [idata.posterior.get(x).mean().values.tolist(), idata.posterior.get(x).std().values.tolist()] for x in idata.posterior.mean() ]
    vector_2_latex(data, names)

def get_random_sample(idata, chain=0):
    names = [x for x in idata.posterior.mean()]
    n_samples = idata.posterior.get(names[0]).values.shape[1]
    sample = np.random.randint(0,n_samples)

    return [idata.posterior.get(n).values[chain, sample] for n in names]



def full_column(name, idata):
    a_cov, a_corr, names = chain_covariance(idata)
    print(f"{name} & ")
    mean_2_latex(idata)
    print(f" & ")
    matrix_2_latex(a_cov, names)
    print(f" & ")
    matrix_2_latex(a_corr, names)
    print("  \\\\  \\hline")

def chain_covariance(idata):
    names = [x for x in idata.posterior.mean()]
    chain = 0
    a = [idata.posterior.get(n)[chain, :] for n in names]
    a = np.array(a)
    return np.cov(a), np.corrcoef(a), names


def run_or_load(mcmc_model, fname, n_samples = 5000, n_tune=3000, n_chains=4):
    if os.path.exists(fname):
        ret = az.from_netcdf(fname)
    else:
        with mcmc_model:
            #approximation = pm.fit(n=n_samples, method='fullrank_advi') # Reutrns 
            #ret = approximation.sample(n_samples)
            start = pm.find_MAP()
            ret = pm.sample(n_samples, init='advi+adapt_diag', tune=n_tune, chains=n_chains, start=start, return_inferencedata=True, discard_tuned_samples=True)
        ret.to_netcdf(fname);
    return ret


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
