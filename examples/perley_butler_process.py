import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import specfit as sf

def cleanup(_S, _sigma, _freq):
    S = np.array(_S)
    sigma = np.array(_sigma)
    freq = np.array(_freq)
    good = np.where(np.isnan(S) == False)
    return S[good], sigma[good], freq[good]


def process(outfile, key, _S, _sigma, _frequency, order):
    global full_names
    official_name = full_names[key][0]
    print(f"%%%%% Processing {key} order={order}", file=outfile, flush=True)
    nu0 = 1.4e9
    S, sigma, freq = cleanup(_S, _sigma, _frequency)
    names, stats, a_cov, a_corr, idata = sf.data_inference(key, freq, S, sigma, order=order, nu0=nu0)
    sf.full_column(outfile, full_names[key], idata, np.array(freq))
    #print(pm.summary(idata))
    fig, ax = sf.dataplot(plt, official_name, freq, S, sigma)
    a, da = stats
    nu = np.geomspace(freq[0], freq[-1], 100)
    ax.plot(nu/1e9, sf.flux(nu, a, nu0), label='Polynomial model')
    ax.legend()
    #plt.show()
    fig.savefig(f"./output/{key}_data.pdf")
    plt.close(fig)
    
    fig, ax = sf.posterior_plot(plt, official_name, freq, idata, nu0)
    fig.savefig(f"./output/{key}_pdf.pdf")
    plt.close(fig)



if False:
    data = pd.read_csv('perley_butler_ascii.txt', skiprows=5, skipfooter=1, skipinitialspace=True)

    print(data)



    frequency = data['(GHz)']*1e9

    S_3C123 = data['S(Jy)']
    sigma_3C123 = data['sigma(Jy)']

    S_3C196 = data['S(Jy).1']
    sigma_3C196 = data['sigma(Jy).1']

    S_3C286 = data['S(Jy).2']
    sigma_3C286 = data['sigma(Jy).2']

    S_3C295 = data['S(Jy).3']
    sigma_3C295 = data['sigma(Jy).3']


    N_obs = data['N_obs']


    process("3C123_2013", S_3C123, sigma_3C123)
    process("3C196_2013", S_3C196, sigma_3C196)
    process("3C286_2013", S_3C286, sigma_3C286)
    process("3C295_2013", S_3C295, sigma_3C295)


### Now 2017 data


data = pd.read_csv('2016Flux-B.dat', skiprows=6, sep='\s+', skipfooter=1, skipinitialspace=True, quotechar="'", keep_default_na=True, engine='python')
#print(data)

frequency = data['Freq']*1e9
err = np.array(data[r"%Err"])


stars  = ["J0133", "For_A", "3C48", "3C123", "J0444", "3C138", "PicA", "3C144", "3C147", "3C196", "3C218", "3C274", "3C286", "3C295", "3C348", "3C353", "3C380", "3C405", "3C444", "3C461"]
orders = {
    'J0133': 3, # J0133-3629
    'For_A': 2, # J0322-3712 Fornax A
    '3C48': 4,  # J0137+3309 3C48
    '3C123': 5, # J0437+2940 3C123
    'J0444': 3, # J0444-2809
    '3C138': 5, # J0521+1638 3C138
    'PicA': 3,  # J0519-4546 Pictor A
    '3C144': 4, # J0534+2200 3C144, Taurus A, Crab
    '3C147': 6, # J0542+4951 3C147
    '3C196': 5, # J0813+4813 3C196
    '3C218': 5, # J0918-1205 3C218, Hydra A
    '3C274': 3, # J1230+1223 3C274, Virgo A, M87
    '3C286': 4, # J1331+3030 3C286
    '3C295': 5, # J1411+5212 3C295
    '3C348': 3, # J1651+0459 3C348, Hercules A
    '3C353': 4, # J1720-0058 3C353
    '3C380': 6, # J1829+4844 3C380
    '3C405': 5, # J1959+4044 3C405, Cygnus A
    '3C444': 4, # J2214-1701 3C444
    '3C461': 4  # J2323+5848 3C461, Cassiopeia A
}

full_names = {
    'J0133': ['J0133-3629'],
    'For_A': ['J0322-3712', 'Fornax A'],
    '3C48':  ['J0137+3309', '3C48'],
    '3C123': ['J0437+2940', '3C123'],
    'J0444': ['J0444-2809'],
    '3C138': ['J0521+1638', '3C138'],
    'PicA':  ['J0519-4546', 'Pictor A'],
    '3C144': ['J0534+2200', '3C144', 'Taurus A', 'Crab'],
    '3C147': ['J0542+4951', '3C147'],
    '3C196': ['J0813+4813', '3C196'],
    '3C218': ['J0918-1205', '3C218', 'Hydra A'],
    '3C274': ['J1230+1223', '3C274', 'Virgo A', 'M87'],
    '3C286': ['J1331+3030', '3C286'],
    '3C295': ['J1411+5212', '3C295'],
    '3C348': ['J1651+0459', '3C348', 'Hercules A'],
    '3C353': ['J1720-0058', '3C353'],
    '3C380': ['J1829+4844', '3C380'],
    '3C405': ['J1959+4044', '3C405', 'Cygnus A'],
    '3C444': ['J2214-1701', '3C444'],
    '3C461': ['J2323+5848', '3C461', 'Cassiopeia A']
}

with open('perley_butler_2017.tex', 'w') as outfile:
    for calibrator in stars:
        mu = data[calibrator]
        sigma = mu*(err / 100)
        process(outfile, calibrator, mu, sigma, frequency, orders[calibrator])
