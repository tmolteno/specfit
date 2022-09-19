import numpy as np
import specfit as sf
import matplotlib.pyplot as plt

# Data from J.E. Reynolds for J1939-6342
original_polynomial = [-30.7667,  26.4908, -7.0977, 0.605334]
original_nu0 = 1.0e6


sigma = 0.5

fig, ax = plt.subplots()
ax.set_xscale("log", nonpositive='clip')
ax.set_yscale("log", nonpositive='clip')
ax.grid(True);

nu = np.linspace(0.5e9, 10e9, 100)
S = sf.flux(nu, original_polynomial, nu0=original_nu0)
ax.plot(nu/1e9, S, label="polynomial fit")
ax.legend()
fig.tight_layout()
plt.show()

names, stats, a_cov, a_corr, f, fake_data = \
    sf.datafree_inference(
                        name="J1939-6342",  
                        freq_min=0.5e9, 
                        freq_max = 10e9, 
                        nfreq = 20, 
                        sigma = 0.5, 
                        a = original_polynomial,  
                        nu0 = original_nu0)

a = stats[0] # Means


print(f"Variables: {names}")
print(f"Means: {stats[0]}")
print(f"SDev: {stats[1]}")
print(f"Covariance Matrix:\n{np.array2string(a_cov, separator=',', precision=4)}")
print(f"Correlation Matrix:\n{np.array2string(a_corr, separator=',', precision=4)}")
