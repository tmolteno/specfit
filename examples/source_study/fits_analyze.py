import numpy as np
import matplotlib
matplotlib.use('PDF')

import matplotlib.pyplot as plt

from astropy.io import fits
from astropy import units as u
from astropy.coordinates import Angle

import specfit as sf

from piecewise_linear import piecewise_linear
import arviz as az


if __name__ == "__main__":
        
    with fits.open('Abell22_full.fits') as hdul:
        print(hdul.info())

        data = hdul[0].data
        h1 = hdul[0].header

        print(repr(h1))
        print(h1['COMMENT'])
        print(h1['SIMPLE'])
        print(h1['NTABLE'])
        print(data)

        data = hdul[1].data
        h1 = hdul[1].header
        print(repr(h1))

        ra = data['RA']
        dec = data['DEC']

        freq = []
        total_flux = []
        E_total_flux = []
        for i in range(12):
            freq.append(data[f"Frequency_{i}"])
            total_flux.append(data[f"Total_flux_{i}"])
            E_total_flux.append(data[f"E_Total_flux_{i}"])

        freq = np.array(freq).T
        total_flux = np.array(total_flux).T
        E_total_flux = np.array(E_total_flux).T
        print(f"RA shape {ra.shape}")
        print(f"DEC shape {dec.shape}")
        print(f"Freq Shape {freq.shape}")
        print(f"Flux Shape {total_flux.shape}")

        result_csv = []
        header = "name, order, ra, dec, slopes, slopes_sigma, change_point, change_point_sigma, log_marginal_likelihood"
        result_csv.append(header)
        
        for i in range(len(ra)):

            r = ra[i]
            d = dec[i]
            nu = freq[i,:]
            S = total_flux[i,:]
            ES = E_total_flux[i,:]
            nu0 = 1e9
            sigma = ES

            def todms(a):
                angle = Angle(a * u.deg) 
                return angle.to_string(unit=u.degree)

            name=f"Source_RA:{todms(r)}_DEC:{todms(d)}"

            if False:
                x = np.log(nu/nu0)
                y = np.log(S)
                y0 = (S-ES/2)
                y1 = (S+ES/2)
                y_sigma = np.log(y1) - y
                x_curve = np.linspace(x[0], x[-1], 100)

                spline = sf.NaturalCubicSpline([x[0], x[-1]], n_interior_knots=1, degree=1)

                f = spline.get_expr()

                ret = spline.regression(x, y, y_sigma=y_sigma)
                print(ret)

                with plt.rc_context({"axes.grid": True, "axes.formatter.min_exponent": 2}):
                    fig, ax = plt.subplots(figsize=(6,4), layout='constrained')

                    ax.set_xscale("log", nonpositive='clip')
                    ax.set_yscale("log", nonpositive='clip')

                    ax.set_xlabel("Frequency (GHz)")
                    ax.set_ylabel("Flux (mJy)")

                    knot_y = ret["knot_y"]
                    y_best = f(k_0=ret["interior_knots"][0], x=x_curve, y_0=knot_y[0],
                            y_1=knot_y[1],
                            y_2=knot_y[2])

                    ax.plot(np.exp(x_curve), np.exp(y_best)*1000, label="fit")
                    ax.errorbar(np.exp(x), np.exp(y)*1000, yerr=ES, fmt='o', label="data")
                    ax.set_title(name)
                    ax.legend()
                    plt.savefig(f"{name}.pdf")
                # break
            else:
                idata, json_data = piecewise_linear(name, freq=nu, S=S, sigma=ES, nu0=nu0)
                json_data['ra'] = r
                json_data['dec'] = d

                line = f"{json_data['name']}, {json_data['order']}, {json_data['ra']}, {json_data['dec']}, {json_data['slopes']}, {json_data['slopes_sigma']}, {json_data['change_point']}, {json_data['change_point_sigma']}, {json_data['log_marginal_likelihood']}"
                result_csv.append(line)

                with open("results.csv", 'w') as csv_file:
                    for line in result_csv:
                        print(line, file=csv_file)

                del idata
