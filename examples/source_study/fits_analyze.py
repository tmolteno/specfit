import numpy as np
import matplotlib.pyplot as plt

from astropy.io import fits
# from astropy.wcs import WCS
import specfit as sf


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

    for i in range(len(ra)):

        r = ra[i]
        d = dec[i]
        nu = freq[i,:]
        S = total_flux[i,:]
        ES = E_total_flux[i,:]
        nu0 = 1e9
        sigma = ES


        def todms(a):
            deg = int(np.floor(a))
            mm = int(np.floor((a - deg)*60))
            sss = (a - deg - mm/60)
            return f"{deg :02d}:{mm :02d}:{sss :04.2f}"

        name=f"Source RA:{todms(r)}, DEC:{todms(d)}"

        if True:
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
                plt.show()
            # break
        else:
            fig, ax = sf.dataplot(plt, name=name, freq=nu, mu=S, sigma=ES)

        # ax.ticklabel_format(style='plain')
        # plt.plot(nu, S, '.')
        # plt.title(f"Source RA:{r}, DEC:{d}")
        # plt.xlabel('Frequency ()')
        # plt.ylabel('Normalized flux')
        plt.show()
