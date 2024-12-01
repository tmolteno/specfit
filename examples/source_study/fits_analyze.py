import json
import argparse
import os

import numpy as np
import matplotlib

import matplotlib.pyplot as plt

from astropy.io import fits
from astropy import units as u
from astropy.coordinates import Angle

import specfit as sf

from pathlib import Path

from piecewise_linear import piecewise_linear

matplotlib.use('PDF')


def process_natural_cubic(name, nu, S, ES, nu0, output_dir):
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


def process_json_file(name, nu, S, ES, nu0, output_dir):
    idata, json_data = piecewise_linear(name, freq=nu, S=S,
                                        sigma=ES, nu0=nu0, 
                                        order=1,
                                        n_samples=3000,
                                        output_dir=output_dir)
    json_data['ra'] = r
    json_data['dec'] = d

    line = json_data['name']
    for key in ['order', 'ra', 'dec', 'slopes', 'slopes_sigma',
                'change_point', 'change_point_sigma',
                'log_marginal_likelihood']:
        item = json_data[key]
        if isinstance(item, list):
            line = line + f", {vect2csv(item)}"
        else:
            line = line + f", {item}"

    result_csv.append(line)

    with open("results.csv", 'w') as csv_file:
        for line in result_csv:
            print(line, file=csv_file)

    del idata


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Plot single piecewise fit.')

    parser.add_argument('--fits', required=True,
                        help="Spectral FITS file.")

    parser.add_argument('--process', action="store_true",
                        help="Process each file.")

    parser.add_argument('--output-dir', required=False, default="output",
                        help="Output directory.")

    ARGS = parser.parse_args()

    fits_file = ARGS.fits   # "'Abell22_full.fits'"

    Path(ARGS.output_dir).mkdir(parents=True, exist_ok=True)

    with fits.open(fits_file) as hdul:
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

        def vect2csv(a_list):
            converted_list = [str(element) for element in a_list]
            return ", ".join(converted_list)

        def vect2hdr(name, dim):
            x = [f"{name}[{i}]" for i in range(dim)]
            return vect2csv(x)

        result_csv = []
        header = f'name, order, ra, dec, {vect2hdr("slopes_mean",2)},  {vect2hdr("slopes_sigma",2)}, change_point_mean, change_point_sigma, log_marginal_likelihood'
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

            name = f"Source_RA:{todms(r)}_DEC:{todms(d)}"
            # Dump the raw data for the source to a json file.
            raw_data = {
                'nu': nu.tolist(),
                'S': S.tolist(),
                'sigma': ES.tolist(),
                'nu0': nu0,
                'ra': r,
                'dev': d
                }

            outfile = os.path.join(ARGS.output_dir, f"{name}_data.json")
            with open(outfile, 'w') as data_file:
                print(json.dumps(raw_data, indent=4), file=data_file)

            if ARGS.process:
                process_json_file(name, nu, S, ES, nu0, output_dir=ARGS.output_dir)
