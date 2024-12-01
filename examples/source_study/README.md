# Processing FITS files

These examples show piecewise linear fitting of spectral data stored in a FITS file. The processing is done in several steps

## Step 1: Split the FITS into JSON files

The FITS file is split into a separate JSON file for each source. This is done using 

    python3 fits_analyze.py --fits="" --output-dir="output"

It will generate a separate JSON file (with ending ```_data.json```) for each target. For example, here is a single target output:

```
{
    "nu": [
        908037109.375,
        952341796.875,
        996646484.375,
        1043458984.375,
        1092779296.875,
        1144607421.875,
        1317228515.625,
        1381177734.375,
        1448052734.375,
        1519943359.375,
        1593923828.125,
        1656201171.875
    ],
    "S": [
        0.0002981782049150504,
        0.00025634331841303316,
        0.00034713640209420397,
        0.0003420256429556299,
        0.0003340862582649761,
        0.00037505784328982055,
        0.00031134723540424976,
        0.0003296269303113061,
        0.0003128251940982223,
        0.0003326067878133761,
        0.0004288621893355432,
        0.0003389767250436867
    ],
    "sigma": [
        7.198934407619225e-05,
        8.036091829966124e-05,
        4.6583878901544084e-05,
        4.1558142504298154e-05,
        3.737381825774445e-05,
        4.6848063813339124e-05,
        2.5919513000915667e-05,
        2.607462645054816e-05,
        2.5422437866387376e-05,
        3.555588307303389e-05,
        7.213247459514352e-05,
        4.277989749287384e-05
    ],
    "nu0": 1000000000.0,
    "ra": 4.505749663282475,
    "dev": -25.68286066655953
}
```

## Step 2: Process all files

    python3 fits_analyze.py --fits="" --output-dir="output" --process

## Plotting a single result

This code shows how to manually plot (rather than using specfit to plot). The code performs a fit, and then
draws samples from the spectral fit, and plots them using matplotlib. Use this code as a base when you want
to control the plotting process for your own publications.

    python3 plot_single_source.py --json="ouput/Source_RA:4d30m20.69878782s_DEC:-25d40m58.29839961s_data.json"
