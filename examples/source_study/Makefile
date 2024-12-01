split:
	python3 fits_analyze.py --fits 'Abell22_full.fits' --output-dir "output"

process:
	python3 fits_analyze.py --fits 'Abell22_full.fits' --output-dir "output" --process

single:
	python3 plot_single_source.py --json="output/Source_RA:4d30m20.69878782s_DEC:-25d40m58.29839961s_data.json"

test:
	python3 piecewise_linear.py
