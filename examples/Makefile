
catalogue:
	rm -f calibrator_catalogue.hdf5
	mkdir -p output
	python3 perley_butler_process.py
	python3 SourceOriginal.py

j1939:
	mkdir -p output
	python3 SourceOriginal.py

raw:
	mkdir -p output
	python3 raw_j1939_6342.py


install:
	cd ..; pip3 install -e .
