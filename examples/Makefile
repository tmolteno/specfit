VENV=../.venv/bin

perley_butler:
	mkdir -p output
	$(VENV)/python3 perley_butler_process.py

j1939:
	mkdir -p output
	$(VENV)/python3 SourceOriginal.py

install:
	cd ..; pip3 install -e .
