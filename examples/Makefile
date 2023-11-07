VENV=../.venv/bin

all:
	mkdir -p output
	$(VENV)/python3 perley_butler_process.py

original:
	mkdir -p output
	$(VENV)/python3 SourceOriginal.py

install:
	cd ..; pip3 install -e .
