# For development purposes, use the 'develop' target to install this package from the source repository (rather than the public python packages). This means that local changes to the code will automatically be used by the package on your machine.
# To do this type
#     make develop
# in this directory.

develop: venv
	${VENV}/pip3 install -e .

requirements: venv
	${VENV}/pip3 freeze > requirements.txt

example:
	${VENV}/python3 examples/raw_j1939_6342.py

lint:
	pylint --extension-pkg-whitelist=numpy --ignored-modules=numpy,tart_tools,dask,dask.array --extension-pkg-whitelist=astropy --extension-pkg-whitelist=dask specfit

test_upload:
	rm -rf specfit.egg-info dist
	python3 setup.py sdist
	twine upload --repository testpypi dist/*

upload:
	rm -rf specfit.egg-info dist
	python3 setup.py sdist
	twine upload --repository pypi dist/*


include Makefile.venv

$(VENV):
	$(PY) -m venv --system-site-packages $(VENVDIR)
#	$(PY) -m venv $(VENVDIR)
	$(VENV)/python3 -m pip install --upgrade pip setuptools wheel

