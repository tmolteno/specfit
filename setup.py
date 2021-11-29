'''
    A convert TART JSON data into a measurement set.
    Author: Tim Molteno, tim@elec.ac.nz
    Copyright (c) 2019.

    License. GPLv3.
'''
from setuptools import setup, find_packages

with open('README.md') as f:
    readme = f.read()

setup(name='specfit',
    version='0.1.0b3',
    description='Infer polynomial spectral models with covariancess',
    long_description=readme,
    long_description_content_type="text/markdown",
    url='http://github.com/tmolteno/specfit',
    author='Tim Molteno',
    author_email='tim@elec.ac.nz',
    license='GPLv3',
    install_requires=['numpy', 'h5py', 'pymc3', 'arviz', 'matplotlib'],
    test_suite='nose.collector',
    tests_require=['nose'],
    packages=['specfit'],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Topic :: Scientific/Engineering",
        "Operating System :: OS Independent",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        "Intended Audience :: Science/Research"])
