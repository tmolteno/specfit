## Examples for using specfit

Processing all the perley butler catalogue, as well as j1939_6342 can be done as follows.
 (in the directory above)

    pip3 install -e .


To generate the catalogue of sources

    make catalogue
    
This will generate an HDF5 file that contains a machine readable version of this catalogue in the file calibrator_catalogue.hdf5.
