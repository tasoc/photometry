This directory serves as the building ground for a FITS image simulator for TESS.

The goal of this simulator is to provide the means to qualify various photometry methods.
For this purpose, full frame images of a reduced size, 200x200px compared to the TESS 2048x2048px, are created.
The code is partially inspired by the much more extensive SPyFFI, which is streamlined for catalog data.
This project aims to provide a more general and customisable platform with which to create simulations of a wide range of stellar images.

When finished, the output FITS files will be run through the TASOC photometry pipeline.
Here it will be converted to the HDF5 format for faster I/O, the background will be estimated and, finally, photometry will be made for each target star in the FFI after separation into stamps.
The output of the pipeline is then a FITS file containing the time series for a given star.
