===========================
The TASOC Photometry module
===========================
.. image:: https://zenodo.org/badge/103402174.svg
    :target: https://zenodo.org/badge/latestdoi/103402174
.. image:: https://travis-ci.org/tasoc/photometry.svg?branch=devel
    :target: https://travis-ci.org/tasoc/photometry
.. image:: https://img.shields.io/codecov/c/github/tasoc/photometry
    :target: https://codecov.io/github/tasoc/photometry
.. image:: https://hitsofcode.com/github/tasoc/photometry?branch=devel
    :alt: Hits-of-Code
    :target: https://hitsofcode.com/view/github/tasoc/photometry?branch=devel
.. image:: https://img.shields.io/github/license/tasoc/photometry.svg
    :alt: license
    :target: https://github.com/tasoc/photometry/blob/devel/LICENSE

This module provides the basic photometry setup for the TESS Asteroseismic Science Operations Center (TASOC).

The code is available through our GitHub organisation (https://github.com/tasoc/photometry) and full documentation for this code can be found on https://tasoc.dk/code/.

.. note::
    Even though the full code and documentation are freely available, we highly encourage users to not attempt to use the code to generate their own photometry from TESS. Instead we encourage you to use the fully processed data products from the full TASOC pipeline, which are available from `TASOC <https://tasoc.dk>`_ and `MAST <https://archive.stsci.edu/hlsp/tasoc>`_. If you are interested in working on details in the processing, we welcome you to join the T'DA working group.

Installation instructions
=========================
* Start by making sure that you have `Git Large File Storage (LFS) <https://git-lfs.github.com/>`_ installed. You can verify that is installed by running the command:

  >>> git lfs version

* Go to the directory where you want the Python code to be installed and simply download it or clone it via *git* as::

  >>> git clone https://github.com/tasoc/photometry.git .

* All dependencies can be installed using the following command. It is recommended to do this in a dedicated `virtualenv <https://virtualenv.pypa.io/en/stable/>`_ or similar:

  >>> pip install -r requirements.txt

How to run tests
================
You can test your installation by going to the root directory where you cloned the repository and run the command::

>>> pytest

Running the program
===================

Just trying it out
------------------
For simply trying out the code straight after installation, you can simply run the photometry code directly. This will automatically load some test input data and run the photometry (see more details in the full documentation or below).

>>> python run_tessphot.py --starid=182092046

The number refers to the TIC-number of the star, and the above one can replaced with any TIC-number that is available in the TODO-list (see below).

Set up directories
------------------
The next thing to do is to set up the directories where input data is stored, and where output data (e.g. lightcurves) should be put. This is done by setting the enviroment variables ``TESSPHOT_INPUT`` and ``TESSPHOT_OUTPUT``.
Depending on your operating system and shell this is done in slightly different ways.

The directory defined in ``TESSPHOT_INPUT`` should contain all the data in FITS files that needs to be processed. The FITS files can be structured into sub-directories as you wish and may also be GZIP compressed (\*.fits.gz). When the different programs runs, some of them will also add some more files to the ``TESSPHOT_INPUT`` directory. The directory in ``TESSPHOT_OUTPUT`` is used to store all the lightcurve FITS file that will be generated at the end.

Make star catalogs
------------------
The first program to be run is the ``make_catalog.py`` program, which will create full catalogs of all stars known to fall on or near the TESS detectors during a given observing sector. These catalogs are created directly from the TESS Input Catalog (TIC), and since this is such a huge table this program relies on internal databases running at TASOC at Aarhus University. You therefore need to be connected to the network at TASOC at Aarhus Univsity to run this program.
The program is simply run as shown here for sector #14 (see full documentation for more options):

>>> python make_catalog.py 14

Prepare photometry
------------------
The next part of the program is to prepare photometry on individual stars by doing all the operations which requires the full-size FFI images, like the following:

* Estimating sky background for all images.
* Estimating spacecraft jitter.
* Creating average image.
* Restructuring data into HDF5 files for efficient I/O operations.

The program can simply be run like the following, which will create a number of HDF5 files (`*.hdf5`) in the ``TESSPHOT_INPUT`` directory.

>>> python run_prepare_photometry.py

Make TODO list
--------------
A TODO-list is a list of targets that should be processed by the photometry code. It includes information about which cameras and CCDs they fall on and which photometric methods they should be processed with. A TODO-list can be generated directly from the catalog files (since these contain all targets near the field-of-view) and the details stored in the HDF5 files.
In order to create a full TODO list of all stars that can be observed, simply run the command:

>>> python make_todo.py

This will create the file ``todo.sqlite`` in the ``TESSPHOT_INPUT`` directory, which is needed for running the photometry. See the full documentation for more options.

Running the photometry
----------------------
The photometry program can by run on a single star by running the program::

  >>> python run_tessphot.py --starid=182092046

Here, the number gives the TIC identifier of the star. The program accepts various other command-line parameters - Try running::

  >>> python run_tessphot.py --help

This is very usefull for testing different methods and settings.

Contributing to the code
========================
You are more than welcome to contribute to this code!
Please contact `Rasmus Handberg <rasmush@phys.au.dk>`_ or `Derek Buzasi <dbuzasi@fgcu.edu>`_ if you wish to contribute.
