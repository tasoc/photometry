#!/bin/env python
# -*- coding: utf-8 -*-
"""
This program will prepare the photometry on individual stars by doing all the operations which requires the full-size FFI images, like the following:

* Estimating sky background for all images.
* Estimating spacecraft jitter.
* Creating average image.
* Restructuring data into HDF5 files for efficient I/O operations.

The program can simply be run like the following, which will create a number of HDF5 files (`\*.hdf5`) in the ``TESSPHOT_INPUT`` directory.

>>> python prepare_photometry.py

The program internally calls the function :py:func:`photometry.prepare.prepare_photometry` with the given parameters.

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import argparse
import os.path
import logging
from photometry.prepare import prepare_photometry
from photometry.utilities import TqdmLoggingHandler

#------------------------------------------------------------------------------
if __name__ == '__main__':

	# Parse command line arguments:
	parser = argparse.ArgumentParser(description='Run TESS Photometry pipeline on single star.')
	parser.add_argument('-d', '--debug', help='Print debug messages.', action='store_true')
	parser.add_argument('-q', '--quiet', help='Only report warnings and errors.', action='store_true')
	parser.add_argument('--camera', type=int, choices=(1,2,3,4), default=None, help='TESS Camera. Default is to run all cameras.')
	parser.add_argument('--ccd', type=int, choices=(1,2,3,4), default=None, help='TESS CCD. Default is to run all CCDs.')
	parser.add_argument('input_folder', type=str, help='TESSPhot input directory to create HDF5 files in.', nargs='?', default=None)
	args = parser.parse_args()

	logging_level = logging.INFO
	if args.quiet:
		logging_level = logging.WARNING
	elif args.debug:
		logging_level = logging.DEBUG

	# Setup logging:
	formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
	logger = logging.getLogger(__name__)
	logger.setLevel(logging_level)
	console = TqdmLoggingHandler()
	console.setFormatter(formatter)
	logger_parent = logging.getLogger('photometry')
	logger_parent.setLevel(logging_level)
	if not logger.hasHandlers(): logger.addHandler(console)
	if not logger_parent.hasHandlers(): logger_parent.addHandler(console)

	# Check that the given input directory is indeed a directory:
	if args.input_folder is not None and not os.path.isdir(args.input_folder):
		parser.error("The given path does not exist or is not a directory")

	# Run the program for the selected camera/ccd combinations:
	prepare_photometry(args.input_folder, cameras=args.camera, ccds=args.ccd)
