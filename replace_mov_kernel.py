#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rewrite movement kernel in simulated data HDF5 file.

@author: Jonas Svenstrup Hansen <jonas.svenstrup@gmail.com>
"""

import os
import argparse
import numpy as np
import logging
import warnings
warnings.filterwarnings('ignore', category=FutureWarning, module='h5py')
import h5py


if __name__ == '__main__':

	""" Parse command line arguments """
	parser = argparse.ArgumentParser(description='Run simulations of the TESS Photometry pipeline.')
	parser.add_argument('-n', '--dirname', help='Folder name in which the HDF5 file is located.')
	parser.add_argument('-c', '--correction', type=float, default=0, help='Fraction of movement kernel correction. 1 is completely corrected, 0 is no correction which overwrites the movement kernel with zeros (default).')
	parser.add_argument('-t', '--trueJitter', type=int, default=0, help='1 if the true jitter from the simulation is to be used as movement kernel instead. Default is 0.')
	parser.add_argument('-d', '--debug', help='Print debug messages.', action='store_true')
	parser.add_argument('-q', '--quiet', help='Only report warnings and errors.', action='store_true')
	args = parser.parse_args()
	dirname = args.dirname
	correction = args.correction
	trueJitter = args.trueJitter

	""" Set up logging """
	# Set logging level:
	logging_level = logging.INFO
	if args.quiet:
		logging_level = logging.WARNING
	elif args.debug:
		logging_level = logging.DEBUG

	# Setup logging:
	formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
	console = logging.StreamHandler()
	console.setFormatter(formatter)
	logger = logging.getLogger(__name__)
	logger.addHandler(console)
	logger.setLevel(logging_level)

	# Log command line arguments:
	logger.debug("HDF5 file folder name is: {}".format(dirname))
	logger.debug("Correction fraction is: {}".format(correction))
	logger.debug("Is true jitter used?: {}".format(trueJitter))

	# Get TESSPHOT input environment variables:
	input_folder = os.environ.get('TESSPHOT_INPUT')

	# Rewrite motion_kernel in hdf5 file:
	hdf_file = os.path.join(input_folder,dirname,'camera{0:d}_ccd{1:d}.hdf5'.format(1,1))
	logger.info("Rewriting motion_kernel in hdf5 file: \n{}".format(hdf_file))
	with h5py.File(hdf_file, 'a') as hdf:
		# Get original movement kernel
		movement_kernel = np.array(hdf['movement_kernel'])
		logger.info("Original movement kernel: \n{}".format(movement_kernel))
		logger.debug(np.shape(movement_kernel))

		# Define new movement kernel:
		if trueJitter:
			logger.info("Setting new movement kernel to a {} fraction of the true jitter.".format(correction))
			movement_kernel_new = correction*np.loadtxt(os.path.join(input_folder, dirname, 'jitter.txt'))
		else:
			logger.info("Setting new movement kernel to a {} fraction of previous jitter.".format(correction))
			movement_kernel_new = correction*movement_kernel

		# Replace values of movement kernel in hdf5 file:
		hdf['movement_kernel'][:] = movement_kernel_new
		logger.info("New movement kernel: \n{}".format(np.array(hdf['movement_kernel'])))