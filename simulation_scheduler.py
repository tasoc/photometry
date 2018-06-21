#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simulate data, prepare it for and do photometry with the TESS Photometry pipeline.

@author: Jonas Svenstrup Hansen <jonas.svenstrup@gmail.com>
"""

import os
import argparse
import numpy as np
import logging
import warnings
warnings.filterwarnings('ignore', category=FutureWarning, module='h5py')
import h5py

from photometry.prepare import create_hdf5
from simulation.simulateFITS import simulateFITS


if __name__ == '__main__':

	""" Parse command line arguments """
	parser = argparse.ArgumentParser(description='Run simulations of the TESS Photometry pipeline.')
	parser.add_argument('-d', '--debug', help='Print debug messages.', action='store_true')
	parser.add_argument('-q', '--quiet', help='Only report warnings and errors.', action='store_true')
	args = parser.parse_args()


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


	""" Create dictionaries with simulations """

	multi_star_2000 = {
		'name':				'multi_star_2000',
		'ignore_mov_kernel': 	False,
		'run_simulateFITS': 	[2000, 350], # stars, samples
#		'run_simulateFITS': 	[1, 27*24*2], # 1 star, 27 days long cadence
#		'run_simulateFITS': 	[1, 2], # test run with just 2 time steps
		'create_hdf5': 		[0, 1, 1], # sector, camera, ccd
		'methods': 			['aperture', 'linpsf'], # photometry methods
		'stars': 			np.arange(1,2001, dtype=int) # stars to do photometry on
	}

	multi_star_test = {
		'name':				'multi_star_test',
		'ignore_mov_kernel': 	False,
		'run_simulateFITS': 	[2000, 2], # 2 stars, 2 samples
#		'run_simulateFITS': 	[1, 27*24*2], # 1 star, 27 days long cadence
#		'run_simulateFITS': 	[1, 2], # test run with just 2 time steps
		'create_hdf5': 		[0, 1, 1], # sector, camera, ccd
		'methods': 			['linpsf'], # photometry methods
		'stars': 			np.arange(1,3, dtype=int) # stars to do photometry on
	}

	# Collect dictionaries in list:
	simulations = [multi_star_test]
	logger.info("Simulations being run: \n %s", simulations)


	""" Create I/O directories for simulation """
	# Get TESSPHOT I/O environment variables:
	input_folder = os.environ.get('TESSPHOT_INPUT')
	output_folder = os.environ.get('TESSPHOT_OUTPUT')

	# For each simulation and photometry method, create I/O directories:
	for simulation in simulations:
		simulation['input_folder'] = os.path.join(input_folder, simulation['name'])

		# Create simulation input directory:
		if not os.path.exists(simulation['input_folder']):
			os.makedirs(simulation['input_folder'])
		else:
			logger.warning("Directory '%s' already exists", simulation['input_folder'])

		# Preallocate list in simulation dict for output directories:
		simulation['output_folders'] = []

		# Loop through photometry methods:
		for method in simulation['methods']:
			# Define output directory:
			sim_output_folder = os.path.join(output_folder, simulation['name'] + '_' + method)

			# Save output directory to dictionary:
			simulation['output_folders'].append(sim_output_folder)

			# Create output directory if it does not exist:
			if not os.path.exists(sim_output_folder):
				os.makedirs(sim_output_folder)
			else:
				logger.warning("Directory '%s' already exists",sim_output_folder)


	""" Loop through all the simulations """
	for simulation in simulations:
		# Set the input environment variable:
		os.environ['TESSPHOT_INPUT'] = simulation['input_folder']
		logger.info("TESSPHOT_INPUT set to '%s'", os.environ.get('TESSPHOT_INPUT'))

		# Run run_simulateFITS.py:
		Nstars = simulation['run_simulateFITS'][0]
		Ntimes = simulation['run_simulateFITS'][1]
		simulateFITS(Nstars=Nstars, Ntimes=Ntimes, save_images=True, overwrite_images=True)

		# Run create_hdf5 from prepare_photometry.py:
		logger.info("Running create_hdf5 from prepare_photometry.py")
		create_hdf5(
			simulation['input_folder'],
			cameras = simulation['create_hdf5'][1],
			ccds    = simulation['create_hdf5'][2]
		)

		# Rewrite motion_kernel in hdf5 file:
		if simulation['ignore_mov_kernel']:
			hdf_file = os.path.join(input_folder,
				simulation['name'],
				'camera{0:d}_ccd{1:d}.hdf5'.format(
					simulation['create_hdf5'][1],
					simulation['create_hdf5'][2]
					)
			)
			logger.info("Rewriting motion_kernel in hdf5 file: \n{}".format(hdf_file))
			with h5py.File(hdf_file, 'a') as hdf:
				# Get original movement kernel
				movement_kernel = np.array(hdf['movement_kernel'])
				logger.debug("Original movement kernel: \n{}".format(movement_kernel))
				logger.debug(np.shape(movement_kernel))

				# Define new movement kernel:
				movement_kernel_new = np.zeros_like(movement_kernel)

				# Replace values of movement kernel in hdf5 file:
				hdf['movement_kernel'][:] = movement_kernel_new
				logger.debug("New movement kernel: \n{}".format(np.array(hdf['movement_kernel'])))
