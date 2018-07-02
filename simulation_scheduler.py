#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simulate data for the TASOC photometry pipeline.

@author: Jonas Svenstrup Hansen <jonas.svenstrup@gmail.com>
"""

import os
import argparse
import logging
import warnings
warnings.filterwarnings('ignore', category=FutureWarning, module='h5py')

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
		'create_hdf5': 		[0, 1, 1], # sector, camera, ccd
		'methods': 			['aperture', 'linpsf'] # photometry methods
	}

	multi_star_test_noise = {
		'name':				'multi_star_test_noise',
		'ignore_mov_kernel': 	True,
		'run_simulateFITS': 	[100, 2, True, True, True, True, True, True, 1, True],
		'create_hdf5': 		[0, 1, 1], # sector, camera, ccd
		'methods': 			['linpsf'] # photometry methods
	}

	multi_star_test_no_noise = {
		'name':				'multi_star_test_no_noise',
		'ignore_mov_kernel': 	True,
		'run_simulateFITS': 	[100, 2, True, True, True, False, True, True, 1, True],
		'create_hdf5': 		[0, 1, 1], # sector, camera, ccd
		'methods': 			['linpsf', 'aperture'] # photometry methods
	}


	""" Noise investigation simulation runs """
	# jitter, noise, inaccurate catalog, variables

	multi_star_no_noise = {
		'name':				'multi_star_no_noise',
		'ignore_mov_kernel': 	False,
		'run_simulateFITS': 	[2000, 150, True, True, False, False, True, False, 0, True],
		'create_hdf5': 		[0, 1, 1], # sector, camera, ccd
		'methods': 			['linpsf', 'aperture'] # photometry methods
	}

	multi_star_noise = {
		'name':				'multi_star_noise',
		'ignore_mov_kernel': 	False,
		'run_simulateFITS': 	[2000, 150, True, True, False, True, True, False, 0, True],
		'create_hdf5': 		[0, 1, 1], # sector, camera, ccd
		'methods': 			['linpsf', 'aperture'] # photometry methods
	}

	multi_star_jitter = {
		'name':				'multi_star_jitter',
		'ignore_mov_kernel': 	False,
		'run_simulateFITS': 	[2000, 150, True, True, True, False, True, False, 0, True],
		'create_hdf5': 		[0, 1, 1], # sector, camera, ccd
		'methods': 			['linpsf', 'aperture'] # photometry methods
	}

	multi_star_inaccurate = {
		'name':				'multi_star_inaccurate',
		'ignore_mov_kernel': 	False,
		'run_simulateFITS': 	[2000, 150, True, True, False, False, True, True, 0, True],
		'create_hdf5': 		[0, 1, 1], # sector, camera, ccd
		'methods': 			['linpsf', 'aperture'] # photometry methods
	}

	multi_star_variable = {
		'name':				'multi_star_variable',
		'ignore_mov_kernel': 	False,
		'run_simulateFITS': 	[2000, 150, True, True, False, False, True, False, 1000, True],
		'create_hdf5': 		[0, 1, 1], # sector, camera, ccd
		'methods': 			['linpsf', 'aperture'] # photometry methods
	}

	multi_star_noise_jitter = {
		'name':				'multi_star_noise_jitter',
		'ignore_mov_kernel': 	False,
		'run_simulateFITS': 	[2000, 150, True, True, True, True, True, False, 0, True],
		'create_hdf5': 		[0, 1, 1], # sector, camera, ccd
		'methods': 			['linpsf', 'aperture'] # photometry methods
	}

	multi_star_noise_inaccurate = {
		'name':				'multi_star_noise_inaccurate',
		'ignore_mov_kernel': 	False,
		'run_simulateFITS': 	[2000, 150, True, True, False, True, True, True, 0, True],
		'create_hdf5': 		[0, 1, 1], # sector, camera, ccd
		'methods': 			['linpsf', 'aperture'] # photometry methods
	}

	multi_star_noise_variable = {
		'name':				'multi_star_noise_variable',
		'ignore_mov_kernel': 	False,
		'run_simulateFITS': 	[2000, 150, True, True, False, True, True, False, 1000, True],
		'create_hdf5': 		[0, 1, 1], # sector, camera, ccd
		'methods': 			['linpsf', 'aperture'] # photometry methods
	}

	multi_star_no_noise_jitter_inaccurate_variable = {
		'name':				'multi_star_no_noise_jitter_inaccurate_variable',
		'ignore_mov_kernel': 	False,
		'run_simulateFITS': 	[2000, 150, True, True, True, False, True, True, 1000, True],
		'create_hdf5': 		[0, 1, 1], # sector, camera, ccd
		'methods': 			['linpsf', 'aperture'] # photometry methods
	}

	multi_star_noise_jitter_inaccurate_variable = {
		'name':				'multi_star_noise_jitter_inaccurate_variable',
		'ignore_mov_kernel': 	False,
		'run_simulateFITS': 	[2000, 150, True, True, True, True, True, True, 1000, True],
		'create_hdf5': 		[0, 1, 1], # sector, camera, ccd
		'methods': 			['linpsf', 'aperture'] # photometry methods
	}


	# Collect dictionaries in list:
	simulations = [
				multi_star_no_noise,
				multi_star_noise,
				multi_star_jitter,
				multi_star_inaccurate,
				multi_star_variable,
				multi_star_noise_jitter,
				multi_star_noise_inaccurate,
				multi_star_noise_variable,
				multi_star_no_noise_jitter_inaccurate_variable,
				multi_star_noise_jitter_inaccurate_variable
				]
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
		Nstars =				simulation['run_simulateFITS'][0]
		Ntimes =				simulation['run_simulateFITS'][1]
		save_images =		simulation['run_simulateFITS'][2]
		overwrite_images =	simulation['run_simulateFITS'][3]
		include_jitter =		simulation['run_simulateFITS'][4]
		include_noise =		simulation['run_simulateFITS'][5]
		include_bkg =		simulation['run_simulateFITS'][6]
		inaccurate_catalog =	simulation['run_simulateFITS'][7]
		Nvariables =			simulation['run_simulateFITS'][8]
		multiprocess =		simulation['run_simulateFITS'][9]
		# TODO: add the rest of the parameters to the simulateFITS call
		simulateFITS(Nstars=Nstars, Ntimes=Ntimes,
					save_images=save_images, overwrite_images=overwrite_images,
					include_jitter=include_jitter, include_noise=include_noise,
					include_bkg=include_bkg, inaccurate_catalog=inaccurate_catalog,
					Nvariables=Nvariables, multiprocess=multiprocess)

		# Run create_hdf5 from prepare_photometry.py:
		logger.info("Running create_hdf5 from prepare_photometry.py")
		create_hdf5(
			simulation['input_folder'],
			cameras = simulation['create_hdf5'][1],
			ccds    = simulation['create_hdf5'][2]
		)
