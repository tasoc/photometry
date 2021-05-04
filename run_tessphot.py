#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Command-line utility to run TESS photometry from command-line.

Example:
	To run a random star from the TODO list:

	>>> python run_tessphot.py --random

Example:
	To run a specific star, you can provide the TIC-identifier:

	>>> python run_tessphot.py --starid=182092046

Example:
	You can be very specific in the photometry methods and input to use.
	The following example runs PSF photometry on Target Pixel Files (tpf) of TIC 182092046,
	and produces plots in the output directory as well.

	>>> python run_tessphot.py --source=tpf --method=psf --plot --starid=182092046

Note:
	run_tessphot is only meant for small tests and running single stars.
	For large scale calculation with many stars, you should use m:py:func:`mpi_scheduler`.

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import os
import argparse
import logging
import functools
from timeit import default_timer
from photometry import tessphot, TaskManager

#--------------------------------------------------------------------------------------------------
def main():
	# Parse command line arguments:
	parser = argparse.ArgumentParser(description='Run TESS Photometry pipeline on single star.')
	parser.add_argument('-d', '--debug', help='Print debug messages.', action='store_true')
	parser.add_argument('-q', '--quiet', help='Only report warnings and errors.', action='store_true')
	parser.add_argument('-o', '--overwrite', help='Overwrite existing results.', action='store_true')
	parser.add_argument('-p', '--plot', help='Save plots when running.', action='store_true')
	parser.add_argument('-t', '--test', help='Use test data and ignore TESSPHOT_INPUT environment variable.', action='store_true')
	parser.add_argument('-m', '--method', choices=('aperture', 'psf', 'linpsf', 'halo'), default=None, help='Photometric method to use.')

	group = parser.add_argument_group('Filter which targets to run')
	group.add_argument('--all', help='Run all stars, one by one. Please consider using the MPI program instead.', action='store_true')
	group.add_argument('-r', '--random', help='Run on random target from TODO-list.', action='store_true')
	group.add_argument('--priority', type=int, default=None, action='append', help='Priority of target.')
	group.add_argument('--starid', type=int, default=None, action='append', help='TIC identifier of target.')
	group.add_argument('--sector', type=int, default=None, action='append', help='TESS Sector. Default is to run all Sectors.')
	group.add_argument('--cadence', type=int, choices=(20,120,600,1800), default=None, action='append', help='Observing cadence. Default is to run all cadences.')
	group.add_argument('--camera', type=int, choices=(1,2,3,4), default=None, action='append', help='TESS Camera. Default is to run all cameras.')
	group.add_argument('--ccd', type=int, choices=(1,2,3,4), default=None, action='append', help='TESS CCD. Default is to run all CCDs.')
	group.add_argument('--datasource', type=str, choices=('ffi', 'tpf'), default=None, help='Data-source to load.')
	group.add_argument('--tmag_min', type=float, default=None, help='Lower/bright limit on Tmag.')
	group.add_argument('--tmag_max', type=float, default=None, help='Upper/faint limit on Tmag.')

	parser.add_argument('--version', type=int, required=True, help='Data release number to store in output files.')
	parser.add_argument('--output', type=str, help='Directory to put lightcurves into.', nargs='?', default=None)
	parser.add_argument('input_folder', type=str, help='Directory to create catalog files in.', nargs='?', default=None)
	args = parser.parse_args()

	# Make sure at least one setting is given:
	if not args.all and args.starid is None and args.priority is None and not args.random:
		parser.error("Please select either a specific STARID or RANDOM.")

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
	logger_parent = logging.getLogger('photometry')
	logger_parent.addHandler(console)
	logger_parent.setLevel(logging_level)

	# Get input and output folder from environment variables:
	test_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), 'tests', 'input'))
	if args.test:
		input_folder = test_folder
	elif args.input_folder:
		input_folder = args.input_folder
	else:
		input_folder = os.environ.get('TESSPHOT_INPUT', test_folder)

	if os.path.isfile(input_folder):
		input_folder = os.path.dirname(input_folder)

	if args.output:
		output_folder = args.output
	else:
		output_folder = os.environ.get('TESSPHOT_OUTPUT', os.path.join(input_folder, 'lightcurves'))

	logger.info("Loading input data from '%s'", input_folder)
	logger.info("Putting output data in '%s'", output_folder)

	# Constraints on which targets to process:
	constraints = {
		'priority': args.priority,
		'starid': args.starid,
		'sector': args.sector,
		'cadence': args.cadence,
		'camera': args.camera,
		'ccd': args.ccd,
		'datasource': args.datasource,
		'tmag_min': args.tmag_min,
		'tmag_max': args.tmag_max,
	}

	# Create partial function of tessphot, setting the common keywords:
	f = functools.partial(
		tessphot,
		input_folder=input_folder,
		output_folder=output_folder,
		plot=args.plot,
		version=args.version)

	# Run the program:
	with TaskManager(input_folder, overwrite=args.overwrite, cleanup_constraints=constraints) as tm:
		while True:
			if args.random:
				task = tm.get_random_task()
			else:
				task = tm.get_task(**constraints)

			if task is None:
				parser.error("No task found matching constraints.")
				break

			# If specific method provided, overwrite whatever
			# was defined in the TODO-file:
			if args.method:
				task['method'] = args.method

			result = task.copy()
			del task['priority'], task['tmag']

			t1 = default_timer()
			pho = f(**task)
			t2 = default_timer()

			# Construct result message:
			result.update({
				'status': pho.status,
				'method_used': pho.method,
				'time': t2 - t1,
				'details': pho._details
			})
			tm.save_result(result)

			if not args.all:
				break

#--------------------------------------------------------------------------------------------------
if __name__ == '__main__':
	main()
