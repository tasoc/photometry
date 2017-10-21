#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Command-line utility to run TESS photometry of single star.

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

from __future__ import with_statement, print_function
import os
import argparse
import logging
from photometry import tessphot

#------------------------------------------------------------------------------
if __name__ == '__main__':

	logging_level = logging.INFO

	parser = argparse.ArgumentParser(description='Run TESS Photometry pipeline on single star.')
	parser.add_argument('-m', '--method', help='Photometric method to use.', default='aperture', choices=('aperture', 'psf'))
	parser.add_argument('-d', '--debug', help='Print debug messages.', action='store_true')
	parser.add_argument('-q', '--quiet', help='Only report warnings and errors.', action='store_true')
	parser.add_argument('starid', type=int, help='TIC identifier of target.')
	args = parser.parse_args()
	starid = args.starid
	method = args.method

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

	# Get input and output folder from enviroment variables:
	input_folder = os.environ.get('TESSPHOT_INPUT', os.path.abspath(os.path.join(os.path.dirname(__file__), 'tests', 'input')))
	output_folder = os.environ.get('TESSPHOT_OUTPUT', os.path.abspath('.'))
	logger.info("Loading input data from '%s'", input_folder)
	logger.info("Putting output data in '%s'", output_folder)
	
	# Run the program:
	pho = tessphot(starid, method, input_folder=input_folder, output_folder=output_folder)

	# TODO: Write out the results?
	if not args.quiet:
		print("=======================")
		print("STATUS: {0}".format(pho.status.name))
