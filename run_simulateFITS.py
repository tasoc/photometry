#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Command-line utility to simulate TESS FITS images for the photometry pipeline.

Structure inspired by `run_tessphot` by Rasmus Handberg.

.. codeauthor:: Jonas Svenstrup Hansen <jonas.svenstrup@gmail.com>
"""

#import os
import argparse
#import logging
from simulation.simulateFITS import simulateFITS

if __name__ == '__main__':
	
#	# Set logging level:
#	logging_level = logging.INFO
	
	# Parse command-line arguments:
	parser = argparse.ArgumentParser(description=
		'Simulate FITS images to be used by the photometry pipeline.')
	parser.add_argument('-s', '--Nstars',
					help='Number of stars in image', default=5)
	parser.add_argument('-t', '--Ntimes', 
					help='Number of time steps and FITS images', default=5)
#	parser.add_argument('-d', '--debug', 
#					help='Print debug messages.',
#					action='store_true')
#	parser.add_argument('-q', '--quiet', 
#					help='Only report warnings and errors.',
#					action='store_true')
	args = parser.parse_args()
	Nstars = args.Nstars
	Ntimes = args.Ntimes

#	if args.quiet:
#		logging_level = logging.WARNING
#	elif args.debug:
#		logging_level = logging.DEBUG

	# Run the program:
	simulateFITS(Nstars=Nstars, Ntimes=Ntimes,
			save_images=True, overwrite_images=True)