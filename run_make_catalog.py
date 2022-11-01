#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create catalogs of stars in a given TESS observing sector.

Example:
	In order to create catalogs for sector 14 simple call this program from the
	command-line like so:

	>>> python run_make_catalog.py 14

	This will create the catalog files (`*.sqlite`) corresponding to sector 14
	in the directory defined in the ``TESSPHOT_INPUT`` environment variable.

Note:
	This function requires the user to be connected to the TASOC network
	at Aarhus University. It connects to the TASOC database to get a complete
	list of all stars in the TESS Input Catalog (TIC), which is a very large
	table.

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import argparse
import logging
import photometry

#--------------------------------------------------------------------------------------------------
def main():
	# Parse command line arguments:
	parser = argparse.ArgumentParser(description='Create CATALOG files for TESS Photometry.')
	parser.add_argument('-d', '--debug', help='Print debug messages.', action='store_true')
	parser.add_argument('-q', '--quiet', help='Only report warnings and errors.', action='store_true')
	parser.add_argument('-o', '--overwrite', help='Overwrite existing files.', action='store_true')
	group = parser.add_argument_group('Filter which CCDs to include')
	group.add_argument('--camera', type=int, choices=(1,2,3,4), default=None, action='append', help='TESS Camera. Default is to run all cameras.')
	group.add_argument('--ccd', type=int, choices=(1,2,3,4), default=None, action='append', help='TESS CCD. Default is to run all CCDs.')
	group = parser.add_argument_group('Settings')
	group.add_argument('--buffer', type=float, default=0.2, help="Buffer in degrees around each CCD to include in catalogs.")
	parser.add_argument('sector', type=int, help='TESS observing sector to generate catalogs for.')
	parser.add_argument('input_folder', type=str, help='Directory to create catalog files in.', nargs='?', default=None)
	args = parser.parse_args()

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
	logger = logging.getLogger('photometry')
	logger.setLevel(logging_level)
	logger.addHandler(console)

	# Make sure we have turned plotting to non-interactive:
	photometry.plots.plots_noninteractive()

	# Run the program:
	photometry.catalog.make_catalog(args.sector,
		input_folder=args.input_folder,
		cameras=args.camera,
		ccds=args.ccd,
		overwrite=args.overwrite,
		coord_buffer=args.buffer)

#--------------------------------------------------------------------------------------------------
if __name__ == '__main__':
	main()
