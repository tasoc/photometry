#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Create catalogs of stars in a given TESS observing sector.

Example:
	In order to create catalogs for sector 14 simple call this program from the
	commandline like so:

	>>> python make_catalog.py 14

	This will create the catalog files (`*.sqlite`) corresponding to sector 14
	in the directory defined in the ``TESSPHOT_INPUT`` envirnonment variable.

Note:
	This function requires the user to be connected to the TASOC network
	at Aarhus University. It connects to the TASOC database to get a complete
	list of all stars in the TESS Input Catalog (TIC), which is a very large
	table.

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

from __future__ import division, with_statement, print_function, absolute_import
import argparse
import logging
from photometry.catalog import make_catalog

#------------------------------------------------------------------------------
if __name__ == '__main__':

	# Parse command line arguments:
	parser = argparse.ArgumentParser(description='Create CATALOG files for TESS Photometry.')
	parser.add_argument('-d', '--debug', help='Print debug messages.', action='store_true')
	parser.add_argument('-q', '--quiet', help='Only report warnings and errors.', action='store_true')
	parser.add_argument('-o', '--overwrite', help='Overwrite existing files.', action='store_true')
	parser.add_argument('--camera', type=int, choices=(1,2,3,4), default=None, help='TESS Camera. Default is to run all cameras.')
	parser.add_argument('--ccd', type=int, choices=(1,2,3,4), default=None, help='TESS CCD. Default is to run all CCDs.')
	parser.add_argument('sector', type=int, help='TESS observing sector to generate catalogs for.')
	args = parser.parse_args()

	# Set logging level:
	logging_level = logging.INFO
	if args.quiet:
		logging_level = logging.WARNING
	elif args.debug:
		logging_level = logging.DEBUG

	# Setup logging:
	formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
	logger = logging.getLogger('photometry')
	logger.setLevel(logging.INFO)
	console = logging.StreamHandler()
	console.setFormatter(formatter)
	logger.addHandler(console)

	# Run the program:
	make_catalog(args.sector, cameras=args.camera, ccds=args.ccd, overwrite=args.overwrite)
