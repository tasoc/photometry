#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create the TODO list which is used by the pipeline to keep track of the
targets that needs to be processed.

Example:
	In order to create to TODO list for the directory in the ``TESSPHOT_INPUT``
	environment variable simply run the program without any further input:

	>>> python run_make_todo.py

	This will create the file ``todo.sqlite`` in the directory defined in the
	``TESSPHOT_INPUT`` environment variable.

Example:
	If you want to create the TODO file for a specific directory (ignoring the
	``TESSPHOT_INPUT`` environment variable), you can simply call the script
	with the directory you want to process:

	>>> python run_make_todo.py /where/ever/you/want/

Note:
	This program assumes that the directory already contains "catalog" files for
	the given sector. These can be create using the :func:`run_make_catalog`
	utility.

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import argparse
import logging
import os.path
import photometry

#------------------------------------------------------------------------------
def main():
	# Parse command line arguments:
	parser = argparse.ArgumentParser(description='Create TODO file for TESS Photometry.')
	parser.add_argument('-d', '--debug', help='Print debug messages.', action='store_true')
	parser.add_argument('-q', '--quiet', help='Only report warnings and errors.', action='store_true')
	parser.add_argument('-o', '--overwrite', help='Overwrite existing TODO file.', action='store_true')
	group = parser.add_argument_group('Filter which targets to include')
	group.add_argument('--sector', type=int, default=None, action='append', help='TESS Sector. Default is to run all sectors.')
	group.add_argument('--camera', type=int, choices=(1,2,3,4), default=None, action='append', help='TESS Camera. Default is to run all cameras.')
	group.add_argument('--ccd', type=int, choices=(1,2,3,4), default=None, action='append', help='TESS CCD. Default is to run all CCDs.')
	parser.add_argument('input_folder', type=str, help='TESSPhot input directory to create TODO file in.', nargs='?', default=None)
	args = parser.parse_args()

	# Check that the given input directory is indeed a directory:
	if args.input_folder is not None and not os.path.isdir(args.input_folder):
		parser.error("The given path does not exist or is not a directory")

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

	# Make sure we have turned plotting to non-interactive:
	photometry.plots.plots_noninteractive()

	# Run the program:
	photometry.todolist.make_todo(args.input_folder,
		sectors=args.sector,
		cameras=args.camera,
		ccds=args.ccd,
		overwrite=args.overwrite)

#--------------------------------------------------------------------------------------------------
if __name__ == '__main__':
	main()
