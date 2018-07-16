#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Create the TODO list which is used by the pipeline to keep track of the
targets that needs to be processed.

Example:
	In order to create to TODO list for the directory in the ``TESSPHOT_INPUT``
	envirnonment variable simply run the program without any further input:

	>>> python make_todo.py

	This will create the file ``todo.sqlite`` in the directory defined in the
	``TESSPHOT_INPUT`` envirnonment variable.

Example:
	If you want to create the TODO file for a specific directory (ignoring the
	``TESSPHOT_INPUT`` envirnonment variable), you can simply call the script
	with the directory you want to process:

	>>> python make_todo.py /where/ever/you/want/

Note:
	This program assumes that the directory already contains "catalog" files for
	the given sector. These can be create using the :py:func:`make_catalog`
	utility.

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

from __future__ import division, with_statement, print_function, absolute_import
import argparse
import logging
import os.path
from photometry.todolist import make_todo

#------------------------------------------------------------------------------
if __name__ == '__main__':

	# Parse command line arguments:
	parser = argparse.ArgumentParser(description='Create TODO file for TESS Photometry.')
	parser.add_argument('-d', '--debug', help='Print debug messages.', action='store_true')
	parser.add_argument('-q', '--quiet', help='Only report warnings and errors.', action='store_true')
	parser.add_argument('-o', '--overwrite', help='Overwrite existing TODO file.', action='store_true')
	parser.add_argument('--camera', type=int, choices=(1,2,3,4), default=None, help='TESS Camera. Default is to run all cameras.')
	parser.add_argument('--ccd', type=int, choices=(1,2,3,4), default=None, help='TESS CCD. Default is to run all CCDs.')
	parser.add_argument('input_folder', type=str, help='TESSPhot input directory to create TODO file in.', nargs='?', default=None)
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
	logger = logging.getLogger(__name__)
	logger.addHandler(console)
	logger.setLevel(logging_level)
	logger_parent = logging.getLogger('photometry')
	logger_parent.addHandler(console)
	logger_parent.setLevel(logging_level)

	# Check that the given input directory is indeed a directory:
	if args.input_folder is not None and not os.path.isdir(args.input_folder):
		parser.error("The given path does not exist or is not a directory")

	# Run the program:
	make_todo(args.input_folder, cameras=args.camera, ccds=args.ccd, overwrite=args.overwrite)
