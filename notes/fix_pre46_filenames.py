#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rename outout FITS lightcurve files from old to new naming scheme.

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import logging
import argparse
import os.path
import re
from tqdm import tqdm
import sys
if sys.path[0] != os.path.abspath(os.path.join(os.path.dirname(__file__), '..')):
	sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from photometry import TaskManager
from photometry.utilities import TqdmLoggingHandler

#--------------------------------------------------------------------------------------------------
if __name__ == '__main__':

	# Parse command line arguments:
	parser = argparse.ArgumentParser(description='Rename output FITS lightcurve files to new naming scheme.')
	parser.add_argument('-d', '--debug', help='Print debug messages.', action='store_true')
	parser.add_argument('-q', '--quiet', help='Only report warnings and errors.', action='store_true')
	parser.add_argument('input_folder', type='str', help='Input directory')
	args = parser.parse_args()

	# Set logging level:
	logging_level = logging.INFO
	if args.quiet:
		logging_level = logging.WARNING
	elif args.debug:
		logging_level = logging.DEBUG

	# Setup logging:
	formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
	console = TqdmLoggingHandler()
	console.setFormatter(formatter)
	logger = logging.getLogger(__name__)
	logger.addHandler(console)
	logger.setLevel(logging_level)

	inp = args.input_folder
	input_folder = os.path.abspath(os.path.dirname(inp))
	logger.info("Renaming files in %s", input_folder)
	if not os.path.isdir(input_folder):
		parser.error("Input folder does not exist")

	tqdm_settings = {'disable': not logger.isEnabledFor(logging.INFO)}

	regex_old = re.compile(r'^tess(\d+)-s(\d+)-c(\d+)-dr(\d+)-v(\d+)-tasoc_lc\.fits\.gz$')
	regex_new = re.compile(r'^tess(\d+)-s(\d+)-[1-4]-[1-4]-c(\d+)-dr(\d+)-v(\d+)-tasoc_lc\.fits\.gz$')

	with TaskManager(input_folder) as tm:

		tm.cursor.execute("SELECT todolist.priority,lightcurve,camera,ccd FROM todolist INNER JOIN diagnostics ON todolist.priority=diagnostics.priority WHERE lightcurve IS NOT NULL;")
		results = tm.cursor.fetchall()

		try:
			for k, row in enumerate(tqdm(results, **tqdm_settings)):

				fpath = os.path.join(input_folder, row['lightcurve'])
				if not os.path.isfile(fpath):
					logger.error("File not found: '%s'", row['lightcurve'])
					continue

				bname = os.path.basename(fpath)
				m = regex_old.match(bname)
				if m:
					# The new filename:
					filename = 'tess{starid:011d}-s{sector:03d}-{camera:d}-{ccd:d}-c{cadence:04d}-dr{datarel:02d}-v{version:02d}-tasoc_lc.fits.gz'.format(
						starid=int(m.group(1)),
						sector=int(m.group(2)),
						camera=row['camera'],
						ccd=row['ccd'],
						cadence=int(m.group(3)),
						datarel=int(m.group(4)),
						version=int(m.group(5))
					)

					newpath = os.path.join(os.path.dirname(fpath), filename)
					relpath = os.path.relpath(newpath, input_folder)

					os.rename(fpath, newpath)
					tm.cursor.execute("UPDATE diagnostics SET lightcurve=? WHERE priority=?;", [relpath, row['priority']])

					if k % 1000 == 0:
						tm.conn.commit()

				else:
					m = regex_new.match(bname)
					if not m:
						logger.error("Does not match old naming scheme")

			tm.conn.commit()
		except (KeyboardInterrupt, SystemExit):
			tm.conn.commit()
		except:
			tm.conn.rollback()
			raise

	print("Done")
