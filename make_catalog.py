#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Create catalogs of stars in a given TESS observing sector.

Note:
	This function requires the user to be connected to the TASOC network
	at Aarhus University. It connects to the TASOC database to get a complete
	list of all stars in the TESS Input Catalog (TIC), which is a very large
	table.

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

from __future__ import division, with_statement, print_function, absolute_import
import argparse
import numpy as np
import os
import sqlite3
import logging
import itertools
from tasoc_db import TASOC_DB
from photometry.utilities import add_proper_motion, load_settings

#------------------------------------------------------------------------------
def make_catalog(sector, cameras=None, ccds=None, coord_buffer=0.1, overwrite=False):
	"""
	Create catalogs of stars in a given TESS observing sector.

	Parameters:
		sector (integer): TESS observing sector.
		cameras (iterable or None): TESS cameras (1-4) to create catalogs for. If ``None`` all cameras are created.
		ccds (iterable or None): TESS ccds (1-4) to create catalogs for. If ``None`` all ccds are created.
		coord_buffer (float): Buffer in degrees around each CCD to include in catalogs. Default=0.1.
		overwrite (boolean): Overwrite existing catalogs. Default=``False``.

	Note:
		This function requires the user to be connected to the TASOC network
		at Aarhus University. It connects to the TASOC database to get a complete
		list of all stars in the TESS Input Catalog (TIC), which is a very large
		table.

	.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
	"""

	logger = logging.getLogger(__name__)

	if cameras is None: cameras = [1, 2, 3, 4]
	if ccds is None: ccds = [1, 2, 3, 4]

	settings = load_settings(sector=sector)
	sector_reference_time = settings['reference_time']
	epoch = (sector_reference_time - 2451544.5)/365.25

	input_folder = os.environ.get('TESSPHOT_INPUT', os.path.join(os.path.dirname(__file__), 'tests', 'input'))
	logger.info("Saving results to '%s'", input_folder)

	# Open connection to the central TASOC database.
	# This requires that users are on the TASOC network at Aarhus University.
	with TASOC_DB() as tasocdb:
		# Loop through the cameras and CCDs that should have catalogs created:
		for camera, ccd in itertools.product(cameras, ccds):

			logger.info("Running SECTOR=%s, CAMERA=%s, CCD=%s", sector, camera, ccd)

			# Create SQLite file:
			catalog_file = os.path.join(input_folder, 'catalog_camera{0:d}_ccd{1:d}.sqlite'.format(camera, ccd))
			if os.path.exists(catalog_file):
				if overwrite:
					os.remove(catalog_file)
				else:
					logger.info("Already done")
					continue
			conn = sqlite3.connect(catalog_file)
			conn.row_factory = sqlite3.Row
			cursor = conn.cursor()

			# Table which stores information used to generate catalog:
			cursor.execute("""CREATE TABLE settings (
				sector INT NOT NULL,
				camera INT NOT NULL,
				ccd INT NOT NULL,
				reference_time DOUBLE PRECISION NOT NULL,
				epoch DOUBLE PRECISION NOT NULL,
				coord_buffer DOUBLE PRECISION NOT NULL,
				footprint TEXT NOT NULL
			);""")

			cursor.execute("""CREATE TABLE catalog (
				starid BIGINT PRIMARY KEY NOT NULL,
				ra DOUBLE PRECISION NOT NULL,
				decl DOUBLE PRECISION NOT NULL,
				ra_J2000 DOUBLE PRECISION NOT NULL,
				decl_J2000 DOUBLE PRECISION NOT NULL,
				tmag REAL NOT NULL
			);""")

			# Get the footprint on the sky of this sector:
			tasocdb.cursor.execute("SELECT footprint FROM tasoc.pointings WHERE sector=%s AND camera=%s AND ccd=%s;", (
				sector,
				camera,
				ccd
			))
			footprint = tasocdb.cursor.fetchone()
			if footprint is None:
				raise IOError("The given sector, camera, ccd combination was not found in TASOC database: (%s,%s,%s)", sector, camera, ccd)
			footprint = footprint[0]

			# Transform footprint into numpy array:
			a = footprint[2:-2].split('),(')
			a = np.array([b.split(',') for b in a], dtype='float64')

			# Center of footprint:
			origin = np.mean(a, axis=0)

			# Add buffer zone, by expanding polygon away from origin:
			for k in range(a.shape[0]):
				vec = a[k,:] - origin
				uvec = vec/np.linalg.norm(vec)
				a[k,:] = origin + vec + uvec*coord_buffer

			# Make footprint into string that will be understood by database:
			footprint = '{' + ",".join([str(s) for s in a.flatten()]) + '}'
			logger.info(footprint)

			# Save settings to SQLite:
			cursor.execute("INSERT INTO settings (sector,camera,ccd,reference_time,epoch,coord_buffer,footprint) VALUES (?,?,?,?,?,?,?);", (
				sector,
				camera,
				ccd,
				sector_reference_time,
				epoch + 2000.0,
				coord_buffer,
				footprint
			))
			conn.commit()

			# Query the TESS Input Catalog table for all stars in the footprint.
			# This is a MASSIVE table, so this query may take a while.
			# Don't include anything left over from before 2016-04-11 (TIC-2).
			tasocdb.cursor.execute("SELECT starid,ra,decl,pm_ra,pm_decl,\"Tmag\",version FROM tasoc.tic WHERE q3c_poly_query(ra, decl, %s) AND version > '2016-04-11'::date;", (
				footprint,
			))

			# We need a list of when the sectors are in time:
			logger.info('Projecting catalog {0:.3f} years relative to 2000'.format(epoch))

			for row in tasocdb.cursor.fetchall():
				# Add the proper motion to each coordinate:
				if row['pm_ra'] and row['pm_decl']:
					ra, dec = add_proper_motion(row['ra'], row['decl'], row['pm_ra'], row['pm_decl'], sector_reference_time, epoch=2000.0)
					logger.debug("(%f, %f) => (%f, %f)", row[1], row[2], ra, dec)
				else:
					ra = row['ra']
					dec = row['decl']

				# Save the coordinates in SQLite database:
				cursor.execute("INSERT INTO catalog (starid,ra,decl,ra_J2000,decl_J2000,tmag) VALUES (?,?,?,?,?,?);", (
					int(row['starid']),
					ra,
					dec,
					row['ra'],
					row['decl'],
					row['Tmag']
				))

			cursor.execute("CREATE UNIQUE INDEX starid_idx ON catalog (starid);")
			cursor.execute("CREATE INDEX ra_dec_idx ON catalog (ra, decl);")
			conn.commit()
			cursor.close()
			conn.close()

		logger.info("Catalog done.")

	logger.info("All catalogs done.")

#------------------------------------------------------------------------------
if __name__ == '__main__':

	# Parse command line arguments:
	parser = argparse.ArgumentParser(description='Create CATALOG files for TESS Photometry.')
	parser.add_argument('-d', '--debug', help='Print debug messages.', action='store_true')
	parser.add_argument('-q', '--quiet', help='Only report warnings and errors.', action='store_true')
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
	logger = logging.getLogger(__name__)
	logger.setLevel(logging.INFO)
	console = logging.StreamHandler()
	console.setFormatter(formatter)
	logger.addHandler(console)

	# Run the program:
	make_catalog(args.sector)
