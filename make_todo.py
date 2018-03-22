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
import six
import os
import numpy as np
import logging
import sqlite3
import warnings
warnings.filterwarnings('ignore', category=FutureWarning, module='h5py')
import h5py
import itertools
import argparse
from astropy.table import Table, vstack
from astropy.io import fits
from astropy.wcs import WCS
from photometry.utilities import find_tpf_files
import multiprocessing

def _ffi_todo_wrapper(args):
	return _ffi_todo(*args)

def _ffi_todo(input_folder, camera, ccd):

	logger = logging.getLogger(__name__)

	# Create the TODO list as a table which we will fill with targets:
	cat = Table(
		names=('starid', 'camera', 'ccd', 'datasource', 'tmag'),
		dtype=('int64', 'int32', 'int32', 'S3', 'float32')
	)

	# See if there are any FFIs for this camera and ccd.
	# We just check if an HDF5 file exist.
	hdf5_file = os.path.join(input_folder, 'camera{0:d}_ccd{1:d}.hdf5'.format(camera, ccd))
	if os.path.exists(hdf5_file):
		# Load the relevant information from the HDF5 file for this camera and ccd:
		with h5py.File(hdf5_file, 'r') as hdf:
			hdr_string = hdf['wcs'][0]
			if not isinstance(hdr_string, six.string_types): hdr_string = hdr_string.decode("utf-8") # For Python 3
			wcs = WCS(header=fits.Header().fromstring(hdr_string))
			offset_rows = hdf['images'].attrs.get('PIXEL_OFFSET_ROW', 0)
			offset_cols = hdf['images'].attrs.get('PIXEL_OFFSET_COLUMN', 0)
			image_shape = hdf['images']['0000'].shape

		# Load the corresponding catalog:
		catalog_file = os.path.join(input_folder, 'catalog_camera{0:d}_ccd{1:d}.sqlite'.format(camera, ccd))
		conn = sqlite3.connect(catalog_file)
		conn.row_factory = sqlite3.Row
		cursor = conn.cursor()

		# Find all the stars in the catalog brigher than a certain limit:
		cursor.execute("SELECT starid,tmag,ra,decl FROM catalog WHERE tmag < 15 ORDER BY tmag;")
		for k, row in enumerate(cursor.fetchall()):
			logger.info("%011d - %.3f", row['starid'], row['tmag'])

			# Calculate the position of this star on the CCD using the WCS:
			ra_dec = np.atleast_2d([row['ra'], row['decl']])
			x, y = wcs.all_world2pix(ra_dec, 0)[0]

			# Subtract the pixel offset if there is one:
			x -= offset_cols
			y -= offset_rows

			# If the target falls outside silicon, do not add it to the todo list:
			if x < 0 or y < 0 or x > image_shape[1] or y > image_shape[0]:
				continue

			# The targets is on silicon, so add it to the todo list:
			cat.add_row({
				'starid': row['starid'],
				'camera': camera,
				'ccd': ccd,
				'datasource': 'ffi',
				'tmag': row['tmag']
			})

		cursor.close()
		conn.close()

	return cat

#------------------------------------------------------------------------------
def make_todo(input_folder=None):
	"""
	Create the TODO list which is used by the pipeline to keep track of the
	targets that needs to be processed.

	Will create the file `todo.sqlite` in the directory.
	It will be overwritten if it already exists.

	Parameters:
		input_folder (string, optional): Input folder to create TODO list for.
			If ``None``, the input directory in the environment variable ``TESSPHOT_INPUT`` is used.

	.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
	"""

	logger = logging.getLogger(__name__)

	# Check the input folder, and load the default if not provided:
	if input_folder is None:
		input_folder = os.environ.get('TESSPHOT_INPUT', os.path.join(os.path.dirname(__file__), 'tests', 'input'))

	# The TODO file that we want to create. Delete it if it already exits:
	todo_file = os.path.join(input_folder, 'todo.sqlite')
	if os.path.exists(todo_file): os.remove(todo_file)

	# Create the TODO list as a table which we will fill with targets:
	cat = Table(
		names=('starid', 'camera', 'ccd', 'datasource', 'tmag'),
		dtype=('int64', 'int32', 'int32', 'S3', 'float32')
	)

	# Load list of all Target Pixel files in the directory:
	tpf_files = find_tpf_files(input_folder)
	logger.info("Number of TPF files: %d", len(tpf_files))
	for fname in tpf_files:
		logger.debug("Processing TPF file: '%s'", fname)
		with fits.open(fname, memmap=True, mode='readonly') as hdu:
			starid = hdu[0].header['TICID']
			camera = hdu[0].header['CAMERA']
			ccd = hdu[0].header['CCD']
			tmag = hdu[0].header['TESSMAG']

			cat.add_row({
				'starid': starid,
				'camera': camera,
				'ccd': ccd,
				'datasource': 'tpf',
				'tmag': tmag
			})

		# TODO: Load all other targets in this stamp!

	# Find all targets in Full Frame Images:
	inputs = itertools.product([input_folder], [1,2,3,4], [1,2,3,4])

	# Open a pool of workers:
	logger.info("Starting pool of workers for FFIs...")
	threads = int(os.environ.get('SLURM_CPUS_PER_TASK', multiprocessing.cpu_count()))
	threads = min(threads, 16) # No reason to use more than the number of jobs in total
	logger.info("Using %d processes.", threads)

	pool = multiprocessing.Pool(threads)
	ccds_done = 0
	for cat2 in pool.imap_unordered(_ffi_todo_wrapper, inputs):
		cat = vstack([cat, cat2], join_type='exact')
		ccds_done += 1
		logger.info("CCDs done: %d/16", ccds_done)
	pool.close()
	pool.join()

	# Sort the final list:
	cat.sort('tmag')

	# TODO: Remove duplicates!
	logger.info("Removing duplicate entries...")
	_, idx = np.unique(cat[('starid', 'camera', 'ccd', 'datasource')], return_index=True, axis=0)
	cat = cat[np.sort(idx)]

	# TODO: Can we make decisions already now on methods?

	# Write the TODO list to the SQLite database file:
	logger.info("Writing TODO file...")
	conn = sqlite3.connect(todo_file)
	cursor = conn.cursor()

	cursor.execute("""CREATE TABLE todolist (
		priority BIGINT NOT NULL,
		starid BIGINT NOT NULL,
		datasource TEXT NOT NULL DEFAULT 'ffi',
		camera INT NOT NULL,
		ccd INT NOT NULL,
		method TEXT DEFAULT NULL,
		tmag REAL,
		status INT DEFAULT NULL
	);""")

	for pri, row in enumerate(cat):
		cursor.execute("INSERT INTO todolist (priority,starid,camera,ccd,datasource,tmag) VALUES (?,?,?,?,?,?);", (
			pri+1,
			int(row['starid']),
			int(row['camera']),
			int(row['ccd']),
			row['datasource'],
			float(row['tmag'])
		))

	cursor.execute("CREATE UNIQUE INDEX priority_idx ON todolist (priority);")
	cursor.execute("CREATE UNIQUE INDEX starid_datasource_idx ON todolist (starid, datasource);")
	cursor.execute("CREATE INDEX status_idx ON todolist (status);")
	cursor.execute("CREATE INDEX starid_idx ON todolist (starid);")
	conn.commit()
	cursor.close()
	conn.close()

	logger.info("TODO done.")

#------------------------------------------------------------------------------
if __name__ == '__main__':

	# Parse command line arguments:
	parser = argparse.ArgumentParser(description='Create TODO file for TESS Photometry.')
	parser.add_argument('-d', '--debug', help='Print debug messages.', action='store_true')
	parser.add_argument('-q', '--quiet', help='Only report warnings and errors.', action='store_true')
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

	make_todo(args.input_folder)
