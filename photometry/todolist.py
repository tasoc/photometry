#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Create the TODO list which is used by the pipeline to keep track of the
targets that needs to be processed.
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
import functools
from astropy.table import Table, vstack
from astropy.io import fits
from astropy.wcs import WCS
from timeit import default_timer
from .utilities import find_tpf_files, sphere_distance
import multiprocessing

def calc_cbv_area(catalog_row, settings):
	# The distance from the camera centre to the corner furthest away:
	camera_radius = np.sqrt( 12**2 + 12**2 ) # np.max(sphere_distance(a[:,0], a[:,1], settings['camera_centre_ra'], settings['camera_centre_dec']))

	# Distance to centre of the camera in degrees:
	camera_centre_dist = sphere_distance(catalog_row['ra'], catalog_row['decl'], settings['camera_centre_ra'], settings['camera_centre_dec'])

	cbv_area = settings['camera']*100 + settings['ccd']*10

	if camera_centre_dist < 0.25*camera_radius:
		cbv_area += 1
	elif camera_centre_dist < 0.5*camera_radius:
		cbv_area += 2
	elif camera_centre_dist < 0.75*camera_radius:
		cbv_area += 3
	else:
		cbv_area += 4

	return cbv_area

def _ffi_todo_wrapper(args):
	return _ffi_todo(*args)

def _ffi_todo(input_folder, camera, ccd):

	logger = logging.getLogger(__name__)

	# Create the TODO list as a table which we will fill with targets:
	cat = Table(
		names=('starid', 'camera', 'ccd', 'datasource', 'tmag', 'cbv_area'),
		dtype=('int64', 'int32', 'int32', 'S256', 'float32', 'int32')
	)

	# See if there are any FFIs for this camera and ccd.
	# We just check if an HDF5 file exist.
	hdf5_file = os.path.join(input_folder, 'camera{0:d}_ccd{1:d}.hdf5'.format(camera, ccd))
	if os.path.exists(hdf5_file):
		# Load the relevant information from the HDF5 file for this camera and ccd:
		with h5py.File(hdf5_file, 'r') as hdf:
			if isinstance(hdf['wcs'], h5py.Group):
				refindx = hdf['wcs'].attrs['ref_frame']
				hdr_string = hdf['wcs']['%04d' % refindx][0]
			else:
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

		# Load the settings:
		cursor.execute("SELECT * FROM settings WHERE camera=? AND ccd=? LIMIT 1;", (camera, ccd))
		settings = cursor.fetchone()

		# Find all the stars in the catalog brigher than a certain limit:
		cursor.execute("SELECT starid,tmag,ra,decl FROM catalog WHERE tmag < 15 ORDER BY tmag;")
		for row in cursor.fetchall():
			logger.debug("%011d - %.3f", row['starid'], row['tmag'])

			# Calculate the position of this star on the CCD using the WCS:
			ra_dec = np.atleast_2d([row['ra'], row['decl']])
			x, y = wcs.all_world2pix(ra_dec, 0)[0]

			# Subtract the pixel offset if there is one:
			x -= offset_cols
			y -= offset_rows

			# If the target falls outside silicon, do not add it to the todo list:
			# The reason for the strange 0.5's is that pixel centers are at integers.
			if x < -0.5 or y < -0.5 or x > image_shape[1]-0.5 or y > image_shape[0]-0.5:
				continue

			# Calculate the Cotrending Basis Vector area the star falls in:
			cbv_area = calc_cbv_area(row, settings)

			# The targets is on silicon, so add it to the todo list:
			cat.add_row({
				'starid': row['starid'],
				'camera': camera,
				'ccd': ccd,
				'datasource': 'ffi',
				'tmag': row['tmag'],
				'cbv_area': cbv_area
			})

		cursor.close()
		conn.close()

	return cat

#------------------------------------------------------------------------------
def _tpf_todo(fname, input_folder=None, cameras=None, ccds=None):

	logger = logging.getLogger(__name__)

	# Create the TODO list as a table which we will fill with targets:
	# TODO: Could we avoid fixed-size strings in datasource column?
	cat = Table(
		names=('starid', 'camera', 'ccd', 'datasource', 'tmag', 'cbv_area'),
		dtype=('int64', 'int32', 'int32', 'S256', 'float32', 'int32')
	)

	logger.debug("Processing TPF file: '%s'", fname)
	with fits.open(fname, memmap=True, mode='readonly') as hdu:
		starid = hdu[0].header['TICID']
		camera = hdu[0].header['CAMERA']
		ccd = hdu[0].header['CCD']

		if camera in cameras and ccd in ccds:
			# Load the corresponding catalog:
			catalog_file = os.path.join(input_folder, 'catalog_camera{0:d}_ccd{1:d}.sqlite'.format(camera, ccd))
			if not os.path.exists(catalog_file):
				raise IOError("Catalog file not found: %s" % catalog_file)
			conn = sqlite3.connect(catalog_file)
			conn.row_factory = sqlite3.Row
			cursor = conn.cursor()

			cursor.execute("SELECT * FROM settings WHERE camera=? AND ccd=? LIMIT 1;", (camera, ccd))
			settings = cursor.fetchone()
			if settings is None:
				logger.error("Settings could not be loaded for camera=%d, ccd=%d.", camera, ccd)
				raise ValueError("Settings could not be loaded for camera=%d, ccd=%d." % (camera, ccd))

			# Get information about star:
			cursor.execute("SELECT * FROM catalog WHERE starid=? LIMIT 1;", (starid, ))
			row = cursor.fetchone()
			if row is None:
				logger.error("Starid %d was not found in catalog (camera=%d, ccd=%d).", starid, camera, ccd)
				raise ValueError("Starid %d was not found in catalog (camera=%d, ccd=%d)." %(starid, camera, ccd))

			# Calculate CBV area that target falls in:
			cbv_area = calc_cbv_area(row, settings)

			# Add the main target to the list:
			cat.add_row({
				'starid': starid,
				'camera': camera,
				'ccd': ccd,
				'datasource': 'tpf',
				'tmag': row['tmag'],
				'cbv_area': cbv_area
			})

			# Load all other targets in this stamp:
			# Use the WCS of the stamp to find all stars that fall within
			# the footprint of the stamp.
			image_shape = hdu[2].shape
			wcs = WCS(header=hdu[2].header)
			footprint = wcs.calc_footprint()
			radec_min = np.min(footprint, axis=0)
			radec_max = np.max(footprint, axis=0)
			# TODO: This can fail to find all targets e.g. if the footprint is across the ra=0 line
			cursor.execute("SELECT * FROM catalog WHERE ra BETWEEN ? AND ? AND decl BETWEEN ? AND ? AND starid != ? AND tmag < 15;", (radec_min[0], radec_max[0], radec_min[1], radec_max[1], starid))
			for row in cursor.fetchall():
				# Calculate the position of this star on the CCD using the WCS:
				ra_dec = np.atleast_2d([row['ra'], row['decl']])
				x, y = wcs.all_world2pix(ra_dec, 0)[0]

				# If the target falls outside silicon, do not add it to the todo list:
				# The reason for the strange 0.5's is that pixel centers are at integers.
				if x < -0.5 or y < -0.5 or x > image_shape[1]-0.5 or y > image_shape[0]-0.5:
					continue

				# Add this secondary target to the list:
				# Note that we are storing the starid of the target
				# in which target pixel file the target can be found.
				logger.debug("Adding extra target: TIC %d", row['starid'])
				cat.add_row({
					'starid': row['starid'],
					'camera': camera,
					'ccd': ccd,
					'datasource': 'tpf:' + str(starid),
					'tmag': row['tmag'],
					'cbv_area': cbv_area
				})

			# Close the connection to the catalog SQLite database:
			cursor.close()
			conn.close()

	return cat

#------------------------------------------------------------------------------
def make_todo(input_folder=None, cameras=None, ccds=None, overwrite=False):
	"""
	Create the TODO list which is used by the pipeline to keep track of the
	targets that needs to be processed.

	Will create the file `todo.sqlite` in the directory.

	Parameters:
		input_folder (string, optional): Input folder to create TODO list for.
			If ``None``, the input directory in the environment variable ``TESSPHOT_INPUT`` is used.
		cameras (iterable of integers, optional): TESS camera number (1-4). If ``None``, all cameras will be included.
		ccds (iterable of integers, optional): TESS CCD number (1-4). If ``None``, all cameras will be included.
		overwrite (boolean): Overwrite existing TODO file. Default=``False``.

	Raises:
		IOError: If the specified ``input_folder`` is not an existing directory.

	.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
	"""

	logger = logging.getLogger(__name__)

	# Check the input folder, and load the default if not provided:
	if input_folder is None:
		input_folder = os.environ.get('TESSPHOT_INPUT', os.path.join(os.path.dirname(__file__), 'tests', 'input'))

	# Check that the given input directory is indeed a directory:
	if not os.path.isdir(input_folder):
		raise IOError("The given path does not exist or is not a directory")

	# Make sure cameras and ccds are iterable:
	cameras = (1, 2, 3, 4) if cameras is None else (cameras, )
	ccds = (1, 2, 3, 4) if ccds is None else (ccds, )
	Nccds = len(cameras)*len(ccds)

	# The TODO file that we want to create. Delete it if it already exits:
	todo_file = os.path.join(input_folder, 'todo.sqlite')
	if os.path.exists(todo_file):
		if overwrite:
			os.remove(todo_file)
		else:
			logger.info("TODO file already exists")
			return

	# Create the TODO list as a table which we will fill with targets:
	cat = Table(
		names=('starid', 'camera', 'ccd', 'datasource', 'tmag', 'cbv_area'),
		dtype=('int64', 'int32', 'int32', 'S256', 'float32', 'int32')
	)

	# Load list of all Target Pixel files in the directory:
	tpf_files = find_tpf_files(input_folder)
	logger.info("Number of TPF files: %d", len(tpf_files))

	if len(tpf_files) > 0:
		# Open a pool of workers:
		logger.info("Starting pool of workers for TPFs...")
		threads = int(os.environ.get('SLURM_CPUS_PER_TASK', multiprocessing.cpu_count()))
		threads = min(threads, len(tpf_files)) # No reason to use more than the number of jobs in total
		logger.info("Using %d processes.", threads)
		pool = multiprocessing.Pool(threads)

		# Run the TPF files in parallel:
		tic = default_timer()
		_tpf_todo_wrapper = functools.partial(_tpf_todo, input_folder=input_folder, cameras=cameras, ccds=ccds)
		for cat2 in pool.imap_unordered(_tpf_todo_wrapper, tpf_files):
			cat = vstack([cat, cat2], join_type='exact')

		pool.close()
		pool.join()

		# Amount of time it took to process TPF files:
		toc = default_timer()
		logger.info("Elaspsed time: %f seconds (%f per file)", toc-tic, (toc-tic)/len(tpf_files))

		# Remove secondary TPF targets if they are also the primary target:
		indx_remove = np.zeros(len(cat), dtype='bool')
		cat.add_index('starid')
		for k, row in enumerate(cat):
			if row['datasource'].startswith('tpf:'):
				indx = cat.loc['starid', row['starid']]['datasource'] == 'tpf'
				if np.any(indx):
					indx_remove[k] = True
		cat.remove_indices('starid')
		logger.info("Removing %d secondary TPF files as they are also primary", np.sum(indx_remove))
		cat = cat[~indx_remove]

	# Find all targets in Full Frame Images:
	inputs = itertools.product([input_folder], cameras, ccds)

	# Open a pool of workers:
	logger.info("Starting pool of workers for FFIs...")
	threads = int(os.environ.get('SLURM_CPUS_PER_TASK', multiprocessing.cpu_count()))
	threads = min(threads, Nccds) # No reason to use more than the number of jobs in total
	logger.info("Using %d processes.", threads)

	pool = multiprocessing.Pool(threads)
	ccds_done = 0
	for cat2 in pool.imap_unordered(_ffi_todo_wrapper, inputs):
		cat = vstack([cat, cat2], join_type='exact')
		ccds_done += 1
		logger.info("CCDs done: %d/%d", ccds_done, Nccds)
	pool.close()
	pool.join()

	# Remove duplicates!
	logger.info("Removing duplicate entries...")
	_, idx = np.unique(cat[('starid', 'camera', 'ccd', 'datasource')], return_index=True, axis=0)
	cat = cat[np.sort(idx)]

	# Sort the final list:
	cat.sort('tmag')

	# TODO: Can we make decisions already now on methods?
	# tmag < 2.5 : Halo photometry
	# tmag < 6.5 : Aperture (saturated)
	# tmag > 6.5 : Aperture (with PSF fallback)

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
		status INT DEFAULT NULL,
		cbv_area INT NOT NULL
	);""")

	for pri, row in enumerate(cat):
		cursor.execute("INSERT INTO todolist (priority,starid,camera,ccd,datasource,tmag,cbv_area) VALUES (?,?,?,?,?,?,?);", (
			pri+1,
			int(row['starid']),
			int(row['camera']),
			int(row['ccd']),
			row['datasource'].strip(),
			float(row['tmag']),
			int(row['cbv_area'])
		))

	conn.commit()
	cursor.execute("CREATE UNIQUE INDEX priority_idx ON todolist (priority);")
	cursor.execute("CREATE INDEX starid_datasource_idx ON todolist (starid, datasource);") # FIXME: Should be "UNIQUE", but something is weird in ETE-6?!
	cursor.execute("CREATE INDEX status_idx ON todolist (status);")
	cursor.execute("CREATE INDEX starid_idx ON todolist (starid);")
	conn.commit()

	# Change settings of SQLite file:
	cursor.execute("PRAGMA page_size=4096;")
	# Run a VACUUM of the table which will force a recreation of the
	# underlying "pages" of the file.
	# Please note that we are changing the "isolation_level" of the connection here,
	# but since we closing the conmnection just after, we are not changing it back
	conn.isolation_level = None
	cursor.execute("VACUUM;")

	# Close connection:
	cursor.close()
	conn.close()
	logger.info("TODO done.")
