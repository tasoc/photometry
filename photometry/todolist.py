#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create the TODO list which is used by the pipeline to keep track of the
targets that needs to be processed.
"""

import os
import numpy as np
import logging
import sqlite3
import h5py
import re
import itertools
import functools
import contextlib
import warnings
import multiprocessing
from scipy.ndimage.morphology import distance_transform_edt
from scipy.interpolate import RectBivariateSpline
from astropy.table import Table, vstack, Column
from astropy.io import fits
from astropy.wcs import WCS, FITSFixedWarning
from timeit import default_timer
from .utilities import find_tpf_files, find_hdf5_files, find_catalog_files, sphere_distance
from .catalog import catalog_sqlite_search_footprint, download_catalogs

# Filter out annoying warnings:
warnings.filterwarnings('ignore', category=FITSFixedWarning, module="astropy")

#--------------------------------------------------------------------------------------------------
def calc_cbv_area(catalog_row, settings):
	"""
	CBV area that a given target falls within.

	Parameters:
		catalog_row (dict): Target catalog entry.
		settings (dict): Catalog settings.

	Returns:
		int: CBV area that the star falls within.
	"""
	# The distance from the camera centre to the corner furthest away:
	camera_radius = np.sqrt( 12**2 + 12**2 )

	# Distance to centre of the camera in degrees:
	camera_centre_dist = sphere_distance(
		catalog_row['ra'],
		catalog_row['decl'],
		settings['camera_centre_ra'],
		settings['camera_centre_dec'])

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

#--------------------------------------------------------------------------------------------------
def edge_distance(row, column, aperture=None, image_shape=None):
	"""
	Distance to nearest edge.

	Parameters:
		row (ndarray): Array of row positions to calculate distance of.
		column (ndarray): Array of column positions to calculate distance of.
		aperture (ndarray, optional): Boolean array indicating pixels to be
			considered "holes" (False) and good (True).
		image_shape (tuple, optional): Shape of aperture image.

	Returns:
		float: Distance in pixels to the nearest edge (outer or internal).
	"""
	# Basic check of input:
	if image_shape is None and aperture is None:
		raise Exception("Please provide either aperture or image_shape.")

	if image_shape is None and aperture is not None:
		image_shape = aperture.shape

	# Distance from position to outer edges of image:
	EdgeDistOuter = np.minimum.reduce([
		column+0.5,
		row+0.5,
		image_shape[1]-(column+0.5),
		image_shape[0]-(row+0.5)
	])

	# If we have been provided with an aperture and it contains "holes",
	# we should include the distance to these holes:
	if aperture is not None and np.any(~aperture):
		# TODO: This doesn't return the correct answer near internal corners.
		aperture_dist = distance_transform_edt(aperture)
		EdgeDistFunc = RectBivariateSpline(
			np.arange(image_shape[0]),
			np.arange(image_shape[1]),
			np.clip(aperture_dist-0.5, 0, None),
			kx=1, ky=1)

		return np.minimum(EdgeDistFunc(row, column), EdgeDistOuter)

	return EdgeDistOuter

#--------------------------------------------------------------------------------------------------
def _ffi_todo(hdf5_file, exclude=[]):

	logger = logging.getLogger(__name__)

	cat_tmp = []

	# Load the relevant information from the HDF5 file for this camera and ccd:
	with h5py.File(hdf5_file, 'r') as hdf:
		sector = int(hdf['images'].attrs['SECTOR'])
		camera = int(hdf['images'].attrs['CAMERA'])
		ccd = int(hdf['images'].attrs['CCD'])
		cadence = int(hdf['images'].attrs.get('CADENCE', 1800))
		datarel = int(hdf['images'].attrs['DATA_REL'])
		if isinstance(hdf['wcs'], h5py.Group):
			refindx = hdf['wcs'].attrs['ref_frame']
			hdr_string = hdf['wcs']['%04d' % refindx][0]
		else:
			hdr_string = hdf['wcs'][0]
		if not isinstance(hdr_string, str):
			hdr_string = hdr_string.decode("utf-8") # For Python 3
		wcs = WCS(header=fits.Header().fromstring(hdr_string))
		offset_rows = hdf['images'].attrs.get('PIXEL_OFFSET_ROW', 0)
		offset_cols = hdf['images'].attrs.get('PIXEL_OFFSET_COLUMN', 0)
		image_shape = hdf['images']['0000'].shape

	# Load the corresponding catalog:
	input_folder = os.path.dirname(hdf5_file)
	catalog_file = find_catalog_files(input_folder, sector=sector, camera=camera, ccd=ccd)
	if len(catalog_file) != 1:
		raise FileNotFoundError("Catalog file not found: SECTOR=%s, CAMERA=%s, CCD=%s" % (sector, camera, ccd))

	with contextlib.closing(sqlite3.connect(catalog_file[0])) as conn:
		conn.row_factory = sqlite3.Row
		cursor = conn.cursor()

		# Load the settings:
		cursor.execute("SELECT * FROM settings WHERE sector=? AND camera=? AND ccd=? LIMIT 1;", (sector, camera, ccd))
		settings = cursor.fetchone()
		if settings is None:
			raise Exception("Settings not found in catalog (SECTOR=%d, CAMERA=%d, CCD=%d)" % (sector, camera, ccd))

		# Find all the stars in the catalog brigher than a certain limit:
		cursor.execute("SELECT starid,tmag,ra,decl FROM catalog WHERE tmag < 15 ORDER BY tmag;")
		for row in cursor.fetchall():
			logger.debug("%011d - %.3f", row['starid'], row['tmag'])

			# Exclude targets from exclude list:
			if (row['starid'], sector, 'ffi', datarel) in exclude:
				logger.debug("Target excluded: STARID=%d, SECTOR=%d, DATASOURCE=ffi, DATAREL=%d", row['starid'], sector, datarel)
				continue

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

			# Calculate distance from target to edge of image:
			EdgeDist = edge_distance(y, x, image_shape=image_shape)

			# Calculate the Co-trending Basis Vector area the star falls in:
			cbv_area = calc_cbv_area(row, settings)

			# The targets is on silicon, so add it to the todo list:
			cat_tmp.append({
				'starid': row['starid'],
				'sector': sector,
				'camera': camera,
				'ccd': ccd,
				'cadence': cadence,
				'datasource': 'ffi',
				'tmag': row['tmag'],
				'cbv_area': cbv_area,
				'edge_dist': EdgeDist
			})

		cursor.close()

	# Create the TODO list as a table which we will fill with targets:
	return Table(
		rows=cat_tmp,
		names=('starid', 'sector', 'camera', 'ccd', 'cadence', 'datasource', 'tmag', 'cbv_area', 'edge_dist'),
		dtype=('int64', 'int32', 'int32', 'int32', 'int32', 'S256', 'float32', 'int32', 'float32')
	)

#--------------------------------------------------------------------------------------------------
def _tpf_todo(fname, input_folder=None, cameras=None, ccds=None,
	find_secondary_targets=True, exclude=[]):

	logger = logging.getLogger(__name__)

	# Create the TODO list as a table which we will fill with targets:
	cat_tmp = []
	empty_table = Table(
		names=('starid', 'sector', 'camera', 'ccd', 'cadence', 'datasource', 'tmag', 'cbv_area', 'edge_dist'),
		dtype=('int64', 'int32', 'int32', 'int32', 'int32', 'S256', 'float32', 'int32', 'float32')
	)

	logger.debug("Processing TPF file: '%s'", fname)
	with fits.open(fname, memmap=True, mode='readonly') as hdu:
		starid = hdu[0].header['TICID']
		sector = hdu[0].header['SECTOR']
		camera = hdu[0].header['CAMERA']
		ccd = hdu[0].header['CCD']
		datarel = hdu[0].header['DATA_REL']
		aperture_observed_pixels = (hdu['APERTURE'].data & 1 != 0)
		cadence = int(np.round(hdu[1].header['TIMEDEL']*86400))

		if (starid, sector, 'tpf', datarel) in exclude:
			logger.debug("Target excluded: STARID=%d, SECTOR=%d, DATASOURCE=tpf, DATAREL=%d", starid, sector, datarel)
			return empty_table

		if camera in cameras and ccd in ccds:
			# Load the corresponding catalog:
			catalog_file = find_catalog_files(input_folder, sector=sector, camera=camera, ccd=ccd)
			if len(catalog_file) != 1:
				raise FileNotFoundError("Catalog file not found: SECTOR=%s, CAMERA=%s, CCD=%s" % (sector, camera, ccd))

			with contextlib.closing(sqlite3.connect(catalog_file[0])) as conn:
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
					return empty_table

				# Calculate CBV area that target falls in:
				cbv_area = calc_cbv_area(row, settings)

				# Add the main target to the list:
				cat_tmp.append({
					'starid': starid,
					'sector': sector,
					'camera': camera,
					'ccd': ccd,
					'cadence': cadence,
					'datasource': 'tpf',
					'tmag': row['tmag'],
					'cbv_area': cbv_area,
					'edge_dist': np.NaN
				})

				if find_secondary_targets:
					# Load all other targets in this stamp:
					# Use the WCS of the stamp to find all stars that fall within
					# the footprint of the stamp.
					image_shape = hdu[2].shape
					wcs = WCS(header=hdu[2].header)
					footprint = wcs.calc_footprint(center=False)

					secondary_targets = catalog_sqlite_search_footprint(cursor, footprint, constraints='starid != %d AND tmag < 15' % starid, buffer_size=2)
					for row in secondary_targets:
						# Calculate the position of this star on the CCD using the WCS:
						ra_dec = np.atleast_2d([row['ra'], row['decl']])
						x, y = wcs.all_world2pix(ra_dec, 0)[0]

						# If the target falls outside silicon, do not add it to the todo list:
						# The reason for the strange 0.5's is that pixel centers are at integers.
						if x < -0.5 or y < -0.5 or x > image_shape[1]-0.5 or y > image_shape[0]-0.5:
							continue

						# Make sure that the pixel that the target falls on has actually been
						# collected by the spacecraft:
						if not aperture_observed_pixels[int(np.round(y)), int(np.round(x))]:
							logger.debug("Secondary target rejected. Falls on non-observed pixel. (primary=%d, secondary=%d)", starid, row['starid'])
							continue

						# Calculate distance from target to edge of image:
						EdgeDist = edge_distance(y, x, aperture=aperture_observed_pixels)

						# Add this secondary target to the list:
						# Note that we are storing the starid of the target
						# in which target pixel file the target can be found.
						logger.debug("Adding extra target: TIC %d", row['starid'])
						cat_tmp.append({
							'starid': row['starid'],
							'sector': sector,
							'camera': camera,
							'ccd': ccd,
							'cadence': cadence,
							'datasource': 'tpf:' + str(starid),
							'tmag': row['tmag'],
							'cbv_area': cbv_area,
							'edge_dist': EdgeDist
						})

				# Close the connection to the catalog SQLite database:
				cursor.close()
		else:
			logger.debug("Target not on requested CAMERA and CCD")
			return empty_table

	# TODO: Could we avoid fixed-size strings in datasource column?
	return Table(
		rows=cat_tmp,
		names=('starid', 'sector', 'camera', 'ccd', 'cadence', 'datasource', 'tmag', 'cbv_area', 'edge_dist'),
		dtype=('int64', 'int32', 'int32', 'int32', 'int32', 'S256', 'float32', 'int32', 'float32')
	)

#--------------------------------------------------------------------------------------------------
def make_todo(input_folder=None, cameras=None, ccds=None, overwrite=False,
	find_secondary_targets=True, output_file=None):
	"""
	Create the TODO list which is used by the pipeline to keep track of the
	targets that needs to be processed.

	Will create the file `todo.sqlite` in the directory.

	Parameters:
		input_folder (string, optional): Input folder to create TODO list for.
			If ``None``, the input directory in the environment variable
			``TESSPHOT_INPUT`` is used.
		cameras (iterable of integers, optional): TESS camera number (1-4). If ``None``,
			all cameras will be included.
		ccds (iterable of integers, optional): TESS CCD number (1-4). If ``None``,
			all cameras will be included.
		overwrite (boolean): Overwrite existing TODO file. Default=``False``.
		find_secondary_targets (boolean): Should secondary targets from TPFs be included?
			Default=True.
		output_file (string, optional): The file path where the output file should be saved.
			If not specified, the file will be saved into the input directory.
			Should only be used for testing, since the file would (properly) otherwise end up with
			a wrong file name for running with the rest of the pipeline.

	Raises:
		NotADirectoryError: If the specified ``input_folder`` is not an existing directory.

	.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
	"""

	logger = logging.getLogger(__name__)

	# Check the input folder, and load the default if not provided:
	if input_folder is None:
		input_folder = os.environ.get('TESSPHOT_INPUT', os.path.join(os.path.dirname(__file__), 'tests', 'input'))

	# Check that the given input directory is indeed a directory:
	if not os.path.isdir(input_folder):
		raise NotADirectoryError("The given path does not exist or is not a directory")

	# Make sure cameras and ccds are iterable:
	cameras = (1, 2, 3, 4) if cameras is None else (cameras, )
	ccds = (1, 2, 3, 4) if ccds is None else (ccds, )

	# The TODO file that we want to create. Delete it if it already exits:
	if output_file is None:
		todo_file = os.path.join(input_folder, 'todo.sqlite')
	else:
		output_file = os.path.abspath(output_file)
		if not output_file.endswith('.sqlite'):
			output_file = output_file + '.sqlite'
		todo_file = output_file

	if os.path.exists(todo_file):
		if overwrite:
			os.remove(todo_file)
		else:
			logger.info("TODO file already exists")
			return

	# Number of threads available for parallel processing:
	threads_max = int(os.environ.get('SLURM_CPUS_PER_TASK', multiprocessing.cpu_count()))

	# Load file with targets to be excluded from processing for some reason:
	exclude_file = os.path.join(os.path.dirname(__file__), 'data', 'todolist-exclude.dat')
	exclude = np.genfromtxt(exclude_file, usecols=(0,1,2,3), dtype=None, encoding='utf-8')
	exclude = set([tuple(e) for e in exclude])

	# Create the TODO list as a table which we will fill with targets:
	cat = Table(
		names=('starid', 'sector', 'camera', 'ccd', 'cadence', 'datasource', 'tmag', 'cbv_area', 'edge_dist'),
		dtype=('int64', 'int32', 'int32', 'int32', 'int32', 'S256', 'float32', 'int32', 'float32')
	)
	sectors = set()

	# Load list of all Target Pixel files in the directory:
	tpf_files = find_tpf_files(input_folder)
	logger.info("Number of TPF files: %d", len(tpf_files))

	# TODO: Could we change this so we don't have to parse the filename?
	regex_tpf = re.compile(r'-s(\d+)[-_]')
	for fname in tpf_files:
		m = regex_tpf.search(os.path.basename(fname))
		sectors.add(int(m.group(1)))

	# Find list of all HDF5 files:
	hdf_files = find_hdf5_files(input_folder, camera=cameras, ccd=ccds)
	logger.info("Number of HDF5 files: %d", len(hdf_files))

	# TODO: Could we change this so we don't have to parse the filename?
	regex_hdf = re.compile(r'^sector(\d+)_camera(\d)_ccd(\d)\.hdf5$')
	for fname in hdf_files:
		m = regex_hdf.match(os.path.basename(fname))
		sectors.add(int(m.group(1)))

	# Make sure that catalog files are available in the input directory.
	# If they are not already, they will be downloaded from the cache:
	for sector, camera, ccd in itertools.product(sectors, cameras, ccds):
		download_catalogs(input_folder, sector, camera=camera, ccd=ccd)

	# Add the target pixel files to the TODO list:
	if len(tpf_files) > 0:
		# Open a pool of workers:
		logger.info("Starting pool of workers for TPFs...")
		threads = min(threads_max, len(tpf_files)) # No reason to use more than the number of jobs
		logger.info("Using %d processes.", threads)

		if threads > 1:
			pool = multiprocessing.Pool(threads)
			m = pool.imap_unordered
		else:
			m = map

		# Run the TPF files in parallel:
		tic = default_timer()
		_tpf_todo_wrapper = functools.partial(_tpf_todo,
			input_folder=input_folder,
			cameras=cameras,
			ccds=ccds,
			find_secondary_targets=find_secondary_targets,
			exclude=exclude)

		for cat2 in m(_tpf_todo_wrapper, tpf_files):
			cat = vstack([cat, cat2], join_type='exact')

		if threads > 1:
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

	if len(hdf_files) > 0:
		# Open a pool of workers:
		logger.info("Starting pool of workers for FFIs...")
		threads = min(threads_max, len(hdf_files)) # No reason to use more than the number of jobs
		logger.info("Using %d processes.", threads)

		if threads > 1:
			pool = multiprocessing.Pool(threads)
			m = pool.imap_unordered
		else:
			m = map

		_ffi_todo_wrapper = functools.partial(_ffi_todo, exclude=exclude)

		tic = default_timer()
		ccds_done = 0
		for cat2 in m(_ffi_todo_wrapper, hdf_files):
			cat = vstack([cat, cat2], join_type='exact')
			ccds_done += 1
			logger.info("CCDs done: %d/%d", ccds_done, len(hdf_files))

		# Amount of time it took to process TPF files:
		toc = default_timer()
		logger.info("Elaspsed time: %f seconds (%f per file)", toc-tic, (toc-tic)/len(hdf_files))

		if threads > 1:
			pool.close()
			pool.join()

	# Check if any targets were found:
	if len(cat) == 0:
		logger.error("No targets found")
		return

	# Remove duplicates!
	logger.info("Removing duplicate entries...")
	_, idx = np.unique(cat[('starid', 'sector', 'camera', 'ccd', 'datasource')], return_index=True, axis=0)
	cat = cat[np.sort(idx)]

	# If the target is present in more than one TPF file, pick the one
	# where the target is the furthest from the edge of the image
	# and discard the target in all the other TPFs:
	if find_secondary_targets:
		# Add an index column to the table for later use:
		cat.add_column(Column(name='priority', data=np.arange(len(cat))))

		# Create index that will only find secondary targets:
		indx = [row['datasource'].strip().startswith('tpf:') for row in cat]

		# Group the table on the starids and find groups with more than 1 target:
		# Equivalent to the SQL code "GROUP BY starid HAVING COUNT(*) > 1"
		remove_indx = []
		for g in cat[indx].group_by('starid').groups:
			if len(g) > 1:
				# Find the target farthest from the edge and mark the rest
				# for removal:
				logger.debug(g)
				im = np.argmax(g['edge_dist'])
				ir = np.ones(len(g), dtype='bool')
				ir[im] = False
				remove_indx += list(g[ir]['priority'])

		# Remove the list of duplicate secondary targets:
		logger.info("Removing %d secondary targets as duplicates.", len(remove_indx))
		logger.debug(remove_indx)
		cat.remove_rows(remove_indx)

	# Load file with specific method settings and create lookup-table of them:
	methods_file = os.path.join(os.path.dirname(__file__), 'data', 'todolist-methods.dat')
	methods_file = np.genfromtxt(methods_file, usecols=(0,1,2,3), dtype=None, encoding='utf-8')
	methods = {}
	for m in methods_file:
		methods[(m[0], m[1], m[2])] = m[3].strip().lower()

	# Sort the final list:
	cat.sort('tmag')

	# Write the TODO list to the SQLite database file:
	logger.info("Writing TODO file...")
	with contextlib.closing(sqlite3.connect(todo_file)) as conn:
		cursor = conn.cursor()

		# Change settings of SQLite file:
		cursor.execute("PRAGMA page_size=4096;")
		cursor.execute("PRAGMA foreign_keys=ON;")
		cursor.execute("PRAGMA locking_mode=EXCLUSIVE;")
		cursor.execute("PRAGMA journal_mode=TRUNCATE;")

		# Create TODO-list table:
		cursor.execute("""CREATE TABLE todolist (
			priority INTEGER PRIMARY KEY ASC NOT NULL,
			starid INTEGER NOT NULL,
			sector INTEGER NOT NULL,
			datasource TEXT NOT NULL DEFAULT 'ffi',
			camera INTEGER NOT NULL,
			ccd INTEGER NOT NULL,
			cadence INTEGER NOT NULL,
			method TEXT DEFAULT NULL,
			tmag REAL,
			status INTEGER DEFAULT NULL,
			cbv_area INTEGER NOT NULL
		);""")

		for pri, row in enumerate(cat):
			# Find if there is a specific method defined for this target:
			method = methods.get((int(row['starid']), int(row['sector']), row['datasource'].strip()), None)

			# For very bright stars, we might as well just use Halo photometry right away:
			if method is None and row['tmag'] <= 2.0 and row['datasource'] == 'ffi':
				method = 'halo'

			# Add target to TODO-list:
			cursor.execute("INSERT INTO todolist (priority,starid,sector,camera,ccd,cadence,datasource,tmag,cbv_area,method) VALUES (?,?,?,?,?,?,?,?,?,?);", (
				pri+1,
				int(row['starid']),
				int(row['sector']),
				int(row['camera']),
				int(row['ccd']),
				int(row['cadence']),
				row['datasource'].strip(),
				float(row['tmag']),
				int(row['cbv_area']),
				method
			))

		conn.commit()
		cursor.execute("CREATE UNIQUE INDEX unique_target_idx ON todolist (starid, datasource, sector, camera, ccd, cadence);")
		cursor.execute("CREATE INDEX status_idx ON todolist (status);")
		cursor.execute("CREATE INDEX starid_idx ON todolist (starid);")
		conn.commit()

		# Analyze the tables for better query planning:
		cursor.execute("ANALYZE;")
		conn.commit()

		# Run a VACUUM of the table which will force a recreation of the
		# underlying "pages" of the file.
		# Please note that we are changing the "isolation_level" of the connection here,
		# but since we closing the connection just after, we are not changing it back
		conn.isolation_level = None
		cursor.execute("VACUUM;")

		# Close connection:
		cursor.close()

	logger.info("TODO done.")
