#!/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, with_statement, print_function, absolute_import
import six
from six.moves import range
import os
import glob
import numpy as np
import h5py
import logging
import astropy.io.fits as pyfits
from astropy.wcs import WCS
import sqlite3
from astropy.table import Table
import multiprocessing
from bottleneck import replace, nanmedian
from photometry.backgrounds import fit_background
from photometry.utilities import add_proper_motion
from timeit import default_timer

def create_todo(sector):

	logger = logging.getLogger(__name__)

	input_folder = os.environ['TESSPHOT_INPUT']

	cat = np.genfromtxt(os.path.join(input_folder, 'catalog.txt.gz'), skip_header=1, usecols=(4,5,6), dtype='float64')
	cat = np.column_stack((np.arange(1, cat.shape[0]+1, dtype='int64'), cat))
	# Convert data to astropy table for further use:
	cat = Table(
		data=cat,
		names=('starid', 'x', 'y', 'tmag'),
		dtype=('int64', 'float64', 'float64', 'float32')
	)

	todo_file = os.path.join(input_folder, 'todo.sqlite')
	if os.path.exists(todo_file): os.remove(todo_file)
	conn = sqlite3.connect(todo_file)
	cursor = conn.cursor()

	cursor.execute("""CREATE TABLE todolist (
		priority BIGINT NOT NULL,
		starid BIGINT NOT NULL,
		datasource INT NOT NULL DEFAULT 0,
		camera INT NOT NULL,
		ccd INT NOT NULL,
		method INT DEFAULT NULL,
		status INT DEFAULT NULL,
		elaptime REAL DEFAULT NULL,
		x REAL,
		y REAL,
		tmag REAL
	);""")

	indx = (cat['tmag'] <= 20) & (cat['x'] > 0) & (cat['x'] < 2048) & (cat['y'] > 0) & (cat['y'] < 2048)
	cat = cat[indx]
	cat.sort('tmag')
	for pri, row in enumerate(cat):
		starid = int(row['starid'])
		tmag = float(row['tmag'])
		cursor.execute("INSERT INTO todolist (priority,starid,camera,ccd,x,y,tmag) VALUES (?,?,?,?,?,?,?);", (pri+1, starid, 1, 1, row['x'], row['y'], tmag))

	cursor.execute("CREATE UNIQUE INDEX starid_idx ON todolist (starid);")
	cursor.execute("CREATE UNIQUE INDEX priority_idx ON todolist (priority);")
	conn.commit()
	cursor.close()
	conn.close()

	logger.info("TODO done.")


def create_catalog(sector, camera, ccd):

	logger = logging.getLogger(__name__)

	input_folder = os.environ['TESSPHOT_INPUT']

	# We need a list of when the sectors are in time:
	sector_reference_time = 2457827.0 + 13.5
	logger.info('Projecting catalog {0:.3f} years relative to 2000'.format((sector_reference_time - 2451544.5)/365.25))

	# Load the catalog from file:
	# TODO: In the future this will be loaded from the TASOC database:
	cat = np.genfromtxt(os.path.join(input_folder, 'catalog.txt.gz'), skip_header=1, usecols=(0,1,2,3,6), dtype='float64')
	cat = np.column_stack((np.arange(1, cat.shape[0]+1, dtype='int64'), cat))

	# Create SQLite file:
	catalog_file = os.path.join(input_folder, 'catalog_camera{0:d}_ccd{1:d}.sqlite'.format(camera, ccd))
	if os.path.exists(catalog_file): os.remove(catalog_file)
	conn = sqlite3.connect(catalog_file)
	conn.row_factory = sqlite3.Row
	cursor = conn.cursor()

	cursor.execute("""CREATE TABLE catalog (
		starid BIGINT PRIMARY KEY NOT NULL,
		ra DOUBLE PRECISION NOT NULL,
		decl DOUBLE PRECISION NOT NULL,
		ra_J2000 DOUBLE PRECISION NOT NULL,
		decl_J2000 DOUBLE PRECISION NOT NULL,
		tmag REAL NOT NULL
	);""")

	for row in cat:
		# Add the proper motion to each coordinate:
		ra, dec = add_proper_motion(row[1], row[2], row[3], row[4], sector_reference_time, epoch=2000.0)
		logger.debug("(%f, %f) => (%f, %f)", row[1], row[2], ra, dec)

		# Save the coordinates in SQLite database:
		cursor.execute("INSERT INTO catalog (starid,ra,decl,ra_J2000,decl_J2000,tmag) VALUES (?,?,?,?,?,?);", (
			int(row[0]),
			ra,
			dec,
			row[1],
			row[2],
			row[5]
		))

	cursor.execute("CREATE UNIQUE INDEX starid_idx ON catalog (starid);")
	cursor.execute("CREATE INDEX ra_dec_idx ON catalog (ra, decl);")
	conn.commit()
	cursor.close()
	conn.close()

	logger.info("Catalog done.")



def create_hdf5(camera, ccd):

	logger = logging.getLogger(__name__)

	input_folder = os.environ['TESSPHOT_INPUT']

	sector_reference_time = 2457827.0 + 13.5

	hdf_file = os.path.join(input_folder, 'camera{0:d}_ccd{1:d}.hdf5'.format(camera, ccd))

	files = glob.glob(os.path.join(input_folder, 'images', '*.fits'))
	#files = files[0:2]
	numfiles = len(files)
	logger.info("Number of files: %d", numfiles)
	if numfiles == 0:
		return

	args = {
		'compression': 'lzf',
		'shuffle': True,
		'fletcher32': True,
		'chunks': True
	}

	threads = int(os.environ.get('SLURM_CPUS_PER_TASK', 1))
	logger.info("Using %d processes.", threads)

	with h5py.File(hdf_file, 'a') as hdf:

		images = hdf.require_group('images')
		backgrounds = hdf.require_group('backgrounds')
		time_smooth = backgrounds.attrs.get('time_smooth', 3)

		if 'backgrounds_unsmoothed' in hdf or len(backgrounds) < numfiles:

			masks = hdf.require_group('backgrounds_masks')

			if len(masks) < numfiles:

				dset_bck_us = hdf.require_dataset('backgrounds_unsmoothed', (2048, 2048, numfiles), dtype='float32')

				tic = default_timer()
				if threads > 1:
					pool = multiprocessing.Pool(threads)
					m = pool.imap
				else:
					m = map

				last_bck_fit = int(masks.keys()[-1])
				k = last_bck_fit+1
				for bck, mask in m(fit_background, files[k:]):
					dset_name = '%04d' % k
					logger.info(k)

					dset_bck_us[:,:,k] = bck

					indicies = np.asarray(np.nonzero(mask), dtype='uint16')
					masks.create_dataset(dset_name, data=indicies, **args)

					k += 1

				if threads > 1:
					pool.close()
					pool.join()
				hdf.flush()
				toc = default_timer()
				logger.info("%f sec/image", (toc-tic)/(numfiles-last_bck_fit))

			# Smooth the backgrounds along the time axis:
			backgrounds.attrs['time_smooth'] = time_smooth
			w = time_smooth//2
			for k in range(numfiles):
				dset_name = '%04d' % k
				if dset_name in backgrounds: continue

				indx1 = max(k-w, 0)
				indx2 = min(k+w+1, numfiles)
				logger.info("%d: %d -> %d", k, indx1, indx2)

				bck = nanmedian(dset_bck_us[:, :, indx1:indx2], axis=2)

				backgrounds.create_dataset(dset_name, data=bck, **args)

			# FIXME: Because HDF5 is stupid, this might not actually delete the data
			#        Maybe we need to run h5repack in the file at the end?
			del hdf['backgrounds_unsmoothed']
			hdf.flush()


		if len(images) < numfiles:
			SumImage = np.zeros((2048, 2048), dtype='float64')
			time = np.empty(numfiles, dtype='float64')
			cadenceno = np.empty(numfiles, dtype='int32')

			# Save list of file paths to the HDF5 file:
			filenames = ['images/' + os.path.basename(fname).rstrip('.gz') for fname in files]
			filenames = [fname.encode('ascii', 'strict') for fname in filenames]
			hdf.require_dataset('imagespaths', (numfiles,), data=filenames, dtype=h5py.special_dtype(vlen=bytes), **args)

			for k, fname in enumerate(files):
				logger.info("%.2f%% - %s", 100*k/numfiles, fname)
				dset_name ='%04d' % k
				#if dset_name in hdf['images']: continue # Dont do this, because it will mess up the sumimage and time vector

				with pyfits.open(fname, mode='readonly', memmap=True) as hdu:
					time[k] = hdu[0].header['BJD']
					cadenceno[k] = k+1
					flux0 = np.asarray(hdu[0].data, dtype='float32')

				# # Load background from HDF file and subtract background from image:
				flux0 -= backgrounds[dset_name]

				# Save image subtracted the background in HDF5 file:
				images.create_dataset(dset_name, data=flux0, **args)

				# Add together images for sum-image:
				replace(flux0, np.nan, 0)
				SumImage += flux0

			SumImage /= numfiles

			logger.info("Saving file...")

			# Save WCS to the file:
			refindx = np.searchsorted(time, sector_reference_time, side='left')
			if refindx > 0 and (refindx == len(time) or abs(sector_reference_time - time[refindx-1]) < abs(sector_reference_time - time[refindx])):
				refindx -= 1

			logger.info("WCS reference frame: %d", refindx)
			with pyfits.open(files[refindx], mode='readonly', memmap=True) as hdu:
				# Store FITS header for later use:
				hdr = hdu[0].header

			dset = hdf.require_dataset('wcs', (1,), dtype=h5py.special_dtype(vlen=bytes), **args)
			dset[0] = WCS(hdr).to_header_string().strip().encode('ascii', 'strict')
			dset.attrs['ref_frame'] = refindx

			# Add other arrays to HDF5 file:
			if 'time' in hdf: del hdf['time']
			if 'sumimage' in hdf: del hdf['sumimage']
			if 'cadenceno' in hdf: del hdf['cadenceno']
			hdf.create_dataset('sumimage', data=SumImage, **args)
			hdf.create_dataset('time', data=time, **args)
			hdf.create_dataset('cadenceno', data=cadenceno, **args)
			hdf.flush()

		logger.info("Done.")

#------------------------------------------------------------------------------
if __name__ == '__main__':

	# Setup logging:
	formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

	logger = logging.getLogger(__name__)
	logger.setLevel(logging.INFO)
	if not logger.hasHandlers():
		console = logging.StreamHandler()
		console.setFormatter(formatter)
		logger.addHandler(console)
	#logger_parent = logging.getLogger('photometry')
	#logger_parent.addHandler(console)
	#logger_parent.setLevel(logging.INFO)

	sector = 0
	camera = 1
	ccd = 1

	create_todo(sector)
	create_catalog(sector, camera, ccd)
	#create_hdf5(camera, ccd)
