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

input_folder = os.environ['TESSPHOT_INPUT']

def create_todo():

	logger = logging.getLogger(__name__)

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

	indx = (cat['tmag'] <= 12) & (cat['x'] > 0) & (cat['x'] < 2048) & (cat['y'] > 0) & (cat['y'] < 2048)
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


def create_catalog(camera, ccd):

	logger = logging.getLogger(__name__)

	cat = np.genfromtxt(os.path.join(input_folder, 'catalog.txt.gz'), skip_header=1, usecols=(0,1,6), dtype='float64')
	cat = np.column_stack((np.arange(1, cat.shape[0]+1, dtype='int64'), cat))

	catalog_file = os.path.join(input_folder, 'catalog_camera{0:d}_ccd{1:d}.sqlite'.format(camera, ccd))
	if os.path.exists(catalog_file): os.remove(catalog_file)
	conn = sqlite3.connect(catalog_file)
	conn.row_factory = sqlite3.Row
	cursor = conn.cursor()

	cursor.execute("""CREATE TABLE catalog (
		starid BIGINT NOT NULL,
		ra DOUBLE PRECISION NOT NULL,
		decl DOUBLE PRECISION NOT NULL,
		tmag REAL NOT NULL
	);""")

	for row in cat:
		cursor.execute("INSERT INTO catalog (starid,ra,decl,tmag) VALUES (?,?,?,?);", row)

	cursor.execute("CREATE UNIQUE INDEX starid_idx ON catalog (starid);")
	cursor.execute("CREATE INDEX ra_dec_idx ON catalog (ra, decl);")
	conn.commit()
	cursor.close()
	conn.close()


def create_hdf5(camera, ccd):

	logger = logging.getLogger(__name__)

	hdf_file = os.path.join(input_folder, 'camera{0:d}_ccd{1:d}.hdf5'.format(camera, ccd))

	files = glob.glob(os.path.join(input_folder, 'images', '*.fits'))
	#files = files[0:10]
	numfiles = len(files)
	logger.info(numfiles)
	if numfiles == 0:
		return

	args = {
		'compression': 'lzf',
		'shuffle': True,
		'fletcher32': True
	}

	with h5py.File(hdf_file, 'a') as hdf:

		dset_bck = hdf.require_dataset('backgrounds', (2048, 2048, numfiles), dtype='float32', **args)
		dset_msk = hdf.require_dataset('backgrounds_masks', (2048, 2048, numfiles), dtype='bool', **args)

		time_smooth = dset_bck.attrs.get('time_smooth', 3)

		last_bck_fit = dset_bck.attrs.get('last_fit', -1)
		logger.info("Last fitted background: %d", last_bck_fit)
		if last_bck_fit < numfiles:
			pool = multiprocessing.Pool()
			k = last_bck_fit+1
			for bck, mask in pool.imap(fit_background, files[k:]):
				logger.info(k)
				dset_bck[:,:,k] = bck
				dset_msk[:,:,k] = mask

				hdf.flush()
				dset_bck.attrs['last_fit'] = k
				k += 1

			pool.close()
			pool.join()
			hdf.flush()

		last_bck_smooth = dset_bck.attrs.get('last_smooth', -1)
		logger.info("Last smoothed background: %d", last_bck_smooth)
		if last_bck_smooth < numfiles:
			dset_bck.attrs['time_smooth'] = time_smooth
			w = time_smooth//2
			for k in range(last_bck_smooth+1, numfiles):
				indx1 = max(k-w, 0)
				indx2 = min(k+w+1, numfiles)
				logger.info("%d: %d -> %d", k, indx1, indx2)
				logger.info(dset_bck[:,:,indx1:indx2].shape)
				dset_bck[:,:,k] = nanmedian(dset_bck[:,:,indx1:indx2], axis=2)

				hdf.flush()
				dset_bck.attrs['last_smooth'] = k

			hdf.flush()

		SumImage = np.zeros((2048, 2048), dtype='float64')
		time = np.empty(numfiles, dtype='float64')
		cadenceno = np.empty(numfiles, dtype='int32')

		dset_img = hdf.require_dataset('images', (2048, 2048, numfiles), **args)
		filenames = ['images/' + os.path.basename(fname).rstrip('.gz') for fname in files]
		hdf.create_dataset('imagespaths', data=filenames, **args)

		for k, fname in enumerate(files):
			logger.info("%.2f%% - %s", 100*k/numfiles, fname)

			# Load background from HDF file:
			bck = hdf['backgrounds'][:, :, k]

			with pyfits.open(fname, mode='readonly', memmap=True) as hdu:
				time[k] = hdu[0].header['BJD']
				cadenceno[k] = k+1
				flux0 = hdu[0].data
				# Store FITS header for later use:
				hdr = hdu[0].header

			# Subtract background from image:
			flux0 -= bck

			# Save image subtracted the background in HDF5 file:
			dset_img[:, :, k] = flux0

			# Add together images for sum-image:
			replace(flux0, np.nan, 0)
			SumImage += flux0

		SumImage /= numfiles

		logger.info("Saving file...")
		hdf.create_dataset('sumimage', data=SumImage, **args)
		hdf.create_dataset('time', data=time, **args)
		hdf.create_dataset('cadenceno', data=cadenceno, **args)

		dset = hdf.create_dataset('wcs', (1,), dtype=h5py.special_dtype(vlen=six.text_type), **args)
		dset[0] = WCS(hdr).to_header_string().strip()

		logger.info("Done.")

#------------------------------------------------------------------------------
if __name__ == '__main__':

	# Setup logging:
	formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
	console = logging.StreamHandler()
	console.setFormatter(formatter)
	logger = logging.getLogger(__name__)
	logger.addHandler(console)
	logger.setLevel(logging.INFO)

	sector = 0
	camera = 1
	ccd = 1

	create_todo()
	create_catalog(camera, ccd)
	create_hdf5(camera, ccd)
