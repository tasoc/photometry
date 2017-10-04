#!/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, with_statement, print_function, absolute_import
from six.moves import range, zip
import os
import glob
import numpy as np
import h5py
import logging
import astropy.io.fits as pyfits
from astropy.wcs import WCS

#------------------------------------------------------------------------------
if __name__ == '__main__':

	# Setup logging:
	formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
	console = logging.StreamHandler()
	console.setFormatter(formatter)
	logger = logging.getLogger(__name__)
	logger.addHandler(console)
	logger.setLevel(logging.INFO)

	run = 'ffi_north'
	folder = os.path.join(r'/aadc/kasoc/conferences/TDA1/data/', run)

	with pyfits.open(os.path.join(folder, 'backgrounds.fits.gz'), mode='readonly', memmap=True) as hdu:
		background = hdu[0].data

	files = glob.glob(os.path.join(folder, 'simulated', 'simulated_*.fits.gz'))
	#files = files[0:50]
	numfiles = len(files)
	logger.info(numfiles)

	SUMIMAGE = np.zeros_like(background, dtype='float64')
	time = np.empty(numfiles, dtype='float64')
	cadenceno = np.empty(numfiles, dtype='int32')

	#args = {}
	args = {
		'compression': 'lzf',
		'shuffle': True,
		'fletcher32': True
	}

	with h5py.File('input/camera{0:d}_ccd{1:d}.hdf5'.format(1, 1), 'w') as hdf:

		#dset1 = hdf.create_dataset('images', (SUMIMAGE.shape[0], SUMIMAGE.shape[1], numfiles), **args)
		dset2 = hdf.create_dataset('backgrounds', (SUMIMAGE.shape[0], SUMIMAGE.shape[1], numfiles), **args)

		filenames = ['images/' + os.path.basename(fname).rstrip('.gz') for fname in files]
		hdf.create_dataset('images', data=filenames, **args)

		for k,fname in enumerate(files):
			logger.info("%.2f%% - %s", 100*k/numfiles, fname)

			with pyfits.open(fname, mode='readonly', memmap=True) as hdu:
				time[k] = hdu[0].header['BJD']
				cadenceno[k] = k+1
				flux0 = hdu[0].data

				bck = background

				# Save data in HDF5 file:
				#dset1[:, :, k] = flux0
				dset2[:, :, k] = bck

				# Add together images for sum-image:
				#replace(flux0, np.nan, 0)
				SUMIMAGE += (flux0 - bck)

				# Store FITS header for later use:
				hdr = hdu[0].header

		SUMIMAGE /= numfiles

		logger.info("Saving file...")
		hdf.create_dataset('sumimage', data=SUMIMAGE, **args)
		hdf.create_dataset('time', data=time, **args)
		hdf.create_dataset('cadenceno', data=cadenceno, **args)

		dset = hdf.create_dataset('wcs', (1,), dtype=h5py.special_dtype(vlen=str), **args)
		dset[0] = WCS(hdr).to_header_string().strip()

