#!/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, with_statement, print_function, absolute_import
from six.moves import range
import os
import numpy as np
import warnings
warnings.filterwarnings('ignore', category=FutureWarning, module='h5py')
import h5py
import sqlite3
import logging
import multiprocessing
from astropy.wcs import WCS
from bottleneck import replace, nanmean
from timeit import default_timer
import itertools
from .backgrounds import fit_background
from .utilities import load_ffi_fits, find_ffi_files
from photometry import TESSQualityFlags, ImageMovementKernel

#------------------------------------------------------------------------------
def _iterate_hdf_group(dset):
	for d in dset:
		yield np.asarray(dset[d])

#------------------------------------------------------------------------------
def create_hdf5(input_folder=None, cameras=None, ccds=None):
	"""
	Restructure individual FFI images (in FITS format) into
	a combined HDF5 file which is used in the photometry
	pipeline.

	In this process the background flux in each FFI is
	estimated using the `backgrounds.fit_background` function.

	Parameters:
		input_folder (string): Input folder to create TODO list for. If ``None``, the input directory in the environment variable ``TESSPHOT_INPUT`` is used.
		cameras (iterable of integers, optional): TESS camera number (1-4). If ``None``, all cameras will be processed.
		ccds (iterable of integers, optional): TESS CCD number (1-4). If ``None``, all cameras will be processed.

	Raises:
		IOError: If the specified ``input_folder`` is not an existing directory or if settings table could not be loaded from the catalog SQLite file.

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

	# Common settings for HDF5 datasets:
	args = {
		'compression': 'lzf',
		'shuffle': True,
		'fletcher32': True
	}
	imgchunks = (64, 64)

	# Get the number of processes we can spawn in case it is needed for calculations:
	threads = int(os.environ.get('SLURM_CPUS_PER_TASK', multiprocessing.cpu_count()))
	logger.info("Using %d processes.", threads)

	# Loop over each combination of camera and CCD:
	for camera, ccd in itertools.product(cameras, ccds):
		logger.info("Running CAMERA=%s, CCD=%s", camera, ccd)
		tic_total = default_timer()

		# Find all the FFI files associated with this camera and CCD:
		files = find_ffi_files(input_folder, camera, ccd)
		numfiles = len(files)
		logger.info("Number of files: %d", numfiles)
		if numfiles == 0:
			continue

		# Catalog file:
		catalog_file = os.path.join(input_folder, 'catalog_camera{0:d}_ccd{1:d}.sqlite'.format(camera, ccd))
		logger.debug("Catalog File: %s", catalog_file)
		if not os.path.exists(catalog_file):
			logger.error("Catalog file could not be found: '%s'", catalog_file)
			continue

		# Load catalog settings from the SQLite database:
		conn = sqlite3.connect(catalog_file)
		conn.row_factory = sqlite3.Row
		cursor = conn.cursor()
		cursor.execute("SELECT sector,reference_time FROM settings LIMIT 1;")
		row = cursor.fetchone()
		if row is None:
			raise IOError("Settings could not be loaded from catalog")
		#sector = row['sector']
		sector_reference_time = row['reference_time']
		cursor.close()
		conn.close()

		# HDF5 file to be created/modified:
		hdf_file = os.path.join(input_folder, 'camera{0:d}_ccd{1:d}.hdf5'.format(camera, ccd))
		logger.debug("HDF5 File: %s", hdf_file)

		# Get image shape from the first file:
		img = load_ffi_fits(files[0])
		img_shape = img.shape

		# Open the HDF5 file for editing:
		with h5py.File(hdf_file, 'a', libver='latest') as hdf:

			images = hdf.require_group('images')
			backgrounds = hdf.require_group('backgrounds')
			if 'wcs' in hdf and isinstance(hdf['wcs'], h5py.Dataset): del hdf['wcs']
			wcs = hdf.require_group('wcs')
			time_smooth = backgrounds.attrs.get('time_smooth', 3)

			if 'backgrounds_unsmoothed' in hdf or len(backgrounds) < numfiles:

				dset_bck_us = hdf.require_group('backgrounds_unsmoothed')
				masks = hdf.require_group('backgrounds_masks')

				if len(masks) < numfiles:

					tic = default_timer()
					if threads > 1:
						pool = multiprocessing.Pool(threads)
						m = pool.imap
					else:
						m = map

					last_bck_fit = -1 if len(masks) == 0 else int(list(masks.keys())[-1])
					k = last_bck_fit+1
					for bck, mask in m(fit_background, files[k:]):
						dset_name = '%04d' % k
						logger.debug("Background %d complete", k)
						logger.debug("Estimate: %f sec/image", (default_timer()-tic)/(k-last_bck_fit))

						dset_bck_us.create_dataset(dset_name, data=bck)

						indicies = np.asarray(np.nonzero(mask), dtype='uint16')
						masks.create_dataset(dset_name, data=indicies, **args)

						k += 1

					if threads > 1:
						pool.close()
						pool.join()

					hdf.flush()
					toc = default_timer()
					logger.info("Background estimation: %f sec/image", (toc-tic)/(numfiles-last_bck_fit))

				# Smooth the backgrounds along the time axis:
				backgrounds.attrs['time_smooth'] = time_smooth
				w = time_smooth//2
				tic = default_timer()
				for k in range(numfiles):
					dset_name = '%04d' % k
					if dset_name in backgrounds: continue

					indx1 = max(k-w, 0)
					indx2 = min(k+w+1, numfiles)
					logger.debug("Smoothing background %d: %d -> %d", k, indx1, indx2)

					block = np.empty((img_shape[0], img_shape[1], indx2-indx1), dtype='float32')
					logger.debug(block.shape)
					for i, k in enumerate(range(indx1, indx2)):
						block[:, :, i] = dset_bck_us['%04d' % k]

					bck = nanmean(block, axis=2)

					backgrounds.create_dataset(dset_name, data=bck, chunks=imgchunks, **args)

				toc = default_timer()
				logger.info("Background smoothing: %f sec/image", (toc-tic)/numfiles)

				# FIXME: Because HDF5 is stupid, this might not actually delete the data
				#        Maybe we need to run h5repack in the file at the end?
				del hdf['backgrounds_unsmoothed']
				hdf.flush()


			if len(images) < numfiles or len(wcs) < numfiles or 'sumimage' not in hdf:
				SumImage = np.zeros((img_shape[0], img_shape[1]), dtype='float64')
				time = np.empty(numfiles, dtype='float64')
				timecorr = np.empty(numfiles, dtype='float32')
				cadenceno = np.empty(numfiles, dtype='int32')
				quality = np.empty(numfiles, dtype='int32')

				# Save list of file paths to the HDF5 file:
				filenames = [os.path.basename(fname).rstrip('.gz').encode('ascii', 'strict') for fname in files]
				hdf.require_dataset('imagespaths', (numfiles,), data=filenames, dtype=h5py.special_dtype(vlen=bytes), **args)

				is_tess = False
				attributes = {
					'DATA_REL': None,
					'NUM_FRM': None,
					'CRMITEN': None,
					'CRBLKSZ': None,
					'CRSPOC': None
				}
				for k, fname in enumerate(files):
					logger.debug("Processing image: %.2f%% - %s", 100*k/numfiles, fname)
					dset_name ='%04d' % k

					# Load the FITS file data and the header:
					flux0, hdr = load_ffi_fits(fname, return_header=True)

					# Check if this is real TESS data:
					# Could proberly be done more elegant, but if it works, it works...
					if not is_tess and hdr.get('TELESCOP') == 'TESS' and hdr.get('NAXIS1') == 2136 and hdr.get('NAXIS2') == 2078:
						is_tess = True

					# Pick out the important bits from the header:
					time[k] = 0.5*(hdr['TSTART'] + hdr['TSTOP']) + hdr.get('BJDREFI', 0) + hdr.get('BJDREFF', 0)
					timecorr[k] = hdr.get('BARYCORR', 0)
					cadenceno[k] = k+1
					quality[k] = hdr.get('DQUALITY', 0)

					if k == 0:
						for key in attributes.keys():
							attributes[key] = hdr.get(key)
					else:
						for key, value in attributes.items():
							if hdr.get(key) != value:
								logger.error("%s is not constant!", key)

					#if hdr.get('SECTOR') != sector:
					#	logger.error("Incorrect SECTOR: Catalog=%s, FITS=%s", sector, hdr.get('SECTOR'))
					if hdr.get('CAMERA') != camera or hdr.get('CCD') != ccd:
						logger.error("Incorrect CAMERA/CCD: FITS=(%s, %s)", hdr.get('CAMERA'), hdr.get('CCD'))

					if dset_name not in images:
						# Load background from HDF file and subtract background from image,
						# if the background has not already been subtracted:
						if not hdr.get('BACKAPP', False):
							flux0 -= backgrounds[dset_name]

						# Save image subtracted the background in HDF5 file:
						images.create_dataset(dset_name, data=flux0, chunks=imgchunks, **args)
					else:
						flux0 = np.asarray(images[dset_name])

					# Save the World Coordinate System of each image:
					if dset_name not in wcs:
						dset = wcs.create_dataset(dset_name, (1,), dtype=h5py.special_dtype(vlen=bytes), **args)
						dset[0] = WCS(header=hdr).to_header_string(relax=True).strip().encode('ascii', 'strict')

					# Add together images for sum-image:
					if TESSQualityFlags.filter(quality[k]):
						replace(flux0, np.nan, 0)
						SumImage += flux0

				SumImage /= numfiles

				# Save attributes
				for key, value in attributes.items():
					logger.debug("Saving attribute %s = %s", key, value)
					images.attrs[key] = value

				# Set pixel offsets:
				if is_tess:
					images.attrs['PIXEL_OFFSET_ROW'] = 0
					images.attrs['PIXEL_OFFSET_COLUMN'] = 44
				else:
					images.attrs['PIXEL_OFFSET_ROW'] = 0
					images.attrs['PIXEL_OFFSET_COLUMN'] = 0

				# Add other arrays to HDF5 file:
				if 'time' in hdf: del hdf['time']
				if 'timecorr' in hdf: del hdf['timecorr']
				if 'sumimage' in hdf: del hdf['sumimage']
				if 'cadenceno' in hdf: del hdf['cadenceno']
				if 'quality' in hdf: del hdf['quality']
				hdf.create_dataset('sumimage', data=SumImage, **args)
				hdf.create_dataset('time', data=time, **args)
				hdf.create_dataset('timecorr', data=timecorr, **args)
				hdf.create_dataset('cadenceno', data=cadenceno, **args)
				hdf.create_dataset('quality', data=quality, **args)
				hdf.flush()

			# Check that the time vector is sorted:
			if not np.all(hdf['time'][:-1] < hdf['time'][1:]):
				logger.error("Time vector is not sorted")
				return

			# Check that the sector reference time is within the timespan of the time vector:
			if sector_reference_time < hdf['time'][0] or sector_reference_time > hdf['time'][-1]:
				logger.error("Sector reference time outside timespan of data")
				#return

			# Find the reference image:
			refindx = np.searchsorted(hdf['time'], sector_reference_time, side='left')
			if refindx > 0 and (refindx == len(hdf['time']) or abs(sector_reference_time - hdf['time'][refindx-1]) < abs(sector_reference_time - hdf['time'][refindx])):
				refindx -= 1
			logger.info("WCS reference frame: %d", refindx)

			# Save WCS to the file:
			wcs.attrs['ref_frame'] = refindx

			if 'movement_kernel' not in hdf:
				# Calculate image motion:
				logger.info("Calculation Image Movement Kernels...")
				imk = ImageMovementKernel(image_ref=images['%04d' % refindx], warpmode='translation')
				kernel = np.empty((numfiles, imk.n_params), dtype='float64')

				tic = default_timer()
				if threads > 1:
					pool = multiprocessing.Pool(threads)

					datasets = _iterate_hdf_group(images)
					for k, knl in enumerate(pool.imap(imk.calc_kernel, datasets)):
						kernel[k, :] = knl
						logger.debug("Kernel: %s", knl)
						logger.debug("Estimate: %f sec/image", (default_timer()-tic)/(k+1))

					pool.close()
					pool.join()
				else:
					for k, dset in enumerate(images):
						kernel[k, :] = imk.calc_kernel(images[dset])
						logger.info("Kernel: %s", kernel[k, :])
						logger.debug("Estimate: %f sec/image", (default_timer()-tic)/(k+1))

				toc = default_timer()
				logger.info("Movement Kernel: %f sec/image", (toc-tic)/numfiles)

				# Save Image Motion Kernel to HDF5 file:
				dset = hdf.create_dataset('movement_kernel', data=kernel, **args)
				dset.attrs['warpmode'] = imk.warpmode
				dset.attrs['ref_frame'] = refindx

		logger.info("Done.")
		logger.info("Total: %f sec/image", (default_timer()-tic_total)/numfiles)
