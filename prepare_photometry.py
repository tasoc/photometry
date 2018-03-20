#!/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, with_statement, print_function, absolute_import
from six.moves import range
import argparse
import os
import numpy as np
import warnings
warnings.filterwarnings('ignore', category=FutureWarning, module='h5py')
import h5py
import logging
import multiprocessing
from astropy.wcs import WCS
from bottleneck import replace, nanmean
from photometry.backgrounds import fit_background
from photometry.utilities import load_ffi_fits, load_settings, find_ffi_files
from photometry import TESSQualityFlags, ImageMovementKernel
from timeit import default_timer
import itertools

#------------------------------------------------------------------------------
def create_hdf5(sector, camera, ccd):
	"""
	Restructure individual FFI images (in FITS format) into
	a combined HDF5 file which is used in the photometry
	pipeline.

	In this process the background flux in each FFI is
	estimated using the `backgrounds.fit_background` function.

	Parameters:
		sector (integer): The TESS observing sector.
		camera (integer): TESS camera number (1-4).
		ccd (integer): TESS CCD number (1-4).

	.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
	"""

	logger = logging.getLogger(__name__)

	settings = load_settings(sector=sector)
	sector_reference_time = settings.get('reference_time')

	input_folder = os.environ.get('TESSPHOT_INPUT', os.path.join(os.path.dirname(__file__), 'tests', 'input'))

	files = find_ffi_files(input_folder, camera, ccd)
	numfiles = len(files)
	logger.info("Number of files: %d", numfiles)
	if numfiles == 0:
		return

	# HDF5 file to be created/modified:
	hdf_file = os.path.join(input_folder, 'camera{0:d}_ccd{1:d}.hdf5'.format(camera, ccd))
	logger.debug("HDF5 File: %s", hdf_file)

	# Get image shape from the first file:
	img = load_ffi_fits(files[0])
	img_shape = img.shape

	# Common settings for HDF5 datasets:
	args = {
		'compression': 'lzf',
		'shuffle': True,
		'fletcher32': True,
		'chunks': True
	}

	threads = int(os.environ.get('SLURM_CPUS_PER_TASK', multiprocessing.cpu_count()))
	logger.info("Using %d processes.", threads)

	with h5py.File(hdf_file, 'a') as hdf:

		images = hdf.require_group('images')
		backgrounds = hdf.require_group('backgrounds')
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
				logger.info("%f sec/image", (toc-tic)/(numfiles-last_bck_fit))

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

				backgrounds.create_dataset(dset_name, data=bck, **args)

			toc = default_timer()
			logger.info("%f sec/image", (toc-tic)/(numfiles))

			# FIXME: Because HDF5 is stupid, this might not actually delete the data
			#        Maybe we need to run h5repack in the file at the end?
			del hdf['backgrounds_unsmoothed']
			hdf.flush()


		if len(images) < numfiles:
			SumImage = np.zeros((img_shape[0], img_shape[1]), dtype='float64')
			time = np.empty(numfiles, dtype='float64')
			timecorr = np.empty(numfiles, dtype='float32')
			cadenceno = np.empty(numfiles, dtype='int32')
			quality = np.empty(numfiles, dtype='int32')

			# Save list of file paths to the HDF5 file:
			filenames = [os.path.basename(fname).rstrip('.gz').encode('ascii', 'strict') for fname in files]
			hdf.require_dataset('imagespaths', (numfiles,), data=filenames, dtype=h5py.special_dtype(vlen=bytes), **args)

			is_tess = False
			for k, fname in enumerate(files):
				logger.debug("Processing image: %.2f%% - %s", 100*k/numfiles, fname)
				dset_name ='%04d' % k
				#if dset_name in hdf['images']: continue # Dont do this, because it will mess up the sumimage and time vector

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
					num_frm = hdr.get('NUM_FRM')
				elif hdr.get('NUM_FRM') != num_frm:
					logger.error("NUM_FRM is not constant!")

				# Load background from HDF file and subtract background from image,
				# if the background has not already been subtracted:
				if not hdr.get('BACKAPP', False):
					flux0 -= backgrounds[dset_name]

				# Save image subtracted the background in HDF5 file:
				images.create_dataset(dset_name, data=flux0, **args)

				# Add together images for sum-image:
				if TESSQualityFlags.filter(quality[k]):
					replace(flux0, np.nan, 0)
					SumImage += flux0

			SumImage /= numfiles

			# Save attributes
			images.attrs['NUM_FRM'] = num_frm
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
		if not np.all(hdf['time'][:-1] <= hdf['time'][1:]):
			logger.error("Time vector is not sorted")
			return

		# Check that the sector reference time is within the timespan of the time vector:
		if sector_reference_time < hdf['time'][0] or sector_reference_time > hdf['time'][-1]:
			logger.error("Sector reference time outside timespan of data")
			return

		if 'wcs' not in hdf or 'movement_kernel' not in hdf:
			# Find the reference image:
			refindx = np.searchsorted(hdf['time'], sector_reference_time, side='left')
			if refindx > 0 and (refindx == len(hdf['time']) or abs(sector_reference_time - hdf['time'][refindx-1]) < abs(sector_reference_time - hdf['time'][refindx])):
				refindx -= 1
			logger.info("WCS reference frame: %d", refindx)

			# Load the reference image and associated header:
			ref_image, hdr = load_ffi_fits(files[refindx], return_header=True)

			# Save WCS to the file:
			dset = hdf.require_dataset('wcs', (1,), dtype=h5py.special_dtype(vlen=bytes), **args)
			dset[0] = WCS(hdr).to_header_string(relax=True).strip().encode('ascii', 'strict')
			dset.attrs['ref_frame'] = refindx

			# Calculate image motion:
			imk = ImageMovementKernel(image_ref=ref_image, warpmode='translation')
			kernel = np.empty((numfiles, imk.n_params), dtype='float64')
			for k, dset in enumerate(images):
				kernel[k, :] = imk.calc_kernel(images[dset])
				logger.info("Kernel: %s", kernel[k, :])

			# Save Image Motion Kernel to HDF5 file:
			if 'movement_kernel' in hdf: del hdf['movement_kernel']
			dset = hdf.create_dataset('movement_kernel', data=kernel, **args)
			dset.attrs['warpmode'] = imk.warpmode
			dset.attrs['ref_frame'] = refindx

		logger.info("Done.")

#------------------------------------------------------------------------------
if __name__ == '__main__':

	# Parse command line arguments:
	parser = argparse.ArgumentParser(description='Run TESS Photometry pipeline on single star.')
	parser.add_argument('-d', '--debug', help='Print debug messages.', action='store_true')
	parser.add_argument('-q', '--quiet', help='Only report warnings and errors.', action='store_true')
	#parser.add_argument('-t', '--test', help='Use test data and ignore TESSPHOT_INPUT environment variable.', action='store_true')
	parser.add_argument('sector', type=int, help='TESS observing sector.')
	args = parser.parse_args()

	if args.quiet:
		logging_level = logging.WARNING
	elif args.debug:
		logging_level = logging.DEBUG
	else:
		logging_level = logging.INFO

	# Setup logging:
	formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
	logger = logging.getLogger(__name__)
	logger.setLevel(logging_level)
	console = logging.StreamHandler()
	console.setFormatter(formatter)
	logger_parent = logging.getLogger('photometry')
	logger_parent.setLevel(logging_level)
	if not logger.hasHandlers(): logger.addHandler(console)
	if not logger_parent.hasHandlers(): logger_parent.addHandler(console)

	for camera, ccd in itertools.product([1,2,3,4], [1,2,3,4]):
		create_hdf5(args.sector, camera, ccd)
