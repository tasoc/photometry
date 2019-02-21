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
import re
import multiprocessing
from astropy.wcs import WCS
from bottleneck import replace, nanmean
from timeit import default_timer
import itertools
import contextlib
from .backgrounds import fit_background
from .utilities import load_ffi_fits, find_ffi_files, find_catalog_files
from photometry import TESSQualityFlags, ImageMovementKernel

#------------------------------------------------------------------------------
def _iterate_hdf_group(dset):
	for d in dset:
		yield np.asarray(dset[d])

#------------------------------------------------------------------------------
def create_hdf5(input_folder=None, sectors=None, cameras=None, ccds=None, calc_movement_kernel=False):
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
		calc_movement_kernel (boolean, optional): Should Image Movement Kernels be calculated for each image?
			If it is not calculated, only the default WCS movement kernel will be available when doing the folllowing photometry. Default=False.

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

	# If no sectors are provided, find all the available FFI files and figure out
	# which sectors they are all from:
	if sectors is None:
		# TODO: Could we change this so we don't have to parse the filenames?
		files = find_ffi_files(input_folder)
		sectors = []
		for fname in files:
			m = re.match(r'^tess.+-s(\d+)-.+\.fits', os.path.basename(fname))
			if int(m.group(1)) not in sectors:
				sectors.append(int(m.group(1)))
		logger.debug("Sectors found: %s", sectors)
	else:
		sectors = (sectors,)

	# Check if any sectors were found/provided:
	if not sectors:
		logger.error("No sectors were found")
		return

	# Loop over each combination of camera and CCD:
	for sector, camera, ccd in itertools.product(sectors, cameras, ccds):
		logger.info("Running SECTOR=%s, CAMERA=%s, CCD=%s", sector, camera, ccd)
		tic_total = default_timer()

		# Find all the FFI files associated with this camera and CCD:
		files = find_ffi_files(input_folder, sector=sector, camera=camera, ccd=ccd)
		numfiles = len(files)
		logger.info("Number of files: %d", numfiles)
		if numfiles == 0:
			continue

		# Catalog file:
		catalog_file = find_catalog_files(input_folder, sector=sector, camera=camera, ccd=ccd)
		logger.debug("Catalog File: %s", catalog_file)
		if len(catalog_file) != 1:
			logger.error("Catalog file could not be found: SECTOR=%s, CAMERA=%s, CCD=%s", sector, camera, ccd)
			continue

		# Load catalog settings from the SQLite database:
		with contextlib.closing(sqlite3.connect(catalog_file[0])) as conn:
			conn.row_factory = sqlite3.Row
			cursor = conn.cursor()
			cursor.execute("SELECT sector,reference_time FROM settings LIMIT 1;")
			row = cursor.fetchone()
			if row is None:
				raise IOError("Settings could not be loaded from catalog")
			#sector = row['sector']
			sector_reference_time = row['reference_time']
			cursor.close()

		# HDF5 file to be created/modified:
		hdf_file = os.path.join(input_folder, 'sector{0:03d}_camera{1:d}_ccd{2:d}.hdf5'.format(sector, camera, ccd))
		logger.debug("HDF5 File: %s", hdf_file)

		# Get image shape from the first file:
		img = load_ffi_fits(files[0])
		img_shape = img.shape

		# Open the HDF5 file for editing:
		with h5py.File(hdf_file, 'a', libver='latest') as hdf:

			images = hdf.require_group('images')
			images_err = hdf.require_group('images_err')
			backgrounds = hdf.require_group('backgrounds')
			masks = hdf.require_group('backgrounds_masks')
			if 'wcs' in hdf and isinstance(hdf['wcs'], h5py.Dataset): del hdf['wcs']
			wcs = hdf.require_group('wcs')
			time_smooth = backgrounds.attrs.get('time_smooth', 3)

			if len(backgrounds) < numfiles:
				# Because HDF5 is stupid, and it cant figure out how to delete data from
				# the file once it is in, we are creating another temp hdf5 file that
				# will hold thing we dont need in the final HDF5 file.
				tmp_hdf_file = hdf_file.replace('.hdf5', '.tmp.hdf5')
				with h5py.File(tmp_hdf_file, 'a', libver='latest') as hdftmp:
					dset_bck_us = hdftmp.require_group('backgrounds_unsmoothed')

					if len(masks) < numfiles:

						tic = default_timer()
						if threads > 1:
							pool = multiprocessing.Pool(threads)
							m = pool.imap
						else:
							m = map

						last_bck_fit = -1 if len(masks) == 0 else int(sorted(list(masks.keys()))[-1])
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
						hdftmp.flush()
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
						#bck_err = np.sqrt(nansum(block_err**2, axis=2)) / time_smooth

						backgrounds.create_dataset(dset_name, data=bck, chunks=imgchunks, **args)

					toc = default_timer()
					logger.info("Background smoothing: %f sec/image", (toc-tic)/numfiles)

				# Flush changes to the permanent HDF5 file:
				hdf.flush()

				# Delete the temporary HDF5 file again:
				if os.path.exists(tmp_hdf_file):
					os.remove(tmp_hdf_file)


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
					'CAMERA': None,
					'CCD': None,
					'DATA_REL': None,
					'NUM_FRM': None,
					'CRMITEN': None,
					'CRBLKSZ': None,
					'CRSPOC': None
				}
				for k, fname in enumerate(files):
					logger.debug("Processing image: %.2f%% - %s", 100*k/numfiles, fname)
					dset_name = '%04d' % k

					# Load the FITS file data and the header:
					flux0, hdr, flux0_err = load_ffi_fits(fname, return_header=True, return_uncert=True)

					# Check if this is real TESS data:
					# Could proberly be done more elegant, but if it works, it works...
					if not is_tess and hdr.get('TELESCOP') == 'TESS' and hdr.get('NAXIS1') == 2136 and hdr.get('NAXIS2') == 2078:
						is_tess = True

					# Pick out the important bits from the header:
					# Keep time in BTJD. If we want BJD we could
					# simply add BJDREFI + BJDREFF:
					time[k] = 0.5*(hdr['TSTART'] + hdr['TSTOP'])
					timecorr[k] = hdr.get('BARYCORR', 0)

					# Get cadence-numbers from headers, if they are available.
					# This header is not added before sector 6, so in that case
					# we are doing a simple scaling of the timestamps.
					if 'FFIINDEX' in hdr:
						cadenceno[k] = hdr['FFIINDEX']
					elif is_tess:
						# The following numbers comes from unofficial communication
						# with Doug Caldwell and Roland Vanderspek:
						# The timestamp in TJD and the corresponding cadenceno:
						first_time = 0.5*(1325.317007851970 + 1325.337841177751) - 3.9072474e-03
						first_cadenceno = 4697
						timedelt = 1800/86400
						# Extracpolate the cadenceno as a simple linear relation:
						offset = first_cadenceno - first_time/timedelt
						cadenceno[k] = np.round((time[k] - timecorr[k])/timedelt + offset)
					else:
						cadenceno[k] = k+1

					# Data quality flags:
					quality[k] = hdr.get('DQUALITY', 0)

					if k == 0:
						for key in attributes.keys():
							attributes[key] = hdr.get(key)
					else:
						for key, value in attributes.items():
							if hdr.get(key) != value:
								logger.error("%s is not constant!", key)

					if dset_name not in images:
						# Load background from HDF file and subtract background from image,
						# if the background has not already been subtracted:
						if not hdr.get('BACKAPP', False):
							flux0 -= backgrounds[dset_name]

						# Save image subtracted the background in HDF5 file:
						images.create_dataset(dset_name, data=flux0, chunks=imgchunks, **args)
						images_err.create_dataset(dset_name, data=flux0_err, chunks=imgchunks, **args)
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
				images.attrs['SECTOR'] = sector
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
			sector_reference_time_tjd = sector_reference_time - 2457000
			if sector_reference_time_tjd < hdf['time'][0] or sector_reference_time_tjd > hdf['time'][-1]:
				logger.error("Sector reference time outside timespan of data")
				#return

			# Find the reference image:
			refindx = np.searchsorted(hdf['time'], sector_reference_time_tjd, side='left')
			if refindx > 0 and (refindx == len(hdf['time']) or abs(sector_reference_time_tjd - hdf['time'][refindx-1]) < abs(sector_reference_time_tjd - hdf['time'][refindx])):
				refindx -= 1
			logger.info("WCS reference frame: %d", refindx)

			# Save WCS to the file:
			wcs.attrs['ref_frame'] = refindx

			if calc_movement_kernel and 'movement_kernel' not in hdf:
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
