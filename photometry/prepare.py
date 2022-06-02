#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Preparation for Photometry extraction.

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import os
import numpy as np
import h5py
import sqlite3
import logging
import re
import multiprocessing
import itertools
import functools
import contextlib
import warnings
from astropy.io import fits
from astropy.wcs import WCS, NoConvergence, FITSFixedWarning
from bottleneck import replace, nanmean, nanmedian
from timeit import default_timer
from tqdm import tqdm, trange
from .catalog import download_catalogs
from .backgrounds import fit_background
from .utilities import (load_ffi_fits, find_ffi_files, find_catalog_files, find_nearest,
	find_tpf_files, load_sector_settings, to_tuple)
from .pixel_flags import pixel_manual_exclude, pixel_background_shenanigans
from . import TESSQualityFlags, PixelQualityFlags, ImageMovementKernel, fixes

#--------------------------------------------------------------------------------------------------
def quality_from_tpf(tpffile, time_start, time_end):
	"""
	Transfer quality flags from Target Pixel File to FFIs.

	Only quality flags in ``TESSQualityFlags.FFI_RELEVANT_BITMASK`` are transfered.

	Parameters:
		tpffile (string): Path to Target Pixel File to load quality flags from.
		time_start (ndarray): Array of start-timestamps for FFIs.
		time_end (ndarray): Array of end-timestamps for FFIs.

	Returns:
		ndarray: Quality flags for FFIs coming from Target Pixel File.

	.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
	"""
	# Load the timestamps and qualities from Target Pixel File:
	with fits.open(tpffile, mode='readonly', memmap=True) as hdu:
		time_tpf = hdu[1].data['TIME'] - hdu[1].data['TIMECORR']
		quality_tpf = hdu[1].data['QUALITY']

	# Remove any non-defined timestamps
	indx_goodtimes = np.isfinite(time_tpf)
	time_tpf = time_tpf[indx_goodtimes]
	quality_tpf = quality_tpf[indx_goodtimes]

	# Loop trough the FFI timestamps and see if any TPF qualities matches:
	Ntimes = len(time_start)
	quality = np.zeros(Ntimes, dtype='int32')
	for k in range(Ntimes):
		# Timestamps from TPF that correspond to the FFI:
		# TODO: Rewrite to using np.searchsorted
		indx = (time_tpf > time_start[k]) & (time_tpf < time_end[k])
		# Combine all TPF qualities, so if a flag is set in any
		# of the TPF timestamps it is set:
		quality[k] = np.bitwise_or.reduce(quality_tpf[indx])

	# Only transfer quality flags that are relevant for FFIs:
	quality = np.bitwise_and(quality, TESSQualityFlags.FFI_RELEVANT_BITMASK)

	return quality

#--------------------------------------------------------------------------------------------------
def _iterate_hdf_group(dset, start=0, stop=None):
	for d in range(start, stop if stop is not None else len(dset)):
		yield np.asarray(dset[f'{d:04d}'])

#--------------------------------------------------------------------------------------------------
def prepare_photometry(input_folder=None, sectors=None, cameras=None, ccds=None,
		calc_movement_kernel=False, backgrounds_pixels_threshold=0.5, output_file=None):
	"""
	Restructure individual FFI images (in FITS format) into
	a combined HDF5 file which is used in the photometry
	pipeline.

	In this process the background flux in each FFI is
	estimated using the :func:`backgrounds.fit_background` function.

	Parameters:
		input_folder (str): Input folder to create TODO list for. If ``None``, the input
			directory in the environment variable ``TESSPHOT_INPUT`` is used.
		sectors (iterable of int, optional): TESS Sectors. If ``None``,
			all sectors will be processed.
		cameras (iterable of int, optional): TESS camera number (1-4). If ``None``,
			all cameras will be processed.
		ccds (iterable of int, optional): TESS CCD number (1-4).
			If ``None``, all cameras will be processed.
		calc_movement_kernel (bool, optional): Should Image Movement Kernels be
			calculated for each image? If it is not calculated, only the default WCS
			movement kernel will be available when doing the folllowing photometry. Default=False.
		backgrounds_pixels_threshold (float): Percentage of times a pixel has to use used in
			background calculation in order to be included in the
			final list of contributing pixels. Default=0.5.
		output_file (str, optional): The file path where the output file should be saved.
			If not specified, the file will be saved into the input directory.
			Should only be used for testing, since the file would (proberly) otherwise end up with
			a wrong file name for running with the rest of the pipeline.

	Raises:
		NotADirectoryError: If the specified ``input_folder`` is not an existing directory
			or if settings table could not be loaded from the catalog SQLite file.

	.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
	"""

	logger = logging.getLogger(__name__)
	tqdm_settings = {
		'disable': None if logger.isEnabledFor(logging.INFO) else True,
		'dynamic_ncols': True
	}

	# Check the input folder, and load the default if not provided:
	if input_folder is None:
		input_folder = os.environ.get('TESSPHOT_INPUT', os.path.join(os.path.dirname(__file__), 'tests', 'input'))

	# Check that the given input directory is indeed a directory:
	if not os.path.isdir(input_folder):
		raise NotADirectoryError("The given path does not exist or is not a directory")

	# Make sure cameras and ccds are iterable:
	sectors = to_tuple(sectors)
	cameras = to_tuple(cameras, (1,2,3,4))
	ccds = to_tuple(ccds, (1,2,3,4))

	# Common settings for HDF5 datasets:
	args = {
		'compression': 'lzf',
		'shuffle': True,
		'fletcher32': True
	}
	imgchunks = (64, 64)
	# Commons settings for HDF5 datasets containing strings:
	# Note: Fletcher32 filter does not work with variable length strings.
	args_strings = {
		'dtype': h5py.special_dtype(vlen=bytes),
		'compression': 'lzf',
		'shuffle': True,
		'fletcher32': False
	}

	# If no sectors are provided, find all the available FFI files and figure out
	# which sectors they are all from:
	if sectors is None:
		sectors = []

		# TODO: Could we change this so we don't have to parse the filenames?
		for fname in find_ffi_files(input_folder):
			m = re.match(r'^tess.+-s(\d+)-.+\.fits', os.path.basename(fname))
			if int(m.group(1)) not in sectors:
				sectors.append(int(m.group(1)))

		# Also collect sectors from TPFs. They are needed for ensuring that
		# catalogs are available. Can be added directly to the sectors list,
		# since the HDF5 creation below will simply skip any sectors with
		# no FFIs available
		for fname in find_tpf_files(input_folder):
			m = re.match(r'^.+-s(\d+)[-_].+_(fast-)?tp\.fits', os.path.basename(fname))
			if int(m.group(1)) not in sectors:
				sectors.append(int(m.group(1)))

		sectors = tuple(sorted(sectors))
		logger.debug("Sectors found: %s", sectors)

	# Check if any sectors were found/provided:
	if not sectors:
		logger.error("No sectors were found")
		return

	# Make sure that catalog files are available in the input directory.
	# If they are not already, they will be downloaded from the cache:
	for sector, camera, ccd in itertools.product(sectors, cameras, ccds):
		download_catalogs(input_folder, sector, camera=camera, ccd=ccd)

	# Force multiprocessing to start fresh python interpreter processes.
	# This is already the default on Windows and MacOS, and the default "fork"
	# has sometimes caused deadlocks on Linux:
	# https://docs.python.org/3/library/multiprocessing.html#contexts-and-start-methods
	mp = multiprocessing.get_context('spawn')

	# Get the number of processes we can spawn in case it is needed for calculations:
	threads = int(os.environ.get('SLURM_CPUS_PER_TASK', mp.cpu_count()))
	logger.info("Using %d processes.", threads)

	# Start pool of workers:
	if threads > 1:
		pool = mp.Pool(threads)
		m = pool.imap
	else:
		m = map

	# Loop over each combination of camera and CCD:
	for sector, camera, ccd in itertools.product(sectors, cameras, ccds):
		logger.info("Running SECTOR=%d, CAMERA=%d, CCD=%d", sector, camera, ccd)
		tic_total = default_timer()

		# Find all the FFI files associated with this camera and CCD:
		files = find_ffi_files(input_folder, sector=sector, camera=camera, ccd=ccd)
		numfiles = len(files)
		logger.info("Number of files: %d", numfiles)
		if numfiles == 0:
			continue

		# Catalog file:
		catalog_file = find_catalog_files(input_folder, sector=sector, camera=camera, ccd=ccd)
		if len(catalog_file) != 1:
			logger.error("Catalog file could not be found: SECTOR=%d, CAMERA=%d, CCD=%d", sector, camera, ccd)
			continue
		logger.debug("Catalog File: %s", catalog_file[0])

		# Load catalog settings from the SQLite database:
		with contextlib.closing(sqlite3.connect(catalog_file[0])) as conn:
			conn.row_factory = sqlite3.Row
			cursor = conn.cursor()
			cursor.execute("SELECT sector,reference_time FROM settings LIMIT 1;")
			row = cursor.fetchone()
			if row is None:
				raise RuntimeError("Settings could not be loaded from catalog")
			#sector = row['sector']
			sector_reference_time = row['reference_time']
			cursor.close()

		# Cadence is simply changed in sector 28:
		cadence = load_sector_settings(sector)['ffi_cadence']

		# HDF5 file to be created/modified:
		if output_file is None:
			hdf_file = os.path.join(input_folder, f'sector{sector:03d}_camera{camera:d}_ccd{ccd:d}.hdf5')
		else:
			output_file = os.path.abspath(output_file)
			if not output_file.endswith('.hdf5'):
				output_file = output_file + '.hdf5'
			hdf_file = output_file
		logger.debug("HDF5 File: %s", hdf_file)

		# Get image shape from the first file:
		img = load_ffi_fits(files[0])
		img_shape = img.shape

		# Open the HDF5 file for editing:
		with h5py.File(hdf_file, 'a', libver='latest') as hdf:

			images = hdf.require_group('images')
			images_err = hdf.require_group('images_err')
			backgrounds = hdf.require_group('backgrounds')
			pixel_flags = hdf.require_group('pixel_flags')
			if 'wcs' in hdf and isinstance(hdf['wcs'], h5py.Dataset):
				del hdf['wcs']
			wcs = hdf.require_group('wcs')
			time_smooth = backgrounds.attrs.get('time_smooth', {1800: 3, 600: 9}[cadence])
			flux_cutoff = backgrounds.attrs.get('flux_cutoff', 8e4)
			bkgiters = backgrounds.attrs.get('bkgiters', 3)
			radial_cutoff = backgrounds.attrs.get('radial_cutoff', 2400)
			radial_pixel_step = backgrounds.attrs.get('radial_pixel_step', 15)
			radial_smooth = backgrounds.attrs.get('radial_smooth', 3)

			if len(backgrounds) < numfiles:
				# Because HDF5 is stupid, and it cant figure out how to delete data from
				# the file once it is in, we are creating another temp hdf5 file that
				# will hold thing we dont need in the final HDF5 file.
				tmp_hdf_file = hdf_file.replace('.hdf5', '.tmp.hdf5')
				with h5py.File(tmp_hdf_file, 'a', libver='latest') as hdftmp:
					dset_bck_us = hdftmp.require_group('backgrounds_unsmoothed')

					if len(pixel_flags) < numfiles:
						logger.info('Calculating backgrounds...')

						# Create wrapper function freezing some of the
						# additional keyword inputs:
						fit_background_wrapper = functools.partial(
							fit_background,
							flux_cutoff=flux_cutoff,
							bkgiters=bkgiters,
							radial_cutoff=radial_cutoff,
							radial_pixel_step=radial_pixel_step,
							radial_smooth=radial_smooth
						)

						tic = default_timer()

						last_bck_fit = -1 if len(pixel_flags) == 0 else int(sorted(list(pixel_flags.keys()))[-1])
						k = last_bck_fit+1
						for bck, mask in tqdm(m(fit_background_wrapper, files[k:]), initial=k, total=numfiles, **tqdm_settings):
							dset_name = f'{k:04d}'
							logger.debug("Background %d complete", k)
							logger.debug("Estimate: %f sec/image", (default_timer()-tic)/(k-last_bck_fit))

							dset_bck_us.create_dataset(dset_name, data=bck)

							# If we ever defined pixel flags above 256, we have to change this to uint16
							mask = np.asarray(np.where(mask, PixelQualityFlags.NotUsedForBackground, 0), dtype='uint8')
							pixel_flags.create_dataset(dset_name, data=mask, chunks=imgchunks, **args)

							k += 1

						hdf.flush()
						hdftmp.flush()
						toc = default_timer()
						logger.info("Background estimation: %f sec/image", (toc-tic)/(numfiles-last_bck_fit))

					# Smooth the backgrounds along the time axis:
					logger.info('Smoothing backgrounds in time...')
					backgrounds.attrs['time_smooth'] = time_smooth
					backgrounds.attrs['flux_cutoff'] = flux_cutoff
					backgrounds.attrs['bkgiters'] = bkgiters
					backgrounds.attrs['radial_cutoff'] = radial_cutoff
					backgrounds.attrs['radial_pixel_step'] = radial_pixel_step
					backgrounds.attrs['radial_smooth'] = radial_smooth
					w = time_smooth//2
					tic = default_timer()
					for k in trange(numfiles, **tqdm_settings):
						dset_name = f'{k:04d}'
						if dset_name in backgrounds: continue

						indx1 = max(k-w, 0)
						indx2 = min(k+w+1, numfiles)
						logger.debug("Smoothing background %d: %d -> %d", k, indx1, indx2)

						block = np.empty((img_shape[0], img_shape[1], indx2-indx1), dtype='float32')
						logger.debug(block.shape)
						for i, n in enumerate(range(indx1, indx2)):
							block[:, :, i] = dset_bck_us[f'{n:04d}']

						bck = nanmean(block, axis=2)
						#bck_err = np.sqrt(nansum(block_err**2, axis=2)) / time_smooth

						backgrounds.create_dataset(dset_name, data=bck, chunks=imgchunks, **args)

					toc = default_timer()
					logger.info("Background smoothing: %f sec/image", (toc-tic)/numfiles)

				# Flush changes to the permanent HDF5 file:
				hdf.flush()

				# Delete the temporary HDF5 file again:
				with contextlib.suppress(FileNotFoundError):
					os.remove(tmp_hdf_file)

			if len(images) < numfiles or len(wcs) < numfiles or 'sumimage' not in hdf or 'backgrounds_pixels_used' not in hdf or 'time_start' not in hdf:
				SumImage = np.zeros((img_shape[0], img_shape[1]), dtype='float64')
				Nimg = np.zeros_like(SumImage, dtype='int32')
				time = np.empty(numfiles, dtype='float64')
				timecorr = np.empty(numfiles, dtype='float32')
				time_start = np.empty(numfiles, dtype='float64')
				time_stop = np.empty(numfiles, dtype='float64')
				cadenceno = np.empty(numfiles, dtype='int32')
				quality = np.empty(numfiles, dtype='int32')
				UsedInBackgrounds = np.zeros_like(SumImage, dtype='int32')

				# Save list of file paths to the HDF5 file:
				filenames = [os.path.basename(fname).rstrip('.gz').encode('ascii', 'strict') for fname in files]
				hdf.require_dataset('imagespaths', (numfiles,), data=filenames, **args_strings)

				is_tess = False
				attributes = {
					'CAMERA': None,
					'CCD': None,
					'DATA_REL': None,
					'PROCVER': None,  # Used to distinguish bad timestamps in sectors 20 and 21
					'NUM_FRM': None,
					'NREADOUT': None,
					'CRMITEN': None,
					'CRBLKSZ': None,
					'CRSPOC': None
				}
				logger.info('Final processing of individual images...')
				tic = default_timer()
				for k, fname in enumerate(tqdm(files, **tqdm_settings)):
					dset_name = f'{k:04d}'

					# Load the FITS file data and the header:
					flux0, hdr, flux0_err = load_ffi_fits(fname, return_header=True, return_uncert=True)

					# Check if this is real TESS data:
					# Could proberly be done more elegant, but if it works, it works...
					if not is_tess and hdr.get('TELESCOP') == 'TESS' and hdr.get('NAXIS1') == 2136 and hdr.get('NAXIS2') == 2078:
						is_tess = True

					# Pick out the important bits from the header:
					# Keep time in BTJD. If we want BJD we could
					# simply add BJDREFI + BJDREFF:
					time_start[k] = hdr['TSTART']
					time_stop[k] = hdr['TSTOP']
					time[k] = 0.5*(hdr['TSTART'] + hdr['TSTOP'])
					timecorr[k] = hdr.get('BARYCORR', 0)

					# Get cadence-numbers from headers, if they are available.
					# This header is not added before sector 6, so in that case
					# we are doing a simple scaling of the timestamps.
					if 'FFIINDEX' in hdr:
						cadenceno[k] = hdr['FFIINDEX']
					elif is_tess and cadence == 1800:
						# The following numbers comes from unofficial communication
						# with Doug Caldwell and Roland Vanderspek:
						# The timestamp in TJD and the corresponding cadenceno:
						first_time = 0.5*(1325.317007851970 + 1325.337841177751) - 3.9072474e-03
						first_cadenceno = 4697
						timedelt = 1800/86400
						# Extracpolate the cadenceno as a simple linear relation:
						offset = first_cadenceno - first_time/timedelt
						cadenceno[k] = np.round((time[k] - timecorr[k])/timedelt + offset)
					elif is_tess:
						raise RuntimeError("Could not determine CADENCENO for TESS data")
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
								logger.error("%s: %s is not constant! (%s, %s)", dset_name, key, value, hdr.get(key))

					# Find pixels marked for manual exclude:
					manexcl = pixel_manual_exclude(flux0, hdr)

					# Add manual excludes to pixel flags:
					if np.any(manexcl):
						pixel_flags[dset_name][manexcl] |= PixelQualityFlags.ManualExclude

					if dset_name not in images:
						# Mask out manually excluded data before saving:
						flux0[manexcl] = np.nan
						flux0_err[manexcl] = np.nan

						# Load background from HDF file and subtract background from image,
						# if the background has not already been subtracted:
						if not hdr.get('BACKAPP', False):
							flux0 -= backgrounds[dset_name]

						# Save image subtracted the background in HDF5 file:
						images.create_dataset(dset_name, data=flux0, chunks=imgchunks, **args)
						images_err.create_dataset(dset_name, data=flux0_err, chunks=imgchunks, **args)
					else:
						flux0 = np.asarray(images[dset_name])
						flux0[manexcl] = np.nan

					# Save the World Coordinate System of each image:
					# Check if the WCS actually works for each image, and if not, set it to an empty string
					if dset_name not in wcs:
						dset = wcs.create_dataset(dset_name, (1,), **args_strings)

						# Test the World Coordinate System solution.
						with warnings.catch_warnings():
							warnings.filterwarnings('ignore', category=FITSFixedWarning)
							w = WCS(header=hdr, relax=True)
						fp = w.calc_footprint()
						test_coords = np.atleast_2d(fp[0, :])
						try:
							w.all_world2pix(test_coords, 0, ra_dec_order=True, maxiter=50)
						except (NoConvergence, ValueError):
							logger.info("%s has bad WCS.", dset_name)
							dset[0] = ''
						else:
							dset[0] = w.to_header_string(relax=True).strip().encode('ascii', 'strict')

					# Add together images for sum-image:
					if TESSQualityFlags.filter(quality[k]):
						Nimg += np.isfinite(flux0)
						replace(flux0, np.nan, 0)
						SumImage += flux0

					# Add together the number of times each pixel was used in the background estimation:
					UsedInBackgrounds += (np.asarray(pixel_flags[dset_name]) & PixelQualityFlags.NotUsedForBackground == 0)

				# Normalize sumimage
				SumImage /= Nimg

				# Correct timestamp offset that was in early data releases:
				time_start = fixes.time_offset(time_start, attributes, datatype='ffi', timepos='start')
				time_stop = fixes.time_offset(time_stop, attributes, datatype='ffi', timepos='end')
				time, fixed_time_offset = fixes.time_offset(time, attributes, datatype='ffi', timepos='mid', return_flag=True)

				# Single boolean image indicating if the pixel was (on average) used in the background estimation:
				if 'backgrounds_pixels_used' not in hdf:
					UsedInBackgrounds = (UsedInBackgrounds/numfiles > backgrounds_pixels_threshold)
					dset_uibkg = hdf.create_dataset('backgrounds_pixels_used', data=UsedInBackgrounds, dtype='bool', chunks=imgchunks, **args)
					dset_uibkg.attrs['threshold'] = backgrounds_pixels_threshold

				# Save attributes
				images.attrs['SECTOR'] = sector
				images.attrs['CADENCE'] = cadence
				images.attrs['TIME_OFFSET_CORRECTED'] = fixed_time_offset
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
				if 'time_start' in hdf: del hdf['time_start']
				if 'time_stop' in hdf: del hdf['time_stop']
				if 'sumimage' in hdf: del hdf['sumimage']
				if 'cadenceno' in hdf: del hdf['cadenceno']
				if 'quality' in hdf: del hdf['quality']
				hdf.create_dataset('sumimage', data=SumImage, **args)
				hdf.create_dataset('time', data=time, **args)
				hdf.create_dataset('timecorr', data=timecorr, **args)
				hdf.create_dataset('time_start', data=time_start, **args)
				hdf.create_dataset('time_stop', data=time_stop, **args)
				hdf.create_dataset('cadenceno', data=cadenceno, **args)
				hdf.create_dataset('quality', data=quality, **args)
				hdf.flush()

				logger.info("Individual image processing: %f sec/image", (default_timer()-tic)/numfiles)
			else:
				# Extract things that are needed further down:
				SumImage = np.asarray(hdf['sumimage'])
				timecorr = np.asarray(hdf['timecorr'])
				time_start = np.asarray(hdf['time_start'])
				time_stop = np.asarray(hdf['time_stop'])
				quality = np.asarray(hdf['quality'])

			# Detections and flagging of Background Shenanigans:
			if pixel_flags.attrs.get('bkgshe_done', -1) < numfiles-1:
				logger.info("Detecting background shenanigans...")
				tic_bkgshe = default_timer()

				# Load settings and create wrapper function with keywords set:
				bkgshe_threshold = pixel_flags.attrs.get('bkgshe_threshold', 40)
				pixel_flags.attrs['bkgshe_threshold'] = bkgshe_threshold
				pixel_background_shenanigans_wrapper = functools.partial(
					pixel_background_shenanigans,
					SumImage=SumImage
				)

				tmp_hdf_file = hdf_file.replace('.hdf5', '.tmp.hdf5')
				with h5py.File(tmp_hdf_file, 'a', libver='latest') as hdftmp:
					# Temporary dataset that will be used to store large array
					# of background shenanigans indicator images:
					pixel_flags_ind = hdftmp.require_dataset('pixel_flags_individual',
						shape=(SumImage.shape[0], SumImage.shape[1], numfiles),
						chunks=(SumImage.shape[0], SumImage.shape[1], 1),
						dtype='float32'
					)

					# Run the background shenanigans extractor in parallel:
					last_bkgshe = pixel_flags_ind.attrs.get('bkgshe_done', -1)
					if last_bkgshe < numfiles-1:
						tic = default_timer()
						k = last_bkgshe + 1
						for bckshe in tqdm(m(pixel_background_shenanigans_wrapper, _iterate_hdf_group(images, start=k)), initial=k, total=numfiles, **tqdm_settings):
							pixel_flags_ind[:, :, k] = bckshe
							pixel_flags_ind.attrs['bkgshe_done'] = k
							k += 1
							hdftmp.flush()
						logger.info("Background Shenanigans: %f sec/image", (default_timer()-tic)/(numfiles-last_bkgshe))

					# Calculate the mean Background Shenanigans indicator:
					if 'mean_shenanigans' not in hdftmp:
						logger.info("Calculating mean shenanigans...")
						tic = default_timer()

						# Calculate robust mean by calculating the
						# median in chunks and then taking the mean of them.
						# This is to avoid loading the entire array into memory
						mean_shenanigans = np.zeros_like(SumImage, dtype='float64')
						block = 25
						indicies = list(range(numfiles))
						np.random.seed(0)
						np.random.shuffle(indicies)
						mean_shenanigans_block = np.empty((SumImage.shape[0], SumImage.shape[1], block))
						for k in trange(0, numfiles, block, **tqdm_settings):
							# Take median of a random block of images:
							for j, i in enumerate(indicies[k:k+block]):
								mean_shenanigans_block[:, :, j] = pixel_flags_ind[:, :, i]
							bckshe = nanmedian(mean_shenanigans_block, axis=2)

							# Add the median block to the mean image:
							replace(bckshe, np.NaN, 0)
							mean_shenanigans += bckshe
						mean_shenanigans /= np.ceil(numfiles/block)
						logger.info("Mean Background Shenanigans: %f sec/image", (default_timer()-tic)/numfiles)

						# Save the mean shenanigans to the HDF5 file:
						hdftmp.create_dataset('mean_shenanigans', data=mean_shenanigans)
					else:
						mean_shenanigans = np.asarray(hdftmp['mean_shenanigans'])

					#msmax = max(np.abs(np.min(mean_shenanigans)), np.abs(np.max(mean_shenanigans)))
					#fig = plt.figure()
					#plot_image(mean_shenanigans, scale='linear', vmin=-msmax, vmax=msmax, cmap='coolwarm', cbar=True)
					#fig.savefig('test.png', bbox_inches='tight')

					logger.info("Setting background shenanigans...")
					tic = default_timer()
					for k in trange(numfiles, **tqdm_settings):
						dset_name = f'{k:04d}'
						bckshe = np.asarray(pixel_flags_ind[:, :, k])

						#img = bckshe - mean_shenanigans
						#img[np.abs(img) <= bkgshe_threshold/2] = 0
						#fig = plt.figure(figsize=(8,9))
						#ax = fig.add_subplot(111)
						#plot_image(img, ax=ax, scale='linear', vmin=-bkgshe_threshold, vmax=bkgshe_threshold, cmap="RdBu_r", cbar=True)
						#ax.set_xticks([])
						#ax.set_yticks([])
						#fig.savefig(dset_name + '.png', bbox_inches='tight')
						#plt.close(fig)

						# Create the mask as anything that significantly pops out
						# (both positive and negative) in the image:
						bckshe = np.abs(bckshe - mean_shenanigans) > bkgshe_threshold

						# Clear any old flags:
						indx = (np.asarray(pixel_flags[dset_name]) & PixelQualityFlags.BackgroundShenanigans != 0)
						if np.any(indx):
							pixel_flags[dset_name][indx] -= PixelQualityFlags.BackgroundShenanigans

						# Save the new flags to the permanent HDF5 file:
						if np.any(bckshe):
							pixel_flags[dset_name][bckshe] |= PixelQualityFlags.BackgroundShenanigans

						pixel_flags.attrs['bkgshe_done'] = k
						hdf.flush()
					logger.info("Setting Background Shenanigans: %f sec/image", (default_timer()-tic)/numfiles)

				# Delete the temporary HDF5 file again:
				with contextlib.suppress(FileNotFoundError):
					os.remove(tmp_hdf_file)

				logger.info("Total Background Shenanigans: %f sec/image", (default_timer()-tic_bkgshe)/numfiles)

			# Check that the time vector is sorted:
			if not np.all(hdf['time'][:-1] < hdf['time'][1:]):
				logger.error("Time vector is not sorted")
				return

			# Transfer quality flags from TPF files from the same CAMERA and CCD to the FFIs:
			if not hdf['quality'].attrs.get('TRANSFER_FROM_TPF', False):
				logger.info("Transfering QUALITY flags from TPFs to FFIs...")

				# Select (max) five random TPF targets from the given sector, camera and ccd:
				tpffiles = find_tpf_files(input_folder, sector=sector, camera=camera, ccd=ccd, findmax=5)
				if len(tpffiles) == 0:
					logger.warning("No TPF files found for SECTOR=%d, CAMERA=%d, CCD=%d and quality flags could therefore not be propergated.", sector, camera, ccd)
				else:
					# Run through each of the found TPF files and build the quality column from them,
					# by simply setting the flag if it is found in any of the files:
					quality_tpf = np.zeros(numfiles, dtype='int32')
					for tpffile in tpffiles:
						quality_tpf |= quality_from_tpf(tpffile, time_start-timecorr, time_stop-timecorr)

					# Inspect the differences with the the qualities set in
					indx_diff = (quality | quality_tpf != quality)
					logger.info("%d qualities will be updated (%.1f%%).", np.sum(indx_diff), 100*np.sum(indx_diff)/numfiles)

					# New quality:
					quality |= quality_tpf

					# Update the quality column in the HDF5 file:
					hdf['quality'][:] = quality
					hdf['quality'].attrs['TRANSFER_FROM_TPF'] = True
					hdf.flush()

			# Check that the sector reference time is within the timespan of the time vector:
			sector_reference_time_tjd = sector_reference_time - 2457000
			if sector_reference_time_tjd < hdf['time'][0] or sector_reference_time_tjd > hdf['time'][-1]:
				logger.error("Sector reference time outside timespan of data")

			# Find the reference image:
			# Create numpy masked array of timestamps with good quality data and
			# find the index in the good timestamps closest to the reference time.
			bad_wcs_mask = np.asarray([not bool(wcs[w][0].strip()) for w in wcs], dtype='bool')
			bad_times_mask = (quality != 0) | bad_wcs_mask
			good_times = np.ma.masked_array(hdf['time'], mask=bad_times_mask)
			refindx = find_nearest(good_times, sector_reference_time_tjd)
			logger.info("WCS reference frame: %d", refindx)

			# Make sure that the WCS solution and quality at the chosen refindx are good.
			# They should be, this is just to catch any errors early.
			if quality[refindx] != 0 or not wcs[f'{refindx:04d}'][0]:
				raise RuntimeError("The chosen refindx does not contain good values.")

			# Save WCS to the file:
			wcs.attrs['ref_frame'] = refindx

			if calc_movement_kernel and 'movement_kernel' not in hdf:
				# Calculate image motion:
				logger.info("Calculation Image Movement Kernels...")
				imk = ImageMovementKernel(image_ref=images[f'{refindx:04d}'], warpmode='translation')
				kernel = np.empty((numfiles, imk.n_params), dtype='float64')

				tic = default_timer()

				datasets = _iterate_hdf_group(images)
				for k, knl in enumerate(tqdm(m(imk.calc_kernel, datasets), **tqdm_settings)):
					kernel[k, :] = knl
					logger.debug("Kernel: %s", knl)
					logger.debug("Estimate: %f sec/image", (default_timer()-tic)/(k+1))

				toc = default_timer()
				logger.info("Movement Kernel: %f sec/image", (toc-tic)/numfiles)

				# Save Image Motion Kernel to HDF5 file:
				dset = hdf.create_dataset('movement_kernel', data=kernel, **args)
				dset.attrs['warpmode'] = imk.warpmode
				dset.attrs['ref_frame'] = refindx

		logger.info("Done.")
		logger.info("Total: %f sec/image", (default_timer()-tic_total)/numfiles)

	# Close workers again:
	if threads > 1:
		pool.close()
		pool.join()
