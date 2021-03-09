#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Collection of utility functions that can be used throughout
the photometry package.

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import numpy as np
from astropy.io import fits
from bottleneck import move_median, nanmedian, nanmean, allnan, nanargmin, nanargmax
import logging
import tqdm
from scipy.special import erf
from scipy.stats import binned_statistic
import configparser
import json
import os.path
import fnmatch
import glob
import re
import itertools
from functools import lru_cache
import requests
import concurrent.futures
from threading import Lock
from collections import defaultdict

# Constants:
mad_to_sigma = 1.482602218505602 #: Constant for converting from MAD to SIGMA. Constant is 1/norm.ppf(3/4)

#--------------------------------------------------------------------------------------------------
@lru_cache(maxsize=1)
def load_settings():
	"""
	Load settings.

	Returns:
		:class:`configparser.ConfigParser`:
	"""

	settings = configparser.ConfigParser()
	settings.read(os.path.join(os.path.dirname(__file__), 'data', 'settings.ini'))
	return settings

#--------------------------------------------------------------------------------------------------
@lru_cache(maxsize=10)
def load_sector_settings(sector=None):

	with open(os.path.join(os.path.dirname(__file__), 'data', 'sectors.json'), 'r') as fid:
		settings = json.load(fid)

	if sector is not None:
		return settings['sectors'][str(sector)]

	return settings

#--------------------------------------------------------------------------------------------------
def find_ffi_files(rootdir, sector=None, camera=None, ccd=None):
	"""
	Search directory recursively for TESS FFI images in FITS format.

	Parameters:
		rootdir (str): Directory to search recursively for TESS FFI images.
		sector (int or None, optional): Only return files from the given sector.
			If ``None``, files from all sectors are returned.
		camera (int or None, optional): Only return files from the given camera number (1-4).
			If ``None``, files from all cameras are returned.
		ccd (int or None, optional): Only return files from the given CCD number (1-4).
			If ``None``, files from all CCDs are returned.

	Returns:
		list: List of full paths to FFI FITS files found in directory. The list will
			be sorted accoridng to the filename of the files, e.g. primarily by time.
	"""

	logger = logging.getLogger(__name__)

	# Create the filename pattern to search for:
	sector_str = '????' if sector is None else f'{sector:04d}'
	camera = '?' if camera is None else str(camera)
	ccd = '?' if ccd is None else str(ccd)
	filename_pattern = f'tess*-s{sector_str:s}-{camera:s}-{ccd:s}-????-[xsab]_ffic.fits*'
	logger.debug("Searching for FFIs in '%s' using pattern '%s'", rootdir, filename_pattern)

	# Do a recursive search in the directory, finding all files that match the pattern:
	matches = []
	for root, dirnames, filenames in os.walk(rootdir, followlinks=True):
		for filename in fnmatch.filter(filenames, filename_pattern):
			matches.append(os.path.join(root, filename))

	# Sort the list of files by thir filename:
	matches.sort(key=lambda x: os.path.basename(x))

	return matches

#--------------------------------------------------------------------------------------------------
@lru_cache(maxsize=10)
def _find_tpf_files(rootdir, sector=None, cadence=None):

	logger = logging.getLogger(__name__)

	# Create the filename pattern to search for:
	sector_str = r'\d{4}' if sector is None else f'{sector:04d}'
	suffix = {None: '(fast-)?tp', 120: 'tp', 20: 'fast-tp'}[cadence]
	re_pattern = r'^tess\d+-s(?P<sector>' + sector_str + r')-(?P<starid>\d+)-\d{4}-[xsab]_' + suffix + r'\.fits(\.gz)?$'
	regexps = [re.compile(re_pattern)]
	logger.debug("Searching for TPFs in '%s' using pattern '%s'", rootdir, re_pattern)

	# Pattern used for TESS Alert data:
	if cadence is None or cadence == 120:
		sector_str = r'\d{2}' if sector is None else f'{sector:02d}'
		re_pattern2 = r'^hlsp_tess-data-alerts_tess_phot_(?P<starid>\d+)-s(?P<sector>' + sector_str + r')_tess_v\d+_tp\.fits(\.gz)?$'
		regexps.append(re.compile(re_pattern2))
		logger.debug("Searching for TPFs in '%s' using pattern '%s'", rootdir, re_pattern2)

	# Do a recursive search in the directory, finding all files that match the pattern:
	filedict = defaultdict(list)
	for root, dirnames, filenames in os.walk(rootdir, followlinks=True):
		for filename in filenames:
			for regex in regexps:
				m = regex.match(filename)
				if m:
					starid = int(m.group('starid'))
					filedict[starid].append(os.path.join(root, filename))
					break

	# Ensure that each list is sorted by itself. We do this once here
	# so we don't have to do it each time a specific starid is requested:
	for key in filedict.keys():
		filedict[key].sort(key=lambda x: os.path.basename(x))

	return filedict

#--------------------------------------------------------------------------------------------------
def find_tpf_files(rootdir, starid=None, sector=None, camera=None, ccd=None, cadence=None,
	findmax=None):
	"""
	Search directory recursively for TESS Target Pixel Files.

	The function is cached, meaning the first time it is run on a particular ``rootdir``
	the list of files in that directory will be read and cached to memory and used in
	subsequent calls to the function. This means that any changes to files on disk after
	the first call of the function will not be picked up in subsequent calls to this function.

	Parameters:
		rootdir (str): Directory to search recursively for TESS TPF files.
		starid (int, optional): Only return files from the given TIC number.
			If ``None``, files from all TIC numbers are returned.
		sector (int, optional): Only return files from the given sector.
			If ``None``, files from all sectors are returned.
		camera (int or None, optional): Only return files from the given camera number (1-4).
			If ``None``, files from all cameras are returned.
		ccd (int, optional): Only return files from the given CCD number (1-4).
			If ``None``, files from all CCDs are returned.
		cadence (int, optional): Only return files from the given cadence (20 or 120).
			If ``None``, files from all cadences are returned.
		findmax (int, optional): Maximum number of files to return.
			If ``None``, return all files.

	Note:
		Filtering on camera and/or ccd will cause the program to read the headers
		of the files in order to determine the camera and ccd from which they came.
		This can significantly slow down the query.

	Returns:
		list: List of full paths to TPF FITS files found in directory. The list will
			be sorted according to the filename of the files, i.e. primarily by time.
	"""

	if cadence is not None and cadence not in (120, 20):
		raise ValueError("Invalid cadence. Must be either 20 or 120.")

	# Call cached function which searches for files on disk:
	filedict = _find_tpf_files(rootdir, sector=sector, cadence=cadence)

	if starid is not None:
		files = filedict.get(starid, [])
	else:
		# If we are not searching for a particilar starid,
		# simply flatten the dict to a list of all found files
		# and sort the list of files by thir filename:
		files = list(itertools.chain(*filedict.values()))
		files.sort(key=lambda x: os.path.basename(x))

	# Expensive check which involve opening the files and reading headers:
	# We are only removing elements, and preserving the ordering, so there
	# is no need for re-sorting the list afterwards.
	if camera is not None or ccd is not None:
		matches = []
		for fpath in files:
			if camera is not None and fits.getval(fpath, 'CAMERA', ext=0) != camera:
				continue
			if ccd is not None and fits.getval(fpath, 'CCD', ext=0) != ccd:
				continue

			# Add the file to the list, but stop if we have already
			# reached the number of files we need to find:
			matches.append(fpath)
			if findmax is not None and len(matches) >= findmax:
				break

		files = matches

	# Just to ensure that we are not returning more than we should:
	if findmax is not None and len(files) > findmax:
		files = files[:findmax]

	return files

#--------------------------------------------------------------------------------------------------
@lru_cache(maxsize=32)
def find_hdf5_files(rootdir, sector=None, camera=None, ccd=None):
	"""
	Search the input directory for HDF5 files matching constraints.

	Parameters:
		rootdir (str): Directory to search for HDF5 files.
		sector (int, list or None, optional): Only return files from the given sectors.
			If ``None``, files from all TIC numbers are returned.
		camera (int, list or None, optional): Only return files from the given camera.
			If ``None``, files from all cameras are returned.
		ccd (int, list or None, optional): Only return files from the given ccd.
			If ``None``, files from all ccds are returned.

	Returns:
		list: List of paths to HDF5 files matching constraints.
	"""

	sector = to_tuple(sector, (None,))
	camera = to_tuple(camera, (1,2,3,4))
	ccd = to_tuple(ccd, (1,2,3,4))

	filelst = []
	for sector, camera, ccd in itertools.product(sector, camera, ccd):
		sector_str = '???' if sector is None else f'{sector:03d}'
		filelst += glob.glob(os.path.join(rootdir, f'sector{sector_str:s}_camera{camera:d}_ccd{ccd:d}.hdf5'))

	return filelst

#--------------------------------------------------------------------------------------------------
@lru_cache(maxsize=32)
def find_catalog_files(rootdir, sector=None, camera=None, ccd=None):
	"""
	Search the input directory for CATALOG (sqlite) files matching constraints.

	Parameters:
		rootdir (str): Directory to search for CATALOG files.
		sector (int, list or None, optional): Only return files from the given sectors.
			If ``None``, files from all TIC numbers are returned.
		camera (int, list or None, optional): Only return files from the given camera.
			If ``None``, files from all cameras are returned.
		ccd (int, list or None, optional): Only return files from the given ccd.
			If ``None``, files from all ccds are returned.

	Returns:
		list: List of paths to CATALOG files matching constraints.
	"""

	sector = to_tuple(sector, (None,))
	camera = to_tuple(camera, (1,2,3,4))
	ccd = to_tuple(ccd, (1,2,3,4))

	filelst = []
	for sector, camera, ccd in itertools.product(sector, camera, ccd):
		sector_str = '???' if sector is None else f'{sector:03d}'
		filelst += glob.glob(os.path.join(rootdir, f'catalog_sector{sector_str:s}_camera{camera:d}_ccd{ccd:d}.sqlite'))

	return filelst

#--------------------------------------------------------------------------------------------------
def load_ffi_fits(path, return_header=False, return_uncert=False):
	"""
	Load FFI FITS file.

	Calibrations columns and rows are trimmed from the image.

	Parameters:
		path (str): Path to FITS file.
		return_header (boolean, optional): Return FITS headers as well. Default is ``False``.

	Returns:
		numpy.ndarray: Full Frame Image.
		list: If ``return_header`` is enabled, will return a dict of the FITS headers.
	"""

	with fits.open(path, memmap=True, mode='readonly') as hdu:
		hdr = hdu[0].header
		if hdr.get('TELESCOP') == 'TESS' and hdu[1].header.get('NAXIS1') == 2136 and hdu[1].header.get('NAXIS2') == 2078:
			img = hdu[1].data[0:2048, 44:2092]
			if return_uncert:
				imgerr = np.asarray(hdu[2].data[0:2048, 44:2092], dtype='float32')
			headers = dict(hdu[0].header)
			headers.update(dict(hdu[1].header))
		else:
			img = hdu[0].data
			headers = dict(hdu[0].header)
			if return_uncert:
				imgerr = np.asarray(hdu[1].data, dtype='float32')

	# Make sure its an numpy array with the correct data type:
	img = np.asarray(img, dtype='float32')

	if return_uncert and return_header:
		return img, headers, imgerr
	elif return_uncert:
		return img, imgerr
	elif return_header:
		return img, headers
	else:
		return img

#--------------------------------------------------------------------------------------------------
def to_tuple(input, default=None):
	"""
	Convert iterable or single values to tuple.

	This function is used for converting inputs, perticularly for
	preparing input to functions cached with :func:`functools.lru_cache`,
	to ensure inputs are hashable.

	Parameters:
		input: Input to convert to tuple.
		default: If ``input`` is ``None`` return this instead.

	Returns:
		tuple: ``input`` converted to tuple.
	"""
	if input is None:
		return default
	if isinstance(input, (list, set, frozenset, np.ndarray)):
		return tuple(input)
	if isinstance(input, (int, float)):
		return (input, )
	return input

#--------------------------------------------------------------------------------------------------
def _move_median_central_1d(x, width_points):
	y = move_median(x, width_points, min_count=1)
	y = np.roll(y, -width_points//2+1)
	for k in range(width_points//2+1):
		y[k] = nanmedian(x[:(k+2)])
		y[-(k+1)] = nanmedian(x[-(k+2):])
	return y

#--------------------------------------------------------------------------------------------------
def move_median_central(x, width_points, axis=0):
	return np.apply_along_axis(_move_median_central_1d, axis, x, width_points)

#--------------------------------------------------------------------------------------------------
def add_proper_motion(ra, dec, pm_ra, pm_dec, bjd, epoch=2000.0):
	"""
	Project coordinates (ra,dec) with proper motions to new epoch.

	Parameters:
		ra (float) : Right ascension.
		dec (float) : Declination.
		pm_ra (float) : Proper motion in RA (mas/year).
		pm_dec (float) : Proper motion in Declination (mas/year).
		bjd (float) : Julian date to calculate coordinates for.
		epoch (float, optional) : Epoch of ``ra`` and ``dec``. Default=2000.

	Returns:
		(float, float) : RA and Declination at the specified date.
	"""

	# Convert BJD to epoch (year):
	epoch_now = (bjd - 2451544.5)/365.25 + 2000.0

	# How many years since the catalog's epoch?
	timeelapsed = epoch_now - epoch # in years

	# Calculate the dec:
	decrate = pm_dec/3600000.0  # in degrees/year (assuming original was in mas/year)
	decindegrees = dec + timeelapsed*decrate

	# Calculate the unprojected rate of RA motion, using the mean declination between
	# the catalog and present epoch.
	rarate = pm_ra/np.cos((dec + timeelapsed*decrate/2.0)*np.pi/180.0)/3600000.0  # in degress of RA/year (assuming original was in mas/year)
	raindegrees = ra + timeelapsed*rarate

	# Return the current positions
	return raindegrees, decindegrees

#--------------------------------------------------------------------------------------------------
def integratedGaussian(x, y, flux, x_0, y_0, sigma=1):
	"""
	Evaluate a 2D symmetrical Gaussian integrated in pixels.

	Parameters:
		x (numpy array) : x coordinates at which to evaluate the PSF.
		y (numpy array) : y coordinates at which to evaluate the PSF.
		flux (float) : Integrated value.
		x_0 (float) : Centroid position.
		y_0 (float) : Centroid position.
		sigma (float, optional) : Standard deviation of Gaussian. Default=1.

	Returns:
		numpy array : 2D Gaussian integrated pixel values at (x,y).

	Note:
		Inspired by
		https://github.com/astropy/photutils/blob/master/photutils/psf/models.py

	Example:

	>>> import numpy as np
	>>> X, Y = np.meshgrid(np.arange(-1,2), np.arange(-1,2))
	>>> integratedGaussian(X, Y, 10, 0, 0)
	array([[ 0.58433556,  0.92564571,  0.58433556],
		[ 0.92564571,  1.46631496,  0.92564571],
		[ 0.58433556,  0.92564571,  0.58433556]])
	"""
	denom = np.sqrt(2) * sigma
	return (flux / 4 * ((erf((x - x_0 + 0.5) / denom)
		- erf((x - x_0 - 0.5) / denom)) * (erf((y - y_0 + 0.5) / denom)
		- erf((y - y_0 - 0.5) / denom)))) # noqa: ET126

#--------------------------------------------------------------------------------------------------
def mag2flux(mag, zp=20.60654144):
	"""
	Convert from magnitude to flux using scaling relation from
	aperture photometry. This is an estimate.

	The scaling is based on fast-track TESS data from sectors 1 and 2.

	Parameters:
		mag (float): Magnitude in TESS band.

	Returns:
		float: Corresponding flux value
	"""
	return np.clip(10**(-0.4*(mag - zp)), 0, None)

#--------------------------------------------------------------------------------------------------
def sphere_distance(ra1, dec1, ra2, dec2):
	"""
	Calculate the great circle distance between two points using the Vincenty formulae.

	Parameters:
		ra1 (float or ndarray): Longitude of first point in degrees.
		dec1 (float or ndarray): Lattitude of first point in degrees.
		ra2 (float or ndarray): Longitude of second point in degrees.
		dec2 (float or ndarray): Lattitude of second point in degrees.

	Returns:
		ndarray: Distance between points in degrees.

	Note:
		https://en.wikipedia.org/wiki/Great-circle_distance
	"""

	# Convert angles to radians:
	ra1 = np.deg2rad(ra1)
	ra2 = np.deg2rad(ra2)
	dec1 = np.deg2rad(dec1)
	dec2 = np.deg2rad(dec2)

	# Calculate distance using Vincenty formulae:
	return np.rad2deg(np.arctan2(
		np.sqrt( (np.cos(dec2)*np.sin(ra2-ra1))**2 + (np.cos(dec1)*np.sin(dec2) - np.sin(dec1)*np.cos(dec2)*np.cos(ra2-ra1))**2 ),
		np.sin(dec1)*np.sin(dec2) + np.cos(dec1)*np.cos(dec2)*np.cos(ra2-ra1)
	))

#--------------------------------------------------------------------------------------------------
def radec_to_cartesian(radec):
	"""
	Convert spherical coordinates as (ra, dec) pairs to cartesian coordinates (x,y,z).

	Parameters:
		radec (ndarray): Array with ra-dec pairs in degrees.

	Returns:
		ndarray: (x,y,z) coordinates corresponding to input coordinates.
	"""
	radec = np.atleast_2d(radec)
	xyz = np.empty((radec.shape[0], 3), dtype='float64')

	phi = np.radians(radec[:,0])
	theta = np.pi/2 - np.radians(radec[:,1])

	xyz[:,0] = np.sin(theta) * np.cos(phi)
	xyz[:,1] = np.sin(theta) * np.sin(phi)
	xyz[:,2] = np.cos(theta)
	return xyz

#--------------------------------------------------------------------------------------------------
def cartesian_to_radec(xyz):
	"""
	Convert cartesian coordinates (x,y,z) to spherical coordinates in ra-dec form.

	Parameters:
		radec (ndarray): Array with ra-dec pairs.

	Returns:
		ndarray: ra-dec coordinates in degrees corresponding to input coordinates.
	"""
	xyz = np.atleast_2d(xyz)
	radec = np.empty((xyz.shape[0], 2), dtype='float64')
	radec[:,1] = np.pi/2 - np.arccos(xyz[:,2])
	radec[:,0] = np.arctan2(xyz[:,1], xyz[:,0])

	indx = radec[:,0] < 0
	radec[indx,0] = 2*np.pi - np.abs(radec[indx,0])
	indx = radec[:,0] > 2*np.pi
	radec[indx,0] -= 2*np.pi

	return np.degrees(radec)

#--------------------------------------------------------------------------------------------------
def rms_timescale(time, flux, timescale=3600/86400):
	"""
	Compute robust RMS on specified timescale. Using MAD scaled to RMS.

	Parameters:
		time (ndarray): Timestamps in days.
		flux (ndarray): Flux to calculate RMS for.
		timescale (float, optional): Timescale to bin timeseries before calculating RMS.
			Default=1 hour.

	Returns:
		float: Robust RMS on specified timescale.

	.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
	"""

	time = np.asarray(time)
	flux = np.asarray(flux)
	if len(flux) == 0 or allnan(flux):
		return np.nan
	if len(time) == 0 or allnan(time):
		raise ValueError("Invalid time-vector specified. No valid timestamps.")

	time_min = np.nanmin(time)
	time_max = np.nanmax(time)
	if not np.isfinite(time_min) or not np.isfinite(time_max) or time_max - time_min <= 0:
		raise ValueError("Invalid time-vector specified")

	# Construct the bin edges seperated by the timescale:
	bins = np.arange(time_min, time_max, timescale)
	bins = np.append(bins, time_max)

	# Bin the timeseries to one hour:
	indx = np.isfinite(flux)
	flux_bin, _, _ = binned_statistic(time[indx], flux[indx], nanmean, bins=bins)

	# Compute robust RMS value (MAD scaled to RMS)
	return mad_to_sigma * nanmedian(np.abs(flux_bin - nanmedian(flux_bin)))

#--------------------------------------------------------------------------------------------------
def find_nearest(array, value):
	"""
	Search array for value and return the index where the value is closest.

	Parameters:
		array (ndarray): Array to search.
		value: Value to search array for.

	Returns:
		int: Index of ``array`` closest to ``value``.

	Raises:
		ValueError: If ``value`` is NaN.

	.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
	"""
	if np.isnan(value):
		raise ValueError("Invalid search value")
	if np.isposinf(value):
		return nanargmax(array)
	if np.isneginf(value):
		return nanargmin(array)
	return nanargmin(np.abs(array - value))
	#idx = np.searchsorted(array, value, side='left')
	#if idx > 0 and (idx == len(array) or abs(value - array[idx-1]) <= abs(value - array[idx])):
	#	return idx-1
	#else:
	#	return idx

#--------------------------------------------------------------------------------------------------
def download_file(url, destination, desc=None, timeout=60,
	position_holders=None, position_lock=None):
	"""
	Download file from URL and place into specified destination.

	Parameters:
		url (str): URL to file to be downloaded.
		destination (str): Path where to save file.
		desc (str, optional): Description to write next to progress-bar.
		timeout (float): Time to wait for server response in seconds. Default=60.

	.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
	"""

	logger = logging.getLogger(__name__)
	tqdm_settings = {
		'unit': 'B',
		'unit_scale': True,
		'position': None,
		'leave': True,
		'disable': None if logger.isEnabledFor(logging.INFO) else True,
		'desc': desc
	}

	if position_holders is not None:
		tqdm_settings['leave'] = False
		position_lock.acquire()
		tqdm_settings['position'] = position_holders.index(False)
		position_holders[tqdm_settings['position']] = True
		position_lock.release()

	try:
		# Start stream from URL and throw an error for bad status codes:
		response = requests.get(url, stream=True, allow_redirects=True, timeout=timeout)
		response.raise_for_status()

		total_size = int(response.headers.get('content-length', 0))
		block_size = 1024

		with open(destination, 'wb') as handle:
			with tqdm.tqdm(total=total_size, **tqdm_settings) as pbar:
				for block in response.iter_content(block_size):
					handle.write(block)
					pbar.update(len(block))

		if os.path.getsize(destination) != total_size:
			raise Exception("File not downloaded correctly")

	except: # noqa: E722, pragma: no cover
		logger.exception("Could not download file")
		if os.path.exists(destination):
			os.remove(destination)
		raise

	finally:
		# Pause before returning to give progress bar time to write.
		if position_holders is not None:
			position_lock.acquire()
			position_holders[tqdm_settings['position']] = False
			position_lock.release()

#--------------------------------------------------------------------------------------------------
def download_parallel(urls, workers=4, timeout=60):
	"""
	Download several files in parallel using multiple threads.

	Parameters:
		urls (iterable): List of files to download. Each element should consist of a list or tuple,
			containing two elements: The URL to download, and the path to the destination where the
			file should be saved.
		workers (int, optional): Number of threads to use for downloading. Default=4.
		timeout (float): Time to wait for server response in seconds. Default=60.

	.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
	"""

	# Don't overcomplicate things for a singe file:
	if len(urls) == 1:
		download_file(urls[0][0], urls[0][1], timeout=timeout)
		return

	workers = min(workers, len(urls))
	position_holders = [False] * workers
	plock = Lock()

	def _wrapper(arg):
		download_file(arg[0], arg[1],
			timeout=timeout,
			position_holders=position_holders,
			position_lock=plock)

	errors = []
	with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
		# Start the load operations and mark each future with its URL
		future_to_url = {executor.submit(_wrapper, url): url for url in urls}
		for future in concurrent.futures.as_completed(future_to_url):
			url = future_to_url[future]
			try:
				future.result()
			except: # noqa: E722, pragma: no cover
				errors.append(url[0])

	if errors:
		raise Exception("Errors encountered during download of the following URLs:\n%s" % '\n'.join(errors))

#--------------------------------------------------------------------------------------------------
class TqdmLoggingHandler(logging.Handler):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)

	def emit(self, record):
		try:
			msg = self.format(record)
			tqdm.tqdm.write(msg)
			self.flush()
		except (KeyboardInterrupt, SystemExit): # pragma: no cover
			raise
		except: # noqa: E722, pragma: no cover
			self.handleError(record)

#--------------------------------------------------------------------------------------------------
class ListHandler(logging.Handler):
	"""
	A logging.Handler that writes messages into a list object.

	The standard logging.QueueHandler/logging.QueueListener can not be used
	for this because the QueueListener runs in a private thread, not the
	main thread.

	.. warning::
		This handler is not thread-safe. Do not use it in threaded environments.
	"""

	def __init__(self, *args, message_queue, **kwargs):
		"""Initialize by copying the queue and sending everything else to superclass."""
		super().__init__(*args, **kwargs)
		self.message_queue = message_queue

	def emit(self, record):
		"""Add the formatted log message (sans newlines) to the queue."""
		self.message_queue.append(self.format(record).rstrip('\n'))

#--------------------------------------------------------------------------------------------------
class LoggerWriter(object):
	"""
	File-like object which passes input into a logger.

	Can be used together with :py:func:`contextlib.redirect_stdout`
	or :py:func:`contextlib.redirect_stderr` to redirect streams to the given logger.
	Can be useful for wrapping codes which uses normal :py:func:`print` functions for logging.

	.. code-block:: python
		:linenos:

		logger = logging.getLogger(__name__)
		with contextlib.redirect_stdout(LoggerWriter(logger, logging.INFO)):
			print("This goes into the logger instead of STDOUT")

	.. warning::
		This object is not thread-safe. Do not use it in threaded environments.

	.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
	"""
	def __init__(self, logger, level=logging.INFO):
		self.logger = logger
		self.level = level

	def write(self, message):
		if message.strip() != '':
			self.logger.log(self.level, message)

#--------------------------------------------------------------------------------------------------
def sqlite_drop_column(conn, table, col):
	"""
	Drop table column from SQLite table.

	Since SQLite does not have functionality for dropping/deleting columns
	in existing tables, this function can provide this functionality.
	This is done by temporarily copying the entire table, so this can be
	quite an expensive operation.

	Parameters:
		conn (:class:`sqlite3.Connection`): Connection to SQLite database.
		table (str): Table to drop column from.
		col (str): Column to be dropped from table.

	.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
	"""

	# Get a list of columns in the existing table:
	cursor = conn.cursor()
	cursor.execute("PRAGMA table_info({table:s})".format(table=table))
	columns = [col[1] for col in cursor.fetchall()]
	if col not in columns:
		raise ValueError("Column '%s' not found in table '%s'" % (col, table))
	columns.remove(col)
	columns = ','.join(columns)

	# Get list of index associated with the table:
	cursor.execute("SELECT name,sql FROM sqlite_master WHERE type='index' AND tbl_name=?", [table])
	index = cursor.fetchall()
	index_names = [row[0] for row in index]
	index_sql = [row[1] for row in index]

	# Warn if any index exist with the column to be removed:
	regex_index = re.compile(r'^CREATE( UNIQUE)? INDEX (.+) ON ' + re.escape(table) + r'\s*\((.+)\).*$', re.IGNORECASE)
	for sql in index_sql:
		m = regex_index.match(sql)
		if not m:
			raise Exception("COULD NOT UNDERSTAND SQL") # pragma: no cover
		index_columns = [i.strip() for i in m.group(3).split(',')]
		if col in index_columns:
			raise Exception("Column is used in INDEX %s." % m.group(2))

	# Store the current foreign_key setting:
	cursor.execute("PRAGMA foreign_keys;")
	current_foreign_keys = cursor.fetchone()[0]

	#BEGIN TRANSACTION;
	#cursor.execute('BEGIN')
	try:
		cursor.execute("PRAGMA foreign_keys=off;")

		# Drop all indexes associated with table:
		for name in index_names:
			cursor.execute("DROP INDEX {0:s};".format(name))

		cursor.execute("ALTER TABLE {table:s} RENAME TO {table:s}_backup;".format(table=table))
		cursor.execute("CREATE TABLE {table:s} AS SELECT {columns:s} FROM {table:s}_backup;".format(table=table, columns=columns))
		cursor.execute("DROP TABLE {table:s}_backup;".format(table=table))

		# Recreate all index associated with table:
		for sql in index_sql:
			cursor.execute(sql)

		conn.commit()
	except: # noqa: E722, pragma: no cover
		conn.rollback()
		raise
	finally:
		cursor.execute("PRAGMA foreign_keys=%s;" % current_foreign_keys)
