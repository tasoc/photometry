#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Collection of utility functions that can be used throughout
the photometry package.

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

import logging
import os.path
import contextlib
import numpy as np
from bottleneck import move_median, nanmedian, nanmean, allnan, nanargmin, nanargmax
import tqdm
from scipy.special import erf
from scipy.stats import binned_statistic
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import concurrent.futures
from threading import Lock

# Constants:
mad_to_sigma = 1.482602218505602 #: Constant for converting from MAD to SIGMA. Constant is 1/norm.ppf(3/4)

#--------------------------------------------------------------------------------------------------
def to_tuple(inp, default=None):
	"""
	Convert iterable or single values to tuple.

	This function is used for converting inputs, perticularly for
	preparing input to functions cached with :func:`functools.lru_cache`,
	to ensure inputs are hashable.

	Parameters:
		inp: Input to convert to tuple.
		default: If ``input`` is ``None`` return this instead.

	Returns:
		tuple: ``inp`` converted to tuple.
	"""
	if inp is None:
		return default
	if isinstance(inp, (list, set, frozenset, np.ndarray)):
		return tuple(inp)
	if isinstance(inp, (int, float, bool, str)):
		return (inp, )
	return inp

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
		x (numpy.ndarray): x coordinates at which to evaluate the PSF.
		y (numpy.ndarray): y coordinates at which to evaluate the PSF.
		flux (float): Integrated value.
		x_0 (float): Centroid position.
		y_0 (float): Centroid position.
		sigma (float, optional): Standard deviation of Gaussian. Default=1.

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
def mag2flux(mag, zp=20.451):
	"""
	Convert from magnitude to flux using scaling relation from
	aperture photometry. This is an estimate.

	The default scaling is based on TASOC Data Release 5 from sectors 1-5.

	Parameters:
		mag (ndarray): Magnitude in TESS band.
		zp (float): Zero-point to use in scaling. Default is estimated from
			TASOC Data Release 5 from TESS sectors 1-5.

	Returns:
		ndarray: Corresponding flux value
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
	position_holders=None, position_lock=None, showprogress=None):
	"""
	Download file from URL and place into specified destination.

	Parameters:
		url (str): URL to file to be downloaded.
		destination (str): Path where to save file.
		desc (str, optional): Description to write next to progress-bar.
		timeout (float): Time to wait for server response in seconds. Default=60.
		showprogress (bool): Force showing the progress bar. If ``None``, the
			progressbar is shown based on the logging level and output type.

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
	if showprogress is not None:
		tqdm_settings['disable'] = not showprogress

	if position_holders is not None:
		tqdm_settings['leave'] = False
		position_lock.acquire()
		tqdm_settings['position'] = position_holders.index(False)
		position_holders[tqdm_settings['position']] = True
		position_lock.release()

	# Strategy for retrying failing requests several times
	# with a small increasing sleep in between:
	retry_strategy = Retry(
		total=3,
		backoff_factor=1,
		status_forcelist=[413, 429, 500, 502, 503, 504],
		allowed_methods=['HEAD', 'GET'],
	)
	adapter = HTTPAdapter(max_retries=retry_strategy)

	try:
		with requests.Session() as http:
			http.mount("https://", adapter)
			http.mount("http://", adapter)

			# Start stream from URL and throw an error for bad status codes:
			response = http.get(url, stream=True, allow_redirects=True, timeout=timeout)
			response.raise_for_status()

			total_size = response.headers.get('content-length', None)
			if total_size is not None:
				total_size = int(total_size)
			block_size = 1024

			with open(destination, 'wb') as handle:
				with tqdm.tqdm(total=total_size, **tqdm_settings) as pbar:
					for block in response.iter_content(block_size):
						handle.write(block)
						pbar.update(len(block))

		if total_size is not None and os.path.getsize(destination) != total_size:
			raise RuntimeError("File not downloaded correctly")

	except: # noqa: E722, pragma: no cover
		logger.exception("Could not download file")
		with contextlib.suppress(FileNotFoundError):
			os.remove(destination)
		raise

	finally:
		# Pause before returning to give progress bar time to write.
		if position_holders is not None:
			position_lock.acquire()
			position_holders[tqdm_settings['position']] = False
			position_lock.release()

#--------------------------------------------------------------------------------------------------
def download_parallel(urls, workers=4, timeout=60, showprogress=None):
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
		download_file(urls[0][0], urls[0][1], timeout=timeout, showprogress=showprogress)
		return

	workers = min(workers, len(urls))
	position_holders = [False] * workers
	plock = Lock()

	def _wrapper(arg):
		download_file(arg[0], arg[1],
			timeout=timeout,
			showprogress=showprogress,
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
		raise RuntimeError("Errors encountered during download of the following URLs:\n%s" % '\n'.join(errors))

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

	def flush(self):
		pass
