#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Collection of utility functions that can be used throughout
the photometry package.

.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
"""

from __future__ import division, with_statement, print_function, absolute_import
from six.moves import range
import numpy as np
from astropy.io import fits
from bottleneck import move_median, nanmedian, nanmean
import logging
from scipy.special import erf
from scipy.stats import binned_statistic
import json
import os.path
import fnmatch
import glob
import itertools
import warnings

# Filter out annoying warnings:
warnings.filterwarnings('ignore', module='scipy', category=FutureWarning, message='Using a non-tuple sequence for multidimensional indexing is deprecated;', lineno=607)

# Constants:
mad_to_sigma = 1.482602218505602 #: Constant for converting from MAD to SIGMA. Constant is 1/norm.ppf(3/4)

#------------------------------------------------------------------------------
def load_settings(sector=None):

	with open(os.path.join(os.path.dirname(__file__), 'data', 'settings.json'), 'r') as fid:
		settings = json.load(fid)

	if sector is not None:
		return settings['sectors'][str(sector)]

	return settings

#------------------------------------------------------------------------------
def find_ffi_files(rootdir, sector=None, camera=None, ccd=None):
	"""
	Search directory recursively for TESS FFI images in FITS format.

	Parameters:
		rootdir (string): Directory to search recursively for TESS FFI images.
		sector (integer or None, optional): Only return files from the given sector. If ``None``, files from all sectors are returned.
		camera (integer or None, optional): Only return files from the given camera number (1-4). If ``None``, files from all cameras are returned.
		ccd (integer or None, optional): Only return files from the given CCD number (1-4). If ``None``, files from all CCDs are returned.

	Returns:
		list: List of full paths to FFI FITS files found in directory. The list will
		      be sorted accoridng to the filename of the files, e.g. primarily by time.
	"""

	logger = logging.getLogger(__name__)

	# Create the filename pattern to search for:
	sector_str = '????' if sector is None else '{0:04d}'.format(sector)
	camera = '?' if camera is None else str(camera)
	ccd = '?' if ccd is None else str(ccd)
	filename_pattern = 'tess*-s{sector:s}-{camera:s}-{ccd:s}-????-[xsab]_ffic.fits*'.format(
		sector=sector_str,
		camera=camera,
		ccd=ccd
	)
	logger.debug("Searching for FFIs in '%s' using pattern '%s'", rootdir, filename_pattern)

	# Do a recursive search in the directory, finding all files that match the pattern:
	matches = []
	for root, dirnames, filenames in os.walk(rootdir, followlinks=True):
		for filename in fnmatch.filter(filenames, filename_pattern):
			matches.append(os.path.join(root, filename))

	# Sort the list of files by thir filename:
	matches.sort(key = lambda x: os.path.basename(x))

	return matches

#------------------------------------------------------------------------------
def find_tpf_files(rootdir, starid=None, sector=None, camera=None, ccd=None, findmax=None):
	"""
	Search directory recursively for TESS Target Pixel Files.

	Parameters:
		rootdir (string): Directory to search recursively for TESS TPF files.
		starid (integer or None, optional): Only return files from the given TIC number. If ``None``, files from all TIC numbers are returned.
		sector (integer or None, optional): Only return files from the given sector. If ``None``, files from all sectors are returned.
		camera (integer or None, optional): Only return files from the given camera number (1-4). If ``None``, files from all cameras are returned.
		ccd (integer or None, optional): Only return files from the given CCD number (1-4). If ``None``, files from all CCDs are returned.
		findmax (integer or None, optional): Maximum number of files to return. If ``None``, return all files.

	Note:
		Filtering on camera and/or ccd will cause the program to read the headers
		of the files in order to determine the camera and ccd from which they came.
		This can significantly slow down the query.

	Returns:
		list: List of full paths to TPF FITS files found in directory. The list will
		      be sorted accoriding to the filename of the files, e.g. primarily by time.
	"""

	logger = logging.getLogger(__name__)

	# Create the filename pattern to search for:
	sector_str = '????' if sector is None else '{0:04d}'.format(sector)
	starid_str = '*' if starid is None else '{0:016d}'.format(starid)
	filename_pattern = 'tess*-s{sector:s}-{starid:s}-????-[xsab]_tp.fits*'.format(
		sector=sector_str,
		starid=starid_str
	)

	# Pattern used for TESS Alert data:
	sector_str = '??' if sector is None else '{0:02d}'.format(sector)
	starid_str = '*' if starid is None else '{0:011d}'.format(starid)
	filename_pattern2 = 'hlsp_tess-data-alerts_tess_phot_{starid:s}-s{sector:s}_tess_v?_tp.fits*'.format(
		sector=sector_str,
		starid=starid_str
	)

	logger.debug("Searching for TPFs in '%s' using pattern '%s'", rootdir, filename_pattern)
	logger.debug("Searching for TPFs in '%s' using pattern '%s'", rootdir, filename_pattern2)

	# Do a recursive search in the directory, finding all files that match the pattern:
	breakout = False
	matches = []
	for root, dirnames, filenames in os.walk(rootdir, followlinks=True):
		for filename in filenames:
			if fnmatch.fnmatch(filename, filename_pattern) or fnmatch.fnmatch(filename, filename_pattern2):
				fpath = os.path.join(root, filename)
				if camera is not None and fits.getval(fpath, 'CAMERA', ext=0) != camera:
					continue

				if ccd is not None and fits.getval(fpath, 'CCD', ext=0) != ccd:
					continue

				matches.append(fpath)
				if findmax is not None and len(matches) >= findmax:
					breakout=True
					break
		if breakout: break

	# Sort the list of files by thir filename:
	matches.sort(key = lambda x: os.path.basename(x))

	return matches

#------------------------------------------------------------------------------
def find_hdf5_files(rootdir, sector=None, camera=None, ccd=None):
	"""
	Search the input directory for HDF5 files matching constraints.

	Parameters:
		rootdir (string): Directory to search for HDF5 files.
		sector (integer, list or None, optional): Only return files from the given sectors. If ``None``, files from all TIC numbers are returned.
		camera (integer, list or None, optional): Only return files from the given camera. If ``None``, files from all cameras are returned.
		ccd (integer, list or None, optional): Only return files from the given ccd. If ``None``, files from all ccds are returned.

	Returns:
		list: List of paths to HDF5 files matching constraints.
	"""

	if not isinstance(sector, (list, tuple)): sector = (sector,)
	if not isinstance(camera, (list, tuple)): camera = (1,2,3,4) if camera is None else (camera,)
	if not isinstance(ccd, (list, tuple)): ccd = (1,2,3,4) if ccd is None else (ccd,)

	filelst = []
	for sector, camera, ccd in itertools.product(sector, camera, ccd):
		filelst += glob.glob(os.path.join(rootdir, 'sector{0:s}_camera{1:d}_ccd{2:d}.hdf5'.format(
			'???' if sector is None else '%03d' % sector,
			camera,
			ccd
		)))

	return filelst

#------------------------------------------------------------------------------
def find_catalog_files(rootdir, sector=None, camera=None, ccd=None):
	"""
	Search the input directory for CATALOG (sqlite) files matching constraints.

	Parameters:
		rootdir (string): Directory to search for CATALOG files.
		sector (integer, list or None, optional): Only return files from the given sectors. If ``None``, files from all TIC numbers are returned.
		camera (integer, list or None, optional): Only return files from the given camera. If ``None``, files from all cameras are returned.
		ccd (integer, list or None, optional): Only return files from the given ccd. If ``None``, files from all ccds are returned.

	Returns:
		list: List of paths to CATALOG files matching constraints.
	"""

	if not isinstance(sector, (list, tuple)): sector = (sector,)
	if not isinstance(camera, (list, tuple)): camera = (1,2,3,4) if camera is None else (camera,)
	if not isinstance(ccd, (list, tuple)): ccd = (1,2,3,4) if ccd is None else (ccd,)

	filelst = []
	for sector, camera, ccd in itertools.product(sector, camera, ccd):
		filelst += glob.glob(os.path.join(rootdir, 'catalog_sector{0:s}_camera{1:d}_ccd{2:d}.sqlite'.format(
			'???' if sector is None else '%03d' % sector,
			camera,
			ccd
		)))

	return filelst

#------------------------------------------------------------------------------
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

#------------------------------------------------------------------------------
def _move_median_central_1d(x, width_points):
	y = move_median(x, width_points, min_count=1)
	y = np.roll(y, -width_points//2+1)
	for k in range(width_points//2+1):
		y[k] = nanmedian(x[:(k+2)])
		y[-(k+1)] = nanmedian(x[-(k+2):])
	return y

#------------------------------------------------------------------------------
def move_median_central(x, width_points, axis=0):
	return np.apply_along_axis(_move_median_central_1d, axis, x, width_points)

#------------------------------------------------------------------------------
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
	timeelapsed = epoch_now - epoch  # in years

	# Calculate the dec:
	decrate = pm_dec/3600000.0  # in degrees/year (assuming original was in mas/year)
	decindegrees = dec + timeelapsed*decrate

	# Calculate the unprojected rate of RA motion, using the mean declination between the catalog and present epoch:
	rarate = pm_ra/np.cos((dec + timeelapsed*decrate/2.0)*np.pi/180.0)/3600000.0  # in degress of RA/year (assuming original was in mas/year)
	raindegrees = ra + timeelapsed*rarate

	# Return the current positions
	return raindegrees, decindegrees

#------------------------------------------------------------------------------
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
	return (flux / 4 *
		((erf((x - x_0 + 0.5) / (np.sqrt(2) * sigma)) -
		  erf((x - x_0 - 0.5) / (np.sqrt(2) * sigma))) *
		 (erf((y - y_0 + 0.5) / (np.sqrt(2) * sigma)) -
		  erf((y - y_0 - 0.5) / (np.sqrt(2) * sigma)))))

#------------------------------------------------------------------------------
def mag2flux(mag):
	"""
	Convert from magnitude to flux using scaling relation from
	aperture photometry. This is an estimate.

	The scaling is based on fast-track TESS data from sectors 1 and 2.

	Parameters:
		mag (float): Magnitude in TESS band.

	Returns:
		float: Corresponding flux value
	"""
	return 10**(-0.4*(mag - 20.60654144))

#------------------------------------------------------------------------------
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

#------------------------------------------------------------------------------
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

#------------------------------------------------------------------------------
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

#------------------------------------------------------------------------------
def rms_timescale(time, flux, timescale=3600/86400):
	"""
	Compute robust RMS on specified timescale. Using MAD scaled to RMS.

	Parameters:
		time (ndarray): Timestamps in days.
		flux (ndarray): Flux to calculate RMS for.
		timescale (float, optional): Timescale to bin timeseries before calculating RMS. Default=1 hour.

	Returns:
		float: Robust RMS on specified timescale.

	.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
	"""

	# Construct the bin edges seperated by the timescale:
	bins = np.arange(np.nanmin(time), np.nanmax(time), timescale)
	bins = np.append(bins, np.nanmax(time))

	# Bin the timeseries to one hour:
	indx = np.isfinite(flux)
	flux_bin, _, _ = binned_statistic(time[indx], flux[indx], nanmean, bins=bins)

	# Compute robust RMS value (MAD scaled to RMS)
	return mad_to_sigma * nanmedian(np.abs(flux_bin - nanmedian(flux_bin)))

#------------------------------------------------------------------------------
def find_nearest(array, value):
	"""
	Search array for value and return the index where the value is closest.

	Parameters:
		array (ndarray): Array to search.
		value: Value to search array for.

	Returns:
		int: Index of ``array`` closest to ``value``.

	.. codeauthor:: Rasmus Handberg <rasmush@phys.au.dk>
	"""
	idx = np.searchsorted(array, value, side='left')
	if idx > 0 and (idx == len(array) or abs(value - array[idx-1]) < abs(value - array[idx])):
		return idx-1
	else:
		return idx