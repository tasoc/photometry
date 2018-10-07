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
from bottleneck import move_median, nanmedian
import logging
from scipy.special import erf
import json
import os.path
import fnmatch

def load_settings(sector=None):

	with open(os.path.join(os.path.dirname(__file__), 'data', 'settings.json'), 'r') as fid:
		settings = json.load(fid)

	if sector is not None:
		return settings['sectors'][str(sector)]

	return settings

#------------------------------------------------------------------------------
def find_ffi_files(rootdir, camera=None, ccd=None):
	"""
	Search directory recursively for TESS FFI images in FITS format.

	Parameters:
		rootdir (string): Directory to search recursively for TESS FFI images.
		camera (integer or None, optional): Only return files from the given camera number (1-4). If ``None``, files from all cameras are returned.
		ccd (integer or None, optional): Only return files from the given CCD number (1-4). If ``None``, files from all CCDs are returned.

	Returns:
		list: List of full paths to FFI FITS files found in directory. The list will
		      be sorted accoridng to the filename of the files, e.g. primarily by time.
	"""

	logger = logging.getLogger(__name__)

	# Create the filename pattern to search for:
	camera = '?' if camera is None else str(camera)
	ccd = '?' if ccd is None else str(ccd)
	filename_pattern = 'tess*-{camera:s}-{ccd:s}-????-[xsab]_ffic.fits*'.format(camera=camera, ccd=ccd)
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
def find_tpf_files(rootdir, starid=None):
	"""
	Search directory recursively for TESS Target Pixel Files.

	Parameters:
		rootdir (string): Directory to search recursively for TESS TPF files.
		starid (integer or None, optional): Only return files from the TIC number. If ``None``, files from all TIC numbers are returned.

	Returns:
		list: List of full paths to TPF FITS files found in directory. The list will
		      be sorted accoridng to the filename of the files, e.g. primarily by time.
	"""

	logger = logging.getLogger(__name__)

	# Create the filename pattern to search for:
	starid_str = '*' if starid is None else '{0:016d}'.format(starid)
	filename_pattern = 'tess*-{starid:s}-????-[xsab]_tp.fits*'.format(starid=starid_str)

	# Pattern used for TESS Alert data:
	starid_str = '*' if starid is None else '{0:011d}'.format(starid)
	filename_pattern2 = 'hlsp_tess-data-alerts_tess_phot_{starid:s}-s??_tess_v?_tp.fits*'.format(starid=starid_str)

	logger.debug("Searching for TPFs in '%s' using pattern '%s'", rootdir, filename_pattern)

	# Do a recursive search in the directory, finding all files that match the pattern:
	matches = []
	for root, dirnames, filenames in os.walk(rootdir, followlinks=True):
		for filename in filenames:
			if fnmatch.fnmatch(filename, filename_pattern) or fnmatch.fnmatch(filename, filename_pattern2):
				matches.append(os.path.join(root, filename))

	# Sort the list of files by thir filename:
	matches.sort(key = lambda x: os.path.basename(x))

	return matches

#------------------------------------------------------------------------------
def load_ffi_fits(path, return_header=False):
	"""
	Load FFI FITS file.

	Calibrations columns and rows are trimmed from the image.

	Parameters:
		path (str): Path to FITS file.
		return_header (boolean, optional): Return FITS headers as well. Default is ``False``.

	Returns:
		numpy.ndarray: Full Frame Image.
		list: If ``return_header`` is enabled, will return a list of the FITS headers.
	"""

	with fits.open(path, memmap=True, mode='readonly') as hdu:
		hdr = hdu[0].header
		if hdr.get('TELESCOP') == 'TESS' and hdu[1].header.get('NAXIS1') == 2136 and hdu[1].header.get('NAXIS2') == 2078:
			img = hdu[1].data[0:2048, 44:2092]
			headers = hdu[1].header
		else:
			img = hdu[0].data
			headers = hdu[0].header

	# Make sure its an numpy array with the correct data type:
	img = np.asarray(img, dtype='float32')

	if return_header:
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

	Parameters:
		mag (float): Magnitude in TESS band.

	Returns:
		float: Corresponding flux value
	"""
	return 10**(-0.4*(mag - 28.24))

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