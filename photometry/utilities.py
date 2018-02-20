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

def load_ffi_fits(fname, return_header=False):

	with fits.open(fname, memmap=True, mode='readonly') as hdu:
		hdr = hdu[0].header
		if hdr.get('TELESCOP', '') == 'TESS':
			img = hdu[1].data[0:2048, 44:2092]
			headers = [hdu[0].header, hdu[1].header]
		else:
			img = hdu[0].data
			headers = [hdu[0].header]

	img = np.array(img, dtype='float32')

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
	'''
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
	'''
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
		mag (float) : Magnitude in TESS band.

	Returns:
		float : Corresponding flux value
	"""
	return 10**(-0.4*(mag - 28.24))
