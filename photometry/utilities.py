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
from bottleneck import move_median, nanmedian
import logging

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
