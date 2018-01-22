#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 17:21:56 2018

.. codeauthor:: Jonas Svenstrup Hansen <jonas.svenstrup@gmail.com>
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy
import random
from astropy.io import fits
from astropy.table import Table, Column

# Import stuff from the photometry directory:
if __package__ is None:
	import sys
	from os import path
	sys.path.append( path.dirname( path.dirname( path.abspath(__file__))))
	
	from photometry.psf import PSF
	from photometry.utilities import mag2flux


class simulateFITS(object):
	def __init__(self):
		"""
		Simulate a FITS image with stars, background and noise
		"""
		# Set random number generator seed:
		random.seed(0)
		
		""" Set image parameters """
		self.pixel_scale = 21.0 # Size of single pixel in arcsecs
		self.Nrows = 200
		self.Ncols = 200
		# TODO: check that the following stamp definition is correct
		self.stamp = (
						- self.Nrows//2,
						self.Nrows//2,
						- self.Ncols//2,
						self.Ncols//2)

		""" Make catalog """
		# Set number of stars in image:
		Nstars = 2
		# Set star identification:
		starids = np.arange(Nstars, dtype=int)
		# Set buffer pixel size around edge where not to put stars:
		bufferpx = 3
		# Draw uniform row positions:
		starrows = np.asarray([random.uniform(bufferpx, self.Nrows-bufferpx) \
							for i in range(Nstars)])
		# Draw uniform column positions:
		starcols = np.asarray([random.uniform(bufferpx, self.Ncols-bufferpx) \
							for i in range(Nstars)])
		# Draw stellar fluxes:
		starflux = np.asarray([random.uniform(1e4,1e6) for i in range(Nstars)])
		# Collect star parameters in list for catalog:
		cat = [starids, starrows, starcols, starflux]
		# Make astropy table with catalog:
		self.catalog = Table(
			cat,
			names=('starid', 'row', 'col', 'tmag'),
			dtype=('int64', 'float64', 'float64', 'float32')
		)

		""" Make stars from catalog """
		# Preallocate array for stars:
		stars = np.zeros([self.Nrows, self.Ncols])
		# Create PSF class instance:
		KPSF = PSF(camera=20, ccd=1, stamp=self.stamp)
		# Convert catalog to parameters for the pixel integrater:
		params = np.empty([len(cat), 3], dtype='float64')
		# Add PRF of each star to the image:
		for (row,col,flux) in zip(starrows,starcols,starflux):
			stars += KPSF.integrate_to_image(params, cutoff_radius=20)
#			stars += integratedGaussian(X, Y, flux, col, row, sigma)

		""" Make uniform background """
		bkg_level = 1e3
		bkg = bkg_level * np.ones_like(stars)

		""" Make Gaussian noise """
		noise = np.zeros_like(stars)

		""" Sum image from its parts """
		img = stars + bkg + noise

		""" Output image to FITS file """



if __name__ == '__main__':
	sim = simulateFITS()
	KPSF = PSF(20, 1, sim.stamp)
#	KPSF.plot()
	
	print(sim.catalog)


