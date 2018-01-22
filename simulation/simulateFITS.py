#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 17:21:56 2018

.. codeauthor:: Jonas Svenstrup Hansen <jonas.svenstrup@gmail.com>
"""

import os
import numpy as np
import random
from astropy.io import fits
from astropy.table import Table

# Import stuff from the photometry directory:
if __package__ is None:
	import sys
	from os import path
	sys.path.append( path.dirname( path.dirname( path.abspath(__file__))))
	
	from photometry.psf import PSF
	from photometry.utilities import mag2flux
#	from photometry.plots import plot_image


class simulateFITS(object):
	def __init__(self, Nstars = 5, Ntimes = 5, save_images=True):
		"""
		Simulate a FITS image with stars, background and noise
		"""
		self.Nstars = Nstars # Number of stars in image
		self.Ntimes = Ntimes # Number of images in time series
		
		# Get output folder from enviroment variables:
		self.output_folder = os.environ.get('TESSPHOT_OUTPUT', 
									os.path.abspath('.'))

		# Set random number generator seed:
		random.seed(0)

		# Set image parameters:
		self.pixel_scale = 21.1 # Size of single pixel in arcsecs
		self.Nrows = 200
		self.Ncols = 200
		# TODO: check that the following stamp definition is correct
		self.stamp = (
						- self.Nrows//2,
						self.Nrows//2,
						- self.Ncols//2,
						self.Ncols//2
		)

		# Define time stamps:
		self.times = self.make_times()

		# Run through the time stamps:
		for i, timestamp in enumerate(self.times):
			# Make catalog:
			self.catalog = self.make_catalog()
			
			# Change catalog:
			# TODO: apply time-dependent changes to catalog parameters
	
			# Make stars from catalog:
			# FIXME: move catalog flux calculation to here
			stars = self.make_stars()
	
			# Make uniform background:
			bkg = self.make_background()
	
			# Make Gaussian noise:
			noise = self.make_noise()
	
			# Sum image from its parts:
			img = stars + bkg + noise
	
			if save_images:
				# Output image to FITS file:
				hdu = fits.PrimaryHDU(img)
				hdu.header['TIME'] = (timestamp/3600/24, 'time in days')
				# TODO: write target info to header
				hdu.writeto(os.path.join(self.output_folder, 'test%02d.fits' % i))



	def make_times(self, cadence = 1800.0):
		"""
		Make the time stamps
		
		Parameters:
			cadence (float): Time difference between frames. Default is 1800 
			seconds.
		"""
		# Define time stamps:
		times = np.arange(0, cadence*self.Ntimes, cadence)
		
		# Ensure correct number of time steps:
		if len(times) > self.Ntimes:
			times = times[0:10]
		
		return times


	def make_catalog(self):
		"""
		Returns:
			`astropy.table.Table`: Table with stars in the image.
		"""
		# Set star identification:
		starids = np.arange(self.Nstars, dtype=int)
		
		# Set buffer pixel size around edge where not to put stars:
		bufferpx = 3
		
		# Draw uniform row positions:
		starrows = np.asarray([random.uniform(bufferpx, self.Nrows-bufferpx) \
							for i in range(self.Nstars)])
	
		# Draw uniform column positions:
		starcols = np.asarray([random.uniform(bufferpx, self.Ncols-bufferpx) \
							for i in range(self.Nstars)])
	
		# Draw stellar fluxes:
		starmag = np.asarray([random.uniform(5,10) for i in range(self.Nstars)])
		
		# Collect star parameters in list for catalog:
		cat = [starids, starrows, starcols, starmag, mag2flux(starmag)]
		
		# Make astropy table with catalog:
		return Table(
			cat,
			names=('starid', 'row', 'col', 'tmag', 'flux'),
			dtype=('int64', 'float64', 'float64', 'float32', 'float64')
		)


	def make_stars(self):
		# Create PSF class instance:
		KPSF = PSF(camera=20, ccd=1, stamp=self.stamp)
		
		# Make list with parameter arrays for the pixel integrater:
		params = [np.array(
					[self.catalog['row'][i], 
					self.catalog['col'][i], 
					self.catalog['flux'][i]]
				) 
				for i in range(self.Nstars)]
		
		# Integrate stars to image:
		stars = KPSF.integrate_to_image(params, cutoff_radius=20)
		
		return stars


	def make_background(self):
		# Set background level:
		bkg_level = 1e3
		
		# Apply background level by multiplying:
		return bkg_level * np.ones([self.Nrows, self.Ncols])


	def make_noise(self):
		# Set sigma value:
		sigma = 10
		
		# Preallocate noise array:
		noise = np.zeros([self.Nrows, self.Ncols])
		
		# Loop over each pixel:
		for row in range(self.Nrows):
			for col in range(self.Ncols):
				# Draw a random value from a Gaussian (normal) distribution:
				noise[row,col] = random.gauss(mu=0, sigma=sigma)
		
		return noise


if __name__ == '__main__':
	sim = simulateFITS()
	catalog = sim.catalog
	print(catalog)

