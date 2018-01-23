#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 17:21:56 2018

.. codeauthor:: Jonas Svenstrup Hansen <jonas.svenstrup@gmail.com>
"""

import os
import numpy as np
import random
from copy import deepcopy
from astropy.io import fits
from astropy.table import Table, Column

# Import stuff from the photometry directory:
if __package__ is None:
	import sys
	from os import path
	sys.path.append( path.dirname( path.dirname( path.abspath(__file__))))
	
	from photometry.psf import PSF
	from photometry.utilities import mag2flux
#	from photometry.plots import plot_image


class simulateFITS(object):
	def __init__(self, Nstars = 5, Ntimes = 5, 
			save_images=True, overwrite_images=True):
		"""
		Simulate FITS images with stars, background and noise.
		
		The purpose of this code is not to replace SPyFFI, but to supplement it
		in making simulated images simpler and more customizable. The aim is to
		supply simulated images that can illustrate the performance of various
		photometry methods in the photometry pipeline.
		
		Parameters:
			Nstars (int): Number of stars in image. Default is 5.
			Ntimes (int): Number of time steps in timeseries. Default is 5.
			save_images (boolean): True if images and catalog should be saved. 
			Default is True.
			overwrite_images (boolean): True if image and catalog files should 
			be overwritten.
		
		Example:
			Default use. Write 5 FITS images of shape 200x200px with 5 stars in
			them to five separate files in a subdirectory called images in the 
			directory specified by the TESSPHOT_INPUT environment variable:
				
			>>> sim = simulateFITS()
		
		.. codeauthor:: Jonas Svenstrup Hansen <jonas.svenstrup@gmail.com>
		"""
		self.Nstars = Nstars # Number of stars in image
		self.Ntimes = Ntimes # Number of images in time series
		self.save_images = save_images # True if images+catalog should be saved
		self.overwrite_images = overwrite_images # True if overwrite in saving
		
		# Get output directory from enviroment variable:
		self.output_folder = os.environ.get('TESSPHOT_INPUT', 
									os.path.abspath('.'))

		# Set random number generator seed:
		random.seed(0)

		# Set image parameters:
		self.pixel_scale = 21.0 # Size of single pixel in arcsecs
		self.Nrows = 200
		self.Ncols = 200
		self.stamp = (0,200,0,200)

		# TODO: move the rest of __init__ to a run_simulateFITS in parent dir
		# Define time stamps:
		self.times = self.make_times()

		# Make catalog:
		self.catalog = self.make_catalog()
		self.make_catalog_file(self.catalog)
		
		# Change catalog:
		# TODO: apply time-independent changes to catalog
		
		# Loop through the time stamps:
		for i, timestamp in enumerate(self.times):
			
			# Change catalog:
			# TODO: apply time-dependent changes to catalog parameters
	
			# Make stars from catalog:
			stars = self.make_stars()
	
			# Make uniform background:
			bkg = self.make_background()
	
			# Make Gaussian noise:
			noise = self.make_noise()
	
			# Sum image from its parts:
			img = stars + bkg + noise
	
			if self.save_images:
				# Write img to FITS file:
				# TODO: Add possibility to write to custom directory
				self.make_fits(img, timestamp, i)
				# TODO: Save catalog as txt file to output_folder


	def make_times(self, cadence = 1800.0):
		"""
		Make the time stamps.
		
		Parameters:
			cadence (float): Time difference between frames. Default is 1800 
			seconds.
		
		Returns:
			times (numpy array): Timestamps of all images to be made.
		"""
		# Define time stamps:
		times = np.arange(0, cadence*self.Ntimes, cadence)
		
		# Force correct number of time steps:
		# (this is necessary because cadence is not an int)
		if len(times) > self.Ntimes:
			times = times[0:10]
		
		return times


	def make_catalog(self):
		"""
		Make catalog of stars in the current image.
		
		The table contains the following columns:
		 * starid: Identifier. Starts at 0.
		 * row:    Pixel row in image.
		 * col:    Pixel column in image.
		 * tmag:   TESS magnitude.
		
		Returns:
			catalog (`astropy.table.Table`): Table with stars in the current 
			image.
		"""
		# Set star identification:
		starids = np.arange(self.Nstars, dtype=int)
		
		# Set buffer pixel size around edge where not to put stars:
		# TODO: Add possibility of stars on the edges
		bufferpx = 3
		
		# Draw uniform row positions:
		starrows = np.asarray([random.uniform(bufferpx, self.Nrows-bufferpx) \
							for i in range(self.Nstars)])
	
		# Draw uniform column positions:
		starcols = np.asarray([random.uniform(bufferpx, self.Ncols-bufferpx) \
							for i in range(self.Nstars)])
	
		# Draw stellar magnitudes:
		starmag = np.asarray([random.uniform(5,10) for i in range(self.Nstars)])
		
		# Collect star parameters in list for catalog:
		cat = [starids, starrows, starcols, starmag]
		
		# Make astropy table with catalog:
		return Table(
			cat,
			names=('starid', 'row', 'col', 'tmag'),
			dtype=('int64', 'float64', 'float64', 'float32')
		)


	def make_catalog_file(self, catalog, fname='catalog', compress=True):
		"""
		Write catalog to an ASCII file.
		
		Parameters:
			catalog (`astropy.table.Table`): Table with stars in the current 
			image. Columns must be starid, row, col, tmag.
			fname (string): Filename of catalog. Default is catalog.
			compress (boolean): True if catalog txt file is to be compressed
			and the uncompressed version deleted. Default is True
		"""
		if self.save_images:
			# Set ra and dec positions:
			# TODO: Convert (row, col) to (ra, dec)
			ra = None
			dec = None
			
			# Set proper motion:
			prop_mot_ra = np.zeros_like(catalog['tmag'])
			prop_mot_dec = np.zeros_like(catalog['tmag'])
			
			# Define extra columns:
			Col_ra = Column(ra, dtype=np.float64)
			Col_dec = Column(dec, dtype=np.float64)
			Col_prop_mot_ra = Column(prop_mot_ra, dtype=np.float32)
			Col_prop_mot_dec = Column(prop_mot_dec, dtype=np.float32)
			
			# Add extra columns to catalog:
			catalog.add_columns([Col_ra, Col_dec, 
								Col_prop_mot_ra, Col_prop_mot_dec],
								indexes=[0,0,0,0],
								names=['ra', 'dec', 
									'prop_mot_ra', 'prop_mot_dec'])
			
			# Convert catalog to numpy array:
			catalog_out = np.asarray(catalog)
			
			if self.overwrite_images:
				# Directory with filename of catalog output file:
				if compress:
					fextension = '.txt.gz'
				else:
					fextension = '.txt'
				txtfiledir = os.path.join(self.output_folder, fname+fextension)
				
				# Write catalog to txt file:
				np.savetxt(txtfiledir, np.column_stack(catalog_out), 
							delimiter='\t', header=catalog.colnames)
			else:
				# TODO: add check and error if file exists
				pass


	def make_stars(self, camera=20, ccd=1):
		"""
		Make stars for the image and append catalog with flux column.
		
		Parameters:
			camera (int): Kepler camera. Used to get PSF. Default is 20.
			ccd (int): Kepler CCD. Used to get PSF. Default is 1.
		
		Returns:
			stars (numpy array): Summed PRFs of stars in the image of the same
			shape as image.
		"""
		
		# Create PSF class instance:
		KPSF = PSF(camera=camera, ccd=ccd, stamp=self.stamp)
		
		# Make list with parameter numpy arrays for the pixel integrater:
		params = [
					np.array(
						[self.catalog['row'][i], 
						self.catalog['col'][i], 
						mag2flux(self.catalog['tmag'][i])]
					) 
				for i in range(self.Nstars)
				]
		
		# Integrate stars to image:
		return KPSF.integrate_to_image(params, cutoff_radius=20)


	def make_background(self, bkg_level=1e3):
		"""
		Make a background for the image.
		
		Parameters:
			bkg_level (float): Background level of uniform background. Default
			is 1000.
		
		Returns:
			bkg (numpy array): Background array of the same shape as image.
		"""
		
		# Apply background level by multiplying:
		return bkg_level * np.ones([self.Nrows, self.Ncols])


	def make_noise(self, sigma=500.0):
		"""
		Make Gaussian noise uniformily across the image.
		
		Parameters:
			sigma (float): Sigma parameter of Gaussian distribution for noise.
			Default is 500.0.
		
		Returns:
			noise (numpy array): Noise array of the same shape as image.
		"""
		
		# Preallocate noise array:
		noise = np.zeros([self.Nrows, self.Ncols])
		
		# Loop over each pixel:
		for row in range(self.Nrows):
			for col in range(self.Ncols):
				# Draw a random value from a Gaussian (normal) distribution:
				noise[row,col] = random.gauss(mu=0, sigma=sigma)
		
		return noise


	def make_fits(self, img, timestamp, i, outdir=None):
		"""
		Write image to FITS file.
		
		Parameters:
			img (numpy array): Image to write to file.
			timestamp (float): Timestamp in seconds of image.
			i (int): Timestamp index that is used in filename.
		"""
		
		# Instantiate primary header data unit:
		hdu = fits.PrimaryHDU(img)
		
		# Add timestamp to header with a unit of days:
		hdu.header['TIME'] = (timestamp/3600/24, 'time in days')
		# TODO: write more info to header
		
		if outdir is None:
			# Specify output directory:
			outdir = os.path.join(self.output_folder, 'images')
		
		# Write FITS file to output directory:
		hdu.writeto(os.path.join(outdir, 'test%02d.fits' % i),
					overwrite=self.overwrite_images)



if __name__ == '__main__':
	sim = simulateFITS(save_images=False)
	print(sim.catalog)

